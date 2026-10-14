"""Regression checks for page provenance, mixed OCR, and bounded archives."""

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pymupdf
import pytest

from src.ingestion import parser
from src.ingestion.parser import FileTooLarge, InvalidFile, config, parse_document
from src.ingestion.parser.chunker import chunk_text
from src.ingestion.parser.figures import describe_figures
from src.ingestion.parser.readers import RawDocument, _docx_figures
from src.ingestion.parser.validator import validate_file

FIX = Path(__file__).parent / 'fixtures'


def test_pdf_text_preserves_page_and_section(monkeypatch):
    monkeypatch.setattr(parser, 'read_document', lambda path: RawDocument(pages=[
        '# Project Plans\n\nAtlas is owned by Maya.',
        'Delta is owned by Priya.',
    ], ocr_pages={2}))
    chunks = parse_document(FIX / 'simple.pdf')
    assert [(chunk.page, chunk.section, chunk.is_ocr) for chunk in chunks] == [
        (1, 'Project Plans', False), (2, 'Project Plans', True),
    ]
    assert 'Delta' not in chunks[0].text
    assert 'Atlas' not in chunks[1].text


def test_non_pdf_has_no_physical_page():
    assert all(chunk.page is None for chunk in parse_document(FIX / 'with_markdown.md'))


def test_short_assertion_without_punctuation_is_kept():
    pieces = chunk_text('Atlas is managed by Maya')
    assert len(pieces) == 1
    assert pieces[0].text == 'Atlas is managed by Maya'
    assert pieces[0].section is None


def test_overlap_does_not_exceed_hard_word_limit():
    pieces = chunk_text(' '.join(['unpunctuated'] * (config.MAX_CHUNK_WORDS + 50)))
    assert len(pieces) == 2
    assert all(len(piece.text.split()) <= config.MAX_CHUNK_WORDS for piece in pieces)


def test_mixed_pdf_ocr_only_missing_page(tmp_path, monkeypatch):
    from src.ingestion.parser import readers
    path = tmp_path / 'mixed.pdf'
    with pymupdf.open() as combined:
        for source in (FIX / 'simple.pdf', FIX / 'scanned_page.pdf'):
            with pymupdf.open(source) as part:
                combined.insert_pdf(part)
        combined.save(path)
    rendered = []

    def render(path, page_numbers=None):
        rendered.extend(page_numbers)
        return [b'page'] * len(page_numbers)

    monkeypatch.setattr(config, 'ENABLE_OCR', True)
    monkeypatch.setattr(readers, '_render_pdf_pages', render)
    monkeypatch.setattr(readers, 'ocr_pages', lambda pages: ['Scanned project Delta is owned by Priya.'])
    chunks = parse_document(path)
    assert rendered == [2]
    assert any('OpenAI' in chunk.text and chunk.page == 1 and not chunk.is_ocr for chunk in chunks)
    assert any('Delta' in chunk.text and chunk.page == 2 and chunk.is_ocr for chunk in chunks)


def test_figure_with_unknown_dimensions_is_described(monkeypatch):
    part = SimpleNamespace(blob=b'unknown image format')
    document = SimpleNamespace(part=SimpleNamespace(rels={
        'r1': SimpleNamespace(reltype='image', target_part=part),
    }))
    figures = _docx_figures(document)
    assert figures[0].width is None and figures[0].height is None
    monkeypatch.setattr(config, 'ENABLE_FIGURE_DESCRIPTIONS', True)
    assert len(describe_figures(figures, lambda image: 'Diagram description.')) == 1


def office_archive(path, extra=None):
    with ZipFile(path, 'w', compression=ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', '<Types/>')
        archive.writestr('word/document.xml', '<Document/>')
        for name, contents in (extra or {}).items():
            archive.writestr(name, contents)


def test_compressed_office_bomb_rejected_before_reader(tmp_path, monkeypatch):
    path = tmp_path / 'bomb.docx'
    office_archive(path, {'word/large.xml': b'x' * (1024 * 1024 + 1)})
    assert path.stat().st_size < 2000
    monkeypatch.setattr(config, 'MAX_ARCHIVE_SIZE_MB', 1)
    with pytest.raises(FileTooLarge, match='expanded Office archive'):
        validate_file(path)


def test_office_archive_member_limit(tmp_path, monkeypatch):
    path = tmp_path / 'many.docx'
    office_archive(path, {'word/extra.xml': '<Data/>'})
    monkeypatch.setattr(config, 'MAX_ARCHIVE_MEMBERS', 2)
    with pytest.raises(FileTooLarge, match='too many members'):
        validate_file(path)


def test_random_zip_is_not_office_document(tmp_path):
    path = tmp_path / 'renamed.docx'
    with ZipFile(path, 'w') as archive:
        archive.writestr('random.txt', 'Hello')
    with pytest.raises(InvalidFile, match='required Office content'):
        validate_file(path)


def test_csv_reader_limits_rows_before_materializing(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'rows.csv'
    path.write_text('Name,Owner\nAtlas,Maya\n\nDelta,Priya\nGamma,Lee\n', encoding='utf-8')
    monkeypatch.setattr(config, 'MAX_TABLE_ROWS', 2)
    chunks = parse_document(path)
    joined = ' '.join(chunk.text for chunk in chunks)
    assert 'Atlas' in joined and 'Delta' in joined
    assert 'Gamma' not in joined
    assert 'keeping the first 2' in caplog.text


def test_xlsx_reader_limits_rows_per_sheet(tmp_path, monkeypatch, caplog):
    from openpyxl import Workbook
    path = tmp_path / 'rows.xlsx'
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Projects'
    for row in [('Name', 'Owner'), ('Atlas', 'Maya'), ('Delta', 'Priya'), ('Gamma', 'Lee')]:
        sheet.append(row)
    workbook.save(path)
    workbook.close()
    monkeypatch.setattr(config, 'MAX_TABLE_ROWS', 2)
    joined = ' '.join(chunk.text for chunk in parse_document(path))
    assert 'Atlas' in joined and 'Delta' in joined
    assert 'Gamma' not in joined
    assert "sheet 'Projects'" in caplog.text
