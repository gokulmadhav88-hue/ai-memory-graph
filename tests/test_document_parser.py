from pathlib import Path

import pytest

from src.ingestion.parser import (
    EncryptedFile,
    FileTooLarge,
    InvalidFile,
    UnsupportedFileType,
    parse_document,
)

ROOT = Path(__file__).parent.parent
FIX = Path(__file__).parent / "fixtures"
AI_NEWS = ROOT / "data" / "raw_documents" / "ai_news.txt"


def joined(chunks):
    return " ".join(c.text for c in chunks)


def texts_of(chunks, kind):
    return [c for c in chunks if c.content_type == kind]


# --- Tier 1 ---

def test_txt_chunks_have_contract_fields():
    chunks = parse_document(AI_NEWS)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.doc_id == "ai_news"
    assert c.chunk_index == 0
    assert c.source_filename == "ai_news.txt"
    assert c.content_hash and c.parser_version
    assert c.content_type == "text"
    assert c.is_numeric_table is False


def test_same_input_gives_same_output():
    assert parse_document(AI_NEWS) == parse_document(AI_NEWS)


def test_chunk_indexes_are_sequential():
    chunks = parse_document(FIX / "long_paragraph.txt")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_and_whitespace_files_give_no_chunks():
    assert parse_document(FIX / "empty.txt") == []
    assert parse_document(FIX / "whitespace_only.txt") == []


def test_blank_pdf_gives_no_chunks():
    assert parse_document(FIX / "blank.pdf") == []


def test_simple_pdf_is_read():
    chunks = parse_document(FIX / "simple.pdf")
    assert chunks
    assert "OpenAI released GPT-4" in joined(chunks)


def test_pdf_headers_footers_and_page_numbers_removed():
    text = joined(parse_document(FIX / "repeated_footers.pdf"))
    assert "Acme Research Report" not in text
    assert "Confidential - Acme Corp" not in text
    assert "OpenAI released GPT-4" in text


def test_fake_pdf_is_rejected():
    with pytest.raises(InvalidFile):
        parse_document(FIX / "fake_extension.pdf")


def test_encrypted_pdf_is_rejected():
    with pytest.raises(EncryptedFile):
        parse_document(FIX / "encrypted.pdf")


def test_unsupported_type_is_rejected():
    with pytest.raises(UnsupportedFileType):
        parse_document(FIX / "unsupported_audio.mp3")


@pytest.mark.skipif(not (FIX / "too_large.txt").exists(), reason="too_large.txt not generated")
def test_oversized_file_is_rejected():
    with pytest.raises(FileTooLarge):
        parse_document(FIX / "too_large.txt")


def test_missing_file_is_rejected():
    with pytest.raises(InvalidFile):
        parse_document(FIX / "does_not_exist.txt")


def test_markdown_file_headings_become_sections():
    chunks = parse_document(FIX / "with_markdown.md")
    assert [c.section for c in chunks] == ["Company History", "Products"]
    assert all("#" not in c.text for c in chunks)


def test_markdown_bullets_stay_on_separate_lines():
    chunks = parse_document(FIX / "with_markdown.md")
    assert "- Text input\n- Image input" in chunks[1].text


# --- Tier 2 ---

def test_csv_becomes_table_chunk():
    chunks = parse_document(FIX / "models.csv")
    assert [c.content_type for c in chunks] == ["table"]
    assert "Model GPT-4: Company is OpenAI, Year is 2023." in chunks[0].text


def test_xlsx_becomes_table_chunk():
    chunks = parse_document(FIX / "models.xlsx")
    assert [c.content_type for c in chunks] == ["table"]
    assert "Model Claude 2: Company is Anthropic, Year is 2023." in chunks[0].text


def test_docx_text_and_table_are_separate_chunks():
    chunks = parse_document(FIX / "report_with_table.docx")
    prose, tables = texts_of(chunks, "text"), texts_of(chunks, "table")
    assert prose and tables
    assert prose[0].section == "Company History"
    assert all("Table 1" not in c.text for c in prose)
    assert "Table 1: Model release dates." in tables[0].text
    assert "Model GPT-4: Company is OpenAI, Year is 2023." in tables[0].text


def test_markdown_table_is_extracted():
    chunks = parse_document(FIX / "with_table.md")
    prose, tables = texts_of(chunks, "text"), texts_of(chunks, "table")
    assert prose and len(tables) == 1
    assert "|" not in " ".join(c.text for c in prose)
    assert "Table 1: Model release dates." in tables[0].text


def test_pdf_table_is_extracted_and_not_duplicated_in_text():
    chunks = parse_document(FIX / "table.pdf")
    prose, tables = texts_of(chunks, "text"), texts_of(chunks, "table")
    assert tables
    assert tables[0].page == 1
    assert "Model GPT-4: Company is OpenAI, Year is 2023." in tables[0].text
    assert all("Anthropic" not in c.text for c in prose)
    assert all("Table 1" not in c.text for c in prose)


def test_fake_docx_is_rejected():
    with pytest.raises(InvalidFile):
        parse_document(FIX / "fake_docx.docx")


# --- gap fixes ---

def test_numeric_csv_table_is_flagged():
    chunks = parse_document(FIX / "numeric_table.csv")
    assert chunks[0].content_type == "table"
    assert chunks[0].is_numeric_table is True


def test_named_table_is_not_flagged_numeric():
    chunks = parse_document(FIX / "models.csv")
    assert chunks[0].is_numeric_table is False


def test_docx_table_inherits_preceding_heading_as_section():
    chunks = parse_document(FIX / "pricing.docx")
    tables = texts_of(chunks, "table")
    assert tables
    assert tables[0].section == "Pricing"


def test_pdf_spanning_table_is_merged_not_split_with_false_header():
    chunks = parse_document(FIX / "spanning_table.pdf")
    tables = texts_of(chunks, "table")
    assert len(tables) == 1                     # merged into one logical table
    joined_text = " ".join(t.text for t in tables)
    for i in range(10):
        assert f"Model M{i}: Score is {i}." in joined_text


# --- Tier 3: figures and OCR ---

def _fake_describer(image_bytes: bytes) -> str:
    return "A bar chart showing example data."


def test_figures_are_skipped_by_default_no_image_chunks(monkeypatch):
    from src.ingestion.parser import config
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", False)
    chunks = parse_document(FIX / "pdf_with_figure.pdf")
    assert texts_of(chunks, "image_description") == []
    assert texts_of(chunks, "text")   # the real text is still there


def test_figures_are_described_when_enabled_pdf(monkeypatch):
    from src.ingestion.parser import config
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    chunks = parse_document(FIX / "pdf_with_figure.pdf", describe_image=_fake_describer)
    images = texts_of(chunks, "image_description")
    assert len(images) == 1
    assert "bar chart" in images[0].text
    assert images[0].page == 1


def test_figures_are_described_when_enabled_docx(monkeypatch):
    from src.ingestion.parser import config
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    chunks = parse_document(FIX / "report_with_figure.docx", describe_image=_fake_describer)
    images = texts_of(chunks, "image_description")
    assert len(images) == 1
    assert "bar chart" in images[0].text


def test_no_describer_passed_means_no_image_chunks_even_if_enabled(monkeypatch):
    from src.ingestion.parser import config
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    chunks = parse_document(FIX / "pdf_with_figure.pdf")   # no describe_image argument
    assert texts_of(chunks, "image_description") == []


def test_scanned_pdf_still_skipped_when_ocr_off():
    assert parse_document(FIX / "scanned_page.pdf") == []


def test_scanned_pdf_read_via_ocr_when_enabled(monkeypatch):
    from src.ingestion.parser import config
    monkeypatch.setattr(config, "ENABLE_OCR", True)
    chunks = parse_document(FIX / "scanned_page.pdf")
    assert chunks
    text = joined(chunks)
    # OCR isn't pixel-perfect (fonts can make "I" read as "l"); check the words OCR
    # reliably gets right rather than requiring an exact match.
    assert "released" in text
    assert "GPT-4" in text or "GPT" in text
    assert all(c.is_ocr for c in chunks)


def test_normal_pdf_chunks_are_not_marked_ocr():
    chunks = parse_document(FIX / "simple.pdf")
    assert all(c.is_ocr is False for c in chunks)
