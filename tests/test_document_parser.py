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


def test_txt_chunks_have_contract_fields():
    chunks = parse_document(AI_NEWS)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.doc_id == "ai_news"
    assert c.chunk_index == 0
    assert c.source_filename == "ai_news.txt"
    assert c.content_hash and c.parser_version
    assert c.content_type == "text"


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