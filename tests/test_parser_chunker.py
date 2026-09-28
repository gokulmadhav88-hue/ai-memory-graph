from src.ingestion.parser import config
from src.ingestion.parser.chunker import chunk_text, split_sentences


def words(text):
    return len(text.split())


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_long_paragraph_splits_at_sentence_boundaries():
    text = ("OpenAI released a new model and the press covered it widely. " * 120).strip()
    pieces = chunk_text(text)
    assert len(pieces) > 1
    for p in pieces:
        assert p.text.rstrip().endswith(".")                       # never cut mid-sentence
        assert words(p.text) <= config.MAX_CHUNK_WORDS


def test_short_paragraphs_are_merged():
    text = "Overview.\n\nOpenAI is a company.\n\nIt makes AI models.\n\nGPT-4 is one of them."
    assert len(chunk_text(text)) == 1


def test_junk_paragraphs_are_dropped():
    good = ("OpenAI released GPT-4 in March 2023 and it was widely covered by the "
            "technology press across many outlets worldwide today.")
    text = f"{good}\n\n-------- ******** ========\n\nhttps://www.example.com/page?id=1\n\n1234 5678 9012 3456"
    pieces = chunk_text(text)
    assert len(pieces) == 1
    assert pieces[0].text == good


def test_repeated_boilerplate_kept_only_once():
    boiler = "Confidential: this document is for internal use only and must not be shared outside the company."
    body = "OpenAI released GPT-4 in March 2023 and Microsoft has invested billions in the company since 2019."
    joined = " ".join(p.text for p in chunk_text(f"{boiler}\n\n{body}\n\n{boiler}"))
    assert joined.count("Confidential") == 1


def test_heading_becomes_section_not_chunk():
    text = ("# Company History\n\nOpenAI was founded in 2015 in San Francisco.\n\n"
            "# Products\n\nOpenAI released GPT-4 in March 2023.")
    pieces = chunk_text(text)
    assert [p.section for p in pieces] == ["Company History", "Products"]
    assert all("#" not in p.text for p in pieces)


def test_next_chunk_starts_with_last_sentence_of_previous():
    text = " ".join(f"Fact {i} about OpenAI models is documented here at length." for i in range(150))
    pieces = chunk_text(text)
    assert len(pieces) > 1
    assert pieces[1].text.startswith(split_sentences(pieces[0].text)[-1])