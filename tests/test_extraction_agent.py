from src.ingestion.extraction_agent import ExtractionAgent


def test_extraction_agent_returns_note():
    agent = ExtractionAgent()
    note = agent.extract("Sample text about Alpha and Beta.", "doc_1")

    assert note.source_document_id == "doc_1"
    assert note.content == "Sample text about Alpha and Beta."
    assert isinstance(note.entities, list)
    assert isinstance(note.keywords, list)
    assert isinstance(note.tags, list)


def test_extraction_agent_extract_many():
    agent = ExtractionAgent()
    notes = agent.extract_many(["A", "B"], "doc_2")

    assert len(notes) == 2
