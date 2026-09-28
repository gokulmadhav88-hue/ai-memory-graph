from src.ingestion.parser.tables import Table, extract_captions, table_to_texts


def make(rows, caption=None):
    return Table(rows=rows, caption=caption)


def test_rows_become_sentences_with_headers():
    t = make([["Model", "Company", "Year"], ["GPT-4", "OpenAI", "2023"], ["Claude 2", "Anthropic", "2023"]],
             "Table 2: Model release dates.")
    (text,) = table_to_texts(t)
    assert "Table 2: Model release dates." in text
    assert "Model GPT-4: Company is OpenAI, Year is 2023." in text
    assert "Model Claude 2: Company is Anthropic, Year is 2023." in text


def test_long_table_splits_and_repeats_header(monkeypatch):
    from src.ingestion.parser import config
    monkeypatch.setattr(config, "TABLE_ROWS_PER_CHUNK", 2)
    rows = [["Model", "Year"]] + [[f"M{i}", str(2000 + i)] for i in range(5)]
    texts = table_to_texts(make(rows))
    assert len(texts) == 3
    assert all("Columns: Model, Year." in t for t in texts)     # header in every part
    joined = " ".join(texts)
    for i in range(5):                                          # no row lost or cut in half
        assert f"Model M{i}: Year is {2000 + i}." in joined


def test_wide_table_falls_back_to_markdown():
    header = [f"c{i}" for i in range(9)]
    row = [str(i) for i in range(9)]
    (text,) = table_to_texts(make([header, row]))
    assert "| c0 | c1 |" in text
    assert "| --- |" in text


def test_empty_header_cell_falls_back_to_markdown():
    (text,) = table_to_texts(make([["", "Q1", "Q2"], ["Rev", "5", "6"]]))
    assert "| Rev | 5 | 6 |" in text


def test_row_longer_than_header_is_logged_and_kept(caplog):
    with caplog.at_level("WARNING"):
        (text,) = table_to_texts(make([["A", "B"], ["1", "2", "3"]]))
    assert "longer than the header" in caplog.text
    assert "column 3 is 3" in text


def test_header_only_table_gives_nothing():
    assert table_to_texts(make([["A", "B"]])) == []


def test_empty_rows_and_columns_dropped():
    rows = [["Model", "", "Year"], ["GPT-4", "", "2023"], ["", "", ""], ["Claude 2", "", "2023"]]
    (text,) = table_to_texts(make(rows))
    assert "Model GPT-4: Year is 2023." in text
    assert "Model Claude 2: Year is 2023." in text


def test_caption_detected_and_removed_from_text():
    text, caps = extract_captions("Intro text.\nTable 1: Model release dates.\nMore text.", 1)
    assert caps == ["Table 1: Model release dates."]
    assert "Table 1" not in text


def test_prose_sentence_starting_with_table_is_not_a_caption():
    text, caps = extract_captions("Table 2 shows that revenue grew.", 1)
    assert caps == [None]
    assert "Table 2 shows" in text