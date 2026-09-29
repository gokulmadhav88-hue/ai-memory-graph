from src.ingestion.parser.tables import Table, extract_captions, merge_continuations, table_to_texts


def make(rows, caption=None, page=None, section=None):
    return Table(rows=rows, caption=caption, page=page, section=section)


def test_rows_become_sentences_with_headers():
    t = make([["Model", "Company", "Year"], ["GPT-4", "OpenAI", "2023"], ["Claude 2", "Anthropic", "2023"]],
             "Table 2: Model release dates.")
    (piece,) = table_to_texts(t)
    assert "Table 2: Model release dates." in piece.text
    assert "Model GPT-4: Company is OpenAI, Year is 2023." in piece.text
    assert "Model Claude 2: Company is Anthropic, Year is 2023." in piece.text
    assert piece.is_numeric is False


def test_long_table_splits_and_repeats_header(monkeypatch):
    from src.ingestion.parser import config
    monkeypatch.setattr(config, "TABLE_ROWS_PER_CHUNK", 2)
    rows = [["Model", "Year"]] + [[f"M{i}", str(2000 + i)] for i in range(5)]
    pieces = table_to_texts(make(rows))
    assert len(pieces) == 3
    assert all("Columns: Model, Year." in p.text for p in pieces)
    joined = " ".join(p.text for p in pieces)
    for i in range(5):
        assert f"Model M{i}: Year is {2000 + i}." in joined


def test_wide_table_falls_back_to_markdown():
    header = [f"c{i}" for i in range(9)]
    row = [str(i) for i in range(9)]
    (piece,) = table_to_texts(make([header, row]))
    assert "| c0 | c1 |" in piece.text
    assert "| --- |" in piece.text


def test_empty_header_cell_falls_back_to_markdown():
    (piece,) = table_to_texts(make([["", "Q1", "Q2"], ["Rev", "5", "6"]]))
    assert "| Rev | 5 | 6 |" in piece.text


def test_row_longer_than_header_is_logged_and_kept(caplog):
    with caplog.at_level("WARNING"):
        (piece,) = table_to_texts(make([["A", "B"], ["1", "2", "3"]]))
    assert "longer than the header" in caplog.text
    assert "column 3 is 3" in piece.text


def test_header_only_table_gives_nothing():
    assert table_to_texts(make([["A", "B"]])) == []


def test_empty_rows_and_columns_dropped():
    rows = [["Model", "", "Year"], ["GPT-4", "", "2023"], ["", "", ""], ["Claude 2", "", "2023"]]
    (piece,) = table_to_texts(make(rows))
    assert "Model GPT-4: Year is 2023." in piece.text
    assert "Model Claude 2: Year is 2023." in piece.text


def test_caption_detected_and_removed_from_text():
    text, caps = extract_captions("Intro text.\nTable 1: Model release dates.\nMore text.", 1)
    assert caps == ["Table 1: Model release dates."]
    assert "Table 1" not in text


def test_prose_sentence_starting_with_table_is_not_a_caption():
    text, caps = extract_captions("Table 2 shows that revenue grew.", 1)
    assert caps == [None]
    assert "Table 2 shows" in text


# --- gap fix: numeric-heavy tables are flagged ---

def test_numeric_heavy_table_is_flagged():
    rows = [["Year", "Revenue", "Growth"]] + [[str(2015 + i), str(1000 * i), f"{i}%"] for i in range(6)]
    (piece,) = table_to_texts(make(rows))
    assert piece.is_numeric is True


def test_name_heavy_table_is_not_flagged_numeric():
    rows = [["Model", "Company"], ["GPT-4", "OpenAI"], ["Claude 2", "Anthropic"]]
    (piece,) = table_to_texts(make(rows))
    assert piece.is_numeric is False


# --- gap fix: page-spanning table continuation ---

def test_headerless_continuation_is_merged_not_treated_as_new_header():
    t1 = make([["Model", "Score"]] + [[f"M{i}", str(i)] for i in range(5)], page=1)
    t2 = make([[f"M{i}", str(i)] for i in range(5, 10)], page=2)   # no header row, no caption
    merged = merge_continuations([t1, t2])
    assert len(merged) == 1
    pieces = table_to_texts(merged[0])
    joined = " ".join(p.text for p in pieces)
    for i in range(10):
        assert f"Model M{i}: Score is {i}." in joined


def test_table_with_its_own_caption_is_not_merged():
    t1 = make([["Model", "Score"], ["M1", "1"]], page=1)
    t2 = make([["Model", "Score"], ["M2", "2"]], page=2, caption="Table 2: A different table.")
    merged = merge_continuations([t1, t2])
    assert len(merged) == 2


def test_table_after_a_page_gap_is_not_merged():
    t1 = make([["Model", "Score"], ["M1", "1"]], page=1)
    t2 = make([["Model", "Score"], ["M2", "2"]], page=5)   # far away, unrelated table
    merged = merge_continuations([t1, t2])
    assert len(merged) == 2
