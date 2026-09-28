from src.ingestion.parser.cleaner import clean_text


# --- Rule 10: encoding and quotes ---

def test_curly_quotes_become_straight():
    assert clean_text("OpenAI\u2019s \u201cGPT-4\u201d") == 'OpenAI\'s "GPT-4"'


def test_non_breaking_space_becomes_space():
    assert clean_text("March\u00a02023") == "March 2023"


# --- Rule 11: invisible characters ---

def test_zero_width_characters_removed():
    assert clean_text("Open\u200bAI") == "OpenAI"


# --- Rule 14: whitespace ---

def test_extra_spaces_collapsed():
    assert clean_text("OpenAI    released   GPT-4.") == "OpenAI released GPT-4."


def test_many_blank_lines_become_one():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_paragraph_break_is_kept():
    assert clean_text("one\n\ntwo") == "one\n\ntwo"


# --- Rule 13: broken lines ---

def test_hyphenated_word_is_rejoined():
    assert clean_text("The company was co-found-\ned by Sam.") == "The company was co-founded by Sam."


def test_broken_line_becomes_one_sentence():
    assert clean_text("released GPT-4 in March\n2023.") == "released GPT-4 in March 2023."


# --- Rule 15: junk sections ---

def test_toc_lines_removed():
    text = "Contents\nIntroduction ........ 1\nConclusion ........ 7\n\nReal text starts here."
    assert clean_text(text) == "Real text starts here."


def test_copyright_line_removed():
    text = "Real text about OpenAI.\n\n(c) 2023 Example Corp. All rights reserved."
    assert clean_text(text) == "Real text about OpenAI."


def test_references_section_cut():
    body = "OpenAI released GPT-4 in March 2023. " * 5
    text = f"{body}\n\nReferences\n\n[1] Smith, J. (2023). A history of AI."
    assert "Smith" not in clean_text(text)
    assert "OpenAI released GPT-4" in clean_text(text)


def test_references_word_at_start_does_not_wipe_document():
    text = "References\n\nThis document explains OpenAI and its models in detail."
    assert "OpenAI" in clean_text(text)

def test_blank_text_does_not_trigger_loss_warning(caplog):
    with caplog.at_level("WARNING"):
        clean_text("   \n\n   ")
    assert "Cleaning removed" not in caplog.text