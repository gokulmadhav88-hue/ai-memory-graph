from src.ingestion.parser.cleaner import clean_text


def test_curly_quotes_become_straight():
    assert clean_text("OpenAI\u2019s \u201cGPT-4\u201d") == 'OpenAI\'s "GPT-4"'


def test_non_breaking_space_becomes_space():
    assert clean_text("March\u00a02023") == "March 2023"


def test_zero_width_characters_removed():
    assert clean_text("Open\u200bAI") == "OpenAI"


def test_extra_spaces_collapsed():
    assert clean_text("OpenAI    released   GPT-4.") == "OpenAI released GPT-4."


def test_many_blank_lines_become_one():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_paragraph_break_is_kept():
    assert clean_text("one\n\ntwo") == "one\n\ntwo"