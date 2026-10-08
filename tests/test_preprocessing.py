from src.preprocessing import (
    clean_text,
    clean_whitespace,
    is_english,
    normalize_unicode,
    strip_html,
)


def test_strip_html_removes_tags():
    assert strip_html("<p>Hello</p>").strip() == "Hello"


def test_strip_html_decodes_entities():
    assert "&" in strip_html("Tom &amp; Jerry")


def test_normalize_unicode_nfkc():
    assert normalize_unicode("ﬁne") == "fine"  # ligature -> "fi"


def test_normalize_unicode_removes_zero_width():
    assert normalize_unicode("a\u200bb") == "ab"


def test_clean_whitespace_collapses():
    assert clean_whitespace("  a   b \n\n c  ") == "a b c"


def test_is_english_true():
    assert is_english("This is a simple sentence written in English language.")


def test_is_english_false_other_language():
    assert not is_english("Ceci est une phrase écrite en langue française.")


def test_is_english_false_too_short():
    assert not is_english("Hi")


def test_clean_text_full_pipeline():
    assert clean_text("<p>Hello   &amp;\n welcome</p>") == "Hello & welcome"


def test_clean_text_non_string():
    assert clean_text(None) == ""
