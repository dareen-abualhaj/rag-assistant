import pytest

from src.chunking import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("Short text.", 100, 10) == ["Short text."]


def test_empty_text_returns_empty_list():
    assert chunk_text("   ", 100, 10) == []


def test_chunks_respect_max_size():
    text = "This is a sentence. " * 200
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_shared_between_consecutive_chunks():
    text = "word " * 400
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert chunks[0][-20:].strip() in chunks[1]


def test_no_content_lost_without_overlap():
    text = "Alpha beta gamma. " * 100
    chunks = chunk_text(text, chunk_size=200, overlap=0)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_long_word_hard_cut():
    chunks = chunk_text("x" * 1000, chunk_size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)
