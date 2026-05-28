from app.ingest import chunk_text
import pytest

def test_chunk_text():
    text = "abcdefghij"
    res = chunk_text(text, chunk_size=5, overlap=0)
    assert res == ["abcde", "fghij"]


def test_chunk_text_empty_string():
    text = ""
    res = chunk_text(text, chunk_size=5, overlap=0)
    assert res == []


def test_chunk_text_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("test", chunk_size=0)


def test_chunk_text_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text("test", chunk_size=10, overlap=-5)


def test_chunk_text_overlap_too_large():
    with pytest.raises(ValueError):
        chunk_text("test", chunk_size=5, overlap=10)