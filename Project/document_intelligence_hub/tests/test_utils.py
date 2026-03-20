import pytest
from src.utils import normalize_text, count_words, format_bytes

def test_normalize_text():
    # Test whitespace collapsing
    assert normalize_text("  hello    world  ") == "hello world"
    # Test null character removal
    assert normalize_text("hello\x00world") == "hello world"
    # Test empty input
    assert normalize_text(None) == ""

def test_count_words():
    assert count_words("This is a test.") == 4
    assert count_words("") == 0
    assert count_words(None) == 0

def test_format_bytes():
    assert format_bytes(500) == "500 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1048576) == "1.00 MB"