from src.search import _build_snippet, _highlight_snippet

def test_build_snippet():
    text = "The quick brown fox jumps over the lazy dog."
    query = "fox"
    snippet = _build_snippet(text, query, max_len=20)
    assert "fox" in snippet
    assert len(snippet) <= 220 # default max_len in your code is 220

def test_highlight_snippet():
    snippet = "The quick brown fox jumps"
    query = "fox"
    highlighted = _highlight_snippet(snippet, query)
    assert "**fox**" in highlighted