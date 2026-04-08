"""
query_rewriter.py — Phase 3 History-Aware Query Rewriting
---------------------------------------------------------
Rewrites short follow-up questions into clearer standalone retrieval queries
using recent conversation history from Streamlit session state.

Rewrite type classification (rule-based, no LLM required):
- reuse_previous   : "tell me more", "elaborate", "explain that", "why?", "how?"
- simplify         : "in simpler terms", "explain simply", "eli5"
- comparative      : "compare that with X", "what about the other report"
- location_add     : "in Mexico", "for rural areas", "regarding children"
- topic_switch     : "what about X", "and X", "how about Y"

Design choice:
- Fully rule-based — no extra LLM call, no added latency
- Easy to explain and demonstrate in project presentations
- Each rewrite carries a `rewrite_type` field for UI display

Returns:
{
    "original_query":  str,
    "rewritten_query": str,
    "used_history":    "yes" | "no",
    "rewrite_reason":  str,
    "rewrite_type":    str,   ← NEW — used by chat_page.py for UI badge
}
"""

from __future__ import annotations

import re
from typing import Dict, List

# ------------------------------------------------------------------
# Pattern tables
# ------------------------------------------------------------------

# These trigger a full reuse of the previous question as the retrieval query
REUSE_PREVIOUS_TRIGGERS = (
    "tell me more",
    "tell me more about that",
    "elaborate",
    "elaborate on that",
    "explain that",
    "can you explain",
    "give me an example",
    "can you give an example",
    "an example",
    "go on",
    "continue",
    "and why",
    "but why",
    "why?",
    "how?",
    "really?",
    "are you sure",
    "what do you mean",
)

# These trigger a simplification rewrite
SIMPLIFY_TRIGGERS = (
    "in simpler terms",
    "explain simply",
    "explain in simple terms",
    "explain like",
    "make it simpler",
    "simpler",
    "simplify",
    "simplify that",
    "eli5",
    "in plain english",
    "in plain language",
)

# These trigger a comparative rewrite
COMPARATIVE_TRIGGERS = (
    "compare that",
    "compare with",
    "compare to",
    "how does that compare",
    "versus",
    " vs ",
    "what about the other",
    "what about the second",
    "what about the first",
    "the other document",
    "the other report",
    "the other paper",
    "the second document",
    "the second report",
    "the first document",
    "the first report",
)

# Standard topic-switch follow-up openers
TOPIC_SWITCH_PREFIXES = (
    "what about",
    "how about",
    "what of",
    "and what about",
    "but what about",
    "also",
    "what else about",
    "more about",
    "now about",
)

# Location / scope addition openers
LOCATION_PREFIXES = (
    "in ",
    "for ",
    "regarding ",
    "concerning ",
    "within ",
    "across ",
    "among ",
    "between ",
    "outside ",
    "beyond ",
    "around ",
)

# Short openers that only qualify as follow-ups when few words follow
WEAK_PREFIXES = ("and", "also", "but", "so", "then")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def normalize_query(text: str) -> str:
    return " ".join((text or "").split()).strip()


def is_likely_follow_up(query: str) -> bool:
    """
    Return True when the query depends on prior context to make sense.
    Stricter than before — avoids false positives on long "and..." questions.
    """
    q = normalize_query(query).lower()
    if not q:
        return False

    word_count = len(q.split())

    # Definite reuse/simplify/comparative triggers — always follow-ups
    for pattern in (*REUSE_PREVIOUS_TRIGGERS, *SIMPLIFY_TRIGGERS, *COMPARATIVE_TRIGGERS):
        if q == pattern or q.startswith(pattern):
            return True

    # Topic-switch openers — always follow-ups regardless of length
    if any(q.startswith(p) for p in TOPIC_SWITCH_PREFIXES):
        return True

    # Location/scope additions — short only
    if any(q.startswith(p) for p in LOCATION_PREFIXES) and word_count <= 6:
        return True

    # Weak prefixes (and/but/so) — only follow-up if ≤ 5 words total
    # Prevents: "And what was the unemployment rate in 2019?" from being rewritten
    if any(q.startswith(p + " ") for p in WEAK_PREFIXES) and word_count <= 5:
        return True

    # Very short queries with no standalone meaning
    if word_count <= 3:
        return True

    return False


def _classify_follow_up(query: str) -> str:
    """
    Classify the type of follow-up for targeted rewrite construction.
    Returns one of: reuse_previous | simplify | comparative | location_add | topic_switch
    """
    q = normalize_query(query).lower()

    for pattern in SIMPLIFY_TRIGGERS:
        if q.startswith(pattern) or pattern in q:
            return "simplify"

    for pattern in REUSE_PREVIOUS_TRIGGERS:
        if q == pattern or q.startswith(pattern):
            return "reuse_previous"

    for pattern in COMPARATIVE_TRIGGERS:
        if q.startswith(pattern) or pattern in q:
            return "comparative"

    if any(q.startswith(p) for p in LOCATION_PREFIXES):
        return "location_add"

    return "topic_switch"


def _strip_known_prefix(query: str) -> str:
    """
    Strip a leading follow-up opener and return the core phrase.
    Tries longest matches first to avoid partial strips.
    """
    q = normalize_query(query)
    q_lower = q.lower()

    all_prefixes = sorted(
        (*TOPIC_SWITCH_PREFIXES, *WEAK_PREFIXES, "what about", "how about"),
        key=len,
        reverse=True,
    )

    for prefix in all_prefixes:
        if q_lower.startswith(prefix):
            tail = q[len(prefix):].strip(" ,.:;?-")
            if tail:
                return tail

    return q.strip(" ,.:;?-")


def _build_rewrite(
    follow_up_type: str,
    original_query: str,
    previous_question: str,
) -> tuple[str, str]:
    """
    Construct the rewritten query and a human-readable reason label.
    Returns (rewritten_query, rewrite_reason).
    """
    base = normalize_query(previous_question).rstrip("?").strip()
    q_lower = original_query.lower()

    # ── Reuse previous question as-is ──
    if follow_up_type == "reuse_previous":
        return previous_question, "elaboration_on_previous"

    # ── Simplification request ──
    if follow_up_type == "simplify":
        return f"Explain in simple terms: {base}", "simplify_previous"

    # ── Comparative ──
    if follow_up_type == "comparative":
        tail = _strip_known_prefix(original_query)
        # Remove comparative trigger words from tail
        for t in COMPARATIVE_TRIGGERS:
            if tail.lower().startswith(t.strip()):
                tail = tail[len(t):].strip(" ,.:;?-")
                break
        if tail:
            return f"Compare {base} with {tail}", "comparative_followup"
        return f"Compare this with related findings: {base}", "comparative_no_target"

    # ── Location / scope addition ──
    if follow_up_type == "location_add":
        # Keep the original "in X" / "for X" fragment and append to base
        return f"{base} {original_query.rstrip('?')}", "location_scope_addition"

    # ── Topic switch (default) ──
    topic = _strip_known_prefix(original_query)
    if not topic:
        return previous_question, "empty_followup_reused_previous"

    # Construct a natural sentence based on how the previous question opened
    prev_lower = base.lower()
    if prev_lower.startswith(("what does", "what do")):
        rewritten = f"{base} about {topic}"
    elif prev_lower.startswith(("what is", "what are", "what were", "what was")):
        rewritten = f"{base} regarding {topic}"
    elif prev_lower.startswith(("how does", "how do", "how is", "how are")):
        rewritten = f"{base} in relation to {topic}"
    elif prev_lower.startswith(("describe", "explain", "summarize", "tell")):
        rewritten = f"{base} focusing on {topic}"
    else:
        rewritten = f"{base} — specifically about {topic}"

    return rewritten, "topic_switch_followup"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def rewrite_query_with_history(
    current_query: str,
    chat_history: List[Dict],
    max_history_turns: int = 2,
) -> Dict[str, str]:
    """
    Rewrite a user query into a clearer standalone retrieval query.

    Returns
    -------
    {
        "original_query":  str,
        "rewritten_query": str,
        "used_history":    "yes" | "no",
        "rewrite_reason":  str,
        "rewrite_type":    str,   ← reuse_previous | simplify | comparative
                                     | location_add | topic_switch | none
    }
    """
    original_query = normalize_query(current_query)

    _no_rewrite = lambda reason: {
        "original_query":  original_query,
        "rewritten_query": original_query,
        "used_history":    "no",
        "rewrite_reason":  reason,
        "rewrite_type":    "none",
    }

    if not original_query:
        return {**_no_rewrite("empty_query"), "original_query": ""}

    if not chat_history:
        return _no_rewrite("no_history")

    if not is_likely_follow_up(original_query):
        return _no_rewrite("standalone_query")

    recent = chat_history[-max_history_turns:]
    previous_question = normalize_query(recent[-1].get("question", ""))

    if not previous_question:
        return _no_rewrite("missing_previous_question")

    follow_up_type = _classify_follow_up(original_query)
    rewritten, reason = _build_rewrite(
        follow_up_type=follow_up_type,
        original_query=original_query,
        previous_question=previous_question,
    )

    # Clean up spacing and ensure question mark
    rewritten = re.sub(r"\s+", " ", rewritten).strip(" -")
    if not rewritten.endswith("?"):
        rewritten = rewritten.rstrip(".") + "?"

    # Safety: if rewrite is identical to original, mark as no rewrite
    if rewritten.rstrip("?").strip().lower() == original_query.rstrip("?").strip().lower():
        return _no_rewrite("rewrite_identical_to_original")

    return {
        "original_query":  original_query,
        "rewritten_query": rewritten,
        "used_history":    "yes",
        "rewrite_reason":  reason,
        "rewrite_type":    follow_up_type,
    }