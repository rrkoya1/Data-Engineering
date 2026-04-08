"""
rag_ollama.py — Phase 3 Local RAG Answer Generation with Ollama
---------------------------------------------------------------
Uses retrieved semantic chunks as grounded context and sends them
to a local Ollama model for answer generation.

Supports two calling modes:
- Blocking   (call_ollama / generate_grounded_answer) — returns full answer
- Streaming  (call_ollama_stream)                     — yields tokens one-by-one
                                                        compatible with st.write_stream()

Supported answer modes:
- strict_grounding   — answer ONLY from retrieved sources
- summarization      — clear summary of retrieved content
- exploratory        — compare and connect ideas across sources
"""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, List

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 120


# ------------------------------------------------------------------
# Source map helpers
# ------------------------------------------------------------------

def build_source_map(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a stable source map with citation labels like [S1], [S2], [S3].
    """
    source_map: List[Dict[str, Any]] = []

    for i, item in enumerate(results, start=1):
        title = item.get("display_title", "Untitled Document")
        file_name = item.get("file_name", "")
        page_number = item.get("page_number", "N/A")
        chunk_index = item.get("chunk_index", "N/A")
        chunk_text = (item.get("chunk_text") or "").strip()

        citation_label = f"[S{i}]"
        source_label = f"{title} — page {page_number}, chunk {chunk_index}"

        source_map.append(
            {
                "citation_label": citation_label,
                "source_label": source_label,
                "display_title": title,
                "file_name": file_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
            }
        )

    return source_map


def format_retrieved_context(results: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved semantic chunks into a structured context block
    for the LLM prompt, using stable citation labels.
    """
    if not results:
        return "No retrieved context available."

    source_map = build_source_map(results)
    blocks: List[str] = []

    for source in source_map:
        block = f"""
{source["citation_label"]}
Title: {source["display_title"]}
File: {source["file_name"]}
Page: {source["page_number"]}
Chunk: {source["chunk_index"]}

Content:
{source["chunk_text"]}
""".strip()

        blocks.append(block)

    return "\n\n" + ("\n\n" + "-" * 80 + "\n\n").join(blocks)


def extract_source_labels(results: List[Dict[str, Any]]) -> List[str]:
    """
    Return clean display labels with stable citation markers.
    """
    source_map = build_source_map(results)
    return [f'{item["citation_label"]} {item["source_label"]}' for item in source_map]


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------

def build_prompt(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    mode: str = "strict_grounding",
) -> str:
    """
    Build a grounded prompt using retrieved chunks only.
    The model is instructed to cite sources using [S1], [S2], [S3] etc.
    """
    context_block = format_retrieved_context(retrieved_results)
    mode = (mode or "strict_grounding").strip().lower()

    common_rules = """
Citation Rules:
- When you make a claim, cite supporting sources using labels like [S1], [S2].
- Only use the citation labels provided in the retrieved context.
- Do not invent citations.
- If the context is insufficient, say so clearly.
""".strip()

    if mode == "summarization":
        instructions = f"""
You are a document intelligence assistant.

Your task is to answer the user's question ONLY using the retrieved context below.

Rules:
- Summarize clearly and simply.
- Stay grounded in the retrieved context.
- Do not invent facts not supported by the sources.
- If the retrieved context is incomplete, say so.
- Write a concise answer in plain language.
- End with a short line titled "Sources used:" and list only the citation labels used.

{common_rules}
""".strip()

    elif mode == "exploratory":
        instructions = f"""
You are a document intelligence assistant.

Your task is to answer the user's question using ONLY the retrieved context below.

Rules:
- Provide a thoughtful exploratory answer grounded in the sources.
- You may compare or connect ideas across the retrieved sources.
- Do not hallucinate or introduce facts not present in the context.
- If the evidence is partial or ambiguous, say that clearly.
- End with a short line titled "Sources used:" and list only the citation labels used.

{common_rules}
""".strip()

    else:  # strict_grounding
        instructions = f"""
You are a document intelligence assistant.

Your task is to answer the user's question STRICTLY using the retrieved context below.

Rules:
- Use only the information found in the retrieved context.
- Do not use outside knowledge.
- If the answer is not supported by the context, say exactly:
  "The retrieved sources do not contain enough information to answer this confidently."
- Be precise and concise.
- Prefer short paragraphs or bullet points when useful.
- End with a short line titled "Sources used:" and list only the citation labels used.

{common_rules}
""".strip()

    prompt = f"""
{instructions}

User Question:
{query}

Retrieved Context:
{context_block}

Now write the answer.
""".strip()

    return prompt


# ------------------------------------------------------------------
# Ollama — blocking call
# ------------------------------------------------------------------

def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Send a prompt to a local Ollama model and return the full text response.
    Blocking — waits for the complete response before returning.
    """
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    return (data.get("response") or "").strip()


# ------------------------------------------------------------------
# Ollama — streaming call  ← NEW
# ------------------------------------------------------------------

def call_ollama_stream(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> Generator[str, None, None]:
    """
    Stream tokens from a local Ollama model one by one.

    Yields individual string tokens as they arrive from the Ollama API.
    Fully compatible with Streamlit's st.write_stream().

    Usage in chat_page.py:
        full_answer = st.write_stream(
            call_ollama_stream(prompt=rag_prompt, model=ollama_model)
        )

    Raises:
        requests.RequestException — if Ollama is unreachable or returns an error.
    """
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
        },
    }

    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            token = data.get("response", "")
            if token:
                yield token

            if data.get("done"):
                break


# ------------------------------------------------------------------
# High-level blocking API (kept for non-streaming use cases)
# ------------------------------------------------------------------

def generate_grounded_answer(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    mode: str = "strict_grounding",
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Generate a grounded answer from retrieved semantic chunks using Ollama.
    Blocking version — use call_ollama_stream() for streaming in the chat UI.
    """
    query = (query or "").strip()

    if not query:
        return {
            "success": False,
            "answer": "",
            "mode": mode,
            "model": model,
            "sources": [],
            "source_map": [],
            "message": "Query is empty.",
        }

    if not retrieved_results:
        return {
            "success": False,
            "answer": "",
            "mode": mode,
            "model": model,
            "sources": [],
            "source_map": [],
            "message": "No retrieved results were provided.",
        }

    source_map = build_source_map(retrieved_results)

    prompt = build_prompt(
        query=query,
        retrieved_results=retrieved_results,
        mode=mode,
    )

    answer = call_ollama(
        prompt=prompt,
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
    )

    return {
        "success": True,
        "answer": answer,
        "mode": mode,
        "model": model,
        "sources": extract_source_labels(retrieved_results),
        "source_map": source_map,
        "message": "Grounded answer generated successfully.",
    }