"""
ner.py — Named Entity Recognition Helpers
-----------------------------------------
Provides spaCy-based named entity extraction for Document Intelligence Hub.

Responsibilities:
- Load the spaCy NER model
- Extract entities from raw text
- Handle long document text by chunking
- Return normalized entity dictionaries
- Summarize extracted entities by label/text frequency

Supported entity labels kept by default:
- PERSON
- ORG
- DATE
- GPE
- LOC
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import spacy

logger = logging.getLogger(__name__)

ENTITY_TYPES_TO_KEEP = {"PERSON", "ORG", "DATE", "GPE", "LOC"}


def load_ner_model(model_name: str = "en_core_web_sm"):
    """
    Load and return a spaCy NER model.
    """
    try:
        nlp = spacy.load(model_name)
        logger.info("Loaded spaCy model: %s", model_name)
        return nlp
    except Exception as exc:
        logger.error("Failed to load spaCy model %s: %s", model_name, exc)
        raise


def extract_entities_from_text(
    text: str,
    nlp,
    allowed_labels: set[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Extract named entities from one text string.
    """
    if not text or not text.strip():
        return []

    if allowed_labels is None:
        allowed_labels = ENTITY_TYPES_TO_KEEP

    doc = nlp(text)

    entities: List[Dict[str, Any]] = []
    for ent in doc.ents:
        if ent.label_ in allowed_labels:
            cleaned_text = ent.text.strip()
            if cleaned_text:
                entities.append(
                    {
                        "entity_text": cleaned_text,
                        "entity_label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                    }
                )

    return entities


def extract_entities_from_document(
    full_text: str,
    nlp,
    allowed_labels: set[str] | None = None,
    chunk_size: int = 50000,
) -> List[Dict[str, Any]]:
    """
    Extract entities from document-level text.

    For long documents, splits the text into chunks before running NER.
    Character offsets are adjusted so returned start/end positions still
    correspond to the original full document text.
    """
    if not full_text or not full_text.strip():
        return []

    if allowed_labels is None:
        allowed_labels = ENTITY_TYPES_TO_KEEP

    if len(full_text) <= chunk_size:
        return extract_entities_from_text(
            text=full_text,
            nlp=nlp,
            allowed_labels=allowed_labels,
        )

    all_entities: List[Dict[str, Any]] = []

    for start_idx in range(0, len(full_text), chunk_size):
        chunk = full_text[start_idx:start_idx + chunk_size]

        chunk_entities = extract_entities_from_text(
            text=chunk,
            nlp=nlp,
            allowed_labels=allowed_labels,
        )

        for ent in chunk_entities:
            all_entities.append(
                {
                    "entity_text": ent["entity_text"],
                    "entity_label": ent["entity_label"],
                    "start_char": ent["start_char"] + start_idx,
                    "end_char": ent["end_char"] + start_idx,
                }
            )

    return all_entities


def summarize_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate extracted entities by (entity_label, entity_text) and count frequency.
    """
    counts: dict[tuple[str, str], int] = {}

    for ent in entities:
        key = (ent["entity_label"], ent["entity_text"])
        counts[key] = counts.get(key, 0) + 1

    summary = [
        {
            "entity_label": label,
            "entity_text": text,
            "count": count,
        }
        for (label, text), count in counts.items()
    ]

    summary.sort(key=lambda x: (x["entity_label"], -x["count"], x["entity_text"]))
    return summary