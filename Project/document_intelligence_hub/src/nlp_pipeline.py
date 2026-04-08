from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence, Tuple

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Optional NLTK support
# -------------------------------------------------------------------

NLTK_AVAILABLE = True
WORDNET_AVAILABLE = True
PUNKT_AVAILABLE = True

try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    _stemmer = PorterStemmer()
    _lemmatizer = WordNetLemmatizer()
except Exception:
    NLTK_AVAILABLE = False
    WORDNET_AVAILABLE = False
    PUNKT_AVAILABLE = False
    _stemmer = None
    _lemmatizer = None
    word_tokenize = None


def ensure_nltk_resources() -> None:
    global PUNKT_AVAILABLE, WORDNET_AVAILABLE

    if not NLTK_AVAILABLE:
        logger.warning("NLTK is not installed. Falling back to regex tokenization.")
        return

    try:
        nltk.data.find("tokenizers/punkt")
        PUNKT_AVAILABLE = True
    except LookupError:
        PUNKT_AVAILABLE = False
        logger.warning("NLTK punkt resource not found. Falling back to regex tokenization.")

    try:
        nltk.data.find("corpora/wordnet.zip")
        WORDNET_AVAILABLE = True
    except LookupError:
        try:
            nltk.data.find("corpora/wordnet")
            WORDNET_AVAILABLE = True
        except LookupError:
            WORDNET_AVAILABLE = False
            logger.warning(
                "NLTK wordnet resource not found. Lemmatization will fall back to token identity."
            )


ensure_nltk_resources()

# -------------------------------------------------------------------
# Stopwords
# -------------------------------------------------------------------

CUSTOM_STOPWORDS = {
    "et",
    "al",
    "figure",
    "table",
    "page",
    "pages",
    "also",
    "using",
    "used",
    "use",
}

STOPWORDS = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)

# -------------------------------------------------------------------
# Internal schema helpers
# -------------------------------------------------------------------


def _get_table_columns(conn: Any, table_name: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cur.fetchall()}


def _resolve_documents_pk(document_columns: set[str]) -> str:
    for candidate in ("id", "document_id", "doc_id"):
        if candidate in document_columns:
            return candidate
    raise ValueError(
        f"Could not find document primary key column in documents table. Found: {sorted(document_columns)}"
    )


def _resolve_pages_document_fk(page_columns: set[str]) -> str:
    for candidate in ("document_id", "doc_id"):
        if candidate in page_columns:
            return candidate
    raise ValueError(
        f"Could not find document foreign key column in pages table. Found: {sorted(page_columns)}"
    )


def _resolve_page_text_column(page_columns: set[str]) -> str:
    for candidate in ("page_text", "text", "text_content"):
        if candidate in page_columns:
            return candidate
    raise ValueError(
        f"Could not find page text column in pages table. Found: {sorted(page_columns)}"
    )


def _resolve_page_number_column(page_columns: set[str]) -> str:
    for candidate in ("page_number", "page_num"):
        if candidate in page_columns:
            return candidate
    raise ValueError(
        f"Could not find page number column in pages table. Found: {sorted(page_columns)}"
    )


def _resolve_file_name_column(document_columns: set[str]) -> str | None:
    for candidate in ("file_name", "filename"):
        if candidate in document_columns:
            return candidate
    return None


def _resolve_title_column(document_columns: set[str], fallback: str | None) -> str | None:
    if "title" in document_columns:
        return "title"
    return fallback


def _resolve_author_column(document_columns: set[str]) -> str | None:
    if "author" in document_columns:
        return "author"
    return None


# -------------------------------------------------------------------
# Database / corpus loading helpers
# -------------------------------------------------------------------


def fetch_document_corpus(conn: Any) -> List[Dict[str, Any]]:
    """
    Load document-level corpus from SQLite by aggregating all pages for each document.
    """

    document_columns = _get_table_columns(conn, "documents")
    page_columns = _get_table_columns(conn, "pages")

    document_pk = _resolve_documents_pk(document_columns)
    page_document_fk = _resolve_pages_document_fk(page_columns)
    page_text_col = _resolve_page_text_column(page_columns)
    page_number_col = _resolve_page_number_column(page_columns)
    file_name_col = _resolve_file_name_column(document_columns)
    title_col = _resolve_title_column(document_columns, file_name_col)
    author_col = _resolve_author_column(document_columns)

    file_name_select = f"d.{file_name_col} AS file_name," if file_name_col else "'' AS file_name,"
    display_title_select = (
        f"COALESCE(d.{title_col}, 'Untitled Document') AS display_title,"
        if title_col
        else "'Untitled Document' AS display_title,"
    )
    author_select = f"COALESCE(d.{author_col}, '') AS author," if author_col else "'' AS author,"

    query = f"""
        SELECT
            d.{document_pk} AS document_id,
            {file_name_select}
            {display_title_select}
            {author_select}
            p.{page_number_col} AS page_number,
            COALESCE(p.{page_text_col}, '') AS page_text
        FROM documents d
        JOIN pages p ON d.{document_pk} = p.{page_document_fk}
        ORDER BY d.{document_pk}, p.{page_number_col}
    """

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    docs: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        document_id, file_name, display_title, author, page_number, page_text = row

        if document_id not in docs:
            docs[document_id] = {
                "document_id": document_id,
                "file_name": file_name,
                "display_title": display_title,
                "author": author,
                "page_count": 0,
                "pages": [],
            }

        docs[document_id]["pages"].append(page_text or "")
        docs[document_id]["page_count"] += 1

    corpus: List[Dict[str, Any]] = []
    for doc in docs.values():
        full_text = "\n".join(doc["pages"]).strip()
        corpus.append(
            {
                "document_id": doc["document_id"],
                "file_name": doc["file_name"],
                "display_title": doc["display_title"],
                "author": doc["author"],
                "page_count": doc["page_count"],
                "full_text": full_text,
            }
        )

    logger.info("Loaded %s documents into document-level corpus.", len(corpus))
    return corpus


def fetch_page_level_corpus(conn: Any) -> List[Dict[str, Any]]:
    """
    Load page-level corpus from SQLite.
    """

    document_columns = _get_table_columns(conn, "documents")
    page_columns = _get_table_columns(conn, "pages")

    document_pk = _resolve_documents_pk(document_columns)
    page_document_fk = _resolve_pages_document_fk(page_columns)
    page_text_col = _resolve_page_text_column(page_columns)
    page_number_col = _resolve_page_number_column(page_columns)
    file_name_col = _resolve_file_name_column(document_columns)
    title_col = _resolve_title_column(document_columns, file_name_col)

    file_name_select = f"d.{file_name_col} AS file_name," if file_name_col else "'' AS file_name,"
    display_title_select = (
        f"COALESCE(d.{title_col}, 'Untitled Document') AS display_title,"
        if title_col
        else "'Untitled Document' AS display_title,"
    )

    query = f"""
        SELECT
            d.{document_pk} AS document_id,
            {file_name_select}
            {display_title_select}
            p.{page_number_col} AS page_number,
            COALESCE(p.{page_text_col}, '') AS page_text
        FROM documents d
        JOIN pages p ON d.{document_pk} = p.{page_document_fk}
        ORDER BY d.{document_pk}, p.{page_number_col}
    """

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        document_id, file_name, display_title, page_number, page_text = row
        results.append(
            {
                "document_id": document_id,
                "file_name": file_name,
                "display_title": display_title,
                "page_number": page_number,
                "page_text": page_text,
            }
        )

    logger.info("Loaded %s pages into page-level corpus.", len(results))
    return results


def fetch_single_document(conn: Any, document_id: int) -> Dict[str, Any] | None:
    corpus = fetch_document_corpus(conn)
    for doc in corpus:
        if doc["document_id"] == document_id:
            return doc
    return None


# -------------------------------------------------------------------
# Text cleaning / preprocessing
# -------------------------------------------------------------------


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = text.replace("\x00", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    if not text:
        return []

    if NLTK_AVAILABLE and PUNKT_AVAILABLE and word_tokenize is not None:
        try:
            return word_tokenize(text)
        except Exception as exc:
            logger.warning("NLTK tokenization failed, using regex fallback: %s", exc)

    return re.findall(r"\b[a-zA-Z0-9]+\b", text)


def remove_stopwords(tokens: Sequence[str]) -> List[str]:
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


def stem_tokens(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []

    if NLTK_AVAILABLE and _stemmer is not None:
        try:
            return [_stemmer.stem(tok) for tok in tokens]
        except Exception as exc:
            logger.warning("Stemming failed, returning original tokens: %s", exc)

    return list(tokens)


def lemmatize_tokens(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []

    if NLTK_AVAILABLE and WORDNET_AVAILABLE and _lemmatizer is not None:
        try:
            return [_lemmatizer.lemmatize(tok) for tok in tokens]
        except Exception as exc:
            logger.warning("Lemmatization failed, returning original tokens: %s", exc)

    return list(tokens)


def preprocess_text(
    text: str,
    method: str = "lemmatize",
    remove_stops: bool = True,
) -> str:
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    if method == "lemmatize":
        tokens = lemmatize_tokens(tokens)
    elif method == "stem":
        tokens = stem_tokens(tokens)
    elif method == "none":
        pass
    else:
        raise ValueError("method must be one of: 'lemmatize', 'stem', 'none'")

    return " ".join(tokens)


def preprocess_corpus(
    texts: Sequence[str],
    method: str = "lemmatize",
    remove_stops: bool = True,
) -> List[str]:
    return [preprocess_text(text, method=method, remove_stops=remove_stops) for text in texts]


def preview_preprocessing(text: str, max_tokens: int = 40) -> Dict[str, Any]:
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    no_stops = remove_stopwords(tokens)
    stemmed = stem_tokens(no_stops)
    lemmatized = lemmatize_tokens(no_stops)

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "tokens": tokens[:max_tokens],
        "tokens_no_stopwords": no_stops[:max_tokens],
        "stemmed_tokens": stemmed[:max_tokens],
        "lemmatized_tokens": lemmatized[:max_tokens],
        "stemmed_text": " ".join(stemmed),
        "lemmatized_text": " ".join(lemmatized),
    }


# -------------------------------------------------------------------
# Vectorization
# -------------------------------------------------------------------


def build_bow_features(
    texts: Sequence[str],
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 1),
):
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    logger.info("Built BoW matrix with shape %s", X.shape)
    return vectorizer, X


def build_tfidf_features(
    texts: Sequence[str],
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    logger.info("Built TF-IDF matrix with shape %s", X.shape)
    return vectorizer, X


def get_top_tfidf_terms(vectorizer, matrix_row, top_n: int = 15) -> List[Tuple[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    row = matrix_row.toarray().flatten()
    top_indices = row.argsort()[::-1][:top_n]

    top_terms = []
    for idx in top_indices:
        score = float(row[idx])
        if score > 0:
            top_terms.append((feature_names[idx], score))

    return top_terms


# -------------------------------------------------------------------
# Convenience helpers for ML page
# -------------------------------------------------------------------


def build_document_texts_for_ml(
    conn: Any,
    preprocess: bool = True,
    method: str = "lemmatize",
) -> List[Dict[str, Any]]:
    corpus = fetch_document_corpus(conn)

    for item in corpus:
        raw_text = item.get("full_text", "") or ""
        item["processed_text"] = preprocess_text(raw_text, method=method) if preprocess else raw_text

    return corpus


def build_page_texts_for_ml(
    conn: Any,
    preprocess: bool = True,
    method: str = "lemmatize",
) -> List[Dict[str, Any]]:
    pages = fetch_page_level_corpus(conn)

    for item in pages:
        raw_text = item.get("page_text", "") or ""
        item["processed_text"] = preprocess_text(raw_text, method=method) if preprocess else raw_text

    return pages