# test_nlp_phase2.py

from src.db import get_connection
from src.nlp_pipeline import (
    fetch_document_corpus,
    preview_preprocessing,
    build_tfidf_features,
)

def main():
    conn = get_connection()

    corpus = fetch_document_corpus(conn)
    print(f"Loaded documents: {len(corpus)}")

    if not corpus:
        print("No documents found. Ingest PDFs first.")
        return

    first_doc = corpus[0]
    print("\nFirst document:")
    print("Document ID:", first_doc["document_id"])
    print("Title:", first_doc["display_title"])
    print("Pages:", first_doc["page_count"])

    sample_text = first_doc["full_text"][:1000]
    preview = preview_preprocessing(sample_text)

    print("\nCleaned text preview:")
    print(preview["cleaned_text"][:300])

    print("\nTokens:")
    print(preview["tokens"][:20])

    print("\nTokens without stopwords:")
    print(preview["tokens_no_stopwords"][:20])

    print("\nStemmed tokens:")
    print(preview["stemmed_tokens"][:20])

    print("\nLemmatized tokens:")
    print(preview["lemmatized_tokens"][:20])

    processed_texts = [doc["full_text"] for doc in corpus]
    vectorizer, X = build_tfidf_features(processed_texts)

    print("\nTF-IDF matrix shape:", X.shape)
    print("Sample features:", vectorizer.get_feature_names_out()[:20])

if __name__ == "__main__":
    main()