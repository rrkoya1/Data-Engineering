"""
ml_page.py — ML / NLP Analysis Page
-------------------------------------
Renders the full Phase 2 ML analysis interface for the Document Intelligence Hub.

Sections:
- Document Preprocessing Demo
- TF-IDF Feature Analysis
- K-Means Clustering with Elbow Method and PCA
- Hierarchical Clustering with Dendrogram
- Heuristic Label Generation
- Label-Based Supervised Classification
- Named Entity Recognition (NER)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from src.db import (
    delete_entities_by_doc_id,
    get_connection,
    get_entities_by_doc_id,
    get_entity_counts_by_label,
    get_entity_summary_by_doc_id,
    insert_entities,
)
from src.ml_models import (
    attach_pca_coordinates,
    build_dendrogram_data,
    build_tfidf_matrix,
    compute_elbow_scores,
    generate_heuristic_labels,
    get_cluster_members,
    prepare_document_dataset,
    prepare_labeled_dataset,
    project_pca_2d,
    run_hierarchical_clustering,
    run_kmeans_clustering,
    summarize_clusters,
    train_and_evaluate_models,
)
from src.ner import extract_entities_from_document, load_ner_model, summarize_entities
from src.nlp_pipeline import (
    build_tfidf_features,
    clean_text,
    fetch_document_corpus,
    get_top_tfidf_terms,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    stem_tokens,
    tokenize_text,
)


@st.cache_resource
def get_ner_model():
    """
    Load and cache the spaCy NER model once per app session.
    """
    return load_ner_model()


def _format_token_preview(tokens: list[str], max_items: int = 40) -> str:
    if not tokens:
        return "No tokens available."

    shown = tokens[:max_items]
    suffix = "" if len(tokens) <= max_items else f" ... (+{len(tokens) - max_items} more)"
    return ", ".join(shown) + suffix


def render_ml_page() -> None:
    st.title("ML / NLP Analysis")
    st.caption(
        "Phase 2 - NLP preprocessing, TF-IDF analysis, clustering, label generation, classification, and NER"
    )

    conn = get_connection()
    try:
        corpus = fetch_document_corpus(conn)

        if not corpus:
            st.warning("No documents found. Please ingest PDFs first.")
            return

        corpus_df = pd.DataFrame(corpus).reset_index(drop=True)

        # ------------------------------------------------------------------
        # Document preprocessing demo
        # ------------------------------------------------------------------
        st.subheader("Document Preprocessing Demo")

        selected_title = st.selectbox(
            "Select a document",
            corpus_df["display_title"].tolist(),
        )

        selected_doc_index = int(
            corpus_df[corpus_df["display_title"] == selected_title].index[0]
        )
        selected_row = corpus_df.iloc[selected_doc_index]

        st.markdown("### Document Info")
        col1, col2, col3 = st.columns(3)
        col1.metric("Document ID", selected_row["document_id"])
        col2.metric("Pages", selected_row["page_count"])
        col3.metric("Author", selected_row["author"] if selected_row["author"] else "Unknown")

        raw_text = selected_row["full_text"] or ""

        if not raw_text.strip():
            st.info("This document has no extracted text.")
            return

        preview_text = raw_text[:3000]

        cleaned_text = clean_text(raw_text)
        raw_tokens = tokenize_text(cleaned_text)
        tokens_no_stop = remove_stopwords(raw_tokens)
        stemmed_tokens = stem_tokens(tokens_no_stop)
        lemmatized_tokens = lemmatize_tokens(tokens_no_stop)

        preview_cleaned = clean_text(preview_text)
        preview_tokens = tokenize_text(preview_cleaned)
        preview_tokens_no_stop = remove_stopwords(preview_tokens)
        preview_stemmed_tokens = stem_tokens(preview_tokens_no_stop)
        preview_lemmatized_tokens = lemmatize_tokens(preview_tokens_no_stop)

        st.markdown("### Original and Cleaned Text")
        col_a, col_b = st.columns(2)

        with col_a:
            st.text_area(
                "Original text",
                preview_text[:2000],
                height=220,
                key="ml_original_text",
            )

        with col_b:
            st.text_area(
                "Cleaned text",
                preview_cleaned[:2000],
                height=220,
                key="ml_cleaned_text",
            )

        st.markdown("### Token Summary")
        col4, col5, col6, col7 = st.columns(4)
        col4.metric("Raw Tokens", len(raw_tokens))
        col5.metric("After Stopword Removal", len(tokens_no_stop))
        col6.metric("Stemmed Tokens", len(stemmed_tokens))
        col7.metric("Lemmatized Tokens", len(lemmatized_tokens))

        st.markdown("### Token-Level Preview")
        col_left, col_right = st.columns(2)

        with col_left:
            st.text_area(
                "Tokens",
                _format_token_preview(preview_tokens),
                height=220,
                key="ml_tokens_preview",
            )

            st.text_area(
                "Tokens without stopwords",
                _format_token_preview(preview_tokens_no_stop),
                height=220,
                key="ml_tokens_no_stop_preview",
            )

        with col_right:
            st.text_area(
                "Stemmed tokens",
                _format_token_preview(preview_stemmed_tokens),
                height=220,
                key="ml_stemmed_tokens_preview",
            )

            st.text_area(
                "Lemmatized tokens",
                _format_token_preview(preview_lemmatized_tokens),
                height=220,
                key="ml_lemmatized_tokens_preview",
            )

        st.markdown("### Reconstructed Outputs")
        col8, col9 = st.columns(2)

        with col8:
            st.text_area(
                "Stemmed text",
                " ".join(preview_stemmed_tokens)[:2000],
                height=220,
                key="ml_stemmed_text",
            )

        with col9:
            st.text_area(
                "Lemmatized text",
                " ".join(preview_lemmatized_tokens)[:2000],
                height=220,
                key="ml_lemmatized_text",
            )

        # ------------------------------------------------------------------
        # TF-IDF feature analysis
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("TF-IDF Feature Analysis")

        method = st.selectbox(
            "Preprocessing method for TF-IDF",
            options=["lemmatize", "stem", "none"],
            index=0,
            key="tfidf_preprocess_method",
        )

        max_features = st.slider(
            "Maximum TF-IDF features",
            min_value=500,
            max_value=5000,
            value=3000,
            step=250,
            key="tfidf_max_features",
        )

        processed_texts = [
            preprocess_text(text or "", method=method)
            for text in corpus_df["full_text"].fillna("").tolist()
        ]

        vectorizer, X = build_tfidf_features(processed_texts, max_features=max_features)

        selected_processed_text = processed_texts[selected_doc_index]
        selected_nonzero_terms = int(X[selected_doc_index].count_nonzero())
        selected_word_count = len(selected_processed_text.split())

        st.markdown("### TF-IDF Corpus Summary")
        col10, col11, col12, col13 = st.columns(4)
        col10.metric("Documents", X.shape[0])
        col11.metric("Vocabulary Size", X.shape[1])
        col12.metric("Selected Doc Words", selected_word_count)
        col13.metric("Nonzero TF-IDF Terms", selected_nonzero_terms)

        top_terms = get_top_tfidf_terms(vectorizer, X[selected_doc_index], top_n=20)

        st.markdown("### Top TF-IDF Terms for Selected Document")

        if top_terms:
            top_terms_df = pd.DataFrame(top_terms, columns=["term", "score"])
            st.dataframe(top_terms_df, use_container_width=True)

            st.markdown("**Top terms preview**")
            st.write(", ".join(top_terms_df["term"].tolist()))
        else:
            st.info("No TF-IDF terms found for the selected document.")

        with st.expander("Show sample TF-IDF vocabulary"):
            vocab_sample = vectorizer.get_feature_names_out()[:100]
            st.write(list(vocab_sample))

        # ------------------------------------------------------------------
        # K-Means clustering
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Clustering Analysis")

        clustering_method = st.selectbox(
            "Preprocessing method for clustering",
            options=["lemmatize", "stem", "none"],
            index=0,
            key="clustering_preprocess_method",
        )

        clustering_max_features = st.slider(
            "Maximum TF-IDF features for clustering",
            min_value=500,
            max_value=5000,
            value=2000,
            step=250,
            key="clustering_max_features",
        )

        dataset_df = prepare_document_dataset(conn, method=clustering_method)
        _, clustering_X = build_tfidf_matrix(
            dataset_df,
            text_column="processed_text",
            max_features=clustering_max_features,
        )

        st.markdown("### Elbow Method Summary")
        elbow_df = compute_elbow_scores(
            clustering_X,
            k_values=range(2, min(9, len(dataset_df))),
        )
        st.dataframe(elbow_df, use_container_width=True)

        default_k = 3 if len(dataset_df) > 3 else 2
        max_k = min(8, len(dataset_df) - 1)

        if max_k < 2:
            st.info("Not enough documents for clustering.")
            return

        selected_k = st.slider(
            "Choose number of clusters (K)",
            min_value=2,
            max_value=max_k,
            value=min(default_k, max_k),
            step=1,
            key="selected_k_clusters",
        )

        clustered_df, _ = run_kmeans_clustering(
            dataset_df,
            clustering_X,
            n_clusters=selected_k,
        )

        cluster_summary_df = summarize_clusters(clustered_df, cluster_column="cluster")

        st.markdown("### Cluster Summary")
        st.dataframe(cluster_summary_df, use_container_width=True)

        selected_cluster = st.selectbox(
            "Select cluster to inspect",
            options=sorted(clustered_df["cluster"].unique().tolist()),
            key="selected_cluster_inspect",
        )

        cluster_members_df = get_cluster_members(
            clustered_df,
            cluster_id=selected_cluster,
            cluster_column="cluster",
        )

        st.markdown(f"### Documents in Cluster {selected_cluster}")
        st.dataframe(cluster_members_df, use_container_width=True)

        coords, _ = project_pca_2d(clustering_X)
        clustered_with_pca_df = attach_pca_coordinates(clustered_df, coords)

        st.markdown("### PCA Scatter View")

        chart_df = clustered_with_pca_df.copy()
        chart_df["cluster"] = chart_df["cluster"].astype(str)

        st.scatter_chart(
            chart_df,
            x="pca_x",
            y="pca_y",
            color="cluster",
        )

        st.caption(
            "Each point represents one document projected into 2D PCA space and colored by K-Means cluster."
        )

        st.markdown("### PCA Projection Data")
        st.dataframe(
            clustered_with_pca_df[
                ["document_id", "display_title", "cluster", "pca_x", "pca_y"]
            ],
            use_container_width=True,
        )

        # ------------------------------------------------------------------
        # Hierarchical clustering
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Hierarchical Clustering")

        linkage_method = st.selectbox(
            "Linkage method",
            options=["ward", "complete", "average", "single"],
            index=0,
            key="hierarchical_linkage_method",
        )

        hierarchical_k = st.slider(
            "Choose number of hierarchical clusters",
            min_value=2,
            max_value=max_k,
            value=min(3, max_k),
            step=1,
            key="hierarchical_selected_k",
        )

        hierarchical_df = run_hierarchical_clustering(
            dataset_df,
            clustering_X,
            n_clusters=hierarchical_k,
            linkage_method=linkage_method,
        )

        hierarchical_summary_df = summarize_clusters(
            hierarchical_df,
            cluster_column="hierarchical_cluster",
        )

        st.markdown("### Hierarchical Cluster Summary")
        st.dataframe(hierarchical_summary_df, use_container_width=True)

        selected_h_cluster = st.selectbox(
            "Select hierarchical cluster to inspect",
            options=sorted(hierarchical_df["hierarchical_cluster"].unique().tolist()),
            key="selected_hierarchical_cluster_inspect",
        )

        hierarchical_members_df = get_cluster_members(
            hierarchical_df,
            cluster_id=selected_h_cluster,
            cluster_column="hierarchical_cluster",
        )

        st.markdown(f"### Documents in Hierarchical Cluster {selected_h_cluster}")
        st.dataframe(hierarchical_members_df, use_container_width=True)

        st.markdown("### Dendrogram")

        dendro_labels = [
            f"{row.document_id}: {str(row.display_title)[:35]}"
            for row in hierarchical_df.itertuples()
        ]

        dendro_data = build_dendrogram_data(
            clustering_X,
            labels=dendro_labels,
            method=linkage_method,
        )

        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(
            dendro_data["linkage_matrix"],
            labels=dendro_labels,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax,
        )
        ax.set_title(
            f"Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Linkage)"
        )
        ax.set_xlabel("Documents")
        ax.set_ylabel("Distance")

        st.pyplot(fig, clear_figure=True)

        st.caption(
            "The dendrogram shows how documents merge into larger groups as the distance threshold increases."
        )

        # ------------------------------------------------------------------
        # Heuristic label generation
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Heuristic Label Generation")

        st.write(
            "Generate a draft label file based on simple keyword heuristics. "
            "This is for bootstrapping and review, not final ground truth."
        )

        label_output_path = st.text_input(
            "Output CSV path",
            value="data/labels_generated.csv",
            key="heuristic_labels_output_path",
        )

        preview_chars = st.slider(
            "Text preview size for heuristic scan",
            min_value=500,
            max_value=8000,
            value=4000,
            step=250,
            key="heuristic_preview_chars",
        )

        if st.button("Generate Draft Labels", key="generate_draft_labels_button"):
            labels_generated_df = generate_heuristic_labels(
                dataset_df,
                output_path=label_output_path,
                preview_chars=preview_chars,
            )

            st.success(f"Draft labels generated and saved to: {label_output_path}")
            st.dataframe(labels_generated_df, use_container_width=True)

            label_counts_df = (
                labels_generated_df["label"]
                .value_counts()
                .rename_axis("label")
                .reset_index(name="count")
            )

            st.markdown("### Draft Label Distribution")
            st.dataframe(label_counts_df, use_container_width=True)

            csv_data = labels_generated_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Draft Labels CSV",
                data=csv_data,
                file_name="labels_generated.csv",
                mime="text/csv",
                key="download_draft_labels_csv",
            )

        # ------------------------------------------------------------------
        # Label-based classification
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Label-Based Classification")

        st.write(
            "Train and compare supervised classifiers using reviewed labels or generated labels. "
            "Reviewed labels are recommended for more meaningful evaluation."
        )

        label_source = st.radio(
            "Choose label source",
            options=["Reviewed labels", "Generated labels", "Custom path"],
            horizontal=True,
            key="classification_label_source",
        )

        if label_source == "Reviewed labels":
            default_path = "data/labels_reviewed.csv"
        elif label_source == "Generated labels":
            default_path = "data/labels_generated.csv"
        else:
            default_path = "data/labels_reviewed.csv"

        classification_labels_path = st.text_input(
            "Label CSV path for classification",
            value=default_path,
            key="classification_labels_path",
        )

        classification_method = st.selectbox(
            "Preprocessing method for classification",
            options=["lemmatize", "stem", "none"],
            index=0,
            key="classification_preprocess_method",
        )

        classification_max_features = st.slider(
            "Maximum TF-IDF features for classification",
            min_value=500,
            max_value=5000,
            value=3000,
            step=250,
            key="classification_max_features",
        )

        if st.button("Run Classification", key="run_classification_button"):
            try:
                labeled_df = prepare_labeled_dataset(
                    conn,
                    labels_path=classification_labels_path,
                    method=classification_method,
                )

                st.markdown("### Labeled Dataset Summary")
                col_a1, col_a2, col_a3 = st.columns(3)
                col_a1.metric("Labeled Documents", len(labeled_df))
                col_a2.metric("Classes", labeled_df["label"].nunique())
                col_a3.metric("Avg Processed Length", round(labeled_df["text_length"].mean(), 2))

                label_dist_df = (
                    labeled_df["label"]
                    .value_counts()
                    .rename_axis("label")
                    .reset_index(name="count")
                )

                st.markdown("### Label Distribution")
                st.dataframe(label_dist_df, use_container_width=True)

                results_df, artifacts = train_and_evaluate_models(
                    labeled_df,
                    test_size=0.2,
                    max_features=classification_max_features,
                )

                st.markdown("### Model Comparison")
                st.dataframe(results_df, use_container_width=True)
                st.bar_chart(results_df.set_index("model")["f1_score"])

                best_model_name = results_df.iloc[0]["model"]
                best_model_artifacts = artifacts[best_model_name]

                st.success(f"Best model: {best_model_name}")

                st.markdown("### Confusion Matrix")
                classes = sorted(labeled_df["label"].unique())
                cm = best_model_artifacts["confusion_matrix"]
                cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                st.dataframe(cm_df, use_container_width=True)

                st.markdown("### Classification Report")
                st.code(best_model_artifacts["classification_report"])

            except Exception as exc:
                st.error(f"Classification failed: {exc}")

        # ------------------------------------------------------------------
        # Named Entity Recognition (NER)
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Named Entity Recognition (NER)")

        st.write(
            "Extract named entities such as people, organizations, dates, and locations "
            "from the selected document. You can preview them and optionally store them in SQLite."
        )

        ner_preview_chars = st.slider(
            "Text length for NER extraction",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=500,
            key="ner_preview_chars",
        )

        ner_text = raw_text[:ner_preview_chars]

        col_ner1, col_ner2, col_ner3 = st.columns(3)
        run_ner_clicked = col_ner1.button("Run NER", key="run_ner_button")
        save_ner_clicked = col_ner2.button("Save NER to DB", key="save_ner_button")
        load_saved_ner_clicked = col_ner3.button("Load Saved NER", key="load_saved_ner_button")

        if run_ner_clicked or save_ner_clicked:
            try:
                nlp = get_ner_model()
                entities = extract_entities_from_document(ner_text, nlp)
                summary = summarize_entities(entities)

                st.markdown("### Extracted Entities")
                if summary:
                    summary_df = pd.DataFrame(summary)
                    st.dataframe(summary_df, use_container_width=True)

                    label_counts = (
                        summary_df.groupby("entity_label")["count"]
                        .sum()
                        .reset_index()
                        .sort_values("count", ascending=False)
                    )

                    st.markdown("### Entity Counts by Label")
                    st.dataframe(label_counts, use_container_width=True)

                    if save_ner_clicked:
                        delete_entities_by_doc_id(int(selected_row["document_id"]))
                        inserted_count = insert_entities(
                            doc_id=int(selected_row["document_id"]),
                            page_number=None,
                            entities=entities,
                        )
                        st.success(f"Saved {inserted_count} entities to the database.")
                else:
                    st.info("No supported entities found in the selected text.")
            except Exception as exc:
                st.error(f"NER failed: {exc}")

        if load_saved_ner_clicked:
            try:
                stored_entities = get_entities_by_doc_id(int(selected_row["document_id"]))
                entity_summary = get_entity_summary_by_doc_id(int(selected_row["document_id"]))
                entity_counts = get_entity_counts_by_label(int(selected_row["document_id"]))

                st.markdown("### Stored Entities")
                if stored_entities:
                    st.dataframe(
                        pd.DataFrame([dict(row) for row in stored_entities]),
                        use_container_width=True,
                    )

                    st.markdown("### Stored Entity Summary")
                    st.dataframe(
                        pd.DataFrame([dict(row) for row in entity_summary]),
                        use_container_width=True,
                    )

                    st.markdown("### Stored Entity Counts by Label")
                    st.dataframe(
                        pd.DataFrame([dict(row) for row in entity_counts]),
                        use_container_width=True,
                    )
                else:
                    st.info("No saved entities found for this document yet.")
            except Exception as exc:
                st.error(f"Loading saved NER failed: {exc}")

    finally:
        conn.close()