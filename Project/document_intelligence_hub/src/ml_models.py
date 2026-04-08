from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.cluster.hierarchy import dendrogram, linkage

from src.nlp_pipeline import fetch_document_corpus, preprocess_text

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Corpus preparation
# -------------------------------------------------------------------


def prepare_document_dataset(
    conn: Any,
    method: str = "lemmatize",
) -> pd.DataFrame:
    """
    Load all documents from SQLite and prepare processed text for ML tasks.
    This is the main entry point for unsupervised Phase 2 work.
    """
    corpus = fetch_document_corpus(conn)
    corpus_df = pd.DataFrame(corpus)

    if corpus_df.empty:
        raise ValueError("No documents found in the database.")

    corpus_df["full_text"] = corpus_df["full_text"].fillna("")
    corpus_df["processed_text"] = corpus_df["full_text"].apply(
        lambda text: preprocess_text(text, method=method)
    )
    corpus_df["text_length"] = corpus_df["processed_text"].apply(lambda x: len(x.split()))

    logger.info("Prepared document dataset with %s documents.", len(corpus_df))
    return corpus_df


def build_tfidf_matrix(
    dataset_df: pd.DataFrame,
    text_column: str = "processed_text",
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
):
    """
    Build TF-IDF matrix for clustering or classification.
    """
    if dataset_df.empty:
        raise ValueError("Dataset is empty.")

    if text_column not in dataset_df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    texts = dataset_df[text_column].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    X = vectorizer.fit_transform(texts)

    logger.info("Built TF-IDF matrix with shape %s", X.shape)
    return vectorizer, X


# -------------------------------------------------------------------
# Unsupervised learning: K-Means
# -------------------------------------------------------------------


def compute_elbow_scores(
    X,
    k_values=range(2, 9),
) -> pd.DataFrame:
    """
    Compute inertia and silhouette score for multiple K values.
    Used to support Elbow Method analysis.
    """
    rows = []

    for k in k_values:
        if k >= X.shape[0]:
            continue

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(X)

        row = {
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_score": None,
        }

        unique_labels = len(set(cluster_labels))
        if unique_labels > 1:
            try:
                row["silhouette_score"] = float(silhouette_score(X, cluster_labels))
            except Exception as exc:
                logger.warning("Silhouette score failed for k=%s: %s", k, exc)

        rows.append(row)

    if not rows:
        raise ValueError("Not enough documents to compute elbow scores.")

    return pd.DataFrame(rows)


def run_kmeans_clustering(
    dataset_df: pd.DataFrame,
    X,
    n_clusters: int = 4,
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Run K-Means clustering and attach cluster labels to the dataset.
    """
    if dataset_df.empty:
        raise ValueError("Dataset is empty.")

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    if n_clusters >= len(dataset_df):
        raise ValueError("n_clusters must be smaller than the number of documents.")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X)

    result_df = dataset_df.copy()
    result_df["cluster"] = cluster_labels

    logger.info("K-Means clustering complete with %s clusters.", n_clusters)
    return result_df, model


# -------------------------------------------------------------------
# PCA projection
# -------------------------------------------------------------------


def project_pca_2d(X) -> Tuple[np.ndarray, PCA]:
    """
    Reduce TF-IDF features to 2 dimensions for visualization.
    """
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 documents for PCA projection.")

    dense_X = X.toarray()
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(dense_X)

    logger.info(
        "PCA projection complete. Explained variance ratio: %s",
        pca.explained_variance_ratio_,
    )
    return coords, pca


def attach_pca_coordinates(
    dataset_df: pd.DataFrame,
    coords: np.ndarray,
) -> pd.DataFrame:
    """
    Add PCA coordinates to a dataframe for plotting or inspection.
    """
    if len(dataset_df) != len(coords):
        raise ValueError("Dataset length and PCA coordinate length do not match.")

    result_df = dataset_df.copy()
    result_df["pca_x"] = coords[:, 0]
    result_df["pca_y"] = coords[:, 1]
    return result_df


# -------------------------------------------------------------------
# Hierarchical clustering
# -------------------------------------------------------------------


def run_hierarchical_clustering(
    dataset_df: pd.DataFrame,
    X,
    n_clusters: int = 4,
    linkage_method: str = "ward",
) -> pd.DataFrame:
    """
    Run Agglomerative (hierarchical) clustering and attach labels.
    """
    if dataset_df.empty:
        raise ValueError("Dataset is empty.")

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    dense_X = X.toarray()

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
    )
    cluster_labels = model.fit_predict(dense_X)

    result_df = dataset_df.copy()
    result_df["hierarchical_cluster"] = cluster_labels

    logger.info(
        "Hierarchical clustering complete with %s clusters using %s linkage.",
        n_clusters,
        linkage_method,
    )
    return result_df


def compute_linkage_matrix(
    X,
    method: str = "ward",
):
    """
    Compute linkage matrix for dendrogram plotting.
    """
    dense_X = X.toarray()
    return linkage(dense_X, method=method)


def build_dendrogram_data(
    X,
    labels: list[str] | None = None,
    method: str = "ward",
) -> Dict[str, Any]:
    """
    Produce dendrogram data for later plotting in Streamlit / matplotlib.
    """
    Z = compute_linkage_matrix(X, method=method)
    dendro = dendrogram(Z, labels=labels, no_plot=True)

    return {
        "linkage_matrix": Z,
        "icoord": dendro["icoord"],
        "dcoord": dendro["dcoord"],
        "ivl": dendro["ivl"],
        "leaves": dendro["leaves"],
    }


# -------------------------------------------------------------------
# Cluster inspection helpers
# -------------------------------------------------------------------


def summarize_clusters(
    clustered_df: pd.DataFrame,
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    """
    Return simple cluster counts and average document length.
    """
    if cluster_column not in clustered_df.columns:
        raise ValueError(f"Missing cluster column: {cluster_column}")

    summary_df = (
        clustered_df.groupby(cluster_column)
        .agg(
            document_count=("document_id", "count"),
            avg_text_length=("text_length", "mean"),
        )
        .reset_index()
        .sort_values(cluster_column)
    )

    summary_df["avg_text_length"] = summary_df["avg_text_length"].round(2)
    return summary_df


def get_cluster_members(
    clustered_df: pd.DataFrame,
    cluster_id: int,
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    """
    Return the documents belonging to one cluster.
    """
    if cluster_column not in clustered_df.columns:
        raise ValueError(f"Missing cluster column: {cluster_column}")

    cols = [
        col for col in [
            "document_id",
            "display_title",
            "file_name",
            "author",
            "page_count",
            "text_length",
            cluster_column,
        ]
        if col in clustered_df.columns
    ]

    return clustered_df[clustered_df[cluster_column] == cluster_id][cols].reset_index(drop=True)


# -------------------------------------------------------------------
# Optional: heuristic auto-label generation
# -------------------------------------------------------------------

def generate_heuristic_labels(
    dataset_df: pd.DataFrame,
    output_path: str = "data/labels_generated.csv",
    preview_chars: int = 4000,
) -> pd.DataFrame:
    """
    Generate draft labels using simple keyword heuristics.
    This is meant for bootstrapping, not final ground truth.
    """
    if dataset_df.empty:
        raise ValueError("Dataset is empty.")

    rows = []

    for _, row in dataset_df.iterrows():
        text = (row.get("full_text", "") or "")[:preview_chars].lower()

        # Research papers first
        if (
            "abstract" in text
            or ("introduction" in text and "conclusion" in text)
            or ("references" in text and "doi" in text)
            or ("literature review" in text)
        ):
            label = "research_paper"

        # Technical reports
        elif (
            "technical" in text
            or "specification" in text
            or "model report" in text
            or "project appraisal document" in text
            or "implementation completion and results report" in text
        ):
            label = "technical_report"

        # Policy / legal / formal guidance docs
        elif (
            "policy" in text
            or "framework" in text
            or "directive" in text
            or "act" in text
            or "code" in text
            or "guidance note" in text
            or "hereby" in text
            or "plaintiff" in text
            or "jurisdiction" in text
        ):
            label = "policy_document"

        else:
            label = "general_document"

        rows.append(
            {
                "document_id": int(row["document_id"]),
                "display_title": row.get("display_title", ""),
                "label": label,
            }
        )

    labels_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_path, index=False)

    logger.info("Generated heuristic labels for %s documents at %s", len(labels_df), output_path)
    return labels_df


# -------------------------------------------------------------------
# Optional supervised classification section
# Kept here for later reuse after you finish cluster-based discovery.
# -------------------------------------------------------------------


def load_labels(labels_path: str = "data/labels.csv") -> pd.DataFrame:
    path = Path(labels_path)

    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels_df = pd.read_csv(path)

    required_columns = {"document_id", "label"}
    missing = required_columns - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")

    labels_df = labels_df.dropna(subset=["document_id", "label"]).copy()
    labels_df["document_id"] = labels_df["document_id"].astype(int)
    labels_df["label"] = labels_df["label"].astype(str).str.strip()

    return labels_df


def prepare_labeled_dataset(
    conn: Any,
    labels_path: str = "data/labels.csv",
    method: str = "lemmatize",
) -> pd.DataFrame:
    dataset_df = prepare_document_dataset(conn, method=method)
    labels_df = load_labels(labels_path)

    merged_df = dataset_df.merge(labels_df, on="document_id", how="inner")

    if merged_df.empty:
        raise ValueError("No labeled documents matched the current database corpus.")

    return merged_df


def get_model_registry(random_state: int = 42) -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
        "SVM": SVC(kernel="linear", probability=True, random_state=random_state),
        "Naive Bayes": MultinomialNB(),
    }


def _resolve_test_size(
    num_samples: int,
    num_classes: int,
    requested_test_size: float,
) -> float:
    min_test_samples = num_classes
    requested_test_samples = math.ceil(num_samples * requested_test_size)
    final_test_samples = max(min_test_samples, requested_test_samples)

    if final_test_samples >= num_samples:
        raise ValueError(
            "Dataset is too small for a stratified train/test split. "
            f"Samples={num_samples}, classes={num_classes}. "
            "Add more labeled documents or reduce the number of classes."
        )

    return final_test_samples / num_samples


def train_and_evaluate_models(
    dataset_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 3000,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    if dataset_df.empty:
        raise ValueError("Dataset is empty.")

    if "processed_text" not in dataset_df.columns or "label" not in dataset_df.columns:
        raise ValueError("Dataset must contain 'processed_text' and 'label' columns.")

    label_counts = dataset_df["label"].value_counts()
    num_classes = len(label_counts)
    num_samples = len(dataset_df)

    if num_classes < 2:
        raise ValueError("Need at least 2 classes for classification.")

    if (label_counts < 2).any():
        raise ValueError(
            "Each class must have at least 2 samples for train/test split. "
            f"Current counts: {label_counts.to_dict()}"
        )

    adjusted_test_size = _resolve_test_size(
        num_samples=num_samples,
        num_classes=num_classes,
        requested_test_size=test_size,
    )

    X = dataset_df["processed_text"].tolist()
    y = dataset_df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=adjusted_test_size,
        random_state=random_state,
        stratify=y,
    )

    results = []
    artifacts: Dict[str, Dict[str, Any]] = {}

    for model_name, model in get_model_registry(random_state=random_state).items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
                ("clf", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        results.append(
            {
                "model": model_name,
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1_score), 4),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )

        artifacts[model_name] = {
            "pipeline": pipeline,
            "y_test": list(y_test),
            "y_pred": list(y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test,
                y_pred,
                zero_division=0,
            ),
        }

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return results_df, artifacts