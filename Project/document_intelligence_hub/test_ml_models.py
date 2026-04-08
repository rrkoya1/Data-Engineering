from src.db import get_connection
from src.ml_models import prepare_labeled_dataset, train_and_evaluate_models

def main():
    conn = get_connection()

    dataset_df = prepare_labeled_dataset(
        conn,
        labels_path="data/labels.csv",
        method="lemmatize",
    )

    print("Labeled dataset shape:", dataset_df.shape)
    print(dataset_df[["document_id", "display_title", "label"]].head())

    results_df, artifacts = train_and_evaluate_models(dataset_df)

    print("\nModel comparison:")
    print(results_df)

    best_model = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_model}")
    print("\nClassification report:")
    print(artifacts[best_model]["classification_report"])

if __name__ == "__main__":
    main()