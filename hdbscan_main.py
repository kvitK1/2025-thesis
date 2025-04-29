from datetime import datetime
import os
import polars as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import load_data, one_hot_encode, dump_pkl, load_pkl
from umap_visualise import reduce_dimensions_umap, visualize_clusters
from scores import calculate_silhouette_score, calculate_davies_bouldin_score, calculate_calinski_harabasz_score
from hdbscan_umap import find_best_hdbscan_params
from hdbscan_validation import validate_clustering
from cluster_summary import summarize_clusters, save_cluster_summary
from cluster_distributions import plot_feature_distributions

categorical_cols = [
    'what_summary_is_answer', 'survey_answer',
    'statement_1', 'statement_2', 'statement_3',
    'question_0', 'question_1', 'question_2', 'push_allow',
    'version_grouped', 'country_grouped', 'age_grouped', 'device_grouped',
    'gender_grouped', 'quiz_duration_grouped', 'goal_adjust_grouped',
    'streak_commitment_level', 'engagement_window', 'life_desires_engagement',
    'selected_books_grouped', 'season_category'
]

multilabel_cols = ['life_desires', 'time_periods_eng']

target = ["purchase"]
random_state = 42
np.random.seed(random_state)

direct = "data/hdbscan_umap"

def create_data_dirs():
    os.makedirs(direct, exist_ok=True)
    os.makedirs(f"{direct}/hdbscan", exist_ok=True)


def main():
    create_data_dirs()

    df = load_data("data/processed_onboarding.parquet").to_pandas()

    df = (
        df[categorical_cols + multilabel_cols + target + ["product_id_grouped"]]
        .sample(frac=0.1, random_state=random_state)
        .reset_index(drop=True)
    )

    X_raw = df[categorical_cols + multilabel_cols]
    X, encoders = one_hot_encode(X_raw, categorical_cols, multilabel_cols)
    print("One-hot encoded data shape:", X.shape)
    dump_pkl(X, "X_encoded.pkl", base_path=direct)

    clustering_embeddings_size = int(np.sqrt(X.shape[1]) * 2)
    print(f"Estimated UMAP embedding dimension: {clustering_embeddings_size}")

    X_umap = load_pkl("X_umap_hdbscan.pkl", base_path=direct)
    if X_umap is None:
        X_umap = reduce_dimensions_umap(
            X,
            n_components=clustering_embeddings_size,
            metric="hamming",
            save_embedding=True,
            embedding_filename="X_umap_hdbscan.pkl"
        )

    labels, best_cluster_size, best_validity_index = find_best_hdbscan_params(X_umap.astype(np.float64))
    print(f"Best cluster size: {best_cluster_size} | Validity index: {best_validity_index}")

    with open(os.path.join(direct, "best_hdbscan_metrics.txt"), "w") as f:
        f.write(f"best_cluster_size:{best_cluster_size}, validity_index:{best_validity_index}")


    embedding_2d = load_pkl("embedding_2d.pkl", base_path=direct)
    if embedding_2d is None:
        embedding_2d = reduce_dimensions_umap(
            X,
            n_components=2,
            metric="hamming",
            save_embedding=True,
            embedding_filename="embedding_2d.pkl"
        )

    df["labels"] = labels
    df.to_parquet(f"{direct}/processed_onboarding_hdbscan_clustered.parquet")
    print(f"Clustered data saved to {direct}/processed_onboarding_hdbscan_clustered.parquet")

    visualize_clusters(embedding_2d, labels, save_plot=True, filename=os.path.join(direct, "hdbscan_clusters.png"))

    silhouette = calculate_silhouette_score(X_umap, labels)
    davies_bouldin = calculate_davies_bouldin_score(X_umap, labels)
    ch_score = calculate_calinski_harabasz_score(X_umap, labels)
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
    print(f"Calinski-Harabasz Score: {ch_score:.3f}")

    pd.DataFrame([{
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "calinski_harabasz_index": ch_score,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
    }]).to_csv(os.path.join(direct, "hdbscan_validation_metrics.csv"), index=False)

    summary = summarize_clusters(df, label_col="labels", categorical_cols=categorical_cols, target="purchase")
    save_cluster_summary(summary, algorithm_name="hdbscan", path=f"{direct}")

    plot_feature_distributions(df, cluster_col="labels", categorical_cols=categorical_cols+multilabel_cols, algorithm_name="hdbscan")

    distribution_text = ""
    for label, group in df.groupby("labels"):
        distribution_text += f"Cluster {label}:\n"
        percents = group["product_id_grouped"].value_counts(normalize=True).mul(100).round(2)
        for product, percent in percents.items():
            distribution_text += f"  - {product}: {percent:.2f}%\n"
        distribution_text += "\n"
    

    plot_feature_distributions(
        df=df,
        cluster_col="labels",
        categorical_cols=["product_id_grouped"],
        algorithm_name="hdbscan",
        base_path="data/cluster_distributions"
    )

    txt_file = os.path.join(direct, "product_id_grouped_distribution.txt")
    with open(txt_file, "w") as f:
        f.write(distribution_text)
    print(f"Product ID distribution text saved to {txt_file}")

    subsampling_results = validate_clustering(X_umap, labels, best_cluster_size)

    print("\nSubsampling Validation:")
    print(f"Average ARI: {np.mean([r[0] for r in subsampling_results]):.3f} ",
          f"Std Dev: {np.std([r[0] for r in subsampling_results]):.3f}")
    print(f"Average NMI: {np.mean([r[1] for r in subsampling_results]):.3f} ",
          f"Std Dev: {np.std([r[1] for r in subsampling_results]):.3f}")

if __name__ == "__main__":
    main()