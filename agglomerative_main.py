from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from preprocess import load_data, dump_pkl, load_pkl
from scores import calculate_silhouette_score, calculate_davies_bouldin_score, calculate_calinski_harabasz_score
from cluster_summary import summarize_clusters
from cluster_distributions import plot_feature_distributions
from preprocess import one_hot_encode
from umap_visualise import reduce_dimensions_umap, visualize_clusters

random_state = 42
np.random.seed(random_state)

categorical_cols = [
    'what_summary_is_answer', 'survey_answer',
    'statement_1', 'statement_2', 'statement_3',
    'question_0', 'question_1', 'question_2', 'push_allow',
    'version_grouped', 'country_grouped', 'age_grouped', 'device_grouped',
    'gender_grouped', 'quiz_duration_grouped', 'goal_adjust_grouped',
    'streak_commitment_level', 'engagement_window', 'life_desires_engagement',
    'selected_books_grouped', 'season_category'
]

target = ["purchase"]

agglomerative_dir = "data/agglomerative_hamming"

def create_data_dirs():
    os.makedirs(agglomerative_dir, exist_ok=True)

def compute_hamming_distance(df: pd.DataFrame) -> np.ndarray:
    X, encoders = one_hot_encode(df, categorical_cols)
    return squareform(pdist(X, metric="hamming"))

def find_best_n_clusters(distance_matrix: np.ndarray, max_k=10) -> tuple:
    scores = []
    range_n = range(2, max_k + 1)
    best_k = 2
    best_score = -np.inf
    best_labels = None

    for k in range_n:
        model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        labels = model.fit_predict(distance_matrix)
        score = calculate_silhouette_score(distance_matrix, labels, metric="precomputed")
        print(f"k={k}, silhouette={score:.3f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

        if k > 5 and len(scores) > 2 and abs(scores[-1] - scores[-2]) < 0.001:
            print(f"Early stopping at k={k}")
            break

        scores.append(score)

    plt.figure(figsize=(10, 5))
    plt.plot(range_n[:len(scores)], scores, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs Number of Clusters (Agglomerative)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(agglomerative_dir, "silhouette_scores.png"))
    plt.close()

    dump_pkl((best_k, best_labels, best_score), "best_k.pkl", base_path=agglomerative_dir)
    return best_k, best_labels, best_score

def run_agglomerative(df_full: pd.DataFrame):
    sample_idx = df_full.sample(n=10000, random_state=random_state).index
    df_clust = df_full.loc[sample_idx, categorical_cols + target].copy()
    dist_matrix = compute_hamming_distance(df_clust)

    best_k_data = load_pkl("best_k.pkl", base_path=agglomerative_dir)
    if best_k_data is None:
        best_k, labels, best_score = find_best_n_clusters(dist_matrix, max_k=8)
    else:
        best_k, labels, best_score = best_k_data

    df_clust["labels"] = labels
    df_dendro = df_full[categorical_cols].sample(n=1000, random_state=random_state).copy()
    dist_matrix_dendro = compute_hamming_distance(df_dendro)

    return df_clust, dist_matrix, labels, best_score, df_dendro, dist_matrix_dendro, sample_idx

def plot_dendrogram(distance_matrix: np.ndarray, df_dendro: pd.DataFrame, dendrogram_cutoff_distance: float = None):
    print("Plotting dendrogram...")
    linkage_matrix = linkage(distance_matrix, method="complete")

    plt.figure(figsize=(15, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=90,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True
    )

    if dendrogram_cutoff_distance is not None:
        plt.axhline(y=dendrogram_cutoff_distance, c='red', linestyle='--', label=f"Cutoff = {dendrogram_cutoff_distance}")
        plt.legend()

    plt.title("Dendrogram for Agglomerative Clustering (Hamming Distance)")
    plt.xlabel("Sample index or (cluster size)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(agglomerative_dir, "dendrogram.png"), dpi=300)
    plt.close()

    if dendrogram_cutoff_distance is not None:
        dendrogram_clusters = fcluster(linkage_matrix, t=dendrogram_cutoff_distance, criterion='distance')
        df_dendro = df_dendro.reset_index(drop=True)
        df_dendro['dendrogram_clusters'] = dendrogram_clusters
        return df_dendro
    return None

def main():
    create_data_dirs()

    df_full = load_data("data/processed_onboarding.parquet").to_pandas()

    clustered_df, dist_matrix, labels, best_score, df_dendro, dist_matrix_dendro, sample_idx = run_agglomerative(df_full)

    df_full["cluster_labels"] = pd.NA
    df_full.loc[sample_idx, "cluster_labels"] = clustered_df["labels"].values

    print(f"\nSilhouette Score (Hamming, reused): {best_score:.3f}")

    db_index = calculate_davies_bouldin_score(dist_matrix, labels)
    sil_score = calculate_silhouette_score(dist_matrix, labels, metric="precomputed")
    ch_score = calculate_calinski_harabasz_score(dist_matrix, labels)
    pd.DataFrame([{
        "silhouette_score": sil_score,
        "davies_bouldin_index": db_index,
        "calinski_harabasz_score": ch_score,
        "n_clusters": len(set(labels)),
    }]).to_csv(os.path.join(agglomerative_dir, "agglomerative_validation_metrics.csv"), index=False)

    summary_df = summarize_clusters(clustered_df, categorical_cols=categorical_cols)
    summary_df.to_csv(os.path.join(agglomerative_dir, "summary_agglomerative.csv"))
    plot_feature_distributions(clustered_df, cluster_col="labels", categorical_cols=categorical_cols, algorithm_name="agglomerative")

    if "product_id_grouped" not in clustered_df.columns and "product_id_grouped" in df_full.columns:
        clustered_df = clustered_df.merge(df_full[["product_id_grouped"]], left_index=True, right_index=True, how="left")
    
    distribution_text = ""
    for label, group in clustered_df.groupby("labels"):
        distribution_text += f"Cluster {label}:\n"
        percents = group["product_id_grouped"].value_counts(normalize=True).mul(100).round(2)
        for product, percent in percents.items():
            distribution_text += f"  - {product}: {percent:.2f}%\n"
        distribution_text += "\n"
    
    txt_file = os.path.join(agglomerative_dir, "product_id_grouped_distribution.txt")
    with open(txt_file, "w") as f:
        f.write(distribution_text)
    print(f"Product ID distribution text saved to {txt_file}")


    plot_feature_distributions(
        df=clustered_df,
        cluster_col="labels",
        categorical_cols=["product_id_grouped"],
        algorithm_name="agglomerative",
        base_path="data/cluster_distributions"
    )

    print("\n--- Starting UMAP Visualization ---")
    X_raw_clust = clustered_df[categorical_cols]
    X_clust_ohe, _ = one_hot_encode(X_raw_clust, categorical_cols, [])
    print(f"One-hot encoded data shape for UMAP: {X_clust_ohe.shape}")

    embedding_2d_filename = "embedding_2d_agg.pkl"
    embedding_2d = load_pkl(embedding_2d_filename, base_path=agglomerative_dir)

    if embedding_2d is None:
        print("Embedding not found or failed to load. Generating...")
        try:
            embedding_2d = reduce_dimensions_umap(
                X_clust_ohe,
                n_components=2,
                metric="hamming",
                random_state=random_state,
                save_embedding=True,
                embedding_filename=embedding_2d_filename,
                base_path=agglomerative_dir
            )
        except Exception as e:
            print(f"Error during UMAP embedding generation: {e}")

    print("UMAP embedding is ready for visualization.")

    umap_plot_filename = os.path.join(agglomerative_dir, "agglomerative_clusters_umap.png")

    visualize_clusters(
                embedding_2d,
                labels,
                save_plot=True,
                filename=umap_plot_filename,
                title="Agglomerative Clusters on 2D UMAP"
            )
    print("UMAP visualization saved.")
    
    dendrogram_cutoff = 4.45
    df_dendro_clustered = plot_dendrogram(dist_matrix_dendro, df_dendro.copy(), dendrogram_cutoff)

    df_full.to_parquet(f"{agglomerative_dir}/processed_onboarding_agglomerative_clustered.parquet")
    print("Agglomerative clustering complete.")

if __name__ == "__main__":
    main()