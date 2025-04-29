import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from cluster_distributions import plot_feature_distributions
from cluster_summary import summarize_clusters
from preprocess import load_data, one_hot_encode, dump_pkl, load_pkl
from umap_visualise import reduce_dimensions_umap, visualize_clusters
from scores import calculate_silhouette_score, calculate_davies_bouldin_score, calculate_calinski_harabasz_score

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
columns_to_select = categorical_cols + multilabel_cols + target + ["product_id_grouped"]

random_state = 42
np.random.seed(random_state)
direct = "data/spectral"

def create_data_dirs():
    os.makedirs(direct, exist_ok=True)

def evaluate_clustering(feature_data, labels, metric="cosine"):
    unique_labels = np.unique(labels)
    mask = labels != -1
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return {}
    else:
        if metric == "cosine":
            distance_matrix = 1 - cosine_similarity(feature_data)
            distance_matrix = np.clip(distance_matrix, 0, 1)
            silhouette = calculate_silhouette_score(distance_matrix, labels[mask], metric="precomputed")
        else:
            silhouette = calculate_silhouette_score(feature_data, labels, metric=metric)
        davies_bouldin = calculate_davies_bouldin_score(feature_data, labels)
        ch_score = calculate_calinski_harabasz_score(feature_data, labels)
        return {
            "silhouette_score": silhouette,
            "davies_bouldin_index": davies_bouldin,
            "calinski_harabasz_score": ch_score,
            "n_clusters": len(unique_labels),
        }

def run_spectral_clustering_precomputed(affinity_matrix: np.ndarray, n_clusters: int = 2):
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state,
        n_init=10,
        assign_labels='discretize',
        verbose=False
    )
    labels = model.fit_predict(affinity_matrix)
    dump_pkl(labels, f"labels_k{n_clusters}.pkl", base_path=direct)
    return labels

def find_optimal_n_clusters_precomputed(feature_data: np.ndarray, affinity_matrix: np.ndarray, max_clusters: int = 10, metric: str = "cosine"):
    cache_file = f"optimal_n_clusters_max{max_clusters}.pkl"
    cached = load_pkl(cache_file, base_path=direct)
    if cached is not None:
        return cached

    silhouette_scores = []
    davies_bouldin_scores = []
    n_clusters_range = range(2, max_clusters + 1)

    for n_clusters in n_clusters_range:
        labels = run_spectral_clustering_precomputed(affinity_matrix, n_clusters)
        eval_metrics = evaluate_clustering(feature_data, labels, metric=metric)
        if eval_metrics:
            silhouette_scores.append(eval_metrics['silhouette_score'])
            davies_bouldin_scores.append(eval_metrics['davies_bouldin_index'])
        else:
            silhouette_scores.append(-1)
            davies_bouldin_scores.append(np.inf)

    valid_indices = [i for i, score in enumerate(silhouette_scores) if score is not None and score > -1]
    if not valid_indices:
        optimal_n_clusters = 3
    else:
        best_silhouette_index = valid_indices[np.argmax([silhouette_scores[i] for i in valid_indices])]
        optimal_n_clusters = n_clusters_range[best_silhouette_index]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(n_clusters_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(n_clusters_range, davies_bouldin_scores, marker='o')
    plt.title('Davies-Bouldin Index vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Index')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(direct, f"cluster_evaluation_metrics_max{max_clusters}.png"))

    dump_pkl(optimal_n_clusters, cache_file, base_path=direct)
    return optimal_n_clusters

def visualize_spectral_clusters_umap(feature_data: np.ndarray, labels: np.ndarray, n_components: int = 2, metric: str = "cosine", save_path: str = "spectral_clusters_umap_viz.png"):
    embedding = reduce_dimensions_umap(
        feature_data,
        n_components=n_components,
        metric=metric,
        save_embedding=False
    )
    visualize_clusters(
        embedding,
        labels,
        title="UMAP Visualization of Spectral Clusters (Precomputed Cosine)",
        save_plot=True,
        filename=save_path
    )
    silhouette_2d = calculate_silhouette_score(embedding, labels, metric="euclidean")
    davies_bouldin_2d = calculate_davies_bouldin_score(embedding, labels)
    pd.DataFrame([{
        "silhouette_score_2d_umap": silhouette_2d,
        "davies_bouldin_index_2d_umap": davies_bouldin_2d,
        "n_clusters": len(set(labels)),
    }]).to_csv(os.path.join(direct, "spectral_validation_metrics_2d_umap.csv"), index=False)

def main():
    create_data_dirs()
    df = load_data("data/processed_onboarding.parquet").to_pandas()
    sample_frac = 0.01
    df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    X_raw = df[categorical_cols + multilabel_cols]
    X, encoders = one_hot_encode(X_raw, categorical_cols, multilabel_cols)
    X = X.astype(np.float64)
    dump_pkl(X, "X_encoded.pkl", base_path=direct)

    assert not np.isnan(X).any(), "X contains NaN"
    assert np.isfinite(X).all(), "X contains Inf"

    cosine_matrix_path = os.path.join(direct, "cosine_sim_matrix.pkl")
    if os.path.exists(cosine_matrix_path):
        cosine_sim_matrix = load_pkl("cosine_sim_matrix.pkl", base_path=direct)
        if cosine_sim_matrix is None or cosine_sim_matrix.shape[0] != X.shape[0]:
            cosine_sim_matrix = None
    else:
        cosine_sim_matrix = None

    if cosine_sim_matrix is None:
        cosine_sim_matrix = cosine_similarity(X, dense_output=True)
        cosine_sim_matrix = np.clip((cosine_sim_matrix + cosine_sim_matrix.T) / 2, 0, 1).astype(np.float32)
        dump_pkl(cosine_sim_matrix, "cosine_sim_matrix.pkl", base_path=direct)

    max_k = 10
    optimal_n_clusters = find_optimal_n_clusters_precomputed(X, cosine_sim_matrix, max_clusters=max_k, metric="cosine")
    spectral_labels = run_spectral_clustering_precomputed(cosine_sim_matrix, n_clusters=optimal_n_clusters)
    dump_pkl(spectral_labels, "final_labels.pkl", base_path=direct)

    df["cluster"] = spectral_labels
    df.to_parquet(os.path.join(direct, "processed_onboarding_spectral_clustered.parquet"))

    eval_metrics_spectral = evaluate_clustering(X, spectral_labels, metric="cosine")
    if eval_metrics_spectral:
        pd.DataFrame([eval_metrics_spectral]).to_csv(os.path.join(direct, "final_evaluation_metrics.csv"), index=False)

    visualize_spectral_clusters_umap(
        feature_data=X,
        labels=spectral_labels,
        n_components=2,
        metric="cosine",
        save_path=os.path.join(direct, f"spectral_clusters_umap_viz.png")
    )

    dist_base_path = 'data/cluster_distributions/spectral'

    os.makedirs(dist_base_path, exist_ok=True)

    plot_feature_distributions(
        df=df,
        cluster_col="cluster",
        categorical_cols=categorical_cols + multilabel_cols,
        algorithm_name=f"spectral",
        base_path=dist_base_path
    )

    plot_feature_distributions(
        df=df,
        cluster_col="cluster",
        categorical_cols=["product_id_grouped"],
        algorithm_name=f"spectral",
        base_path=dist_base_path
    )

    distribution_text = ""
    for label, group in df.groupby("cluster"):
        distribution_text += f"Cluster {label}:\n"
        if group.empty:
            distribution_text += "  - No users in this cluster.\n"
            continue
        percents = group["product_id_grouped"].value_counts(normalize=True).mul(100)
        for product, percent in percents.items():
            distribution_text += f"  - {product}: {percent:.2f}%\n"
        distribution_text += "\n"

    txt_file = os.path.join(direct, f"product_id_grouped_distribution.txt")
    with open(txt_file, "w") as f:
        f.write(distribution_text)

    summary_df = summarize_clusters(df, label_col="cluster", categorical_cols=categorical_cols)
    summary_csv = os.path.join(direct, f"summary_spectral.csv")
    summary_df.to_csv(summary_csv, index=False)

if __name__ == "__main__":
    main()