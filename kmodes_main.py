from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from kneed import KneeLocator
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples

from preprocess import load_data, dump_pkl, load_pkl, one_hot_encode
from umap_visualise import reduce_dimensions_umap, visualize_clusters
from scores import calculate_silhouette_score, calculate_davies_bouldin_score, calculate_calinski_harabasz_score
from cluster_summary import summarize_clusters
from cluster_distributions import plot_feature_distributions

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

columns_to_select = categorical_cols + target + ["product_id_grouped"]

kmodes_dir = "data/kmodes"

def create_data_dirs():
    os.makedirs(kmodes_dir, exist_ok=True)

def run_kmodes_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df[columns_to_select].dropna().reset_index(drop=True)

    df_sampled = load_pkl("kmodes_sampled_df.pkl", base_path=kmodes_dir)
    if df_sampled is None:
        df_sampled = df.sample(frac=0.1, random_state=random_state).copy()
        dump_pkl(df_sampled, "kmodes_sampled_df.pkl", base_path=kmodes_dir)
    else:
        print("Loaded cached sampled dataset.")

    labels_cached = load_pkl("kmodes_labels.pkl", base_path=kmodes_dir)
    if labels_cached is not None:
        df_sampled["labels"] = labels_cached
        print("Loaded cached KModes labels.")
    else:
        best_k = load_pkl("best_k.pkl", base_path=kmodes_dir)
        if best_k is None:
            best_k = find_best_k(df_sampled)
            dump_pkl(best_k, "best_k.pkl", base_path=kmodes_dir)
        else:
            print(f"Loaded best_k from cache: {best_k}")

        km = KModes(n_clusters=best_k, init='Huang', n_init=5, verbose=1, random_state=random_state)
        labels = km.fit_predict(df_sampled[categorical_cols])
        df_sampled["labels"] = labels
        dump_pkl(labels, "kmodes_labels.pkl", base_path=kmodes_dir)
        print("KModes clustering completed.")

    print("DataFrame columns after clustering:", df_sampled.columns.tolist())
    df_sampled.to_parquet(f"{kmodes_dir}/processed_onboarding_kmodes_clustered.parquet")
    return df_sampled


def plot_silhouette_multiplot(X, k_list, labels_list, save_path):
    """
    Generates silhouette plots for multiple k-values in a grid layout.
    
    Args:
        X (np.ndarray): One-hot encoded input data.
        k_list (list[int]): List of k-values.
        labels_list (list[np.ndarray]): List of label arrays corresponding to each k.
        save_path (str): Output file path for the figure.
    """
    num_plots = len(k_list)
    ncols = 2
    nrows = (num_plots + 1) // 2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axs = axs.flatten()
    cmap = plt.cm.nipy_spectral

    for i, (k, labels) in enumerate(zip(k_list, labels_list)):
        ax = axs[i]
        silhouette_vals = silhouette_samples(X, labels, metric="hamming")
        y_lower = 10
        for j in range(k):
            j_vals = silhouette_vals[labels == j]
            j_vals.sort()
            size_j = j_vals.shape[0]
            y_upper = y_lower + size_j
            color = cmap(float(j) / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, j_vals, facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_j, str(j))
            y_lower = y_upper + 10

        avg_score = np.mean(silhouette_vals)
        ax.axvline(avg_score, color="red", linestyle="--")
        ax.set_title(f"The silhouette plot for k={k}.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X) + (k + 1) * 10])
        ax.grid(False)

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def find_best_k(df: pd.DataFrame) -> int:
    print("Finding best k for KModes (Cost, Silhouette, UMAP, Multiplot)...")
    costs = []
    silhouette_scores = []
    labels_all = []
    k_values = list(range(2, 10))

    df_enc_elbow, _ = one_hot_encode(df[categorical_cols], categorical_cols, [])

    embedding_2d = reduce_dimensions_umap(
        df_enc_elbow,
        n_components=2,
        metric="hamming",
        random_state=random_state,
        save_embedding=False
    )

    for k in k_values:
        print(f"Running KModes for k={k}...")
        km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0, random_state=random_state)
        labels = km.fit_predict(df[categorical_cols])
        costs.append(km.cost_)
        labels_all.append(labels)

        silhouette = calculate_silhouette_score(df_enc_elbow, labels, metric='hamming')
        silhouette_scores.append(silhouette)
        print(f"  k={k}, Cost: {km.cost_:.4f}, Silhouette: {silhouette:.4f}")

        visualize_clusters(
            embedding_2d,
            labels,
            save_plot=True,
            filename=os.path.join(kmodes_dir, f"umap_k_{k}.png"),
            title=f"KModes Clusters on 2D UMAP (k={k})"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, costs, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Cost")
    plt.title("KModes Cost vs k")
    plt.grid(True)
    try:
        best_k = KneeLocator(k_values, costs, curve="convex", direction="decreasing").elbow
        if best_k is None:
            raise ValueError("Elbow not found")
    except:
        best_k = 6
        print("Defaulting to k=6")
    plt.axvline(x=best_k, color='red', linestyle='--', label=f"Best k = {best_k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(kmodes_dir, "kmodes_cost_elbow_vs_k.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker='x')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k (KModes)")
    plt.grid(True)
    max_sil_k = k_values[np.argmax(silhouette_scores)]
    plt.axvline(x=max_sil_k, color='purple', linestyle='--', label=f"Max Silhouette = {max_sil_k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(kmodes_dir, "kmodes_silhouette_vs_k.png"))
    plt.close()

    plot_silhouette_multiplot(
        df_enc_elbow,
        k_values,
        labels_all,
        save_path=os.path.join(kmodes_dir, "kmodes_silhouette_multiplot.png")
    )

    print(f"Plots saved. Best k = {best_k}")
    return best_k


def evaluate_and_visualize_kmodes(df_sampled: pd.DataFrame):
    df_enc, _ = one_hot_encode(df_sampled[categorical_cols], categorical_cols)

    db_index = calculate_davies_bouldin_score(df_enc, df_sampled["labels"])
    sil_score = calculate_silhouette_score(df_enc, df_sampled["labels"], metric='hamming')
    ch_score = calculate_calinski_harabasz_score(df_enc, df_sampled["labels"])
    print(f"Davies-Bouldin Index: {db_index:.3f}")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Calinski-Harabasz: {ch_score:.3f}")

    pd.DataFrame([{
        "silhouette_score": sil_score,
        "davies_bouldin_index": db_index,
        "calinski_harabasz_score": ch_score,
        "n_clusters": len(set(df_sampled["labels"])),
    }]).to_csv(os.path.join(kmodes_dir, "kmodes_validation_metrics.csv"), index=False)


    summary_df = summarize_clusters(df_sampled, categorical_cols=categorical_cols)
    summary_df.to_csv(os.path.join(kmodes_dir, "summary_kmodes.csv"), index=False)
    plot_feature_distributions(df_sampled, cluster_col="labels", categorical_cols=categorical_cols, algorithm_name="kmodes")

    plot_feature_distributions(
        df=df_sampled,
        cluster_col="labels",
        categorical_cols=["product_id_grouped"],
        algorithm_name="kmodes",
        base_path="data/cluster_distributions"
    )

    distribution_text = ""
    for label, group in df_sampled.groupby("labels"):
        distribution_text += f"Cluster {label}:\n"
        percents = group["product_id_grouped"].value_counts(normalize=True).mul(100).round(2)
        for product, percent in percents.items():
            distribution_text += f"  - {product}: {percent:.2f}%\n"
        distribution_text += "\n"

    txt_file = os.path.join(kmodes_dir, "product_id_grouped_distribution.txt")
    with open(txt_file, "w") as f:
        f.write(distribution_text)
    print(f"Product ID distribution text saved to {txt_file}")

def main():
    create_data_dirs()
    df = load_data("data/processed_onboarding.parquet").to_pandas()
    df_sampled = run_kmodes_pipeline(df)
    evaluate_and_visualize_kmodes(df_sampled)

if __name__ == "__main__":
    main()