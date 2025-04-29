from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import hdbscan


def validate_clustering(
    data, original_labels, best_cluster_size,
    num_subsamples=3, subsample_fraction=0.8
):
    """
    Validates clustering using subsampling and perturbation.

    Args:
        data (np.ndarray): The data used for clustering.
        original_labels (np.ndarray): The cluster labels from the original run.
        best_cluster_size (int): The best min_cluster_size found by HDBSCAN.
        num_subsamples (int, optional): Number of subsamples to create. Defaults to 3.
        subsample_fraction (float, optional): Fraction of data to include in each subsample. Defaults to 0.8.

    Returns:
        tuple: A tuple containing lists of ARI and NMI scores for subsampling and perturbation.
    """

    subsampling_results = []
    perturbation_results = []

    for _ in range(num_subsamples):
        subsample_indices = np.random.choice(
            len(data), size=int(len(data) * subsample_fraction), replace=False
        )
        subsample_data = data[subsample_indices]
        subsample_labels = hdbscan.HDBSCAN(
            min_cluster_size=best_cluster_size
        ).fit_predict(subsample_data)

        mapped_subsample_labels = np.full(
            len(data), -2
        )
        mapped_subsample_labels[subsample_indices] = subsample_labels

        ari = adjusted_rand_score(original_labels, mapped_subsample_labels)
        nmi = normalized_mutual_info_score(
            original_labels, mapped_subsample_labels
        )
        subsampling_results.append((ari, nmi))
    
    print("Subsampling Validation is finished")

    return subsampling_results