import hdbscan
import numpy as np
from preprocess import load_pkl, dump_pkl, cluster_subsample  # Assuming preprocess.py is in the same directory
from datetime import datetime


def find_best_hdbscan_params(
    embeddings, max_cluster_size_fraction=0.1, lin_space_size=10,
    min_samples_factor=2,  # Added min_samples_factor
    save_params=True,
    base_path="data/hdbscan_umap/hdbscan"
):
    """
    Finds the best HDBSCAN parameters using a validity index.

    Args:
        embeddings (np.ndarray): The input embeddings.
        max_cluster_size_fraction (float, optional): Maximum cluster size
            as a fraction of total samples. Defaults to 0.1.
        lin_space_size (int, optional): Number of cluster sizes to test.
            Defaults to 10.
        min_samples_factor (int, optional): Factor to multiply the log-estimated
            min_samples. Defaults to 2.
        save_params (bool, optional): Whether to save the best cluster size.
            Defaults to True.
        base_path (str, optional): Base path for saving/loading data.
            Defaults to "data/hdbscan_umap/hdbscan".

    Returns:
        tuple: A tuple containing the best cluster labels, the best cluster
            size, and the best validity index.
    """

    num_samples = embeddings.shape[0]
    min_samples_estimate = int(np.ceil(np.log10(num_samples)) * min_samples_factor)
    max_cluster_size = int(num_samples * max_cluster_size_fraction)
    lin_space = np.linspace(1000, max_cluster_size, num=lin_space_size).astype(int)

    best_cluster_size = 0
    best_labels = None
    best_validity_index = -1

    best_cluster_size_file_exists = load_pkl("best_cluster_size.pkl", base_path) is not None
    if best_cluster_size_file_exists:
        lin_space = load_pkl("best_cluster_size.pkl", base_path)

    for i in lin_space:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=i,
            min_samples=min_samples_estimate,
            memory=base_path,
            algorithm="boruvka_kdtree",
        )

        labels = clusterer.fit_predict(embeddings)
        subsampled_embeddings, sampled_labels = cluster_subsample(
            embeddings, labels, fraction=0.1
        )
        validity_index = hdbscan.validity.validity_index(
            subsampled_embeddings, sampled_labels
        )

        print(f"Cluster Size: {i}, Validity index: {validity_index}")

        if validity_index > best_validity_index:
            best_validity_index = validity_index
            best_labels = labels
            best_cluster_size = i
        else:
            # Early stopping: if validity decreases, it's unlikely to improve
            break

    if save_params and not best_cluster_size_file_exists:
        dump_pkl(np.array([best_cluster_size]), "best_cluster_size.pkl", base_path)

    return best_labels, best_cluster_size, best_validity_index