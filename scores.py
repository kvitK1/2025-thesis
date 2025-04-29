from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np


def calculate_silhouette_score(data, labels, metric="euclidean"):
    """
    Calculates the Silhouette Score.

    Args:
        data (np.ndarray): The data.
        labels (np.ndarray): The cluster labels.

    Returns:
        float: The Silhouette Score. Returns None if there are fewer than 2 clusters.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        return silhouette_score(data, labels, metric=metric)
    else:
        print("Silhouette Score requires more than one cluster.")
        return None


def calculate_davies_bouldin_score(data, labels):
    """
    Calculates the Davies-Bouldin Score.

    Args:
        data (np.ndarray): The data.
        labels (np.ndarray): The cluster labels.

    Returns:
        float: The Davies-Bouldin Score. Returns None if there are fewer than 2 clusters.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        return davies_bouldin_score(data, labels)
    else:
        print("Davies-Bouldin Score requires more than one cluster.")
        return None


def calculate_calinski_harabasz_score(data, labels):
    """
    Calculates the Calinski-Harabasz Score.

    Args:
        data (np.ndarray): The data.
        labels (np.ndarray): The cluster labels.

    Returns:
        float: The Calinski-Harabasz Score. Returns None if there are fewer than 2 clusters.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        return calinski_harabasz_score(data, labels)
    else:
        print("Calinski-Harabasz Score requires more than one cluster.")
        return None
