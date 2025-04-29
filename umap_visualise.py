import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess import dump_pkl, load_pkl  # Assuming preprocess.py is in the same directory
import numpy as np


def reduce_dimensions_umap(
    data, n_components=2, n_neighbors=30, min_dist=0.0, metric="hamming",
    random_state=42,  # Make sure to include random_state for reproducibility
    save_embedding=True,
    embedding_filename="embedding.pkl",
    base_path="data/hdbscan_umap/"
):
    """
    Reduces the dimensionality of the input data using UMAP.

    Args:
        data (np.ndarray): The input data to reduce.
        n_components (int, optional): The number of dimensions to reduce to.
            Defaults to 2.
        n_neighbors (int, optional): UMAP parameter. Defaults to 30.
        min_dist (float, optional): UMAP parameter. Defaults to 0.0.
        metric (str, optional): UMAP metric. Defaults to "hamming".
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.
        save_embedding (bool, optional): Whether to save the embedding.
            Defaults to True.
        embedding_filename (str, optional): Filename to save the embedding.
            Defaults to "embedding.pkl".
        base_path (str, optional): Base path for saving data.
            Defaults to "data/hdbscan_umap/".

    Returns:
        np.ndarray: The reduced dimensionality representation of the data.
    """

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(data)

    if save_embedding:
        dump_pkl(embedding, embedding_filename, base_path)

    return embedding


def visualize_clusters(
    embedding, labels, title="UMAP projection of the dataset",
    save_plot=False,
    filename="cluster_visualization.png"
):
    """
    Visualizes clusters using a scatter plot.

    Args:
        embedding (np.ndarray): The 2D embedding of the data.
        labels (np.ndarray): The cluster labels.
        title (str, optional): The title of the plot.
            Defaults to "UMAP projection of the dataset".
        save_plot (bool, optional): Whether to save the plot to a PNG file.
            Defaults to False.
        filename (str, optional): The filename to save the plot as.
            Defaults to "cluster_visualization.png".
    """

    plot_df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "cluster": labels})

    unique_labels = np.unique(labels)
    num_unique_labels = len(unique_labels)

    # Generate a color palette that includes gray for noise points (-1)
    if -1 in unique_labels:
        base_palette = sns.color_palette("tab20", n_colors=num_unique_labels - 1)
        custom_palette = ["#d3d3d3"] + base_palette  # Gray for -1, then tab20
    else:
        custom_palette = sns.color_palette("tab20", n_colors=num_unique_labels)

    # Ensure the palette has enough colors
    custom_palette = custom_palette[:num_unique_labels]

    plt.figure(figsize=(10, 8))  # Adjust figure size for better readability
    sns.scatterplot(
        data=plot_df, x="x", y="y", hue="cluster", palette=custom_palette, s=50
    )
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position

    if save_plot:
        plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box
    plt.show()