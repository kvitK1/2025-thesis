import os
import pandas as pd

def summarize_clusters(df, label_col="labels", categorical_cols=None, target="purchase"):
    """
    Summarizes clusters by computing mode for categorical features,
    cluster size, and target rate (e.g., purchase rate).

    Args:
        df (pd.DataFrame): DataFrame containing clusters.
        label_col (str): Column name of cluster labels.
        categorical_cols (list): List of categorical columns to summarize.
        target (str): Target variable for calculating mean (e.g., conversion rate).

    Returns:
        pd.DataFrame: Summary statistics for each cluster.
    """
    if categorical_cols is None:
        raise ValueError("categorical_cols must be provided")

    summary = (
        df.groupby(label_col)[categorical_cols]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    summary["size"] = df[label_col].value_counts().sort_index().values
    if target in df.columns:
        summary["purchase_rate"] = df.groupby(label_col)[target].mean().values

    return summary


def save_cluster_summary(summary_df, algorithm_name, path="data/cluster_summaries"):
    """
    Saves the summary DataFrame to CSV under a standardized path.

    Args:
        summary_df (pd.DataFrame): The summary DataFrame to save.
        algorithm_name (str): Name of the clustering algorithm (used in filename).
        path (str): Directory to save the CSV.
    """
    os.makedirs(path, exist_ok=True)
    summary_df.to_csv(f"{path}/summary_{algorithm_name}.csv")
