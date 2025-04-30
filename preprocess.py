import polars as pl
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import pickle as pkl
import pandas as pd  # Import pandas


def load_data(path):
    """Loads a Polars DataFrame from a parquet file."""
    df = pl.read_parquet(path)
    return df


def dump_pkl(data, filename, base_path="data/hdbscan_umap/"):
    """Saves data to a pickle file."""
    with open(f"{base_path}/{filename}", "wb") as f:
        pkl.dump(data, f)


def load_pkl(filename, base_path="data/hdbscan_umap/"):
    """Loads data from a pickle file."""
    try:
        with open(f"{base_path}/{filename}", "rb") as f:
            data = pkl.load(f)
        return data
    except FileNotFoundError:
        return None


def one_hot_encode(df, categorical_cols, multilabel_cols=None, handle_unknown="ignore"):
    """
    Performs one-hot encoding on specified categorical columns of a DataFrame,
    and handles multi-label columns.

    Args:
        df (pl.DataFrame or pd.DataFrame): The input DataFrame.
        categorical_cols (list[str]): List of column names to encode.
        multilabel_cols (list[str], optional): List of columns containing
            multi-labels (lists of strings). Defaults to None.
        handle_unknown (str, optional): How to handle unknown values for
            simple one-hot encoding. Defaults to "ignore".

    Returns:
        tuple: A tuple containing the encoded NumPy array and the encoder objects
               (OneHotEncoder and a list of MultiLabelBinarizers).
    """

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()  # Convert to pandas if it's a Polars DataFrame

    encoded_cols = []
    encoders = []

    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
        X = encoder.fit_transform(df[categorical_cols])
        encoded_cols.append(X)
        encoders.append(encoder)  # Store the encoder
    else:
        X = np.array([])  # Empty array if no categorical cols
        encoders.append(None)

    if multilabel_cols:
        mlb_encoders = []
        for col in multilabel_cols:
            df.loc[:, col] = df[col].fillna("missing")
            df.loc[:, col] = df[col].apply(
                lambda x: [i.strip() for i in x.split(",")] if isinstance(x, str)
                else [] if not isinstance(x, list) else x
            )
            mlb = MultiLabelBinarizer()
            one_hot = mlb.fit_transform(df[col])
            one_hot_df = pd.DataFrame(one_hot, columns=[f"{col}_{cls}" for cls in mlb.classes_])
            encoded_cols.append(one_hot_df.values)
            mlb_encoders.append(mlb)
        encoders.append(mlb_encoders)  # Store MLB encoders
    else:
        encoders.append(None)  # Placeholder if no multilabel cols

    if encoded_cols:
        return np.concatenate(encoded_cols, axis=1), encoders
    else:
        return np.array([]), encoders  # Return empty array and None if no encoding


def cluster_subsample(embeddings, labels, fraction=0.1):
    """
    Subsamples embeddings and labels for each cluster.

    Args:
        embeddings (np.ndarray): The embedding vectors.
        labels (np.ndarray): The cluster labels.
        fraction (float, optional): The fraction of samples to retain
            from each cluster. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the subsampled embeddings and labels.
    """

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    unique_labels = np.unique(labels)
    sampled_embeddings = []
    sampled_labels = []

    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        sample_size = max(1, int(len(idx) * fraction))
        sampled_idx = np.random.choice(idx, size=sample_size, replace=False)

        sampled_embeddings.append(embeddings[sampled_idx])
        sampled_labels.append(labels[sampled_idx])

    return np.concatenate(sampled_embeddings, axis=0), np.concatenate(
        sampled_labels, axis=0
    )
