import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math

def plot_feature_distributions(df, cluster_col, categorical_cols, algorithm_name, base_path="data/cluster_distributions"):
    """
    Plots and saves feature distributions for each feature in a single image file.
    Each image contains subplots for each cluster showing the distribution of the feature.
    Bar colors are assigned based on the feature values consistently across clusters.

    Args:
        df (pd.DataFrame): Input dataframe with clusters assigned.
        cluster_col (str): Name of the column containing cluster labels.
        categorical_cols (list): List of categorical features to plot.
        algorithm_name (str): Identifier for the algorithm (used for folder naming).
        base_path (str): Base directory to save plots and CSVs.
    """
    output_dir = os.path.join(base_path, algorithm_name)
    os.makedirs(output_dir, exist_ok=True)

    for feature in categorical_cols:
        feature_dir = os.path.join(output_dir, feature)
        os.makedirs(feature_dir, exist_ok=True)
        
        # Create a color mapping for this feature using a Seaborn palette.
        # We use sorted unique values to ensure consistency.
        all_categories = sorted(df[feature].dropna().unique())
        palette = sns.color_palette("viridis", len(all_categories))
        color_mapping = dict(zip(all_categories, palette))
        
        # Get unique clusters
        unique_clusters = sorted(df[cluster_col].dropna().unique())
        n_clusters = len(unique_clusters)
        
        # Define grid: e.g., 3 columns (or fewer if not enough clusters)
        n_cols = min(3, n_clusters)
        n_rows = math.ceil(n_clusters / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        # If there's only one subplot, make axes iterable
        if n_clusters == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Loop over clusters and plot distribution for each in its subplot
        for i, cluster in enumerate(unique_clusters):
            cluster_df = df[df[cluster_col] == cluster]
            counts = cluster_df[feature].value_counts(normalize=True).sort_values(ascending=False)
            
            # Use the color mapping to create a list of colors corresponding to counts.index
            colors = [color_mapping.get(val, "#333333") for val in counts.index]
            
            ax = axes[i]
            sns.barplot(x=counts.values * 100, y=counts.index, ax=ax, palette=colors)
            ax.set_title(f"Cluster {cluster}")
            ax.set_xlabel("% share")
            ax.set_ylabel(feature)
        
        # Remove any extra axes if n_clusters doesn't fill the grid
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.suptitle(f"{feature} Distribution Across Clusters", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save the combined plot as an image
        plot_filename = os.path.join(feature_dir, f"{feature}.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Saved combined plot for {feature} to {plot_filename}")
        
        # Optionally, save CSV files that combine all clusters
        combined_list = []
        for cluster in unique_clusters:
            cluster_df = df[df[cluster_col] == cluster]
            counts = cluster_df[feature].value_counts(normalize=True).mul(100).round(2)
            dist_df = counts.reset_index()
            dist_df.columns = [feature, "percentage"]
            dist_df["cluster"] = cluster
            combined_list.append(dist_df)
        combined_df = pd.concat(combined_list, ignore_index=True)
        csv_filename = os.path.join(feature_dir, f"{feature}_combined_distribution.csv")
        combined_df.to_csv(csv_filename, index=False)
        print(f"Saved combined CSV for {feature} to {csv_filename}")