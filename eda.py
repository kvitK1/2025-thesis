# eda.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
import polars as pl
from sklearn.preprocessing import LabelEncoder
import warnings
import math

warnings.filterwarnings("ignore")

from preprocess import load_data

EDA_OUTPUT_DIR = "data/eda_outputs"
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

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


def plot_categorical_distributions(df, categorical_cols):
    print("\nPlotting categorical distributions...")
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        df[col].value_counts(normalize=True).head(20).plot(kind='bar')
        plt.title(f"Distribution of {col}")
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.savefig(f"{EDA_OUTPUT_DIR}/dist_{col}.png")
        plt.close()


def print_nulls(df):
    null_counts_dict = df.select([
        pl.col(col).is_null().sum().alias(col) for col in df.columns
    ]).to_dict(as_series=False)

    null_counts_df = df.select([
        pl.col(col).is_null().sum().alias(col) for col in df.columns
    ]).to_dict(as_series=False)

    total_rows = df.height
    null_ratio_df = pl.DataFrame({
        "column": list(null_counts_df.keys()),
        "null_ratio": [v[0] / total_rows for v in null_counts_df.values()]
    })

    null_ratio_df = null_ratio_df.sort("null_ratio", descending=True)

    null_ratio_df.to_pandas()

    print(f"{'Column name':<{52}} {'Nulls'}")
    for row in null_ratio_df.iter_rows():
        print(f"{row[0]:<50} {round(row[1] * 100, 2):>6.2f}%")


def contingency_matrices_purchase(df, categorical_cols):
    n_cols = 3
    n_rows = math.ceil(len(categorical_cols[:-2]) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols[:-2]):
        ct = round(pd.crosstab(
            df.select(col).to_pandas()[col],
            df.select("purchase").to_pandas()["purchase"],
            normalize='index') * 100, 2)
        
        sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"{col} vs Purchase (%)")
        axes[i].set_xlabel("purchase")
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/contingency_heatmaps.png")
    plt.close()



def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))


def cramers_v_matrix(df, categorical_cols):
    cramer_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

    for col1 in categorical_cols:
        for col2 in categorical_cols:
            cramer_matrix.loc[col1, col2] = cramers_v(df.select(categorical_cols).to_pandas()[col1], df.select(categorical_cols).to_pandas()[col2])

    cramer_matrix = cramer_matrix.astype(float)

    plt.figure(figsize=(14, 12)) 
    sns.heatmap(cramer_matrix, annot=False, cmap="coolwarm", square=True, 
                linewidths=0.5, vmin=0, vmax=1, cbar_kws={"label": "Cramér's V"})
    plt.title("Cramér's V Heatmap Between Categorical Features", fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/cramers_v_heatmap.png")
    plt.close()


def run_eda():
    df = load_data("data/processed_onboarding.parquet").to_pandas()

    print_nulls(pl.DataFrame(df))
    plot_categorical_distributions(df, categorical_cols)
    contingency_matrices_purchase(pl.DataFrame(df), categorical_cols)
    cramers_v_matrix(pl.DataFrame(df), categorical_cols)

    print("\nEDA complete. Outputs saved to", EDA_OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
