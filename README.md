# 👩‍🎓 ML-Driven Clustering of Onboarding Users: Strategies for Enhanced Conversion
Implementation of the "ML-Driven Clustering of Onboarding Users: Strategies for Enhanced Conversion" modules as the part of the bachelor thesis conducted by [Kvitoslava Kolodii](https://www.linkedin.com/in/kvitkolodii/) under the supervision of [Tetiana Marynych](https://www.linkedin.com/in/tetiana-marynych/). It was submitted in fulfilment of the requirements for the Bachelor of Science degree in the Department of Business Analytics and Information Technologies at the Faculty of Applied Sciences.

## 🎀 Abstract
Effective onboarding is crucial for user engagement and conversion, yet many applications employ universal strategies that fail to address diverse user needs and motivations. This study investigates unsupervised clustering techniques to segment users based on their interactions during the onboarding process. The thesis is based on anonymised user data from a book summaries mobile application, focusing on key onboarding steps. We apply clustering algorithms, including K-Modes, HDBSCAN, Agglomerative Clustering, and Spectral Clustering, to identify distinct user segments. Appropriate metrics are applied to evaluate the clustering solutions, and the most effective one is selected for in-depth analysis. The analysis results are then used to generate actionable insights and strategies to enhance the user experience and, therefore, subscription conversions.

## 📁 Repository Organisation

| Name                          | Module description                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `dataset.py` | loads raw onboarding data, cleans and filters it, performs feature engineering and categorical transformations, and saves the enriched dataset for further analysis or modeling; |
| `eda.py` | performs exploratory data analysis on processed onboarding data by visualizing distributions, missing values, contingency tables with purchase, and Cramér’s V correlations between categorical features; |
| `preprocess.py` | provides utility functions for loading data, encoding categorical and multi-label features, saving/loading pickles, and subsampling clustered embeddings for further analysis; |
| `scores.py` | calculates three standard clustering validation metrics—Silhouette, Davies-Bouldin, and Calinski-Harabasz scores—to evaluate the quality of clustering results; |
| `umap_visualise.py` | performs dimensionality reduction using UMAP and visualizes clustering results in 2D space via scatter plots; |
| `kmodes_main.py` | runs the full K-Modes clustering pipeline on onboarding data, including optimal k selection, clustering, validation, visualization, and per-cluster summary generation; |
| `agglomerative_main.py` | performs Agglomerative Clustering on categorical onboarding data using Hamming distance, evaluates clustering quality, generates cluster summaries and UMAP visualizations, and saves all outputs for analysis; |
| `spectral_main.py` | performs Spectral Clustering using a cosine similarity matrix on onboarding data, selects the optimal number of clusters, evaluates clustering quality, visualizes clusters with UMAP, and saves detailed outputs for further analysis; |
| `hdbscan_umap.py` | optimizes HDBSCAN clustering parameters by searching for the best `min_cluster_size` using a validity index, and returns the corresponding cluster labels and metrics; |
| `hdbscan_main.py` | performs HDBSCAN clustering on UMAP-reduced embeddings of onboarding data, selects optimal parameters using a validity index, evaluates and visualizes cluster quality, and generates detailed per-cluster summaries and feature distributions; |
| `cluster_summary.py` | generates per-cluster summaries by computing modal feature values, cluster sizes, and target conversion rates, and saves the results as CSV files; |
| `cluster_distributions.py` | plots and saves per-cluster bar charts of categorical feature distributions and exports corresponding percentage breakdowns to CSV files for detailed analysis. |

## ℹ️ Instructions
### 🐑 Clone the repository.
```
git clone https://github.com/kvitK1/ML-Driven-Clustering-of-Onboarding-Users-Strategies-for-Enhanced-Conversion
cd ML-Driven-Clustering-of-Onboarding-Users-Strategies-for-Enhanced-Conversion
```

### 💠 Create and activate a virtual environment
for macOS\Linux:
```
python3 -m venv venv
source venv/bin/activate
```

for Windows:
```
python -m venv venv
venv\Scripts\activate
```

### 🛠️ Install the required packages
```
pip install -r requirements.txt
```

### 👾 Start using the modules
⚠️ `dataset.py` is view-only. Don't run it as you have an already processed file in `data/`.

```
python kmodes_main.py
python agglomerative_main.py
python spectral_main.py
python hdbscan_main.py
python eda.py
```

🗃️ Results of modules are organised in data/ directories, there are their descriptions below:
- `processed onboarding.parquet`: an already preprocessed dataset (on missing values, features, etc.);
- `eda_outputs`: contains visual outputs from the exploratory data analysis phase, including individual feature distribution plots, contingency heatmaps, and Cramér’s V correlations to assess categorical variable relationships and their association with purchase behavior;
- `kmodes`: contains KModes clustering evaluation results, including optimal cluster selection plots (`elbow`, `silhouette`), UMAP visualizations across k values (`umap_k_*.png`), validation metrics, cluster summaries, and product ID distribution analysis;
- `agglomerative_hamming`: contains agglomerative clustering results, including UMAP visualizations, a dendrogram, silhouette scores, validation metrics, optimal k selection, product ID distributions, and per-cluster summaries;
- `hdbscan_umap`: contains the full output of the HDBSCAN clustering process, including the optimal cluster size selection, final clustering results, UMAP visualization, validation metrics, cluster summaries, and product ID distributions, along with cached model components and parameters in the subdirectory;
- `spectral`: contains Spectral Clustering results, including UMAP visualizations, evaluation plots and metrics, per-cluster summaries, and product ID distribution analysis;
- `cluster_distributions/algorithm_name/feature_name`: contain feature distrubutions per cluster in `csv` format and visualisations.

🐍 Python version used: `Python 3.9.6`

## 👩‍🎤 Contributors
- [Kvitoslava Kolodii](https://www.linkedin.com/in/kvitkolodii/)
- [Tetiana Marynych](https://www.linkedin.com/in/tetiana-marynych/)
