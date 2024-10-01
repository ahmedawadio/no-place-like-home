# import pandas as pd
# import re

# df = pd.read_csv("api/analysis/data/query.csv",dtype={'cid': str})

# df= df.dropna()


# # Extract 'variable' and 'year' from 'variable_year'
# # Assuming the last 4 characters represent the year
# df_long['year'] = df_long['variable_year'].apply(lambda x: re.search(r'(\d{4})$', x).group(1) if re.search(r'(\d{4})$', x) else 'Unknown')
# df_long['variable'] = df_long['variable_year'].apply(lambda x: re.sub(r'_(\d{4})$', '', x) if re.search(r'_(\d{4})$', x) else x)

# # Display the reshaped DataFrame
# # print("\nReshaped DataFrame (long format):")
# # print(df_long.head(10))

# num_variables = df_long['variable'].nunique()

# # Print the count of unique variables
# print(f"\nNumber of unique variables: {num_variables}")



import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, AffinityPropagation, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, OPTICS, Birch, SpectralClustering, AffinityPropagation

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances



# not always good to do, but because i am looping through a bunch of different clustering algorithms, i will suppress warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress all warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Suppress ConvergenceWarnings


# -------------------------
# 1. Load and Prepare Data
# -------------------------

def plot_row_column_thresholds(df):
    # Load the CSV file with 'cid' as string
    df.replace(['', 'None', None, '-888888888'], pd.NA, inplace=True)

    # Calculate invalid percentages
    invalid_percentages = df.isna().mean()

    # Lists to store threshold percentages, rows, and columns
    thresholds = []
    num_rows = []
    num_cols = []

    # Loop through thresholds from 0% to 99%
    for threshold_percentage in range(1, 100):  # Start from 1% to avoid threshold=0
        temp_df = df.copy()  # Use .copy() to avoid modifying the original dataframe

        # Filter out columns with high invalid percentages
        cols_to_keep = invalid_percentages[invalid_percentages <= threshold_percentage / 100].index.tolist()

        # Ensure 'cid' is included
        if 'cid' not in cols_to_keep:
            cols_to_keep.insert(0, 'cid')

        # Filter the DataFrame to only keep columns with valid percentages
        temp_df = temp_df[cols_to_keep]
        
        # Drop rows with NaN values
        temp_df = temp_df.dropna()

        # Store the threshold percentage, number of rows, and columns
        thresholds.append(threshold_percentage)
        num_rows.append(temp_df.shape[0])
        num_cols.append(temp_df.shape[1])

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, num_rows, label='Number of Rows', color='blue', marker='o')
    plt.plot(thresholds, num_cols, label='Number of Columns', color='red', marker='x')

    plt.xlabel('Threshold Percentage')
    plt.ylabel('Count')
    plt.title('Number of Rows and Columns vs Threshold Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()


def shape_threshold(df,threshold_percentage):
    # Load the CSV file with 'cid' as string
    df.replace(['', 'None', None, '-888888888'], pd.NA, inplace=True)

    # Calculate invalid percentages
    invalid_percentages = df.isna().mean()

    temp_df = df.copy()  # Use .copy() to avoid modifying the original dataframe

    # Filter out columns with high invalid percentages
    cols_to_keep = invalid_percentages[invalid_percentages <= threshold_percentage / 100].index.tolist()

    # Ensure 'cid' is included
    if 'cid' not in cols_to_keep:
        cols_to_keep.insert(0, 'cid')

    # Filter the DataFrame to only keep columns with valid percentages
    temp_df = temp_df[cols_to_keep]
    
    # Drop rows with NaN values
    temp_df = temp_df.dropna()

    # Remove any columns that are not of type float or int, excluding 'cid'
    numeric_cols = temp_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'cid' in temp_df.columns:
        numeric_cols.insert(0, 'cid')  # Ensure 'cid' stays in the DataFrame
    temp_df = temp_df[numeric_cols]

    return temp_df


filepath  = "api/analysis/data/query.csv"


# Load the CSV file with 'cid' as string
df = pd.read_csv(filepath, dtype={'cid': str}, low_memory=False)
df.replace(['', 'None', None, '-888888888'], pd.NA, inplace=True)

# Checking the tradeoff of rows / columns to drop based on threshold. Goal is to keep more more rows to ahve more ciities in model
# plot_row_column_thresholds(df)


chosen_threshold = 10
df = shape_threshold(df, chosen_threshold)

# # Display the first few rows
# print("First few rows of the cleaned DataFrame:")
# print(df.head())

# # Exclude 'cid' column for analysis
features = df.drop('cid', axis=1)

# # # Display feature names
# # print("\nFeature columns:")
# # print(features.columns.tolist())

# -------------------------
# 2. Data Scaling
# -------------------------

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
scaled_features = scaler.fit_transform(features)

# Convert back to DataFrame for easier handling
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)


# -------------------------
# 3. Dimensionality Reduction
# -------------------------

def graph_PCA():
    # Initialize PCA (without specifying n_components to consider all components)
    pca = PCA()
    pca_result = pca.fit(scaled_features)

    # Get the explained variance ratio for each component
    explained_variance = pca.explained_variance_ratio_

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)

    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Number of components required to explain 95% variance: {n_components}")

    # Choose how many components to display (e.g., first 20 components)
    chunk_to_viz = 100  # Adjust this number as needed

    # Plot the explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, chunk_to_viz + 1), explained_variance[:chunk_to_viz], 'bo-', label='Explained Variance per Component')
    plt.plot(range(1, chunk_to_viz + 1), cumulative_variance[:chunk_to_viz], 'ro-', label='Cumulative Explained Variance')
    plt.title('Explained Variance vs Number of Components (First 20 Components)')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.xticks(range(1, chunk_to_viz + 1))
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

# graph_PCA()

# Use PCA for preprocessing
n_components = 81  # Adjust the number of components as needed
pca = PCA(n_components=n_components, random_state=42)
pca_result = pca.fit_transform(scaled_features)


# -------------------------
# 4. Clustering Algorithms
# -------------------------


def find_optimal_k():

    inertia_values = []
    k_values = range(2, 15)  # Adjust the range as needed

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(pca_result)
        inertia_values.append(kmeans.inertia_)

    print(f"kvalues{k_values} inertia_values{inertia_values}")
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia_values, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    silhouette_avg_values = []
    k_values = range(2, 15)  # Adjust the range as needed

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=1)
        cluster_labels = kmeans.fit_predict(pca_result)
        silhouette_avg = silhouette_score(pca_result, cluster_labels)
        silhouette_avg_values.append(silhouette_avg)


    print (f"kvalues{k_values} silhouette_avg_values{silhouette_avg_values}")
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_avg_values, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.grid(True)
    plt.show()

# find_optimal_k()


# Define a function to perform clustering and evaluate
def perform_clustering(name, algorithm, data, **kwargs):
    algorithm_instance = algorithm(**kwargs)
    cluster_labels = algorithm_instance.fit_predict(data)
    n_clusters = len(np.unique(cluster_labels))
    
    # Exclude noise points for DBSCAN or algorithms that label noise as -1
    if n_clusters > 1 and -1 not in cluster_labels:
        silhouette = silhouette_score(data, cluster_labels)
        db_index = davies_bouldin_score(data, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
        print(f"{name} Sil: {silhouette:.2f}, DB: {db_index:.2f}, CH: {calinski_harabasz:.2f}  ({kwargs}) - ")
    else:
        silhouette = np.nan
        db_index = np.nan
        calinski_harabasz = np.nan
    
    return cluster_labels, n_clusters, silhouette, db_index, calinski_harabasz

# Initialize a list to store results
clustering_results = []

# List of clustering algorithms and parameters
clustering_algorithms = [
    ('KMeans', KMeans, {'n_clusters': range(6,7), 'random_state': 1}),
    ('AgglomerativeClustering', AgglomerativeClustering, {'n_clusters': range(8,9)}),
    ('Birch', Birch, {'n_clusters': range(6,7)}),
    ('SpectralClustering', SpectralClustering, { 'n_clusters': range(7, 10), 'affinity': 'rbf', 'eigen_tol': 1e-4, }),
    ('GaussianMixture', GaussianMixture, {'n_components': range(3, 5), 'random_state': 42}),
    # Add other algorithms if desired
]

# Perform clustering
for name, algorithm, params in clustering_algorithms:
    param_range = params[list(params.keys())[0]]
    param_name = list(params.keys())[0]
    for param_value in param_range:
        kwargs = params.copy()
        kwargs[param_name] = param_value
        cluster_labels, n_clusters, silhouette, db_index, calinski_harabasz = perform_clustering(
            name=f'{name}_{param_value}',
            algorithm=algorithm,
            data=pca_result,
            **kwargs
        )
        clustering_results.append({
            'Algorithm': name,
            'Parameters': f'{param_name}={param_value}',
            'Labels': cluster_labels,
            'Number of Clusters': n_clusters,
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': db_index,
            'Calinski-Harabasz Index': calinski_harabasz
        })

# -------------------------
# 5. Visualization with t-SNE
# -------------------------

# Use t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(pca_result)
tsne_df = pd.DataFrame(data=tsne_result, columns=['Dim1', 'Dim2'])

# Prepare for plotting
num_plots = len(clustering_results)
cols = 5
rows = int(np.ceil(num_plots / cols))

plt.figure(figsize=(6, 5 * rows))

for idx, result in enumerate(clustering_results):
    tsne_df['Cluster'] = result['Labels']
    n_clusters = result['Number of Clusters']
    silhouette = result['Silhouette Score']
    db_index = result['Davies-Bouldin Index']
    ch_index = result['Calinski-Harabasz Index']
    
    plt.subplot(rows, cols, idx + 1)
    sns.scatterplot(
        x='Dim1', y='Dim2',
        hue='Cluster',
        palette='tab10',
        data=tsne_df,
        legend=False,
        alpha=0.6
    )
    plt.title(f"{result['Algorithm']} \C: {n_clusters}")

    # plt.title(f"{result['Algorithm']} \C: {n_clusters}, S: {silhouette:.2f}, DB: {db_index:.2f}, CH: {ch_index:.2f}")
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

# -------------------------
# 6. Feature Importance Analysis
# -------------------------

from sklearn.ensemble import RandomForestClassifier

# Analyze feature importance for each clustering algorithm
for result in clustering_results:
    algo_name = result['Algorithm']
    params = result['Parameters']
    print(f"\nFeature Importance for clusters from {algo_name} with {params}")
    
    # Use the original scaled features
    X = scaled_features
    y = result['Labels']
    
    # Ensure that y is suitable for classification
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    feature_importances = pd.Series(clf.feature_importances_, index=features.columns)
    feature_importances.sort_values(ascending=False, inplace=True)
    print(feature_importances.head(10))
    
    # Plot top features
    feature_importances.head(10).plot(kind='barh')
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Features Contributing to Clusters ({algo_name} with {params})')
    plt.gca().invert_yaxis()
    plt.show()
