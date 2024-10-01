import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import re
# Suppress warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Suppress ConvergenceWarnings

# -------------------------
# 1. Load and Prepare Data
# -------------------------

def shape_threshold(df, threshold_percentage):
    # Replace invalid entries with NaN
    df.replace(['', 'None', None, '-888888888'], pd.NA, inplace=True)

    # Calculate the percentage of missing values per column
    invalid_percentages = df.isna().mean()

    # Filter columns based on the threshold
    cols_to_keep = invalid_percentages[invalid_percentages <= threshold_percentage / 100].index.tolist()

    # Ensure 'cid' is included
    if 'cid' not in cols_to_keep:
        cols_to_keep.insert(0, 'cid')

    # Keep only the selected columns
    temp_df = df[cols_to_keep].copy()

    # Drop rows with any remaining NaN values
    temp_df.dropna(inplace=True)

    # Keep only numeric columns (float64 and int64), excluding 'cid'
    numeric_cols = temp_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'cid' in temp_df.columns:
        numeric_cols.insert(0, 'cid')  # Ensure 'cid' is included

    temp_df = temp_df[numeric_cols].copy()

    return temp_df

# File path to your CSV
filepath  = "api/analysis/data/query.csv"

# Load the CSV file with 'cid' as string
df = pd.read_csv(filepath, dtype={'cid': str}, low_memory=False)
df.replace(['', 'None', None, '-888888888'], pd.NA, inplace=True)

# Apply thresholding to clean the data
chosen_threshold = 10
df = shape_threshold(df, chosen_threshold)

# Exclude 'cid' column for analysis
features = df.drop('cid', axis=1)

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

# Apply PCA
n_components = 81  # Number of principal components
pca = PCA(n_components=n_components, random_state=42)
pca_result = pca.fit_transform(scaled_features)

# -------------------------
# 4. Clustering Algorithms
# -------------------------

# Perform KMeans clustering with k=6
kmeans = KMeans(n_clusters=6, random_state=1)
cluster_labels = kmeans.fit_predict(pca_result)

# Create a DataFrame to store the cluster labels and PCA results
clustered_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
clustered_df['Cluster'] = cluster_labels
clustered_df['cid'] = df['cid'].values  # Ensure alignment with original data

# -------------------------
# 5. Define the Nearest Neighbors Function
# -------------------------

def find_nearest_neighbors(target_cid, clustered_df, n_neighbors=5):
    """
    Find the nearest neighbors for a given 'cid' within its cluster.

    Parameters:
    - target_cid (str): The 'cid' of the target city.
    - clustered_df (pd.DataFrame): DataFrame containing PCA components, 'Cluster', and 'cid'.
    - n_neighbors (int): Number of nearest neighbors to find.

    Returns:
    - List of nearest neighbor 'cid's.
    """
    # Check if target_cid exists
    if target_cid not in clustered_df['cid'].values:
        raise ValueError(f"cid '{target_cid}' not found in the data.")

    # Get the cluster of the target_cid
    target_cluster = clustered_df.loc[clustered_df['cid'] == target_cid, 'Cluster'].values[0]

    # Get the data points in the same cluster, excluding the target_cid
    cluster_data = clustered_df[clustered_df['Cluster'] == target_cluster].copy()
    cluster_data = cluster_data[cluster_data['cid'] != target_cid]

    cluster_size = cluster_data.shape[0]

    if cluster_size == 0:
        print(f"No other points in cluster {target_cluster} for cid '{target_cid}'.")
        return []
    
    # Adjust n_neighbors if necessary
    effective_n_neighbors = min(n_neighbors, cluster_size)

    # Get the feature vectors (assuming PCA components are columns PC1, PC2, ..., PC81)
    pca_columns = [col for col in clustered_df.columns if col.startswith('PC')]
    target_features = clustered_df.loc[clustered_df['cid'] == target_cid, pca_columns].values

    # Fit NearestNeighbors on the cluster data
    nbrs = NearestNeighbors(n_neighbors=effective_n_neighbors, metric='euclidean')
    nbrs.fit(cluster_data[pca_columns])

    # Find the nearest neighbors
    distances, indices = nbrs.kneighbors(target_features)

    # Retrieve 'cid's of nearest neighbors
    nearest_cids = cluster_data.iloc[indices[0]]['cid'].values.tolist()

    return nearest_cids

# -------------------------
# 6. Example Usage of the Function
# -------------------------

# Example target cid (replace '12345' with an actual cid from your data)
target_cid = '12345'  # Replace with a valid cid from your dataset
desired_neighbors = 5

try:
    nearest_cities = find_nearest_neighbors(target_cid, clustered_df, n_neighbors=desired_neighbors)
    print(f"Nearest neighbors to cid '{target_cid}': {nearest_cities}")
except ValueError as e:
    print(e)

# -------------------------
# 7. (Optional) Nearest Neighbors for All Clusters
# -------------------------

def get_all_nearest_neighbors(clustered_df, n_neighbors=5):
    """
    Find the nearest neighbors for each 'cid' within its respective cluster.

    Parameters:
    - clustered_df (pd.DataFrame): DataFrame containing PCA components, 'Cluster', and 'cid'.
    - n_neighbors (int): Number of nearest neighbors to find.

    Returns:
    - Dictionary where keys are cluster IDs and values are DataFrames with 'cid' and their nearest neighbors.
    """
    from collections import defaultdict

    # Initialize a dictionary to store nearest neighbors for each cluster
    nearest_neighbors_by_cluster = defaultdict(pd.DataFrame)
    
    # Get list of PCA columns
    pca_columns = [col for col in clustered_df.columns if col.startswith('PC')]

    # Loop over each cluster
    for cluster_id in clustered_df['Cluster'].unique():
        # Subset data for the current cluster
        cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id].copy()
        
        # Check the size of the cluster
        cluster_size = cluster_data.shape[0]
        if cluster_size <= 1:
            print(f"Cluster {cluster_id} has only {cluster_size} sample(s). Skipping Nearest Neighbors search.")
            continue  # Skip clusters with only one sample
        
        # Adjust n_neighbors
        current_n_neighbors = min(n_neighbors, cluster_size - 1)
        print(f"Processing Cluster {cluster_id} with {cluster_size} samples. Finding {current_n_neighbors} neighbors per 'cid'.")
        
        # Fit NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=current_n_neighbors, metric='euclidean')
        nbrs.fit(cluster_data[pca_columns])
        
        # Find neighbors
        distances, indices = nbrs.kneighbors(cluster_data[pca_columns])
        
        # Create a list of nearest neighbors for each 'cid'
        neighbors_list = []
        for idx, neighbors in enumerate(indices):
            neighbor_cids = cluster_data.iloc[neighbors]['cid'].values.tolist()
            neighbors_list.append(neighbor_cids)
        
        # Add to the dictionary
        nearest_neighbors_by_cluster[cluster_id] = pd.DataFrame({
            'cid': cluster_data['cid'].values,
            'Nearest Neighbors': neighbors_list
        })
    
    return nearest_neighbors_by_cluster

# Exampl Get nearest neighbors for all clusters
nearest_neighbors_all = get_all_nearest_neighbors(clustered_df, n_neighbors=5)

# # Display nearest neighbors for a specific cluster (e.g., Cluster 0)
# cluster_id_to_inspect = 1
# if cluster_id_to_inspect in nearest_neighbors_all:
#     print(f"\nNearest Neighbors for Cluster {cluster_id_to_inspect}:")
#     print(nearest_neighbors_all[cluster_id_to_inspect])
# else:
#     print(f"\nNo nearest neighbors found for Cluster {cluster_id_to_inspect}.")

print("done")



# -------------------------
# 6. Feature Importance Analysis
# -------------------------

def get_top_features_sanitized(scaled_features, cluster_labels, feature_names, top_n=10):
    """
    Identify the top N most important features contributing to the clustering after sanitizing feature names by removing the '_year' suffix.
    
    Parameters:
    - scaled_features (np.ndarray or pd.DataFrame): Scaled feature matrix used for clustering.
    - cluster_labels (np.ndarray or pd.Series): Cluster labels assigned by the clustering algorithm.
    - feature_names (list or pd.Index): Original feature names with '_year' suffix.
    - top_n (int): Number of top features to return.
    
    Returns:
    - top_features (list): List of top N sanitized feature names sorted by aggregated importance.
    """
    # Initialize the RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the classifier to predict cluster labels
    clf.fit(scaled_features, cluster_labels)
    
    # Extract feature importances
    importances = clf.feature_importances_
    
    # Create a DataFrame with feature names and their importances
    feature_importances = pd.DataFrame({
        'Original_Feature': feature_names,
        'Importance': importances
    })
    
    # Sanitize feature names by removing the '_year' suffix
    feature_importances['Sanitized_Feature'] = feature_importances['Original_Feature'].apply(
        lambda x: re.sub(r'_\d{4}$', '', x)  # Removes '_year' where year is 4 digits at the end
    )
    
    # Aggregate importances by sanitized feature names
    aggregated_importances = feature_importances.groupby('Sanitized_Feature')['Importance'].sum().reset_index()
    
    # Sort the features by aggregated importance in descending order
    aggregated_importances_sorted = aggregated_importances.sort_values(by='Importance', ascending=False)
    
    # Retrieve the top N features
    top_features = aggregated_importances_sorted.head(top_n)['Sanitized_Feature'].tolist()
    
    return top_features

# Get feature names
feature_names = features.columns.tolist()

# Call the function to get top 10 sanitized features
top_10_sanitized_features = get_top_features_sanitized(scaled_features, cluster_labels, feature_names, top_n=10)

variables_dictionary = get_census_variables_for_1_year(2023)

print("Top 10 Important Features for KMeans Clusters (k=6) after Sanitization:")
for idx, feature in enumerate(top_10_sanitized_features, start=1):
    print(f"{idx}. {feature}: {variables_dictionary[feature]}  ")


