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
import os
import requests

# -------------------------
# 1. Load and Prepare Data
# -------------------------

def shape_threshold(df, threshold_percentage):
    # As of my year writing this (2024) full data is not available for all micro cbsa areas, so cleaning the df in this manner will exclude all 2023 data until more data is available from census

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
print(df.shape)

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


cluster_counts = clustered_df['Cluster'].value_counts().sort_index()

# print("cluster_counts",cluster_counts)



# -------------------------
# 6. Nearest Neighbors for All Clusters
# -------------------------

def get_all_nearest_neighbors(clustered_df, n_neighbors=5):
    """
    Find the nearest neighbors for each 'cid' within its cluster.
    If not enough neighbors are found, look into the next closest clusters.

    Parameters:
    - clustered_df (pd.DataFrame): DataFrame containing PCA components, 'Cluster', and 'cid'.
    - n_neighbors (int): Number of nearest neighbors to find.

    Returns:
    - Dictionary where keys are cluster IDs and values are DataFrames with 'cid' and their nearest neighbors.
    """
    from collections import defaultdict
    import numpy as np
    from scipy.spatial.distance import cdist

    # Initialize a dictionary to store nearest neighbors for each cluster
    nearest_neighbors_by_cluster = defaultdict(pd.DataFrame)
    
    # Get list of PCA columns
    pca_columns = [col for col in clustered_df.columns if col.startswith('PC')]

    # Compute centroids of clusters
    cluster_centroids = clustered_df.groupby('Cluster')[pca_columns].mean()

    # Compute pairwise distances between centroids
    centroid_distances = cdist(cluster_centroids.values, cluster_centroids.values)

    # Create cluster proximity mapping
    cluster_proximity = {}
    for i, cluster_id in enumerate(cluster_centroids.index):
        distances = centroid_distances[i]
        # Get indices of clusters ordered by distance
        sorted_indices = np.argsort(distances)
        ordered_clusters = cluster_centroids.index[sorted_indices]
        # Exclude itself
        ordered_clusters = [cid for cid in ordered_clusters if cid != cluster_id]
        cluster_proximity[cluster_id] = ordered_clusters

    # Now, for each cluster
    for cluster_id in clustered_df['Cluster'].unique():
        # Subset data for the current cluster
        cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id].copy()

        # Get list of 'cid's in the cluster
        cids_in_cluster = cluster_data['cid'].values

        # Check the size of the cluster
        cluster_size = cluster_data.shape[0]
        if cluster_size == 0:
            # Skip empty clusters
            continue

        # For each 'cid' in the cluster
        neighbors_list = []
        for idx, cid in enumerate(cids_in_cluster):
            own_pca = cluster_data[cluster_data['cid'] == cid][pca_columns].values  # 1 x n_components

            # Initialize lists for neighbors and distances
            neighbor_cids = []
            neighbor_distances = []

            # First, get distances to other points in the same cluster (excluding itself)
            same_cluster_data = cluster_data[cluster_data['cid'] != cid]
            if not same_cluster_data.empty:
                distances = cdist(own_pca, same_cluster_data[pca_columns].values)[0]
                neighbor_cids.extend(same_cluster_data['cid'].values)
                neighbor_distances.extend(distances)

            # If not enough neighbors, look into other clusters
            if len(neighbor_cids) < n_neighbors:
                # Loop over other clusters in order of proximity
                for other_cluster_id in cluster_proximity[cluster_id]:
                    other_cluster_data = clustered_df[clustered_df['Cluster'] == other_cluster_id]
                    if other_cluster_data.empty:
                        continue
                    # Compute distances to all points in the other cluster
                    distances = cdist(own_pca, other_cluster_data[pca_columns].values)[0]
                    neighbor_cids.extend(other_cluster_data['cid'].values)
                    neighbor_distances.extend(distances)
                    if len(neighbor_cids) >= n_neighbors:
                        break  # We have enough neighbors

            # Create a DataFrame of neighbors and distances
            neighbors_df = pd.DataFrame({
                'cid': neighbor_cids,
                'distance': neighbor_distances
            })

            # Sort by distance
            neighbors_df = neighbors_df.sort_values('distance')

            # Exclude any duplicates (unlikely, but just in case)
            neighbors_df = neighbors_df.drop_duplicates(subset='cid')

            # Take top n_neighbors
            top_neighbors = neighbors_df.head(n_neighbors)['cid'].tolist()

            # Add to neighbors list
            neighbors_list.append(top_neighbors)

        # Add to the dictionary
        nearest_neighbors_by_cluster[cluster_id] = pd.DataFrame({
            'cid': cids_in_cluster,  
            'Nearest Neighbors': neighbors_list
        })
    
    return nearest_neighbors_by_cluster

# Exampl Get nearest neighbors for all clusters
nearest_neighbors_all = get_all_nearest_neighbors(clustered_df, n_neighbors=5, )

# Define the target cid
# target_cid = "35620"

# # Initialize a variable to store the result
# found_neighbor = None

# # Iterate over each cluster's DataFrame
# for cluster_id, df in nearest_neighbors_all.items():
#     # Check if the target_cid exists in the current DataFrame
#     if target_cid in df['cid'].values:
#         # Extract the row where 'cid' equals the target_cid
#         found_neighbor = df[df['cid'] == target_cid]
#         # print(f"Found cid {target_cid} in cluster {cluster_id}:")
#         print(found_neighbor)
#         break  # Stop once we find the first match


def save_similar_metros_to_csv(nearest_neighbors_by_cluster, output_filepath, overwrite=False):
    """
    Saves the nearest neighbors data to a CSV file with two columns:
    'mid' (as a string) and 'similar_mid' (as a comma-separated list of mids).

    Parameters:
    - nearest_neighbors_by_cluster (dict): 
        Dictionary where keys are cluster IDs and values are DataFrames 
        with 'cid' and 'Nearest Neighbors'.
    - output_filepath (str): 
        The file path where the CSV will be saved.
    - overwrite (bool): 
        If True, overwrite the file if it exists. 
        If False, skip saving to prevent overwriting existing files. Default is False.

    Returns:
    - None
    """
    try:
        # Create a list to hold all rows to write to CSV
        all_rows = []

        # Loop through each cluster in the dictionary
        for cluster_id, df in nearest_neighbors_by_cluster.items():
            for idx, row in df.iterrows():
                # Extract 'cid' and 'Nearest Neighbors'
                mid = row['cid']
                similar_mid_list = row['Nearest Neighbors']
                
                # Ensure similar_mid_list is a list
                if isinstance(similar_mid_list, list):
                    # Convert list to comma-separated string
                    similar_mid_str = ','.join(map(str, similar_mid_list))
                else:
                    # If it's not a list, convert it to string directly
                    similar_mid_str = str(similar_mid_list)
                
                # Append to all_rows
                all_rows.append([mid, similar_mid_str])

        # Create a DataFrame from the rows
        output_df = pd.DataFrame(all_rows, columns=['mid', 'similar_mid'])

        # Check if the file already exists
        if not os.path.exists(output_filepath):
            # Save the DataFrame to a CSV file
            output_df.to_csv(output_filepath, index=False)
            print(f"File saved to {output_filepath}")
        elif overwrite:
            # Overwrite the existing file
            output_df.to_csv(output_filepath, index=False)
            print(f"File '{output_filepath}' existed and was overwritten.")
        else:
            # Skip saving to prevent overwriting
            print(f"File '{output_filepath}' already exists. Skipping save to prevent overwriting.")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
similar_metros_filepath = "api/analysis/data/database/similar_metros.csv"
save_similar_metros_to_csv(nearest_neighbors_all, similar_metros_filepath,overwrite=False)



# Display nearest neighbors for a specific cluster (e.g., Cluster 0)
# cluster_id_to_inspect = 1
# if cluster_id_to_inspect in nearest_neighbors_all:
#     print(f"\nNearest Neighbors for Cluster {cluster_id_to_inspect}:")
#     print(nearest_neighbors_all[cluster_id_to_inspect])
# else:
#     print(f"\nNo nearest neighbors found for Cluster {cluster_id_to_inspect}.")


# -------------------------
# 7. Feature Importance Analysis
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
top_sanitized_features = get_top_features_sanitized(scaled_features, cluster_labels, feature_names, top_n=5)

# print("Top Important Features for KMeans Clusters after Sanitization:")
# for idx, feature in enumerate(top_sanitized_features, start=1):
#     print(f"{idx}. {feature}: {variables_dictionary[feature]}  ")


# print(top_sanitized_features)


#I translated census variables regular english
variables_dictionary = {
    # found from top features
    'DP04_0026PE': {
        "name": 'Pre-1939 Housing', 
        "description": 'Percent of housing units built in 1939 or earlier', 
        "type": 'percent'
    },
    'DP04_0134E': {
        "name": 'Median Rent', 
        "description": 'Median rent of occupied units paying rent (dollars)', 
        "type": 'dollars'
    },
    'DP04_0103PE': {
        "name": 'No Mortgage <$250', 
        "description": 'Percent of housing units without a mortgage with monthly costs less than $250', 
        "type": 'percent'
    },
    'DP04_0087E': {
        "name": '$500k-$999k Homes', 
        "description": 'Value of owner-occupied units between $500,000 and $999,999', 
        "type": 'count'
    },
    'DP03_0119PE': {
        "name": 'Families Below Poverty', 
        "description": 'Percent of families with income below poverty level in the past 12 months', 
        "type": 'percent'
    },
    'DP04_0131E': {
        "name": 'Rent $2k to $2.5k', 
        "description": 'Rent of occupied units paying $2,000 to $2,499', 
        "type": 'count'
    },
    'DP04_0132E': {
        "name": 'Rent $2.5k to $3k', 
        "description": 'Rent of occupied units paying $2,500 to $2,999', 
        "type": 'count'
    },
    'DP03_0136PE': {
        "name": 'People Below Poverty', 
        "description": 'Percent of people in families with income below poverty level in the past 12 months', 
        "type": 'percent'
    },
    'DP05_0023PE': {
        "name": 'Pop Age 62+', 
        "description": 'Percent of total population aged 62 years and over', 
        "type": 'percent'
    },
    'DP04_0065PE': {
        "name": 'Electric Heating', 
        "description": 'Percent of housing units using electricity for heating', 
        "type": 'percent'
    },

    # added myself
    'DP05_0001E': {
        "name": 'Total Population', 
        "description": 'Total population', 
        "type": 'count'
    },    
    'DP05_0018E': {
        "name": 'Median Age',
        "description": 'Median age of the population',
        "type": 'years'
    },
    'DP04_0046E': {
        "name": 'Owner-Occupied Housing Units',
        "description": 'Number of housing units that are owner-occupied',
        "type": 'count'
    },
    'DP04_0093E': {
        "name": 'Housing Units w/ Mortgage',
        "description": 'Number of housing units that have a mortgage',
        "type": 'count'
    },}

    
    # 'B19013_001E': {
    #     "name": 'Median Income', 
    #     "description": 'Median household income', 
    #     "type": 'dollars'
    # },
    # 'B25077_001E': {
    #     "name": 'Median Home Value', 
    #     "description": 'Median home value', 
    #     "type": 'dollars'
    # },
    # 'B25064_001E': {
    #     "name": 'Median Rent', 
    #     "description": 'Median gross rent (dollars)', 
    #     "type": 'dollars'
    # },
    # 'B23025_005E': {
    #     "name": 'Unemployed', 
    #     "description": 'Number of unemployed individuals', 
    #     "type": 'count'
    # },
    # 'B15003_022E': {
    #     "name": "Bachelor's Degrees", 
    #     "description": 'Number of bachelor’s degrees', 
    #     "type": 'count'
    # },
    # 'B08301_003E': {
    #     "name": 'Solo Commuters', 
    #     "description": 'Number of solo commuters in vehicles', 
    #     "type": 'count'
    # },
    # 'B27010_018E': {
    #     "name": 'Insured Middle-Aged', 
    #     "description": 'Number of middle-aged adults with health insurance', 
    #     "type": 'count'
    # },







def save_variables(var_dict, output_filepath, overwrite=False):
    """
    Processes and saves variables to a CSV file.

    Parameters:
    - var_dict (dict): Dictionary mapping ACS variable codes to their details including name, description, and type.
    - output_filepath (str): The file path where the CSV will be saved.
    - overwrite (bool): If True, overwrite the file if it exists. If False, skip saving to prevent overwriting. Default is False.

    Returns:
    - None
    """
    try:
        # Check if the output file already exists
        if os.path.exists(output_filepath):
            if overwrite:
                print(f"File '{output_filepath}' exists and will be overwritten as per the overwrite parameter.")
            else:
                print(f"File '{output_filepath}' already exists. Skipping save to prevent overwriting.")
                return  # Exit the function without saving
        else:
            print(f"File '{output_filepath}' does not exist and will be created.")

        # Convert the variables dictionary to a DataFrame with columns for variable_code, name, description, and type
        data = [
            {
                "variable_code": variable_code,
                "name": details["name"],
                "description": details["description"],
                "type": details["type"]
            }
            for variable_code, details in var_dict.items()
        ]
        
        df_variables = pd.DataFrame(data, columns=['variable_code', 'name', 'description', 'type'])

        # Save the DataFrame to a CSV file
        df_variables.to_csv(output_filepath, index=False)
        print(f"Variables saved successfully to '{output_filepath}'.")

    except Exception as e:
        print(f"An error occurred while saving the variables: {e}")

# Define the output file path for variables
variables_filepath = "api/analysis/data/database/variables.csv"

# Call the function to save variables
save_variables(variables_dictionary, variables_filepath, overwrite=True)

# -------------------------
# 8. Create variable files
# -------------------------


# Now lets filter based on population, and top_10_sanitized_features, and other important features

# Step 1: Melt the DataFrame to long format
df_long = df.melt(id_vars=['cid'], var_name='variable_year', value_name='value')

df_long['year'] = df_long['variable_year'].apply(lambda x: re.search(r'(\d{4})$', x).group(1) if re.search(r'(\d{4})$', x) else 'Unknown')
df_long['variable'] = df_long['variable_year'].apply(lambda x: re.sub(r'_(\d{4})$', '', x) if re.search(r'_(\d{4})$', x) else x)


# Step 3: Drop the intermediate 'variable_year' column
df_long.drop('variable_year', axis=1, inplace=True)

df_long.rename(columns={'cid': 'mid'}, inplace=True)

# Step 4: Reorder columns for better readability
df_long = df_long[['mid', 'variable', 'year', 'value']]



print("Shape before subsetting:", df_long.shape)

df_long = df_long[df_long['variable'].isin(variables_dictionary.keys())]

print("Shape after subsetting:", df_long.shape)

print(df_long.head(50))

def save_metro_metrics(long_data_frame, var_dict, output_filepath, overwrite=False):
    """
    Processes and saves metro metrics to a CSV file.

    Parameters:
    - long_data_frame (pd.DataFrame): The long-format DataFrame containing 'mid', 'variable', 'year', and 'value'.
    - var_dict (dict): Dictionary mapping ACS variable codes to their descriptions.
    - output_filepath (str): The file path where the CSV will be saved.
    - overwrite (bool): If True, overwrite the file if it exists. If False, skip saving to prevent overwriting. Default is False.

    Returns:
    - None
    """
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_processed = long_data_frame.copy()

        # Map 'variable' to 'description' using var_dict
        df_processed['description'] = df_processed['variable'].map(var_dict)

        
        # Handle variables that might not be in var_dict
        missing_descriptions = df_processed['description'].isnull().sum()
        if missing_descriptions > 0:
            print(f"Warning: {missing_descriptions} variables were not found in variables_dictionary and will have NaN descriptions.")

        # Reorder columns: 'mid', 'year', 'variable', 'value'

        df_processed['variable_code'] = df_processed['variable']
        df_processed = df_processed[['mid', 'year', 'variable_code',  'value']]

        # Check if the output file already exists
        if os.path.exists(output_filepath):
            if overwrite:
                # Overwrite the existing file
                df_processed.to_csv(output_filepath, index=False)
                print(f"File '{output_filepath}' existed and was overwritten.")
            else:
                # Skip saving to prevent overwriting
                print(f"File '{output_filepath}' already exists. Skipping save to prevent overwriting.")
        else:
            # Save the DataFrame to a CSV file
            df_processed.to_csv(output_filepath, index=False)
            print(f"File saved to {output_filepath}")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


# Define the output file path
metro_metrics_filepath = "api/analysis/data/database/metro_metrics.csv"

# Call the function to save metro metrics
save_metro_metrics(df_long, variables_dictionary, metro_metrics_filepath, overwrite=True)


# -------------------------
# 9. Update metros file
# -------------------------


def save_metros(start_year=2019, end_year=2023, overwrite=False, webscrape=False):
    """
    Fetches metropolitan/micropolitan statistical area data from the Census API for each year,
    processes it to extract 'NAME' and 'mid',
    optionally performs web scraping to collect additional data,
    and saves all data to a single 'metros.csv' file at the end.

    Parameters:
    - start_year (int): The starting year for data retrieval. Default is 2019.
    - end_year (int): The ending year for data retrieval. Default is 2024.
    - overwrite (bool): If True, overwrite existing 'metros.csv'. If False, skip saving to prevent overwriting. Default is False.
    - webscrape (bool): If True, performs web scraping to collect additional data. Default is False.

    Returns:
    - None
    """
    try:
        # Ensure the output directory exists
        output_dir = "api/analysis/data/database/"
        os.makedirs(output_dir, exist_ok=True)

        # Initialize a list to collect DataFrames for each year
        all_years_data = []

        # Iterate through each year
        for year in range(start_year, end_year + 1):
            print(f"\nProcessing year: {year}")

            # Define both ACS 5-Year and ACS 1-Year API URLs
            acs5_url = f"https://api.census.gov/data/{year}/acs/acs5/profile?get=NAME&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*"
            acs1_url = f"https://api.census.gov/data/{year}/acs/acs1/profile?get=NAME&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*"

            # Initialize variables
            data = None
            used_url = None

            # Attempt to fetch data using ACS 5-Year
            try:
                print(f"Attempting to fetch data from ACS 5-Year for year {year}...")
                response = requests.get(acs5_url)
                response.raise_for_status()  # Raise an error for bad status codes
                data = response.json()
                used_url = "acs5"
                print(f"Successfully fetched data from ACS 5-Year for year {year}.")
            except requests.exceptions.RequestException as e:
                print(f"ACS 5-Year request failed for year {year}: {e}")
                print(f"Switching to ACS 1-Year for year {year}...")
                # Attempt to fetch data using ACS 1-Year
                try:
                    response = requests.get(acs1_url)
                    response.raise_for_status()
                    data = response.json()
                    used_url = "acs1"
                    print(f"Successfully fetched data from ACS 1-Year for year {year}.")
                except requests.exceptions.RequestException as e1:
                    print(f"ACS 1-Year request also failed for year {year}: {e1}")
                    print(f"Skipping year {year} due to data retrieval failures.")
                    continue  # Skip to the next year

            # Proceed if data was successfully fetched
            if data:
                try:
                    # Extract headers and rows
                    header = data[0]
                    rows = data[1:]

                    # Create DataFrame
                    df_metro = pd.DataFrame(rows, columns=header)

                    # Rename columns for clarity
                    df_metro.rename(columns={
                        "NAME": "name",
                        "metropolitan statistical area/micropolitan statistical area": "mid"
                    }, inplace=True)

                    # Convert 'mid' to string (if not already)
                    df_metro['mid'] = df_metro['mid'].astype(str)

                    # Add 'year' column
                    df_metro['year'] = year

                    # Optional: Check for duplicate 'mid' values
                    duplicates = df_metro[df_metro.duplicated('mid', keep=False)]
                    if not duplicates.empty:
                        print(f"Year {year}: Found duplicate 'mid' values. Keeping the first occurrence and removing duplicates.")
                        df_metro = df_metro.drop_duplicates(subset='mid', keep='first')
                        print(f"Year {year}: Number of records after removing duplicates: {len(df_metro)}")
                    else:
                        print(f"Year {year}: No duplicate 'mid' values found.")

                    # Append the processed DataFrame to the list
                    all_years_data.append(df_metro)

                except KeyError as e:
                    print(f"Year {year}: Key error - Missing expected column {e}. Skipping this year.")
                except Exception as e:
                    print(f"Year {year}: An unexpected error occurred during data processing: {e}")

        # After processing all years, concatenate all DataFrames
        if all_years_data:
            combined_df = pd.concat(all_years_data, ignore_index=True)
            combined_df = combined_df[['mid', 'name']]
            combined_df = combined_df.drop_duplicates(subset='mid', keep='first')

            # Remove the last 10 characters from 'name' (assuming ' Metro Area' or similar)
            combined_df['name'] = combined_df['name'].apply(lambda x: x[:-10] if isinstance(x, str) else x)

            print(f"\nTotal records collected from {start_year} to {end_year}: {len(combined_df)}")

            # If webscrape is True, perform web scraping
            if webscrape:
                print("Starting web scraping for each metro area...")
                # Collect additional data via web scraping
                scraped_data = asyncio.run(web_scrape_multiple(combined_df['name'].tolist()))

                # Merge scraped data into combined_df
                combined_df = pd.merge(combined_df, scraped_data, on='name', how='left')

            # Define the output filepath
            metros_filepath = os.path.join(output_dir, "metros2.csv")

            # Check if the file exists and handle overwrite
            if os.path.exists(metros_filepath) and not overwrite:
                print(f"File '{metros_filepath}' already exists. Skipping save to prevent overwriting.")
            else:
                # Save the combined DataFrame to CSV
                combined_df.to_csv(metros_filepath, index=False)
                print(f"Combined metros data saved to '{metros_filepath}' successfully with {len(combined_df)} records.")
        else:
            print("\nNo data was fetched and combined.")
    except Exception as e:
        print(f"An error occurred during data retrieval and processing: {e}")

def adjust_place_name(name):
    """
    Adjusts the place name by taking the part before the first dash
    and appending the comma and state abbreviation (if not already present).

    Parameters:
    - name (str): The original metro name.

    Returns:
    - str: The adjusted place name.
    """
    dash_index = name.find('-')
    if dash_index != -1:
        name_part = name[:dash_index].strip()
    else:
        name_part = name.strip()
    
    comma_index = name.find(',')
    if comma_index != -1:
        # Include comma and next two letters (state abbreviation)
        state_part = name[comma_index:comma_index+4]
        # Check if state abbreviation is already in name_part, if so, skip appending it
        if state_part in name_part:
            adjusted_name = name_part
        else:
            adjusted_name = name_part + state_part
    else:
        adjusted_name = name_part  # No state abbreviation available, return name_part alone

    return adjusted_name

async def web_scrape_data_usa(browser, search_query, total_names, current_index):
    """
    Webscrapes the Data USA website to find information for the given search query,
    retries with adjusted parameters if initial attempt fails,
    and collects section titles and descriptions on the resulting page.

    Parameters:
    browser: The browser instance from Playwright.
    search_query (str): The query to search for.
    total_names (int): Total number of names being processed (for progress tracking).
    current_index (int): The current index of the name being processed.

    Returns:
    dict: A dictionary containing the scraped data.
    """
    # Mapping dictionary to convert titles to desired lowercase format
    title_map = {
        'About': 'about',
        'Population & Diversity': 'population',
        'Economy': 'economy',
        'Civics': 'civics',
        'Education': 'education',
        'Housing & Living': 'housing',
        'Health': 'health'
    }

    section_data = {'name': search_query}
    data_found = False  # Flag to indicate whether data was found
    data_source_url = ""  # To store the URL used

    # Define initial and alternative URLs
    initial_url = "https://datausa.io/search/?q=&dimension=Geography&hierarchy=MSA"
    alternative_url = "https://datausa.io/search/?q=&dimension=Geography"

    try:
        page = await browser.new_page()

        # First attempt with initial URL and search query
        await page.goto(initial_url)
        await page.wait_for_timeout(2000)

        # Perform search
        search_input = page.locator("input.bp3-input[placeholder='Find a report...']")
        await search_input.fill(search_query)
        await page.wait_for_timeout(2000)
        await search_input.press("ArrowDown")
        await page.wait_for_timeout(500)
        await search_input.press("Enter")
        await page.wait_for_timeout(3000)

        # Collect sections
        sections = page.locator(".section-title, .section-description")
        section_count = await sections.count()

        if section_count > 0:
            data_found = True
            data_source_url = initial_url
        else:
            # No data found, try alternative URL and adjusted search query
            adjusted_query = adjust_place_name(search_query)
            await page.goto(alternative_url)
            await page.wait_for_timeout(2000)

            # Perform search with adjusted query
            search_input = page.locator("input.bp3-input[placeholder='Find a report...']")
            await search_input.fill(adjusted_query)
            await page.wait_for_timeout(2000)
            await search_input.press("ArrowDown")
            await page.wait_for_timeout(500)
            await search_input.press("Enter")
            await page.wait_for_timeout(3000)

            # Collect sections
            sections = page.locator(".section-title, .section-description")
            section_count = await sections.count()

            if section_count > 0:
                data_found = True
                data_source_url = alternative_url
                section_data['name'] = adjusted_query  # Update name to adjusted query
            else:
                print(f"No data found for '{search_query}' even after adjusting the query.")

        # If data was found, process sections
        if data_found:
            for i in range(0, section_count, 2):
                title_element = sections.nth(i)
                description_element = sections.nth(i + 1) if i + 1 < section_count else None
                title = await title_element.text_content()

                # Collecting all <p> elements within the description and joining with \n\n delimiter
                if description_element:
                    paragraphs = await description_element.locator('p').all_text_contents()
                    description = "\n\n".join(paragraphs)
                else:
                    description = ""

                if title:
                    # Apply the title mapping and use lowercase keys
                    mapped_title = title_map.get(title.strip(), title.strip().lower())
                    section_data[mapped_title] = description.strip() if description else ""

                    # Print the length of each section's description
                    description_length = len(description.strip()) if description else 0
                    print(f"Processed '{mapped_title}' section for '{search_query}': Description Length = {description_length} characters")

            # Add 'data_source' to section_data
            section_data['data_source'] = data_source_url

    except Exception as e:
        print(f"An error occurred during web scraping for '{search_query}': {e}")
    finally:
        await page.close()

    return section_data

async def web_scrape_multiple(names):
    """
    Performs web scraping for multiple names and collects the data into a DataFrame.

    Parameters:
    - names (iterable): An iterable of names to search and scrape.

    Returns:
    - DataFrame containing the scraped data for each name.
    """
    total_names = len(names)
    completed = 0

    # Limit the number of concurrent tasks to prevent overwhelming the server
    semaphore = asyncio.Semaphore(5)  # Adjust the concurrency limit as appropriate

    async def sem_task(name, browser, index):
        async with semaphore:
            data = await web_scrape_data_usa(browser, name, total_names, index)
            return data

    results = []

    async with async_playwright() as p:
        # Set up the Playwright browser
        browser = await p.chromium.launch(headless=True)
        tasks = [sem_task(name, browser, idx + 1) for idx, name in enumerate(names)]
        # Use asyncio.as_completed to process tasks as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            percent_complete = (completed / total_names) * 100
            print(f"Progress: {percent_complete:.2f}% complete ({completed}/{total_names})")

        await browser.close()

    # Convert the results list to a DataFrame
    scraped_df = pd.DataFrame(results)
    return scraped_df


def check_metros_for_nulls(filepath="api/analysis/data/database/metros.csv"):
    """
    Reads the metros CSV file and checks for null values and empty strings
    in each column, providing a count of occurrences.

    Parameters:
    - filepath (str): The path to the metros CSV file. Default is 'api/analysis/data/database/metros.csv'.

    Returns:
    - None: Prints the results to the console.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        print(f"File '{filepath}' loaded successfully. Total records: {len(df)}\n")

        # Check for null values
        null_counts = df.isnull().sum()
        # print("Null values count per column:")
        # print(null_counts)
        # print()

        # Check for empty strings
        empty_string_counts = (df.applymap(lambda x: x == '')).sum()
        # print("Empty string count per column:")
        # print(empty_string_counts)
        # print()

        # Combine both null and empty counts for a total 'missing' count
        total_missing_counts = null_counts + empty_string_counts
        print("Total missing values (nulls + empty strings) per column:")
        print(total_missing_counts)

    except FileNotFoundError:
        print(f"File not found at '{filepath}'. Please check the file path.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


def check_empty_rows(filepath="api/analysis/data/database/metros.csv"):
    """
    Reads the metros CSV file and identifies rows where all columns 
    (excluding 'mid' and 'name') are either NULL or empty strings.

    Parameters:
    - filepath (str): The path to the metros CSV file. Default is 'api/analysis/data/database/metros.csv'.

    Returns:
    - None: Prints the results to the console.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        print(f"File '{filepath}' loaded successfully. Total records: {len(df)}\n")

        # Define the columns to check for emptiness (excluding 'mid' and 'name')
        columns_to_check = ['about', 'population', 'economy', 'education', 'housing', 'health', 'civics']
        
        # Create a boolean mask for rows where all specified columns are either null or empty
        mask_empty = df[columns_to_check].applymap(lambda x: pd.isnull(x) or x == '').all(axis=1)
        
        # Count and display the rows with no meaningful data in these columns
        empty_rows_count = mask_empty.sum()
        print(f"Number of rows with all '{columns_to_check}' columns empty or NULL (excluding 'mid' and 'name'): {empty_rows_count}")

        # Optionally, show the specific rows if desired
        if empty_rows_count > 0:
            print("\nRows with no data in specified columns (excluding 'mid' and 'name'):")
            print(df[mask_empty])
        return df[mask_empty]

    except FileNotFoundError:
        print(f"File not found at '{filepath}'. Please check the file path.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


# save_metros(overwrite=False, webscrape=False)

# check_metros_for_nulls("api/analysis/data/database/metros.csv")


# -------------------------
# 9. Update zipcode file
# -------------------------

def save_zipcodes(read_filepath, save_filepath, overwrite=False):
    """
    Reads the 'zipcodes' CSV, filters rows based on clustered metro ids from the global 'df' DataFrame,
    and saves the result to a new CSV.

    Parameters:
    - read_filepath (str): The file path to read the 'zipcodes' CSV from.
    - save_filepath (str): The file path to save the filtered 'zipcodes' CSV to.
    - overwrite (bool): If True, overwrite the file if it exists. If False, skip saving to prevent overwriting.
      Default is False.

    Returns:
    - None
    """
    try:
        # Read the 'zipcodes' CSV into a DataFrame
        df_zipcodes = pd.read_csv(read_filepath,dtype={'mid': str},)
        print(f"'zipcodes' CSV loaded successfully. Number of records: {len(df_zipcodes)}")
        print(df_zipcodes)
        # Access the global 'df' DataFrame
        global df

        mid_df = df.copy()
        mid_df["mid"] = df["cid"]
        print("mid_df",mid_df)

        df_zipcodes = df_zipcodes[df_zipcodes['mid'].isin(mid_df['mid'].to_list())]

        print("after",df_zipcodes)

        # Check if the output file already exists
        if os.path.exists(save_filepath):
            if overwrite:
                # Overwrite the existing file
                df_zipcodes.to_csv(save_filepath, index=False)
                print(f"File '{save_filepath}' existed and was overwritten.")
            else:
                # Skip saving to prevent overwriting
                print(f"File '{save_filepath}' already exists. Skipping save to prevent overwriting.")
        else:
            # Save the DataFrame to a CSV file
            df_zipcodes.to_csv(save_filepath, index=False)
            print(f"File saved to {save_filepath}")

    except Exception as e:
        print(f"An error occurred while processing 'zipcodes': {e}")

# Define file paths
zipcodes_read_filepath = "api/analysis/data/zipcodes.csv"
zipcodes_save_filepath = "api/analysis/data/database/zipcodes.csv"



# Call the function to save filtered 'zipcodes'
save_zipcodes(zipcodes_read_filepath, zipcodes_save_filepath, overwrite=False)

