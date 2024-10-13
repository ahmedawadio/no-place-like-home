import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import time
from playwright.async_api import async_playwright
import asyncio

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


#####################
##################### identify_years_to_query.py
#####################
"""
My goal is to use a model that has a large amount of variables used to describe a locaiton,
in a resonable amount of query time. Because of this, I will use the  Census api on
the ACS 1 and 5 years. I am choosing the Profile api, because the larger data tables are mostly subsets The dataset has the most popular and relevant averages for each 
geographic location. 

In order to choose which variables ot query for each year, I will see which variables are being
contiuousy used across the years. 

The more time we go back the better, but the worse the overlapping variables. 

I will choose the year right before the big drop. I chosse 2019-2023(most recent year).

"""

# import matplotlib.pyplot as plt
# from get_variables import get_census_variables

# x=[]
# y=[]



# for year in range(2015, 2023+1):
#     print(year)
#     x.append(year)
#     y.append(len(get_census_variables(list(range(year, 2023+1)))) )


# # Plot the line chart
# plt.plot(x, y, marker='o', linestyle='-', color='b', label='Number of Variables')

# # Add titles and labels
# plt.title('Census Variable Overlaps Over the Years')
# plt.xlabel('Year')
# plt.ylabel('Number of Variables')

# # Show the grid
# plt.grid(True)

# # Display the chart
# plt.legend()
# plt.show()


#####################
##################### identify_years_to_query.py
#####################





#####################
##################### analysis.py
#####################
# # import pandas as pd
# # import re

# # df = pd.read_csv("api/analysis/data/query.csv",dtype={'mid': str})

# # df= df.dropna()


# # # Extract 'variable' and 'year' from 'variable_year'
# # # Assuming the last 4 characters represent the year
# # df_long['year'] = df_long['variable_year'].apply(lambda x: re.search(r'(\d{4})$', x).group(1) if re.search(r'(\d{4})$', x) else 'Unknown')
# # df_long['variable'] = df_long['variable_year'].apply(lambda x: re.sub(r'_(\d{4})$', '', x) if re.search(r'_(\d{4})$', x) else x)

# # # Display the reshaped DataFrame
# # # print("\nReshaped DataFrame (long format):")
# # # print(df_long.head(10))

# # num_variables = df_long['variable'].nunique()

# # # Print the count of unique variables
# # print(f"\nNumber of unique variables: {num_variables}")



# import pandas as pd
# import numpy as np
# import re
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import SpectralClustering, AffinityPropagation, AgglomerativeClustering, DBSCAN
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import silhouette_score, davies_bouldin_score
# from sklearn.metrics.pairwise import euclidean_distances
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, OPTICS, Birch, SpectralClustering, AffinityPropagation

# from sklearn.metrics import (
#     silhouette_score,
#     davies_bouldin_score,
#     calinski_harabasz_score,
# )

# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import silhouette_score
# from sklearn.metrics.pairwise import euclidean_distances



# # not always good to do, but because i am looping through a bunch of different clustering algorithms, i will suppress warnings
# import warnings
# from sklearn.exceptions import ConvergenceWarning

# # Suppress all warnings
# warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings
# warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Suppress ConvergenceWarnings


# # -------------------------
# # 1. Load and Prepare Data
# # -------------------------

# def plot_row_column_thresholds(df):
#     # Load the CSV file with 'mid' as string
#     df.replace(['', 'None', None, '-888888888'], pd.NA, inplace=True)

#     # Calculate invalid percentages
#     invalid_percentages = df.isna().mean()

#     # Lists to store threshold percentages, rows, and columns
#     thresholds = []
#     num_rows = []
#     num_cols = []

#     # Loop through thresholds from 0% to 99%
#     for threshold_percentage in range(1, 100):  # Start from 1% to avoid threshold=0
#         temp_df = df.copy()  # Use .copy() to avoid modifying the original dataframe

#         # Filter out columns with high invalid percentages
#         cols_to_keep = invalid_percentages[invalid_percentages <= threshold_percentage / 100].index.tolist()

#         # Ensure 'mid' is included
#         if 'mid' not in cols_to_keep:
#             cols_to_keep.insert(0, 'mid')

#         # Filter the DataFrame to only keep columns with valid percentages
#         temp_df = temp_df[cols_to_keep]
        
#         # Drop rows with NaN values
#         temp_df = temp_df.dropna()

#         # Store the threshold percentage, number of rows, and columns
#         thresholds.append(threshold_percentage)
#         num_rows.append(temp_df.shape[0])
#         num_cols.append(temp_df.shape[1])

#     # Plotting the results
#     plt.figure(figsize=(10, 6))
#     plt.plot(thresholds, num_rows, label='Number of Rows', color='blue', marker='o')
#     plt.plot(thresholds, num_cols, label='Number of Columns', color='red', marker='x')

#     plt.xlabel('Threshold Percentage')
#     plt.ylabel('Count')
#     plt.title('Number of Rows and Columns vs Threshold Percentage')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# def shape_threshold(df,threshold_percentage):
#     # Load the CSV file with 'mid' as string
#     df.replace(['', 'None', None, '-888888888'], pd.NA, inplace=True)

#     # Calculate invalid percentages
#     invalid_percentages = df.isna().mean()

#     temp_df = df.copy()  # Use .copy() to avoid modifying the original dataframe

#     # Filter out columns with high invalid percentages
#     cols_to_keep = invalid_percentages[invalid_percentages <= threshold_percentage / 100].index.tolist()

#     # Ensure 'mid' is included
#     if 'mid' not in cols_to_keep:
#         cols_to_keep.insert(0, 'mid')

#     # Filter the DataFrame to only keep columns with valid percentages
#     temp_df = temp_df[cols_to_keep]
    
#     # Drop rows with NaN values
#     temp_df = temp_df.dropna()

#     # Remove any columns that are not of type float or int, excluding 'mid'
#     numeric_cols = temp_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     if 'mid' in temp_df.columns:
#         numeric_cols.insert(0, 'mid')  # Ensure 'mid' stays in the DataFrame
#     temp_df = temp_df[numeric_cols]

#     return temp_df


# filepath  = "api/analysis/data/query.csv"


# # Load the CSV file with 'mid' as string
# df = pd.read_csv(filepath, dtype={'mid': str}, low_memory=False)
# df.replace(['', 'None', None, '-888888888'], pd.NA, inplace=True)

# # Checking the tradeoff of rows / columns to drop based on threshold. Goal is to keep more more rows to ahve more ciities in model
# # plot_row_column_thresholds(df)


# chosen_threshold = 10
# df = shape_threshold(df, chosen_threshold)

# # # Display the first few rows
# # print("First few rows of the cleaned DataFrame:")
# # print(df.head())

# # # Exclude 'mid' column for analysis
# features = df.drop('mid', axis=1)

# # # # Display feature names
# # # print("\nFeature columns:")
# # # print(features.columns.tolist())

# # -------------------------
# # 2. Data Scaling
# # -------------------------

# # Initialize the scaler
# scaler = StandardScaler()

# # Fit and transform the features
# scaled_features = scaler.fit_transform(features)

# # Convert back to DataFrame for easier handling
# scaled_df = pd.DataFrame(scaled_features, columns=features.columns)


# # -------------------------
# # 3. Dimensionality Reduction
# # -------------------------

# def graph_PCA():
#     # Initialize PCA (without specifying n_components to consider all components)
#     pca = PCA()
#     pca_result = pca.fit(scaled_features)

#     # Get the explained variance ratio for each component
#     explained_variance = pca.explained_variance_ratio_

#     # Calculate cumulative explained variance
#     cumulative_variance = np.cumsum(explained_variance)

#     n_components = np.argmax(cumulative_variance >= 0.95) + 1
#     print(f"Number of components required to explain 95% variance: {n_components}")

#     # Choose how many components to display (e.g., first 20 components)
#     chunk_to_viz = 100  # Adjust this number as needed

#     # Plot the explained variance
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(1, chunk_to_viz + 1), explained_variance[:chunk_to_viz], 'bo-', label='Explained Variance per Component')
#     plt.plot(range(1, chunk_to_viz + 1), cumulative_variance[:chunk_to_viz], 'ro-', label='Cumulative Explained Variance')
#     plt.title('Explained Variance vs Number of Components (First 20 Components)')
#     plt.xlabel('Number of Components')
#     plt.ylabel('Variance Explained')
#     plt.xticks(range(1, chunk_to_viz + 1))
#     plt.grid(True)
#     plt.legend(loc='best')
#     plt.show()

# # graph_PCA()

# # Use PCA for preprocessing
# n_components = 81  # Adjust the number of components as needed
# pca = PCA(n_components=n_components, random_state=42)
# pca_result = pca.fit_transform(scaled_features)


# # -------------------------
# # 4. Clustering Algorithms
# # -------------------------


# def find_optimal_k():

#     inertia_values = []
#     k_values = range(2, 15)  # Adjust the range as needed

#     for k in k_values:
#         kmeans = KMeans(n_clusters=k, random_state=1)
#         kmeans.fit(pca_result)
#         inertia_values.append(kmeans.inertia_)

#     print(f"kvalues{k_values} inertia_values{inertia_values}")
#     plt.figure(figsize=(8, 6))
#     plt.plot(k_values, inertia_values, 'bo-')
#     plt.xlabel('Number of Clusters (k)')
#     plt.ylabel('Inertia')
#     plt.title('Elbow Method for Optimal k')
#     plt.grid(True)
#     plt.show()

#     silhouette_avg_values = []
#     k_values = range(2, 15)  # Adjust the range as needed

#     for k in k_values:
#         kmeans = KMeans(n_clusters=k, random_state=1)
#         cluster_labels = kmeans.fit_predict(pca_result)
#         silhouette_avg = silhouette_score(pca_result, cluster_labels)
#         silhouette_avg_values.append(silhouette_avg)


#     print (f"kvalues{k_values} silhouette_avg_values{silhouette_avg_values}")
#     plt.figure(figsize=(8, 6))
#     plt.plot(k_values, silhouette_avg_values, 'ro-')
#     plt.xlabel('Number of Clusters (k)')
#     plt.ylabel('Average Silhouette Score')
#     plt.title('Silhouette Analysis for Optimal k')
#     plt.grid(True)
#     plt.show()

# # find_optimal_k()


# # Define a function to perform clustering and evaluate
# def perform_clustering(name, algorithm, data, **kwargs):
#     algorithm_instance = algorithm(**kwargs)
#     cluster_labels = algorithm_instance.fit_predict(data)
#     n_clusters = len(np.unique(cluster_labels))
    
#     # Exclude noise points for DBSCAN or algorithms that label noise as -1
#     if n_clusters > 1 and -1 not in cluster_labels:
#         silhouette = silhouette_score(data, cluster_labels)
#         db_index = davies_bouldin_score(data, cluster_labels)
#         calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
#         print(f"{name} Sil: {silhouette:.2f}, DB: {db_index:.2f}, CH: {calinski_harabasz:.2f}  ({kwargs}) - ")
#     else:
#         silhouette = np.nan
#         db_index = np.nan
#         calinski_harabasz = np.nan
    
#     return cluster_labels, n_clusters, silhouette, db_index, calinski_harabasz

# # Initialize a list to store results
# clustering_results = []

# # List of clustering algorithms and parameters
# clustering_algorithms = [
#     ('KMeans', KMeans, {'n_clusters': range(6,7), 'random_state': 1}),
#     ('AgglomerativeClustering', AgglomerativeClustering, {'n_clusters': range(8,9)}),
#     ('Birch', Birch, {'n_clusters': range(6,7)}),
#     ('SpectralClustering', SpectralClustering, { 'n_clusters': range(7, 10), 'affinity': 'rbf', 'eigen_tol': 1e-4, }),
#     ('GaussianMixture', GaussianMixture, {'n_components': range(3, 5), 'random_state': 42}),
#     # Add other algorithms if desired
# ]

# # Perform clustering
# for name, algorithm, params in clustering_algorithms:
#     param_range = params[list(params.keys())[0]]
#     param_name = list(params.keys())[0]
#     for param_value in param_range:
#         kwargs = params.copy()
#         kwargs[param_name] = param_value
#         cluster_labels, n_clusters, silhouette, db_index, calinski_harabasz = perform_clustering(
#             name=f'{name}_{param_value}',
#             algorithm=algorithm,
#             data=pca_result,
#             **kwargs
#         )
#         clustering_results.append({
#             'Algorithm': name,
#             'Parameters': f'{param_name}={param_value}',
#             'Labels': cluster_labels,
#             'Number of Clusters': n_clusters,
#             'Silhouette Score': silhouette,
#             'Davies-Bouldin Index': db_index,
#             'Calinski-Harabasz Index': calinski_harabasz
#         })

# # -------------------------
# # 5. Visualization with t-SNE
# # -------------------------

# # Use t-SNE for visualization
# tsne = TSNE(n_components=2, random_state=42)
# tsne_result = tsne.fit_transform(pca_result)
# tsne_df = pd.DataFrame(data=tsne_result, columns=['Dim1', 'Dim2'])

# # Prepare for plotting
# num_plots = len(clustering_results)
# cols = 5
# rows = int(np.ceil(num_plots / cols))

# plt.figure(figsize=(6, 5 * rows))

# for idx, result in enumerate(clustering_results):
#     tsne_df['Cluster'] = result['Labels']
#     n_clusters = result['Number of Clusters']
#     silhouette = result['Silhouette Score']
#     db_index = result['Davies-Bouldin Index']
#     ch_index = result['Calinski-Harabasz Index']
    
#     plt.subplot(rows, cols, idx + 1)
#     sns.scatterplot(
#         x='Dim1', y='Dim2',
#         hue='Cluster',
#         palette='tab10',
#         data=tsne_df,
#         legend=False,
#         alpha=0.6
#     )
#     plt.title(f"{result['Algorithm']} \C: {n_clusters}")

#     # plt.title(f"{result['Algorithm']} \C: {n_clusters}, S: {silhouette:.2f}, DB: {db_index:.2f}, CH: {ch_index:.2f}")
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.xticks([])
#     plt.yticks([])

# plt.tight_layout()
# plt.show()

# # -------------------------
# # 6. Feature Importance Analysis
# # -------------------------

# from sklearn.ensemble import RandomForestClassifier

# # Analyze feature importance for each clustering algorithm
# for result in clustering_results:
#     algo_name = result['Algorithm']
#     params = result['Parameters']
#     print(f"\nFeature Importance for clusters from {algo_name} with {params}")
    
#     # Use the original scaled features
#     X = scaled_features
#     y = result['Labels']
    
#     # Ensure that y is suitable for classification
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X, y)
#     feature_importances = pd.Series(clf.feature_importances_, index=features.columns)
#     feature_importances.sort_values(ascending=False, inplace=True)
#     print(feature_importances.head(10))
    
#     # Plot top features
#     feature_importances.head(10).plot(kind='barh')
#     plt.xlabel('Feature Importance')
#     plt.title(f'Top 10 Features Contributing to Clusters ({algo_name} with {params})')
#     plt.gca().invert_yaxis()
#     plt.show()




#####################
##################### local_database_analysis.py
#####################

# import os
# from dotenv import load_dotenv, find_dotenv
# from supabase import create_client, Client

# envs = [".env.local", ".env.production", ".env.preview"]

# SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
# SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
# CENSUS_API_KEY: str = os.getenv('CENSUS_API_KEY')

# def load_env_files(env_files):
#     """Attempts to load each .env file in the given list"""
#     for env_file in env_files:
#         # Check if the env file exists
#         dotenv_path = find_dotenv(env_file)
#         if dotenv_path:
#             # print(f"Loading environment variables from {env_file}")
#             load_dotenv(dotenv_path=dotenv_path)
#             break  # Stop at the first successfully loaded .env file

# # Attempt to load environment variables from the list of env files
# load_env_files(envs)


#####################
##################### local_database_analysis.py
#####################



#####################
##################### get_images.py
#####################
# from openai import OpenAI
# import pandas as pd
# import requests
# import os
# from keys import OPENAI_API_KEY


# client = OpenAI(api_key=OPENAI_API_KEY)

# def generate_metro_images(
#     csv_filepath: str,
#     output_dir: str,
#     image_size: str = "1024x1024",
#     model: str = "dall-e-3",
#     quality: str = "standard"
# ):
#     print("Reading CSV file...")

#     # Read the first 3 rows of the CSV file
#     try:
#         df = pd.read_csv(csv_filepath)
#         print("CSV file loaded successfully.")

#         # Check if the required columns are present
#         if 'name' in df.columns and 'mid' in df.columns:
#             print("Iterating through each row (first 3 rows):")
            
#             # Ensure output directory exists
#             os.makedirs(output_dir, exist_ok=True)
#             total_locations = len(df)

#             # Loop through each row and generate images
#             for index, row in df.iterrows():
#                 name = row['name']
#                 mid = row['mid']
#                 prompt = f"Make me a single pastel paintbrush-style depiction of the quintessential downtown of: {name}. Do not include text or a paintbrush."
#                 print(f"Row {index}/{total_locations}:, mid:{mid}, name: {name}")

#                 image_path = os.path.join(output_dir, f"{mid}.png")

#                 # Check if the image already exists, if so, skip
#                 if os.path.exists(image_path):
#                     print(f"Image {image_path} already exists. Skipping...")
#                     continue

#                 try:
#                     # Generate image using OpenAI's DALL-E API
#                     response = client.images.generate(
#                         model=model,
#                         prompt=prompt,
#                         size=image_size,
#                         quality=quality,
#                         n=1,
#                     )

#                     # Extract image URL
#                     image_url = response.data[0].url

#                     # Download and save the image
#                     img_response = requests.get(image_url)
#                     if img_response.status_code == 200:
#                         # Construct file path
#                         with open(image_path, 'wb') as img_file:
#                             img_file.write(img_response.content)
#                         print(f"Saved image to {image_path}")
#                     else:
#                         print(f"Failed to download image from {image_url}")

#                 except Exception as e:
#                     print(f"Error generating image for row {index} with value '{name}': {e}")
#         else:
#             print("The specified columns 'name' or 'mid' are missing in the CSV file.")
#             return

#     except FileNotFoundError:
#         print(f"File at {csv_filepath} not found.")
#     except Exception as e:
#         print(f"An error occurred while reading the CSV file: {e}")

# # Example usage
# if __name__ == "__main__":
#     csv_filepath = "api/analysis/data/database/metros.csv"
#     output_dir = "api/analysis/data/bucket"
    
#     generate_metro_images(
#         csv_filepath=csv_filepath,
#         output_dir=output_dir
#     )


#####################
##################### get_images.py
#####################



#####################
##################### extraxt_raw_data.py
#####################
# import pandas as pd

# """
# Extracting all US zip codes and connecting them to metropolitan/micropolitan areas(CBSA).

# I cleaned the format and added it to the db.

# """

# ############################################################################################################
# ###  
# # METROPOLITAN/MICROPOLITAN STATISTICAL AREAS (CBSA) 

# # URL: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
# # Title: Core based statistical areas (CBSAs), metropolitan divisions, and combined statistical areas (CSAs)
# # Category: Jul. 2023 

# ###

# #file path
# cbsa_name_type = "api/analysis/raw_data/cbsa_name_type.xlsx"

# # Read the Excel file into a DataFrame, keeping dtype as streing for leading zeros
# cbsa_df = pd.read_excel(cbsa_name_type, header=0, dtype=str)

# # Extract the specific columns I want to save
# cbsa_cleaned_df = cbsa_df[['CBSA Code', 'CBSA Title', 'Metropolitan/Micropolitan Statistical Area']]

# # Remove duplicates based on 'CBSA Code'
# cbsa_cleaned_df = cbsa_cleaned_df.drop_duplicates(subset='CBSA Code')

# final_cbsa_df = pd.DataFrame()
# final_cbsa_df["cid"] = cbsa_cleaned_df["CBSA Code"]
# final_cbsa_df["name"] = cbsa_cleaned_df["CBSA Title"]
# # final_cbsa_df["type"] = cbsa_cleaned_df["Metropolitan/Micropolitan Statistical Area"]

# final_cbsa_df.to_csv("api/analysis//data/cbsa.csv", index=False)
# # Display the first few rows of the cleaned DataFrame to confirm it worked correctly
# # print(final_cbsa_df.head())



# ############################################################################################################
# ###  ZIP CODES AND CITY,STATE ###


# # (note: a free account is required to download the dataset)
# # URL dataset: https://www.huduser.gov/apps/public/uspscrosswalk/home
# # Category ZIP-CBSA snd Quarter 2024

# # URL dataset Description:  https://www.huduser.gov/portal/datasets/usps_crosswalk.html
# ###

# #file path
# zip_city_cbsa = "api/analysis/raw_data/zip_city_cbsa.xlsx"

# #  keeping dtype as streing for leading zeros
# zip_city_cbsa_df = pd.read_excel(zip_city_cbsa, header=0,dtype=str)

# # Extract the specific columns I want to save
# zip_city_cbsa_df_cleaned = zip_city_cbsa_df[['ZIP', 'CBSA','USPS_ZIP_PREF_CITY', 'USPS_ZIP_PREF_STATE']]

# zip_city_cbsa_df_cleaned = zip_city_cbsa_df_cleaned.drop_duplicates(subset='ZIP')

# #Puerto Rico is not included in model
# zip_city_cbsa_df_cleaned = zip_city_cbsa_df_cleaned[zip_city_cbsa_df_cleaned['USPS_ZIP_PREF_STATE'] != 'PR']

# #99999 is a false zipcode and is often used as placeholder, I belive similar is true for CBSA Any row with this value is removed
# zip_city_cbsa_df_cleaned = zip_city_cbsa_df_cleaned[zip_city_cbsa_df_cleaned['CBSA'] != '99999']

# # Cleaning text formating. Making copy to avoid returning-a-view-versus-a-copy warning
# zip_city_cbsa_df_cleaned = zip_city_cbsa_df_cleaned.copy()
# zip_city_cbsa_df_cleaned['USPS_ZIP_PREF_CITY'] = zip_city_cbsa_df['USPS_ZIP_PREF_CITY'].str.title().copy()


# final_zipcode_df = pd.DataFrame()
# final_zipcode_df["zipcode"] = zip_city_cbsa_df_cleaned["ZIP"]
# final_zipcode_df["city"] = zip_city_cbsa_df_cleaned["USPS_ZIP_PREF_CITY"]
# final_zipcode_df["state"] = zip_city_cbsa_df_cleaned["USPS_ZIP_PREF_STATE"]
# final_zipcode_df["mid"] = zip_city_cbsa_df_cleaned["CBSA"]


# final_zipcode_df.to_csv("api/analysis/data/zipcodes.csv", index=False)
# # print(final_zipcode_df.head())

#####################
##################### extraxt_raw_data.py
#####################
