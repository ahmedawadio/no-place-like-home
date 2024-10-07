import pandas as pd
import os
import requests



def save_metros(start_year=2019, end_year=2023, overwrite=False):
    """
    Fetches metropolitan/micropolitan statistical area data from the Census API for each year,
    processes it to extract 'NAME' and 'mid', maps 'mid' using the global 'df' DataFrame's 'cid',
    and saves all data to a single 'metros.csv' file at the end.

    Parameters:
    - start_year (int): The starting year for data retrieval. Default is 2019.
    - end_year (int): The ending year for data retrieval. Default is 2024.
    - overwrite (bool): If True, overwrite existing 'metros.csv'. If False, skip saving to prevent overwriting. Default is False.

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

                    # Map 'mid' using global 'df' DataFrame's 'cid'
                    if 'cid' in df.columns:
                        # Ensure 'cid' is also string
                        df['cid'] = df['cid'].astype(str)
                        cid_set = set(df['cid'].unique())

                        original_count = len(df_metro)
                        df_metro = df_metro[df_metro['mid'].isin(cid_set)]
                        filtered_count = len(df_metro)
                        removed_count = original_count - filtered_count

                        print(f"Year {year}: Filtered metros to include only those with 'mid' present in 'df' 'cid' column.")
                        print(f"Year {year}: Removed {removed_count} records; {filtered_count} records remain.")
                    else:
                        print("Warning: Global DataFrame 'df' does not contain 'cid' column. Skipping 'mid' mapping.")

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
        
            print(f"\nTotal records collected from {start_year} to {end_year}: {len(combined_df)}")

            # Define the output filepath
            metros_filepath = os.path.join(output_dir, "metros.csv")

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


save_metros(overwrite=false)


