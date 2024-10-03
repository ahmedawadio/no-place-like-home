import pandas as pd
import os


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
# from get_variables import get_census_variables

# kmeans_vars = [
#     'DP04_0026PE',
#     'DP04_0134E',
#     'DP04_0026PE',
#     'DP04_0026PE',
#     'DP04_0134E',
#     'DP04_0103PE',
#     'DP04_0026PE',
#     'DP03_0108PE',
#     'DP04_0131E',
#     'DP04_0132E'
# ]


# birch_vars = [
#     'DP03_0136PE',
#     'DP04_0026PE',
#     'DP03_0108PE',
#     'DP04_0088E',
#     'DP04_0134E',
#     'DP03_0119PE',
#     'DP03_0097PE',
#     'DP03_0119PE',
#     'DP04_0026PE',
#     'DP03_0097PE'
# ]


# variables = get_census_variables([2023])

# kmeans_vars = [variables[var] for var in kmeans_vars if var in variables]
# """
# [
#     'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
#     'Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)',
#     'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
#     'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
#     'Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)',
#     'Percent!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage!!Less than $250',
#     'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
#     'Percent!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population 19 to 64 years!!In labor force:!!Employed:!!No health insurance coverage',
#     'Estimate!!GROSS RENT!!Occupied units paying rent!!$2,000 to $2,499',
#     'Estimate!!GROSS RENT!!Occupied units paying rent!!$2,500 to $2,999'
# ]
# """
# birch_vars = [variables[var] for var in birch_vars if var in variables]

# """[
#     'Percent!!PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL!!All people!!People in families',
#     'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
#     'Percent!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population 19 to 64 years!!In labor force:!!Employed:!!No health insurance coverage',
#     'Estimate!!VALUE!!Owner-occupied units!!$1,000,000 or more',
#     'Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)',
#     'Percent!!PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL!!All families',
#     'Percent!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population!!With health insurance coverage!!With private health insurance',
#     'Percent!!PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL!!All families',
#     'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
#     'Percent!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population!!With health insurance coverage!!With private health insurance'
# ]
# """

# print(kmeans_vars)
# print(birch_vars)


# """printed pretty


# """

import pandas as pd

# # Path to your metros.csv
# metros_filepath = "api/analysis/data/database/metros.csv"

# # Load the CSV
# df_metros = pd.read_csv(metros_filepath)

# # Display the number of records
# print(f"'metros.csv' contains {len(df_metros)} records.")

# # Optionally, display the first few rows
# print(df_metros.tail())

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
zipcodes_save_filepath = "api/analysis/data/database/zipcodes_filtered.csv"



# Call the function to save filtered 'zipcodes'
save_zipcodes(zipcodes_read_filepath, zipcodes_save_filepath, overwrite=True)

