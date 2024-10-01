import os
import pandas as pd
from keys import CENSUS_API_KEY
import requests
from analysis.extraxt_raw_data import cbsa_cleaned_df
from get_variables import get_census_variables
from datetime import datetime
import time
import logging
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException
import re

years = list(range(2019, 2023 + 1))

acs_variables_dict = get_census_variables(years)

# 3. Get the current timestamp
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 4. Define the base output directory and create it
output_base_dir = 'api/analysis/data/queries'
output_dir = os.path.join(output_base_dir, f"query_{current_time}")
os.makedirs(output_dir, exist_ok=True)

# 5. Define the file paths for the query file and log file
query_file_path = os.path.join(output_dir, "query.csv")
log_file_path = os.path.join(output_dir, "query_log.log")

# 6. Configure logging to write to the log file
logging.basicConfig(
    filename=log_file_path,
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

max_retries = 3
initial_retry_delay = 1  # Starting delay for retries

# Initialize a session for connection pooling
session = requests.Session()

def process_variables_in_chunks(variables_dict: dict, chunk_size: int = 49):
    # Get all the keys from the dictionary
    variable_keys = list(variables_dict.keys())
    # Calculate the number of chunks required
    total_chunks = (len(variable_keys) + chunk_size - 1) // chunk_size  # This ensures rounding up
    # Initialize an empty list to store the chunks
    chunks_list = []
    # Loop through each chunk
    for i in range(total_chunks):
        # Get the start and end indices for the current chunk
        start = i * chunk_size
        end = min(start + chunk_size, len(variable_keys))  # Ensures we don't go out of bounds
        # Get the current chunk of keys (not key-value pairs)
        current_chunk = variable_keys[start:end]
        # Append this chunk of keys to the list
        chunks_list.append(current_chunk)
    # Return the list of chunks
    return chunks_list

# Prepare the header for the CSV
query_header = ["cid"]
for year in years:
    for variable in acs_variables_dict.keys():
        query_header.append(f"{variable}_{year}")

# Initialize a dictionary to hold data for each CBSA
cbsa_data_dict = {}
invalid_value_counts = {}
total_cbsa_count = 0


for index, chunk in enumerate(process_variables_in_chunks(acs_variables_dict)):
    logging.info(f"Processing chunk {index + 1} of {(len(acs_variables_dict) + 48) // 49}")
    chunk_variables = chunk  # Variables in the current chunk

    for year in years:
        logging.info(f"   Year: {year}")
        variables = ','.join(chunk_variables)

        retry_delay = initial_retry_delay  # Initialize retry delay for each request
        for attempt in range(max_retries):
            try:
                base_url = f"https://api.census.gov/data/{year}/acs/acs5/profile"
                
                #acs5 is more reliable than acs1, but acs5 is not avialable for all years
                # NOTE - acs5 is not available for 2023 as of my time now(2024), only acs1 check for ur time if this is still the case
                if retry_delay!=initial_retry_delay or year==2023:
                    base_url = f"https://api.census.gov/data/{year}/acs/acs1/profile"
                    logging.warning(f"   Attempting acs1 for Year {year}")

                query_url = f"{base_url}?get={variables}&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*&key={CENSUS_API_KEY}"

                response = session.get(query_url, timeout=10)  # Increased timeout
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 1:
                        header = data[0]
                        for row in data[1:]:
                            cbsa_code = row[-1]
                            if cbsa_code not in cbsa_data_dict:
                                cbsa_data_dict[cbsa_code] = {}
                            # Map variable names with year suffix
                            for var_name, value in zip(header[:-1], row[:-1]):  # Exclude 'for' variable
                                key = f"{var_name}_{year}"
                                cbsa_data_dict[cbsa_code][key] = value
                        logging.info(f"   Success: Year {year}")
                    else:
                        logging.warning(f"   Unexpected data format for Year {year}")
                        sanitized_url = re.sub(r'key=\w+', 'key=***', query_url)  # removing key
                        logging.warning(f"   URL: {sanitized_url}")
                    break  # Exit retry loop on success or unexpected data format
                elif response.status_code == 204:
                    logging.warning(f"   No data for Year {year}")
                else:
                    logging.warning(f"   Unknown error for Year {year}")
                    response.raise_for_status()  # Raise HTTPError for bad responses
            except (Timeout, ConnectionError, HTTPError, RequestException) as e:
                if attempt == max_retries - 1:
                    logging.error(f"   Error: Year {year}: {e}")
                else:
                    logging.warning(f"   Error occurred: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        else:
            # If all retries failed
            logging.error(f"   All retries failed for Year {year}")
        break
    # Optional: Sleep briefly to avoid hitting rate limits
    time.sleep(0.1)  # Sleep for 100 milliseconds
    break


# 12. Close the session after all requests are done
session.close()


# After data collection is complete
# Calculate invalid value percentages
invalid_value_percentages = {}
for var_key, invalid_count in invalid_value_counts.items():
    percentage = (invalid_count / total_cbsa_count) * 100
    invalid_value_percentages[var_key] = percentage

# Set threshold percentage
threshold_percentage = 10

# Variables to keep
variables_to_keep = [var for var, perc in invalid_value_percentages.items() if perc <= threshold_percentage]

# Update query_header to include only variables to keep
query_header = ["cid"] + variables_to_keep

# Update cbsa_data_dict to include only variables to keep
for cbsa_code, data_dict in cbsa_data_dict.items():
    # Retain only variables that are in variables_to_keep
    filtered_data_dict = {var: data_dict.get(var, '') for var in variables_to_keep}
    cbsa_data_dict[cbsa_code] = filtered_data_dict
    print(cbsa_code, data_dict)



# Write the data to the CSV file
with open(query_file_path, mode='w', newline='', encoding='utf-8') as queryfile:
    # Write headers
    queryfile.write(','.join(query_header) + "\n")
    # Write data for each CBSA
    for cbsa_code, data_dict in cbsa_data_dict.items():
        row = [cbsa_code]
        for col in query_header[1:]:  # Skip 'cid'
            row.append(str(data_dict.get(col, '')))
        queryfile.write(','.join(row) + "\n")

# Variables removed
variables_removed = [var for var in invalid_value_percentages if var not in variables_to_keep]

# Log removed variables
logging.info(f"Variables removed due to high invalid percentages (> {threshold_percentage}%):")
for var in variables_removed:
    logging.info(f"{var}: {invalid_value_percentages[var]:.2f}% invalid values")

logging.info(f"Query completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
print("done", datetime.now().strftime("%Y%m%d_%H%M%S"))