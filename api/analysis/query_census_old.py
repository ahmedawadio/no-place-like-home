"""
In order to not consume all my free credits for my cloud functions, 
I will only query Census API locally(which can take a good amountof time), 
then populate the db. 

The cloud functions will only connect to the db and compute similarty scores 

"""
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


years = list(range(2019, 2023+1))

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


variables_count = len(acs_variables_dict.keys())
max_retries = 3
initial_retry_delay = 1  # Starting delay for retries

# Initialize a session for connection pooling
session = requests.Session()

def process_variables_in_chunks(variables_dict:dict, chunk_size: int = 49):
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



with open(query_file_path, mode='w', newline='', encoding='utf-8') as queryfile:

    # 8. Create headers for the query CSV
    query_header = ["mid"]
    for year in years:
        for variable in acs_variables_dict.keys():
            query_header.append(f"{variable}_{year}")
    queryfile.write(','.join(query_header) + "\n")
    queryfile.flush()


    # 10. Iterate through each CBSA code

    for index, chunk in enumerate(process_variables_in_chunks(acs_variables_dict)):
        logging.info(f"Processing chunk {index + 1} of {len(acs_variables_dict) // 49 + 1}")
        variables = ','.join(chunk)

    # for index, row in enumerate(cbsafile):
        # columns = row.strip().split(',')
        # cbsa_code = columns[0]
        cbsa_row = []
        # completion_rate = round((index + 1) / cbsa_count, 3)
        # logging.info(f"Processing CBSA {cbsa_code} - Completion rate: {completion_rate}")

        for year in years:
            logging.info(f"   Year: {year}")
            variables = ','.join(acs_variables_dict.keys())
            base_url = f"https://api.census.gov/data/{year}/acs/acs1/profile"
            query_url = f"{base_url}?get={variables}&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*&key={CENSUS_API_KEY}"

            retry_delay = initial_retry_delay  # Initialize retry delay for each request
            for attempt in range(max_retries):
                try:
                    response = session.get(query_url, timeout=10)  # Increased timeout
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 1:

                            for cbsa_data in data[1]:                        
                                cbsa_row = [lst[-1]] + lst[:-1]  # Move the CBSA code to the beginning
                                cbsa_row = [str(v) if v is not None else '' for v in data[1]] #enturing all elements are strings
                                queryfile.write(','.join(cbsa_row) + "\n")
                                queryfile.flush() # Flush the buffer to write to the file
                            # cbsa_row.extend(values)
                            logging.info(f"   Success: Year {year}")
                        else:
                            logging.warning(f"   Unexpected data format for Year {year}")
                            sanitized_url = re.sub(r'key=\w+', 'key=***', url) #removing key
                            logging.warning(f"   URL: {sanitized_url}")
                        break  # Exit retry loop on success or unexpected data format
                    elif response.status_code == 204:
                        logging.warning(f"   No data for Year {year}")
                        session.close()  # Close the session to release resources, because variables will be out of sync
                    else:
                        logging.warning(f"   Unkown error  for Year {year}")
                        response.raise_for_status()  # Raise HTTPError for bad responses
                        session.close()  # Close the session to release resources, because variables will be out of sync
                except Timeout:
                    if attempt == max_retries - 1:
                        logging.error(f"   Timeout: Year {year}")
                        session.close()  # Close the session to release resources, because variables will be out of sync
                    else:
                        logging.warning(f"   Timeout occurred. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except ConnectionError as ce:
                    if attempt == max_retries - 1:
                        logging.error(f"   Connection error: Year {year}: {ce}")
                        session.close()  # Close the session to release resources, because variables will be out of sync

                    else:
                        logging.warning(f"   Connection error. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                except HTTPError as he:
                    if attempt == max_retries - 1:
                        logging.error(f"   HTTP error: CBSA Year {year}: {he}")
                        session.close()  # Close the session to release resources, because variables will be out of sync
                    else:
                        logging.warning(f"   HTTP error. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                except RequestException as e:
                    if attempt == max_retries - 1:
                        logging.error(f"   Request exception: Year {year}: {e}")
                        session.close()  # Close the session to release resources, because variables will be out of sync
                    else:
                        logging.warning(f"   Request exception. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
            else:
                # If all retries failed
                logging.error(f"   All retries failed for Year {year}")
                session.close()  # Close the session to release resources, because variables will be out of sync


        # Optional: Sleep briefly to avoid hitting rate limits
        time.sleep(0.1)  # Sleep for 100 milliseconds

    # 12. Close the session after all requests are done
    session.close()

logging.info(f"Query completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
print("done", datetime.now().strftime("%Y%m%d_%H%M%S"))
