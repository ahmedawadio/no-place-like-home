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
from datetime import datetime
import time
import logging
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException





variables_dict = {
    'B01001_001E': 'Total population',
    'B01002_001E': 'Median age by sex',
    'B02001_002E': 'White alone population',
    'B02001_003E': 'Black or African American alone population',
    'B19013_001E': 'Median household income',
    'B19301_001E': 'Per capita income',
    'B25077_001E': 'Median home value',
    'B25064_001E': 'Median gross rent',
    'B08013_001E': 'Aggregate travel time to work',
    'B15003_001E': 'Total population 25 years and over',
    'B15003_017E': 'High school graduate (population 25 years and over)',
    'B15003_022E': 'Bachelor\'s degree (population 25 years and over)',
    'B23025_003E': 'Civilian labor force (16 years and over)',
    'B23025_005E': 'Unemployed population (16 years and over)',
    'B17001_002E': 'Population below poverty level',
    'B25035_001E': 'Median year structure built (housing units)',
    'B16010_001E': 'Total population 5 years and over',
    'B16010_002E': 'Population 5 years and over who speak English less than "very well"',
    'B06012_002E': 'Moved from a different state (1 year ago)',
    'B01003_001E': 'Total population (by county or tract)',
    'B25034_001E': 'Year structure built for housing units'
}
import os
import pandas as pd
from keys import CENSUS_API_KEY
import requests
from analysis.extraxt_raw_data import cbsa_cleaned_df
from datetime import datetime
import time
import logging
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException

# 1. Define your variables and years
variables_dict = {
    'B01001_001E': 'Total population',
    'B01002_001E': 'Median age by sex',
    'B02001_002E': 'White alone population',
    'B02001_003E': 'Black or African American alone population',
    'B19013_001E': 'Median household income',
    'B19301_001E': 'Per capita income',
    'B25077_001E': 'Median home value',
    'B25064_001E': 'Median gross rent',
    'B08013_001E': 'Aggregate travel time to work',
    'B15003_001E': 'Total population 25 years and over',
    'B15003_017E': 'High school graduate (population 25 years and over)',
    'B15003_022E': 'Bachelor\'s degree (population 25 years and over)',
    'B23025_003E': 'Civilian labor force (16 years and over)',
    'B23025_005E': 'Unemployed population (16 years and over)',
    'B17001_002E': 'Population below poverty level',
    'B25035_001E': 'Median year structure built (housing units)',
    'B16010_001E': 'Total population 5 years and over',
    'B16010_002E': 'Population 5 years and over who speak English less than "very well"',
    'B06012_002E': 'Moved from a different state (1 year ago)',
    'B01003_001E': 'Total population (by county or tract)',
    'B25034_001E': 'Year structure built for housing units'
}

years = list(range(2012, 2023))

# 2. Define the CBSA CSV file path and read it
cbsa_file_path = "api/analysis/data/cbsa.csv"
cbsa_cleaned_df = pd.read_csv(cbsa_file_path)

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

# Optional: Also log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# 7. Determine the total number of CBSA codes for progress tracking
with open(cbsa_file_path, mode='r', newline='', encoding='utf-8') as cbsafile:
    cbsa_count = len(cbsafile.readlines()) - 1  # Subtract header

variables_count = len(variables_dict.keys())
max_retries = 3
initial_retry_delay = 1  # Starting delay for retries

# Initialize a session for connection pooling
session = requests.Session()

with open(cbsa_file_path, mode='r', newline='', encoding='utf-8') as cbsafile, \
     open(query_file_path, mode='w', newline='', encoding='utf-8') as queryfile:

    # 8. Create headers for the query CSV
    query_header = ["cid"]
    for year in years:
        for variable in variables_dict.keys():
            query_header.append(f"{variable}_{year}")
    queryfile.write(','.join(query_header) + "\n")
    queryfile.flush()

    # 9. Skip the header row in the CBSA CSV
    next(cbsafile)

    # 10. Iterate through each CBSA code
    for index, row in enumerate(cbsafile):
        columns = row.strip().split(',')
        cbsa_code = columns[0]
        cbsa_row = [cbsa_code]
        completion_rate = round((index + 1) / cbsa_count, 3)
        logging.info(f"Processing CBSA {cbsa_code} - Completion rate: {completion_rate}")

        for year in years:
            logging.info(f"   Year: {year}")
            variables = ','.join(variables_dict.keys())
            base_url = f"https://api.census.gov/data/{year}/acs/acs5"
            query_url = f"{base_url}?get={variables}&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:{cbsa_code}&key={CENSUS_API_KEY}"

            retry_delay = initial_retry_delay  # Initialize retry delay for each request
            for attempt in range(max_retries):
                try:
                    response = session.get(query_url, timeout=10)  # Increased timeout
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 1:
                            values = [v if v is not None else '' for v in data[1]]
                            cbsa_row.extend(values)
                            logging.info(f"   Success: CBSA {cbsa_code}, Year {year}")
                        else:
                            placeholders = [''] * variables_count
                            cbsa_row.extend(placeholders)
                            logging.warning(f"   Unexpected data format for CBSA {cbsa_code}, Year {year}")
                            logging.warning(f"   URL: {query_url}")
                        break  # Exit retry loop on success or unexpected data format
                    elif response.status_code == 204:
                        placeholders = [''] * variables_count
                        cbsa_row.extend(placeholders)
                        logging.warning(f"   No data for CBSA {cbsa_code}, Year {year}")
                        break  # Exit retry loop since there's no data
                    else:
                        response.raise_for_status()  # Raise HTTPError for bad responses
                except Timeout:
                    if attempt == max_retries - 1:
                        placeholders = [''] * variables_count
                        cbsa_row.extend(placeholders)
                        logging.error(f"   Timeout: CBSA {cbsa_code}, Year {year}")
                    else:
                        logging.warning(f"   Timeout occurred. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except ConnectionError as ce:
                    if attempt == max_retries - 1:
                        placeholders = [''] * variables_count
                        cbsa_row.extend(placeholders)
                        logging.error(f"   Connection error: CBSA {cbsa_code}, Year {year}: {ce}")
                    else:
                        logging.warning(f"   Connection error. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                except HTTPError as he:
                    if attempt == max_retries - 1:
                        placeholders = [''] * variables_count
                        cbsa_row.extend(placeholders)
                        logging.error(f"   HTTP error: CBSA {cbsa_code}, Year {year}: {he}")
                    else:
                        logging.warning(f"   HTTP error. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                except RequestException as e:
                    if attempt == max_retries - 1:
                        placeholders = [''] * variables_count
                        cbsa_row.extend(placeholders)
                        logging.error(f"   Request exception: CBSA {cbsa_code}, Year {year}: {e}")
                    else:
                        logging.warning(f"   Request exception. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
            else:
                # If all retries failed and placeholders have not been added, add them here
                placeholders = [''] * variables_count
                cbsa_row.extend(placeholders)
                logging.error(f"   All retries failed for CBSA {cbsa_code}, Year {year}")

        # 11. Ensure all elements in cbsa_row are strings
        cbsa_row_cleaned = [str(x) if x is not None else '' for x in cbsa_row]
        queryfile.write(','.join(cbsa_row_cleaned) + "\n")
        queryfile.flush()

        # Optional: Sleep briefly to avoid hitting rate limits
        time.sleep(0.1)  # Sleep for 100 milliseconds

    # 12. Close the session after all requests are done
    session.close()

logging.info(f"Query completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
print("done", datetime.now().strftime("%Y%m%d_%H%M%S"))
