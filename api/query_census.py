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

# 1. Define the CBSA CSV file path and read it
cbsa_file_path = "api/analysis/data/cbsa.csv"
cbsa_cleaned_df = pd.read_csv(cbsa_file_path)

# 2. Get the current timestamp
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 3. Define the base output directory
output_base_dir = 'api/analysis/data/queries'

# 4. Create a new folder with the timestamp of the request
output_dir = os.path.join(output_base_dir, f"query_{current_time}")
os.makedirs(output_dir, exist_ok=True)

# 5. Define the file paths for the query and error files
query_file_path = os.path.join(output_dir, "query.csv")
error_file_path = os.path.join(output_dir, "errors.csv")

with open(cbsa_file_path, mode='r', newline='', encoding='utf-8') as cbsafile:
    # grabbing count for % completion, subtracting header
    cbsa_count = len(cbsafile.readlines()) -1


variables_count= len(variables_dict.keys())
max_retries = 3
retry_delay = 1 


with open(cbsa_file_path, mode='r', newline='', encoding='utf-8') as cbsafile, \
     open(query_file_path, mode='w', newline='', encoding='utf-8') as queryfile,\
     open(error_file_path, mode='w', newline='', encoding='utf-8') as errorfile:


    # Creating headers
    query_header =  ["cid"]
    for year in years:
        for variable in variables_dict.keys():
            query_header.append(f"{variable}_{year}")
    queryfile.write(','.join(query_header)+"\n")

   

    #not looping through header row
    next(cbsafile)

    for index, row in enumerate(cbsafile):
        columns = row.split(',')
        cbsa_code = columns[0]
        
        cbsa_row = [cbsa_code]
        print(f"Completion rate: {round((index+1)/cbsa_count,3)}")

        for year in years:
            print(f"   {year}")
            variables = ','.join(variables_dict.keys())
            base_url = f"https://api.census.gov/data/{year}/acs/acs5"
            query_url = f"{base_url}?get={variables}&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:{cbsa_code}&key={CENSUS_API_KEY}"

            # Make the request to the Census API

            for attempt in range(max_retries):

                response = requests.get(query_url)

                if response.status_code == 200:
                    data = response.json()
                    # The first row is headers
                    # headers = data[0]
                    values = data[1]
                    cbsa_row.extend(values)

                    retry_delay = 1 #reset
                    break
                elif attempt == max_retries - 1:
                    placeholders = [''] * variables_count  
                    cbsa_row.extend(placeholders)
                    try:
                        error_message = response.json()
                        print(f"Failed CBSA {cbsa_code}, year {year}: {error_message}")
                        errorfile.write(f"Failed CBSA {cbsa_code}, year {year}: {error_message}\n")
                    except ValueError:
                        print(f"Failed CBSA {cbsa_code}, year {year}: HTTP {response.status_code}")
                        errorfile.write(f"Failed CBSA {cbsa_code}, year {year}: HTTP {response.status_code}\n")
                else:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        queryfile.write(','.join(cbsa_row)+"\n")

        #write immediately to file
        queryfile.flush()

print("done")