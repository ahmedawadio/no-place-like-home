
# running local vs prod, requires diff imports styles, thanks turbo repo :/
try: from .keys import SUPABASE_URL,SUPABASE_SERVICE_ROLE_KEY
except: from keys import SUPABASE_URL,SUPABASE_SERVICE_ROLE_KEY

from supabase import create_client, Client
import os
# import pandas as pd


supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def get_location(zipcode):
    """
    - variables
    - metros
    - zipcodes
    - metro_metrics
    - similar_metros
    """
    
    try:
        zipcode_details = supabase.table("zipcodes").select("*").eq("zipcode", zipcode).execute().data[0]
        initial_zipcode_found = True

    except:
        initial_zipcode_found = False
        all_zipcode = supabase.table("zipcodes").select("zipcode").execute().data
        zipcode_list = [int(zipcode['zipcode']) for zipcode in all_zipcode]
        closest_zipcode = min(zipcode_list, key=lambda x: abs(x - int(zipcode)))
        zipcode_details = supabase.table("zipcodes").select("*").eq("zipcode", closest_zipcode).execute().data[0]


    del zipcode_details['creation_date']

    zipcode_mid =  zipcode_details["mid"]


    similar_mids_string = supabase.table("similar_metros").select("similar_mid").eq("mid",zipcode_mid).execute().data[0]['similar_mid']
    mids_list = [zipcode_mid] +  similar_mids_string.split(',')
    metro_metrics = []
    metro_details = []

    # print(mids_list)

    # Populate metro metrics and metro details
    for mid in mids_list:
        # Retrieve metrics for the current metro ID
        metro_metric_data = supabase.table("metro_metrics").select("*").eq("mid", mid).execute().data
        metros_details = supabase.table("metros").select("*").eq("mid", mid).execute().data

        # Remove 'creation_date' from metrics and metro details
        for metric in metro_metric_data:
            if "creation_date" in metric.keys():
                del metric['creation_date']

        for metro in metros_details:
            if "creation_date" in metro.keys():
                del metro['creation_date']

        # Append the structured metrics group to metro_metrics
        metro_metrics.append({
            "metro_id": mid,
            "metrics": metro_metric_data
        })

        # Append metro details to the list
        metro_details.append(metros_details[0])
    variables_list = supabase.table("variables").select("*").execute().data
    for v in variables_list:
        if "creation_date" in v.keys():
            del v['creation_date']




    output = {
            "initial_zipcode_found": initial_zipcode_found,
            "initial_zipcode": zipcode,
            "zipcode": zipcode_details,
            "metro_details": metro_details,
            "metro_metrics": metro_metrics,
            "variables": variables_list
        }
    return output


def insert_into_tables():
    """
    Inserts data from CSV files into corresponding Supabase tables.

    CSV files should be located in 'api/data/database/' and named exactly as the tables:
    - variables.csv
    - metros.csv
    - zipcodes.csv
    - metro_metrics.csv
    - similar_metros.csv

    The function handles upserts to manage conflicts based on primary keys.

    Returns:
    - dict: A summary of insertion results for each table.
    """
    # Define the tables in the order that respects foreign key constraints
    tables = [
        {
            'name': 'variables',
            'primary_keys': ['variable_code']
        },
        {
            'name': 'metros',
            'primary_keys': ['mid']
        },
        {
            'name': 'zipcodes',
            'primary_keys': ['zipcode']
        },
        {
            'name': 'metro_metrics',
            'primary_keys': ['mid', 'year', 'variable_code']
        },
        {
            'name': 'similar_metros',
            'primary_keys': ['mid', 'similar_mid']
        }
    ]

    # Define Base Filepath
    base_filepath = "api/analysis/data/database/"

    # Initialize Result Dictionary
    insertion_results = {}

    # Iterate Through Each Table
    for table in tables:
        table_name = table['name']
        primary_keys = table['primary_keys']
        filepath = os.path.join(base_filepath, f"{table_name}.csv")

        print(f"\nProcessing table: '{table_name}'")
        print(f"CSV file path: '{filepath}'")

        # Check if the CSV file exists
        if not os.path.exists(filepath):
            print(f"Error: CSV file for table '{table_name}' not found at '{filepath}'. Skipping this table.")
            insertion_results[table_name] = {
                'status': 'Skipped',
                'reason': 'CSV file not found'
            }
            continue

        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath)
            print(f"CSV file '{filepath}' loaded successfully. Number of records: {len(df)}")

        

            # Convert DataFrame to list of dictionaries
            records = df.to_dict(orient='records')

            if not records:
                print(f"No records found in '{filepath}'. Skipping insertion.")
                insertion_results[table_name] = {
                    'status': 'Skipped',
                    'reason': 'No records to insert'
                }
                continue


            # Define on_conflict parameter as comma-separated primary keys
            on_conflict = ','.join(primary_keys)

            print("records length", len(records))
            # Perform upsert operation
            response = supabase.table(table_name).upsert(records, on_conflict=on_conflict).execute()

            print(response)
            # Check for errors
    

            # Determine the number of inserted or updated rows
            inserted_count = len(response.data)
            print(f"Successfully inserted/updated {inserted_count} records into '{table_name}'.")

            # Retrieve the first row after insertion for verification
            first_row_response = supabase.table(table_name).select("*").limit(1).execute()
            first_row = first_row_response.data[0] if first_row_response.data else None

            # Store the results
            insertion_results[table_name] = {
                'status': 'Success',
                'inserted_rows': inserted_count,
                'first_row': first_row
            }

        except Exception as e:
            print(f"An exception occurred while processing table '{table_name}': {e}")
            insertion_results[table_name] = {
                'status': 'Failed',
                'error': str(e)
            }
            continue

    # Print Summary of Insertions
    print("\nInsertion Summary:")
    for table, result in insertion_results.items():
        print(f"Table: {table}")
        print(f"  Status: {result['status']}")
        if result['status'] == 'Success':
            print(f"  Inserted/Updated Rows: {result['inserted_rows']}")
            print(f"  First Row: {result['first_row']}")
        elif result['status'] == 'Failed':
            print(f"  Error: {result['error']}")
        elif result['status'] == 'Skipped':
            print(f"  Reason: {result['reason']}")
        print()

    return insertion_results



def confirm_tables_created():

    """
    Supabase does not currently allow table creation in py.

    Go into your supabase dashboard/sql editor and paste this (not counting {}): {
         -- 1. Create metros table
        CREATE TABLE IF NOT EXISTS
        metros (
            mid VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            creation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- 2. Create zipcodes table
        CREATE TABLE IF NOT EXISTS
        zipcodes (
            zipcode VARCHAR PRIMARY KEY,
            city VARCHAR NOT NULL,
            state VARCHAR NOT NULL,
            mid VARCHAR NOT NULL,
            creation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_zip_metro FOREIGN KEY (mid) REFERENCES metros (mid) ON DELETE CASCADE ON UPDATE CASCADE
        );

        -- 3. Create variables table

       CREATE TABLE IF NOT EXISTS
        variables (
            variable_code VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            description TEXT NOT NULL,
            type VARCHAR NOT NULL,
            creation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- 4. Create metro_metrics table
        CREATE TABLE IF NOT EXISTS
        metro_metrics (
            mid VARCHAR NOT NULL,
            year INTEGER NOT NULL,
            variable_code VARCHAR NOT NULL,
            value FLOAT,
            creation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (mid, year, variable_code),
            CONSTRAINT fk_metrics_metro FOREIGN KEY (mid) REFERENCES metros (mid) ON DELETE CASCADE ON UPDATE CASCADE,
            CONSTRAINT fk_metrics_variable FOREIGN KEY (variable_code) REFERENCES variables (variable_code) ON DELETE RESTRICT ON UPDATE CASCADE
        );


        -- 5. Create similar_metros junction table
        CREATE TABLE IF NOT EXISTS
        similar_metros (
            mid VARCHAR NOT NULL,
            similar_mid VARCHAR NOT NULL,
            creation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (mid, similar_mid),
            CONSTRAINT fk_similar_metro FOREIGN KEY (mid) REFERENCES metros (mid) ON DELETE CASCADE ON UPDATE CASCADE,
            CONSTRAINT fk_similar_metro_ref FOREIGN KEY (similar_mid) REFERENCES metros (mid) ON DELETE CASCADE ON UPDATE CASCADE
        );

    }
    
    Confirm table creation by going to Table Editor tab in Supabase dashboard.
    
    Then you can run this function to confirm tables were created and you can connect to supabase.
     
     """
    # Define the tables to confirm
    tables = ['metros', 'zipcodes', 'variables', 'metro_metrics', 'similar_metros']
    
    # Dictionary to store results
    results = {}
    
    try:
        for table in tables:
            # Perform a count query
            count_response = supabase.table(table).select('*', count='exact').execute()
            row_count = count_response.count if count_response.count is not None else 0
            
            # Retrieve the first row (if any)
            first_row_response = supabase.table(table).select('*').limit(1).execute()
            first_row = first_row_response.data[0] if first_row_response.data else None
            
            # Store the results
            results[table] = {
                'count': row_count,
                'first_row': first_row
            }
            
            # Print the results for each table
            print(f"Table '{table}': {row_count} rows.")
            print(f"First row of '{table}': {first_row}\n")
        
        return results

    except Exception as e:
        print("An error occurred while fetching table information:")
        print(e)
        return False


if __name__ == "__main__":
    # insert_into_tables()
    # confirm_tables_created()
    print(get_location("10001"))

