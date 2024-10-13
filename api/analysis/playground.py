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
