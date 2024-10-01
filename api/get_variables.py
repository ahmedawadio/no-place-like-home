import requests
from bs4 import BeautifulSoup

# Function to scrape the census variables page and extract variable names and labels
def get_census_variables_for_1_year(year):
    # The URL of the page with the variables table

    if year%5 == 0:
        url = f"https://api.census.gov/data/{year}/acs/acs5/profile/variables.html"
    else:
        url = f"https://api.census.gov/data/{year}/acs/acs1/profile/variables.html"
    try:
        # Send a request to get the page content
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the table rows (<tr> tags) in the HTML
        rows = soup.find_all("tr")

        # Dictionary to store the variable names and labels
        variable_dict = {}

        # Loop through each row in the table
        for row in rows:
            # Find all columns (<td> tags) in the row
            cols = row.find_all("td")

            # If the row has the required number of columns
            if len(cols) >= 2:
                # Extract the variable name (from the first column)
                variable_name = cols[0].text.strip()

                # Extract the label/description (from the second column)
                label = cols[1].text.strip()

                # Add the variable name and label to the dictionary
                variable_dict[variable_name] = label

        # Filter the dictionary to only include keys that contain underscores because those are the non geographic variable names (ie, state names. geography ids, state numbers)
        variable_dict = {k: v for k, v in variable_dict.items() if "_" in k}
        
        return variable_dict

    except requests.exceptions.RequestException as e:
        print(f"Error fetching variables for {year}: {e}")
        return {}


# Function to find exact matches between multiple dictionaries
def find_exact_matches(*dicts):
    # Start with the keys of the first dictionary as the initial set of overlapping keys
    overlapping_keys = set(dicts[0].keys())

    # Find the intersection of keys across all dictionaries
    for dictionary in dicts[1:]:
        overlapping_keys &= dictionary.keys()

    # Create a new dictionary with only the overlapping keys and matching values across all dictionaries
    exact_matches = {
        key: dicts[0][key] for key in overlapping_keys
        if all(dictionary[key] == dicts[0][key] for dictionary in dicts)
    }

    return exact_matches


# Main function to get census variables for multiple years
def get_census_variables(years: list):
    # List to store the extracted variables and labels from each year
    variable_data = []

    for year in years:
        # Get the variables and labels for the specified year
        variables = get_census_variables_for_1_year(year)

        # If data was retrieved, add it to the list
        if variables:
            variable_data.append(variables)

    if len(variable_data) == 1:
        return variable_data[0]
        
    elif  len(variable_data) == 0:
        return {}

    # Find overlapping variables across all years
    overlapping_variables_across_years = find_exact_matches(*variable_data)

    return overlapping_variables_across_years





if __name__ == "__main__":

    print(len(get_census_variables(list(range(2019, 2020)))))

    print(get_census_variables(list(range(2019, 2020))))
