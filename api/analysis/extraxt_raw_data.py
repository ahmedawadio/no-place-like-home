import pandas as pd

"""
Extracting all US zip codes and connecting them to metropolitan/micropolitan areas(CBSA).

I cleaned the format and added it to the db.

"""

############################################################################################################
###  
# METROPOLITAN/MICROPOLITAN STATISTICAL AREAS (CBSA) 

# URL: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
# Title: Core based statistical areas (CBSAs), metropolitan divisions, and combined statistical areas (CSAs)
# Category: Jul. 2023 

###

#file path
cbsa_name_type = "api/analysis/raw_data/cbsa_name_type.xlsx"

# Read the Excel file into a DataFrame, keeping dtype as streing for leading zeros
cbsa_df = pd.read_excel(cbsa_name_type, header=0, dtype=str)

# Extract the specific columns I want to save
cbsa_cleaned_df = cbsa_df[['CBSA Code', 'CBSA Title', 'Metropolitan/Micropolitan Statistical Area']]

# Remove duplicates based on 'CBSA Code'
cbsa_cleaned_df = cbsa_cleaned_df.drop_duplicates(subset='CBSA Code')

final_cbsa_df = pd.DataFrame()
final_cbsa_df["cid"] = cbsa_cleaned_df["CBSA Code"]
final_cbsa_df["name"] = cbsa_cleaned_df["CBSA Title"]
# final_cbsa_df["type"] = cbsa_cleaned_df["Metropolitan/Micropolitan Statistical Area"]

final_cbsa_df.to_csv("api/analysis//data/cbsa.csv", index=False)
# Display the first few rows of the cleaned DataFrame to confirm it worked correctly
# print(final_cbsa_df.head())



############################################################################################################
###  ZIP CODES AND CITY,STATE ###


# (note: a free account is required to download the dataset)
# URL dataset: https://www.huduser.gov/apps/public/uspscrosswalk/home
# Category ZIP-CBSA snd Quarter 2024

# URL dataset Description:  https://www.huduser.gov/portal/datasets/usps_crosswalk.html
###

#file path
zip_city_cbsa = "api/analysis/raw_data/zip_city_cbsa.xlsx"

#  keeping dtype as streing for leading zeros
zip_city_cbsa_df = pd.read_excel(zip_city_cbsa, header=0,dtype=str)

# Extract the specific columns I want to save
zip_city_cbsa_df_cleaned = zip_city_cbsa_df[['ZIP', 'CBSA','USPS_ZIP_PREF_CITY', 'USPS_ZIP_PREF_STATE']]

zip_city_cbsa_df_cleaned = zip_city_cbsa_df_cleaned.drop_duplicates(subset='ZIP')

#Puerto Rico is not included in model
zip_city_cbsa_df_cleaned = zip_city_cbsa_df_cleaned[zip_city_cbsa_df_cleaned['USPS_ZIP_PREF_STATE'] != 'PR']

#99999 is a false zipcode and is often used as placeholder, I belive similar is true for CBSA Any row with this value is removed
zip_city_cbsa_df_cleaned = zip_city_cbsa_df_cleaned[zip_city_cbsa_df_cleaned['CBSA'] != '99999']

# Cleaning text formating. Making copy to avoid returning-a-view-versus-a-copy warning
zip_city_cbsa_df_cleaned = zip_city_cbsa_df_cleaned.copy()
zip_city_cbsa_df_cleaned['USPS_ZIP_PREF_CITY'] = zip_city_cbsa_df['USPS_ZIP_PREF_CITY'].str.title().copy()


final_zipcode_df = pd.DataFrame()
final_zipcode_df["zipcode"] = zip_city_cbsa_df_cleaned["ZIP"]
final_zipcode_df["city"] = zip_city_cbsa_df_cleaned["USPS_ZIP_PREF_CITY"]
final_zipcode_df["state"] = zip_city_cbsa_df_cleaned["USPS_ZIP_PREF_STATE"]
final_zipcode_df["mid"] = zip_city_cbsa_df_cleaned["CBSA"]


final_zipcode_df.to_csv("api/analysis/data/zipcodes.csv", index=False)
# print(final_zipcode_df.head())
