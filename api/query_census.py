"""
In order to not consume all my free credits for my cloud functions, 
I will only query Census API locally(which can take a good amountof time), 
then populate the db. 

The cloud functions will only connect to the db and compute similarty scores 

"""

import os
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client

load_dotenv(dotenv_path=".env.local")

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(url, key)

response = supabase.table("example").select("*").execute()

print(response)

CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')

# print(url,key,CENSUS_API_KEY)


