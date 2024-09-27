"""
In order to not consume all my free credits for my cloud functions, 
I will only query Census API locally(which can take a good amountof time), 
then populate the db. 

The cloud functions will only connect to the db and compute similarty scores 

"""

import os
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client


envs = [".env.local", ".env.production", ".env.preview"]

def load_env_files(env_files):
    """Attempts to load each .env file in the given list"""
    for env_file in env_files:
        # Check if the env file exists
        dotenv_path = find_dotenv(env_file)
        if dotenv_path:
            # print(f"Loading environment variables from {env_file}")
            load_dotenv(dotenv_path=dotenv_path)
            break  # Stop at the first successfully loaded .env file

# Attempt to load environment variables from the list of env files
load_env_files(envs)


SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
CENSUS_API_KEY: str = os.getenv('CENSUS_API_KEY')


