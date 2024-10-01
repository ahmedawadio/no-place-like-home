from .keys import SUPABASE_URL,SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client, Client

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
response = supabase.table("example").select("*").execute()

def get_location():
    # Check if data exists in the response
    if response.data and len(response.data) > 0:
        # Extract the first row and the 'location' column value
        first_row = response.data[0]  # Get the first row
        location = first_row.get('location')  # Get the 'location' column value
        return location
    return "Not found"

