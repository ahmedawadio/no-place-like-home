from keys import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
import os
from supabase import create_client, Client
from flask import Flask, send_file, jsonify
from io import BytesIO

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def upload_images_to_supabase(
    csv_filepath: str,
    image_dir: str,
    bucket_name: str = "metros"
):
    print("Reading CSV file to get list of mids...")

    try:
        # Read the CSV file
        df = pd.read_csv(csv_filepath)
        print("CSV file loaded successfully.")

        # Check if the required 'mid' column is present
        if 'mid' in df.columns:
            print("Iterating through each 'mid' to upload images:")

            # Loop through each row and get 'mid'
            for index, row in df.iterrows():
                mid = row['mid']
                image_path = os.path.join(image_dir, f"{mid}.webp")
                # hi
                path_on_supabase = f"{mid}.webp"
                
                # Check if the image file exists locally
                if os.path.isfile(image_path):
                    # List files in the bucket to check if file already exists
                    existing_files_response = supabase.storage.from_(bucket_name).list()
                    existing_file_names = [file['name'] for file in existing_files_response]

                    if path_on_supabase in existing_file_names:
                        print(f"File {path_on_supabase} already exists in bucket '{bucket_name}'. Skipping upload.")
                        continue

                    # Upload the image to Supabase Storage
                    with open(image_path, 'rb') as file:
                        print(f"Uploading {image_path} to Supabase bucket '{bucket_name}'...")
                        response = supabase.storage.from_(bucket_name).upload(
                            path=path_on_supabase,
                            file=file,
                            file_options={"content-type": "image/png"}
                        )

                        # Check if the response indicates an error
                        if hasattr(response, 'status_code') and response.status_code != 200:
                            print(f"Error uploading {image_path}: {response.json()}")
                        else:
                            print(f"Successfully uploaded {image_path} as {path_on_supabase} in bucket '{bucket_name}'.")
                else:
                    print(f"Image file not found for MID {mid}: {image_path}")

        else:
            print("The specified column 'mid' is missing in the CSV file.")
            return

    except FileNotFoundError:
        print(f"File at {csv_filepath} not found.")
    except Exception as e:
        print(f"An error occurred while processing the CSV file: {e}")



def get_image(mid: str):
    try:
        # Define the path in Supabase storage
        path_on_supabase = f"{mid}.webp"

        # Download the file from Supabase
        response = supabase.storage.from_("metros").download(path_on_supabase)

        # If no data is returned, assume the file was not found
        if response is None:
            return jsonify({"error": f"Image not found for MID {mid}"}), 404

        # Wrap the binary image data in a BytesIO object
        img_data = BytesIO(response)

        # Return the image as a file response with the appropriate mime type
        return send_file(img_data, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Example usage
if __name__ == "__main__":
    print("running")
    # import pandas as pd

    # csv_filepath = "api/analysis/data/database/metros.csv"
    # image_dir = "api/analysis/data/bucket"
    
    # upload_images_to_supabase(
    #     csv_filepath=csv_filepath,
    #     image_dir=image_dir
    # )
