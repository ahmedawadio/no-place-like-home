from openai import OpenAI
import pandas as pd
import requests
import os
from keys import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)

def generate_metro_images(
    csv_filepath: str,
    output_dir: str,
    image_size: str = "1024x1024",
    model: str = "dall-e-3",
    quality: str = "standard"
):
    print("Reading CSV file...")

    # Read the first 3 rows of the CSV file
    try:
        df = pd.read_csv(csv_filepath)
        print("CSV file loaded successfully.")

        # Check if the required columns are present
        if 'name' in df.columns and 'mid' in df.columns:
            print("Iterating through each row (first 3 rows):")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            total_locations = len(df)

            # Loop through each row and generate images
            for index, row in df.iterrows():
                name = row['name']
                mid = row['mid']
                prompt = f"Make me a single pastel paintbrush-style depiction of the quintessential downtown of: {name}. Do not include text or a paintbrush."
                print(f"Row {index}/{total_locations}:, mid:{mid}, name: {name}")

                image_path = os.path.join(output_dir, f"{mid}.png")

                # Check if the image already exists, if so, skip
                if os.path.exists(image_path):
                    print(f"Image {image_path} already exists. Skipping...")
                    continue

                try:
                    # Generate image using OpenAI's DALL-E API
                    response = client.images.generate(
                        model=model,
                        prompt=prompt,
                        size=image_size,
                        quality=quality,
                        n=1,
                    )

                    # Extract image URL
                    image_url = response.data[0].url

                    # Download and save the image
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        # Construct file path
                        with open(image_path, 'wb') as img_file:
                            img_file.write(img_response.content)
                        print(f"Saved image to {image_path}")
                    else:
                        print(f"Failed to download image from {image_url}")

                except Exception as e:
                    print(f"Error generating image for row {index} with value '{name}': {e}")
        else:
            print("The specified columns 'name' or 'mid' are missing in the CSV file.")
            return

    except FileNotFoundError:
        print(f"File at {csv_filepath} not found.")
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")

# Example usage
if __name__ == "__main__":
    csv_filepath = "api/analysis/data/database/metros.csv"
    output_dir = "api/analysis/data/bucket"
    
    generate_metro_images(
        csv_filepath=csv_filepath,
        output_dir=output_dir
    )
