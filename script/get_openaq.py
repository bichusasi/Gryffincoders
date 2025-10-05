import requests
import pandas as pd
from datetime import datetime, timedelta, timezone 
import os
import json 

# --- Configuration ---

# 🛑 YOUR API KEY IS HERE 🛑
OPENAQ_API_KEY = "51f8f32c204d0346990189b80b62e0d456ae27eeb658bc9879e691c4ec45b2b2" 

# FIX: CHANGED ENDPOINT TO GET LATEST MEASUREMENTS DIRECTLY
# BASE_URL now includes the parameter ID in the path
BASE_URL_TEMPLATE = "https://api.openaq.org/v3/parameters/{parameter_id}/latest" 

# Define the geographic region and pollutants we are interested in.
# Testing with Delhi again to ensure data is found.
TARGET_COORDINATES = "28.7041,77.1025" 
SEARCH_RADIUS_METERS = 25000  
TARGET_PARAMETER = 2 # Parameter ID 2 corresponds to PM2.5 

# Calculate date ranges (optional for this specific V3 endpoint, but good practice)
DATE_TO = datetime.now(timezone.utc) 
DATE_FROM = DATE_TO - timedelta(days=7)

# Parameters to send to the API (using coordinates and radius for filtering)
PARAMS = {
    'coordinates': TARGET_COORDINATES,
    'radius': SEARCH_RADIUS_METERS,
    'limit': 1000, 
    'order_by': 'id',
    'sort_order': 'asc'
}

# Headers for API Key
HEADERS = {
    "X-API-Key": OPENAQ_API_KEY
}

OUTPUT_FILE_PATH = '../data_raw/openaq_data.csv'

# --- Main Logic ---

def fetch_openaq_data():
    """
    Fetches the latest PM2.5 measurements from the OpenAQ API within a specific radius.
    """
    # Construct the final API URL using the template and the parameter ID
    final_url = BASE_URL_TEMPLATE.format(parameter_id=TARGET_PARAMETER)
    
    print(f"--- Starting OpenAQ Data Fetch ---")
    print(f"Querying for PM2.5 data using endpoint: {final_url} within {SEARCH_RADIUS_METERS/1000} km.")

    try:
        # Make the API request
        response = requests.get(final_url, params=PARAMS, headers=HEADERS)

        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            # The 'latest' endpoint results are structured simpler
            latest_readings = data.get('results', [])

            if not latest_readings:
                print("\nNo latest measurements found for the specified query. Try changing coordinates or increasing the radius.")
                return
            
            # --- Data Processing: Extracting values from the simpler structure ---
            flat_data = []
            for reading in latest_readings:
                
                # Extract coordinates and value/time directly from the reading object
                coordinates = reading.get('coordinates', {})
                parameter = reading.get('parameter', {})
                
                latitude = coordinates.get('latitude')
                longitude = coordinates.get('longitude')
                pollutant_value = reading.get('value')
                
                # Ensure we have all necessary components before flattening
                if latitude is not None and longitude is not None and pollutant_value is not None:
                    
                    flat_data.append({
                        'timestamp_utc': reading.get('datetime'),
                        'pollutant_value': pollutant_value,
                        'pollutant_unit': parameter.get('units'),
                        'latitude': latitude,
                        'longitude': longitude,
                        'location_name': reading.get('location'),
                        'parameter_name': parameter.get('displayName')
                    })

            df = pd.DataFrame(flat_data)
            
            # Since we only append records where all three critical values are present, 
            # we can skip the final `df.dropna` filtering step.
            
            # --- Directory Check (kept) ---
            directory = os.path.dirname(OUTPUT_FILE_PATH)
            if directory:
                os.makedirs(directory, exist_ok=True)
            # --- END Directory Check ---

            # Save the processed data
            df.to_csv(OUTPUT_FILE_PATH, index=False)
            
            # Report the final success count
            if len(df) > 0:
                print(f"\nSUCCESS: Downloaded and processed {len(df)} latest measurements from local stations.")
                print(f"Data saved to: {OUTPUT_FILE_PATH}")
                print(f"Columns: {list(df.columns)}")
            else:
                print(f"\nSTATUS: Downloaded 0 measurements after successful API call. No file saved.")
            
        else:
            # Error handling for HTTP status codes
            print(f"ERROR: Failed to fetch data. HTTP Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            try:
                print("Response JSON:")
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError:
                print("Raw Response Text (not JSON):")
                print(response.text)


    except requests.exceptions.RequestException as e:
        print(f"A connection error occurred: {e}")

if __name__ == "__main__":
    fetch_openaq_data()