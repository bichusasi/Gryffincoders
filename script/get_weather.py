import pandas as pd
import requests
import time
from datetime import datetime
import numpy as np

# --- Configuration ---
API_KEY = "41c16e9e79998396f10a2818107fa38d"
CURRENT_URL = "http://api.openweathermap.org/data/2.5/weather" 
INPUT_FILE = '../data_processed/tempo_processed.csv'
OUTPUT_FILE = '../data_processed/weather_processed.csv'
SAMPLE_RATE = 5000  # Query every 10th data point to stay within API limits

# --- Data Preparation ---
try:
    # 1. Load the TEMPO data from Member A
    tempo_df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(tempo_df)} TEMPO data points.")
    
    # **NAN FIX: Drop rows where latitude or longitude are missing**
    initial_len = len(tempo_df)
    tempo_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    dropped_count = initial_len - len(tempo_df)
    print(f"Dropped {dropped_count} rows with missing coordinates (NaN).")
    
    # 2. Select only the necessary columns (lat/lon) and take a sample
    # Drop duplicates in case multiple rows have the same lat/lon (common for TEMPO data)
    query_list = tempo_df[['latitude', 'longitude']].iloc[::SAMPLE_RATE, :].drop_duplicates().reset_index(drop=True)
    print(f"Querying weather for {len(query_list)} unique sampled points (every {SAMPLE_RATE}th point).")
    
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE}. Make sure Member A's task is complete.")
    exit()

# --- API Fetching Loop ---
weather_results = []
queries_sent = 0

print("\nStarting weather data fetch...")

for index, row in query_list.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY,
        'units': 'metric' 
    }
    
    try:
        response = requests.get(CURRENT_URL, params=params)
        data = response.json()
        
        if response.status_code == 200:
            result = {
                'latitude': lat,
                'longitude': lon,
                'current_temp_C': data['main']['temp'],
                'current_wind_speed_m_s': data['wind']['speed'],
                'current_wind_direction_deg': data['wind']['deg'],
                'current_humidity_percent': data['main']['humidity'],
                'timestamp_queried_utc': datetime.utcnow().isoformat() # This line is fine for a prototype
            }
            weather_results.append(result)
            queries_sent += 1
            
        else:
            print(f"Error for ({lat:.2f}, {lon:.2f}): Status {response.status_code}, Message: {data.get('message', 'Unknown error')}")
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        
    time.sleep(1) # Pause to respect the API rate limit (important!)
    
print(f"\n--- Fetch Complete. Sent {queries_sent} successful queries. ---")

# --- Finalize and Save Data ---
if weather_results:
    weather_df = pd.DataFrame(weather_results)
    weather_df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: Weather data saved to: {OUTPUT_FILE}")
else:
    print("WARNING: No weather data was successfully collected.")