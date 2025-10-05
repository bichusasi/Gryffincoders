# combine_data.py
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree 
import os

# --- File Paths ---
# Member A output
TEMPO_FILE = '../data_processed/tempo_processed.csv' 
# FIX: OpenAQ file is saved in '../data_raw' by your get_openaq.py script
OPENAQ_FILE = '../data_raw/openaq_data.csv' 
# Member B output
WEATHER_FILE = '../data_processed/weather_processed.csv' 
MASTER_FILE = '../data_processed/master_data_for_model.csv'

# =========================================================================
# --- Step 2: Load and Standardize Data ---
# =========================================================================
print("Starting Phase 2: Data Alignment and Merging...")

try:
    df_tempo = pd.read_csv(TEMPO_FILE)
    df_openaq = pd.read_csv(OPENAQ_FILE)
    df_weather = pd.read_csv(WEATHER_FILE)
    
    print("All input data files loaded successfully.")

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: One or more input files missing. Error: {e}")
    exit()

# Standardization and Cleanup
# TEMPO: Cleanup NaNs and standardize time
df_tempo.dropna(subset=['latitude', 'longitude'], inplace=True) 
df_tempo['datetime'] = pd.to_datetime(df_tempo['NO2_column_density'], errors='coerce') # NOTE: Assuming 'datetime' column is missing or needs better parsing from TEMPO data
df_tempo.dropna(subset=['datetime'], inplace=True) 

# OPENAQ: Cleanup NaNs, standardize time, and rename columns for clarity
df_openaq.dropna(subset=['latitude', 'longitude', 'pollutant_value'], inplace=True)
df_openaq.rename(columns={'pollutant_value': 'openaq_pm25', 'timestamp_utc': 'datetime'}, inplace=True)
df_openaq['datetime'] = pd.to_datetime(df_openaq['datetime'], utc=True, errors='coerce')

# WEATHER: Already cleaned and contains 51 unique points
df_weather.dropna(inplace=True) 

print(f"Cleaned Data Totals: TEMPO ({len(df_tempo)}), OpenAQ ({len(df_openaq)}), Weather ({len(df_weather)})")

# =========================================================================
# --- Step 3: Merge 1: TEMPO + Weather (Spatial Key Merge) ---
# =========================================================================

print("\nStarting Merge 1: TEMPO + Current Weather...")

# Use rounded coordinates to create a merge key
df_tempo['lat_lon_key'] = df_tempo['latitude'].round(2).astype(str) + '_' + df_tempo['longitude'].round(2).astype(str)
df_weather['lat_lon_key'] = df_weather['latitude'].round(2).astype(str) + '_' + df_weather['longitude'].round(2).astype(str)

# Left Merge: Keep all TEMPO rows and add corresponding weather data
df_master = pd.merge(df_tempo, df_weather, on='lat_lon_key', how='left', suffixes=('_tempo', '_weather'))

# Clean up temporary keys and redundant coordinate columns
df_master.drop(columns=['lat_lon_key', 'latitude_weather', 'longitude_weather', 'timestamp_queried_utc'], inplace=True)
df_master.rename(columns={'latitude_tempo': 'latitude', 'longitude_tempo': 'longitude'}, inplace=True)

print(f"Merge 1 (TEMPO + Weather) complete. Master Data Rows: {len(df_master)}")


# =========================================================================
# --- Step 4: Merge 2: Integrate Ground Data (OpenAQ) via Nearest Neighbor ---
# =========================================================================

print("Starting Merge 2: Nearest OpenAQ Neighbor Calculation...")

coords_master = df_master[['latitude', 'longitude']].values
coords_openaq = df_openaq[['latitude', 'longitude']].values

if len(coords_openaq) > 0:
    # Build a spatial index
    tree = BallTree(np.radians(coords_openaq), metric='haversine')
    
    # Query: Find the nearest OpenAQ station for every point in the master file
    distances, indices = tree.query(np.radians(coords_master), k=1)
    
    # Add the nearest OpenAQ PM2.5 value and its distance
    df_master['nearest_openaq_pm25'] = df_openaq.loc[indices[:, 0], 'openaq_pm25'].values
    df_master['distance_km_to_openaq'] = distances.flatten() * 6371  # Earth radius in km
else:
    print("WARNING: OpenAQ data is empty. Skipping Nearest Neighbor calculation.")
    df_master['nearest_openaq_pm25'] = np.nan
    df_master['distance_km_to_openaq'] = np.nan

print("Nearest OpenAQ neighbor calculation complete.")

# =========================================================================
# --- Step 5: Finalize and Hand Off (The Deliverable) ---
# =========================================================================

# Use the EXACT TEMPO column name from your read_tempo.py
TEMPO_NO2_COL = 'NO2_column_density'

final_columns = ['datetime', 'latitude', 'longitude', 
                 # TEMPO Pollution
                 TEMPO_NO2_COL, 
                 # Weather Data
                 'current_temp_C', 'current_wind_speed_m_s', 'current_wind_direction_deg', 
                 # OpenAQ Validation Data
                 'nearest_openaq_pm25', 'distance_km_to_openaq']

# Filter the master DF to only include the final columns 
final_master_df = df_master.reindex(columns=final_columns, fill_value=np.nan)

# Save the Master File
os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)
final_master_df.to_csv(MASTER_FILE, index=False)

print(f"\n=========================================================================")
print(f"SUCCESS! MEMBER B'S DELIVERABLE IS READY.")
print(f"File: {MASTER_FILE}")
print(f"Total Rows for Model: {len(final_master_df)}")
print(f"Hand off to Member C (The Forecast Engine)!")
print(f"=========================================================================")