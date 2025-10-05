import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# --- Configuration ---
INPUT_FILE_NAME = 'tempo_no2_sample.nc' 
INPUT_FILE_PATH = f'../data_raw/{INPUT_FILE_NAME}'
OUTPUT_FILE_PATH = '../data_processed/tempo_processed.csv'

# --- TEMPO Variable Paths (✅ updated based on your file structure) ---
NO2_VARIABLE_PATH = '/product/vertical_column_troposphere' 
LATITUDE_VARIABLE_PATH = '/geolocation/latitude' 
LONGITUDE_VARIABLE_PATH = '/geolocation/longitude' 
QUALITY_FLAG_PATH = '/product/main_data_quality_flag'  # optional

# ----------------------------------------------------

def read_tempo_data():
    """
    Reads a TEMPO satellite data file (.nc), extracts NO2, latitude, and longitude,
    flattens the arrays, and saves them as a CSV file.
    """
    print(f"--- Starting TEMPO Data Reader ---")
    print(f"Attempting to read file: {INPUT_FILE_PATH}")
    
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"❌ ERROR: File not found at '{INPUT_FILE_PATH}'.")
        print("Please make sure your TEMPO file is named 'tempo_no2_sample.nc' and is in the '../data_raw' folder.")
        return

    try:
        with Dataset(INPUT_FILE_PATH, 'r') as nc_file:
            print("\n--- FILE STRUCTURE EXPLORATION (for reference) ---")
            print(nc_file)
            print("--------------------------------------------------\n")

            # ✅ Read variables
            no2_data = nc_file[NO2_VARIABLE_PATH][:]
            lat_data = nc_file[LATITUDE_VARIABLE_PATH][:]
            lon_data = nc_file[LONGITUDE_VARIABLE_PATH][:]
            
            # Optional: read data quality flag if present
            quality_flag = None
            if QUALITY_FLAG_PATH in nc_file.variables:
                quality_flag = nc_file[QUALITY_FLAG_PATH][:]
                print("Quality flag data found and loaded.")
            
            print(f"NO2 data shape: {no2_data.shape}")
            print(f"Latitude data shape: {lat_data.shape}")
            print(f"Longitude data shape: {lon_data.shape}")

            # ✅ Flatten arrays
            no2_flat = no2_data.flatten()
            lat_flat = lat_data.flatten()
            lon_flat = lon_data.flatten()

            if quality_flag is not None:
                quality_flat = quality_flag.flatten()
            else:
                quality_flat = np.full_like(no2_flat, np.nan)

            # ✅ Handle fill/missing values
            FILL_VALUE = -9999.0
            no2_flat = np.where(no2_flat == FILL_VALUE, np.nan, no2_flat)

            # ✅ Create DataFrame
            tempo_df = pd.DataFrame({
                'latitude': lat_flat,
                'longitude': lon_flat,
                'NO2_column_density': no2_flat,
                'quality_flag': quality_flat
            })

            # Remove missing NO2 values
            tempo_df = tempo_df.dropna(subset=['NO2_column_density'])

            # ✅ Ensure output folder exists
            os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

            # ✅ Save CSV
            tempo_df.to_csv(OUTPUT_FILE_PATH, index=False)
            print(f"\n✅ SUCCESS: Processed {len(tempo_df)} valid data points.")
            print(f"Processed data saved to: {OUTPUT_FILE_PATH}")

    except KeyError as ke:
        print(f"\n❌ ERROR: Missing variable {ke}")
        print("Check the printed file structure and update the variable paths accordingly.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    read_tempo_data()
