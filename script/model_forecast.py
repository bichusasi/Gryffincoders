# model_forecast.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle 
import os # <--- FIX: ADDED OS IMPORT

# --- Configuration ---
MASTER_FILE = '../data_processed/master_data_for_model.csv'
MODEL_FILE = '../model/air_quality_model.pkl'

# --- Simplified AQI Conversion Function ---
def pm25_to_aqi_level(pm25):
    """Converts predicted PM2.5 (µg/m³) to a simplified AQI health level."""
    if pm25 <= 12.0:
        return {"level": "Good", "color": "green"}
    elif pm25 <= 35.4:
        return {"level": "Moderate", "color": "yellow"}
    elif pm25 <= 55.4:
        return {"level": "Unhealthy for Sensitive Groups", "color": "orange"}
    elif pm25 <= 150.4:
        return {"level": "Unhealthy", "color": "red"}
    else:
        return {"level": "Hazardous", "color": "purple"}

def train_and_save_model():
    print("--- Starting Model Training ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(MASTER_FILE)
    except FileNotFoundError:
        print(f"ERROR: Master file not found at {MASTER_FILE}. Please run combine_data.py.")
        return None

    # 2. Define Features (X) and Target (Y)
    # Target: The ground truth PM2.5 from OpenAQ
    Y = df['nearest_openaq_pm25']
    
    # Features (Inputs): TEMPO pollution and Weather conditions
    X = df[[
        'NO2_column_density',             # Ensure this name matches your TEMPO file!
        'current_temp_C', 
        'current_wind_speed_m_s', 
        'current_wind_direction_deg'
    ]].copy()

    # Data Cleaning: Drop rows with NaN values in the features or target
    data_clean = pd.concat([X, Y], axis=1).dropna()
    X_clean = data_clean.drop('nearest_openaq_pm25', axis=1)
    Y_clean = data_clean['nearest_openaq_pm25']

    if len(X_clean) == 0:
        print("ERROR: No clean data available to train the model.")
        return None

    # 3. Train Model
    model = LinearRegression()
    model.fit(X_clean, Y_clean)

    # 4. Evaluate (Optional)
    y_pred = model.predict(X_clean)
    mse = mean_squared_error(Y_clean, y_pred)
    print(f"Model Trained! MSE on training data: {mse:.2f}")

    # 5. Save the Model (FIXED: os.makedirs added)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"SUCCESS: Model saved to {MODEL_FILE}")
    return model

if __name__ == "__main__":
    # Ensure the model is trained when the script is run directly
    trained_model = train_and_save_model()
    # If the script runs successfully, the model is ready!