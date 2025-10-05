# app_backend.py
import os
import pandas as pd
import pickle
from flask import Flask, jsonify, request, render_template
# NEW: Import CORS to allow cross-origin requests from the HTML file
from flask_cors import CORS 
from model_forecast import pm25_to_aqi_level
import numpy as np

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_FILE = os.path.join(ROOT_DIR, 'model', 'air_quality_model.pkl')
MASTER_FILE = os.path.join(ROOT_DIR, 'data_processed', 'master_data_for_model.csv')
template_dir = os.path.join(ROOT_DIR, 'templates')

app = Flask(__name__, template_folder=template_dir)
# FIX: Enable CORS for all routes (*) to allow the HTML file to connect
CORS(app) 

df_master = None
model = None

def load_resources():
    global model, df_master
    try:
        with open(MODEL_FILE, 'rb') as file:
            model = pickle.load(file)
        df_master = pd.read_csv(MASTER_FILE)
        print("Resources loaded: Model and Master Data Ready.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load resources. Path check:\nModel Path: {MODEL_FILE}\nMaster Data Path: {MASTER_FILE}\nError: {e}")
        model = None
        df_master = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """
    API endpoint to get the predicted PM2.5 and AQI level for a location.
    Accepts optional lat/lon query parameters.
    """
    if model is None or df_master is None:
        return jsonify({"error": "Server not ready (model/data missing). Check terminal logs."}), 500

    # Retrieve lat/lon from query parameters (index.html is sending these)
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    # --- Data Retrieval Logic ---
    if lat is not None and lon is not None:
        # Find nearest row in df_master based on passed coordinates
        if 'latitude' in df_master.columns and 'longitude' in df_master.columns:
            # Simple squared Euclidean distance for nearest neighbor in dataset
            df_master['distance'] = ((df_master['latitude'] - lat)**2 + (df_master['longitude'] - lon)**2)
            nearest_row = df_master.loc[df_master['distance'].idxmin()]
            
            # Extract features from the nearest data point
            features = nearest_row[[
                'NO2_column_density',
                'current_temp_C',
                'current_wind_speed_m_s',
                'current_wind_direction_deg'
            ]].values.reshape(1, -1)
            location_context = f"Nearest data to ({lat:.4f}, {lon:.4f})"
        else:
            return jsonify({"error": "Latitude/Longitude columns missing in master data."}), 500
    else:
        # Fallback to the very last data point if no coordinates are provided
        last_row = df_master.tail(1)
        features = last_row[[
            'NO2_column_density',
            'current_temp_C',
            'current_wind_speed_m_s',
            'current_wind_direction_deg'
        ]].values
        location_context = "Prediction based on the model's features from last data record."

    if features.size == 0 or np.any(np.isnan(features)):
        return jsonify({"error": "No valid data to generate prediction."}), 500

    # --- Prediction and AQI Conversion ---
    predicted_pm25 = model.predict(features)[0]
    predicted_pm25 = max(0, predicted_pm25) # Ensure PM2.5 is not negative
    aqi_data = pm25_to_aqi_level(predicted_pm25)

    return jsonify({
        "status": "success",
        "predicted_pm25": round(predicted_pm25, 2),
        "aqi_level": aqi_data["level"],
        "aqi_color": aqi_data["color"],
        "location_context": location_context
    })

@app.route('/api/validation', methods=['GET'])
def get_validation_data():
    """
    API endpoint to show recent data for validation/transparency.
    """
    if df_master is None:
        return jsonify({"error": "Server not ready (model/data missing)."}), 500

    # Get the last 10 rows for recent data comparison
    recent_data = df_master.tail(10).to_dict(orient='records')
    validation_list = []
    for row in recent_data:
        validation_list.append({
            # NOTE: datetime column name used here must match 'datetime' in df_master
            "timestamp": row.get('datetime', ''),
            "satellite_no2": round(row.get('NO2_column_density', 0), 4),
            "ground_pm25": round(row.get('nearest_openaq_pm25', 0), 2),
            "distance_km": round(row.get('distance_km_to_openaq', 0), 1)
        })

    return jsonify({
        "status": "success",
        "recent_validation_data": validation_list
    })

if __name__ == '__main__':
    load_resources()
    print("\n--- Flask Server Starting ---")
    app.run(debug=True)
