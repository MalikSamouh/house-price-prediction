import argparse
import numpy as np
import pandas as pd
import joblib

# Load Model & Scaler
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load feature names used during training
feature_names = joblib.load("models/feature_names.pkl")  # Save this in train.py

# CLI Argument Parsing
parser = argparse.ArgumentParser(description="Predict House Price")
parser.add_argument("--bedrooms", type=int, required=True, help="Number of bedrooms")
parser.add_argument("--bathrooms", type=int, required=True, help="Number of bathrooms")
parser.add_argument("--sqft", type=int, required=True, help="Total square footage")

args = parser.parse_args()

# Prepare Input Data as DataFrame
input_data = pd.DataFrame([{
    "BedroomAbvGr": args.bedrooms,
    "FullBath": args.bathrooms,
    "GrLivArea": args.sqft
}])

# Ensure input has the same columns as the training set
for col in feature_names:
    if col not in input_data:
        input_data[col] = 0  # Add missing features

# Reorder columns to match training set
input_data = input_data[feature_names]

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make Prediction
predicted_price = model.predict(input_scaled)

print(f"Predicted House Price: ${predicted_price[0]:,.2f}")