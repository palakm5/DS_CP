# src/predict.py
import numpy as np
import joblib
import os

# Define paths to model files
MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

# Load the trained model and preprocessing objects
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the best crop based on soil and climate parameters.
    Inputs:
        N, P, K, temperature, humidity, ph, rainfall — float or int values
    Returns:
        Predicted crop name (string)
    """
    # Prepare feature vector
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Scale input features
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Convert numeric label to actual crop name
    crop_name = label_encoder.inverse_transform(prediction)[0]

    return crop_name
