import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# File paths
DATA_PATH = "data/Crop_recommendation.csv"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
PROCESSED_FOLDER = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_FOLDER, "train.csv")
TEST_PATH = os.path.join(PROCESSED_FOLDER, "test.csv")

def load_data():
    """Load the raw dataset"""
    df = pd.read_csv(DATA_PATH)
    print("✅ Data loaded successfully!")
    print(f"Shape of dataset: {df.shape}")
    print("Columns:", df.columns.tolist())
    return df

def preprocess_data(df):
    """Clean, encode, scale, split, and save processed train/test datasets"""
    
    # Handle missing values for numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print("✅ Missing values handled.")

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Encode crop labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("✅ Labels encoded.")

    # Scale feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Features scaled.")

    # Make required directories
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save the scaler and encoder
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print("✅ Scaler and LabelEncoder saved.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print("✅ Data split into train and test sets.")

    # Convert arrays back to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    y_train_df = pd.DataFrame(y_train, columns=['label'])
    y_test_df = pd.DataFrame(y_test, columns=['label'])

    # Merge X and y to save as CSV
    train_df = pd.concat([X_train_df, y_train_df], axis=1)
    test_df = pd.concat([X_test_df, y_test_df], axis=1)

    # Save CSV files in data/processed/
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"✅ train.csv and test.csv saved in '{PROCESSED_FOLDER}'.")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    preprocess_data(df)
