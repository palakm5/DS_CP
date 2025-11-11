import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os



DATA_PATH = "data/Crop_recommendation.csv"
SCALER_PATH = "models/scaler.pkl"
PROCESSED_FOLDER = "data/processed"

def load_data():
    
    df = pd.read_csv(DATA_PATH)
    print(" Data loaded successfully!")
    print(f"Shape of dataset: {df.shape}")
    print("Columns:", df.columns.tolist())
    return df

def preprocess_data(df):
    
    
   
   
    df = df.fillna(df.mean()) 
    print(" Missing values handled.")

    

    X = df.drop('label', axis=1)
    y = df['label']

   

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled.")

    
    
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    

    joblib.dump(scaler, SCALER_PATH)

   

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(" Data split into train and test sets.")

    

    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    y_train_df = pd.DataFrame(y_train, columns=['label'])
    y_test_df = pd.DataFrame(y_test, columns=['label'])



   
    train_df = pd.concat([X_train_df, y_train_df], axis=1)
    test_df = pd.concat([X_test_df, y_test_df], axis=1)

 

    train_df.to_csv(f"{PROCESSED_FOLDER}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_FOLDER}/test.csv", index=False)
    print(f"Preprocessed train and test CSVs saved in '{PROCESSED_FOLDER}'.")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
