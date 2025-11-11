# src/model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# File paths
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/random_forest_model.pkl"
REPORT_PATH = "reports/model_results.txt"
CONFUSION_MATRIX_PATH = "visuals/confusion_matrix.png"

def load_data():
    """Load the preprocessed train and test datasets"""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Missing values before cleaning:")
    print(train_df.isnull().sum())

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train a Random Forest model and evaluate it"""
    print("🚀 Training Random Forest model...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Save report
    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save confusion matrix as a heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs("visuals", exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.show()

    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    train_and_evaluate_model(X_train, X_test, y_train, y_test)
