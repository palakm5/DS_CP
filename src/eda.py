import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


df = pd.read_csv("data/Crop_recommendation.csv")


os.makedirs("visuals", exist_ok=True)




numeric_df = df.drop('label', axis=1)
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("visuals/correlation_heatmap.png")
plt.show()




plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='label')
plt.title("Class Distribution of Crops")
plt.xticks(rotation=45)
plt.savefig("visuals/class_distribution.png")
plt.show()




key_features = ['N', 'P', 'K', 'temperature', 'humidity']

for col in key_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='label', y=col, data=df)
    plt.title(f"{col} vs Crop")
    plt.xticks(rotation=45)
    plt.savefig(f"visuals/{col}_boxplot.png")
    plt.show()
