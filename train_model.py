import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('crop_data.csv')

# Features and target variable
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Input features
y = data['label']  # Target variable (crop type)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
