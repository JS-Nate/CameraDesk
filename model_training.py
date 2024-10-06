import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from feature_extraction import extract_features
import joblib

# Load and prepare your data
def load_data():
    data = []
    labels = []
    with open('data/labels.csv', 'r') as f:
        for line in f:
            path, label = line.strip().split(',')
            features = extract_features(path)
            data.append(features)
            labels.append(float(label))
    return np.array(data), np.array(labels)

# Train the model
def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/model.pkl')
    print('Model trained and saved.')

if __name__ == "__main__":
    train_model()
