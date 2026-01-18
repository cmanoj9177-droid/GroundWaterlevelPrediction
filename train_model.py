# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# Load your dataset
df = pd.read_csv("C:/Users/uday0/OneDrive/Desktop/Project-02/final_groundwater_dataset (3).csv")

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["Soil Type", "Land Use Type", "Area Name"])

# Features and target
X = df.drop(columns=["Water Table Depth (m)", "Well_ID", "Date"])
y = df["Water Table Depth (m)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Test MSE: {mse}")

# Save model
joblib.dump(model, "groundwater_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
