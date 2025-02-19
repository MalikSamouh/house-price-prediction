import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("data/house_prices.csv")

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Numeric: Replace with median
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])  # Categorical: Replace with most frequent value (mode)


# One-Hot Encoding for Categorical Features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate Features & Target Variable
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save Model & Scaler
joblib.dump(model, "models/house_price_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save Feature Names for Prediction Script
joblib.dump(list(X.columns), "models/feature_names.pkl")
print("Model and Scaler saved successfully.")
