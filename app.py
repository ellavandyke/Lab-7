import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import openpyxl
import xlrd
import numpy as np

#Step 1

url = "https://raw.githubusercontent.com/ellavandyke/Lab-7/refs/heads/main/AmesHousing.csv"

data = pd.read_csv(url)

# Display the first few rows
print(data.head())

# Select relevant features and target variable
features = ["Bedroom AbvGr", "Full Bath", "Lot Area", "Year Built"]
target = "SalePrice"

#Step 2

# Drop rows with missing values
data = data[features + [target]].dropna()

# Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: {mse}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: ${rmse:,.2f}")

#Step 3

# Streamlit Web App Interface
st.title("Ames Housing Price Predictor")
st.write("Enter property details to predict house price.")

# User input fields
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=4)
bathooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=2)
lot_area = st.number_input("Lot Area sf", min_value=1300, max_value=215245, value=100000)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)

# Predict price based on user input
input_data = pd.DataFrame([[bedrooms, bathrooms, lot_area, year_built]], columns=features)
predicted_price = model.predict(input_data)[0]

st.write(f"Predicted Sale Price: ${predicted_price:,.2f}")

