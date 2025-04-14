# Streamlit Dashboard for Housing Price Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("/Users/rushabarram/Desktop/big_data_projects/Housing.csv")

# Convert categorical columns to lowercase
df = df.applymap(lambda x: x.lower() if type(x) == str else x)

# Load or train the model (replace with joblib.load if using saved model)
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Define app title
st.title("🏠 Housing Price Prediction Dashboard")
st.write("Enter the details of the house below to predict its price.")

# Sidebar Inputs
area = st.sidebar.slider("Area (sqft)", 1000, 15000, 5000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
stories = st.sidebar.slider("Stories", 1, 4, 2)
parking = st.sidebar.slider("Parking (spots)", 0, 4, 1)

mainroad = st.sidebar.selectbox("Main Road Access", ["yes", "no"])
guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"])
basement = st.sidebar.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.sidebar.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Create input data frame
input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "parking": [parking],
    "mainroad_yes": [1 if mainroad == "yes" else 0],
    "guestroom_yes": [1 if guestroom == "yes" else 0],
    "basement_yes": [1 if basement == "yes" else 0],
    "hotwaterheating_yes": [1 if hotwaterheating == "yes" else 0],
    "airconditioning_yes": [1 if airconditioning == "yes" else 0],
    "prefarea_yes": [1 if prefarea == "yes" else 0],
    "furnishingstatus_semi-furnished": [1 if furnishingstatus == "semi-furnished" else 0],
    "furnishingstatus_unfurnished": [1 if furnishingstatus == "unfurnished" else 0],
})

# Predict and display result
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ₹{predicted_price:,.0f}")
