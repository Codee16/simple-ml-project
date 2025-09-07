import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")

# App title
st.title("Iris Flower Predictor 🌸")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_class = model.predict(features)[0]
    class_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"Predicted Iris Species: {class_map[pred_class]}")
