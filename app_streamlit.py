import streamlit as st
import numpy as np
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "iris_model.pkl"

# Load or train model
if not os.path.exists(MODEL_PATH):
    st.info("No trained model found. Training a new one...")

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    st.success("Model trained and saved!")
else:
    model = joblib.load(MODEL_PATH)

# Iris target mapping
class_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# App title
st.title("🌸 Iris Flower Predictor")

st.markdown("Enter flower measurements below and click **Predict** to identify the Iris species.")

# Input fields with default example values
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1, value=1.4)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1, value=0.2)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_class = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    st.success(f"🌼 Predicted Iris Species: **{class_map[pred_class]}**")
    st.write(
        f"Confidence: Setosa {probs[0]:.2f}, Versicolor {probs[1]:.2f}, Virginica {probs[2]:.2f}"
    )
