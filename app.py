from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]  # Expecting list of features
    prediction = model.predict([np.array(data)])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
