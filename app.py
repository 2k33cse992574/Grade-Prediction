from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

THRESHOLD = 0.4   # same threshold used in evaluation

@app.route("/")
def home():
    return "Student Grade Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Use DataFrame to preserve feature names (no warnings)
        X = pd.DataFrame([{
            "weekly_self_study_hours": float(data["weekly_self_study_hours"]),
            "attendance_percentage": float(data["attendance_percentage"]),
            "class_participation": float(data["class_participation"])
        }])
    except Exception:
        return jsonify({"error": "Invalid input format"}), 400

    # Apply same preprocessing as training
    X_scaled = scaler.transform(X)

    # Probability of class 1 (B)
    prob_B = model.predict_proba(X_scaled)[0][1]

    # Threshold-based prediction
    grade = "B" if prob_B >= THRESHOLD else "A"

    return jsonify({
        "grade": grade,
        "probability": round(float(prob_B), 3),
        "threshold": THRESHOLD
    })

if __name__ == "__main__":
    app.run(debug=True)
