from flask import Blueprint, request, jsonify
from app.utils import runStrokePrediction

main = Blueprint('main', __name__)

@main.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required_fields = [
        "gender", "age", "hypertension", "heartdisease", "ever_married",
        "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
    ]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        result = runStrokePrediction(
            gender=data["gender"],
            age=float(data["age"]),
            hypertension=data["hypertension"],
            heartdisease=data["heartdisease"],
            ever_married=data["ever_married"],
            work_type=data["work_type"],
            Residence_type=data["Residence_type"],
            avg_glucose_level=float(data["avg_glucose_level"]),
            bmi=float(data["bmi"]),
            smoking_status=data["smoking_status"],
            result=0
        )
        return jsonify({"probability": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500