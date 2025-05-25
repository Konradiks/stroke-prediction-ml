from joblib import load
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

MODEL_PATH = "models/stroke_prediction_model_pca8_gb_smote.joblib"
SCALER_PATH = "scalers/scaler_selected_columns.joblib"

def runStrokePrediction(
    gender: str,
    age: float,
    hypertension: str,
    heartdisease: str,
    ever_married: str,
    work_type: str,
    Residence_type: str,
    avg_glucose_level: float,
    bmi: float,
    smoking_status: str,
    result: int = 0
) -> str:
    processed_data = preprocess_input(
        gender, age, hypertension, heartdisease, ever_married,
        work_type, Residence_type, avg_glucose_level, bmi,
        smoking_status)

    prediction = runModel(processed_data)

    if result == 0:
        return str(np.round(prediction[0][1] * 100, 2)) + "%"
    else:
        return "Probability for stroke: " + str(np.round(prediction[0][1] * 100, 2)) + "%"

def preprocess_input(gender, age, hypertension, heartdisease, ever_married,
                     work_type, Residence_type, avg_glucose_level, bmi,
                     smoking_status):

    gender = 0 if gender == 'Female' else 1
    hypertension = 0 if hypertension == 'No' else 1
    heartdisease = 0 if heartdisease == 'No' else 1
    ever_married = 0 if ever_married == 'No' else 1

    private = 1 if work_type == 'Private' else 0
    self_employed = 1 if work_type == 'Self-employed' else 0
    govt_job = 1 if work_type == 'Govt_job' else 0
    children = 1 if work_type == 'Children' else 0

    formerly_smoked = 1 if smoking_status == 'Formerly smoked' else 0
    never_smoked = 1 if smoking_status == "Never smoked" else 0
    smokes = 1 if smoking_status == 'Smokes' else 0
    Unknown = 1 if smoking_status == 'Unknown' else 0

    Residence_type = 0 if Residence_type == 'Rural' else 1

    return pd.DataFrame({
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heartdisease,
        'ever_married': ever_married,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'Private': private,
        'Self-employed': self_employed,
        'Govt_job': govt_job,
        'children': children,
        'formerly smoked': formerly_smoked,
        'never smoked': never_smoked,
        'smokes': smokes,
        'Unknown': Unknown
    }, index=[0])

def runModel(df):
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    columns_to_scale = ['age', 'avg_glucose_level', 'bmi']
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    infer_pipeline = Pipeline([
        ("pca", model.named_steps["pca"]),
        ("clf", model.named_steps["clf"])
    ])

    return infer_pipeline.predict_proba(df)