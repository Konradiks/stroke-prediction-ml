from joblib import load
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


MODEL_PATH = "models/stroke_prediction_model_pca8_gb_smote.joblib"
SCALER_PATH = "scalers/scaler_selected_columns.joblib"

EXAMPLE_DATA = {
    "gender": "Male",
    "age": 45.0,
    "hypertension": "No",
    "heartdisease": "No",
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 85.6,
    "bmi": 23.5,
    "smoking_status": "Never smoked",
    "result": 1
}

def runStrokePrediction (
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
    """
    Predicts the likelihood of a stroke based on provided health and demographic features.

    Parameters
    ----------
    gender : str
        Gender of the individual.
        Expected values: ['Male', 'Female']

    age : float
        Age of the individual.
        Expected range: 0 to 110, step: 1

    hypertension : str
        Indicates whether the individual has hypertension.
        Expected values: ['No', 'Yes']

    heartdisease : str
        Indicates whether the individual has heart disease.
        Expected values: ['No', 'Yes']

    ever_married : str
        Marital status of the individual.
        Expected values: ['Yes', 'No']

    work_type : str
        Type of work the individual performs.
        Expected values: ['Private', 'Self-employed', 'Govt_job', 'children']

    Residence_type : str
        Type of residence.
        Expected values: ['Urban', 'Rural']

    avg_glucose_level : float
        Average glucose level of the individual.
        Expected range: approximately 50 to 280, step 0.01

    bmi : float
        Body Mass Index of the individual.
        Expected range: approximately 14 to 50, step 0.1

    smoking_status : str
        Smoking status of the individual.
        Expected values: ['Formerly smoked', 'Never smoked', 'Smokes', 'Unknown']

    result : int, optional (default=0)
        Format of the returned prediction:
        - 0: Returns a plain string with percentage (e.g., "6.12%")
        - 1: Returns a full description (e.g., "Probability for stroke: 6.12%")

    Returns
    -------
    str
        Prediction result indicating stroke likelihood, formatted according to `result` parameter.
    """
    processed_data = preprocess_input(
        gender, age, hypertension, heartdisease, ever_married,
        work_type, Residence_type, avg_glucose_level, bmi,
        smoking_status)

    if __name__ == "__main__":
        print("Data after processed:")
        print(processed_data.to_string())

    prediction = runModel(processed_data)

    if __name__ == "__main__":
        print("raw prediction:")
        print(prediction)
        print("\n")

    if result == 0:
        return str(np.round(prediction[0][1] * 100, 2), "%")
    else:
        return str("Propability for stroke: " + str(np.round(prediction[0][1] * 100, 2)) + "%")



def preprocess_input(gender, age, hypertension, heartdisease, ever_married,
                     work_type, Residence_type, avg_glucose_level, bmi,
                     smoking_status):

    gender = 0 if gender == 'Female' else 1
    # age is int
    hypertension = 0 if hypertension == 'No' else 1
    heartdisease = 0 if heartdisease == 'No' else 1
    ever_married = 0 if ever_married == 'No' else 1

    private = 1 if work_type == 'Private' else 0
    self_employed = 1 if work_type == 'Self-employed' else 0
    govt_job = 1 if work_type == 'Govt_job' else 0
    children = 1 if work_type == 'Children' else 0

    # avg_glucose_level is float
    # bmi is float

    # smoking_status
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
        # only one work_type
        'Private': private,
        'Self-employed': self_employed,
        'Govt_job': govt_job,
        'children': children,
        # only one smoking_status
        'formerly smoked': formerly_smoked,
        'never smoked': never_smoked,
        'smokes': smokes,
        'Unknown': Unknown
    }, index=[0])

def runModel (df):

    #MODEL_PATH = "./models/stroke_prediction_model_pca8_gb_smote.joblib"
    #SCALER_PATH = "./scalers/scaler_selected_columns.joblib"
    model = MODEL_PATH

    scaler = load(SCALER_PATH )
    columns_to_scale = [
        'age', 'avg_glucose_level', 'bmi'
    ]
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    train_pipeline = load(model)

    infer_pipeline = Pipeline([
        ("pca", train_pipeline.named_steps["pca"]),
        ("clf", train_pipeline.named_steps["clf"])
    ])

    return infer_pipeline.predict_proba(df)



# --- Example Usage ---
if __name__ == "__main__":

    print(runStrokePrediction(**EXAMPLE_DATA))

