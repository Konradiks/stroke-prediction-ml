import pandas as pd

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the input dataframe according to the required format
    for the stroke prediction model. Invalid rows are removed.

    Parameters:
        df (pd.DataFrame): Input DataFrame with raw input data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with only valid rows.
    """
    valid_gender = {'Male', 'Female'}
    valid_ever_married = {'Yes', 'No'}
    valid_work_type = {'Private', 'Self-employed', 'Govt_job', 'children'}
    valid_residence_type = {'Urban', 'Rural'}
    valid_smoking_status = {'Formerly smoked', 'Never smoked', 'Smokes', 'Unknown'}
    valid_stroke_values = {0, 1}

    df_cleaned = df.copy()

    # Normalize capitalization (optional but robust)
    df_cleaned['gender'] = df_cleaned['gender'].str.title()
    df_cleaned['ever_married'] = df_cleaned['ever_married'].str.title()
    df_cleaned['work_type'] = df_cleaned['work_type'].str.strip()
    df_cleaned['Residence_type'] = df_cleaned['Residence_type'].str.title()
    df_cleaned['smoking_status'] = df_cleaned['smoking_status'].str.title()

    # Remove rows with missing required fields
    df_cleaned = df_cleaned.dropna(subset=[
        'age', 'avg_glucose_level', 'bmi', 'gender', 'hypertension',
        'heart_disease', 'ever_married', 'work_type',
        'Residence_type', 'smoking_status'
    ])

    # Apply all validation checks
    conditions = (
        df_cleaned['age'].between(0, 110) &
        df_cleaned['avg_glucose_level'].between(50, 280) &
        df_cleaned['bmi'].between(14, 50) &
        df_cleaned['hypertension'].isin([0, 1]) &
        df_cleaned['heart_disease'].isin([0, 1]) &
        df_cleaned['gender'].isin(valid_gender) &
        df_cleaned['ever_married'].isin(valid_ever_married) &
        df_cleaned['work_type'].isin(valid_work_type) &
        df_cleaned['Residence_type'].isin(valid_residence_type) &
        df_cleaned['smoking_status'].isin(valid_smoking_status) &
        df_cleaned['stroke'].isin(valid_stroke_values)
    )

    invalid_rows = df_cleaned[~conditions]
    if not invalid_rows.empty:
        print(f"Removed {len(invalid_rows)} invalid rows.")

    return df_cleaned[conditions].reset_index(drop=True)
