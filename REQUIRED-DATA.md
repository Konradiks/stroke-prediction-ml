# Required Input Data for Stroke Prediction Model

This model predicts the likelihood of a stroke based on a combination of medical, demographic, and lifestyle features. Below is a description of the accepted input types and value ranges.

---

## Numeric Inputs

| Feature                 | Type  | Expected Range       | Description                                     |
| ----------------------- | ----- | -------------------- | ----------------------------------------------- |
| **age**                 | float | 0–110 (step 1)       | Age of the individual in years.                 |
| **avg\_glucose\_level** | float | \~50–280 (step 0.01) | Average blood glucose level (mg/dL).            |
| **bmi**                 | float | \~14–50 (step 0.1)   | Body Mass Index, calculated from height/weight. |

---

## Binary Health Indicators

| Feature          | Type | Accepted Values | Description                                 |
| ---------------- | ---- | --------------- | ------------------------------------------- |
| **hypertension** | str  | `'Yes'`, `'No'` | Whether the person has high blood pressure. |
| **heartdisease** | str  | `'Yes'`, `'No'` | Whether the person has heart disease.       |

---

## Demographic & Lifestyle Features

| Feature             | Type | Accepted Values                                                | Description                               |
| ------------------- | ---- | -------------------------------------------------------------- | ----------------------------------------- |
| **gender**          | str  | `'Male'`, `'Female'`                                           | Biological sex of the person.             |
| **ever\_married**   | str  | `'Yes'`, `'No'`                                                | Whether the person has ever been married. |
| **work\_type**      | str  | `'Private'`, `'Self-employed'`, `'Govt_job'`, `'children'`     | Type of employment.                       |
| **Residence\_type** | str  | `'Urban'`, `'Rural'`                                           | Location of residence.                    |
| **smoking\_status** | str  | `'Formerly smoked'`, `'Never smoked'`, `'Smokes'`, `'Unknown'` | Smoking habits of the person.             |

---

## Output Format (optional)

| Parameter  | Type | Default | Description                                                                                                                  |
| ---------- | ---- | ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **result** | int  | 0       | Output format:<br>0 → simple percent string (e.g., `"6.12%"`)<br>1 → full sentence (e.g., `"Probability for stroke: 6.12%"`) |

---

## 🔗 Return to [README.md](README.md)
