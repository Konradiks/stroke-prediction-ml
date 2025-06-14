# 🧠 Stroke Prediction ML

Machine learning project for predicting the risk of stroke based on patient health and lifestyle factors.

# 📁 Project Structure
```bash
StrokePredictionML/
├── datasets/     # Dataset files for training and testing the model
├── models/       # Directory for storing trained machine learning models
├── notebooks/    # Jupyter notebooks for data analysis and model development
├── scalers/      # Serialized scalers used for preprocessing model input
└── src/          # Source files with functions and scripts used in notebooks
```

---

## 📦 What You Need to Run
To use the prediction model, make sure you have the following files and folders in your working directory:

✅ "models/" folder (with trained ML model)

✅ "scalers/" folder (with preprocessing scalers)

✅ "runStrokePredictionML.py" file

These are required for the runStrokePrediction() function to work correctly.

---

## 📋 Required Input Format

Detailed description of required data types, formats, and accepted values can be found in the file:  
👉 [REQUIRED-DATA.md](REQUIRED-DATA.md)

---

## 🚀 How to Use the Model

To run a prediction using the trained model, use the `runStrokePrediction()` function from `runStrokePredictionML.py`.

---

### ⚙️ Requirements to Run the Model

To run the model using `runStrokePrediction()`, make sure the following Python packages are installed:

```python
from joblib import load
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
```

You can install them with requirements file:

```bash
pip install -r requirements.txt
```

Or individually, if needed:

```bash
pip install joblib scikit-learn pandas numpy
```


### Example usage:

```python
from runStrokePredictionML import runStrokePrediction

result = runStrokePrediction(
    gender="Male",
    age=45.0,
    hypertension="No",
    heartdisease="No",
    ever_married="Yes",
    work_type="Private",
    Residence_type="Urban",
    avg_glucose_level=85.6,
    bmi=23.5,
    smoking_status="Never smoked",
    result=1  # Optional: 0 = plain %, 1 = full description
)

print(result)
````

### Example output:

```
Probability for stroke: 6.12%
```

---

### 💡 Notes:

* The model expects specific categorical values for string inputs (see [`REQUIRED-DATA.md`](REQUIRED-DATA.md)).
* The model uses a Gradient Boosting Classifier trained on PCA-reduced and SMOTE-balanced data.
* Input features are automatically preprocessed (e.g., encoded and scaled) inside the function.
* The result is the model’s predicted probability that the patient will suffer a stroke.

---


## 📖 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

---

## 📚 Data Source

[`Frontend repository`](https://github.com/Spoky03/stroke-prediction-webapp)
The dataset used for training and testing the model is sourced from:

[`Brain Stroke Dataset on Kaggle`](https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset)

---
## 🧑‍💻 Autorzy

* PowerBI: [NataliaPolak13](https://github.com/NataliaPolak13)
* Model Training and Optimization: [Konradiks](https://github.com/Konradiks)
* Web Application: [Spoky03](https://github.com/Spoky03)
