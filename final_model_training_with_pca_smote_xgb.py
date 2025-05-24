import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from joblib import dump, load
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import set_config
from joblib import Memory

# --------------- Load data ---------------
x_train = pd.read_csv("x_train_unb.csv")
x_test = pd.read_csv("x_test_unb.csv")
y_train = pd.read_csv("y_train_unb.csv").squeeze()
y_test = pd.read_csv("y_test_unb.csv").squeeze()

# --------------- Build pipeline ---------------
# SMOTE in pipeline requires imblearn's Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

def evaluate_model(config_name, y_true, y_pred):
    print(f"Configuration: {config_name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print('-' * 60)

pipeline = ImbPipeline(steps=[
    ("pca", PCA(n_components=5, random_state=42)),
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        # scale_pos_weight is not needed when SMOTE is used
    ))
])

# --------------- Train final model ---------------
print("Training final model (PCA=5, XGBoost + SMOTE)…")
pipeline.fit(x_train, y_train)

# --------------- Save model ---------------
dump(pipeline, "final_model_pca5_xgb_smote.joblib")
print("Model saved as final_model_pca5_xgb_smote.joblib")

# Load model
train_pipeline = load("final_model_pca5_xgb_smote.joblib")

# Remove 'smote' step for inference: keep only PCA → classifier
infer_pipeline = Pipeline([
    ("pca", train_pipeline.named_steps["pca"]),
    ("clf", train_pipeline.named_steps["clf"])
])

# Predict on test data
y_pred = infer_pipeline.predict(x_test)
print(y_pred)
evaluate_model("PCA, SMOTE, XGBOOST", y_test, y_pred)
