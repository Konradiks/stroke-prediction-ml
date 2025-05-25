import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn import set_config
from joblib import Memory

# ------------------ Memory Configuration ------------------
print("Configuring memory...")
set_config(working_memory=12000)
memory = Memory(location='cachedir', verbose=0)
memory.clear(warn=False)

def load_data():
    print("Loading data...")
    x_train = pd.read_csv("x_train_unb.csv")
    x_test = pd.read_csv("x_test_unb.csv")
    y_train = pd.read_csv("y_train_unb.csv").squeeze()
    y_test = pd.read_csv("y_test_unb.csv").squeeze()
    return x_train, x_test, y_train, y_test

def evaluate_model(config_name, y_true, y_pred):
    print(f"Configuration: {config_name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print('-' * 60)

def build_training_pipeline():
    return ImbPipeline(steps=[
        ("pca", PCA(n_components=8, random_state=42)),
        ("smote", SMOTE(random_state=42)),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

def build_inference_pipeline(trained_pipeline):
    return Pipeline([
        ("pca", trained_pipeline.named_steps["pca"]),
        ("clf", trained_pipeline.named_steps["clf"])
    ])
