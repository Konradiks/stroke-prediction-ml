import pandas as pd
import numpy as np
from sklearn import set_config
from joblib import Memory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# ------------------------- Configuration -------------------------
print('Configuring memory...')
set_config(working_memory=12000)
memory = Memory(location='cachedir', verbose=0)
memory.clear(warn=False)

# ------------------------- Data -------------------------
print('Loading data...')
x_train = pd.read_csv("x_train_unb.csv")
x_test = pd.read_csv("x_test_unb.csv")
y_train = pd.read_csv("y_train_unb.csv").squeeze()
y_test = pd.read_csv("y_test_unb.csv").squeeze()

# ------------------------- Helper functions -------------------------
def evaluate_model(config_name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    print(f"Configuration: {config_name}")
    print(confusion_matrix(y_te, y_pred))
    print(classification_report(y_te, y_pred))
    print('-' * 60)

# ------------------------- Standard preprocessing -------------------------
scaler = StandardScaler()
X_train_scaled = x_train
X_test_scaled = x_test

# ------------------------- Model definitions -------------------------
model_constructors = {
    'LogisticRegression': lambda weight: LogisticRegression(class_weight=weight, max_iter=1000, random_state=42),
    'SVM': lambda weight: SVC(class_weight=weight, probability=True, random_state=42),
    'GaussianNB': lambda weight: GaussianNB(),
    'MLP': lambda weight: MLPClassifier(max_iter=1000, random_state=42),
    'XGBoost': lambda weight: XGBClassifier(scale_pos_weight=(np.bincount(y_train)[0] / np.bincount(y_train)[1]) if weight == 'balanced' else 1, use_label_encoder=False, eval_metric='logloss', random_state=42),
    'GradientBoosting': lambda weight: GradientBoostingClassifier(random_state=42),
    'KNN': lambda weight: KNeighborsClassifier()
}

balancing_methods = {
    'No balancing': 'none',
    'Class weights': 'balanced',
    'SMOTE': 'smote'
}

smote = SMOTE(random_state=42)

# ------------------------- Evaluation without PCA -------------------------
for balance_name, balance_method in balancing_methods.items():
    for model_name, constructor in model_constructors.items():
        print(f"\n=== Model: {model_name}, {balance_name} ===")
        # Prepare data
        if balance_method == 'smote':
            X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
            X_tr, y_tr = X_res, y_res
        else:
            X_tr, y_tr = X_train_scaled, y_train
        # Create model
        weight = 'balanced' if balance_method == 'balanced' else None
        model = constructor(weight)
        # Train
        model.fit(X_tr, y_tr)
        # Evaluate
        evaluate_model(f"{model_name}, {balance_name}", model, X_test_scaled, y_test)

# ------------------------- PCA + balancing from 5 to 17 components -------------------------
for n_comp in range(5, 18):
    print(f"\n### PCA: {n_comp} components ###")
    pca = PCA(n_components=n_comp, random_state=42)
    X_tr_pca = pca.fit_transform(X_train_scaled)
    X_te_pca = pca.transform(X_test_scaled)

    for balance_name, balance_method in balancing_methods.items():
        for model_name, constructor in model_constructors.items():
            print(f"\n=== PCA={n_comp}, Model: {model_name}, {balance_name} ===")
            # Prepare data
            if balance_method == 'smote':
                X_res_pca, y_res_pca = smote.fit_resample(X_tr_pca, y_train)
                X_tr2, y_tr2 = X_res_pca, y_res_pca
            else:
                X_tr2, y_tr2 = X_tr_pca, y_train
            # Create model
            weight = 'balanced' if balance_method == 'balanced' else None
            model = constructor(weight)
            # Train
            model.fit(X_tr2, y_tr2)
            # Evaluate
            evaluate_model(f"PCA={n_comp}, {model_name}, {balance_name}", model, X_te_pca, y_test)

print('Done model_evaluation_with_pca_and_balancing!')
