import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def evaluate_model(config_name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    print(f"Configuration: {config_name}")
    print(confusion_matrix(y_te, y_pred))
    print(classification_report(y_te, y_pred, zero_division=0))
    print('-' * 60)

def get_model_constructors(y_train):
    return {
        'LogisticRegression': lambda weight: LogisticRegression(class_weight=weight, max_iter=1000, random_state=42),
        'SVM': lambda weight: SVC(class_weight=weight, probability=True, random_state=42),
        'GaussianNB': lambda weight: GaussianNB(),
        'MLP': lambda weight: MLPClassifier(max_iter=1000, random_state=42),
        'XGBoost': lambda weight: XGBClassifier(
            scale_pos_weight=(np.bincount(y_train)[0] / np.bincount(y_train)[1]) if weight == 'balanced' else 1,
            use_label_encoder=False,
            eval_metric='logloss',
            #missing=1,
            random_state=42
        ),
        'GradientBoosting': lambda weight: GradientBoostingClassifier(random_state=42),
        'KNN': lambda weight: KNeighborsClassifier()
    }

def get_balancing_methods():
    return {
        'No balancing': 'none',
        'Class weights': 'balanced',
        'SMOTE': 'smote'
    }

def train_and_evaluate_models(X_train, y_train, X_test, y_test, model_constructors, balancing_methods):
    smote = SMOTE(random_state=42)
    for balance_name, balance_method in balancing_methods.items():
        for model_name, constructor in model_constructors.items():
            print(f"\n=== Model: {model_name}, {balance_name} ===")
            if balance_method == 'smote':
                X_res, y_res = smote.fit_resample(X_train, y_train)
                X_tr, y_tr = X_res, y_res
            else:
                X_tr, y_tr = X_train, y_train
            weight = 'balanced' if balance_method == 'balanced' else None
            model = constructor(weight)
            model.fit(X_tr, y_tr)
            evaluate_model(f"{model_name}, {balance_name}", model, X_test, y_test)

def pca_train_and_evaluate_models(X_train, y_train, X_test, y_test, model_constructors, balancing_methods, n_comp_range):
    smote = SMOTE(random_state=42)
    for n_comp in n_comp_range:
        print(f"\n### PCA: {n_comp} components ###")
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca.fit_transform(X_train)
        X_te_pca = pca.transform(X_test)

        for balance_name, balance_method in balancing_methods.items():
            for model_name, constructor in model_constructors.items():
                print(f"\n=== PCA={n_comp}, Model: {model_name}, {balance_name} ===")
                if balance_method == 'smote':
                    X_res_pca, y_res_pca = smote.fit_resample(X_tr_pca, y_train)
                    X_tr2, y_tr2 = X_res_pca, y_res_pca
                else:
                    X_tr2, y_tr2 = X_tr_pca, y_train
                weight = 'balanced' if balance_method == 'balanced' else None
                model = constructor(weight)
                model.fit(X_tr2, y_tr2)
                evaluate_model(f"PCA={n_comp}, {model_name}, {balance_name}", model, X_te_pca, y_test)
