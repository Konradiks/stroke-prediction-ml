import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from joblib import Memory
import sklearn
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')

# ------------------------- Konfiguracja -------------------------
print('Konfiguracja pamięci...')
sklearn.set_config(working_memory=12000)
memory = Memory(location='cachedir', verbose=0)
memory.clear(warn=False)

# ------------------------- Dane -------------------------
print('Wczytywanie danych...')
x_train = pd.read_csv("x_train_scaled.csv")
x_test = pd.read_csv("x_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print(y_train.value_counts())

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ------------------------- SVC -------------------------
print('Trenowanie SVC...')
pipeline_svc = Pipeline([
    ('svc', SVC(probability=True))
], memory=memory)

svc_params = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['rbf', 'linear'],
    'svc__gamma': ['scale', 'auto']
}

svc_grid = GridSearchCV(pipeline_svc, svc_params, cv=cv, scoring='f1', verbose=3, n_jobs=-1)
svc_grid.fit(x_train, y_train)
svc_preds = svc_grid.predict(x_test)

# ------------------------- LDA -------------------------
print('Trenowanie LDA...')
pipeline_lda = Pipeline([
    ('lda', LDA(solver='lsqr', shrinkage='auto'))
], memory=memory)

lda_grid = GridSearchCV(pipeline_lda, {}, cv=cv, scoring='f1', verbose=3, n_jobs=-1)
lda_grid.fit(x_train, y_train)
lda_preds = lda_grid.predict(x_test)

# ------------------------- Random Forest -------------------------
print('Trenowanie Random Forest...')
pipeline_rf = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))
], memory=memory)

rf_params = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2],
    'rf__bootstrap': [True, False]
}

rf_grid = GridSearchCV(pipeline_rf, rf_params, cv=cv, scoring='f1', verbose=3, n_jobs=-1)
rf_grid.fit(x_train, y_train)
rf_preds = rf_grid.predict(x_test)

# ------------------------- XGBoost -------------------------
print('Trenowanie XGBoost...')
pipeline_xgb = Pipeline([
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
], memory=memory)

xgb_params = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 6, 9],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.8, 1],
    'xgb__colsample_bytree': [0.8, 1]
}

xgb_grid = GridSearchCV(pipeline_xgb, xgb_params, cv=cv, scoring='f1', verbose=3, n_jobs=-1)
xgb_grid.fit(x_train, y_train)
xgb_preds = xgb_grid.predict(x_test)

# ------------------------- Stacking -------------------------
print('Trenowanie StackingClassifier...')
estimators = [
    ('svc', svc_grid.best_estimator_),
    ('rf', rf_grid.best_estimator_),
    ('xgb', xgb_grid.best_estimator_)
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv,
    n_jobs=-1
)

stacking.fit(x_train, y_train)
stacking_preds = stacking.predict(x_test)

# ------------------------- Wyniki -------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")

print("\nOcena modeli:")
evaluate("SVC", y_test, svc_preds)
evaluate("LDA", y_test, lda_preds)
evaluate("Random Forest", y_test, rf_preds)
evaluate("XGBoost", y_test, xgb_preds)
evaluate("Stacking", y_test, stacking_preds)

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

def train_with_pca(model_name, base_model, param_grid, x_train, y_train, x_test, y_test):
    for n in [5, 10, 11, 12, 13, 14, 15]:
        print(f'\n🔵 Trenowanie {model_name.upper()} z PCA ({n} komponentów)...')
        pipe = Pipeline([
            ('pca', PCA(n_components=n)),
            (model_name, base_model)
        ], memory=memory)

        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1', verbose=2, n_jobs=-1)
        grid.fit(x_train, y_train)
        y_pred = grid.predict(x_test)

        # Ewaluacja
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        print(f"\n📊 {model_name.upper()} + PCA ({n} komponentów)")
        print("-" * 40)
        print(f"Accuracy       : {acc:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        print(f"Precision       : {prec:.4f}")
        print(f"Recall          : {rec:.4f}")
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))

        # Macierz pomyłek
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{model_name.upper()} + PCA ({n}) – Macierz Pomyłek')
        plt.xlabel('Przewidziane')
        plt.ylabel('Rzeczywiste')
        plt.tight_layout()
        plt.show()

# SVC
svc_base = SVC(probability=True)
svc_param = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['rbf', 'linear'],
    'svc__gamma': ['scale', 'auto']
}

# Random Forest
rf_base = RandomForestClassifier(random_state=42)
rf_param = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

# XGBoost
xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_param = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 6],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0]
}


train_with_pca('svc', svc_base, svc_param, x_train, y_train, x_test, y_test)
train_with_pca('rf', rf_base, rf_param, x_train, y_train, x_test, y_test)
train_with_pca('xgb', xgb_base, xgb_param, x_train, y_train, x_test, y_test)
