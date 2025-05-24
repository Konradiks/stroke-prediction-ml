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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

svc_grid = GridSearchCV(pipeline_svc, svc_params, cv=cv, scoring='f1_macro', verbose=2, n_jobs=-1)
svc_grid.fit(x_train, y_train)
svc_preds = svc_grid.predict(x_test)

# ------------------------- LDA -------------------------
print('Trenowanie LDA...')
pipeline_lda = Pipeline([
    ('lda', LDA(solver='lsqr', shrinkage='auto'))
], memory=memory)

lda_grid = GridSearchCV(pipeline_lda, {}, cv=cv, scoring='f1_macro', verbose=2, n_jobs=-1)
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

rf_grid = GridSearchCV(pipeline_rf, rf_params, cv=cv, scoring='f1_macro', verbose=2, n_jobs=-1)
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

xgb_grid = GridSearchCV(pipeline_xgb, xgb_params, cv=cv, scoring='f1_macro', verbose=2, n_jobs=-1)
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
