# ml_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from joblib import Memory

def create_memory_cache():
    memory = Memory(location='cachedir', verbose=0)
    memory.clear(warn=False)
    return memory

def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")

def train_model(name, pipeline, param_grid, x_train, y_train, x_test, y_test, cv):
    print(f"🔹 Training {name}...")
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', verbose=3, n_jobs=-1)
    grid.fit(x_train, y_train)
    preds = grid.predict(x_test)
    evaluate(name, y_test, preds)
    return grid, preds

def train_with_pca(model_name, base_model, param_grid, x_train, y_train, x_test, y_test, cv, memory):
    for n in [5, 10, 11, 12, 13, 14, 15]:
        print(f'\n🔵 Training {model_name.upper()} with PCA ({n} components)...')
        pipe = Pipeline([
            ('pca', PCA(n_components=n)),
            (model_name, base_model)
        ], memory=memory)

        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1', verbose=2, n_jobs=-1)
        grid.fit(x_train, y_train)
        y_pred = grid.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        print(f"\n📊 {model_name.upper()} + PCA ({n})")
        print("-" * 40)
        print(f"Accuracy       : {acc:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        print(f"Precision       : {prec:.4f}")
        print(f"Recall          : {rec:.4f}")
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{model_name.upper()} + PCA ({n}) – Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
