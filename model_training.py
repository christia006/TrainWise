

import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

def train_model(model_name, X_train, y_train, hyperparameters=None):
    """
    Melatih model supervised learning.

    Args:
        model_name (str): Nama model ('RandomForest', 'DecisionTree', 'SVM').
        X_train (array-like): Fitur data pelatihan.
        y_train (array-like): Target data pelatihan.
        hyperparameters (dict): Kamus hyperparameter untuk model.

    Returns:
        tuple: (trained_model, training_time)
    """
    start_time = time.time()
    model = None

    if hyperparameters is None:
        hyperparameters = {}

    if model_name == 'RandomForest':
        model = RandomForestClassifier(**hyperparameters, random_state=42)
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier(**hyperparameters, random_state=42)
    elif model_name == 'SVM':
        # SVC membutuhkan data yang discale dengan baik
        model = SVC(**hyperparameters, probability=True, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Model '{model_name}' trained in {training_time:.2f} seconds.")
    return model, training_time

def evaluate_model(model, X, y, split_type="test"):
    """
    Mengevaluasi model dan mengembalikan metrik performa.

    Args:
        model (estimator): Model yang sudah terlatih.
        X (array-like): Fitur data untuk evaluasi.
        y (array-like): Target data untuk evaluasi.
        split_type (str): Tipe split data ('train', 'validation', 'test').

    Returns:
        dict: Kamus metrik performa.
    """
    y_pred = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
    }


    if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
        y_proba = model.predict_proba(X)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y, y_proba)
    else:
        metrics['roc_auc'] = None 

    print(f"Metrics for {split_type} set:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"  {metric_name}: {value:.4f}")

    return metrics

def save_model(model, filepath):
    """Menyimpan model terlatih ke disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Memuat model dari disk."""
    return joblib.load(filepath)

def plot_confusion_matrix(y_true, y_pred, model_name, plot_path):
    """Membuat dan menyimpan plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")

def plot_feature_importance(model, feature_names, model_name, plot_path):
    """
    Membuat dan menyimpan plot feature importance untuk model yang mendukungnya.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(indices)])
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")
    else:
        print(f"Model {model_name} does not have feature importances.")

if __name__ == '__main__':
  
    from data_preprocessing import load_and_preprocess_data
    import pandas as pd
    import os

 
    data = {
        'feature1': [10, 20, 30, np.nan, 50, 60, 70, 80, 90, 100],
        'feature2': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'feature3': [1.1, 2.2, np.nan, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    dummy_df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    dummy_dataset_path = 'data/dummy_dataset.csv'
    dummy_df.to_csv(dummy_dataset_path, index=False)

    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
        dummy_dataset_path, 'target'
    )

    if X_train is not None:
     
        print("\n--- Training Random Forest ---")
        rf_model, rf_time = train_model(
            'RandomForest', X_train, y_train, {'n_estimators': 100, 'max_depth': 5}
        )
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "test")

      
        os.makedirs('models', exist_ok=True)
        model_path = 'models/rf_dummy_model.joblib'
        save_model(rf_model, model_path)

      
        loaded_rf_model = load_model(model_path)
        print("Loaded model evaluation:")
        evaluate_model(loaded_rf_model, X_test, y_test, "loaded_test")

        os.makedirs('logs', exist_ok=True)
        plot_confusion_matrix(y_test, loaded_rf_model.predict(X_test), "RandomForest", "logs/rf_confusion_matrix.png")

        
        feature_names_after_preprocessing = []
        for name, pipe, cols in preprocessor.transformers:
            if name == 'num':
                feature_names_after_preprocessing.extend(cols)
            elif name == 'cat':
                # Dapatkan nama fitur dari OneHotEncoder
                ohe = pipe.named_steps['onehot']
                feature_names_after_preprocessing.extend(ohe.get_feature_names_out(cols))
        
        plot_feature_importance(loaded_rf_model, feature_names_after_preprocessing, "RandomForest", "logs/rf_feature_importance.png")

        # Contoh pelatihan Decision Tree
        print("\n--- Training Decision Tree ---")
        dt_model, dt_time = train_model('DecisionTree', X_train, y_train, {'max_depth': 3})
        dt_metrics = evaluate_model(dt_model, X_test, y_test, "test")

        # Contoh pelatihan SVM (mungkin membutuhkan waktu lebih lama untuk dataset besar)
        print("\n--- Training SVM ---")
        # Untuk SVM, pastikan data numerik sudah diskalakan (sudah dilakukan di preprocessor)
        # Coba C yang lebih kecil untuk dataset kecil
        svm_model, svm_time = train_model('SVM', X_train, y_train, {'C': 0.1})
        svm_metrics = evaluate_model(svm_model, X_test, y_test, "test")
    else:
        print("Skipping model training due to preprocessing failure.")