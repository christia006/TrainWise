

import os
from datetime import datetime
import json
import logging
import pandas as pd
import numpy as np


from db_manager import DBManager
from data_preprocessing import load_and_preprocess_data
from model_training import train_model, evaluate_model, save_model, plot_confusion_matrix, plot_feature_importance


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "dbname": "neurosordb",
    "user": "neurosord_user",
    "password": "Sayabag",
    "host": "localhost",
    "port": "5432"
}



def run_experiment(
    experiment_name,
    dataset_path,
    target_column,
    models_to_train=['RandomForest', 'DecisionTree', 'SVM'],
    model_hyperparameters={}
):
    """
    Menjalankan satu eksperimen AutoML lengkap.
    """
    db = DBManager(**DB_CONFIG)
    db.connect()

    if not db.conn:
        logger.error("Failed to connect to database. Aborting experiment.")
        return

    # Catat eksperimen baru di DB
    experiment_id = db.insert_experiment(experiment_name, dataset_path, target_column, notes="AutoML experiment run")
    if experiment_id is None:
        logger.error("Failed to insert new experiment record. Aborting.")
        db.disconnect()
        return

    logger.info(f"Starting experiment '{experiment_name}' (ID: {experiment_id})")
    db.update_experiment_status(experiment_id, 'running')

    try:
        # 1. Preprocessing Otomatis
        logger.info("Starting data preprocessing...")
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(dataset_path, target_column)

        if X_train is None:
            raise Exception("Data preprocessing failed. Check dataset path or target column.")

        logger.info("Data preprocessing completed.")
        
        # Dapatkan nama fitur setelah preprocessing (untuk visualisasi feature importance)
        feature_names_after_preprocessing = []
        for name, pipe, cols in preprocessor.transformers:
            if name == 'num':
                feature_names_after_preprocessing.extend(cols)
            elif name == 'cat':
                ohe = pipe.named_steps['onehot']
                try: 
                    # Coba cara modern dulu (scikit-learn >= 0.23)
                    feature_names_after_preprocessing.extend(ohe.get_feature_names_out(cols))
                except AttributeError:
                    # Jika get_feature_names_out tidak ada, gunakan pendekatan fallback yang lebih aman
                    logger.warning(f"OneHotEncoder.get_feature_names_out() not available. Using generic names for {len(cols)} categorical features.")
                    # Fallback yang lebih aman: Buat nama fitur generik.
                    # Ini kurang informatif tapi mencegah crash.
                    for col_name in cols:
                        feature_names_after_preprocessing.append(f"ohe_{col_name}_generic")
        
        # Penjagaan darurat: Pastikan feature_names_after_preprocessing memiliki panjang yang sama dengan X_train_processed.shape[1]
        # Ini penting agar plotting feature importance tidak crash jika jumlah nama fitur tidak cocok
        if len(feature_names_after_preprocessing) != X_train.shape[1]:
            logger.warning("Discrepancy in feature names count after preprocessing. Generating fully generic names for all processed features.")
            feature_names_after_preprocessing = [f"feature_{i}" for i in range(X_train.shape[1])]


        # 2. Pelatihan dan Evaluasi Model
        for model_name in models_to_train:
            logger.info(f"Training {model_name} model...")
            current_hyperparams = model_hyperparameters.get(model_name, {})

            # Buat log file khusus untuk model ini
            log_filename = f"model_log_{experiment_id}_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            log_path = os.path.join('logs', log_filename)
            
            model_logger = logging.getLogger(f'model_{model_name}_{experiment_id}')
            model_logger.setLevel(logging.INFO)
            # Hindari menambahkan handler yang sama berulang kali
            file_handler_exists = any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path) for h in model_logger.handlers)
            
            if not file_handler_exists:
                file_handler = logging.FileHandler(log_path)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                model_logger.addHandler(file_handler)
            else:
                file_handler = None # Tidak perlu membuat lagi

            try:
                model, training_time = train_model(model_name, X_train, y_train, current_hyperparams)

                # Simpan model
                model_filepath = os.path.join('models', f'{model_name}_exp{experiment_id}.joblib')
                save_model(model, model_filepath)

                # Masukkan model ke DB
                model_id = db.insert_model(experiment_id, model_name, model_filepath, current_hyperparams, training_time, log_path)
                if model_id is None:
                    raise Exception(f"Failed to insert model {model_name} into database.")

                # Evaluasi model pada training dan test set
                train_metrics = evaluate_model(model, X_train, y_train, split_type="train")
                test_metrics = evaluate_model(model, X_test, y_test, split_type="test")

                # Simpan metrik ke DB
                for metric_name, value in train_metrics.items():
                    if value is not None:
                        db.insert_metric(model_id, metric_name, value, 'train')
                for metric_name, value in test_metrics.items():
                    if value is not None:
                        db.insert_metric(model_id, metric_name, value, 'test')

                # Visualisasi
                plot_dir = os.path.join('logs', 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                
                plot_confusion_matrix(y_test, model.predict(X_test), model_name,
                                      os.path.join(plot_dir, f'{model_name}_confusion_matrix_exp{experiment_id}.png'))
                
                # Plot feature importance hanya jika model mendukung dan nama fitur tersedia
                if hasattr(model, 'feature_importances_') and feature_names_after_preprocessing:
                    plot_feature_importance(model, feature_names_after_preprocessing, model_name,
                                            os.path.join(plot_dir, f'{model_name}_feature_importance_exp{experiment_id}.png'))
                else:
                    logger.warning(f"Skipping feature importance plot for {model_name}. Model does not support or feature names not available.")

                model_logger.info(f"Model {model_name} training and evaluation completed successfully.")
            except Exception as e:
                logger.error(f"Error during training/evaluation of {model_name} (Experiment ID: {experiment_id}): {e}")
                if file_handler:
                    model_logger.error(f"Error during training/evaluation: {e}")
            finally:
                if file_handler:
                    # Pastikan handler dihapus hanya jika kita yang menambahkannya di sesi ini
                    model_logger.removeHandler(file_handler)
                    file_handler.close()

        db.update_experiment_status(experiment_id, 'completed', end_time=datetime.now())
        logger.info(f"Experiment '{experiment_name}' (ID: {experiment_id}) completed successfully.")

    except Exception as e:
        logger.error(f"Experiment '{experiment_name}' (ID: {experiment_id}) failed overall: {e}")
        db.update_experiment_status(experiment_id, 'failed', end_time=datetime.now())
    finally:
        db.disconnect()

def get_experiment_details(experiment_id):
    """Mengambil dan menampilkan detail eksperimen dari DB."""
    db = DBManager(**DB_CONFIG)
    db.connect()
    if not db.conn:
        logger.error("Failed to connect to database.")
        return

    experiment = db._execute_query("SELECT * FROM experiments WHERE experiment_id = %s", (experiment_id,), fetch_results=True)
    if not experiment:
        print(f"Experiment with ID {experiment_id} not found.")
        db.disconnect()
        return

    experiment = experiment[0]
    print(f"\n--- Experiment Details (ID: {experiment['experiment_id']}) ---")
    print(f"Name: {experiment['experiment_name']}")
    print(f"Dataset: {experiment['dataset_path']}")
    print(f"Target Column: {experiment['target_column']}")
    print(f"Status: {experiment['status']}")
    print(f"Start Time: {experiment['start_time']}")
    print(f"End Time: {experiment['end_time']}")
    print(f"Notes: {experiment['notes']}")

    models = db.get_models_for_experiment(experiment_id)
    if models:
        print("\n--- Models Trained ---")
        for model in models:
            print(f"  Model ID: {model['model_id']}")
            print(f"  Name: {model['model_name']}")
            print(f"  Path: {model['model_path']}")
            # Parse JSONB back to Python dict for printing
            hyperparameters_dict = model['hyperparameters'] if isinstance(model['hyperparameters'], dict) else json.loads(model['hyperparameters'])
            print(f"  Hyperparameters: {json.dumps(hyperparameters_dict, indent=2)}")
            print(f"  Training Time: {model['training_time']:.2f} seconds")
            print(f"  Log Path: {model['log_path']}")

            metrics = db.get_metrics_for_model(model['model_id'])
            if metrics:
                print("    Metrics:")
                for metric in metrics:
                    print(f"      {metric['metric_name']} ({metric['split_type']}): {metric['metric_value']:.4f}")
            print("-" * 30)
    else:
        print("\nNo models found for this experiment.")

    db.disconnect()

def list_experiments():
    """Mencantumkan semua eksperimen yang ada dalam DB."""
    db = DBManager(**DB_CONFIG)
    db.connect()
    if not db.conn:
        logger.error("Failed to connect to database.")
        return

    experiments = db.get_experiments()
    if experiments:
        print("\n--- All Experiments ---")
        for exp in experiments:
            print(f"ID: {exp['experiment_id']}, Name: {exp['experiment_name']}, Status: {exp['status']}, Start: {exp['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("No experiments found.")
    db.disconnect()


if __name__ == '__main__':
    # Pastikan direktori ada sebelum menjalankan
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/plots', exist_ok=True)

    # --- Contoh Penggunaan ---

    # 1. Buat dummy dataset untuk pengujian (jika belum ada)
    dummy_dataset_path = 'data/sample_classification_data.csv'
    if not os.path.exists(dummy_dataset_path):
        print("Creating dummy dataset...")
        data = {
            'numerical_feature_1': np.random.rand(100) * 100,
            'numerical_feature_2': np.random.randint(0, 10, 100),
            'categorical_feature_1': ['A'] * 30 + ['B'] * 40 + ['C'] * 30,
            'categorical_feature_2': ['X'] * 50 + ['Y'] * 50,
            'missing_num_feature': [np.nan if i % 7 == 0 else i for i in range(100)],
            'missing_cat_feature': ['apple' if i % 5 == 0 else (np.nan if i % 3 == 0 else 'banana') for i in range(100)],
            'target_class': [0] * 40 + [1] * 60 # Contoh klasifikasi biner
        }
        dummy_df = pd.DataFrame(data)
        dummy_df.to_csv(dummy_dataset_path, index=False)
        print(f"Dummy dataset created at {dummy_dataset_path}")
    else:
        print(f"Dummy dataset already exists at {dummy_dataset_path}. Skipping creation.")


    # Input 1: Basic Model Comparison
    print("\n--- Running Experiment 1: Basic Model Comparison (RF, DT, SVM) ---")
    run_experiment(
        experiment_name="Basic Model Comparison (RF, DT, SVM)",
        dataset_path=dummy_dataset_path,
        target_column="target_class",
        models_to_train=['RandomForest', 'DecisionTree', 'SVM'],
        model_hyperparameters={
            'RandomForest': {'n_estimators': 100, 'max_depth': 10},
            'DecisionTree': {'max_depth': 5},
            'SVM': {'C': 1.0, 'kernel': 'rbf'}
        }
    )

    # Input 2: Tuned Random Forest
    print("\n--- Running Experiment 2: Tuned Random Forest ---")
    run_experiment(
        experiment_name="Tuned RandomForest",
        dataset_path=dummy_dataset_path,
        target_column="target_class",
        models_to_train=['RandomForest'],
        model_hyperparameters={
            'RandomForest': {'n_estimators': 200, 'max_depth': 15, 'min_samples_leaf': 5}
        }
    )

    # Input 3: Decision Tree with deeper max_depth
    print("\n--- Running Experiment 3: Decision Tree - Deeper ---")
    run_experiment(
        experiment_name="Decision Tree - Deeper",
        dataset_path=dummy_dataset_path,
        target_column="target_class",
        models_to_train=['DecisionTree'],
        model_hyperparameters={
            'DecisionTree': {'max_depth': 8, 'min_samples_leaf': 10}
        }
    )
    
    # Input 4: SVM with different kernel
    print("\n--- Running Experiment 4: SVM - Poly Kernel ---")
    run_experiment(
        experiment_name="SVM - Polynomial Kernel",
        dataset_path=dummy_dataset_path,
        target_column="target_class",
        models_to_train=['SVM'],
        model_hyperparameters={
            'SVM': {'C': 0.5, 'kernel': 'poly', 'degree': 3}
        }
    )

  
    print("\n--- Running Experiment 5: Random Forest - Fewer Estimators ---")
    run_experiment(
        experiment_name="RandomForest - Fewer Estimators",
        dataset_path=dummy_dataset_path,
        target_column="target_class",
        models_to_train=['RandomForest'],
        model_hyperparameters={
            'RandomForest': {'n_estimators': 50, 'max_depth': 8}
        }
    )


    print("\n--- Listing All Experiments ---")
    list_experiments()


    print("\n--- Viewing Details of Experiment ID 1 ---")
    get_experiment_details(1)
    
    print("\n--- Viewing Details of Experiment ID 2 ---")
    get_experiment_details(2)