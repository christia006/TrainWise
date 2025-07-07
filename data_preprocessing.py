# automl_platform/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os 
import logging 

logger = logging.getLogger(__name__) 

def load_and_preprocess_data(dataset_path, target_column, test_size=0.2, random_state=42):
    """
    Memuat dataset, melakukan preprocessing otomatis, dan membagi data.

    Args:
        dataset_path (str): Path ke file dataset CSV.
        target_column (str): Nama kolom target (variabel dependen).
        test_size (float): Proporsi dataset yang akan digunakan sebagai data uji.
        random_state (int): Seed untuk reproduktibilitas pembagian data.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logger.error(f"Error: Dataset not found at {dataset_path}")
        return None, None, None, None, None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None, None, None, None, None

    if target_column not in df.columns:
        logger.error(f"Error: Target column '{target_column}' not found in dataset.")
        return None, None, None, None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# Bagian if __name__ == '__main__': (untuk pengujian internal modul)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Running data_preprocessing.py in test mode...")
    
   
    os.makedirs('data', exist_ok=True)

    data = {
        'feature1': [10, 20, 30, np.nan, 50, 60],
        'feature2': ['A', 'B', 'A', 'C', 'B', np.nan],
        'feature3': [1.1, 2.2, np.nan, 4.4, 5.5, 6.6],
        'target': [0, 1, 0, 1, 0, 1]
    }
    dummy_df = pd.DataFrame(data)
    dummy_dataset_path_test = 'data/dummy_dataset_for_test.csv'
    dummy_df.to_csv(dummy_dataset_path_test, index=False)

    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
        dummy_dataset_path_test, 'target'
    )

    if X_train is not None:
        print("Preprocessing successful!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        feature_names_after_preprocessing = []
        for name, pipe, cols in preprocessor.transformers:
            if name == 'num':
                feature_names_after_preprocessing.extend(cols)
            elif name == 'cat':
                ohe = pipe.named_steps['onehot']
                try:
                    feature_names_after_preprocessing.extend(ohe.get_feature_names_out(cols))
                except AttributeError:
                    logger.warning(f"OneHotEncoder.get_feature_names_out() not available. Using generic names for {len(cols)} categorical features.")
                    for col_name in cols:
                        feature_names_after_preprocessing.append(f"ohe_{col_name}_generic")
        
     
        if len(feature_names_after_preprocessing) != X_train.shape[1]:
            logger.warning("Discrepancy in feature names count after preprocessing. Generating fully generic names.")
            feature_names_after_preprocessing = [f"feature_{i}" for i in range(X_train.shape[1])]

        print(f"Sample feature names after preprocessing: {feature_names_after_preprocessing[:5]}...")
    else:
        print("Preprocessing failed.")