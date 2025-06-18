# src/features/build_features.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess data for model training"""
    logger.info('Preprocessing data')
    
    # Handle categorical variables
    df = pd.get_dummies(df, columns=['gender', 'dataset_source'], drop_first=True)
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['stress_level'])
    
    # Separate features and target
    X = df.drop('stress_level', axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder, X.columns.tolist()

def main():
    """Processes raw data into features for modeling"""
    logger.info('Building features')
    
    # Load dataset
    processed_dir = Path(__file__).resolve().parents[2] / 'data/processed'
    input_path = processed_dir / 'combined_stress_data.csv'
    df = pd.read_csv(input_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = preprocess_data(df)
    
    # Save processed data
    features_dir = Path(__file__).resolve().parents[2] / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(features_dir / 'X_train.npy', X_train)
    np.save(features_dir / 'X_test.npy', X_test)
    np.save(features_dir / 'y_train.npy', y_train)
    np.save(features_dir / 'y_test.npy', y_test)
    
    # Save metadata
    pd.Series(feature_names).to_csv(features_dir / 'feature_names.csv', index=False)
    pd.Series(encoder.classes_).to_csv(features_dir / 'class_names.csv', index=False)
    
    logger.info('Feature engineering complete')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()