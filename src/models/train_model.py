# src/models/train_model.py
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from . import StressPredictionModel

logger = logging.getLogger(__name__)

def main():
    """Trains and saves the stress prediction model"""
    logger.info('Training stress prediction model')
    
    # Load processed data
    features_dir = Path(__file__).resolve().parents[2] / 'features'
    
    X_train = np.load(features_dir / 'X_train.npy')
    X_test = np.load(features_dir / 'X_test.npy')
    y_train = np.load(features_dir / 'y_train.npy')
    y_test = np.load(features_dir / 'y_test.npy')
    
    feature_names = pd.read_csv(features_dir / 'feature_names.csv', header=None)[0].tolist()
    stress_levels = pd.read_csv(features_dir / 'class_names.csv', header=None)[0].tolist()
    
    # Initialize and train model
    model = StressPredictionModel()
    model.train_models(X_train, y_train, X_test, y_test, feature_names)
    
    # Save model
    models_dir = Path(__file__).resolve().parents[2] / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'stress_model.pkl'
    model.save_model(model_path)
    
    logger.info(f'Model saved to {model_path}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()