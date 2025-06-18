# src/data/__init__.py
"""
Data collection and preprocessing module for stress prediction model
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import requests
import json
import logging
from typing import Dict, List, Tuple, Optional

class StressDataCollector:
    """Collects and preprocesses stress-related data from multiple sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_kaggle_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load stress dataset from Kaggle"""
        try:
            # Example datasets that could be used:
            # - Stress and Mental Health Dataset
            # - Employee Stress Prediction
            # - Mental Health in Tech Survey
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Loaded dataset with {len(df)} records")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Kaggle dataset: {e}")
            return pd.DataFrame()
    
    def create_synthetic_stress_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create synthetic stress data for training"""
        np.random.seed(42)
        
        # Generate synthetic features
        data = {
            'age': np.random.randint(18, 65, n_samples),
            'work_hours': np.random.normal(8, 2, n_samples),
            'sleep_hours': np.random.normal(7, 1.5, n_samples),
            'exercise_frequency': np.random.randint(0, 7, n_samples),
            'caffeine_intake': np.random.randint(0, 5, n_samples),
            'social_support': np.random.randint(1, 5, n_samples),
            'job_satisfaction': np.random.randint(1, 5, n_samples),
            'financial_stress': np.random.randint(1, 5, n_samples),
            'relationship_status': np.random.choice(['single', 'married', 'divorced'], n_samples),
            'chronic_illness': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Calculate stress level based on features
        stress_score = (
            (df['work_hours'] - 8) * 0.3 +
            (8 - df['sleep_hours']) * 0.4 +
            (7 - df['exercise_frequency']) * 0.2 +
            df['caffeine_intake'] * 0.15 +
            (5 - df['social_support']) * 0.3 +
            (5 - df['job_satisfaction']) * 0.4 +
            df['financial_stress'] * 0.3 +
            df['chronic_illness'] * 0.5
        )
        
        # Categorize stress levels
        df['stress_level'] = pd.cut(stress_score, 
                                   bins=[-np.inf, 2, 4, 6, np.inf],
                                   labels=['Low', 'Moderate', 'High', 'Severe'])
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for model training"""
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'stress_level':
                df[col] = self.label_encoder.fit_transform(df[col])
        
        # Separate features and target
        X = df.drop('stress_level', axis=1)
        y = df['stress_level']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get feature names for the model"""
        return [col for col in df.columns if col != 'stress_level']