# src/models/__init__.py
"""
AI Prediction Model for Stress Level Analysis and Recommendations
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class StressPredictionModel:
    """Advanced AI model for stress level prediction and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        }
        self.best_model = None
        self.feature_names = None
        self.stress_levels = ['Low', 'Moderate', 'High', 'Severe']
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, 
                    feature_names: List[str]) -> Dict[str, float]:
        """Train multiple models and select the best one"""
        
        self.feature_names = feature_names
        model_scores = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mean_cv_score = np.mean(cv_scores)
            
            # Test score
            test_score = model.score(X_test, y_test)
            
            model_scores[name] = {
                'cv_score': mean_cv_score,
                'test_score': test_score,
                'model': model
            }
            
            self.logger.info(f"{name} - CV Score: {mean_cv_score:.4f}, Test Score: {test_score:.4f}")
        
        # Select best model based on cross-validation score
        best_model_name = max(model_scores.keys(), 
                             key=lambda x: model_scores[x]['cv_score'])
        
        self.best_model = model_scores[best_model_name]['model']
        self.logger.info(f"Best model selected: {best_model_name}")
        
        # Generate detailed evaluation
        y_pred = self.best_model.predict(X_test)
        self.logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        return model_scores
    
    def predict_stress_level(self, features: dict) -> Dict[str, Any]:
        """Predict stress level from feature dictionary"""
        if self.best_model is None:
            raise ValueError("Model not trained yet.")
        
        # Convert to array in correct feature order
        input_vector = []
        for feature in self.feature_names:
            input_vector.append(features.get(feature, 0))
        
        # Predict stress level
        prediction = self.best_model.predict([input_vector])[0]
        probabilities = self.best_model.predict_proba([input_vector])[0]
        
        return {
            'stress_level': self.stress_levels[prediction],
            'confidence': max(probabilities),
            'probabilities': dict(zip(self.stress_levels, probabilities))
        }
    
    
    def analyze_stress_factors(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze key stress factors contributing to the prediction"""
        prediction_result = self.predict_stress_level(features)
        
        if prediction_result['feature_importance']:
            # Sort features by importance
            sorted_features = sorted(
                prediction_result['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_factors = sorted_features[:5]  # Top 5 contributing factors
            
            return {
                'prediction': prediction_result,
                'top_stress_factors': top_factors,
                'risk_assessment': self._assess_risk_level(prediction_result['stress_level'])
            }
        
        return prediction_result
    
    def _assess_risk_level(self, stress_level: str) -> Dict[str, Any]:
        """Assess risk level based on stress prediction"""
        risk_mapping = {
            'Low': {
                'risk_level': 'Minimal',
                'urgency': 'Low',
                'monitoring_frequency': 'Monthly'
            },
            'Moderate': {
                'risk_level': 'Moderate',
                'urgency': 'Medium',
                'monitoring_frequency': 'Weekly'
            },
            'High': {
                'risk_level': 'High',
                'urgency': 'High',
                'monitoring_frequency': 'Daily'
            },
            'Severe': {
                'risk_level': 'Critical',
                'urgency': 'Immediate',
                'monitoring_frequency': 'Continuous'
            }
        }
        
        return risk_mapping.get(stress_level, risk_mapping['Moderate'])
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.best_model,
            'feature_names': self.feature_names,
            'stress_levels': self.stress_levels
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.stress_levels = model_data['stress_levels']
        self.logger.info(f"Model loaded from {filepath}")