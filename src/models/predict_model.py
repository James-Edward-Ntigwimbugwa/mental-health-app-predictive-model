# src/models/predict_model.py
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from ..features.problem_analyzer import StressProblemAnalyzer
from ..features.solution_generator import AIStressSolutionGenerator

logger = logging.getLogger(__name__)

class StressPredictor:
    """End-to-end stress prediction and recommendation system"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.stress_levels = None
        self.problem_analyzer = StressProblemAnalyzer()
        self.solution_generator = AIStressSolutionGenerator()
        self.load_model()
    
    def load_model(self):
        """Load trained stress prediction model"""
        model_path = Path(__file__).resolve().parents[2] / 'models/stress_model.pkl'
        try:
            model_data = joblib.load(model_path)
            
            # Handle different possible formats of the saved model
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', [])
                self.stress_levels = model_data.get('stress_levels', ['Low', 'Moderate', 'High', 'Severe'])
            else:
                self.model = model_data
                self.scaler = None
                self.feature_names = []
                self.stress_levels = ['Low', 'Moderate', 'High', 'Severe']
            
            if self.model is None:
                raise ValueError("No model found in the saved file")
            
            # Validate and correct feature names
            if hasattr(self.model, 'n_features_in_'):
                n_features = self.model.n_features_in_
                if len(self.feature_names) != n_features:
                    logger.warning(f"Feature mismatch: {len(self.feature_names)} names vs {n_features} expected")
                    if '0' in self.feature_names:
                        self.feature_names.remove('0')
                        logger.info("Removed extra feature '0'")
                    if len(self.feature_names) == n_features:
                        logger.info(f"Feature count corrected to {n_features}")
                    else:
                        logger.error(f"Feature count mismatch persists: {len(self.feature_names)} vs {n_features}")
                        raise ValueError(f"Cannot resolve feature mismatch: expected {n_features}, got {len(self.feature_names)}")
            
            logger.info('Model loaded successfully')
            logger.info(f'Model type: {type(self.model)}')
            logger.info(f'Feature names: {len(self.feature_names)} features')
            logger.info(f'Stress levels: {self.stress_levels}')
            
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            raise

    def debug_features(self):
        """Debug method to check feature alignment"""
        print(f"Model expects {len(self.feature_names)} features:")
        for i, feature in enumerate(self.feature_names):
            print(f"  {i+1:2d}. {feature}")
        return self.feature_names
    
    def predict_stress(self, features: dict) -> dict:
        """Predict stress level and generate recommendations"""
        try:
            # Convert input to model format
            input_vector = self._format_input(features)
            
            # Scale input if scaler is available
            if self.scaler is not None:
                input_vector = self.scaler.transform(input_vector)
            
            # Predict stress level
            prediction_idx = self.model.predict(input_vector)[0]
            
            # Handle both string and numeric predictions
            if isinstance(prediction_idx, (int, np.integer)):
                if prediction_idx < len(self.stress_levels):
                    stress_level = self.stress_levels[prediction_idx]
                else:
                    stress_level = 'High'  # Default fallback
            else:
                stress_level = str(prediction_idx)
            
            # Get probabilities if available
            try:
                probabilities = self.model.predict_proba(input_vector)[0]
                confidence = max(probabilities)
            except:
                confidence = 0.8  # Default confidence if predict_proba not available
            
            # Analyze problems
            problem_report = self.problem_analyzer.analyze_problems(
                stress_level, user_factors=features
            )
            
            # Generate solutions
            solution_plan = self.solution_generator.generate_personalized_solutions(
                stress_level, features
            )
            
            # Generate comprehensive report
            final_report = self.solution_generator.generate_comprehensive_report(
                stress_level, features, problem_report
            )
            
            return {
                'stress_level': stress_level,
                'confidence': confidence,
                'problem_analysis': problem_report,
                'solutions': solution_plan,
                'comprehensive_report': final_report
            }
            
        except Exception as e:
            logger.error(f'Prediction failed: {e}')
            raise
    
    def _format_input(self, features: dict) -> np.ndarray:
        """Convert feature dictionary to model input format"""
        # If no feature names are defined, try to infer from input
        if not self.feature_names:
            self.feature_names = list(features.keys())
            logger.warning("No feature names found in model, using input keys")
        
        # Create vector in correct feature order
        input_vector = []
        for feature in self.feature_names:
            if feature in features:
                input_vector.append(features[feature])
            else:
                logger.warning(f'Missing feature {feature}, using default value 0')
                input_vector.append(0)
        
        return np.array([input_vector])

def main():
    """Run sample prediction with feature debugging"""
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    logger.info('Running stress prediction')
    
    try:
        predictor = StressPredictor()
        
        # DEBUG: Print expected features
        print("\n=== DEBUGGING FEATURES ===")
        expected_features = predictor.debug_features()
        
        # Use features that match the model from stress_model.py
        user_data = {
            'sleep_hours': 6.5,
            'work_hours': 9,
            'exercise_minutes': 30,
            'social_interactions': 3,
            'caffeine_intake': 2,
            'screen_time_hours': 6,
            'meditation_minutes': 15,
            'workload_rating': 8,
            'relationship_satisfaction': 6,
            'financial_stress': 7
        }
        
        print(f"\nProvided {len(user_data)} features:")
        for i, (feature, value) in enumerate(user_data.items()):
            print(f"  {i+1:2d}. {feature} = {value}")
        
        # Check for mismatches
        provided_set = set(user_data.keys())
        expected_set = set(expected_features)
        
        missing = expected_set - provided_set
        extra = provided_set - expected_set
        
        if missing:
            print(f"\nMISSING FEATURES: {missing}")
            print("Adding missing features with default value 0")
            for feature in missing:
                user_data[feature] = 0
        
        if extra:
            print(f"\nEXTRA FEATURES: {extra}")
            print("Removing extra features")
            for feature in list(extra):  # Convert to list to avoid modification during iteration
                del user_data[feature]
        
        print(f"\nFinal user_data has {len(user_data)} features")
        print("=" * 30)
        
        result = predictor.predict_stress(user_data)
        
        # Print comprehensive report
        print("\n" + "="*80)
        print(result['comprehensive_report'])
        print("="*80)
        
        logger.info('Prediction complete')
        
    except Exception as e:
        logger.error(f'Prediction failed: {e}')
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()