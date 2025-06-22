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
        self.problem_analyzer = StressProblemAnalyzer()
        self.solution_generator = AIStressSolutionGenerator()
        self.load_model()
    
    def load_model(self):
        """Load trained stress prediction model"""
        model_path = Path(__file__).resolve().parents[2] / 'models/stress_model.pkl'
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.stress_levels = model_data['stress_levels']
            logger.info('Model loaded successfully')
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
        # Convert input to model format
        input_vector = self._format_input(features)
        
        # Predict stress level
        prediction_idx = self.model.predict(input_vector)[0]
        stress_level = self.stress_levels[prediction_idx]
        
        # Get probabilities
        probabilities = self.model.predict_proba(input_vector)[0]
        confidence = max(probabilities)
        
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
    
    def _format_input(self, features: dict) -> np.ndarray:
        """Convert feature dictionary to model input format"""
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
    
    predictor = StressPredictor()
    
    # DEBUG: Print expected features
    print("\n=== DEBUGGING FEATURES ===")
    expected_features = predictor.debug_features()
    
    # Original user data
    user_data = {
        'age': 35,
        'sleep_quality': 3,
        'social_support': 4,
        'work_stress': 5,
        'financial_stress': 4,
        'health_status': 3,
        'exercise_frequency': 2,
        'substance_use': 1,
        'mental_health_history': 0,
        'gender_Male': 1,
        'dataset_source_workplace': 1
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
        for feature in extra:
            del user_data[feature]
    
    print(f"\nFinal user_data has {len(user_data)} features")
    print("=" * 30)
    
    try:
        result = predictor.predict_stress(user_data)
        
        # Print comprehensive report
        print("\n" + "="*80)
        print(result['comprehensive_report'])
        print("="*80)
        
        logger.info('Prediction complete')
        
    except Exception as e:
        logger.error(f'Prediction failed: {e}')
        print(f"\nERROR: {e}")
        
        # Additional debugging
        input_vector = predictor._format_input(user_data)
        print(f"Input vector shape: {input_vector.shape}")
        print(f"Expected features: {len(expected_features)}")