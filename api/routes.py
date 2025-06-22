from flask import Blueprint, request, jsonify
import logging
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
import os

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Global predictor variable
predictor = None

class StressModelWrapper:
    """Wrapper for the stress model to provide consistent interface"""
    
    def __init__(self, model_path):
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.solutions_database = {}
        self.solution_generator = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the stress model from pickle file"""
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            # Required components
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.feature_names = self.model_data.get('feature_names', [])
            
            # Optional components with fallbacks
            self.solutions_database = self.model_data.get('solutions_database', {})
            self.solution_generator = self.model_data.get('solution_generator', None)
            
            # If solutions database is empty, create default one
            if not self.solutions_database:
                self.solutions_database = self._create_default_solutions_database()
                logging.info("Created default solutions database")
            
            # If solution generator is missing, create default one
            if self.solution_generator is None:
                self.solution_generator = self._create_default_solution_generator()
                logging.info("Created default solution generator")
            
            # If feature names are missing, try to infer them
            if not self.feature_names:
                self.feature_names = self._get_default_feature_names()
                logging.warning("Using default feature names - model may not work correctly")
            
            logging.info(f"Stress model loaded successfully from {model_path}")
            logging.info(f"Model expects {len(self.feature_names)} features: {self.feature_names}")
            
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _create_default_solutions_database(self):
        """Create a default solutions database"""
        return {
            'low': {
                'message': 'Your stress levels appear to be manageable',
                'priority': 'maintenance',
                'solutions': [
                    'Continue your current healthy habits',
                    'Practice preventive stress management',
                    'Maintain work-life balance',
                    'Regular exercise routine',
                    'Adequate sleep schedule'
                ]
            },
            'moderate': {
                'message': 'You are experiencing moderate stress levels',
                'priority': 'attention',
                'solutions': [
                    'Implement stress reduction techniques',
                    'Consider meditation or mindfulness',
                    'Improve time management',
                    'Seek social support',
                    'Review work-life balance'
                ]
            },
            'high': {
                'message': 'You are experiencing high stress levels',
                'priority': 'urgent',
                'solutions': [
                    'Seek professional help if needed',
                    'Implement immediate stress relief techniques',
                    'Consider counseling or therapy',
                    'Reduce workload if possible',
                    'Focus on self-care activities'
                ]
            }
        }
    
    def _create_default_solution_generator(self):
        """Create a default solution generator function"""
        def generate_solutions(stress_level, features):
            base_solutions = self.solutions_database.get(stress_level, {}).get('solutions', [])
            
            # Add personalized recommendations based on features
            personalized = base_solutions.copy()
            
            # Convert features array back to feature dict for analysis
            feature_dict = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(features):
                    feature_dict[feature_name] = features[i]
            
            # Add specific recommendations based on feature values
            if feature_dict.get('sleep_hours', 8) < 6:
                personalized.insert(0, 'Prioritize getting 7-9 hours of sleep per night')
            
            if feature_dict.get('exercise_minutes', 30) < 30:
                personalized.insert(0, 'Increase physical activity to at least 30 minutes daily')
            
            if feature_dict.get('work_hours', 8) > 10:
                personalized.insert(0, 'Consider reducing work hours or improving time management')
            
            if feature_dict.get('meditation_minutes', 0) < 10:
                personalized.append('Try 10-15 minutes of daily meditation or mindfulness')
            
            return personalized[:10]  # Return top 10 recommendations
        
        return generate_solutions
    
    def _get_default_feature_names(self):
        """Get default feature names if not provided in model"""
        return [
            'sleep_hours',
            'work_hours', 
            'exercise_minutes',
            'social_interactions',
            'caffeine_intake',
            'screen_time_hours',
            'meditation_minutes',
            'workload_rating',
            'relationship_satisfaction',
            'financial_stress'
        ]
    
    def predict_stress(self, features):
        """Predict stress level and generate recommendations"""
        try:
            # Convert features dict to array in correct order
            feature_array = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_array.append(features[feature_name])
                else:
                    logging.warning(f"Missing feature {feature_name}, using default value 0")
                    feature_array.append(0)
            
            # Scale features
            input_scaled = self.scaler.transform([feature_array])
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # Handle different prediction formats
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(input_scaled)[0]
                    confidence = max(probabilities)
                except:
                    confidence = 0.8  # Default confidence
            else:
                confidence = 0.8  # Default confidence for models without predict_proba
            
            # Convert numeric predictions to string labels if needed
            if isinstance(prediction, (int, float, np.integer, np.floating)):
                if prediction <= 0.33:
                    prediction = 'low'
                elif prediction <= 0.66:
                    prediction = 'moderate'
                else:
                    prediction = 'high'
            
            # Generate personalized solutions
            personalized_solutions = self.solution_generator(prediction, feature_array)
            
            # Get base solution info
            solution_info = self.solutions_database.get(prediction, {
                'message': 'Stress level assessed',
                'priority': 'attention'
            })
            
            # Create comprehensive report
            comprehensive_report = self._generate_comprehensive_report(
                prediction, features, solution_info, personalized_solutions
            )
            
            return {
                'stress_level': prediction,
                'confidence': float(confidence),
                'problem_analysis': {
                    'primary_stressors': self._identify_primary_stressors(features),
                    'risk_factors': self._identify_risk_factors(features),
                    'priority': solution_info.get('priority', 'attention')
                },
                'solutions': personalized_solutions,
                'comprehensive_report': comprehensive_report
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise
    
    def _identify_primary_stressors(self, features):
        """Identify primary stress factors from user input"""
        stressors = []
        
        # Map features to stressor categories
        if features.get('work_hours', 0) > 10:
            stressors.append("Excessive work hours")
        if features.get('sleep_hours', 8) < 6:
            stressors.append("Insufficient sleep")
        if features.get('financial_stress', 0) > 7:
            stressors.append("High financial stress")
        if features.get('workload_rating', 0) > 8:
            stressors.append("Heavy workload")
        if features.get('relationship_satisfaction', 10) < 4:
            stressors.append("Poor relationship satisfaction")
        if features.get('exercise_minutes', 30) < 20:
            stressors.append("Lack of physical activity")
        
        return stressors if stressors else ["General life pressures"]
    
    def _identify_risk_factors(self, features):
        """Identify risk factors that could worsen stress"""
        risk_factors = []
        
        if features.get('caffeine_intake', 0) > 4:
            risk_factors.append("High caffeine consumption")
        if features.get('screen_time_hours', 0) > 10:
            risk_factors.append("Excessive screen time")
        if features.get('social_interactions', 5) < 2:
            risk_factors.append("Limited social support")
        if features.get('meditation_minutes', 0) < 5:
            risk_factors.append("Lack of mindfulness practice")
        
        return risk_factors if risk_factors else ["No significant risk factors identified"]
    
    def _generate_comprehensive_report(self, stress_level, features, solution_info, solutions):
        """Generate a comprehensive stress analysis report"""
        report = f"""
STRESS ANALYSIS REPORT
======================

Stress Level: {stress_level.upper()}
Assessment: {solution_info.get('message', 'Stress level assessed')}
Priority: {solution_info.get('priority', 'attention').upper()}

KEY FINDINGS:
{self._format_findings(features)}

PERSONALIZED RECOMMENDATIONS:
{self._format_recommendations(solutions)}

NEXT STEPS:
- Implement highest priority recommendations first
- Monitor progress weekly
- Seek professional help if stress persists or worsens
- Consider regular stress assessments to track improvement

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        return report
    
    def _format_findings(self, features):
        """Format key findings from the analysis"""
        findings = []
        
        # Sleep analysis
        sleep_hours = features.get('sleep_hours', 8)
        if sleep_hours < 6:
            findings.append(f"• Critical sleep deficiency: {sleep_hours} hours (recommended: 7-9 hours)")
        elif sleep_hours < 7:
            findings.append(f"• Suboptimal sleep: {sleep_hours} hours (recommended: 7-9 hours)")
        
        # Work analysis
        work_hours = features.get('work_hours', 8)
        if work_hours > 10:
            findings.append(f"• Excessive work schedule: {work_hours} hours/day")
        
        # Exercise analysis
        exercise_min = features.get('exercise_minutes', 30)
        if exercise_min < 30:
            findings.append(f"• Insufficient physical activity: {exercise_min} minutes/day")
        
        # Stress factors
        financial_stress = features.get('financial_stress', 0)
        if financial_stress > 7:
            findings.append(f"• High financial stress level: {financial_stress}/10")
        
        workload = features.get('workload_rating', 0)
        if workload > 8:
            findings.append(f"• Overwhelming workload: {workload}/10")
        
        return '\n'.join(findings) if findings else "• No critical issues identified in measured parameters"
    
    def _format_recommendations(self, solutions):
        """Format recommendations for the report"""
        formatted = []
        for i, solution in enumerate(solutions[:8], 1):  # Show top 8 recommendations
            formatted.append(f"{i}. {solution}")
        
        return '\n'.join(formatted)
    
    def get_feature_template(self):
        """Get a template with all expected features and their descriptions"""
        template = {}
        
        descriptions = {
            'sleep_hours': 'Hours of sleep per night (4-12)',
            'work_hours': 'Hours worked per day (4-16)',
            'exercise_minutes': 'Minutes of exercise per day (0-180)',
            'social_interactions': 'Number of social interactions per day (0-15)',
            'caffeine_intake': 'Cups of coffee/caffeine per day (0-10)',
            'screen_time_hours': 'Hours of screen time per day (0-16)',
            'meditation_minutes': 'Minutes of meditation/mindfulness per day (0-120)',
            'workload_rating': 'Perceived workload stress (0-10 scale)',
            'relationship_satisfaction': 'Satisfaction with relationships (0-10 scale)',
            'financial_stress': 'Level of financial stress (0-10 scale)'
        }
        
        for feature in self.feature_names:
            template[feature] = {
                'description': descriptions.get(feature, 'No description available'),
                'required': True,
                'type': 'number'
            }
        
        return template
    
    def validate_input_features(self, features):
        """Validate input features and return missing/extra features"""
        provided_features = set(features.keys())
        expected_features = set(self.feature_names)
        
        missing = expected_features - provided_features
        extra = provided_features - expected_features
        
        return missing, extra

def init_predictor():
    """Initialize the stress predictor"""
    global predictor
    
    if predictor is not None:
        logging.info("Predictor already initialized")
        return
    
    try:
        # Get the current working directory and the directory where this script is located
        current_dir = Path.cwd()
        script_dir = Path(__file__).parent
        
        # Try to find the model file in various locations
        possible_paths = [
            # First check the current working directory (project root)
            current_dir / 'stress_model.pkl',
            # Then check relative to the script location
            script_dir / 'stress_model.pkl',
            script_dir / '../stress_model.pkl',
            script_dir / '../../stress_model.pkl',
            # Check common model directories
            current_dir / 'models' / 'stress_model.pkl',
            current_dir / 'src' / 'models' / 'stress_model.pkl',
            # Check in parent directories
            current_dir.parent / 'stress_model.pkl',
        ]
        
        # Log the paths being checked for debugging
        logging.info(f"Current working directory: {current_dir}")
        logging.info(f"Script directory: {script_dir}")
        logging.info("Searching for stress_model.pkl in the following locations:")
        for path in possible_paths:
            logging.info(f"  - {path.absolute()}")
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                logging.info(f"Found model file at: {path.absolute()}")
                break
        
        if model_path is None:
            # List files in current directory for debugging
            files_in_current_dir = list(current_dir.glob('*'))
            logging.error(f"Could not find stress_model.pkl in any expected location")
            logging.error(f"Files in current directory ({current_dir}): {[f.name for f in files_in_current_dir]}")
            raise FileNotFoundError(f"Could not find stress_model.pkl. Searched in: {[str(p) for p in possible_paths]}")
        
        predictor = StressModelWrapper(model_path)
        logging.info("Stress predictor initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize predictor: {e}")
        raise

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'API is running',
        'predictor_status': 'initialized' if predictor else 'not_initialized',
        'timestamp': datetime.utcnow().isoformat()
    })

@api_bp.route('/predict', methods=['POST'])
def predict_stress():
    """Main prediction endpoint"""
    try:
        if not predictor:
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Make prediction
        result = predictor.predict_stress(data)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/features', methods=['GET'])
def get_features():
    """Get expected features for prediction"""
    try:
        if not predictor:
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized'
            }), 500
        
        features_template = predictor.get_feature_template()
        
        return jsonify({
            'success': True,
            'features': list(features_template.keys()),
            'feature_template': features_template,
            'total_features': len(features_template)
        })
        
    except Exception as e:
        logging.error(f"Features endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/validate', methods=['POST'])
def validate_features():
    """Validate features without making prediction"""
    try:
        if not predictor:
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized'
            }), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate features
        missing, extra = predictor.validate_input_features(data)
        
        return jsonify({
            'success': True,
            'valid': len(missing) == 0,
            'missing_features': list(missing),
            'extra_features': list(extra),
            'total_provided': len(data),
            'total_expected': len(predictor.feature_names)
        })
        
    except Exception as e:
        logging.error(f"Validation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.errorhandler(404)
def api_not_found(error):
    return jsonify({
        'success': False,
        'error': 'API endpoint not found'
    }), 404