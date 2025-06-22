import logging
import traceback
from functools import wraps
from flask import jsonify, request
from typing import Dict, Any
import time

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def handle_errors(f):
    """Decorator to handle errors and return consistent error responses"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logging.error(f"Validation error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Validation error',
                'details': str(e)
            }), 400
        except Exception as e:
            logging.error(f"Internal server error: {str(e)}")
            logging.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'details': str(e) if logging.getLogger().level <= logging.DEBUG else None
            }), 500
    return decorated_function

def validate_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean request data"""
    required_fields = [
        'age', 'gender_Male', 'sleep_quality', 'health_status',
        'mental_health_history', 'exercise_frequency', 'substance_use',
        'social_support', 'work_stress', 'financial_stress'
    ]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Set default values for optional fields
    if 'dataset_source_workplace' not in data:
        data['dataset_source_workplace'] = 1
    
    return data

def log_request():
    """Log incoming requests"""
    logging.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    if request.is_json:
        logging.debug(f"Request data: {request.get_json()}")