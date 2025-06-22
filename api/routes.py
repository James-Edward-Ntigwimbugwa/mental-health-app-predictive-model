from flask import Blueprint, request, jsonify
from .models import PredictionRequest, PredictionResponse, ErrorResponse, HealthResponse
from .utils import handle_errors, validate_request_data, log_request
from datetime import datetime
import logging
# Import your existing model
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.predict_model import StressPredictor

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize predictor (will be done in app.py)
predictor = StressPredictor()

def init_predictor():
    """Initialize the stress predictor"""
    global predictor
    if predictor is None:
        predictor = StressPredictor()
        logging.info("Stress predictor initialized")

@api_bp.route('/health', methods=['GET'])
@handle_errors
def health_check():
    """Health check endpoint"""
    log_request()
    
    response = HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=predictor is not None,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return jsonify(response.dict())

@api_bp.route('/predict', methods=['POST'])
@handle_errors
def predict_stress():
    """Main prediction endpoint"""
    log_request()
    
    # Validate request
    if not request.is_json:
        return jsonify(ErrorResponse(
            error="Content-Type must be application/json"
        ).dict()), 400
    
    data = request.get_json()
    
    # Validate and clean data
    validated_data = validate_request_data(data)
    
    # Validate with Pydantic model
    try:
        prediction_request = PredictionRequest(**validated_data)
    except Exception as e:
        return jsonify(ErrorResponse(
            error="Invalid request data",
            details=str(e)
        ).dict()), 400
    
    # Make prediction
    if predictor is None:
        return jsonify(ErrorResponse(
            error="Model not loaded"
        ).dict()), 500
    
    # Convert to dict for predictor
    features = prediction_request.dict()
    
    # Make prediction
    result = predictor.predict_stress(features)
    
    # Format response
    response = PredictionResponse(
        success=True,
        stress_level=result['stress_level'],
        confidence=float(result['confidence']),
        problem_analysis=result['problem_analysis'],
        solutions=result['solutions'],
        comprehensive_report=result['comprehensive_report']
    )
    
    logging.info(f"Prediction completed: {result['stress_level']} (confidence: {result['confidence']:.2f})")
    
    return jsonify(response.dict())

@api_bp.route('/features', methods=['GET'])
@handle_errors
def get_features():
    """Get expected features for prediction"""
    log_request()
    
    if predictor is None:
        return jsonify(ErrorResponse(
            error="Model not loaded"
        ).dict()), 500
    
    features_template = predictor.debug_features
    
    return jsonify({
        'success': True,
        'features': list(features_template.keys()),
        'feature_template': features_template,
        'total_features': len(features_template)
    })

@api_bp.route('/validate', methods=['POST'])
@handle_errors
def validate_features():
    """Validate features without making prediction"""
    log_request()
    
    if not request.is_json:
        return jsonify(ErrorResponse(
            error="Content-Type must be application/json"
        ).dict()), 400
    
    data = request.get_json()
    
    if predictor is None:
        return jsonify(ErrorResponse(
            error="Model not loaded"
        ).dict()), 500
    
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