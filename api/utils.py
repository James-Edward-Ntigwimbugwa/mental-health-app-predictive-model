import logging
import sys
from datetime import datetime

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    
    # Convert string level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress some noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logging.info(f"Logging setup complete - Level: {log_level}")

def log_request_info(request):
    """Log information about incoming requests"""
    logging.info(f"Request: {request.method} {request.path}")
    if request.get_json():
        logging.debug(f"Request data keys: {list(request.get_json().keys())}")

def log_prediction_result(result):
    """Log prediction results (without sensitive data)"""
    if isinstance(result, dict):
        stress_level = result.get('stress_level', 'unknown')
        confidence = result.get('confidence', 0)
        logging.info(f"Prediction result: {stress_level} (confidence: {confidence:.2f})")
    else:
        logging.info(f"Prediction completed")