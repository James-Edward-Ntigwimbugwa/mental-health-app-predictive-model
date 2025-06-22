from flask import Flask
from flask_cors import CORS
import logging
import os
from api.routes import init_predictor, api_bp
from api.utils import setup_logging

# Configuration classes
class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    LOG_LEVEL = 'DEBUG'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def create_app(config_name=None):
    """Application factory"""
    
    # Determine config
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Setup logging
    setup_logging(app.config['LOG_LEVEL'])
    
    # Setup CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Initialize predictor
    with app.app_context():
        try:
            init_predictor()
            logging.info("Application initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize predictor: {e}")
            # Don't fail app startup, but log the error
    
    # Root route
    @app.route('/')
    def index():
        return {
            'message': 'Mental Health Stress Prediction API',
            'version': '1.0.0',
            'docs': '/api/v1/health'
        }
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'success': False, 'error': 'Endpoint not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'success': False, 'error': 'Internal server error'}, 500
    
    return app

# Create app instance for Flask CLI
app = create_app()

# For running the app directly
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)