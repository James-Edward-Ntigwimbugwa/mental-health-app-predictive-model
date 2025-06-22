import os
from pathlib import Path

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY') or 'default'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    MODEL_PATH = Path(__file__).resolve().parents[0] / 'models' /'stress_model'

    API_VERSION = 'v1'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()


class DevelopmentConfig(Config) :
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default' : DevelopmentConfig
}

