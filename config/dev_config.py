"""
Development configuration settings
"""

# Database settings
DATABASE = {
    'name': 'PSX_investing_Stocks_KMI30.db',
    'path': 'data/processed/'
}

# API settings
API = {
    'base_url': 'https://api.example.com',
    'timeout': 30,
    'retry_attempts': 3
}

# Logging settings
LOGGING = {
    'level': 'DEBUG',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/dev.log'
}

# Model settings
MODEL = {
    'prediction_horizon': 30,  # days
    'training_window': 365,   # days
    'confidence_level': 0.95
}

# Web application settings
WEB = {
    'host': 'localhost',
    'port': 5000,
    'debug': True
} 