"""
Production configuration settings
"""

# Database settings
DATABASE = {
    'name': 'PSX_investing_Stocks_KMI30.db',
    'path': '/var/data/psx/processed/'
}

# API settings
API = {
    'base_url': 'https://api.production.example.com',
    'timeout': 60,
    'retry_attempts': 5
}

# Logging settings
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': '/var/log/psx/app.log'
}

# Model settings
MODEL = {
    'prediction_horizon': 30,  # days
    'training_window': 365,   # days
    'confidence_level': 0.95
}

# Web application settings
WEB = {
    'host': '0.0.0.0',
    'port': 80,
    'debug': False
} 