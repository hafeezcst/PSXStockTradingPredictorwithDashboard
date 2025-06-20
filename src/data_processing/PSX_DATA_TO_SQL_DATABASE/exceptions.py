"""
Custom exceptions for PSX data download operations.
"""

class PSXDataDownloadError(Exception):
    """Custom exception for PSX data download errors"""
    def __init__(self, message: str, symbol: str = None, date_range: str = None):
        self.symbol = symbol
        self.date_range = date_range
        super().__init__(message)

class DatabaseConnectionError(Exception):
    """Custom exception for database connection issues"""
    def __init__(self, message: str, db_path: str = None):
        self.db_path = db_path
        super().__init__(message)

class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    def __init__(self, message: str, symbol: str = None, validation_type: str = None):
        self.symbol = symbol
        self.validation_type = validation_type
        super().__init__(message)

class APIConnectionError(Exception):
    """Custom exception for API connection issues"""
    def __init__(self, message: str, url: str = None, status_code: int = None):
        self.url = url
        self.status_code = status_code
        super().__init__(message)

class ConfigurationError(Exception):
    """Custom exception for configuration issues"""
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(message)
