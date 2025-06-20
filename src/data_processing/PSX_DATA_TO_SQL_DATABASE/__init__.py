"""
PSX Data to SQL Database - Enhanced Module

This package provides comprehensive tools for downloading, processing, and managing PSX stock data.

Main Components:
- EnhancedDataReader: Main class for data operations
- DataValidator: Data quality validation
- MetricsCollector: Performance monitoring
- EnhancedDatabaseManager: Database operations
- PSXAPIClient: PSX API communication

Usage:
    from PSX_DATA_TO_SQL_DATABASE import EnhancedDataReader
    
    with EnhancedDataReader() as reader:
        data = reader.stocks('HBL', start_date, end_date)

Version: 2.0.0
Author: PSX Data Team
"""

from config_manager import AppConfig
from exceptions import (
    PSXDataDownloadError,
    DatabaseConnectionError,
    APIConnectionError,
    DataValidationError,
    ConfigurationError
)
from monitoring import MetricsCollector, HealthChecker
from data_validator import DataValidator
from enhanced_db_manager import EnhancedDatabaseManager
from api_client import PSXAPIClient
from enhanced_data_processor import EnhancedDataProcessor

# Import the main classes
try:
    from enhanced_psx_data_reader import EnhancedDataReader, DataReader
except ImportError as e:
    # Handle import error gracefully
    import logging
    logging.warning(f"Could not import main DataReader classes: {e}")
    EnhancedDataReader = None
    DataReader = None

__version__ = "2.0.0"
__author__ = "PSX Data Team"
__email__ = "data@psx.com"

__all__ = [
    'EnhancedDataReader',
    'DataReader',
    'AppConfig',
    'PSXDataDownloadError',
    'DatabaseConnectionError',
    'APIConnectionError',
    'DataValidationError',
    'ConfigurationError',
    'MetricsCollector',
    'HealthChecker',
    'DataValidator',
    'EnhancedDatabaseManager',
    'PSXAPIClient',
    'EnhancedDataProcessor'
]
