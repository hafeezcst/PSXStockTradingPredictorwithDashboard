"""
Enhanced PSX Data Processing Package

This package provides comprehensive tools for downloading, processing, and managing PSX stock data.

Main Components:
- EnhancedDataReader: Main class for data operations
- DataValidator: Data quality validation
- MetricsCollector: Performance monitoring
- DatabaseManager: Database operations
- APIClient: PSX API communication

Usage:
    from src.data_processing import EnhancedDataReader
    
    with EnhancedDataReader() as reader:
        data = reader.stocks('HBL', start_date, end_date)
"""

from .config_manager import AppConfig
from .exceptions import (
    PSXDataDownloadError,
    DatabaseConnectionError,
    APIConnectionError,
    DataValidationError,
    ConfigurationError
)
from .monitoring import MetricsCollector, HealthChecker
from .data_validator import DataValidator
from .enhanced_db_manager import EnhancedDatabaseManager
from .api_client import PSXAPIClient
from .enhanced_data_processor import EnhancedDataProcessor

# Import the main classes
try:
    from .enhanced_psx_data_reader import EnhancedDataReader
    
    # Import DataReader from numbered file using importlib
    import importlib.util
    import sys
    import os
    
    module_path = os.path.join(os.path.dirname(__file__), "01-PSX_Database_data_download_to_SQL_db_PSX.py")
    spec = importlib.util.spec_from_file_location("psx_data_reader", module_path)
    psx_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(psx_module)
    DataReader = psx_module.DataReader
    
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
