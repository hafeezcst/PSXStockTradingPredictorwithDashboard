"""Enhanced PSX Data Reader with additional functionality"""

import importlib.util
import os
import sys
from .data_validator import DataValidator
from .enhanced_data_processor import EnhancedDataProcessor
from typing import Optional, List, Dict
import pandas as pd
from datetime import date
import logging

# Import DataReader from numbered file
module_path = os.path.join(os.path.dirname(__file__), "01-PSX_Database_data_download_to_SQL_db_PSX.py")
spec = importlib.util.spec_from_file_location("psx_data_reader", module_path)
psx_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psx_module)
DataReader = psx_module.DataReader

class EnhancedDataReader(DataReader):
    """Enhanced version of DataReader with additional analysis capabilities"""
    
    def __init__(self, db_path=None, alt_db_path=None):
        super().__init__(db_path, alt_db_path)
        self.validator = DataValidator()
        self.processor = EnhancedDataProcessor()
        
    def get_enhanced_data(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Get data with additional validation and processing"""
        try:
            data = self.stocks(symbol, start_date, end_date)
            if not self.validator.validate(data):
                logging.warning(f"Data validation failed for {symbol}")
                return None
                
            return self.processor.enhance(data)
        except Exception as e:
            logging.error(f"Error in get_enhanced_data for {symbol}: {e}")
            return None
            
    def batch_get_enhanced_data(self, symbols: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """Batch process multiple symbols with enhanced data"""
        results = {}
        for symbol in symbols:
            data = self.get_enhanced_data(symbol, start_date, end_date)
            if data is not None:
                results[symbol] = data
        return results

    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.engine.dispose()
        self.alt_engine.dispose()
        return False