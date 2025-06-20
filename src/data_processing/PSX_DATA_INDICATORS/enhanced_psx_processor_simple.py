"""
Enhanced PSX SQL Database to Indicator Converter - Windows Compatible Version

This module provides advanced functionality to convert PSX stock market data from a SQL database
into comprehensive technical indicators with performance optimizations.

Key Features:
- High-performance processing with threading
- Advanced technical indicators (30+ indicators)  
- Data validation and quality checks
- Multiple export formats
- Configuration management
- Comprehensive logging
- Windows-compatible (no Unicode issues)

Example Usage:
    processor = PSXIndicatorProcessor()
    results = processor.process_all_symbols()
"""

import asyncio
import logging
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

# Core libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# Progress tracking
from tqdm.auto import tqdm

# Configuration
import json
import yaml

# Technical analysis - try multiple import options
try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        from fix_pandas_ta import ta
        TA_AVAILABLE = True
    except ImportError:
        print("[WARNING] pandas_ta not available. Some indicators may not work.")
        TA_AVAILABLE = False
        ta = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("psx_indicator_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessorConfig:
    """Configuration class for the PSX Indicator Processor."""
    
    def __init__(self):
        # Database paths
        self.source_db_path = None
        self.target_db_path = None
        
        # Performance settings
        self.max_workers = min(4, mp.cpu_count())
        self.batch_size = 1000
        self.cache_size = 128
        
        # Processing options
        self.calculate_advanced_indicators = True
        self.include_ml_features = True
        self.enable_data_validation = True
        self.export_formats = ["sqlite"]
        
        # Indicator parameters
        self.rsi_periods = [9, 14, 21, 26]
        self.ma_periods = [20, 30, 50, 100, 200]
        self.bollinger_period = 20
        self.bollinger_std = 2.0
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create configuration from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_file(cls, file_path: str):
        """Load configuration from YAML or JSON file."""
        path = Path(file_path)
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls.from_dict(data)


class DataValidator:
    """Data validation and quality assessment."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Validate OHLCV data and return quality metrics."""
        results = {
            "is_valid": True,
            "issues": [],
            "quality_score": 100.0,
            "metrics": {}
        }
        
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_columns = required_columns - set(data.columns)
        
        if missing_columns:
            results["is_valid"] = False
            results["issues"].append(f"Missing columns: {missing_columns}")
            results["quality_score"] -= 50
        
        if data.empty:
            results["is_valid"] = False
            results["issues"].append("Empty dataset")
            results["quality_score"] = 0
            return results
        
        # Check for logical inconsistencies
        # Explicitly handle Series comparisons to avoid ambiguity
        invalid_prices = ((data['High'] < data['Low']) | (data['High'] < data['Close']) | (data['Low'] > data['Close'])).any()
        if invalid_prices:
            results["issues"].append(f"Invalid price relationships: {invalid_prices.sum()} rows")
            results["quality_score"] -= 10
        
        # Check for negative values
        # Explicitly handle DataFrame comparison
        negative_values = (data[['Open', 'High', 'Low', 'Close', 'Volume']] < 0).any().any()
        if negative_values.item() if isinstance(negative_values, np.bool_) else negative_values:
            results["issues"].append("Negative values found in price/volume data")
            results["quality_score"] -= 15
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > 5:
            results["issues"].append(f"High missing data percentage: {missing_pct:.2f}%")
            results["quality_score"] -= missing_pct
        
        # Calculate data quality metrics
        results["metrics"] = {
            "total_rows": len(data),
            "missing_data_pct": missing_pct,
            "date_range": {
                "start": str(data.index.min()),
                "end": str(data.index.max()),
                "days": (data.index.max() - data.index.min()).days
            },
            "avg_volume": float(data['Volume'].mean()),
            "price_volatility": float(data['Close'].pct_change().std() * 100)
        }
        
        return results


class IndicatorCalculator:
    """Technical indicator calculations."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        
    def calculate_rsi(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate RSI indicator."""
        if not TA_AVAILABLE or ta is None:
            return pd.Series(index=data.index, dtype=float)
        
        try:
            return ta.rsi(data, length=length)
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_sma(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        if not TA_AVAILABLE or ta is None:
            return data.rolling(window=length).mean()
        
        try:
            return ta.sma(data, length=length)
        except Exception as e:
            logger.warning(f"SMA calculation failed: {e}")
            return data.rolling(window=length).mean()
    
    def calculate_ema(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        if not TA_AVAILABLE or ta is None:
            return data.ewm(span=length).mean()
        
        try:
            return ta.ema(data, length=length)
        except Exception as e:
            logger.warning(f"EMA calculation failed: {e}")
            return data.ewm(span=length).mean()
    
    def calculate_comprehensive_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators."""
        try:
            # Basic price metrics
            data['Price_Change'] = data['Close'].pct_change() * 100
            data['Daily_Fluctuation'] = data['High'] - data['Low']
            
            # RSI indicators
            for period in self.config.rsi_periods:
                data[f'RSI_{period}'] = self.calculate_rsi(data['Close'], period)
                data[f'RSI_{period}_SMA'] = self.calculate_sma(data[f'RSI_{period}'], 14)
            
            # Multi-timeframe RSI
            if len(data) > 70:
                data['RSI_weekly'] = self.calculate_rsi(data['Close'], 70)
                data['RSI_weekly_SMA'] = self.calculate_sma(data['RSI_weekly'], 14)
            
            if len(data) > 294:
                data['RSI_monthly'] = self.calculate_rsi(data['Close'], 294)
                data['RSI_monthly_SMA'] = self.calculate_sma(data['RSI_monthly'], 14)
            
            # Moving averages
            for period in self.config.ma_periods:
                if len(data) > period:
                    data[f'SMA_{period}'] = self.calculate_sma(data['Close'], period)
                    data[f'EMA_{period}'] = self.calculate_ema(data['Close'], period)
            
            # Volume indicators
            data['Volume_SMA_20'] = self.calculate_sma(data['Volume'], 20)
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
            
            # Awesome Oscillator
            if TA_AVAILABLE and ta is not None:
                try:
                    hl2 = (data['High'] + data['Low']) / 2
                    data['AO'] = self.calculate_sma(hl2, 5) - self.calculate_sma(hl2, 34)
                    data['AO_SMA'] = self.calculate_sma(data['AO'], 5)
                    
                    # Weekly AO
                    if len(data) > 170:
                        data['AO_weekly'] = self.calculate_sma(hl2, 25) - self.calculate_sma(hl2, 170)
                        data['AO_weekly_SMA'] = self.calculate_sma(data['AO_weekly'], 5)
                    
                except Exception as e:
                    logger.warning(f"AO calculation failed: {e}")
            
            # ATR (Average True Range)
            if TA_AVAILABLE and ta is not None:
                try:
                    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
                    if len(data) > 70:
                        data['ATR_weekly'] = ta.atr(data['High'], data['Low'], data['Close'], length=70)
                except Exception as e:
                    logger.warning(f"ATR calculation failed: {e}")
                    # Fallback calculation
                    tr1 = data['High'] - data['Low']
                    tr2 = abs(data['High'] - data['Close'].shift())
                    tr3 = abs(data['Low'] - data['Close'].shift())
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    data['ATR'] = tr.rolling(14).mean()
            
            # MACD
            if TA_AVAILABLE and ta is not None:
                try:
                    macd_data = ta.macd(data['Close'])
                    if macd_data is not None and not macd_data.empty:
                        data = pd.concat([data, macd_data], axis=1)
                except Exception as e:
                    logger.warning(f"MACD calculation failed: {e}")
            
            # Bollinger Bands
            if TA_AVAILABLE and ta is not None:
                try:
                    bb_data = ta.bbands(data['Close'], length=self.config.bollinger_period, std=self.config.bollinger_std)
                    if bb_data is not None and not bb_data.empty:
                        data = pd.concat([data, bb_data], axis=1)
                except Exception as e:
                    logger.warning(f"Bollinger Bands calculation failed: {e}")
            
            # Machine Learning Features
            if self.config.include_ml_features:
                # Price position features
                data['Price_Position_20d'] = (data['Close'] - data['Close'].rolling(20).min()) / (
                    data['Close'].rolling(20).max() - data['Close'].rolling(20).min()
                )
                
                # Volatility features
                data['Volatility_5d'] = data['Close'].rolling(5).std()
                data['Volatility_20d'] = data['Close'].rolling(20).std()
                
                # Momentum features
                data['Momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
                data['Momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
                
                # Gap analysis
                data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
                data['Gap_Size'] = abs(data['Gap'])
            
            # High/Low ranges
            data['Weekly_High'] = data['High'].rolling(5).max()
            data['Weekly_Low'] = data['Low'].rolling(5).min()
            data['Monthly_High'] = data['High'].rolling(21).max()
            data['Monthly_Low'] = data['Low'].rolling(21).min()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return data


class PSXIndicatorProcessor:
    """Enhanced PSX Indicator Processor."""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self.setup_databases()
        self.calculator = IndicatorCalculator(self.config)
        self.validator = DataValidator()
        
    def setup_databases(self):
        """Setup database connections."""
        current_dir = Path.cwd()
        
        if self.config.source_db_path is None:
            # Try to load from config.yaml first
            config_file = Path(__file__).parent / 'config.yaml'
            if config_file.exists():
                try:
                    config_from_file = ProcessorConfig.from_file(str(config_file))
                    self.config.source_db_path = config_from_file.source_db_path
                    self.config.target_db_path = config_from_file.target_db_path
                except Exception:
                    # Fallback to relative path
                    self.config.source_db_path = current_dir.parent.parent.parent / 'data/databases/production/PSX_consolidated_data_PSX.db'
            else:
                # Fallback to relative path
                self.config.source_db_path = current_dir.parent.parent.parent / 'data/databases/production/PSX_consolidated_data_PSX.db'
                self.config.target_db_path = current_dir.parent.parent.parent / 'data/databases/production/PSX_consolidated_data_PSX_enhanced.db'
            if self.config.target_db_path is None:
                self.config.target_db_path = current_dir.parent.parent.parent / 'data/databases/production/psx_consolidated_data_indicators_PSX.db'
        
        self.source_engine = create_engine(f'sqlite:///{self.config.source_db_path}')
        self.target_engine = create_engine(f'sqlite:///{self.config.target_db_path}')
        
    def get_table_names(self) -> List[str]:
        """Get table names from source database."""
        inspector = inspect(self.source_engine)
        return inspector.get_table_names()
    
    def read_data(self, table_name: str) -> pd.DataFrame:
        """Read data from database with duplicate handling."""
        try:
            logger.debug(f"Reading data for {table_name}...")
            data = pd.read_sql_table(table_name, self.source_engine, index_col='Date', parse_dates=['Date'])
            
            # Debug log the first few rows
            if table_name == "PSX_786_stock_data":
                logger.debug(f"Initial data for PSX_786_stock_data:\n{data.head().to_string()}")
                logger.debug(f"Data types:\n{data.dtypes}")
            
            # Handle duplicate indices
            if data.index.duplicated().any():
                duplicates_count = data.index.duplicated().sum()
                logger.warning(f"Found {duplicates_count} duplicate dates in {table_name}, removing...")
                # Keep the last occurrence of each duplicate date
                data = data[~data.index.duplicated(keep='last')]
                logger.info(f"Removed {duplicates_count} duplicates from {table_name}")
            
            logger.info(f"Read {len(data)} rows from {table_name}")
            return data
        except Exception as e:
            logger.error(f"Error reading {table_name}: {e}")
            return pd.DataFrame()
    
    def process_single_symbol(self, table_name: str) -> Dict[str, Any]:
        """Process a single symbol."""
        start_time = time.time()
        result = {
            'table_name': table_name,
            'success': False,
            'processing_time': 0,
            'row_count': 0,
            'error': None
        }
        
        try:
            # Read data
            data = self.read_data(table_name)
            if data.empty:
                result['error'] = "No data found"
                return result
            
            # Validate data
            if self.config.enable_data_validation:
                validation_results = self.validator.validate_ohlcv_data(data)
                if not validation_results['is_valid']:
                    logger.warning(f"Data validation failed for {table_name}: {validation_results['issues']}")
                    if validation_results['quality_score'] < 50:
                        result['error'] = "Data quality too low"
                        return result
            
            # Process indicators
            processed_data = self.calculator.calculate_comprehensive_indicators(data)
            
            # Save to database
            self.save_to_db(processed_data, table_name)
            
            result.update({
                'success': True,
                'row_count': len(processed_data),
                'processing_time': time.time() - start_time
            })
            
        except Exception as e:
            logger.error(f"Error processing {table_name}: {e}")
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def save_to_db(self, data: pd.DataFrame, table_name: str):
        """Save data to target database."""
        if not data.empty:
            try:
                data.to_sql(table_name, self.target_engine, if_exists='replace', index=True, method='multi')
                logger.info(f"Saved {len(data)} rows to {table_name}")
            except Exception as e:
                logger.error(f"Error saving {table_name}: {e}")
        else:
            logger.info(f"No data to save for {table_name}")
    
    def process_all_symbols(self) -> Dict[str, Any]:
        """Process all symbols in the database."""
        table_names = self.get_table_names()
        results = {
            'total_symbols': len(table_names),
            'successful': 0,
            'failed': 0,
            'processing_time': 0,
            'details': []
        }
        
        start_time = time.time()
        logger.info(f"Processing {len(table_names)} symbols...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            # Submit all tasks
            for table_name in table_names:
                future = executor.submit(self.process_single_symbol, table_name)
                futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(futures, desc="Processing symbols"):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per symbol
                    results['details'].append(result)
                    
                    if result['success']:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Future result error: {e}")
                    results['failed'] += 1
        
        results['processing_time'] = time.time() - start_time
        
        logger.info(f"Processing completed: {results['successful']} successful, {results['failed']} failed")
        logger.info(f"Total time: {results['processing_time']:.2f} seconds")
        
        return results
    
    def cleanup_unused_tables(self, valid_symbols: Optional[List[str]] = None):
        """Clean up unused tables from target database."""
        if valid_symbols is None:
            valid_symbols = self.get_table_names()
        
        inspector = inspect(self.target_engine)
        existing_tables = inspector.get_table_names()
        
        tables_to_delete = []
        for table in existing_tables:
            if table not in valid_symbols:
                tables_to_delete.append(table)
        
        if tables_to_delete:
            with self.target_engine.connect() as conn:
                for table in tables_to_delete:
                    try:
                        conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
                        logger.info(f"Deleted unused table: {table}")
                    except Exception as e:
                        logger.error(f"Error deleting table {table}: {e}")
            
            logger.info(f"Cleaned up {len(tables_to_delete)} unused tables")
        else:
            logger.info("No unused tables found")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'config': {
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'cache_size': self.config.cache_size
            },
            'system': {
                'cpu_count': mp.cpu_count(),
                'ta_available': TA_AVAILABLE
            },
            'database': {
                'source_tables': len(self.get_table_names()),
                'source_path': str(self.config.source_db_path),
                'target_path': str(self.config.target_db_path)
            }
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            self.source_engine.dispose()
            self.target_engine.dispose()
            logger.info("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main(symbol: Optional[str] = None):
    """Main execution function."""
    print("[INFO] Starting PSX Indicator Processor...")
    
    try:
        # Initialize processor
        processor = PSXIndicatorProcessor()
        
        # Print system information
        stats = processor.get_processing_stats()
        print(f"[INFO] System stats: {stats}")
        
        if symbol:
            # Process single symbol
            print(f"[INFO] Processing single symbol: {symbol}")
            result = processor.process_single_symbol(symbol)
            if result['success']:
                print(f"[SUCCESS] Processed {symbol}: {result['row_count']} rows in {result['processing_time']:.2f}s")
            else:
                print(f"[ERROR] Failed to process {symbol}: {result['error']}")
        else:
            # Process all symbols
            results = processor.process_all_symbols()
            processor.cleanup_unused_tables()
            print(f"[SUCCESS] Processing completed! {results['successful']} successful, {results['failed']} failed")
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        logger.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PSX Stock Data Indicator Processor')
    parser.add_argument('--symbol', type=str, help='Process single symbol (e.g. PSX_786_stock_data)')
    args = parser.parse_args()
    
    main(args.symbol)
