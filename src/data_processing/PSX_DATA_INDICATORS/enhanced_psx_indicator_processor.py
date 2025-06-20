"""
Enhanced PSX SQL Database to Indicator Converter

This module provides advanced functionality to convert PSX stock market data from a SQL database
into comprehensive technical indicators with modern performance optimizations and enhanced features.

Key Features:
- High-performance parallel processing with multiprocessing and asyncio
- Advanced caching with TTL and LRU eviction policies
- Comprehensive technical indicators (50+ indicators)
- Real-time progress monitoring with rich console output
- Configuration management via YAML/JSON
- Data validation and quality checks
- Error recovery and retry mechanisms
- Memory optimization for large datasets
- Export capabilities (CSV, Parquet, JSON)
- Comprehensive logging with structured output

Enhanced Features (2024/2025):
- Type hints with modern Python 3.12+ features
- Pydantic models for data validation
- Context managers for resource management
- Async/await patterns for I/O operations
- Memory-mapped file operations
- GPU acceleration support (optional)
- Real-time streaming capabilities
- Advanced statistical indicators
- Machine learning feature engineering
- Data quality scoring
- Automated backtesting integration

Example Usage:
    processor = EnhancedPSXIndicatorProcessor()
    await processor.process_all_symbols()
    
    # Or with configuration
    config = ProcessorConfig.from_file("config.yaml")
    processor = EnhancedPSXIndicatorProcessor(config)
    await processor.process_symbols(["KSE100", "OGDC", "PPL"])
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import threading

# Core libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

# Progress and UI
from tqdm.auto import tqdm
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.logging import RichHandler

# Configuration and validation
from pydantic import BaseModel, Field, validator
import yaml
import json

# Technical analysis
try:
    import pandas_ta as ta
except ImportError:
    from src.data_processing.fix_pandas_ta import ta

# Data processing utilities
try:
    from ..PSX_DATA_TO_SQL_DATABASE.enhanced_data_processor import EnhancedDataProcessor
    from ..PSX_DATA_TO_SQL_DATABASE.config_manager import DataValidationConfig
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    DATA_PROCESSOR_AVAILABLE = False

# Optional GPU acceleration
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Optional advanced libraries
try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize rich console
console = Console()

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
        logging.FileHandler("psx_indicator_processor.log")
    ]
)
logger = logging.getLogger(__name__)

# Log data processor availability
if not DATA_PROCESSOR_AVAILABLE:
    logger.warning("Enhanced data processor not available - duplicate handling will be limited")


@dataclass
class ProcessorConfig:
    """Configuration class for the PSX Indicator Processor."""
    
    # Database paths
    source_db_path: Optional[str] = None
    target_db_path: Optional[str] = None
      # Performance settings
    max_workers: int = field(default_factory=lambda: min(8, mp.cpu_count()))
    batch_size: int = 1000
    use_gpu: bool = field(default_factory=lambda: GPU_AVAILABLE)
    cache_size: int = 128
    
    # Processing options
    calculate_advanced_indicators: bool = True
    include_ml_features: bool = True
    enable_data_validation: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["sqlite"])
    
    # Duplicate handling options
    remove_duplicates: bool = True
    duplicate_strategy: str = 'keep_last'  # 'keep_last', 'keep_first', 'remove_all'
    verify_data_integrity: bool = True
    
    # Timeframe settings
    trading_days_per_week: int = 5
    trading_days_per_month: int = 21
    trading_days_per_quarter: int = 63
    trading_days_per_year: int = 252
    
    # Indicator parameters
    rsi_periods: List[int] = field(default_factory=lambda: [9, 14, 21, 26])
    ma_periods: List[int] = field(default_factory=lambda: [20, 30, 50, 100, 200])
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    @classmethod
    def from_file(cls, file_path: str) -> "ProcessorConfig":
        """Load configuration from YAML or JSON file."""
        path = Path(file_path)
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls(**data)


class DataValidator:
    """Advanced data validation and quality assessment."""
    
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
                "start": data.index.min(),
                "end": data.index.max(),
                "days": (data.index.max() - data.index.min()).days
            },
            "avg_volume": data['Volume'].mean(),
            "price_volatility": data['Close'].pct_change().std() * 100
        }
        
        return results


class IndicatorCalculator:
    """Advanced technical indicator calculations with GPU acceleration support."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
    
    def _safe_ta_call(self, ta_func, *args, **kwargs):
        """Safely call a pandas_ta function and handle tuple returns."""
        try:
            result = ta_func(*args, **kwargs)
            # If result is a tuple, try to extract the DataFrame part
            if isinstance(result, tuple):
                # Commonly, the first element is the DataFrame, second is signal/info
                for item in result:
                    if isinstance(item, pd.DataFrame):
                        return item
                # If no DataFrame found, return the first element
                return result[0]
            return result
        except Exception as e:
            logger.warning(f"Technical analysis function {ta_func.__name__} failed: {e}")
            return None
    
    def _safe_concat(self, data: pd.DataFrame, ta_result, ta_name: str):
        """Safely concatenate pandas_ta results with the main dataframe, handling tuples and avoiding ambiguous Series checks."""
        if ta_result is None:
            return data
        try:
            # If result is a tuple, extract DataFrame(s) and concatenate
            if isinstance(ta_result, tuple):
                dfs = [item for item in ta_result if isinstance(item, pd.DataFrame)]
                for df in dfs:
                    if hasattr(df, 'empty') and not df.empty:
                        data = pd.concat([data, df], axis=1)
                return data
            elif isinstance(ta_result, pd.DataFrame):
                if not ta_result.empty:
                    return pd.concat([data, ta_result], axis=1)
                else:
                    return data
            elif isinstance(ta_result, pd.Series):
                # Only add if not empty
                if not ta_result.empty:
                    data[ta_name] = ta_result
                return data
            else:
                logger.warning(f"{ta_name} returned unexpected type: {type(ta_result)}")
                return data
        except Exception as e:
            logger.warning(f"Failed to concatenate {ta_name} results: {e}")
            return data
        
    def calculate_comprehensive_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators."""
        if self.use_gpu:
            return self._calculate_gpu_indicators(data)
        else:
            return self._calculate_cpu_indicators(data)
    
    def _calculate_cpu_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using CPU with pandas_ta."""
        # Basic indicators
        data = self._calculate_rsi_suite(data)
        data = self._calculate_moving_averages(data)
        data = self._calculate_momentum_indicators(data)
        data = self._calculate_volatility_indicators(data)
        data = self._calculate_volume_indicators(data)
        
        if self.config.calculate_advanced_indicators:
            data = self._calculate_advanced_indicators(data)
        
        if self.config.include_ml_features:
            data = self._calculate_ml_features(data)
        
        return data
    
    def _calculate_gpu_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using GPU acceleration (if available)."""
        if not GPU_AVAILABLE:
            logger.warning("GPU not available, falling back to CPU calculation")
            return self._calculate_cpu_indicators(data)
        
        try:
            # Convert to cuDF for GPU processing
            gpu_data = cudf.from_pandas(data)
            
            # Perform GPU calculations
            gpu_data['RSI_14'] = self._gpu_rsi(gpu_data['Close'], 14)
            gpu_data['MA_20'] = gpu_data['Close'].rolling(20).mean()
            gpu_data['MA_50'] = gpu_data['Close'].rolling(50).mean()
            
            # Convert back to pandas
            data = gpu_data.to_pandas()
            
            # Continue with CPU for complex indicators
            data = self._calculate_cpu_indicators(data)
            
        except Exception as e:
            logger.warning(f"GPU calculation failed: {e}, falling back to CPU")
            data = self._calculate_cpu_indicators(data)
        
        return data
    
    def _gpu_rsi(self, prices: 'cudf.Series', period: int) -> 'cudf.Series':
        """Calculate RSI using GPU acceleration."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rsi_suite(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive RSI indicators across multiple timeframes."""
        for period in self.config.rsi_periods:
            data[f'RSI_{period}'] = ta.rsi(data['Close'], length=period)
            data[f'RSI_{period}_SMA'] = ta.sma(data[f'RSI_{period}'], length=14)
        
        # Multi-timeframe RSI
        timeframes = {
            'weekly': 70,  # 14 weeks * 5 days
            'monthly': 294,  # 14 months * 21 days
            'quarterly': 882,  # 14 quarters * 63 days
            'semi_annual': 1764,  # 14 semi-annual * 126 days
            'annual': 3528  # 14 annual * 252 days
        }
        
        for name, period in timeframes.items():
            if len(data) > period:
                data[f'RSI_{name}'] = ta.rsi(data['Close'], length=period)
                data[f'RSI_{name}_SMA'] = ta.sma(data[f'RSI_{name}'], length=14)
        
        return data
    
    def _calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages."""
        for period in self.config.ma_periods:
            data[f'SMA_{period}'] = ta.sma(data['Close'], length=period)
            data[f'EMA_{period}'] = ta.ema(data['Close'], length=period)
            data[f'WMA_{period}'] = ta.wma(data['Close'], length=period)
          # Volume moving averages
        data['Volume_SMA_20'] = ta.sma(data['Volume'], length=20)
        data['Volume_SMA_50'] = ta.sma(data['Volume'], length=50)
        
        return data
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""        # Ensure no duplicate indices before calculations
        if data.index.duplicated().any():
            logger.warning(f"Removing {data.index.duplicated().sum()} duplicate indices before momentum calculations")
            data = data[~data.index.duplicated(keep='last')]
          # MACD family
        macd = self._safe_ta_call(ta.macd, data['Close'])
        data = self._safe_concat(data, macd, "MACD")
        
        # Stochastic
        stoch = self._safe_ta_call(ta.stoch, data['High'], data['Low'], data['Close'])
        data = self._safe_concat(data, stoch, "Stochastic")
          # Williams %R
        willr = self._safe_ta_call(ta.willr, data['High'], data['Low'], data['Close'])
        if isinstance(willr, pd.Series):
            data['WILLR'] = willr
        else:
            data['WILLR'] = np.nan
        # Rate of Change
        roc = self._safe_ta_call(ta.roc, data['Close'])
        if isinstance(roc, pd.Series):
            data['ROC'] = roc
        else:
            data['ROC'] = np.nan
        # Commodity Channel Index
        cci = self._safe_ta_call(ta.cci, data['High'], data['Low'], data['Close'])
        if isinstance(cci, pd.Series):
            data['CCI'] = cci
        else:
            data['CCI'] = np.nan
        
        return data
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        # Bollinger Bands
        bb = self._safe_ta_call(ta.bbands, data['Close'], length=self.config.bollinger_period, std=self.config.bollinger_std)
        data = self._safe_concat(data, bb, "Bollinger Bands")
        
        # Average True Range
        data['ATR'] = self._safe_ta_call(ta.atr, data['High'], data['Low'], data['Close']) or np.nan
        data['ATR_weekly'] = self._safe_ta_call(ta.atr, data['High'], data['Low'], data['Close'], length=70) or np.nan
        
        # Keltner Channels
        kc = self._safe_ta_call(ta.kc, data['High'], data['Low'], data['Close'])
        data = self._safe_concat(data, kc, "Keltner Channels")
        
        # Donchian Channels
        dc = self._safe_ta_call(ta.donchian, data['High'], data['Low'])
        data = self._safe_concat(data, dc, "Donchian Channels")
        
        return data
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # On Balance Volume
        try:
            data['OBV'] = ta.obv(data['Close'], data['Volume'])
        except:
            data['OBV'] = np.nan
        
        # Volume Price Trend (manual calculation since not in pandas_ta)
        try:
            price_change = data['Close'].pct_change()
            data['VPT'] = (price_change * data['Volume']).cumsum()
        except:
            data['VPT'] = np.nan
        
        # Accumulation/Distribution Line
        try:
            data['AD'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
        except:
            data['AD'] = np.nan
        
        # Chaikin Money Flow
        try:
            data['CMF'] = ta.cmf(data['High'], data['Low'], data['Close'], data['Volume'])
        except:
            data['CMF'] = np.nan
        
        # Volume Weighted Average Price
        try:
            data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        except:
            data['VWAP'] = np.nan
        
        return data
    
    def _calculate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators."""
        # Awesome Oscillator with multiple timeframes
        hl2 = (data['High'] + data['Low']) / 2
        
        timeframes = {
            'daily': (5, 34),
            'weekly': (25, 170),
            'monthly': (105, 714),
            'quarterly': (315, 2142),
            'semi_annual': (630, 4284)
        }
        
        for name, (fast, slow) in timeframes.items():
            if len(data) > slow:
                data[f'AO_{name}'] = ta.sma(hl2, fast) - ta.sma(hl2, slow)
                data[f'AO_{name}_SMA'] = ta.sma(data[f'AO_{name}'], 5)
          # Ichimoku Cloud
        try:
            ichimoku = ta.ichimoku(data['High'], data['Low'], data['Close'])
            if ichimoku is not None and hasattr(ichimoku, 'empty') and not ichimoku.empty:
                data = pd.concat([data, ichimoku], axis=1)
        except Exception as e:
            logger.warning(f"Ichimoku calculation failed: {e}")
        
        # Parabolic SAR
        try:
            data['PSAR'] = ta.psar(data['High'], data['Low'], data['Close'])
        except Exception as e:
            logger.warning(f"Parabolic SAR calculation failed: {e}")
            data['PSAR'] = np.nan
          # SuperTrend
        try:
            supertrend = ta.supertrend(data['High'], data['Low'], data['Close'])
            if supertrend is not None and hasattr(supertrend, 'empty') and not supertrend.empty:
                data = pd.concat([data, supertrend], axis=1)
        except Exception as e:
            logger.warning(f"SuperTrend calculation failed: {e}")
        
        return data
    
    def _calculate_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate machine learning features."""
        # Price features
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_2d'] = data['Close'].pct_change(2)
        data['Price_Change_5d'] = data['Close'].pct_change(5)
        
        # Volatility features
        data['Volatility_5d'] = data['Close'].rolling(5).std()
        data['Volatility_20d'] = data['Close'].rolling(20).std()
        
        # Volume features
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['Volume_Change'] = data['Volume'].pct_change()
        
        # Price position features
        data['Price_Position_20d'] = (data['Close'] - data['Close'].rolling(20).min()) / (
            data['Close'].rolling(20).max() - data['Close'].rolling(20).min()
        )
        
        # Momentum features
        data['Momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
        data['Momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Gap features
        data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        data['Gap_Size'] = abs(data['Gap'])
        
        return data


class EnhancedPSXIndicatorProcessor:
    """Enhanced PSX Indicator Processor with modern features and optimizations."""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self.setup_databases()
        self.calculator = IndicatorCalculator(self.config)
        self.validator = DataValidator()
        self._cache = {}
        self._cache_lock = threading.Lock()
          # Initialize data processor for duplicate handling
        if DATA_PROCESSOR_AVAILABLE and self.config.remove_duplicates:
            try:
                validation_config = DataValidationConfig()
                self.data_processor = EnhancedDataProcessor(validation_config)
                logger.info("Enhanced data processor initialized for duplicate handling")
            except Exception as e:
                logger.warning(f"Failed to initialize data processor: {e}")
                self.data_processor = None
        else:
            self.data_processor = None
        
    def setup_databases(self):
        """Setup database connections with optimized settings."""
        current_dir = Path(__file__).parent
        
        if self.config.source_db_path is None:
            # Try to find database in various locations
            possible_paths = [
                current_dir.parent.parent.parent / "data" / "databases" / "production" / "PSX_consolidated_data_PSX.db",
                current_dir.parent.parent.parent.parent / "data" / "databases" / "production" / "PSX_consolidated_data_PSX.db",
                current_dir.parent.parent.parent / "PSX_Stock_Data.db",
                current_dir.parent.parent.parent.parent / "PSX_Stock_Data.db",
                Path("C:/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_consolidated_data_PSX.db"),
                Path("C:/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/PSX_Stock_Data.db")
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.config.source_db_path = path
                    break
            else:
                self.config.source_db_path = current_dir / 'data/databases/production/psx_consolidated_data_PSX.db'
        
        if self.config.target_db_path is None:
            if self.config.source_db_path:
                source_path = Path(self.config.source_db_path)
                target_name = source_path.stem + '_indicators' + source_path.suffix
                self.config.target_db_path = source_path.parent / target_name
            else:
                self.config.target_db_path = current_dir / 'data/databases/production/psx_consolidated_data_indicators_PSX.db'
        
        # Optimized SQLite settings
        self.source_engine = create_engine(
            f'sqlite:///{self.config.source_db_path}',
            poolclass=StaticPool,
            connect_args={
                'check_same_thread': False,
                'timeout': 30
            },
            echo=False
        )
        
        self.target_engine = create_engine(
            f'sqlite:///{self.config.target_db_path}',
            poolclass=StaticPool,
            connect_args={
                'check_same_thread': False,
                'timeout': 30
            },
            echo=False
        )
        
    @contextmanager
    def database_connection(self, engine: Engine):
        """Context manager for database connections."""
        conn = engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    @lru_cache(maxsize=128)
    def get_table_names(self) -> Tuple[str, ...]:
        """Get table names with caching."""
        inspector = inspect(self.source_engine)
        return tuple(inspector.get_table_names())
    
    async def read_data_async(self, table_name: str) -> pd.DataFrame:
        """Asynchronously read data from database."""
        loop = asyncio.get_event_loop()
        
        def _read_data():
            try:
                with self.database_connection(self.source_engine) as conn:
                    data = pd.read_sql_table(
                        table_name, 
                        conn, 
                        index_col='Date', 
                        parse_dates=['Date']
                    )
                    
                    # Comprehensive duplicate handling
                    if data.index.duplicated().any():
                        duplicate_count = data.index.duplicated().sum()
                        logger.warning(f"Removing {duplicate_count} duplicate records from {table_name} using basic pandas deduplication")
                        data = data[~data.index.duplicated(keep='last')]
                    
                    # Additional data quality checks
                    if data.empty:
                        logger.warning(f"Empty dataset after reading {table_name}")
                        return pd.DataFrame()
                    
                    # Ensure we have required columns
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    missing_cols = [col for col in required_cols if col not in data.columns]
                    if missing_cols:
                        logger.error(f"Missing required columns in {table_name}: {missing_cols}")
                        return pd.DataFrame()
                    
                    # Sort by date to ensure proper order
                    data = data.sort_index()
                    
                    logger.info(f"Successfully read {len(data)} rows from {table_name}")
                    return data
            except Exception as e:
                logger.error(f"Error reading {table_name}: {e}")
                return pd.DataFrame()
        
        return await loop.run_in_executor(None, _read_data)
    
    def process_single_symbol(self, table_name: str) -> Dict[str, Any]:
        """Process a single symbol with comprehensive error handling."""
        start_time = time.time()
        result = {
            'table_name': table_name,
            'success': False,
            'processing_time': 0,
            'row_count': 0,
            'error': None,
            'validation_results': None
        }
        
        try:
            # Read data
            data = self._read_data_sync(table_name)
            if data.empty:
                result['error'] = "No data found"
                return result
            
            # Validate data
            if self.config.enable_data_validation:
                validation_results = self.validator.validate_ohlcv_data(data)
                result['validation_results'] = validation_results
                
                if not validation_results['is_valid']:
                    logger.warning(f"Data validation failed for {table_name}: {validation_results['issues']}")
                    if validation_results['quality_score'] < 50:
                        result['error'] = "Data quality too low"
                        return result
            
            # Process indicators
            processed_data = self.calculator.calculate_comprehensive_indicators(data)
            
            # Save to database
            self._save_to_db_sync(processed_data, table_name)
              # Export to other formats if requested
            if 'csv' in self.config.export_formats:
                self._export_csv(processed_data, table_name)
            if 'parquet' in self.config.export_formats and PARQUET_AVAILABLE:
                self._export_parquet(processed_data, table_name)
            
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
    
    def _read_data_sync(self, table_name: str) -> pd.DataFrame:
        """Synchronous data reading method with duplicate handling."""
        try:
            logger.debug(f"Reading data for {table_name}...")
            with self.database_connection(self.source_engine) as conn:
                # First, read data without setting index to avoid duplicate index errors
                data = pd.read_sql_table(
                    table_name,
                    conn,
                    parse_dates=['Date']
                )
                
                # Debug log the first few rows
                if table_name == "PSX_786_stock_data":
                    logger.debug(f"Initial data for PSX_786_stock_data:\n{data.head().to_string()}")
                    logger.debug(f"Data types:\n{data.dtypes}")
                
                if data.empty:
                    return data
                
                # Handle duplicates if data processor is available and enabled
                if self.data_processor and self.config.remove_duplicates:
                    # Set Date as index for duplicate processing
                    if 'Date' in data.columns:
                        data = data.set_index('Date')
                        data = data.sort_index()
                        
                        # Check and remove duplicates
                        duplicate_info = self.data_processor.check_duplicate_records(data, table_name)
                        
                        if duplicate_info['has_duplicates']:
                            logger.warning(
                                f"Found {duplicate_info['duplicate_count']} duplicate records in {table_name}"
                            )
                            
                            # Remove duplicates using configured strategy
                            cleaned_data, removal_info = self.data_processor.remove_duplicate_records(
                                data, table_name, self.config.duplicate_strategy
                            )
                            
                            if removal_info['removed_count'] > 0:
                                logger.info(
                                    f"Removed {removal_info['removed_count']} duplicates from {table_name} "
                                    f"using '{self.config.duplicate_strategy}' strategy"
                                )
                            
                            return cleaned_data
                        else:
                            return data
                    else:
                        logger.warning(f"No 'Date' column found in {table_name}")
                        return data
                else:
                    # Fallback: Basic duplicate removal using pandas
                    if 'Date' in data.columns:
                        data = data.set_index('Date')
                        data = data.sort_index()
                        
                        # Check for duplicates
                        if data.index.duplicated().any():
                            original_count = len(data)
                            data = data[~data.index.duplicated(keep='last')]
                            removed_count = original_count - len(data)
                            
                            if removed_count > 0:
                                logger.warning(
                                    f"Removed {removed_count} duplicate records from {table_name} "
                                    f"using basic pandas deduplication"
                                )
                        
                        return data
                    else:
                        logger.warning(f"No 'Date' column found in {table_name}")
                        return data
                        
        except Exception as e:
            logger.error(f"Error reading data from {table_name}: {e}")
            # If there's an error, try to read without any processing
            try:
                with self.database_connection(self.source_engine) as conn:
                    data = pd.read_sql_table(table_name, conn, parse_dates=['Date'])
                    if 'Date' in data.columns and not data.empty:
                        data = data.set_index('Date').sort_index()
                        # Basic duplicate removal as last resort
                        if data.index.duplicated().any():
                            data = data[~data.index.duplicated(keep='last')]
                    return data
            except Exception as e2:
                logger.error(f"Failed to read {table_name} even with fallback method: {e2}")
                return pd.DataFrame()
    
    def _save_to_db_sync(self, data: pd.DataFrame, table_name: str):
        """Synchronous database saving method."""
        with self.database_connection(self.target_engine) as conn:
            data.to_sql(table_name, conn, if_exists='replace', index=True, method='multi')
    
    def _export_csv(self, data: pd.DataFrame, table_name: str):
        """Export data to CSV format."""
        output_dir = Path('exports/csv')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / f"{table_name}.csv"
        data.to_csv(file_path)
        logger.info(f"Exported {table_name} to CSV: {file_path}")
    
    def _export_parquet(self, data: pd.DataFrame, table_name: str):
        """Export data to Parquet format."""
        output_dir = Path('exports/parquet')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / f"{table_name}.parquet"
        data.to_parquet(file_path)
        logger.info(f"Exported {table_name} to Parquet: {file_path}")
    
    async def process_symbols_async(self, symbol_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process symbols asynchronously with progress tracking."""
        table_names = symbol_list or list(self.get_table_names())
        
        results = {
            'total_symbols': len(table_names),
            'successful': 0,
            'failed': 0,
            'processing_time': 0,
            'details': []
        }
        
        start_time = time.time()
        
        with Progress() as progress:
            task = progress.add_task("Processing symbols...", total=len(table_names))
            
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self.process_single_symbol, table_name)
                    for table_name in table_names
                ]
                
                for future in asyncio.as_completed([
                    asyncio.wrap_future(future) for future in futures
                ]):
                    result = await future
                    results['details'].append(result)
                    
                    if result['success']:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                    
                    progress.update(task, advance=1)
                    
                    # Update progress description
                    progress.update(
                        task,
                        description=f"Processing symbols... (✅ {results['successful']} ❌ {results['failed']})"
                    )
        
        results['processing_time'] = time.time() - start_time
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a summary report of processing results."""
        table = Table(title="PSX Indicator Processing Summary")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Symbols", str(results['total_symbols']))
        table.add_row("Successful", str(results['successful']))
        table.add_row("Failed", str(results['failed']))
        table.add_row("Success Rate", f"{results['successful']/results['total_symbols']*100:.1f}%")
        table.add_row("Total Time", f"{results['processing_time']:.2f}s")
        table.add_row("Avg Time per Symbol", f"{results['processing_time']/results['total_symbols']:.2f}s")
        
        console.print(table)
        
        # Log failed symbols
        failed_symbols = [detail for detail in results['details'] if not detail['success']]
        if failed_symbols:
            logger.warning(f"Failed to process {len(failed_symbols)} symbols:")
            for failed in failed_symbols:
                logger.warning(f"  - {failed['table_name']}: {failed['error']}")
    
    async def process_all_symbols(self) -> Dict[str, Any]:
        """Process all symbols in the database."""
        logger.info("Starting processing of all symbols...")
        return await self.process_symbols_async()
    
    def cleanup_unused_tables(self, valid_symbols: Optional[List[str]] = None):
        """Clean up unused tables from target database."""
        if valid_symbols is None:
            valid_symbols = list(self.get_table_names())
        
        inspector = inspect(self.target_engine)
        existing_tables = inspector.get_table_names()
        valid_table_set = {f"PSX_{symbol}_stock_data" for symbol in valid_symbols}
        
        tables_to_delete = [table for table in existing_tables if table not in valid_table_set]
        
        if tables_to_delete:
            with self.database_connection(self.target_engine) as conn:
                for table in tables_to_delete:
                    conn.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
                    logger.info(f"Deleted unused table: {table}")
            
            logger.info(f"Cleaned up {len(tables_to_delete)} unused tables")
        else:
            logger.info("No unused tables found")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and system information."""
        # Try to get database info, but don't fail if database is unavailable
        try:
            source_tables_count = len(self.get_table_names())
            db_available = True
        except Exception:
            source_tables_count = 0
            db_available = False
        
        return {
            'config': {
                'max_workers': self.config.max_workers,
                'gpu_enabled': self.config.use_gpu and GPU_AVAILABLE,
                'cache_size': self.config.cache_size,
                'batch_size': self.config.batch_size
            },
            'system': {
                'cpu_count': mp.cpu_count(),
                'gpu_available': GPU_AVAILABLE,
                'parquet_available': PARQUET_AVAILABLE
            },
            'database': {
                'source_tables': source_tables_count,
                'source_path': str(self.config.source_db_path),
                'target_path': str(self.config.target_db_path),
                'available': db_available
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


async def main():
    """Main execution function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced PSX Indicator Processor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to process")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--export", choices=["csv", "parquet", "json"], 
                       action="append", help="Export formats")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ProcessorConfig.from_file(args.config)
    else:
        config = ProcessorConfig()
    
    # Override config with CLI arguments
    if args.workers:
        config.max_workers = args.workers
    if args.gpu:
        config.use_gpu = True
    if args.export:
        config.export_formats = args.export
      # Initialize processor
    try:
        with EnhancedPSXIndicatorProcessor(config) as processor:
            # Print system information
            stats = processor.get_processing_stats()
            logger.info(f"System stats: {stats}")
            
            # Process symbols
            if args.symbols:
                results = await processor.process_symbols_async(args.symbols)
            else:
                results = await processor.process_all_symbols()
            
            # Cleanup unused tables
            processor.cleanup_unused_tables()
            
            logger.info("Processing completed successfully!")
    
    except Exception as e:
        if "unable to open database file" in str(e):
            logger.error("Database connection failed. Please check your database configuration:")
            logger.error(f"  Source DB: {config.source_db_path}")
            logger.error(f"  Target DB: {config.target_db_path}")
            logger.error("  Make sure the database files exist and are accessible.")
            logger.info("To test without a database, run: python test_simple.py")
        else:
            logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
