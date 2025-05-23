import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine, QueuePool
import sqlite3
import os
import logging
from datetime import datetime, timedelta
from telegram_message import send_telegram_message_with_image
from telegram_message import send_telegram_message
from tabulate import tabulate
import numpy as np
from collections import Counter
import time
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy.engine import Engine
from dataclasses import dataclass
from pathlib import Path
import traceback
import requests
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv, find_dotenv, dotenv_values
from dataclasses import dataclass
from typing import Generator, Optional, List, Tuple, Dict, Any
import re

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging with enhanced formatting and file rotation."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create formatters
    file_formatter = logging.Formatter(log_format, date_format)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / 'stock_analysis.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info("Logging system initialized")
    return logger

def parse_env_file(file_path: Path) -> Dict[str, str]:
    """Custom parser for .env file with better error handling."""
    env_vars = {}
    if not file_path.exists():
        return env_vars
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Parse key-value pairs
                match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
                if match:
                    key, value = match.groups()
                    # Remove quotes if present
                    value = value.strip().strip('"\'')
                    env_vars[key] = value
                else:
                    logger.warning(f"Invalid line format in .env file at line {line_num}: {line}")
                    
    except Exception as e:
        logger.error(f"Error reading .env file: {str(e)}")
        
    return env_vars

def load_environment_variables() -> Dict[str, Any]:
    """Load and validate environment variables with proper error handling."""
    try:
        # Define default values
        defaults = {
            'MAIN_DB_PATH': 'data/databases/production/psx_consolidated_data_indicators_PSX.db',
            'SIGNALS_DB_PATH': 'data/databases/production/PSX_investing_Stocks_KMI30.db',
            'CHARTS_DIR': 'outputs/charts/RSI_AO_CHARTS',
            'DASHBOARDS_DIR': 'outputs/dashboards/PSX_DASHBOARDS',
            'RSI_OVERSOLD': '40',
            'RSI_OVERBOUGHT': '60',
            'MA_PERIOD': '30',
            'NEUTRAL_THRESHOLD': '1.5',
            'MAX_POSSIBLE_SCORE': '10.0',
            'CHART_DPI': '120',
            'CHART_FIGSIZE': '14,14',
            'TELEGRAM_BOT_TOKEN': '',
            'TELEGRAM_CHAT_ID': ''
        }
        
        # Try to find .env file in parent directories
        env_path = find_dotenv(raise_error_if_not_found=False)
        env_vars = defaults.copy()
        
        if env_path:
            logger.info(f"Found .env file at: {env_path}")
            # Use custom parser
            custom_env_vars = parse_env_file(Path(env_path))
            if custom_env_vars:
                env_vars.update(custom_env_vars)
                logger.info("Successfully loaded environment variables from .env file")
            else:
                logger.warning("No valid environment variables found in .env file. Using defaults.")
        else:
            logger.warning("No .env file found. Using default values.")
        
        # Validate and convert values
        for key, value in env_vars.items():
            if key in ['RSI_OVERSOLD', 'RSI_OVERBOUGHT', 'MA_PERIOD', 'CHART_DPI']:
                try:
                    env_vars[key] = str(int(value))
                except ValueError:
                    logger.warning(f"Invalid integer value for {key}: {value}. Using default.")
                    env_vars[key] = defaults[key]
            elif key in ['NEUTRAL_THRESHOLD', 'MAX_POSSIBLE_SCORE']:
                try:
                    env_vars[key] = str(float(value))
                except ValueError:
                    logger.warning(f"Invalid float value for {key}: {value}. Using default.")
                    env_vars[key] = defaults[key]
            elif key == 'CHART_FIGSIZE':
                try:
                    # Ensure it's in the correct format
                    parts = value.split(',')
                    if len(parts) == 2:
                        int(parts[0])
                        int(parts[1])
                        env_vars[key] = value
                    else:
                        raise ValueError("Invalid format")
                except (ValueError, IndexError):
                    logger.warning(f"Invalid CHART_FIGSIZE value: {value}. Using default.")
                    env_vars[key] = defaults[key]
        
        return env_vars
        
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")
        raise

# Initialize logging system
logger = setup_logging()

# Load environment variables
try:
    env_vars = load_environment_variables()
except Exception as e:
    logger.error(f"Failed to load environment variables: {str(e)}")
    raise

@dataclass
class Configuration:
    """Configuration settings for the application."""
    
    # Database paths
    MAIN_DB_PATH: str = env_vars['MAIN_DB_PATH']
    SIGNALS_DB_PATH: str = env_vars['SIGNALS_DB_PATH']
    
    # Output directories
    CHARTS_DIR: str = env_vars['CHARTS_DIR']
    DASHBOARDS_DIR: str = env_vars['DASHBOARDS_DIR']
    
    # Technical analysis parameters
    RSI_OVERSOLD: int = int(env_vars['RSI_OVERSOLD'])
    RSI_OVERBOUGHT: int = int(env_vars['RSI_OVERBOUGHT'])
    MA_PERIOD: int = int(env_vars['MA_PERIOD'])
    
    # Market phase thresholds
    NEUTRAL_THRESHOLD: float = float(env_vars['NEUTRAL_THRESHOLD'])
    MAX_POSSIBLE_SCORE: float = float(env_vars['MAX_POSSIBLE_SCORE'])
    
    # Chart settings
    CHART_DPI: int = int(env_vars['CHART_DPI'])
    CHART_FIGSIZE: tuple = tuple(map(int, env_vars['CHART_FIGSIZE'].split(',')))
    
    # Telegram settings
    TELEGRAM_BOT_TOKEN: str = env_vars['TELEGRAM_BOT_TOKEN']
    TELEGRAM_CHAT_ID: str = env_vars['TELEGRAM_CHAT_ID']
    
    def __post_init__(self):
        """Create required directories and validate configuration."""
        # Create required directories
        for directory in [self.CHARTS_DIR, self.DASHBOARDS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Validate technical parameters
        if self.RSI_OVERSOLD >= self.RSI_OVERBOUGHT:
            raise ValueError("RSI_OVERSOLD must be less than RSI_OVERBOUGHT")
        
        if self.MA_PERIOD <= 0:
            raise ValueError("MA_PERIOD must be positive")
        
        if self.NEUTRAL_THRESHOLD <= 0 or self.MAX_POSSIBLE_SCORE <= 0:
            raise ValueError("Thresholds must be positive")
        
        # Validate Telegram settings
        if not self.TELEGRAM_BOT_TOKEN or not self.TELEGRAM_CHAT_ID:
            logger.warning("Telegram bot token or chat ID not configured. Telegram notifications will be disabled.")
        else:
            logger.info("Telegram credentials are configured")
            
        # Validate database paths
        for db_path in [self.MAIN_DB_PATH, self.SIGNALS_DB_PATH]:
            if not os.path.exists(os.path.dirname(db_path)):
                raise ValueError(f"Database directory does not exist: {os.path.dirname(db_path)}")

# Create global configuration instance
config = Configuration()

class DatabaseConnectionManager:
    """Enhanced database connection manager with connection pooling and better error handling."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._engine = None
        self._pool = None
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        """Initialize the database connection pool."""
        try:
            self._engine = create_engine(
                f'sqlite:///{self.db_path}',
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800
            )
            logger.info(f"Database connection pool initialized for {self.db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database connection pool: {str(e)}")
    
    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine instance."""
        if not self._engine:
            self._initialize_connection_pool()
        return self._engine
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection from the pool with automatic cleanup."""
        connection = None
        try:
            connection = self.engine.raw_connection()
            yield connection
        except Exception as e:
            raise DatabaseError(f"Failed to get database connection: {str(e)}")
        finally:
            if connection:
                connection.close()
    
    def execute_query(self, query: str, params: tuple = None) -> list:
        """Execute a query with retry logic and better error handling."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    return cursor.fetchall()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise DatabaseError(f"Database operation failed: {str(e)}")
            except Exception as e:
                raise DatabaseError(f"Unexpected database error: {str(e)}")
    
    def execute_many(self, query: str, params_list: list[tuple]) -> None:
        """Execute multiple queries with transaction support."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                conn.execute("BEGIN TRANSACTION")
                try:
                    cursor.executemany(query, params_list)
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise DatabaseError(f"Failed to execute batch query: {str(e)}")
        except Exception as e:
            raise DatabaseError(f"Failed to execute batch operation: {str(e)}")
    
    def close(self):
        """Close all database connections in the pool."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connection pool closed")

# Create database connection managers
main_db = DatabaseConnectionManager(config.MAIN_DB_PATH)
signals_db = DatabaseConnectionManager(config.SIGNALS_DB_PATH)

def send_signals_and_charts_summary(buy_df, sell_df, available_symbols, total_processed):
    """Generate a summary of signals and charts processed"""
    try:
        message = "📊 PSX SIGNALS AND CHARTS SUMMARY 📊\n\n"
        
        # Buy signals summary
        buy_symbols = buy_df['Stock'].unique().tolist() if not buy_df.empty else []
        message += f"✅ BUY SIGNALS: {len(buy_symbols)} stocks\n"
        message += f"📈 CHARTS PROCESSED: {total_processed} charts\n"
        
        # Sell signals summary
        sell_symbols = sell_df['Stock'].unique().tolist() if not sell_df.empty else []
        message += f"❌ SELL SIGNALS: {len(sell_symbols)} stocks\n\n"
        
        # Add chart locations info
        message += "🗂️ Charts have been saved in the RSI_AO_CHARTS folder\n"
        
        # Market breadth indicators
        if len(available_symbols) > 0:
            buy_percentage = len(buy_symbols) / len(available_symbols) * 100
            sell_percentage = len(sell_symbols) / len(available_symbols) * 100
            
            message += f"📈 MARKET BREADTH INDICATORS:\n"
            message += f"   - Buy Signals: {buy_percentage:.2f}% of tracked stocks\n"
            message += f"   - Sell Signals: {sell_percentage:.2f}% of tracked stocks\n"
            
            # Simple market interpretation
            if buy_percentage > sell_percentage * 2:
                message += "\n🟢 MARKET INTERPRETATION: Strongly Bullish\n"
            elif buy_percentage > sell_percentage:
                message += "\n🟡 MARKET INTERPRETATION: Moderately Bullish\n"
            elif sell_percentage > buy_percentage * 2:
                message += "\n🔴 MARKET INTERPRETATION: Strongly Bearish\n"
            elif sell_percentage > buy_percentage:
                message += "\n🟠 MARKET INTERPRETATION: Moderately Bearish\n"
            else:
                message += "\n⚪ MARKET INTERPRETATION: Neutral\n"
        
        return message
    except Exception as e:
        logger.error(f"Error generating signals summary: {e}")
        return "Error generating summary"

def get_latest_sell_stocks() -> pd.DataFrame:
    """Get the latest sell stocks from the database.
    
    Returns:
        pd.DataFrame: DataFrame containing the latest sell signals with columns:
            - Stock: Stock symbol
            - Date: Current date
            - Close: Current closing price
            - RSI_Weekly_Avg: Weekly RSI average
            - AO_Weekly: Weekly Awesome Oscillator
            - Signal_Date: Date of sell signal
            - Signal_Close: Price at sell signal
            - update_date: Last update date (if available)
            - days_ago: Days since sell signal
    """
    try:
        with sqlite3.connect(config.SIGNALS_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # First check if the sell_stocks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sell_stocks'")
            if not cursor.fetchone():
                logger.warning("Table 'sell_stocks' does not exist in the database")
                return pd.DataFrame()
            
            cursor.execute("PRAGMA table_info(sell_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            logger.info(f"Actual columns in sell_stocks: {columns}")
            
            # Get a list of all unique stocks
            cursor.execute("SELECT DISTINCT Stock FROM sell_stocks WHERE Signal_Date IS NOT NULL")
            stocks = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(stocks)} unique stocks with sell signal dates")
            
            # For each stock, get the most recent signal
            results = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for stock in stocks:
                if 'update_date' in columns:
                    # If update_date exists, use it to find the most recent entry
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close, 
                              update_date, julianday('{current_date}') - julianday(Signal_Date) AS days_ago
                        FROM sell_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY update_date DESC, Signal_Date DESC
                        LIMIT 1
                    """
                else:
                    # Otherwise just use Signal_Date
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close,
                              julianday('{current_date}') - julianday(Signal_Date) AS days_ago
                        FROM sell_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY Signal_Date DESC
                        LIMIT 1
                    """
                
                cursor.execute(query, (stock,))
                row = cursor.fetchone()
                
                if row:
                    results.append(row)
            
            # Convert the results to a DataFrame
            if 'update_date' in columns:
                column_names = ['Stock', 'Date', 'Close', 'RSI_Weekly_Avg', 'AO_Weekly', 
                               'Signal_Date', 'Signal_Close', 'update_date', 'days_ago']
            else:
                column_names = ['Stock', 'Date', 'Close', 'RSI_Weekly_Avg', 'AO_Weekly', 
                               'Signal_Date', 'Signal_Close', 'days_ago']
                
            df = pd.DataFrame(results, columns=column_names)
            
            # Sort by the most recent update_date first, then by days_ago
            if 'update_date' in columns and not df.empty:
                df['update_date'] = pd.to_datetime(df['update_date'])
                df = df.sort_values(['update_date', 'days_ago'], ascending=[False, True])
            else:
                df = df.sort_values('days_ago')
                
            # Convert days_ago to integer
            if not df.empty:
                df['days_ago'] = df['days_ago'].astype(int)
                
            return df
            
    except Exception as e:
        logger.error(f"Error getting latest sell stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# 1. UTILITY FUNCTIONS
def execute_with_retry(func, *args, max_retries=3, delay=1, backoff_factor=2, **kwargs):
    """Execute a function with exponential backoff retry logic for database operations"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if attempt < max_retries - 1:
                wait_time = delay * (backoff_factor ** attempt)
                logger.warning(f"Operation failed, retrying in {wait_time:.2f}s ({attempt+1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                raise

def check_database_files():
    """Check if required database files exist and are accessible"""
    required_dbs = {
        config.MAIN_DB_PATH: 'Main stock data database',
        config.SIGNALS_DB_PATH: 'Signals database'
    }
    
    missing_dbs = []
    for db_file, description in required_dbs.items():
        if not os.path.exists(db_file):
            missing_dbs.append(f"- {db_file}: {description}")
        else:
            # Check if file is readable
            try:
                with open(db_file, 'rb') as f:
                    # Just read a small part to check access
                    f.read(100)
                # Try to open a sqlite connection to verify integrity
                try:
                    conn = sqlite3.connect(db_file)
                    conn.execute("SELECT 1")
                    conn.close()
                except sqlite3.Error as e:
                    missing_dbs.append(f"- {db_file}: File exists but appears to be corrupted ({e})")
            except IOError as e:
                missing_dbs.append(f"- {db_file}: File exists but cannot be read (permission denied: {e})")
    
    if missing_dbs:
        logger.error("\n⚠️ DATABASE ERROR ⚠️")
        logger.error("The following required database files are missing or inaccessible:")
        for msg in missing_dbs:
            logger.error(msg)
        logger.error("\nTo use this script, please ensure:")
        logger.error("1. You've downloaded the latest database files from the repository")
        logger.error("2. The database files are in the same directory as this script")
        logger.error("3. You have read/write permissions for these files")
        return False
        
    return True

def create_default_symbols_file():
    """Create a default KMI30 symbols file if it doesn't exist"""
    try:
        file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        
        # Check if file already exists
        if os.path.exists(file_path):
            logger.info(f"Symbols file already exists at {file_path}")
            return True
            
        # Default KMI30 symbols (as of March 2024)
        kmi30_symbols = [
            'AICL', 'ATRL', 'BAFL', 'BAHL', 'CNERGY', 'EFERT', 'ENGRO', 
            'FFBL', 'FFC', 'FCCL', 'HUBC', 'HBL', 'ISL', 'ILP', 'LUCK', 
            'MCB', 'MARI', 'MEBL', 'MLCF', 'MTL', 'NBP', 'NML', 'OGDC', 
            'PAKT', 'PPL', 'PIOC', 'PSO', 'SNGP', 'SSGC', 'UBL'
        ]
        
        # Create DataFrame and save to Excel
        symbols_df = pd.DataFrame(kmi30_symbols, columns=['Symbol'])
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            symbols_df.to_excel(writer, sheet_name='KMI30', index=False)
            
        logger.info(f"Created default symbols file at {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating symbols file: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PSX Stock Analysis Tool')
    parser.add_argument('--dashboard-only', action='store_true', 
                       help='Only generate the dashboard, skip individual charts')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to analyze')
    return parser.parse_args()

# 2. DATABASE INTERACTION FUNCTIONS
def fetch_table_names(cursor):
    """Fetch stock tables from the database without requiring Excel file"""
    try:
        # Get all tables from database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Filter tables that match the PSX stock data pattern
        stock_tables = [table[0] for table in tables 
                      if table[0].startswith('PSX_') 
                      and table[0].endswith('_stock_data')]
        
        if not stock_tables:
            logger.warning("No stock tables found in database")
        else:
            logger.info(f"Fetched {len(stock_tables)} stock tables")
            
        return stock_tables
    except Exception as e:
        logger.error(f"Error fetching table names: {e}")
        return []

def fetch_column_names(engine, table_name):
    """Fetch column names from the specified table"""
    try:
        query = f"PRAGMA table_info({table_name})"
        df = pd.read_sql(query, engine)
        columns = df['name'].tolist()
        logger.info(f"Columns in table {table_name}: {columns}")
        return columns
    except Exception as e:
        logger.error(f"Error fetching column names for table {table_name}: {e}")
        return []

def get_available_symbols(cursor):
    """Get a list of available stock symbols from the database"""
    try:
        table_names = fetch_table_names(cursor)
        
        # Define common words that should NOT be treated as stock symbols
        excluded_terms = ['STOCK_DATA', 'META', 'SYSTEM', 'DATA', 'INDEX', 'CONFIG', 'TEMP', 'BACKUP']
        
        symbols = []
        for table_name in table_names:
            # Extract symbol from table name
            symbol = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
            
            # Skip if it's a system table or common word rather than a stock symbol
            if (symbol in excluded_terms or 
                len(symbol) > 10 or  # Most stock symbols aren't this long
                '_' in symbol):      # Real stock symbols typically don't have underscores
                logger.info(f"Skipping non-stock table: {table_name}")
                continue
                
            symbols.append(symbol)
            
        return symbols
    except Exception as e:
        logger.error(f"Error getting available symbols: {e}")
        print(f"\n⚠️ Database access error: {e}")
        print("Please check if your database file is valid and not corrupted.")
        return []

def get_latest_buy_stocks():
    """Get the latest buy stocks from the database"""
    try:
        with sqlite3.connect(config.SIGNALS_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # First check if the buy_stocks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='buy_stocks'")
            if not cursor.fetchone():
                logger.warning("Table 'buy_stocks' does not exist in the database")
                return pd.DataFrame()
            
            cursor.execute("PRAGMA table_info(buy_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            logger.info(f"Actual columns in buy_stocks: {columns}")
            
            # Get a list of all unique stocks
            cursor.execute("SELECT DISTINCT Stock FROM buy_stocks WHERE Signal_Date IS NOT NULL")
            stocks = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(stocks)} unique stocks with buy signal dates")
            
            # For each stock, get the most recent signal
            results = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for stock in stocks:
                if 'update_date' in columns:
                    # If update_date exists, use it to find the most recent entry
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close, 
                              update_date, julianday('{current_date}') - julianday(Signal_Date) AS holding_days
                        FROM buy_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY update_date DESC, Signal_Date DESC
                        LIMIT 1
                    """
                else:
                    # Otherwise just use Signal_Date
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close,
                              julianday('{current_date}') - julianday(Signal_Date) AS holding_days
                        FROM buy_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY Signal_Date DESC
                        LIMIT 1
                    """
                
                cursor.execute(query, (stock,))
                row = cursor.fetchone()
                
                if row:
                    results.append(row)
            
            # Convert the results to a DataFrame
            if 'update_date' in columns:
                column_names = ['Stock', 'Date', 'Close', 'RSI_Weekly_Avg', 'AO_Weekly', 
                               'Signal_Date', 'Signal_Close', 'update_date', 'holding_days']
            else:
                column_names = ['Stock', 'Date', 'Close', 'RSI_Weekly_Avg', 'AO_Weekly', 
                               'Signal_Date', 'Signal_Close', 'holding_days']
                
            df = pd.DataFrame(results, columns=column_names)
            
            # Sort by the most recent update_date first, then by holding_days
            if 'update_date' in columns and not df.empty:
                df['update_date'] = pd.to_datetime(df['update_date'])
                df = df.sort_values(['update_date', 'holding_days'], ascending=[False, True])
            else:
                df = df.sort_values('holding_days')
                
            # Convert holding_days to integer
            if not df.empty:
                df['holding_days'] = df['holding_days'].astype(int)
                
            return df
            
    except Exception as e:
        logger.error(f"Error getting latest buy stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_latest_sell_stocks():
    """Get the latest sell stocks from the database"""
    try:
        with sqlite3.connect(config.SIGNALS_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # First check if the sell_stocks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sell_stocks'")
            if not cursor.fetchone():
                logger.warning("Table 'sell_stocks' does not exist in the database")
                return pd.DataFrame()
            
            cursor.execute("PRAGMA table_info(sell_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            logger.info(f"Actual columns in sell_stocks: {columns}")
            
            # Get a list of all unique stocks
            cursor.execute("SELECT DISTINCT Stock FROM sell_stocks WHERE Signal_Date IS NOT NULL")
            stocks = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(stocks)} unique stocks with sell signal dates")
            
            # For each stock, get the most recent signal
            results = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for stock in stocks:
                if 'update_date' in columns:
                    # If update_date exists, use it to find the most recent entry
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close, 
                              update_date, julianday('{current_date}') - julianday(Signal_Date) AS days_ago
                        FROM sell_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY update_date DESC, Signal_Date DESC
                        LIMIT 1
                    """
                else:
                    # Otherwise just use Signal_Date
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close,
                              julianday('{current_date}') - julianday(Signal_Date) AS days_ago
                        FROM sell_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY Signal_Date DESC
                        LIMIT 1
                    """
                
                cursor.execute(query, (stock,))
                row = cursor.fetchone()
                
                if row:
                    results.append(row)
            
            # Convert the results to a DataFrame
            if 'update_date' in columns:
                column_names = ['Stock', 'Date', 'Close', 'RSI_Weekly_Avg', 'AO_Weekly', 
                               'Signal_Date', 'Signal_Close', 'update_date', 'days_ago']
            else:
                column_names = ['Stock', 'Date', 'Close', 'RSI_Weekly_Avg', 'AO_Weekly', 
                               'Signal_Date', 'Signal_Close', 'days_ago']
                
            df = pd.DataFrame(results, columns=column_names)
            
            # Sort by the most recent update_date first, then by days_ago
            if 'update_date' in columns and not df.empty:
                df['update_date'] = pd.to_datetime(df['update_date'])
                df = df.sort_values(['update_date', 'days_ago'], ascending=[False, True])
            else:
                df = df.sort_values('days_ago')
                
            # Convert days_ago to integer
            if not df.empty:
                df['days_ago'] = df['days_ago'].astype(int)
                
            return df
            
    except Exception as e:
        logger.error(f"Error getting latest sell stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def format_signals_for_telegram(signal_df, signal_type="BUY"):
    """Format signals dataframe for Telegram message"""
    if signal_df.empty:
        return f"No {signal_type.lower()} signals found."
    
    try:
        # Create a copy to avoid modifying original dataframe
        display_df = signal_df.copy()
        
        # Format columns based on signal type
        if signal_type == "BUY":
            if 'update_date' in display_df.columns:
                display_df = display_df[['Stock', 'update_date', 'Signal_Date', 'Signal_Close', 'RSI_Weekly_Avg', 'AO_Weekly', 'holding_days']].copy()
                display_df['update_date'] = pd.to_datetime(display_df['update_date']).dt.strftime('%Y-%m-%d')
                display_df.columns = ['Symbol', 'Updated', 'Buy Date', 'Buy Price', 'RSI', 'AO', 'Days Held']
            else:
                display_df = display_df[['Stock', 'Signal_Date', 'Signal_Close', 'RSI_Weekly_Avg', 'AO_Weekly', 'holding_days']].copy()
                display_df.columns = ['Symbol', 'Buy Date', 'Buy Price', 'RSI', 'AO', 'Days Held']
        else:  # SELL signals
            if 'update_date' in display_df.columns:
                display_df = display_df[['Stock', 'update_date', 'Signal_Date', 'Signal_Close', 'RSI_Weekly_Avg', 'AO_Weekly', 'days_ago']].copy()
                display_df['update_date'] = pd.to_datetime(display_df['update_date']).dt.strftime('%Y-%m-%d')
                display_df.columns = ['Symbol', 'Updated', 'Sell Date', 'Sell Price', 'RSI', 'AO', 'Days Ago']
            else:
                display_df = display_df[['Stock', 'Signal_Date', 'Signal_Close', 'RSI_Weekly_Avg', 'AO_Weekly', 'days_ago']].copy()
                display_df.columns = ['Symbol', 'Sell Date', 'Sell Price', 'RSI', 'AO', 'Days Ago']
        
        # Format date columns
        date_columns = [col for col in display_df.columns if 'Date' in col]
        for col in date_columns:
            display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d')
        
        # Format numeric columns
        numeric_cols = ['RSI', 'AO']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        price_cols = ['Buy Price', 'Sell Price']
        for col in price_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # Add index numbers
        display_df.insert(0, '#', range(1, len(display_df) + 1))
        
        # Format the table using tabulate
        table = tabulate(display_df, headers='keys', tablefmt='plain', showindex=False)
        
        # Add header
        header = f"🔔 ALL {signal_type} SIGNALS (SORTED BY UPDATE DATE) 🔔\n"
        header += "=" * 50 + "\n"
        
        # Add footer with count
        footer = f"\nTotal {signal_type} Signals: {len(display_df)}"
        
        # Combine all parts
        formatted_message = f"```\n{header}{table}\n{footer}\n```"
        
        return formatted_message
    
    except Exception as e:
        logger.error(f"Error formatting {signal_type} signals for Telegram: {e}")
        return f"Error formatting {signal_type} signals for Telegram: {e}"

def get_buy_sell_signals(symbol: str) -> tuple:
    """Get buy and sell signals for a stock"""
    try:
        # Get buy signals
        buy_signals = []
        with sqlite3.connect(config.SIGNALS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT Signal_Date, Signal_Close
                FROM buy_stocks
                WHERE Stock = ?
                ORDER BY Signal_Date DESC
            """, (symbol,))
            for date, price in cursor.fetchall():
                buy_signals.append({
                    'date': date,
                    'price': price
                })
                
        # Get sell signals
        sell_signals = []
        with sqlite3.connect(config.SIGNALS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT Signal_Date, Signal_Close
                FROM sell_stocks
                WHERE Stock = ?
                ORDER BY Signal_Date DESC
            """, (symbol,))
            for date, price in cursor.fetchall():
                sell_signals.append({
                    'date': date,
                    'price': price
                })
                
        logger.info(f"Found {len(buy_signals)} buy signals and {len(sell_signals)} sell signals for {symbol}")
        return buy_signals, sell_signals
        
    except Exception as e:
        logger.error(f"Error getting buy/sell signals for {symbol}: {e}")
        return [], []

def remove_duplicate_buy_stocks():
    """Remove duplicate entries from the buy_stocks table, keeping only the most recent signal for each stock"""
    try:
        with sqlite3.connect(config.SIGNALS_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # First, check if the table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='buy_stocks'")
            if not cursor.fetchone():
                logger.warning("Table 'buy_stocks' does not exist in the database")
                return 0
            
            # Count total records before cleaning
            cursor.execute("SELECT COUNT(*) FROM buy_stocks")
            total_before = cursor.fetchone()[0]
            
            # Get list of all stocks with duplicates
            cursor.execute("""
                SELECT Stock, COUNT(*) as count 
                FROM buy_stocks 
                GROUP BY Stock 
                HAVING count > 1
            """)
            duplicated_stocks = cursor.fetchall()
            
            if not duplicated_stocks:
                logger.info("No duplicate stocks found in buy_stocks table")
                return 0
                
            total_removed = 0
            
            # For each stock with duplicates, keep only the most recent entry
            # We only consider Signal_Date for determining the most recent (ignoring update_date)
            for stock, count in duplicated_stocks:
                cursor.execute("""
                    DELETE FROM buy_stocks 
                    WHERE rowid NOT IN (
                        SELECT rowid FROM buy_stocks
                        WHERE Stock = ?
                        ORDER BY Signal_Date DESC
                        LIMIT 1
                    )
                    AND Stock = ?
                """, (stock, stock))
                
                removed = count - 1  # We keep 1 record, remove the rest
                total_removed += removed
                logger.info(f"Removed {removed} duplicate entries for {stock}")
            
            conn.commit()
            
            # Count total records after cleaning
            cursor.execute("SELECT COUNT(*) FROM buy_stocks")
            total_after = cursor.fetchone()[0]
            
            logger.info(f"Cleaning complete. Records before: {total_before}, after: {total_after}, removed: {total_removed}")
            
            return total_removed
            
    except Exception as e:
        logger.error(f"Error removing duplicate buy stocks: {e}")
        return 0

# 3. ANALYSIS FUNCTIONS
def prepare_analysis_data(df):
    """Prepare dataframe for market phase analysis"""
    analysis_df = df.copy().sort_values('Date')
    
    # Calculate percentage change if it doesn't exist
    if 'pct_change' not in analysis_df.columns:
        analysis_df['pct_change'] = analysis_df['Close'].pct_change() * 100
        
    # Use recent data for analysis (last ~260 trading days / 1 year)
    recent_df = analysis_df.tail(260)
    
    return recent_df

def get_ao_change_dates(df):
    """Get the dates when AO changed from negative to positive and positive to negative"""
    change_dates = {'positive_to_negative': [], 'negative_to_positive': []}
    previous_ao = None

    for index, row in df.iterrows():
        ao_weekly = row['AO_weekly_AVG']
        if previous_ao is not None:
            if previous_ao < 0 <= ao_weekly:
                change_dates['negative_to_positive'].append((row['Date'], row['Close']))
            elif previous_ao > 0 >= ao_weekly:
                change_dates['positive_to_negative'].append((row['Date'], row['Close']))
        previous_ao = ao_weekly

    return change_dates

def analyze_rsi_trend(analysis_df):
    """Analyze RSI trends to detect accumulation/distribution patterns"""
    try:
        # Get recent RSI values (approximately 4 weeks of data)
        recent_rsi = analysis_df['RSI_weekly'].tail(20).values
        
        # Calculate trend
        rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0] if len(recent_rsi) > 1 else 0
        
        # Get longer-term RSI trend for context (approximately 12 weeks)
        longer_rsi = analysis_df['RSI_weekly'].tail(60).values
        longer_rsi_trend = np.polyfit(range(len(longer_rsi)), longer_rsi, 1)[0] if len(longer_rsi) > 2 else 0
        
        # RSI current value
        current_rsi = recent_rsi[-1] if len(recent_rsi) > 0 else 50
        
        # Assign score based on RSI values and trends
        rsi_score = 0
        
        if current_rsi < config.RSI_OVERSOLD and rsi_trend > 0.1:  # Strong oversold with uptrend
            rsi_score = 2.5
        elif current_rsi < 40 and rsi_trend > 0.05:  # Oversold with modest uptrend
            rsi_score = 2
        elif current_rsi > config.RSI_OVERBOUGHT and rsi_trend < -0.1:  # Strong overbought with downtrend
            rsi_score = -2.5
        elif current_rsi > 60 and rsi_trend < -0.05:  # Overbought with modest downtrend
            rsi_score = -2
        elif current_rsi < 45 and rsi_trend > 0.03:  # Below midline with uptrend
            rsi_score = 1
        elif current_rsi > 55 and rsi_trend < -0.03:  # Above midline with downtrend
            rsi_score = -1
        elif current_rsi < 50 and longer_rsi_trend > 0:  # Below midline with long-term uptrend
            rsi_score = 0.5
        elif current_rsi > 50 and longer_rsi_trend < 0:  # Above midline with long-term downtrend
            rsi_score = -0.5
            
        details = {
            'current_rsi': round(current_rsi, 2),
            'rsi_trend': round(rsi_trend, 4),
            'longer_rsi_trend': round(longer_rsi_trend, 4)
        }
            
        # Add analysis of RSI divergence
        if len(recent_rsi) > 10 and len(analysis_df['Close'].tail(20)) > 10:
            price_trend = np.polyfit(range(len(analysis_df['Close'].tail(20))), analysis_df['Close'].tail(20).values, 1)[0]
            
            if price_trend > 0 and rsi_trend < 0:  # Bearish divergence
                rsi_score -= 0.5
                details['divergence'] = 'bearish'
            elif price_trend < 0 and rsi_trend > 0:  # Bullish divergence
                rsi_score += 0.5
                details['divergence'] = 'bullish'
            else:
                details['divergence'] = 'none'
                
        # Add analysis of RSI volatility
        if len(recent_rsi) > 10:
            rsi_volatility = np.std(recent_rsi)
            details['rsi_volatility'] = round(rsi_volatility, 4)
            
            if rsi_volatility > 5:  # High volatility
                rsi_score -= 0.3
            elif rsi_volatility < 3:  # Low volatility
                rsi_score += 0.3
                
        return rsi_score, details
        
    except Exception as e:
        logger.error(f"Error analyzing RSI trend: {e}")
        return 0, {'error': str(e)}

def analyze_ao_trend(analysis_df):
    """Analyze Awesome Oscillator trends to detect momentum shifts"""
    try:
        # Get recent AO values
        recent_ao = analysis_df['AO_weekly_AVG'].tail(30).values
        
        # Current AO value and trend
        current_ao = recent_ao[-1] if len(recent_ao) > 0 else 0
        ao_trend = np.polyfit(range(len(recent_ao)), recent_ao, 1)[0] if len(recent_ao) > 1 else 0
        
        # Check for recent crosses in last ~3 days (15 trading days)
        recent_crosses = recent_ao[-15:] if len(recent_ao) >= 15 else recent_ao
        ao_crosses_up = False
        ao_crosses_down = False
        
        for i in range(1, len(recent_crosses)):
            if recent_crosses[i-1] < 0 and recent_crosses[i] >= 0:
                ao_crosses_up = True
            if recent_crosses[i-1] > 0 and recent_crosses[i] <= 0:
                ao_crosses_down = True
        
        # Assign score
        ao_score = 0
        
        if ao_crosses_up:  # Recent bullish zero-line cross
            ao_score = 2.5
        elif ao_crosses_down:  # Recent bearish zero-line cross
            ao_score = -2.5
        elif current_ao > 0 and ao_trend > 0.02:  # Strong positive momentum above zero
            ao_score = 2
        elif current_ao < 0 and ao_trend < -0.02:  # Strong negative momentum below zero
            ao_score = -2
        elif current_ao > 0:  # Positive but not increasing strongly
            ao_score = 1
        elif current_ao < 0:  # Negative but not decreasing strongly
            ao_score = -1
            
        details = {
            'current_ao': round(current_ao, 2),
            'ao_trend': round(ao_trend, 4),
            'crosses_up': ao_crosses_up,
            'crosses_down': ao_crosses_down
        }
            
        # Add analysis of AO divergence
        if len(recent_ao) > 10 and len(analysis_df['Close'].tail(20)) > 10:
            price_trend = np.polyfit(range(len(analysis_df['Close'].tail(20))), analysis_df['Close'].tail(20).values, 1)[0]
            
            if price_trend > 0 and ao_trend < 0:  # Bearish divergence
                ao_score -= 0.5
                details['divergence'] = 'bearish'
            elif price_trend < 0 and ao_trend > 0:  # Bullish divergence
                ao_score += 0.5
                details['divergence'] = 'bullish'
            else:
                details['divergence'] = 'none'
                
        return ao_score, details
        
    except Exception as e:
        logger.error(f"Error analyzing AO trend: {e}")
        return 0, {'error': str(e)}

def analyze_volume_pattern(analysis_df):
    """Analyze volume patterns for distribution/accumulation signs"""
    try:
        # Get recent volume and price data
        recent_volume = analysis_df['Volume'].tail(60)
        recent_returns = analysis_df['pct_change'].tail(60)
        
        volume_score = 0
        details = {}
        
        if len(recent_returns) >= 40:
            # Calculate avg volume on up days vs down days
            up_days = recent_returns > 0
            down_days = recent_returns < 0
            
            if up_days.sum() > 0 and down_days.sum() > 0:
                up_day_volume = recent_volume[up_days].mean()
                down_day_volume = recent_volume[down_days].mean()
                
                # Calculate volume ratio
                vol_ratio = up_day_volume / down_day_volume if down_day_volume > 0 else 1.0
                details['volume_ratio'] = round(vol_ratio, 2)
                
                # Look at recent volume trend (30 days)
                recent_vol_trend = recent_volume.tail(30)
                vol_trend = np.polyfit(range(len(recent_vol_trend)), recent_vol_trend.values, 1)[0]
                details['volume_trend'] = round(vol_trend, 2)
                
                # Score based on volume patterns
                if vol_ratio > 1.5 and vol_trend > 0:
                    # Higher volume on up days and increasing volume - strong accumulation
                    volume_score = 2.5
                elif vol_ratio < 0.67 and vol_trend > 0:
                    # Higher volume on down days and increasing volume - strong distribution
                    volume_score = -2.5
                elif vol_ratio > 1.2:
                    # Moderately higher volume on up days
                    volume_score = 1.5
                elif vol_ratio < 0.83:
                    # Moderately higher volume on down days
                    volume_score = -1.5
                elif vol_ratio > 1:
                    # Slightly higher volume on up days
                    volume_score = 0.7
                elif vol_ratio < 1:
                    # Slightly higher volume on down days
                    volume_score = -0.7
            
        # Add additional analysis for volume spikes
        if len(recent_volume) >= 30:
            # Calculate volume spikes relative to different time periods
            volume_spike_thresholds = {
                'short_term': recent_volume.tail(10).mean() * 2,
                'medium_term': recent_volume.tail(30).mean() * 2,
                'long_term': recent_volume.tail(60).mean() * 2
            }
            
            volume_spikes = {}
            for period, threshold in volume_spike_thresholds.items():
                volume_spikes[period] = recent_volume[recent_volume > threshold]
            
            details['volume_spikes'] = {}
            for period, spikes in volume_spikes.items():
                if len(spikes) > 0:
                    spike_days = spikes.index
                    spike_returns = recent_returns.loc[spike_days]
                    
                    if spike_returns.mean() > 0:  # Positive returns on spike days
                        volume_score += 0.3
                        details['volume_spikes'][period] = 'positive'
                    elif spike_returns.mean() < 0:  # Negative returns on spike days
                        volume_score -= 0.3
                        details['volume_spikes'][period] = 'negative'
                    else:
                        details['volume_spikes'][period] = 'neutral'
                else:
                    details['volume_spikes'][period] = 'none'
            
        # Add analysis of volume divergence
        if len(recent_volume) > 10 and len(analysis_df['Close'].tail(20)) > 10:
            price_trend = np.polyfit(range(len(analysis_df['Close'].tail(20))), analysis_df['Close'].tail(20).values, 1)[0]
            volume_trend = np.polyfit(range(len(recent_volume.tail(20))), recent_volume.tail(20).values, 1)[0]
            
            if price_trend > 0 and volume_trend < 0:  # Bearish divergence
                volume_score -= 0.5
                details['divergence'] = 'bearish'
            elif price_trend < 0 and volume_trend > 0:  # Bullish divergence
                volume_score += 0.5
                details['divergence'] = 'bullish'
            else:
                details['divergence'] = 'none'
                
        # Add analysis of volume trend relative to price trend
        if len(recent_volume) > 10 and len(analysis_df['Close'].tail(20)) > 10:
            price_trend = np.polyfit(range(len(analysis_df['Close'].tail(20))), analysis_df['Close'].tail(20).values, 1)[0]
            volume_trend = np.polyfit(range(len(recent_volume.tail(20))), recent_volume.tail(20).values, 1)[0]
            
            trend_difference = volume_trend - price_trend
            
            details['volume_price_trend_diff'] = round(trend_difference, 4)
            
            if trend_difference > 0.01:  # Volume trend significantly above price trend
                volume_score += 0.3
            elif trend_difference < -0.01:  # Volume trend significantly below price trend
                volume_score -= 0.3
                
        return volume_score, details
        
    except Exception as e:
        logger.error(f"Error analyzing volume pattern: {e}")
        return 0, {'error': str(e)}

def analyze_price_ma_relationship(analysis_df):
    """Analyze price relationship to moving averages"""
    try:
        # Get recent price and MA data
        recent_prices = analysis_df['Close'].tail(5).values
        recent_ma = analysis_df['MA_30'].tail(5).values
        
        ma_score = 0
        details = {}
        
        # Current relationship between price and multiple MAs
        if len(recent_prices) > 0:
            ma_columns = ['MA_10', 'MA_30', 'MA_50']
            ma_scores = []
            ma_details = {}
            
            for ma_col in ma_columns:
                if ma_col in analysis_df.columns:
                    recent_ma = analysis_df[ma_col].tail(5).values
                    if len(recent_ma) > 0:
                        price_vs_ma = (recent_prices[-1] / recent_ma[-1] - 1) * 100  # % difference
                        price_above_ma = recent_prices[-1] > recent_ma[-1]
                        
                        ma_details[f'{ma_col}_pct'] = round(price_vs_ma, 2)
                        ma_details[f'{ma_col}_above'] = price_above_ma
                        
                        # Score based on price-MA relationship
                        if price_above_ma and price_vs_ma > 5:  # Price significantly above MA
                            ma_scores.append(1.5)
                        elif price_above_ma:  # Price moderately above MA
                            ma_scores.append(1)
                        elif not price_above_ma and price_vs_ma < -5:  # Price significantly below MA
                            ma_scores.append(-1.5)
                        elif not price_above_ma:  # Price moderately below MA
                            ma_scores.append(-1)
            
            # Calculate average MA score
            if ma_scores:
                ma_score = sum(ma_scores) / len(ma_scores)
                details.update(ma_details)
                
        # Add analysis of price trend relative to MA
        if len(recent_prices) > 1 and len(recent_ma) > 1:
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            ma_trend = np.polyfit(range(len(recent_ma)), recent_ma, 1)[0]
            
            trend_difference = price_trend - ma_trend
            
            details['price_trend'] = round(price_trend, 4)
            details['ma_trend'] = round(ma_trend, 4)
            details['trend_difference'] = round(trend_difference, 4)
            
            if trend_difference > 0.01:  # Price trend significantly above MA trend
                ma_score += 0.5
            elif trend_difference < -0.01:  # Price trend significantly below MA trend
                ma_score -= 0.5
                
        return ma_score, details
        
    except Exception as e:
        logger.error(f"Error analyzing price-MA relationship: {e}")
        return 0, {'error': str(e)}

def analyze_price_pattern(analysis_df):
    """Analyze price patterns (higher highs/lows vs lower highs/lows)"""
    try:
        pattern_score = 0
        details = {}
        
        # For daily data, use appropriate window sizes (~80 days = ~16 weeks)
        if len(analysis_df) >= 80:
            # Use proper pivots for swing analysis
            prices = analysis_df['Close'].values
            
            # Look at 2 segments of 40 days each (~8 weeks)
            prev_segment = prices[-80:-40]
            curr_segment = prices[-40:]
            
            prev_high = max(prev_segment)
            prev_low = min(prev_segment)
            curr_high = max(curr_segment)
            curr_low = min(curr_segment)
            
            details['prev_high'] = round(prev_high, 2)
            details['prev_low'] = round(prev_low, 2) 
            details['curr_high'] = round(curr_high, 2)
            details['curr_low'] = round(curr_low, 2)
            
            # Calculate percentage changes
            high_change_pct = (curr_high / prev_high - 1) * 100
            low_change_pct = (curr_low / prev_low - 1) * 100
            
            details['high_change_pct'] = round(high_change_pct, 2)
            details['low_change_pct'] = round(low_change_pct, 2)
            
            # Check for higher highs and higher lows (accumulation)
            if high_change_pct > 3 and low_change_pct > 3:
                pattern_score = 2.5  # Strong higher highs & higher lows
                details['pattern'] = 'strong_higher_highs_lows'
            elif curr_high > prev_high and curr_low > prev_low:
                pattern_score = 1.5  # Higher highs & higher lows
                details['pattern'] = 'higher_highs_lows'
            # Check for lower highs and lower lows (distribution)
            elif high_change_pct < -3 and low_change_pct < -3:
                pattern_score = -2.5  # Strong lower highs & lower lows
                details['pattern'] = 'strong_lower_highs_lows'
            elif curr_high < prev_high and curr_low < prev_low:
                pattern_score = -1.5  # Lower highs & lower lows
                details['pattern'] = 'lower_highs_lows'
            # Check for mixed patterns
            elif curr_high > prev_high and curr_low < prev_low:
                pattern_score = 0.3  # Expanding volatility
                details['pattern'] = 'expanding_volatility'
            elif curr_high < prev_high and curr_low > prev_low:
                pattern_score = -0.3  # Contracting volatility
                details['pattern'] = 'contracting_volatility'
                
            # Add analysis of price trend
            if len(prices) > 10:
                price_trend = np.polyfit(range(len(prices[-20:])), prices[-20:], 1)[0]
                details['price_trend'] = round(price_trend, 4)
                
                if price_trend > 0.01:  # Positive price trend
                    pattern_score += 0.3
                elif price_trend < -0.01:  # Negative price trend
                    pattern_score -= 0.3
                
        return pattern_score, details
        
    except Exception as e:
        logger.error(f"Error analyzing price pattern: {e}")
        return 0, {'error': str(e)}

def calculate_final_phase_score(rsi_score, ao_score, volume_score, ma_score, pattern_score):
    """Calculate the final market phase score and determine accumulation/distribution"""
    try:
        # Calculate total score with weighted indicators
        total_score = (rsi_score * 0.3) + (ao_score * 0.25) + (volume_score * 0.2) + (ma_score * 0.15) + (pattern_score * 0.1)
        
        # Define neutral threshold
        neutral_threshold = config.NEUTRAL_THRESHOLD
        
        # Define max possible score
        max_possible_score = config.MAX_POSSIBLE_SCORE
        
        # Calculate probability and determine phase
        if total_score > neutral_threshold:  # Accumulation
            probability = min(round(((total_score - neutral_threshold) / (max_possible_score - neutral_threshold)) * 100, 2), 100)
            phase = "ACCUMULATION"
        elif total_score < -neutral_threshold:  # Distribution
            probability = min(round(((abs(total_score) - neutral_threshold) / (max_possible_score - neutral_threshold)) * 100, 2), 100)
            phase = "DISTRIBUTION"
        else:  # Neutral zone
            neutral_position = total_score / neutral_threshold if neutral_threshold > 0 else 0
            probability = round(50 + (neutral_position * 25), 2)  # 25-75% within neutral zone
            phase = "NEUTRAL"
            
        # Ensure probability is within 0-100 range
        probability = max(0, min(probability, 100))
            
        return phase, probability, {
            'rsi_score': rsi_score,
            'ao_score': ao_score,
            'volume_score': volume_score,
            'ma_score': ma_score,
            'pattern_score': pattern_score,
            'total_score': round(total_score, 2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating final phase score: {e}")
        return "NEUTRAL", 50, {'error': str(e)}

def calculate_market_phase(df, symbol_name):
    """Calculate market phase (accumulation/distribution) for a stock"""
    analysis_df = prepare_analysis_data(df)
    rsi_score, rsi_details = analyze_rsi_trend(analysis_df)
    ao_score, ao_details = analyze_ao_trend(analysis_df)
    volume_score, volume_details = analyze_volume_pattern(analysis_df)
    ma_score, ma_details = analyze_price_ma_relationship(analysis_df)
    pattern_score, pattern_details = analyze_price_pattern(analysis_df)
    
    return calculate_final_phase_score(
        rsi_score, ao_score, volume_score, ma_score, pattern_score
    )

# 4. VISUALIZATION AND REPORTING FUNCTIONS
class FigureManager:
    """Context manager for handling matplotlib figures"""
    def __init__(self):
        self.fig = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fig is not None:
            plt.close(self.fig)
            
    def create_figure(self, figsize):
        """Create a new figure and store it"""
        self.fig = plt.figure(figsize=figsize)
        return self.fig

def draw_indicator_trend_lines_with_signals(symbol: str) -> tuple:
    """Draw indicator trend lines with buy/sell signals for a stock"""
    try:
        # Get stock data
        table_name = f"PSX_{symbol}_stock_data"
        df = get_stock_data(table_name)
        
        if df.empty:
            logger.error(f"No data found for {symbol}")
            return None, None
            
        # Get buy/sell signals
        buy_signals, sell_signals = get_buy_sell_signals(symbol)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 14))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # Price and MA subplot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax1.plot(df.index, df['MA'], label='MA', color='red')
        
        # Add buy/sell signals
        for date, price in buy_signals:
            ax1.scatter(date, price, color='green', marker='^', s=100, label='Buy Signal')
        for date, price in sell_signals:
            ax1.scatter(date, price, color='red', marker='v', s=100, label='Sell Signal')
            
        ax1.set_title(f'{symbol} Price and Signals')
        ax1.legend()
        ax1.grid(True)
        
        # RSI subplot
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_title('RSI')
        ax2.legend()
        ax2.grid(True)
        
        # AO subplot
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(df.index, df['AO'], label='AO', color='orange')
        ax3.axhline(y=0, color='black', linestyle='-')
        ax3.set_title('Awesome Oscillator')
        ax3.legend()
        ax3.grid(True)
        
        # Volume subplot
        ax4 = fig.add_subplot(gs[3])
        ax4.bar(df.index, df['Volume'], label='Volume', color='gray', alpha=0.5)
        ax4.set_title('Volume')
        ax4.legend()
        ax4.grid(True)
        
        # Format x-axis dates
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        os.makedirs('outputs/charts/RSI_AO_CHARTS', exist_ok=True)
        plot_filename = os.path.join('outputs/charts/RSI_AO_CHARTS', f'{symbol}_trend_lines_with_signals.png')
        plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
        plt.close()
        
        # Generate title text
        title_text = f"{symbol} Technical Analysis"
        
        logger.info(f"Generated chart for {symbol} at {plot_filename}")
        return plot_filename, title_text
        
    except Exception as e:
        logger.error(f"Error drawing chart for {symbol}: {e}")
        return None, None

def generate_stock_dashboard():
    """Generate a dashboard showing buy, sell and neutral stocks with key metrics"""
    try:
        database_path = 'data/databases/production/psx_consolidated_data_indicators_PSX.db'
        
        # Create a connection to the database
        engine = create_engine(f'sqlite:///{database_path}')
        connection = engine.connect()
        cursor = connection.connection.cursor()
        
        # Get all available symbols
        available_symbols = get_available_symbols(cursor)
        
        # Get all buy signals
        buy_df = get_latest_buy_stocks()
        buy_symbols = set(buy_df['Stock'].tolist()) if not buy_df.empty else set()
        
        # Prepare containers for results
        all_results = []
        
        # Process each symbol individually
        logger.info("Analyzing all available stocks for dashboard...")
        for symbol in tqdm(available_symbols, desc="Processing stocks"):
            try:
                # Get stock data for this specific symbol only
                table_name = f"PSX_{symbol}_stock_data"
                
                # First, check which columns exist in this table
                available_columns = fetch_column_names(engine, table_name)
                if not available_columns:
                    logger.warning(f"No columns found for {table_name}")
                    continue
                
                # Define required and optional columns
                required_cols = ["Date", "Close"]
                optional_cols = {
                    "RSI_weekly_Avg": None,
                    "AO_weekly_AVG": None,
                    "MA_30": None,
                    "RSI_weekly": None,
                    "Volume": None
                }
                
                # Check if required columns exist
                if not all(col in available_columns for col in required_cols):
                    logger.warning(f"Missing required columns in {table_name}")
                    continue
                
                # Build SELECT clause with only available columns
                select_cols = required_cols.copy()
                for col in optional_cols:
                    if col in available_columns:
                        select_cols.append(col)
                
                # Build and execute query
                query = f"""SELECT {', '.join(select_cols)} 
                           FROM {table_name}
                           ORDER BY Date DESC
                           LIMIT 60"""
                
                df = pd.read_sql(query, connection)
                
                if df.empty:
                    continue
                
                # Add missing columns as NaN
                for col, default_val in optional_cols.items():
                    if col not in df.columns:
                        df[col] = default_val
                
                # Rest of your processing code...
                # [rest of the function remains the same]
                # Work with a proper copy to avoid SettingWithCopyWarning
                df = df.copy()
                
                # Convert the date column to datetime
                df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
                
                # Calculate percentage change
                df.loc[:, 'pct_change'] = df['Close'].pct_change() * 100
                
                # Get latest data
                latest = df.iloc[0] if not df.empty else None
                if latest is None:
                    continue
                
                # Get buy and sell signals
                buy_signals, sell_signals = get_buy_sell_signals(symbol)
                
                # Determine stock status
                if buy_signals and sell_signals:
                    latest_buy = max(buy_signals, key=lambda x: x[0])
                    latest_sell = max(sell_signals, key=lambda x: x[0])
                    
                    if latest_buy[0] > latest_sell[0]:
                        status = "BUY/HOLD"
                    else:
                        status = "SELL"
                elif buy_signals:
                    status = "BUY/HOLD"
                elif sell_signals:
                    status = "SELL"
                else:
                    status = "OPPORTUNITY"
                
                # Calculate market phase
                market_phase, phase_probability, _ = calculate_market_phase(df, symbol)
                
                # Get holding days and profit/loss for buy stocks
                holding_days = None
                profit_loss = None
                
                if status == "BUY/HOLD" and symbol in buy_symbols:
                    stock_info = buy_df[buy_df['Stock'] == symbol].iloc[0]
                    holding_days = int(stock_info['holding_days']) if 'holding_days' in stock_info else None
                    
                    # Calculate profit/loss if we have signal price
                    if 'Signal_Close' in stock_info:
                        signal_price = float(stock_info['Signal_Close'])
                        current_price = latest['Close']
                        profit_loss = ((current_price - signal_price) / signal_price) * 100
                
                # Collect metrics
                result = {
                    'Symbol': symbol,
                    'Status': status,
                    'Close': latest['Close'],
                    'RSI': latest['RSI_weekly'],
                    'AO': latest['AO_weekly_AVG'],
                    'Market_Phase': market_phase,
                    'Phase_Probability': phase_probability,
                    'Holding_Days': holding_days,
                    'Profit_Loss': profit_loss,
                    'Above_MA30': latest['Close'] > latest['MA_30'] if 'MA_30' in latest and pd.notna(latest['MA_30']) else False
                }
                
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for dashboard: {e}")
                continue
        
        # Convert results to DataFrame
        dashboard_df = pd.DataFrame(all_results)
        
        # Create the dashboard
        create_dashboard_visualization(dashboard_df)
        
        # Close connection
        connection.close()
        
        return dashboard_df
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def create_dashboard_visualization(df):
    """Create visual dashboard using Matplotlib"""
    if df.empty:
        logger.error("No data available for dashboard visualization")
        return False
    
    # Create a figure with multiple subplots - 3x3 grid
    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(hspace=0.8, wspace=0.4)  # Increased spacing between plots
    
    # Add a title
    fig.suptitle('PSX Market Dashboard', fontsize=24, y=0.98)
    
    # 1-7: Keep these plots as they are...
    
    # 1. Status Distribution Pie Chart
    plt.subplot(3, 3, 1)
    status_counts = df['Status'].value_counts()
    colors = {'BUY/HOLD': 'green', 'SELL': 'red', 'OPPORTUNITY': 'blue'}
    status_colors = [colors.get(s, 'gray') for s in status_counts.index]
    plt.pie(status_counts, labels=status_counts.index, autopct='%.2f%%', colors=status_colors)
    plt.title('Stock Signal Distribution')
    
    # 2. Market Phase Distribution Pie Chart
    plt.subplot(3, 3, 2)
    phase_counts = df['Market_Phase'].value_counts()
    phase_colors = {'ACCUMULATION': 'green', 'DISTRIBUTION': 'red', 'NEUTRAL': 'gray'}
    plt.pie(phase_counts, labels=phase_counts.index, autopct='%.2f%%',
            colors=[phase_colors.get(p, 'blue') for p in phase_counts.index])
    plt.title('Market Phase Distribution')
    
    # 3. Market Breadth Indicator
    plt.subplot(3, 3, 3)
    # Calculate key market breadth metrics
    above_ma = df['Above_MA30'].sum() / len(df) * 100
    acc_stocks = len(df[df['Market_Phase'] == 'ACCUMULATION']) / len(df) * 100
    high_rsi = len(df[df['RSI'] > 50]) / len(df) * 100
    pos_ao = len(df[df['AO'] > 0]) / len(df) * 100

    metrics = ['Above MA30', 'Accumulation', 'RSI > 50', 'AO > 0']
    values = [above_ma, acc_stocks, high_rsi, pos_ao]
    colors = ['navy', 'green', 'purple', 'orange']

    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Calculate overall market breadth score
    market_score = sum(values) / len(values)
    plt.axhline(y=market_score, color='black', linestyle='-', linewidth=2, alpha=0.5)
    plt.text(len(metrics) - 0.5, market_score + 2, f'Avg: {market_score:.1f}%', 
             ha='center', va='bottom', fontweight='bold')

    # Determine market condition
    market_condition = "Neutral Market"
    if market_score > 60:
        market_condition = "Strong Bullish Market"
    elif market_score > 50:
        market_condition = "Moderately Bullish Market" 
    elif market_score < 40:
        market_condition = "Strong Bearish Market"
    elif market_score < 50:
        market_condition = "Moderately Bearish Market"

    plt.title(f'Market Breadth Indicator\n{market_condition}')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel('Percentage of Stocks (%)')
    
    # 4. Exit Timing Indicator
    plt.subplot(3, 3, 4)
    buy_df = df[df['Status'] == 'BUY/HOLD'].copy()

    if not buy_df.empty and 'Holding_Days' in buy_df.columns and len(buy_df.dropna(subset=['Holding_Days', 'Profit_Loss'])) > 0:
        # Create holding period buckets
        buy_df['Holding_Bucket'] = pd.cut(
            buy_df['Holding_Days'].fillna(0), 
            bins=[0, 5, 20, 60, 120, float('inf')],
            labels=['0-5d', '6-20d', '21-60d', '61-120d', '>120d']
        )
        
        # Calculate average profit/loss by holding period
        profitability = buy_df.groupby('Holding_Bucket', observed=False)['Profit_Loss'].agg(
            ['mean', 'count']).reset_index()
        
        if not profitability.empty:
            # Create plot
            bars = plt.bar(profitability['Holding_Bucket'], 
                          profitability['mean'], 
                          alpha=0.7,
                          color=['green' if x >= 0 else 'red' for x in profitability['mean']])
            
            # Add count labels
            for i, bar in enumerate(bars):
                count = profitability.iloc[i]['count']
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (1 if bar.get_height() >= 0 else -3),
                        f"n={count}", 
                        ha='center', va='bottom', fontsize=8)
            
            plt.title('Profit/Loss by Holding Period')
            plt.xlabel('Holding Period')
            plt.ylabel('Average Profit/Loss %')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add optimal exit guidance if we have valid data
            if not profitability['mean'].isna().all():
                best_period_idx = profitability['mean'].idxmax()
                best_period = profitability.iloc[best_period_idx]
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.text(0.5, 0.9, 
                        f"Best exit window: {best_period['Holding_Bucket']}", 
                        transform=plt.gca().transAxes, ha='center',
                        bbox=dict(facecolor='yellow', alpha=0.5))
    else:
        plt.text(0.5, 0.5, 'No buy/hold stocks data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 5. Market Momentum Heat Map
    plt.subplot(3, 3, 5)
    # Create a filtered DataFrame with valid data for both axes
    valid_data = df.dropna(subset=['RSI', 'AO'])
    
    if len(valid_data) >= 5:  # Make sure we have enough data points
        # Create a 2D histogram (heatmap)
        h = plt.hist2d(valid_data['RSI'], valid_data['AO'], 
                      bins=[10, 10], cmap='RdYlGn', alpha=0.7,
                      range=[[0, 100], [-5, 5]])
        
        # Add quadrant lines
        plt.axvline(x=50, color='white', linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='white', linestyle='--', alpha=0.7)
        
        # Add quadrant labels
        plt.text(25, 2.5, "Strong Buy\nZone", fontsize=9, ha='center', color='white', weight='bold')
        plt.text(75, 2.5, "Overbought", fontsize=9, ha='center', color='white', weight='bold')
        plt.text(25, -2.5, "Weak", fontsize=9, ha='center', color='white', weight='bold')
        plt.text(75, -2.5, "Distribution\nZone", fontsize=9, ha='center', color='white', weight='bold')
        
        # Calculate and show market center of gravity
        avg_rsi = valid_data['RSI'].mean()
        avg_ao = valid_data['AO'].mean()
        plt.scatter(avg_rsi, avg_ao, color='white', edgecolor='black', s=100, marker='*')
        
        # Determine market status based on center of gravity
        market_status = ""
        if avg_rsi < 50 and avg_ao > 0:
            market_status = "Accumulation Phase"
        elif avg_rsi > 50 and avg_ao > 0:
            market_status = "Bullish Phase"
        elif avg_rsi > 50 and avg_ao < 0:
            market_status = "Distribution Phase"
        else:
            market_status = "Bearish Phase"
        
        plt.colorbar(label='Stock Concentration')
        plt.title(f'Market Momentum Heat Map\n{market_status}')
        plt.xlabel('RSI Value')
        plt.ylabel('AO Value')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for heat map', 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # 6. Top 10 BUY/HOLD stocks with highest phase probability
    plt.subplot(3, 3, 6)
    if not buy_df.empty:
        top_buys = buy_df.sort_values('Phase_Probability', ascending=False).head(10)
        y_pos = range(len(top_buys))
        plt.barh(y_pos, top_buys['Phase_Probability'], color='green', alpha=0.7)
        plt.yticks(y_pos, top_buys['Symbol'])
        plt.title('Top BUY/HOLD Stocks by Accumulation Probability')
        plt.xlabel('Accumulation Probability %')
    
    # 7. Top 10 OPPORTUNITY stocks with highest accumulation probability
    plt.subplot(3, 3, 7)
    opportunity_df = df[(df['Status'] == 'OPPORTUNITY') & (df['Market_Phase'] == 'ACCUMULATION')]
    if not opportunity_df.empty:
        top_opps = opportunity_df.sort_values('Phase_Probability', ascending=False).head(10)
        y_pos = range(len(top_opps))
        plt.barh(y_pos, top_opps['Phase_Probability'], color='blue', alpha=0.7)
        plt.yticks(y_pos, top_opps['Symbol'])
        plt.title('Top OPPORTUNITY Stocks (Accumulation Phase)')
        plt.xlabel('Accumulation Probability %')
    
    # 8. Latest Buy Signals by Holding Days
    plt.subplot(3, 3, 8)

    # Filter only the BUY/HOLD stocks
    buy_df = df[df['Status'] == 'BUY/HOLD'].copy()

    if not buy_df.empty and 'Holding_Days' in buy_df.columns:
        # Sort by holding days (newest first) and take top 10
        recent_buys = buy_df.sort_values('Holding_Days').head(10)
        
        # Prepare data for the plot
        y_pos = range(len(recent_buys))
        symbols = recent_buys['Symbol']
        days = recent_buys['Holding_Days']
        
        # Define colors based on Market Phase
        colors = {'ACCUMULATION': 'green', 'DISTRIBUTION': 'red', 'NEUTRAL': 'gray'}
        bar_colors = [colors.get(phase, 'blue') for phase in recent_buys['Market_Phase']]
        
        # Create horizontal bar chart
        bars = plt.barh(y_pos, days, color=bar_colors, alpha=0.7)
        plt.yticks(y_pos, symbols)
        
        # Add holding days labels at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            phase_prob = recent_buys.iloc[i]['Phase_Probability']
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f"{int(width)} days ({phase_prob:.1f}%)", ha='left', va='center', fontsize=8)
        
        plt.title('Latest Buy Signals by Holding Days')
        plt.xlabel('Days Since Buy Signal')
        
        # Add a legend for market phases
        markers = [plt.Rectangle((0,0),1,1,color=color) for color in [colors['ACCUMULATION'], colors['DISTRIBUTION'], colors['NEUTRAL']]]
        plt.legend(markers, ['Accumulation', 'Distribution', 'Neutral'], loc='upper right')
    else:
        plt.text(0.5, 0.5, 'No buy/hold stocks data available', 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    # 9. Market Rotation Analysis
    ax9 = plt.subplot(3, 3, 9)
    ax9.set_title('Market Rotation Analysis', fontsize=12)

    # Compute average metrics for different stock price ranges
    def get_price_category(price):
        if price < 50:
            return "Small Cap (<50)"
        elif price < 200:
            return "Mid Cap (50-200)"
        else:
            return "Large Cap (>200)"

    # Add price category to dataframe
    df['Price_Category'] = df['Close'].apply(get_price_category)

    # Calculate metrics by price category
    price_cat_metrics = df.groupby('Price_Category').agg({
        'Profit_Loss': lambda x: x.dropna().mean(),
        'RSI': 'mean',
        'AO': 'mean',
        'Symbol': 'count'
    }).reset_index()

    price_cat_metrics = price_cat_metrics.rename(columns={'Symbol': 'Count'})

    if not price_cat_metrics.empty and len(price_cat_metrics) > 1:
        bars = ax9.bar(price_cat_metrics['Price_Category'], price_cat_metrics['RSI'],
                      color=['lightblue', 'royalblue', 'darkblue'])
        
        # Add count labels on top of each bar
        for bar, count in zip(bars, price_cat_metrics['Count']):
            height = bar.get_height()
            ax9.annotate(f'n={count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        ax9.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax9.set_ylim(0, 100)
        
        # Add a short summary text for rotation indicators
        if (price_cat_metrics['RSI'].iloc[0] > price_cat_metrics['RSI'].iloc[-1] and
            len(price_cat_metrics) > 1):
            ax9.set_xlabel("⬆️ Rotation to Smaller Caps", fontweight='bold')
        elif (price_cat_metrics['RSI'].iloc[0] < price_cat_metrics['RSI'].iloc[-1] and
                len(price_cat_metrics) > 1):
            ax9.set_xlabel("⬇️ Rotation to Larger Caps", fontweight='bold')
                
    else:
        plt.text(0.5, 0.5, 'Insufficient data for rotation analysis',
                 ha='center', va='center', transform=ax9.transAxes)
        ax9.axis('off')
    
    # Save dashboard
    dashboards_folder = 'outputs/dashboards/PSX_DASHBOARDS'
    os.makedirs(dashboards_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y-%m-%d')
    dashboard_path = os.path.join(dashboards_folder, f'psx_dashboard_{current_date}.png')
    plt.savefig(dashboard_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Create tabular dashboards
    create_category_tables(df, dashboards_folder, current_date)
    
    # Send dashboard to Telegram
    message = f"PSX Market Dashboard - Generated on {current_date}"
    send_telegram_message_with_image(dashboard_path, message)
    
    logger.info(f"Dashboard saved to {dashboard_path} and sent to Telegram")
    return True

def create_category_tables(df, folder, date):
    """Create and save tabular dashboards for each category"""
    # 1. BUY/HOLD stocks table
    buy_df = df[df['Status'] == 'BUY/HOLD'].copy()
    if not buy_df.empty:
        buy_df = buy_df[['Symbol', 'Close', 'RSI', 'AO', 'Market_Phase', 
                         'Phase_Probability', 'Holding_Days', 'Profit_Loss']]
        buy_df.sort_values('Phase_Probability', ascending=False, inplace=True)
        buy_df['Profit_Loss'] = buy_df['Profit_Loss'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        
        # Save as CSV
        buy_csv = os.path.join(folder, f'buy_stocks_{date}.csv')
        buy_df.to_csv(buy_csv, index=False)
        
        # Create a prettier HTML version with styling
        buy_html = os.path.join(folder, f'buy_stocks_{date}.html')
        buy_df.to_html(buy_html, index=False, classes='table table-striped table-hover', border=0)
    
    # 2. SELL stocks table
    sell_df = df[df['Status'] == 'SELL'].copy()
    if not sell_df.empty:
        sell_df = sell_df[['Symbol', 'Close', 'RSI', 'AO', 'Market_Phase', 'Phase_Probability']]
        sell_df.sort_values('Phase_Probability', ascending=False, inplace=True)
        
        # Save as CSV
        sell_csv = os.path.join(folder, f'sell_stocks_{date}.csv')
        sell_df.to_csv(sell_csv, index=False)
        
        # HTML version
        sell_html = os.path.join(folder, f'sell_stocks_{date}.html')
        sell_df.to_html(sell_html, index=False, classes='table table-striped table-hover', border=0)
    
    # 3. OPPORTUNITY stocks table (in accumulation phase)
    opp_df = df[(df['Status'] == 'OPPORTUNITY') & (df['Market_Phase'] == 'ACCUMULATION')].copy()
    if not opp_df.empty:
        opp_df = opp_df[['Symbol', 'Close', 'RSI', 'AO', 'Phase_Probability']]
        opp_df.sort_values('Phase_Probability', ascending=False, inplace=True)
        
        # Save as CSV
        opp_csv = os.path.join(folder, f'opportunity_stocks_{date}.csv')
        opp_df.to_csv(opp_csv, index=False)
        
        # HTML version
        opp_html = os.path.join(folder, f'opportunity_stocks_{date}.html')
        opp_df.to_html(opp_html, index=False, classes='table table-striped table-hover', border=0)

def create_enhanced_category_tables(df, folder, date):
    """Create enhanced tabular dashboards with holding period analysis"""
    # Keep original tables
    create_category_tables(df, folder, date)
    
    # Enhanced buy/hold table with performance metrics
    buy_df = df[df['Status'] == 'BUY/HOLD'].copy()
    if not buy_df.empty:
        # Calculate additional metrics
        buy_df['Daily_Return'] = buy_df.apply(
            lambda x: x['Profit_Loss'] / x['Holding_Days'] if pd.notna(x['Profit_Loss']) and pd.notna(x['Holding_Days']) and x['Holding_Days'] > 0 else None, 
            axis=1
        )
        
        # Format metrics for display
        buy_df['Daily_Return'] = buy_df['Daily_Return'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        buy_df['Close'] = buy_df['Close'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        buy_df['RSI'] = buy_df['RSI'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        buy_df['AO'] = buy_df['AO'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        buy_df['Phase_Probability'] = buy_df['Phase_Probability'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        buy_df['Profit_Loss'] = buy_df['Profit_Loss'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        
        # Create holdings age categories
        def categorize_holding(days):
            if pd.isna(days):
                return "Unknown"
            if days <= 5:
                return "Very Short (≤5d)"
            elif days <= 20:
                return "Short (≤20d)"
            elif days <= 60:
                return "Medium (≤60d)"
            else:
                return "Long (>60d)"
                
        buy_df['Holding_Category'] = buy_df['Holding_Days'].apply(categorize_holding)
        
        # Sort by holding category and daily return
        buy_df['Sort_Order'] = buy_df['Holding_Category'].map({
            "Very Short (≤5d)": 1, 
            "Short (≤20d)": 2, 
            "Medium (≤60d)": 3, 
            "Long (>60d)": 4,
            "Unknown": 5
        })
        
        # Sort the dataframe first, before selecting columns
        buy_df = buy_df.sort_values(['Sort_Order', 'Holding_Days'])
        
        # Select and order columns (now after sorting)
        enhanced_buy_df = buy_df[[
            'Symbol', 'Close', 'RSI', 'AO', 'Market_Phase', 
            'Phase_Probability', 'Holding_Days', 'Holding_Category',
            'Profit_Loss', 'Daily_Return'
        ]]
        
        # Save as CSV and HTML
        enhanced_buy_csv = os.path.join(folder, f'enhanced_buy_stocks_{date}.csv')
        enhanced_buy_df.to_csv(enhanced_buy_csv, index=False)
        
        # Create HTML with styling for profit/loss cells
        enhanced_buy_html = os.path.join(folder, f'enhanced_buy_stocks_{date}.html')
        
        # Generate HTML with color-coding based on values
        html_content = enhanced_buy_df.to_html(index=False)
        styled_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .very-short {{ background-color: #ffe6e6; }}
                .short {{ background-color: #fff2e6; }}
                .medium {{ background-color: #e6f2ff; }}
                .long {{ background-color: #e6ffe6; }}
            </style>
        </head>
        <body>
            <h1>Enhanced Buy/Hold Stocks Analysis - {date}</h1>
            {html_content}
        </body>
        </html>
        '''
        
        with open(enhanced_buy_html, 'w') as f:
            f.write(styled_html)

def add_decision_matrix(df, fig, pos):
    """Add a decision matrix plot showing buy/sell/hold confidence by risk"""
    ax = fig.add_subplot(pos)
    ax.set_title('Signal Confidence & Decision Matrix', fontsize=12)
    
    # Create confidence scores based on market phase probability and technical indicators
    if not df.empty:
        decision_df = df.copy()
        
        # Calculate confidence score (0-100)
        decision_df['Confidence'] = decision_df.apply(
            lambda row: min(100, (
                (row['Phase_Probability'] * 0.5) +
                (30 if row['Status'] == 'BUY/HOLD' else 0) +
                (20 if row['Above_MA30'] else 0) +
                (20 if row['AO'] > 0 else 0) +
                (10 if row['RSI'] > 40 and row['RSI'] < 60 else 0)
            )), axis=1
        )
        
        # Calculate risk score (0-100)
        decision_df['Risk'] = decision_df.apply(
            lambda row: min(100, (
                (50 if row['Status'] == 'SELL' else 0) +
                (30 if not row['Above_MA30'] else 0) +
                (30 if row['AO'] < 0 else 0) +
                (100 - row['Phase_Probability']) * 0.4
            )), axis=1
        )
        
        # Create decision categories based on confidence and risk
        def get_decision(row):
            if row['Confidence'] > 70 and row['Risk'] < 30:
                return 'Strong Buy'
            elif row['Confidence'] > 60 and row['Risk'] < 40:
                return 'Buy'
            elif row['Confidence'] > 50 and row['Risk'] < 50:
                return 'Accumulate'
            elif row['Confidence'] < 30 and row['Risk'] > 70:
                return 'Strong Sell'
            elif row['Confidence'] < 40 and row['Risk'] > 60:
                return 'Sell'
            elif row['Confidence'] < 50 and row['Risk'] > 50:
                return 'Reduce'
            else:
                return 'Hold'
                
        decision_df['Decision'] = decision_df.apply(get_decision, axis=1)
        
        # Create scatter plot with color-coded decisions
        decision_colors = {
            'Strong Buy': 'darkgreen',
            'Buy': 'green', 
            'Accumulate': 'lightgreen',
            'Hold': 'blue',
            'Reduce': 'orange',
            'Sell': 'red',
            'Strong Sell': 'darkred'
        }
        
        # Plot each decision category
        for decision, group in decision_df.groupby('Decision'):
            ax.scatter(group['Risk'], group['Confidence'], 
                       label=f'{decision} ({len(group)})',
                       color=decision_colors.get(decision, 'gray'),
                       alpha=0.7, s=80)
            
            # Label some key stocks in each category
            if len(group) > 0:
                # Label up to 3 stocks per category
                for _, row in group.nlargest(min(3, len(group)), 'Confidence').iterrows():
                    ax.annotate(row['Symbol'], 
                               (row['Risk'], row['Confidence']),
                               xytext=(5, 0), textcoords='offset points',
                               fontsize=8)
        
        # Add quadrant lines
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(25, 75, "BUY ZONE", fontsize=12, ha='center', color='green', weight='bold')
        ax.text(75, 75, "CAUTION", fontsize=12, ha='center', color='orange', weight='bold')
        ax.text(25, 25, "NEUTRAL", fontsize=12, ha='center', color='blue', weight='bold')
        ax.text(75, 25, "SELL ZONE", fontsize=12, ha='center', color='red', weight='bold')
        
        ax.set_xlabel('Risk Score (Lower is Better)')
        ax.set_ylabel('Confidence Score (Higher is Better)')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Recommendation', loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 ncol=4, fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for decision matrix',
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def generate_portfolio_recommendations(df):
    """Generate detailed portfolio management recommendations based on data analysis"""
    if df.empty:
        return "Insufficient data for portfolio recommendations"
    
    # Calculate overall market status
    accumulation_pct = len(df[df['Market_Phase'] == 'ACCUMULATION']) / len(df) * 100
    bullish_rsi_pct = len(df[df['RSI'] > 50]) / len(df) * 100
    positive_ao_pct = len(df[df['AO'] > 0]) / len(df) * 100
    
    market_score = (accumulation_pct + bullish_rsi_pct + positive_ao_pct) / 3
    
    # Define market condition
    if market_score > 65:
        market_condition = "STRONGLY BULLISH"
    elif market_score > 55:
        market_condition = "MODERATELY BULLISH"
    elif market_score > 45:
        market_condition = "NEUTRAL"
    elif market_score > 35:
        market_condition = "MODERATELY BEARISH"
    else:
        market_condition = "STRONGLY BEARISH"
    
    # Top picks for different strategies
    if len(df) > 5:
        # Value stocks (low RSI but accumulation phase)
        value_picks = df[(df['Market_Phase'] == 'ACCUMULATION') & 
                         (df['RSI'] < 50) & 
                         (df['AO'] > 0)].nlargest(5, 'Phase_Probability')
                         
        # Growth stocks (strong momentum)
        growth_picks = df[(df['AO'] > 0) & 
                         (df['RSI'] > 50) & 
                         (df['RSI'] < 70)].nlargest(5, 'AO')
        
        # Stocks to avoid
        avoid_picks = df[(df['Market_Phase'] == 'DISTRIBUTION') & 
                        (df['Phase_Probability'] > 70)].nlargest(5, 'Phase_Probability')
    else:
        value_picks = growth_picks = avoid_picks = pd.DataFrame()
    
    # Generate recommendations
    recommendations = f"🔍 PORTFOLIO RECOMMENDATIONS ({market_condition} MARKET)\n\n"
    
    # Position sizing recommendation based on market condition
    if market_condition in ["STRONGLY BULLISH", "MODERATELY BULLISH"]:
        recommendations += "📊 POSITION SIZING: Standard to aggressive position sizes recommended\n"
        recommendations += "🎯 TARGET ALLOCATION: 80-100% invested\n"
    elif market_condition == "NEUTRAL":
        recommendations += "📊 POSITION SIZING: Standard position sizes recommended\n"
        recommendations += "🎯 TARGET ALLOCATION: 60-80% invested\n"
    else:
        recommendations += "📊 POSITION SIZING: Reduced position sizes recommended\n" 
        recommendations += "🎯 TARGET ALLOCATION: 30-50% invested\n"
    
    # Strategy recommendations
    recommendations += f"\n💼 STRATEGY RECOMMENDATIONS ({market_score:.1f}% bullish score):\n"
    
    if market_condition in ["STRONGLY BULLISH", "MODERATELY BULLISH"]:
        recommendations += "✅ Focus on growth and momentum stocks\n"
        recommendations += "✅ Consider pyramiding profitable positions\n" 
        recommendations += "✅ Let winners run with trailing stops\n"
    elif market_condition == "NEUTRAL":
        recommendations += "✅ Balance between growth and value stocks\n"
        recommendations += "✅ Tighter stops on all positions\n"
        recommendations += "✅ Take partial profits at resistance levels\n"
    else:
        recommendations += "✅ Focus on capital preservation\n"
        recommendations += "✅ Only high conviction value setups\n"
        recommendations += "✅ Reduce or hedge existing positions\n"
        
    # Top picks
    recommendations += "\n🔝 TOP VALUE PICKS:\n"
    if not value_picks.empty:
        for i, (_, row) in enumerate(value_picks.iterrows(), 1):
            recommendations += f"{i}. {row['Symbol']} - {row['Phase_Probability']:.1f}% acc. probability, RSI: {row['RSI']:.1f}\n"
    else:
        recommendations += "No clear value picks identified\n"
        
    recommendations += "\n🚀 TOP GROWTH PICKS:\n"
    if not growth_picks.empty:
        for i, (_, row) in enumerate(growth_picks.iterrows(), 1):
            recommendations += f"{i}. {row['Symbol']} - AO: {row['AO']:.2f}, RSI: {row['RSI']:.1f}\n"
    else:
        recommendations += "No clear growth picks identified\n"
        
    recommendations += "\n⚠️ STOCKS TO AVOID:\n"
    if not avoid_picks.empty:
        for i, (_, row) in enumerate(avoid_picks.iterrows(), 1):
            recommendations += f"{i}. {row['Symbol']} - {row['Phase_Probability']:.1f}% dist. probability\n"
    else:
        recommendations += "No clear stocks to avoid identified\n"
        
    return recommendations

# 5. MAIN EXECUTION CODE
if __name__ == "__main__":
    args = parse_args()
    
    # Check if required databases exist and are accessible
    if not check_database_files():
        exit(1)
    
    # database path
    database_path = 'data/databases/production/psx_consolidated_data_indicators_PSX.db'
    # Create a connection to the database
    engine = create_engine(f'sqlite:///{database_path}')
    connection = engine.connect()
    cursor = connection.connection.cursor()
    
    # Get available symbols
    available_symbols = get_available_symbols(cursor)
    
    if not available_symbols:
        logger.error("No stock symbols found in the database.")
        connection.close()
        exit(1)
        
    # Add this line at the beginning of your main code
    create_default_symbols_file()
    
    # Display all buy stocks sorted by update_date (most recent first), then by holding days
    logger.info("\n🟢 ALL BUY SIGNALS (SORTED BY UPDATE DATE) 🟢")
    logger.info("============================================")
    
    latest_buy_df = get_latest_buy_stocks()
    
    # Get sell signals as well
    latest_sell_df = get_latest_sell_stocks()
    
    # Format buy and sell signals tables for Telegram
    buy_signals_message = format_signals_for_telegram(latest_buy_df, "BUY")
    sell_signals_message = format_signals_for_telegram(latest_sell_df, "SELL")
    
    # Send buy signals to Telegram
    send_telegram_message(buy_signals_message)
    
    # Send sell signals to Telegram
    send_telegram_message(sell_signals_message)
    
    if not latest_buy_df.empty:
        # Format and display the buy signals in console as before
        # ...existing code for displaying buy signals in console...
        
        # Get unique symbols from the buy signals
        latest_buy_symbols = latest_buy_df['Stock'].unique().tolist()
        
        logger.info("\n📊 GENERATING CHARTS FOR BUY SIGNALS ONLY 📊")
        logger.info("===========================================")
        
        success_count = 0
        fail_count = 0
        
        # Only generate charts for stocks with active buy signals
        if latest_buy_symbols:
            for symbol in tqdm(latest_buy_symbols, desc="Generating charts"):
                if symbol in available_symbols:
                    table_name = f"PSX_{symbol}_stock_data"
                    logger.info(f"Processing {symbol}...")
                    
                    success, title_text = draw_indicator_trend_lines_with_signals(symbol)
                    if success:
                        success_count += 1
                        logger.info(f"✅ Chart for {symbol} has been generated and sent to Telegram")
                    else:
                        fail_count += 1
                        logger.info(f"❌ Failed to generate chart for {symbol}")
                else:
                    logger.warning(f"⚠️ Symbol {symbol} not found in available data tables")
                    fail_count += 1
            
            logger.info(f"\nCompleted processing {len(latest_buy_symbols)} buy signal stocks.")
            logger.info(f"✅ Successfully generated: {success_count} charts")
            logger.info(f"❌ Failed to generate: {fail_count} charts")
            logger.info(f"Charts saved in the RSI_AO_CHARTS folder.")
            
            # Send signals and charts summary to Telegram
            send_signals_and_charts_summary(latest_buy_df, latest_sell_df, available_symbols, success_count)
        else:
            logger.info("No buy signals to process for chart generation.")
    else:
        logger.info("No buy signals found in the database.")
    
    # Generate comprehensive market dashboard
    logger.info("\n📈 GENERATING MARKET DASHBOARD 📈")
    logger.info("================================")
    dashboard_df = generate_stock_dashboard()
    if not dashboard_df.empty:
        logger.info(f"Dashboard generated successfully with {len(dashboard_df)} stocks analyzed")
        
        # Display summary statistics
        buy_count = len(dashboard_df[dashboard_df['Status'] == 'BUY/HOLD'])
        sell_count = len(dashboard_df[dashboard_df['Status'] == 'SELL'])
        opp_count = len(dashboard_df[dashboard_df['Status'] == 'OPPORTUNITY'])
        
        logger.info(f"\nSummary Statistics:")
        logger.info(f"- BUY/HOLD signals: {buy_count}")
        logger.info(f"- SELL signals: {sell_count}")
        logger.info(f"- OPPORTUNITY signals: {opp_count}")
        
        # Display accumulation/distribution stats
        acc_count = len(dashboard_df[dashboard_df['Market_Phase'] == 'ACCUMULATION'])
        dist_count = len(dashboard_df[dashboard_df['Market_Phase'] == 'DISTRIBUTION'])
        neut_count = len(dashboard_df[dashboard_df['Market_Phase'] == 'NEUTRAL'])
        
        logger.info(f"\nMarket Phase Statistics:")
        logger.info(f"- ACCUMULATION: {acc_count} stocks ({acc_count/len(dashboard_df)*100:.2f}%)")
        logger.info(f"- DISTRIBUTION: {dist_count} stocks ({dist_count/len(dashboard_df)*100:.2f}%)")
        logger.info(f"- NEUTRAL: {neut_count} stocks ({neut_count/len(dashboard_df)*100:.2f}%)")
        
        # Dashboard files location
        current_date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"\nDashboard saved in PSX_DASHBOARDS folder with date {current_date}")
        
        # Define dashboards folder (this was missing)
        dashboards_folder = 'outputs/dashboards/PSX_DASHBOARDS'
        os.makedirs(dashboards_folder, exist_ok=True)
        
        # Generate portfolio recommendations
        portfolio_recommendations = generate_portfolio_recommendations(dashboard_df)

        # Save recommendations to file
        rec_file = os.path.join(dashboards_folder, f'portfolio_recommendations_{current_date}.txt')
        with open(rec_file, 'w') as f:
            f.write(portfolio_recommendations)

        # Send recommendations via Telegram
        send_telegram_message(portfolio_recommendations)
    else:
        logger.error("Failed to generate market dashboard")
    
    # Close the connection
    connection.close()
    
def get_latest_sell_stocks():
    """Get the latest sell stocks from the database"""
    try:
        with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI30.db') as conn:
            cursor = conn.cursor()
            
            # First check if the sell_stocks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sell_stocks'")
            if not cursor.fetchone():
                logger.warning("Table 'sell_stocks' does not exist in the database")
                return pd.DataFrame()
            
            cursor.execute("PRAGMA table_info(sell_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            logger.info(f"Actual columns in sell_stocks: {columns}")
            
            # Get a list of all unique stocks
            cursor.execute("SELECT DISTINCT Stock FROM sell_stocks WHERE Signal_Date IS NOT NULL")
            stocks = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(stocks)} unique stocks with sell signal dates")
            
            # For each stock, get the most recent signal
            results = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for stock in stocks:
                if 'update_date' in columns:
                    # If update_date exists, use it to find the most recent entry
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close, 
                              update_date, julianday('{current_date}') - julianday(Signal_Date) AS days_ago
                        FROM sell_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY update_date DESC, Signal_Date DESC
                        LIMIT 1
                    """
                else:
                    # Otherwise just use Signal_Date
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close,
                              julianday('{current_date}') - julianday(Signal_Date) AS days_ago
                        FROM sell_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY Signal_Date DESC
                        LIMIT 1
                    """
                
                cursor.execute(query, (stock,))
                row = cursor.fetchone()
                
                if row:
                    results.append(row)
            
            # Convert the results to a DataFrame
            if 'update_date' in columns:
                column_names = ['Stock', 'Date', 'Close', 'RSI_Weekly_Avg', 'AO_Weekly', 
                               'Signal_Date', 'Signal_Close', 'update_date', 'days_ago']
            else:
                column_names = ['Stock', 'Date', 'Close', 'RSI_Weekly_Avg', 'AO_Weekly', 
                               'Signal_Date', 'Signal_Close', 'days_ago']
                
            df = pd.DataFrame(results, columns=column_names)
            
            # Sort by the most recent update_date first, then by days_ago
            if 'update_date' in columns and not df.empty:
                df['update_date'] = pd.to_datetime(df['update_date'])
                df = df.sort_values(['update_date', 'days_ago'], ascending=[False, True])
            else:
                df = df.sort_values('days_ago')
                
            # Convert days_ago to integer
            if not df.empty:
                df['days_ago'] = df['days_ago'].astype(int)
                
            return df
            
    except Exception as e:
        logger.error(f"Error getting latest sell stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

class StockAnalysisError(Exception):
    """Base exception class for stock analysis errors"""
    pass

class DatabaseError(StockAnalysisError):
    """Exception raised for database-related errors"""
    pass

class DataProcessingError(StockAnalysisError):
    """Exception raised for data processing errors"""
    pass

class ChartGenerationError(StockAnalysisError):
    """Exception raised for chart generation errors"""
    pass

class TelegramError(StockAnalysisError):
    """Exception raised for Telegram-related errors"""
    pass

def handle_error(error: Exception, context: str = "") -> None:
    """Enhanced error handling with detailed logging and notification."""
    error_type = type(error).__name__
    error_msg = str(error)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create detailed error message
    error_details = {
        'timestamp': timestamp,
        'error_type': error_type,
        'error_message': error_msg,
        'context': context,
        'traceback': traceback.format_exc()
    }
    
    # Log error with different levels based on type
    if isinstance(error, DatabaseError):
        logger.error(f"Database Error: {error_msg}", extra=error_details)
    elif isinstance(error, DataProcessingError):
        logger.error(f"Data Processing Error: {error_msg}", extra=error_details)
    elif isinstance(error, ChartGenerationError):
        logger.error(f"Chart Generation Error: {error_msg}", extra=error_details)
    elif isinstance(error, TelegramError):
        logger.error(f"Telegram Notification Error: {error_msg}", extra=error_details)
    else:
        logger.error(f"Unexpected Error: {error_msg}", extra=error_details)
    
    # Send error notification if Telegram is configured
    try:
        config = Configuration()
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
            error_notification = (
                f"🚨 Error Alert\n"
                f"Type: {error_type}\n"
                f"Context: {context}\n"
                f"Message: {error_msg}\n"
                f"Time: {timestamp}"
            )
            notifier.send_message(error_notification)
    except Exception as e:
        logger.error(f"Failed to send error notification: {str(e)}")

class DataValidator:
    """Class for validating data before processing"""
    
    @staticmethod
    def validate_stock_data(df: pd.DataFrame, required_columns: list) -> bool:
        """Validate stock data DataFrame"""
        if df.empty:
            raise DataProcessingError("DataFrame is empty")
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataProcessingError(f"Missing required columns: {missing_columns}")
            
        if df['Date'].dtype != 'datetime64[ns]':
            raise DataProcessingError("Date column must be datetime type")
            
        if not df['Date'].is_monotonic_increasing:
            raise DataProcessingError("Date column must be sorted in ascending order")
            
        return True
    
    @staticmethod
    def validate_technical_indicators(df: pd.DataFrame) -> bool:
        """Validate technical indicators in DataFrame"""
        required_indicators = ['RSI', 'AO', 'Close', 'Volume']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        
        if missing_indicators:
            raise DataProcessingError(f"Missing required technical indicators: {missing_indicators}")
            
        if df['RSI'].isna().any():
            raise DataProcessingError("RSI column contains NaN values")
            
        if df['AO'].isna().any():
            raise DataProcessingError("AO column contains NaN values")
            
        if df['Close'].isna().any():
            raise DataProcessingError("Close column contains NaN values")
            
        if df['Volume'].isna().any():
            raise DataProcessingError("Volume column contains NaN values")
            
        return True
    
    @staticmethod
    def validate_signal_data(signal_df: pd.DataFrame) -> bool:
        """Validate signal data DataFrame"""
        required_columns = ['Stock', 'Date', 'Close', 'Signal_Date', 'Signal_Close']
        missing_columns = [col for col in required_columns if col not in signal_df.columns]
        
        if missing_columns:
            raise DataProcessingError(f"Missing required columns in signal data: {missing_columns}")
            
        if signal_df['Date'].dtype != 'datetime64[ns]':
            raise DataProcessingError("Date column must be datetime type")
            
        if signal_df['Signal_Date'].dtype != 'datetime64[ns]':
            raise DataProcessingError("Signal_Date column must be datetime type")
            
        return True

class ChartGenerator:
    """Class for generating stock analysis charts"""
    
    def __init__(self, symbol: str, df: pd.DataFrame):
        """Initialize chart generator"""
        self.symbol = symbol
        self.df = df
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        
    def setup_figure(self):
        """Setup the figure with subplots"""
        self.fig = plt.figure(figsize=config.CHART_FIGSIZE, dpi=config.CHART_DPI)
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
        
        self.ax1 = plt.subplot(gs[0])  # Price and MA
        self.ax2 = plt.subplot(gs[1], sharex=self.ax1)  # RSI
        self.ax3 = plt.subplot(gs[2], sharex=self.ax1)  # AO
        
        plt.subplots_adjust(hspace=0)
        
    def plot_price_and_ma(self):
        """Plot price and moving average"""
        self.ax1.plot(self.df['Date'], self.df['Close'], label='Close Price', color='blue')
        self.ax1.plot(self.df['Date'], self.df['MA'], label=f'MA{config.MA_PERIOD}', color='red')
        self.ax1.set_title(f'{self.symbol} Price and Indicators')
        self.ax1.legend()
        self.ax1.grid(True)
        
    def plot_rsi(self):
        """Plot RSI indicator"""
        self.ax2.plot(self.df['Date'], self.df['RSI'], label='RSI', color='purple')
        self.ax2.axhline(y=config.RSI_OVERSOLD, color='green', linestyle='--', label=f'RSI {config.RSI_OVERSOLD}')
        self.ax2.axhline(y=config.RSI_OVERBOUGHT, color='red', linestyle='--', label=f'RSI {config.RSI_OVERBOUGHT}')
        self.ax2.set_ylabel('RSI')
        self.ax2.legend()
        self.ax2.grid(True)
        
    def plot_ao(self):
        """Plot Awesome Oscillator"""
        self.ax3.plot(self.df['Date'], self.df['AO'], label='AO', color='orange')
        self.ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self.ax3.set_ylabel('AO')
        self.ax3.legend()
        self.ax3.grid(True)
        
    def add_signals(self, buy_signals: list, sell_signals: list):
        """Add buy and sell signals to the chart"""
        for date, price in buy_signals:
            self.ax1.scatter(date, price, color='green', marker='^', s=100, label='Buy Signal')
            
        for date, price in sell_signals:
            self.ax1.scatter(date, price, color='red', marker='v', s=100, label='Sell Signal')
            
        # Remove duplicate labels
        handles, labels = self.ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax1.legend(by_label.values(), by_label.keys())
        
    def save_chart(self, output_path: str):
        """Save the chart to file"""
        plt.savefig(output_path, bbox_inches='tight', dpi=config.CHART_DPI)
        plt.close()
        
    def generate_chart(self, buy_signals: list, sell_signals: list, output_path: str):
        """Generate complete chart with all components"""
        try:
            self.setup_figure()
            self.plot_price_and_ma()
            self.plot_rsi()
            self.plot_ao()
            self.add_signals(buy_signals, sell_signals)
            self.save_chart(output_path)
            return True
        except Exception as e:
            raise ChartGenerationError(f"Error generating chart for {self.symbol}: {str(e)}")

class DashboardGenerator:
    """Class for generating market dashboard"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize dashboard generator"""
        self.df = df
        self.validator = DataValidator()
        self.fig = None
        self.ax = None
        
    def setup_figure(self):
        """Setup the figure for dashboard"""
        self.fig, self.ax = plt.subplots(figsize=config.CHART_FIGSIZE, dpi=config.CHART_DPI)
        
    def plot_market_phases(self):
        """Plot market phases distribution"""
        phase_counts = self.df['Market_Phase'].value_counts()
        colors = ['green', 'red', 'gray']
        
        self.ax.pie(phase_counts, labels=phase_counts.index, colors=colors, autopct='%1.1f%%')
        self.ax.set_title('Market Phase Distribution')
        
    def add_summary_statistics(self):
        """Add summary statistics to dashboard"""
        total_stocks = len(self.df)
        buy_count = len(self.df[self.df['Status'] == 'BUY/HOLD'])
        sell_count = len(self.df[self.df['Status'] == 'SELL'])
        opp_count = len(self.df[self.df['Status'] == 'OPPORTUNITY'])
        
        stats_text = (
            f"Total Stocks: {total_stocks}\n"
            f"BUY/HOLD Signals: {buy_count} ({buy_count/total_stocks*100:.1f}%)\n"
            f"SELL Signals: {sell_count} ({sell_count/total_stocks*100:.1f}%)\n"
            f"OPPORTUNITY Signals: {opp_count} ({opp_count/total_stocks*100:.1f}%)"
        )
        
        self.ax.text(1.2, 0.5, stats_text, transform=self.ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
    def add_portfolio_recommendations(self):
        """Add portfolio recommendations to dashboard"""
        # Get top 5 stocks by phase score for each category
        buy_stocks = self.df[self.df['Status'] == 'BUY/HOLD'].nlargest(5, 'Phase_Score')
        sell_stocks = self.df[self.df['Status'] == 'SELL'].nlargest(5, 'Phase_Score')
        opp_stocks = self.df[self.df['Status'] == 'OPPORTUNITY'].nlargest(5, 'Phase_Score')
        
        recommendations = (
            "Top 5 BUY/HOLD Stocks:\n" +
            "\n".join([f"{row['Stock']} ({row['Phase_Score']:.1f})" for _, row in buy_stocks.iterrows()]) +
            "\n\nTop 5 SELL Stocks:\n" +
            "\n".join([f"{row['Stock']} ({row['Phase_Score']:.1f})" for _, row in sell_stocks.iterrows()]) +
            "\n\nTop 5 OPPORTUNITY Stocks:\n" +
            "\n".join([f"{row['Stock']} ({row['Phase_Score']:.1f})" for _, row in opp_stocks.iterrows()])
        )
        
        self.ax.text(1.2, 0.2, recommendations, transform=self.ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
    def save_dashboard(self, output_path: str):
        """Save dashboard to file"""
        plt.savefig(output_path, bbox_inches='tight', dpi=config.CHART_DPI)
        plt.close()
        
    def generate_dashboard(self, output_path: str) -> bool:
        """Generate complete dashboard"""
        try:
            self.validator.validate_stock_data(self.df, ['Stock', 'Status', 'Market_Phase', 'Phase_Score'])
            
            self.setup_figure()
            self.plot_market_phases()
            self.add_summary_statistics()
            self.add_portfolio_recommendations()
            self.save_dashboard(output_path)
            
            return True
            
        except Exception as e:
            raise ChartGenerationError(f"Error generating dashboard: {str(e)}")

class TelegramNotifier:
    """Class for sending notifications via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """Initialize Telegram notifier"""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logger
        self.logger.info("Initializing Telegram notifier")
        
    def send_message(self, message: str) -> bool:
        """Send a text message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            if response.status_code == 200:
                self.logger.info("Successfully sent message to Telegram")
                return True
            else:
                self.logger.error(f"Failed to send message to Telegram. Status code: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error sending message to Telegram: {e}")
            return False
            
    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo to Telegram"""
        try:
            if not os.path.exists(photo_path):
                self.logger.error(f"Photo file not found: {photo_path}")
                return False
                
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            
            # Open the file in binary mode
            with open(photo_path, 'rb') as photo:
                files = {
                    'photo': photo
                }
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'HTML'
                }
                
                self.logger.info(f"Attempting to send photo {photo_path} to Telegram")
                response = requests.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    self.logger.info(f"Successfully sent photo {photo_path} to Telegram")
                    return True
                else:
                    self.logger.error(f"Failed to send photo to Telegram. Status code: {response.status_code}")
                    self.logger.error(f"Response: {response.text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error sending photo to Telegram: {e}")
            return False
        finally:
            # Ensure the file is closed
            if 'files' in locals() and 'photo' in files:
                files['photo'].close()

class StockAnalysisExecutor:
    """Main class for executing stock analysis"""
    
    def __init__(self):
        """Initialize stock analysis executor"""
        self.config = config
        self.logger = logger
        self.db_manager = DatabaseConnectionManager(self.config.MAIN_DB_PATH)
        self.signals_db_manager = DatabaseConnectionManager(self.config.SIGNALS_DB_PATH)
        self.history = StockAnalysisHistory(self.db_manager)
        self.telegram_notifier = None
        self.charts_sent = 0
        
        # Initialize Telegram notifier if credentials are available
        if self.config.TELEGRAM_BOT_TOKEN and self.config.TELEGRAM_CHAT_ID:
            self.telegram_notifier = TelegramNotifier(
                self.config.TELEGRAM_BOT_TOKEN,
                self.config.TELEGRAM_CHAT_ID
            )
            logger.info("Telegram notifier initialized")
        else:
            logger.warning("Telegram credentials not configured")

    def run(self):
        """Run the stock analysis process."""
        try:
            # Reset chart counter
            self.charts_sent = 0
            
            # Get latest buy signals
            buy_signals = get_latest_buy_stocks()
            if buy_signals.empty:
                logger.warning("No buy signals found")
                return
            
            # Send signal tables first
            if self.telegram_notifier:
                send_signal_tables_to_telegram(self.telegram_notifier)
            
            # Process each buy signal
            for _, signal in buy_signals.iterrows():
                if self.charts_sent >= 10:
                    break
                    
                symbol = signal['symbol']
                logger.info(f"Processing buy signal for {symbol}")
                
                # Analyze stock and get chart
                phase, score, details = self.analyze_stock(symbol, send_to_telegram=True)
                
                if details and self.telegram_notifier:
                    # Create detailed caption
                    caption = (
                        f"📈 {symbol} Analysis\n"
                        f"Phase: {phase}\n"
                        f"Score: {score:.1f}\n"
                        f"Signal Date: {signal['signal_date']}\n"
                        f"Price: {signal['price']:.2f}\n"
                        f"RSI: {signal['rsi']:.1f}\n"
                        f"AO: {signal['ao']:.3f}"
                    )
                    
                    # Send chart
                    if self.telegram_notifier.send_photo(details['chart_path'], caption):
                        self.charts_sent += 1
                        logger.info(f"Successfully sent chart for {symbol} to Telegram")
                    else:
                        logger.error(f"Failed to send chart for {symbol} to Telegram")
                
                time.sleep(1)  # Prevent rate limiting
            
            logger.info(f"Successfully processed {self.charts_sent} buy signals")
            
        except Exception as e:
            logger.error(f"Error in run method: {str(e)}")
            raise

    def analyze_stock(self, symbol: str, send_to_telegram: bool = False) -> tuple:
        """Analyze a stock and optionally send chart to Telegram"""
        try:
            self.logger.info(f"Analyzing stock {symbol} (send_to_telegram={send_to_telegram})")
            
            # Get stock data
            table_name = f"PSX_{symbol}_stock_data"
            df = get_stock_data(table_name)
            
            if df.empty:
                self.logger.error(f"No data found for {symbol}")
                return "NEUTRAL", 50, {'error': 'No data found'}
                
            # Calculate market phase and score
            analyzer = SignalAnalyzer(df)
            phase, score, details = analyzer.calculate_market_phase()
            
            # Save analysis to history
            self.history.add_analysis(symbol, phase, score)
            
            # Generate and send chart if requested
            if send_to_telegram and self.telegram_notifier and self.charts_sent < 10:
                # Generate chart
                plot_filename, title_text = draw_indicator_trend_lines_with_signals(symbol)
                if not plot_filename:
                    self.logger.error(f"Failed to generate chart for {symbol}")
                    return phase, score, details
                    
                # Send to Telegram
                caption = (
                    f"{title_text}\n"
                    f"Phase: {phase}\n"
                    f"Score: {score:.1f}"
                )
                self.logger.info(f"Attempting to send chart for {symbol} to Telegram")
                success = self.telegram_notifier.send_photo(plot_filename, caption)
                if success:
                    self.charts_sent += 1
                    self.logger.info(f"Successfully sent chart for {symbol} to Telegram (chart {self.charts_sent} of 10)")
                else:
                    self.logger.error(f"Failed to send chart for {symbol} to Telegram")
                    
                details['chart_path'] = plot_filename
            
            return phase, score, details
            
        except Exception as e:
            self.logger.error(f"Error analyzing stock {symbol}: {e}")
            return "NEUTRAL", 50, {'error': str(e)}
            
    def get_available_symbols(self) -> list:
        """Get list of available stock symbols"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                return get_available_symbols(cursor)
        except Exception as e:
            handle_error(e, "getting available symbols")
            return []
            
    def run(self):
        """Run the stock analysis process"""
        try:
            # Get latest buy signals
            buy_df = get_latest_buy_stocks()
            if buy_df.empty:
                self.logger.error("No buy signals found")
                return
                
            # Get the latest 10 buy signals
            latest_buy_signals = buy_df.head(10)['Stock'].tolist()
            self.logger.info(f"Found {len(latest_buy_signals)} latest buy signals")
            
            # Process latest buy signals and send to Telegram
            self.telegram_charts_sent = 0  # Reset counter
            for symbol in latest_buy_signals:
                self.logger.info(f"Processing buy signal for: {symbol}")
                self.analyze_stock(symbol, send_to_telegram=True)
                
            # Send summary message if any charts were sent
            if self.notifier and self.telegram_charts_sent > 0:
                message = f"📊 Latest {self.telegram_charts_sent} Buy Signals Analysis\n\n"
                for symbol in latest_buy_signals[:self.telegram_charts_sent]:
                    history = self.history.get_stock_analysis_history(symbol, limit=1)
                    if history:
                        date, phase, score = history[0]
                        message += f"• {symbol} ({phase}, Score: {score:.1f})\n"
                self.logger.info("Sending summary message to Telegram")
                self.notifier.send_message(message)
                self.logger.info(f"Sent summary message for {self.telegram_charts_sent} stocks")
            else:
                self.logger.warning("No charts were sent to Telegram")
                
            # Generate market dashboard locally
            self.generate_market_dashboard()
            
        except Exception as e:
            self.logger.error(f"Error running stock analysis: {e}")
            
    def generate_market_dashboard(self) -> pd.DataFrame:
        """Generate market dashboard without sending to Telegram"""
        try:
            # Get all analyzed stocks
            all_symbols = self.get_available_symbols()
            dashboard_data = []
            
            for symbol in all_symbols:
                try:
                    phase, score, details = self.analyze_stock(symbol, send_to_telegram=False)
                    dashboard_data.append({
                        'Symbol': symbol,
                        'Phase': phase,
                        'Score': score,
                        'Details': details
                    })
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Create dashboard DataFrame
            dashboard_df = pd.DataFrame(dashboard_data)
            
            # Generate dashboard visualization locally
            if not dashboard_df.empty:
                dashboard_generator = DashboardGenerator(dashboard_df)
                dashboard_path = os.path.join(self.config.DASHBOARDS_DIR, f"market_dashboard_{datetime.now().strftime('%Y%m%d')}.png")
                dashboard_generator.generate_dashboard(dashboard_path)
            
            return dashboard_df
            
        except Exception as e:
            handle_error(e, "generating market dashboard")
            return pd.DataFrame()

class StockAnalysisHistory:
    """Class for managing stock analysis history"""
    
    def __init__(self, db_manager: DatabaseConnectionManager):
        """Initialize stock analysis history"""
        self.db_manager = db_manager
        self._ensure_table_exists()
        
    def _ensure_table_exists(self):
        """Ensure the stock_analysis_history table exists"""
        try:
            query = """
            CREATE TABLE IF NOT EXISTS stock_analysis_history (
                stock TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                phase TEXT,
                score REAL,
                PRIMARY KEY (stock, analysis_date)
            )
            """
            with self.db_manager.get_connection() as conn:
                conn.execute(query)
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating stock_analysis_history table: {e}")
            
    def add_analysis(self, stock: str, phase: str, score: float):
        """Add a new stock analysis record"""
        try:
            query = """
            INSERT OR REPLACE INTO stock_analysis_history (stock, phase, score)
            VALUES (?, ?, ?)
            """
            with self.db_manager.get_connection() as conn:
                conn.execute(query, (stock, phase, score))
                conn.commit()
                logger.info(f"Added analysis record for {stock}: Phase={phase}, Score={score}")
        except Exception as e:
            logger.error(f"Error adding analysis record for {stock}: {e}")
            
    def get_latest_analyzed_stocks(self, limit: int = 10) -> list:
        """Get the most recently analyzed stocks"""
        try:
            query = """
            SELECT DISTINCT stock
            FROM stock_analysis_history
            ORDER BY analysis_date DESC
            LIMIT ?
            """
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (limit,))
                results = cursor.fetchall()
                stocks = [row[0] for row in results]
                logger.info(f"Retrieved {len(stocks)} latest analyzed stocks: {stocks}")
                return stocks
        except Exception as e:
            logger.error(f"Error getting latest analyzed stocks: {e}")
            return []
            
    def get_stock_analysis_history(self, stock: str, limit: int = 10) -> list:
        """Get analysis history for a specific stock"""
        try:
            query = """
            SELECT analysis_date, phase, score
            FROM stock_analysis_history
            WHERE stock = ?
            ORDER BY analysis_date DESC
            LIMIT ?
            """
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (stock, limit))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting analysis history for {stock}: {e}")
            return []

class ChartScheduler:
    """Manages scheduled chart generation and distribution"""
    
    def __init__(self, executor: 'StockAnalysisExecutor'):
        self.executor = executor
        self.history = StockAnalysisHistory(executor.db_manager)
        self.last_monthly_run = None
        self.daily_charts_sent = False
        
    def should_run_monthly(self) -> bool:
        """Check if it's time to run monthly full list"""
        today = datetime.now()
        # Check if it's Monday and first week of month
        is_monday = today.weekday() == 0
        is_first_week = today.day <= 7
        
        # Check if we haven't run this month
        if self.last_monthly_run:
            last_run_month = self.last_monthly_run.month
            last_run_year = self.last_monthly_run.year
            if last_run_month == today.month and last_run_year == today.year:
                return False
                
        return is_monday and is_first_week
        
    def should_run_daily(self) -> bool:
        """Check if daily charts should be sent"""
        if not self.daily_charts_sent:
            return True
            
        # Reset daily flag at midnight
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            self.daily_charts_sent = False
            return True
            
        return False
        
    def get_stock_category(self, phase: str, score: float) -> str:
        """Determine stock category based on phase and score"""
        if phase == "ACCUMULATION" and score >= 70:
            return "BUY/HOLD"
        elif phase == "DISTRIBUTION" and score <= 30:
            return "SELL"
        else:
            return "OPPORTUNITY"
            
    def send_latest_charts(self):
        """Send charts for 10 most recently analyzed stocks"""
        try:
            # Get latest 10 stocks
            latest_stocks = self.history.get_latest_analyzed_stocks(limit=10)
            
            if not latest_stocks:
                self.executor.logger.warning("No recently analyzed stocks found")
                return
                
            # Generate and send charts
            message = "📊 Latest 10 Stock Analyses\n\n"
            for stock in latest_stocks:
                try:
                    # Analyze stock and get phase/score
                    phase, score, _ = self.executor.analyze_stock(stock)
                    
                    # Generate chart
                    chart_path = os.path.join(self.executor.config.CHARTS_DIR, f"{stock}_analysis.png")
                    
                    # Send chart with analysis details
                    caption = (
                        f"Analysis for {stock}\n"
                        f"Phase: {phase}\n"
                        f"Score: {score:.1f}"
                    )
                    self.executor.notifier.send_photo(chart_path, caption)
                    
                    message += f"• {stock} ({phase}, Score: {score:.1f})\n"
                    
                except Exception as e:
                    self.executor.logger.error(f"Error processing {stock}: {e}")
                    continue
                    
            # Send summary message
            self.executor.notifier.send_message(message)
            self.daily_charts_sent = True
            
        except Exception as e:
            handle_error(e, "sending latest charts")
            
    def send_monthly_full_list(self):
        """Send charts for all stocks on first Monday of month"""
        try:
            if not self.should_run_monthly():
                return
                
            # Get all stocks
            all_stocks = self.executor.get_available_symbols()
            
            # Categorize stocks
            categories = {
                'BUY/HOLD': [],
                'SELL': [],
                'OPPORTUNITY': []
            }
            
            # Process each stock
            for stock in tqdm(all_stocks, desc="Processing monthly charts"):
                try:
                    # Analyze stock
                    phase, score, _ = self.executor.analyze_stock(stock)
                    category = self.get_stock_category(phase, score)
                    categories[category].append((stock, score))
                    
                except Exception as e:
                    self.executor.logger.error(f"Error processing {stock}: {e}")
                    continue
                    
            # Send summary message
            message = "📈 Monthly Full Market Analysis\n\n"
            for category, stocks in categories.items():
                message += f"\n{category} Stocks:\n"
                # Sort stocks by score
                stocks.sort(key=lambda x: x[1], reverse=True)
                for stock, score in stocks:
                    message += f"• {stock} (Score: {score:.1f})\n"
                    
            self.executor.notifier.send_message(message)
            self.last_monthly_run = datetime.now()
            
        except Exception as e:
            handle_error(e, "sending monthly full list")

class SignalAnalyzer:
    """Enhanced signal analyzer with validation and error handling."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the signal analyzer with data validation."""
        if df is None or df.empty:
            raise DataProcessingError("Input DataFrame is empty or None")
        
        required_columns = ['close', 'rsi', 'ao', 'volume', 'ma']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataProcessingError(f"Missing required columns: {missing_columns}")
        
        self.df = df.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Validate the input data for analysis."""
        # Check for NaN values
        nan_columns = self.df.columns[self.df.isna().any()].tolist()
        if nan_columns:
            logger.warning(f"NaN values found in columns: {nan_columns}")
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        # Validate numeric ranges
        if (self.df['rsi'] < 0).any() or (self.df['rsi'] > 100).any():
            raise DataProcessingError("RSI values must be between 0 and 100")
        
        if (self.df['close'] <= 0).any():
            raise DataProcessingError("Price values must be positive")
        
        if (self.df['volume'] < 0).any():
            raise DataProcessingError("Volume values must be non-negative")
    
    def calculate_market_phase(self) -> tuple:
        """Calculate market phase with enhanced analysis."""
        try:
            # Calculate individual scores
            rsi_score = self._analyze_rsi()
            ao_score = self._analyze_ao()
            volume_score = self._analyze_volume()
            ma_score = self._analyze_ma()
            pattern_score = self._analyze_patterns()
            
            # Calculate final score
            final_score = self._calculate_final_score(
                rsi_score, ao_score, volume_score, ma_score, pattern_score
            )
            
            # Determine market phase
            phase = self._determine_market_phase(final_score)
            
            # Create details dictionary
            details = {
                'rsi_score': rsi_score,
                'ao_score': ao_score,
                'volume_score': volume_score,
                'ma_score': ma_score,
                'pattern_score': pattern_score,
                'final_score': final_score
            }
            
            return phase, final_score, details
            
        except Exception as e:
            raise DataProcessingError(f"Failed to calculate market phase: {str(e)}")
    
    def _analyze_rsi(self) -> float:
        """Analyze RSI with trend detection."""
        try:
            current_rsi = self.df['rsi'].iloc[-1]
            rsi_trend = self.df['rsi'].diff().iloc[-5:].mean()
            
            # Score based on RSI value and trend
            if current_rsi < 30:
                base_score = 2.0
            elif current_rsi < 40:
                base_score = 1.5
            elif current_rsi > 70:
                base_score = 0.5
            elif current_rsi > 60:
                base_score = 1.0
            else:
                base_score = 1.0
            
            # Adjust score based on trend
            if rsi_trend > 0:
                base_score *= 1.2
            elif rsi_trend < 0:
                base_score *= 0.8
            
            return min(max(base_score, 0), 2.0)
            
        except Exception as e:
            raise DataProcessingError(f"Failed to analyze RSI: {str(e)}")
    
    def _analyze_ao(self) -> float:
        """Analyze Awesome Oscillator with momentum detection."""
        try:
            current_ao = self.df['ao'].iloc[-1]
            ao_trend = self.df['ao'].diff().iloc[-5:].mean()
            
            # Score based on AO value and trend
            if current_ao > 0 and ao_trend > 0:
                return 2.0
            elif current_ao > 0:
                return 1.5
            elif current_ao < 0 and ao_trend < 0:
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            raise DataProcessingError(f"Failed to analyze AO: {str(e)}")
    
    def _analyze_volume(self) -> float:
        """Analyze volume with trend confirmation."""
        try:
            avg_volume = self.df['volume'].rolling(20).mean()
            current_volume = self.df['volume'].iloc[-1]
            volume_trend = self.df['volume'].diff().iloc[-5:].mean()
            
            # Score based on volume relative to average and trend
            volume_ratio = current_volume / avg_volume.iloc[-1]
            
            if volume_ratio > 1.5 and volume_trend > 0:
                return 2.0
            elif volume_ratio > 1.2:
                return 1.5
            elif volume_ratio < 0.8 and volume_trend < 0:
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            raise DataProcessingError(f"Failed to analyze volume: {str(e)}")
    
    def _analyze_ma(self) -> float:
        """Analyze Moving Average relationship."""
        try:
            current_price = self.df['close'].iloc[-1]
            current_ma = self.df['ma'].iloc[-1]
            price_ma_ratio = current_price / current_ma
            
            if price_ma_ratio > 1.05:
                return 2.0
            elif price_ma_ratio > 1.02:
                return 1.5
            elif price_ma_ratio < 0.95:
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            raise DataProcessingError(f"Failed to analyze MA: {str(e)}")
    
    def _analyze_patterns(self) -> float:
        """Analyze price patterns."""
        try:
            # Simple pattern detection
            last_5_prices = self.df['close'].iloc[-5:]
            price_changes = last_5_prices.diff()
            
            # Check for uptrend
            if (price_changes > 0).all():
                return 2.0
            # Check for downtrend
            elif (price_changes < 0).all():
                return 0.5
            # Check for consolidation
            elif abs(price_changes).mean() < 0.01:
                return 1.0
            else:
                return 1.5
                
        except Exception as e:
            raise DataProcessingError(f"Failed to analyze patterns: {str(e)}")
    
    def _calculate_final_score(self, rsi_score: float, ao_score: float,
                             volume_score: float, ma_score: float,
                             pattern_score: float) -> float:
        """Calculate final market phase score with weighted components."""
        weights = {
            'rsi': 0.25,
            'ao': 0.25,
            'volume': 0.20,
            'ma': 0.15,
            'pattern': 0.15
        }
        
        final_score = (
            rsi_score * weights['rsi'] +
            ao_score * weights['ao'] +
            volume_score * weights['volume'] +
            ma_score * weights['ma'] +
            pattern_score * weights['pattern']
        )
        
        return min(max(final_score, 0), 2.0)
    
    def _determine_market_phase(self, score: float) -> str:
        """Determine market phase based on final score."""
        if score >= 1.5:
            return "BULLISH"
        elif score <= 0.5:
            return "BEARISH"
        else:
            return "NEUTRAL"

def get_stock_data(table_name: str) -> pd.DataFrame:
    """Get stock data from the database"""
    try:
        conn = sqlite3.connect(config.MAIN_DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        if not df.empty:
            # Convert Date column to datetime index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Calculate MA if not present
            if 'MA' not in df.columns:
                df['MA'] = df['Close'].rolling(window=config.MA_PERIOD).mean()
                
        return df
        
    except Exception as e:
        logger.error(f"Error getting stock data for {table_name}: {e}")
        return pd.DataFrame()

def create_signal_tables():
    """Create detailed tables for buy, sell, and opportunity signals."""
    try:
        # Get latest signals
        buy_df = get_latest_buy_stocks()
        sell_df = get_latest_sell_stocks()
        
        # Create opportunity signals (stocks with high scores)
        opportunity_df = pd.DataFrame()
        if not buy_df.empty:
            opportunity_df = buy_df[buy_df['score'] >= 7.0].copy()
        
        # Format tables for Telegram
        buy_table = format_signal_table(buy_df, "BUY")
        sell_table = format_signal_table(sell_df, "SELL")
        opportunity_table = format_signal_table(opportunity_df, "OPPORTUNITY")
        
        return buy_table, sell_table, opportunity_table
    except Exception as e:
        logger.error(f"Error creating signal tables: {str(e)}")
        return "", "", ""

def format_signal_table(df: pd.DataFrame, signal_type: str) -> str:
    """Format signal data into a readable table."""
    if df.empty:
        return f"No {signal_type} signals available"
    
    try:
        # Select and rename columns
        columns = {
            'symbol': 'Symbol',
            'signal_date': 'Date',
            'price': 'Price',
            'score': 'Score',
            'phase': 'Phase',
            'rsi': 'RSI',
            'ao': 'AO',
            'volume': 'Volume'
        }
        
        # Format the data
        formatted_df = df[columns.keys()].copy()
        formatted_df.columns = columns.values()
        
        # Format numeric columns
        formatted_df['Price'] = formatted_df['Price'].map('{:.2f}'.format)
        formatted_df['Score'] = formatted_df['Score'].map('{:.1f}'.format)
        formatted_df['RSI'] = formatted_df['RSI'].map('{:.1f}'.format)
        formatted_df['AO'] = formatted_df['AO'].map('{:.3f}'.format)
        formatted_df['Volume'] = formatted_df['Volume'].map('{:.0f}'.format)
        
        # Create table
        table = f"📊 {signal_type} SIGNALS\n\n"
        table += tabulate(formatted_df, headers='keys', tablefmt='grid', showindex=False)
        return table
    except Exception as e:
        logger.error(f"Error formatting {signal_type} table: {str(e)}")
        return f"Error formatting {signal_type} signals"

def send_signal_tables_to_telegram(notifier: TelegramNotifier):
    """Send formatted signal tables to Telegram."""
    try:
        # Create tables
        buy_table, sell_table, opportunity_table = create_signal_tables()
        
        # Send each table
        if buy_table:
            notifier.send_message(buy_table)
            time.sleep(1)  # Prevent rate limiting
        
        if sell_table:
            notifier.send_message(sell_table)
            time.sleep(1)
        
        if opportunity_table:
            notifier.send_message(opportunity_table)
            time.sleep(1)
            
        logger.info("Successfully sent signal tables to Telegram")
    except Exception as e:
        logger.error(f"Error sending signal tables to Telegram: {str(e)}")

