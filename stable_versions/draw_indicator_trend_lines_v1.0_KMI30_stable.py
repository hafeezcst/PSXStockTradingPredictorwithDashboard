import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import sqlite3
import os
import logging
from datetime import datetime, timedelta
from telegram_message import send_telegram_message_with_image
from telegram_message import send_telegram_message
from tabulate import tabulate
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import yaml
import json
import traceback
from functools import partial
import time
import requests
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
def validate_telegram_config(config):
    """Validate Telegram configuration and return True if valid"""
    try:
        bot_token = config['telegram']['bot_token']
        chat_id = config['telegram']['chat_id']
        
        if not bot_token or not chat_id:
            logging.error("Telegram configuration is missing. Please set bot_token and chat_id in config.yaml")
            return False
            
        # Validate chat_id format
        try:
            # Remove any whitespace and convert to string
            chat_id = str(chat_id).strip()
            # Check if it's a valid numeric value (including negative numbers)
            if not chat_id.lstrip('-').isdigit():
                raise ValueError("Chat ID must be a numeric value")
                
            # Validate the chat_id with Telegram API
            test_url = f"https://api.telegram.org/bot{bot_token}/getChat"
            test_data = {'chat_id': chat_id}
            test_response = requests.post(test_url, data=test_data)
            
            if test_response.status_code == 400:
                error_data = test_response.json()
                if 'description' in error_data:
                    if 'chat not found' in error_data['description'].lower():
                        logging.error(f"Chat not found: {chat_id}")
                        logging.error("Please make sure:")
                        logging.error("1. The bot is added to the group/channel")
                        logging.error("2. The bot has admin rights in the group/channel")
                        logging.error("3. The chat_id is correct")
                        logging.error("To verify your chat_id:")
                        logging.error("1. Send a message to your bot")
                        logging.error("2. Visit: https://api.telegram.org/bot{bot_token}/getUpdates")
                        logging.error("3. Look for the 'chat' object in the response")
                        logging.error("4. Copy the 'id' value exactly as shown")
                    else:
                        logging.error(f"Telegram API error: {error_data['description']}")
                return False
                
            test_response.raise_for_status()
            return True
            
        except ValueError as e:
            logging.error(f"Invalid chat_id format: {chat_id}")
            logging.error(str(e))
            return False
        except requests.exceptions.HTTPError as e:
            logging.error(f"Error validating chat_id: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error validating chat_id: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Error validating Telegram configuration: {e}")
        return False

def load_config():
    """Load configuration from config.yaml or create default if not exists"""
    config_path = 'config.yaml'
    default_config = {
        'database': {
            'main_db': 'data/databases/production/PSX_investing_Stocks.db',
            'signals_db': 'data/databases/production/PSX_investing_Stocks_KMI30.db'
        },
        'output': {
            'charts_folder': 'outputs/charts/RSI_AO_CHARTS',
            'dashboards_folder': 'outputs/dashboards/PSX_DASHBOARDS'
        },
        'telegram': {
            'max_images_per_message': 10,
            'bot_token': "6860197701:AAESTzERZLYbqyU6gFKfAwJQL8jJ_HNKLbM",
            'chat_id': "-4152327824"  # From step 1
        },
        'analysis': {
            'lookback_years': 10,
            'rsi_thresholds': [40, 60],
            'ma_periods': [10, 30, 50],
            'volume_ma_period': 20,
            'ao_fast_period': 5,
            'ao_slow_period': 34
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'file': 'logs/psx_analysis.log'
        }
    }

    try:
        # Create necessary directories first
        os.makedirs('data/databases/production', exist_ok=True)
        os.makedirs('outputs/charts/RSI_AO_CHARTS', exist_ok=True)
        os.makedirs('outputs/dashboards/PSX_DASHBOARDS', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Ensure all required keys exist
            for section, default_values in default_config.items():
                if section not in config:
                    config[section] = default_values
                else:
                    for key, value in default_values.items():
                        if key not in config[section]:
                            config[section][key] = value
        else:
            config = default_config
            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logging.warning(f"Created default config file at {config_path}")
            logging.warning("Please update the following settings in config.yaml:")
            logging.warning("1. Telegram bot_token and chat_id")
            logging.warning("2. Database file paths if different from defaults")
            
        # Check if database files exist
        missing_dbs = []
        if not os.path.exists(config['database']['main_db']):
            missing_dbs.append(('main_db', config['database']['main_db']))
        if not os.path.exists(config['database']['signals_db']):
            missing_dbs.append(('signals_db', config['database']['signals_db']))
            
        if missing_dbs:
            logging.error("Missing required database files:")
            for db_type, db_path in missing_dbs:
                logging.error(f"- {db_type}: {db_path}")
            logging.error("\nTo fix this:")
            logging.error("1. Download the required database files:")
            logging.error("   - Main DB: PSX_investing_Stocks.db")
            logging.error("   - Signals DB: PSX_investing_Stocks_KMI30.db")
            logging.error("2. Create the directory structure if it doesn't exist:")
            logging.error("   mkdir -p data/databases/production")
            logging.error("3. Place the database files in the correct location:")
            logging.error("   - Main DB: data/databases/production/PSX_investing_Stocks.db")
            logging.error("   - Signals DB: data/databases/production/PSX_investing_Stocks_KMI30.db")
            logging.error("4. Or update the paths in config.yaml to point to your database locations")
            logging.error("\nNote: The script will continue with default configuration")
            logging.error("but will not be able to process data until the databases are available.")
            logging.error("\nIf you need help obtaining the database files:")
            logging.error("1. Check the project documentation")
            logging.error("2. Contact the project maintainers")
            logging.error("3. Check if there's a database initialization script")
            return default_config
            
        # Validate Telegram configuration
        if not config['telegram']['bot_token'] or not config['telegram']['chat_id']:
            logging.warning("Telegram configuration is missing. The bot will not send messages.")
            logging.warning("To enable Telegram notifications:")
            logging.warning("1. Create a bot using @BotFather on Telegram")
            logging.warning("2. Get your chat_id by sending a message to your bot and visiting:")
            logging.warning(f"   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
            logging.warning("3. Update the bot_token and chat_id in config.yaml")
            # Continue without Telegram
            config['telegram']['enabled'] = False
        else:
            config['telegram']['enabled'] = True
            
        return config
        
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        logging.error("Using default configuration")
        return default_config

def setup_logging(config=None):
    """Setup logging configuration"""
    try:
        if config is None:
            # Use default logging configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filename='logs/psx_analysis.log'
            )
            logging.warning("Using default logging configuration")
            return
            
        # Use configuration from config file
        log_level = getattr(logging, config['logging']['level'])
        log_format = config['logging']['format']
        log_file = config['logging']['file']
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=log_file
        )
        
    except Exception as e:
        # Fallback to default logging if there's an error
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='logs/psx_analysis.log'
        )
        logging.error(f"Error setting up logging: {e}")

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception as e:
        print(f"Warning: Error loading .env file: {e}")
        # Continue without .env file

# Load configuration
config = load_config()

# Setup logging with the loaded config
setup_logging(config)

# 1. UTILITY FUNCTIONS
def execute_with_retry(func, *args, max_retries=3, delay=1, backoff_factor=2, **kwargs):
    """Execute a function with exponential backoff retry logic"""
    import time
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if attempt < max_retries - 1:
                wait_time = delay * (backoff_factor ** attempt)
                logging.warning(f"Operation failed, retrying in {wait_time:.2f}s ({attempt+1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                raise

def check_database_files():
    """Check if required database files exist and are accessible"""
    try:
        if config is None:
            logging.error("Configuration not loaded")
            return False
            
        required_dbs = {
            'main_db': config['database']['main_db'],
            'signals_db': config['database']['signals_db']
        }
        
        missing_dbs = []
        for db_type, db_path in required_dbs.items():
            if not os.path.exists(db_path):
                missing_dbs.append((db_type, db_path))
                
        if missing_dbs:
            logging.error("Required database files are missing:")
            for db_type, db_path in missing_dbs:
                logging.error(f"- {db_type}: {db_path}")
            logging.error("\nPlease ensure all database files are in place before running the analysis.")
            return False
            
        # Try to open the databases to verify they're valid SQLite files
        for db_type, db_path in required_dbs.items():
            try:
                logging.info(f"Attempting to connect to {db_type} at {db_path}")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                logging.info(f"Successfully connected to {db_type}")
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if not tables:
                    logging.error(f"Database {db_type} exists but contains no tables: {db_path}")
                    conn.close()
                    return False
                logging.info(f"Found {len(tables)} tables in {db_type}")
                conn.close()
                logging.info(f"Successfully closed connection to {db_type}")
            except sqlite3.Error as e:
                logging.error(f"Error accessing {db_type} database at {db_path}: {str(e)}")
                return False
                
        logging.info("All database checks passed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error checking database files: {str(e)}")
        return False

def create_default_symbols_file():
    """Create a default KMI30 symbols file if it doesn't exist"""
    try:
        file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        if os.path.exists(file_path):
            logging.info(f"Symbols file already exists at {file_path}")
            return True
            
        # Default KMI30 symbols if file doesn't exist
        kmi30_symbols = [
            'AICL', 'ATRL', 'BAFL', 'BAHL', 'CNERGY', 'EFERT', 'ENGRO', 
            'FFBL', 'FFC', 'FCCL', 'HUBC', 'HBL', 'ISL', 'ILP', 'LUCK', 
            'MCB', 'MARI', 'MEBL', 'MLCF', 'MTL', 'NBP', 'NML', 'OGDC', 
            'PAKT', 'PPL', 'PIOC', 'PSO', 'SNGP', 'SSGC', 'UBL'
        ]
        symbols_df = pd.DataFrame(kmi30_symbols, columns=['Symbol'])
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            symbols_df.to_excel(writer, sheet_name='KMI30', index=False)
        logging.info(f"Created default symbols file at {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error creating symbols file: {e}")
        return False

def get_kmi30_symbols():
    """Read KMI30 symbols from Excel file"""
    try:
        file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        print(f"Attempting to read KMI30 symbols from: {file_path}")
        if not os.path.exists(file_path):
            logging.warning(f"KMI30 symbols file not found at {file_path}")
            print(f"KMI30 symbols file not found at {file_path}")
            return []
            
        # Read symbols from Excel file
        try:
            df = pd.read_excel(file_path, sheet_name='KMI30')
            print(f"Successfully read Excel file, found {len(df)} rows")
            if 'Symbol' not in df.columns:
                logging.error("'Symbol' column not found in KMI30 sheet")
                print("Error: 'Symbol' column not found in KMI30 sheet")
                # Fall back to default symbols
                default_symbols = ['AICL', 'ATRL', 'BAFL', 'BAHL', 'CNERGY', 'EFERT', 'ENGRO', 'FFBL', 'FFC', 'FCCL', 'HUBC', 'HBL', 'ISL', 'ILP', 'LUCK', 'MCB', 'MARI', 'MEBL', 'MLCF', 'MTL', 'NBP', 'NML', 'OGDC', 'PAKT', 'PPL', 'PIOC', 'PSO', 'SNGP', 'SSGC', 'UBL']
                print(f"Falling back to default list of {len(default_symbols)} symbols")
                return default_symbols
                
            symbols = df['Symbol'].str.strip().str.upper().tolist()
            logging.info(f"Successfully read {len(symbols)} KMI30 symbols from Excel file")
            print(f"Successfully read {len(symbols)} KMI30 symbols: {symbols[:5]}...")
            return symbols
        except Exception as e:
            logging.error(f"Error reading Excel file: {str(e)}")
            print(f"Error reading Excel file: {str(e)}")
            # Fall back to default symbols
            default_symbols = ['AICL', 'ATRL', 'BAFL', 'BAHL', 'CNERGY', 'EFERT', 'ENGRO', 'FFBL', 'FFC', 'FCCL', 'HUBC', 'HBL', 'ISL', 'ILP', 'LUCK', 'MCB', 'MARI', 'MEBL', 'MLCF', 'MTL', 'NBP', 'NML', 'OGDC', 'PAKT', 'PPL', 'PIOC', 'PSO', 'SNGP', 'SSGC', 'UBL']
            print(f"Falling back to default list of {len(default_symbols)} symbols")
            return default_symbols
    except Exception as e:
        logging.error(f"Error reading KMI30 symbols: {e}")
        print(f"Error reading KMI30 symbols: {str(e)}")
        return []

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='PSX Stock Analysis Tool')
    parser.add_argument('--dashboard-only', action='store_true', help='Generate only the dashboard')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to analyze')
    parser.add_argument('--backtest', action='store_true', help='Run backtest on signals')
    return parser.parse_args()

# 2. DATABASE INTERACTION FUNCTIONS
def fetch_table_names(cursor):
    """Fetch stock tables from the database"""
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall() 
                  if table[0].startswith('PSX_') and table[0].endswith('_stock_data')]
        logging.info(f"Fetched {len(tables)} stock tables")
        return tables
    except Exception as e:
        logging.error(f"Error fetching table names: {e}")
        return []

def fetch_column_names(engine, table_name):
    """Fetch column names from a table"""
    try:
        query = f"PRAGMA table_info({table_name})"
        df = pd.read_sql(query, engine)
        columns = df['name'].tolist()
        logging.info(f"Columns in table {table_name}: {columns}")
        return columns
    except Exception as e:
        logging.error(f"Error fetching column names for table {table_name}: {e}")
        return []

def get_available_symbols(cursor):
    """Get a list of available stock symbols"""
    try:
        print("Getting available symbols from database")
        # First get KMI30 symbols from Excel
        kmi30_symbols = set(get_kmi30_symbols())
        print(f"Retrieved {len(kmi30_symbols)} KMI30 symbols")
        
        # Then get available tables from database
        table_names = fetch_table_names(cursor)
        print(f"Retrieved {len(table_names)} table names from database")
        excluded_terms = ['STOCK_DATA', 'META', 'SYSTEM', 'DATA', 'INDEX', 'CONFIG', 'TEMP', 'BACKUP']
        available_symbols = []
        
        for table_name in table_names:
            symbol = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
            if symbol in excluded_terms or len(symbol) > 10 or '_' in symbol:
                logging.info(f"Skipping non-stock table: {table_name}")
                continue
            # Only include symbols that are in KMI30
            if symbol in kmi30_symbols:
                available_symbols.append(symbol)
                print(f"Added symbol {symbol} to available symbols")
                
        logging.info(f"Found {len(available_symbols)} available KMI30 symbols")
        print(f"Found {len(available_symbols)} available KMI30 symbols")
        return available_symbols
    except Exception as e:
        logging.error(f"Error getting available symbols: {e}")
        print(f"Error getting available symbols: {str(e)}")
        return []

def get_latest_buy_stocks():
    """Get the latest buy stocks from the database"""
    try:
        with sqlite3.connect(config['database']['signals_db']) as conn:
            query = """
                SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close,
                       update_date, julianday(?) - julianday(Signal_Date) AS holding_days
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY Stock ORDER BY update_date DESC, Signal_Date DESC) as rn
                    FROM buy_stocks
                    WHERE Signal_Date IS NOT NULL
                ) t
                WHERE rn = 1
                ORDER BY update_date DESC, holding_days
            """
            df = pd.read_sql_query(query, conn, params=(datetime.now().strftime('%Y-%m-%d'),))
            if not df.empty:
                df['holding_days'] = df['holding_days'].astype(int)
            return df
    except Exception as e:
        logging.error(f"Error getting latest buy stocks: {e}")
        print(f"Error getting latest buy stocks: {str(e)}")
        return pd.DataFrame()

def get_latest_sell_stocks():
    """Get the latest sell stocks from the database"""
    try:
        with sqlite3.connect(config['database']['signals_db']) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sell_stocks'")
            if not cursor.fetchone():
                logging.warning("Table 'sell_stocks' does not exist")
                print("Warning: Table 'sell_stocks' does not exist")
                return pd.DataFrame()
            query = """
                SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close,
                       update_date, julianday(?) - julianday(Signal_Date) AS days_ago
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY Stock ORDER BY update_date DESC, Signal_Date DESC) as rn
                    FROM sell_stocks
                    WHERE Signal_Date IS NOT NULL
                ) t
                WHERE rn = 1
                ORDER BY update_date DESC, days_ago
            """
            df = pd.read_sql_query(query, conn, params=(datetime.now().strftime('%Y-%m-%d'),))
            if not df.empty:
                df['days_ago'] = df['days_ago'].astype(int)
            return df
    except Exception as e:
        logging.error(f"Error getting latest sell stocks: {e}")
        print(f"Error getting latest sell stocks: {str(e)}")
        return pd.DataFrame()

def format_signals_for_telegram(signal_df, signal_type="BUY"):
    """Format signals dataframe for Telegram message"""
    try:
        if signal_df.empty:
            return None  # Return None for empty dataframes
            
        # Format the dataframe
        formatted_df = signal_df.copy()
        
        # Select and rename columns based on signal type
        if signal_type == "BUY":
            if 'update_date' in formatted_df.columns:
                formatted_df = formatted_df[['Stock', 'update_date', 'Signal_Date', 'Signal_Close', 'RSI_Weekly_Avg', 'AO_Weekly', 'holding_days']].copy()
                formatted_df.columns = ['Symbol', 'Updated', 'Buy Date', 'Buy Price', 'RSI', 'AO', 'Days Held']
            else:
                formatted_df = formatted_df[['Stock', 'Signal_Date', 'Signal_Close', 'RSI_Weekly_Avg', 'AO_Weekly', 'holding_days']].copy()
                formatted_df.columns = ['Symbol', 'Buy Date', 'Buy Price', 'RSI', 'AO', 'Days Held']
        else:  # SELL signals
            if 'update_date' in formatted_df.columns:
                formatted_df = formatted_df[['Stock', 'update_date', 'Signal_Date', 'Signal_Close', 'RSI_Weekly_Avg', 'AO_Weekly', 'days_ago']].copy()
                formatted_df.columns = ['Symbol', 'Updated', 'Sell Date', 'Sell Price', 'RSI', 'AO', 'Days Ago']
            else:
                formatted_df = formatted_df[['Stock', 'Signal_Date', 'Signal_Close', 'RSI_Weekly_Avg', 'AO_Weekly', 'days_ago']].copy()
                formatted_df.columns = ['Symbol', 'Sell Date', 'Sell Price', 'RSI', 'AO', 'Days Ago']
        
        # Format dates and numbers
        date_columns = [col for col in formatted_df.columns if 'Date' in col]
        for col in date_columns:
            formatted_df[col] = pd.to_datetime(formatted_df[col]).dt.strftime('%Y-%m-%d')
        
        numeric_cols = ['RSI', 'AO']
        for col in numeric_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        price_cols = ['Buy Price', 'Sell Price']
        for col in price_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # Add row numbers
        formatted_df.insert(0, '#', range(1, len(formatted_df) + 1))
        
        # Convert to string representation
        table = tabulate(formatted_df, headers='keys', tablefmt='simple', showindex=False)
        
        # Create header and footer
        header = f"🔔 {signal_type} SIGNALS 🔔\n"
        header += "=" * 50 + "\n"
        footer = f"\nTotal {signal_type} signals: {len(formatted_df)}"
        
        # Combine everything into a single message
        message = f"{header}\n{table}\n{footer}"
        
        # Split into chunks if too long
        max_length = 4000  # Telegram's message length limit
        if len(message) > max_length:
            # Split by newlines to preserve table structure
            lines = message.split('\n')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 > max_length:
                    if current_chunk:  # Only add chunk if it's not empty
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = len(line)
                else:
                    current_chunk.append(line)
                    current_length += len(line) + 1
            
            if current_chunk:  # Add the last chunk if it's not empty
                chunks.append('\n'.join(current_chunk))
            
            # Return the first chunk that's not empty and has reasonable length
            for chunk in chunks:
                if len(chunk.strip()) > 10:  # Ensure chunk has meaningful content
                    return chunk
            
            # If no suitable chunk found, return a truncated version of the original message
            return message[:max_length] + "..."
        else:
            return message
            
    except Exception as e:
        logging.error(f"Error formatting {signal_type} signals: {e}")
        return None  # Return None for errors

def _send_telegram_api_call(message=None, image_path=None, max_retries=1):
    """Make the actual Telegram API call with rate limit handling"""
    try:
        if not config.get('telegram', {}).get('enabled', False):
            logging.debug("Telegram notifications are disabled")
            return True
            
        bot_token = config['telegram']['bot_token']
        chat_id = config['telegram']['chat_id']
        
        if not bot_token or not chat_id:
            logging.error("Telegram bot token or chat ID not configured")
            return False
            
        # Validate chat_id before making the request
        try:
            # Test the chat_id with a simple getChat request
            test_url = f"https://api.telegram.org/bot{bot_token}/getChat"
            test_data = {'chat_id': chat_id}
            test_response = requests.post(test_url, data=test_data)
            
            if test_response.status_code == 400:
                error_data = test_response.json()
                if 'description' in error_data:
                    if 'chat not found' in error_data['description'].lower():
                        logging.error(f"Chat not found: {chat_id}")
                        logging.error("Please make sure:")
                        logging.error("1. The bot is added to the group/channel")
                        logging.error("2. The bot has admin rights in the group/channel")
                        logging.error("3. The chat_id is correct")
                        logging.error("To verify your chat_id, send a message to your bot and visit:")
                        logging.error(f"https://api.telegram.org/bot{bot_token}/getUpdates")
                    else:
                        logging.error(f"Telegram API error: {error_data['description']}")
                return False
                
            test_response.raise_for_status()
            
        except requests.exceptions.HTTPError as e:
            logging.error(f"Error validating chat_id: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error validating chat_id: {e}")
            return False
            
        # Validate message
        if message is not None:
            message = str(message).strip()
            if len(message) < 2:  # Skip very short messages
                logging.warning(f"Skipping message that's too short: '{message}'")
                return True
                
        retry_count = 0
        while retry_count <= max_retries:
            try:
                if image_path and os.path.exists(image_path):
                    # Send image with caption
                    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                    with open(image_path, 'rb') as photo:
                        files = {'photo': photo}
                        data = {'chat_id': chat_id}
                        if message:
                            # Escape special characters for Markdown
                            message = message.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                            data['caption'] = message
                            data['parse_mode'] = 'Markdown'
                        response = requests.post(url, files=files, data=data)
                else:
                    # Send text message only
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    if message:
                        # Escape special characters for Markdown
                        message = message.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                    data = {
                        'chat_id': chat_id,
                        'text': message,
                        'parse_mode': 'Markdown'
                    }
                    response = requests.post(url, data=data)
                    
                response.raise_for_status()
                return True
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    retry_after = int(e.response.headers.get('Retry-After', 30))
                    logging.warning(f"Rate limit hit, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                elif e.response.status_code == 400:  # Bad Request
                    logging.error(f"Bad request error: {e.response.text}")
                    # Try sending without Markdown formatting
                    try:
                        if image_path and os.path.exists(image_path):
                            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                            with open(image_path, 'rb') as photo:
                                files = {'photo': photo}
                                data = {'chat_id': chat_id}
                                if message:
                                    data['caption'] = message
                                response = requests.post(url, files=files, data=data)
                        else:
                            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                            data = {
                                'chat_id': chat_id,
                                'text': message
                            }
                            response = requests.post(url, data=data)
                        response.raise_for_status()
                        return True
                    except Exception as retry_e:
                        logging.error(f"Failed to send without Markdown: {retry_e}")
                        return False
                else:
                    logging.error(f"HTTP error: {e}")
                    return False
                    
            except Exception as e:
                logging.error(f"Request failed: {e}")
                retry_count += 1
                if retry_count <= max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                return False
                
    except Exception as e:
        logging.error(f"Telegram API call failed: {e}")
        return False

def send_telegram_message(message):
    """Send message to Telegram with proper error handling"""
    try:
        if not config.get('telegram', {}).get('enabled', False):
            logging.warning("Telegram messaging is disabled in configuration.")
            return False
        # Ensure message is not too long (Telegram limit is 4096 characters)
        max_length = 4000  # Leave some buffer
        if len(message) > max_length:
            # Split message into chunks
            chunks = [message[i:i+max_length] for i in range(0, len(message), max_length)]
            for chunk in chunks:
                _send_telegram_api_call(message=chunk)
                time.sleep(1)  # Add delay between chunks
        else:
            _send_telegram_api_call(message=message)
        return True
    except Exception as e:
        logging.error(f"Error sending message to Telegram: {e}")
        return False

def send_telegram_message_with_image(image_paths, message=None):
    """Send message with image(s) to Telegram"""
    try:
        if not config.get('telegram', {}).get('enabled', False):
            logging.warning("Telegram messaging is disabled in configuration.")
            return False
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
            
        for image_path in image_paths:
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                continue
                
            # Try to send with retries
            success = False
            for attempt in range(3):  # Try up to 3 times
                if _send_telegram_api_call(message, image_path):
                    success = True
                    break
                time.sleep(2)  # Wait between attempts
                
            if not success:
                logging.error(f"Failed to send image {image_path}")
                
            time.sleep(1)  # Rate limiting between images
            
    except Exception as e:
        logging.error(f"Error sending image message: {e}")

def get_buy_signals(symbol):
    """Get buy signals for a specific symbol"""
    try:
        with sqlite3.connect(config['database']['signals_db']) as conn:
            query = """
                SELECT Signal_Date, Signal_Close
                FROM buy_stocks
                WHERE Stock = ?
                ORDER BY Signal_Date DESC
                LIMIT 5
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            return [(row['Signal_Date'], row['Signal_Close']) for _, row in df.iterrows()]
    except Exception as e:
        logging.error(f"Error getting buy signals for {symbol}: {e}")
        print(f"Error getting buy signals for {symbol}: {str(e)}")
        return []

def get_sell_signals(symbol):
    """Get sell signals for a specific symbol"""
    try:
        with sqlite3.connect(config['database']['signals_db']) as conn:
            query = """
                SELECT Signal_Date, Signal_Close
                FROM sell_stocks
                WHERE Stock = ?
                ORDER BY Signal_Date DESC
                LIMIT 5
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            return [(row['Signal_Date'], row['Signal_Close']) for _, row in df.iterrows()]
    except Exception as e:
        logging.error(f"Error getting sell signals for {symbol}: {e}")
        print(f"Error getting sell signals for {symbol}: {str(e)}")
        return []

def get_buy_sell_signals(symbol):
    """Get buy and sell signals for a specific symbol"""
    try:
        with sqlite3.connect(config['database']['signals_db']) as conn:
            buy_query = """
                SELECT Signal_Date, Signal_Close, 'BUY' as Signal_Type
                FROM buy_stocks
                WHERE Stock = ?
                ORDER BY Signal_Date DESC
                LIMIT 5
            """
            sell_query = """
                SELECT Signal_Date, Signal_Close, 'SELL' as Signal_Type
                FROM sell_stocks
                WHERE Stock = ?
                ORDER BY Signal_Date DESC
                LIMIT 5
            """
            buy_df = pd.read_sql_query(buy_query, conn, params=(symbol,))
            sell_df = pd.read_sql_query(sell_query, conn, params=(symbol,))
            buy_signals = [(row['Signal_Date'], row['Signal_Close'], row['Signal_Type']) for _, row in buy_df.iterrows()]
            sell_signals = [(row['Signal_Date'], row['Signal_Close'], row['Signal_Type']) for _, row in sell_df.iterrows()]
            return buy_signals, sell_signals
    except Exception as e:
        logging.error(f"Error getting signals for {symbol}: {e}")
        print(f"Error getting signals for {symbol}: {str(e)}")
        return [], []

def remove_duplicate_buy_stocks():
    """Remove duplicate entries from buy_stocks, keeping the most recent"""
    try:
        with sqlite3.connect(config['database']['signals_db']) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='buy_stocks'")
            if not cursor.fetchone():
                logging.warning("Table 'buy_stocks' does not exist")
                return 0
            cursor.execute("SELECT COUNT(*) FROM buy_stocks")
            total_before = cursor.fetchone()[0]
            cursor.execute("""
                DELETE FROM buy_stocks 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM buy_stocks
                    GROUP BY Stock, Signal_Date
                )
            """)
            total_removed = conn.total_changes
            conn.commit()
            cursor.execute("SELECT COUNT(*) FROM buy_stocks")
            total_after = cursor.fetchone()[0]
            logging.info(f"Removed {total_removed} duplicates. Records: {total_before} -> {total_after}")
            return total_removed
    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return 0

# 3. ANALYSIS FUNCTIONS
def prepare_analysis_data(df):
    """Prepare dataframe for market phase analysis"""
    analysis_df = df.copy().sort_values('Date')
    if 'pct_change' not in analysis_df.columns:
        analysis_df['pct_change'] = analysis_df['Close'].pct_change() * 100
    return analysis_df.tail(260)

def get_ao_change_dates(df):
    """Get dates when AO changes sign"""
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
    """Analyze RSI trends for accumulation/distribution"""
    try:
        recent_rsi = analysis_df['RSI_weekly'].tail(20).values
        rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0] if len(recent_rsi) > 1 else 0
        longer_rsi = analysis_df['RSI_weekly'].tail(60).values
        longer_rsi_trend = np.polyfit(range(len(longer_rsi)), longer_rsi, 1)[0] if len(longer_rsi) > 2 else 0
        current_rsi = recent_rsi[-1] if len(recent_rsi) > 0 else 50
        rsi_score = 0
        if current_rsi < 35 and rsi_trend > 0.1:
            rsi_score = 2.5
        elif current_rsi < 40 and rsi_trend > 0.05:
            rsi_score = 2
        elif current_rsi > 65 and rsi_trend < -0.1:
            rsi_score = -2.5
        elif current_rsi > 60 and rsi_trend < -0.05:
            rsi_score = -2
        elif current_rsi < 45 and rsi_trend > 0.03:
            rsi_score = 1
        elif current_rsi > 55 and rsi_trend < -0.03:
            rsi_score = -1
        elif current_rsi < 50 and longer_rsi_trend > 0:
            rsi_score = 0.5
        elif current_rsi > 50 and longer_rsi_trend < 0:
            rsi_score = -0.5
        details = {
            'current_rsi': round(current_rsi, 2),
            'rsi_trend': round(rsi_trend, 4),
            'longer_rsi_trend': round(longer_rsi_trend, 4)
        }
        if len(recent_rsi) > 10 and len(analysis_df['Close'].tail(20)) > 10:
            price_trend = np.polyfit(range(len(analysis_df['Close'].tail(20))), analysis_df['Close'].tail(20).values, 1)[0]
            if price_trend > 0 and rsi_trend < 0:
                rsi_score -= 0.5
                details['divergence'] = 'bearish'
            elif price_trend < 0 and rsi_trend > 0:
                rsi_score += 0.5
                details['divergence'] = 'bullish'
            else:
                details['divergence'] = 'none'
        if len(recent_rsi) > 10:
            rsi_volatility = np.std(recent_rsi)
            details['rsi_volatility'] = round(rsi_volatility, 4)
            if rsi_volatility > 5:
                rsi_score -= 0.3
            elif rsi_volatility < 3:
                rsi_score += 0.3
        return rsi_score, details
    except Exception as e:
        logging.error(f"Error analyzing RSI trend: {e}")
        return 0, {'error': str(e)}

def analyze_ao_trend(analysis_df):
    """Analyze Awesome Oscillator trends"""
    try:
        recent_ao = analysis_df['AO_weekly_AVG'].tail(30).values
        current_ao = recent_ao[-1] if len(recent_ao) > 0 else 0
        ao_trend = np.polyfit(range(len(recent_ao)), recent_ao, 1)[0] if len(recent_ao) > 1 else 0
        recent_crosses = recent_ao[-15:] if len(recent_ao) >= 15 else recent_ao
        ao_crosses_up = False
        ao_crosses_down = False
        for i in range(1, len(recent_crosses)):
            if recent_crosses[i-1] < 0 and recent_crosses[i] >= 0:
                ao_crosses_up = True
            if recent_crosses[i-1] > 0 and recent_crosses[i] <= 0:
                ao_crosses_down = True
        ao_score = 0
        if ao_crosses_up:
            ao_score = 2.5
        elif ao_crosses_down:
            ao_score = -2.5
        elif current_ao > 0 and ao_trend > 0.02:
            ao_score = 2
        elif current_ao < 0 and ao_trend < -0.02:
            ao_score = -2
        elif current_ao > 0:
            ao_score = 1
        elif current_ao < 0:
            ao_score = -1
        details = {
            'current_ao': round(current_ao, 2),
            'ao_trend': round(ao_trend, 4),
            'crosses_up': ao_crosses_up,
            'crosses_down': ao_crosses_down
        }
        if len(recent_ao) > 10 and len(analysis_df['Close'].tail(20)) > 10:
            price_trend = np.polyfit(range(len(analysis_df['Close'].tail(20))), analysis_df['Close'].tail(20).values, 1)[0]
            if price_trend > 0 and ao_trend < 0:
                ao_score -= 0.5
                details['divergence'] = 'bearish'
            elif price_trend < 0 and ao_trend > 0:
                ao_score += 0.5
                details['divergence'] = 'bullish'
            else:
                details['divergence'] = 'none'
        return ao_score, details
    except Exception as e:
        logging.error(f"Error analyzing AO trend: {e}")
        return 0, {'error': str(e)}

def analyze_volume_pattern(analysis_df):
    """Analyze volume patterns"""
    try:
        recent_volume = analysis_df['Volume'].tail(60)
        recent_returns = analysis_df['pct_change'].tail(60)
        volume_score = 0
        details = {}
        if len(recent_returns) >= 40:
            up_days = recent_returns > 0
            down_days = recent_returns < 0
            if up_days.sum() > 0 and down_days.sum() > 0:
                up_day_volume = recent_volume[up_days].mean()
                down_day_volume = recent_volume[down_days].mean()
                vol_ratio = up_day_volume / down_day_volume if down_day_volume > 0 else 1.0
                details['volume_ratio'] = round(vol_ratio, 2)
                recent_vol_trend = recent_volume.tail(30)
                vol_trend = np.polyfit(range(len(recent_vol_trend)), recent_vol_trend.values, 1)[0]
                details['volume_trend'] = round(vol_trend, 2)
                if vol_ratio > 1.5 and vol_trend > 0:
                    volume_score = 2.5
                elif vol_ratio < 0.67 and vol_trend > 0:
                    volume_score = -2.5
                elif vol_ratio > 1.2:
                    volume_score = 1.5
                elif vol_ratio < 0.83:
                    volume_score = -1.5
                elif vol_ratio > 1:
                    volume_score = 0.7
                elif vol_ratio < 1:
                    volume_score = -0.7
        if len(recent_volume) >= 30:
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
                    if spike_returns.mean() > 0:
                        volume_score += 0.3
                        details['volume_spikes'][period] = 'positive'
                    elif spike_returns.mean() < 0:
                        volume_score -= 0.3
                        details['volume_spikes'][period] = 'negative'
                    else:
                        details['volume_spikes'][period] = 'neutral'
                else:
                    details['volume_spikes'][period] = 'none'
        if len(recent_volume) > 10 and len(analysis_df['Close'].tail(20)) > 10:
            price_trend = np.polyfit(range(len(analysis_df['Close'].tail(20))), analysis_df['Close'].tail(20).values, 1)[0]
            volume_trend = np.polyfit(range(len(recent_volume.tail(20))), recent_volume.tail(20).values, 1)[0]
            if price_trend > 0 and volume_trend < 0:
                volume_score -= 0.5
                details['divergence'] = 'bearish'
            elif price_trend < 0 and volume_trend > 0:
                volume_score += 0.5
                details['divergence'] = 'bullish'
            else:
                details['divergence'] = 'none'
        return volume_score, details
    except Exception as e:
        logging.error(f"Error analyzing volume pattern: {e}")
        return 0, {'error': str(e)}

def analyze_price_ma_relationship(analysis_df):
    """Analyze price relationship to moving averages"""
    try:
        recent_prices = analysis_df['Close'].tail(5).values
        ma_score = 0
        details = {}
        if len(recent_prices) > 0:
            ma_columns = ['MA_10', 'MA_30', 'MA_50']
            ma_scores = []
            ma_details = {}
            for ma_col in ma_columns:
                if ma_col in analysis_df.columns:
                    recent_ma = analysis_df[ma_col].tail(5).values
                    if len(recent_ma) > 0:
                        price_vs_ma = (recent_prices[-1] / recent_ma[-1] - 1) * 100
                        price_above_ma = recent_prices[-1] > recent_ma[-1]
                        ma_details[f'{ma_col}_pct'] = round(price_vs_ma, 2)
                        ma_details[f'{ma_col}_above'] = price_above_ma
                        if price_above_ma and price_vs_ma > 5:
                            ma_scores.append(1.5)
                        elif price_above_ma:
                            ma_scores.append(1)
                        elif not price_above_ma and price_vs_ma < -5:
                            ma_scores.append(-1.5)
                        elif not price_above_ma:
                            ma_scores.append(-1)
            if ma_scores:
                ma_score = sum(ma_scores) / len(ma_scores)
                details.update(ma_details)
        if len(recent_prices) > 1 and 'MA_30' in analysis_df.columns:
            recent_ma = analysis_df['MA_30'].tail(5).values
            if len(recent_ma) > 1:
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                ma_trend = np.polyfit(range(len(recent_ma)), recent_ma, 1)[0]
                trend_difference = price_trend - ma_trend
                details['price_trend'] = round(price_trend, 4)
                details['ma_trend'] = round(ma_trend, 4)
                details['trend_difference'] = round(trend_difference, 4)
                if trend_difference > 0.01:
                    ma_score += 0.5
                elif trend_difference < -0.01:
                    ma_score -= 0.5
        return ma_score, details
    except Exception as e:
        logging.error(f"Error analyzing price-MA relationship: {e}")
        return 0, {'error': str(e)}

def analyze_price_pattern(analysis_df):
    """Analyze price patterns (higher highs/lows)"""
    try:
        pattern_score = 0
        details = {}
        if len(analysis_df) >= 80:
            prices = analysis_df['Close'].values
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
            high_change_pct = (curr_high / prev_high - 1) * 100
            low_change_pct = (curr_low / prev_low - 1) * 100
            details['high_change_pct'] = round(high_change_pct, 2)
            details['low_change_pct'] = round(low_change_pct, 2)
            if high_change_pct > 3 and low_change_pct > 3:
                pattern_score = 2.5
                details['pattern'] = 'strong_higher_highs_lows'
            elif curr_high > prev_high and curr_low > prev_low:
                pattern_score = 1.5
                details['pattern'] = 'higher_highs_lows'
            elif high_change_pct < -3 and low_change_pct < -3:
                pattern_score = -2.5
                details['pattern'] = 'strong_lower_highs_lows'
            elif curr_high < prev_high and curr_low < prev_low:
                pattern_score = -1.5
                details['pattern'] = 'lower_highs_lows'
            elif curr_high > prev_high and curr_low < prev_low:
                pattern_score = 0.3
                details['pattern'] = 'expanding_volatility'
            elif curr_high < prev_high and curr_low > prev_low:
                pattern_score = -0.3
                details['pattern'] = 'contracting_volatility'
            if len(prices) > 10:
                price_trend = np.polyfit(range(len(prices[-20:])), prices[-20:], 1)[0]
                details['price_trend'] = round(price_trend, 4)
                if price_trend > 0.01:
                    pattern_score += 0.3
                elif price_trend < -0.01:
                    pattern_score -= 0.3
        return pattern_score, details
    except Exception as e:
        logging.error(f"Error analyzing price pattern: {e}")
        return 0, {'error': str(e)}

def calculate_final_phase_score(rsi_score, ao_score, volume_score, ma_score, pattern_score):
    """Calculate market phase score"""
    try:
        # Further adjust weights to emphasize RSI and MA relationships
        total_score = (rsi_score * 0.40) + (ao_score * 0.15) + (volume_score * 0.15) + (ma_score * 0.25) + (pattern_score * 0.05)
        
        # Lower thresholds further for phase detection
        strong_threshold = 0.8   # Was 1.0
        weak_threshold = 0.3     # Was 0.5
        
        max_possible_score = 10.0
        
        if total_score > strong_threshold:
            probability = min(round(((total_score - strong_threshold) / (max_possible_score - strong_threshold)) * 100, 2), 100)
            phase = "ACCUMULATION"
        elif total_score > weak_threshold:
            probability = min(round(((total_score - weak_threshold) / (strong_threshold - weak_threshold)) * 100, 2), 100)
            phase = "WEAK_ACCUMULATION"
        elif total_score < -strong_threshold:
            probability = min(round(((abs(total_score) - strong_threshold) / (max_possible_score - strong_threshold)) * 100, 2), 100)
            phase = "DISTRIBUTION"
        else:
            neutral_position = total_score / strong_threshold if strong_threshold > 0 else 0
            probability = round(50 + (neutral_position * 25), 2)
            phase = "NEUTRAL"
            
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
        logging.error(f"Error calculating phase score: {e}")
        return "NEUTRAL", 50, {'error': str(e)}

def calculate_market_phase(df, symbol_name):
    """Calculate market phase for a stock"""
    analysis_df = prepare_analysis_data(df)
    rsi_score, rsi_details = analyze_rsi_trend(analysis_df)
    ao_score, ao_details = analyze_ao_trend(analysis_df)
    volume_score, volume_details = analyze_volume_pattern(analysis_df)
    ma_score, ma_details = analyze_price_ma_relationship(analysis_df)
    pattern_score, pattern_details = analyze_price_pattern(analysis_df)
    return calculate_final_phase_score(rsi_score, ao_score, volume_score, ma_score, pattern_score)

# 4. VISUALIZATION AND REPORTING FUNCTIONS
def draw_indicator_trend_lines_with_signals(database_path, table_name):
    """Draw trend lines and signals for a stock"""
    try:
        symbol_name = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
        engine = create_engine(f'sqlite:///{database_path}')
        connection = engine.connect()
        available_columns = fetch_column_names(engine, table_name)
        monthly_column = 'RSI_monthly_Avg'
        threemonth_column = 'RSI_3months_Avg'
        base_columns = ["Date", "Close", "RSI_weekly_Avg", "AO_weekly_AVG", "MA_30", "RSI_weekly", "Volume", "pct_change"]
        columns_to_select = base_columns.copy()
        if monthly_column in available_columns:
            columns_to_select.append(monthly_column)
        if threemonth_column in available_columns:
            columns_to_select.append(threemonth_column)
        query = f"SELECT {', '.join(columns_to_select)} FROM {table_name}"
        df = pd.read_sql(query, connection)
        connection.close()
        if df.empty:
            logging.warning(f"No data for {table_name}")
            return False
        df['Date'] = pd.to_datetime(df['Date'])
        custom_years_ago = datetime.now() - timedelta(days=config['analysis']['lookback_years']*365)
        df = df[df['Date'] >= custom_years_ago]
        change_dates = get_ao_change_dates(df)
        buy_signals, sell_signals = get_buy_sell_signals(symbol_name)
        buy_signals = [(pd.to_datetime(date), price, signal_type) for date, price, signal_type in buy_signals]
        sell_signals = [(pd.to_datetime(date), price, signal_type) for date, price, signal_type in sell_signals]
        holding_days = None
        profit_loss_pct = None
        signal_price = None
        current_price = None
        with sqlite3.connect(config['database']['signals_db']) as conn:
            cursor = conn.cursor()
            current_date = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("PRAGMA table_info(buy_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            pl_column = next((col for col in columns if 'p/l' in col.lower() or col.lower() in ['profit', 'loss', 'gain']), None)
            query = f"""
                SELECT julianday('{current_date}') - julianday(Signal_Date) AS days_held,
                       Signal_Date, Signal_Close
                FROM buy_stocks 
                WHERE Stock = ? AND Signal_Date IS NOT NULL
                ORDER BY Signal_Date DESC
                LIMIT 1
            """
            cursor.execute(query, (symbol_name,))
            result = cursor.fetchone()
            if result:
                holding_days = int(result[0])
                signal_date = result[1]
                signal_price = result[2]
                if pl_column:
                    cursor.execute(f'SELECT "{pl_column}" FROM buy_stocks WHERE Stock = ? AND Signal_Date = ?', 
                                  (symbol_name, signal_date))
                    pl_result = cursor.fetchone()
                    if pl_result and pl_result[0]:
                        profit_loss_pct = float(pl_result[0])
                if profit_loss_pct is None and signal_price:
                    cursor.execute(f"SELECT Close FROM {table_name} ORDER BY Date DESC LIMIT 1")
                    close_result = cursor.fetchone()
                    if close_result:
                        current_price = float(close_result[0])
                        profit_loss_pct = ((current_price - signal_price) / signal_price) * 100
        stock_status = "OPPORTUNITY"
        if buy_signals and sell_signals:
            latest_buy = max(buy_signals, key=lambda x: x[0])
            latest_sell = max(sell_signals, key=lambda x: x[0])
            if latest_buy[0] > latest_sell[0]:
                stock_status = "BUY/HOLD"
            else:
                stock_status = "SELL"
        elif buy_signals:
            stock_status = "BUY/HOLD"
        elif sell_signals:
            stock_status = "SELL"
        market_phase, phase_probability, phase_details = calculate_market_phase(df, symbol_name)
        plt.figure(figsize=(14, 14))
        plt.subplot(5, 1, 1)
        plt.plot(df['Date'], df['RSI_weekly_Avg'], label='RSI_weekly_Avg', color='blue')
        plt.plot(df['Date'], df['RSI_weekly'], label='RSI_weekly', color='purple')
        plt.axhline(y=config['analysis']['rsi_thresholds'][0], color='green', linestyle='--', label='RSI 40')
        plt.axhline(y=config['analysis']['rsi_thresholds'][1], color='red', linestyle='--', label='RSI 60')
        plt.title(f'{symbol_name} - RSI Weekly')
        plt.xlabel('Date')
        plt.ylabel('RSI Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(5, 1, 2)
        positive_ao = df[df['AO_weekly_AVG'] >= 0]
        negative_ao = df[df['AO_weekly_AVG'] < 0]
        plt.plot(positive_ao['Date'], positive_ao['AO_weekly_AVG'], color='green', label='AO Positive')
        plt.plot(negative_ao['Date'], negative_ao['AO_weekly_AVG'], color='red', label='AO Negative')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title(f'{symbol_name} - AO Weekly')
        plt.xlabel('Date')
        plt.ylabel('AO_weekly_AVG')
        plt.legend()
        plt.grid(True, alpha=0.3)
        for date, close in change_dates['negative_to_positive']:
            plt.plot(date, 0, marker='*', color='green', markersize=10)
        for date, close in change_dates['positive_to_negative']:
            plt.plot(date, 0, marker='*', color='red', markersize=10)
        ax3 = plt.subplot(5, 1, 3)
        plt.plot(df['Date'], df['Close'], label='Close', color='orange')
        plt.plot(df['Date'], df['MA_30'], label='MA_30', color='red')
        plt.title(f'{symbol_name} - Price and MA30')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        for date, price, _ in buy_signals:
            plt.plot(date, price, marker='^', color='green', markersize=10)
            plt.annotate('Buy', (date, price), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='green')
        for date, price, _ in sell_signals:
            plt.plot(date, price, marker='v', color='red', markersize=10)
            plt.annotate('Sell', (date, price), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        plt.legend()
        plt.subplot(5, 1, 4)
        plt.bar(df['Date'], df['Volume'], color='blue', alpha=0.6)
        plt.title(f'{symbol_name} - Trading Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.grid(True, axis='y', alpha=0.3)
        plt.subplot(5, 1, 5)
        has_monthly_data = False
        if monthly_column in df.columns and df[monthly_column].notna().any():
            plt.plot(df['Date'], df[monthly_column], label='RSI Monthly', color='green')
            has_monthly_data = True
        if threemonth_column in df.columns and df[threemonth_column].notna().any():
            plt.plot(df['Date'], df[threemonth_column], label='RSI 3-Month', color='blue')
            has_monthly_data = True
        if not has_monthly_data:
            plt.text(0.5, 0.5, 'No monthly RSI data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.axhline(y=config['analysis']['rsi_thresholds'][0], color='green', linestyle='--', label='RSI 40')
        plt.axhline(y=config['analysis']['rsi_thresholds'][1], color='red', linestyle='--', label='RSI 60')
        plt.title(f'{symbol_name} - Monthly and 3-Month RSI')
        plt.xlabel('Date')
        plt.ylabel('RSI Values')
        plt.grid(True, alpha=0.3)
        plt.legend()
        current_date = datetime.now().strftime('%Y-%m-%d')
        title_text = f'{symbol_name} Technical Analysis - {stock_status}'
        if holding_days and stock_status == "BUY/HOLD":
            title_text += f' - Held for {holding_days} days'
        if profit_loss_pct is not None:
            title_text += f' - P/L: {profit_loss_pct:.2f}%'
        title_text += f' - {market_phase} {phase_probability:.2f}% - Generated on {current_date}'
        watermark_color = {
            "BUY/HOLD": "green",
            "SELL": "red",
            "OPPORTUNITY": "blue"
        }.get(stock_status, "gray")
        watermark_text = stock_status
        if holding_days and stock_status == "BUY/HOLD":
            watermark_text = f"{stock_status}\n{holding_days} DAYS"
        if profit_loss_pct is not None:
            profit_loss_sign = "+" if profit_loss_pct >= 0 else ""
            watermark_text += f"\n{profit_loss_sign}{profit_loss_pct:.2f}%"
        phase_color = {
            "ACCUMULATION": "green",
            "DISTRIBUTION": "red",
            "NEUTRAL": "gray"
        }.get(market_phase, "gray")
        fig = plt.gcf()
        fig.text(0.5, 0.55, watermark_text, fontsize=80, color=watermark_color, 
                 ha='center', va='center', alpha=0.2, rotation=30)
        fig.text(0.5, 0.25, f"{market_phase} {phase_probability:.2f}%", 
                 fontsize=60, color=phase_color, 
                 ha='center', va='center', alpha=0.2, rotation=30)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        charts_folder = config['output']['charts_folder']
        os.makedirs(charts_folder, exist_ok=True)
        plot_filename = os.path.join(charts_folder, f'{symbol_name}_trend_lines.png')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=120)
        message = title_text
        send_telegram_message_with_image(plot_filename, message)
        plt.close()
        return True
    except Exception as e:
        logging.error(f"Error drawing trend for {table_name}: {e}")
        return False

def send_signals_and_charts_summary(buy_df, sell_df, symbols, total_processed):
    """Send summary of signals and charts to Telegram"""
    try:
        # Format and send buy signals
        buy_message = format_signals_for_telegram(buy_df, "BUY")
        if buy_message:
            send_telegram_message(buy_message)
            time.sleep(1)  # Add delay between messages
        
        # Format and send sell signals
        sell_message = format_signals_for_telegram(sell_df, "SELL")
        if sell_message:
            send_telegram_message(sell_message)
            time.sleep(1)  # Add delay between messages
        
        # Send summary message
        summary = f"📊 Analysis Summary 📊\n"
        summary += "=" * 50 + "\n"
        summary += f"Total symbols processed: {total_processed}\n"
        summary += f"Total buy signals: {len(buy_df)}\n"
        summary += f"Total sell signals: {len(sell_df)}\n"
        summary += f"Total symbols with signals: {len(symbols)}"
        
        send_telegram_message(summary)
        
    except Exception as e:
        logging.error(f"Error sending signals summary: {e}")

def process_symbol(symbol, database_path):
    """Process a single symbol for chart generation"""
    table_name = f"PSX_{symbol}_stock_data"
    success = draw_indicator_trend_lines_with_signals(database_path, table_name)
    return symbol, success

def generate_charts_parallel(latest_buy_symbols, database_path, available_symbols):
    """Generate charts in parallel"""
    success_count = 0
    fail_count = 0
    
    # Filter symbols that are available
    symbols_to_process = [s for s in latest_buy_symbols if s in available_symbols]
    
    # Create a pool of workers
    with Pool() as pool:
        # Create a partial function with the database_path
        process_func = partial(process_symbol, database_path=database_path)
        
        # Use imap_unordered for parallel processing
        results = list(tqdm(
            pool.imap_unordered(process_func, symbols_to_process),
            total=len(symbols_to_process),
            desc="Generating charts"
        ))
    
    # Process results
    for symbol, success in results:
        if success:
            success_count += 1
            print(f"✅ Chart for {symbol} generated and sent")
        else:
            fail_count += 1
            print(f"❌ Failed to generate chart for {symbol}")
    
    return success_count, fail_count

def send_batched_telegram_messages(charts_folder, message):
    """Send charts in batches to avoid Telegram rate limits"""
    try:
        chart_files = [os.path.join(charts_folder, f) for f in os.listdir(charts_folder) 
                       if f.endswith('.png')]
        max_images = config['telegram']['max_images_per_message']
        
        # Add delay between batches to avoid rate limits
        delay_between_batches = 2  # seconds
        
        for i in range(0, len(chart_files), max_images):
            batch = chart_files[i:i+max_images]
            if batch:
                try:
                    send_telegram_message_with_image(batch, message)
                    # Add delay between batches
                    time.sleep(delay_between_batches)
                except Exception as e:
                    if "429" in str(e):  # Too Many Requests error
                        logging.warning("Rate limit hit, waiting 30 seconds before retrying...")
                        time.sleep(30)  # Wait longer when hitting rate limit
                        try:
                            send_telegram_message_with_image(batch, message)
                        except Exception as retry_e:
                            logging.error(f"Failed to send batch after retry: {retry_e}")
                    else:
                        logging.error(f"Error sending batch: {e}")
        return True
    except Exception as e:
        logging.error(f"Error sending batched Telegram messages: {e}")
        return False

def create_market_overview_dashboard(df, folder):
    """Create market overview dashboard"""
    if df.empty:
        return False
    date = datetime.now().strftime('%Y-%m-%d')
    
    # Create figure with custom size and style
    plt.style.use('default')  # Using default matplotlib style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'axes.edgecolor': '#CCCCCC',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333'
    })
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Add main title with date
    fig.suptitle(f'PSX Market Overview - {date}', fontsize=20, y=0.95)
    
    # Signal Distribution
    plt.subplot(2, 2, 1)
    signal_counts = df['Status'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green for BUY/HOLD, Red for SELL, Blue for OPPORTUNITY
    plt.pie(signal_counts, labels=signal_counts.index, autopct='%1.1f%%',
            colors=colors[:len(signal_counts)],
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title('Signal Distribution', pad=20, fontsize=12)
    
    # Market Phase Distribution
    plt.subplot(2, 2, 2)
    phase_counts = df['Market_Phase'].value_counts()
    colors = ['#27ae60', '#2ecc71', '#c0392b', '#7f8c8d']  # Different shades for different phases
    plt.pie(phase_counts, labels=phase_counts.index, autopct='%1.1f%%',
            colors=colors[:len(phase_counts)],
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title('Market Phase Distribution', pad=20, fontsize=12)
    
    # Market Breadth Indicators
    plt.subplot(2, 2, 3)
    breadth_data = {
        'Above MA30': len(df[df['Above_MA30']]) / len(df) * 100,
        'RSI > 50': len(df[df['RSI'] > 50]) / len(df) * 100,
        'Accumulation': len(df[df['Market_Phase'].isin(['ACCUMULATION', 'WEAK_ACCUMULATION'])]) / len(df) * 100
    }
    bars = plt.bar(breadth_data.keys(), breadth_data.values(), 
                  color=['#3498db', '#2ecc71', '#27ae60'])
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
    plt.ylim(0, 100)
    plt.title('Market Breadth Indicators', pad=20, fontsize=12)
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Top Performing Stocks
    plt.subplot(2, 2, 4)
    top_stocks = df.nlargest(5, 'Phase_Probability')
    bars = plt.barh(top_stocks['Symbol'], top_stocks['Phase_Probability'],
                   color='#2ecc71', alpha=0.8)
    for i, v in enumerate(top_stocks['Phase_Probability']):
        plt.text(v + 1, i, f'{v:.1f}%', va='center')
    plt.title('Top Performing Stocks', pad=20, fontsize=12)
    plt.xlabel('Accumulation Probability (%)', fontsize=10)
    
    # Add footer with summary
    plt.figtext(0.5, 0.02, 
                f'Total Stocks: {len(df)} | Strong Accumulation: {len(df[df["Market_Phase"] == "ACCUMULATION"])} | '
                f'Weak Accumulation: {len(df[df["Market_Phase"] == "WEAK_ACCUMULATION"])}',
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save with high DPI
    path = os.path.join(folder, f'market_overview_{date}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Send to Telegram
    message = f"📊 PSX Market Overview - {date}\n\n"
    message += f"Total Stocks: {len(df)}\n"
    message += f"Strong Accumulation: {len(df[df['Market_Phase'] == 'ACCUMULATION'])}\n"
    message += f"Weak Accumulation: {len(df[df['Market_Phase'] == 'WEAK_ACCUMULATION'])}\n"
    message += f"Distribution: {len(df[df['Market_Phase'] == 'DISTRIBUTION'])}"
    
    send_telegram_message_with_image(path, message)
    return True

def create_recommendation_dashboard(df, folder):
    """Create recommendation dashboard"""
    if df.empty:
        return False
    date = datetime.now().strftime('%Y-%m-%d')
    
    # Create figure with custom size and style
    plt.style.use('default')  # Using default matplotlib style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'axes.edgecolor': '#CCCCCC',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333'
    })
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Add main title with date
    fig.suptitle(f'PSX Trading Recommendations - {date}', fontsize=20, y=0.95)
    
    # Top 5 BUY/HOLD
    plt.subplot(2, 2, 1)
    buy_df = df[df['Status'] == 'BUY/HOLD'].nlargest(5, 'Phase_Probability')
    if not buy_df.empty:
        bars = plt.barh(buy_df['Symbol'], buy_df['Phase_Probability'], 
                       color='#2ecc71', alpha=0.8)
        for i, v in enumerate(buy_df['Phase_Probability']):
            plt.text(v + 1, i, f'{v:.1f}%', va='center')
        plt.title('Top BUY/HOLD Stocks', pad=20, fontsize=12)
        plt.xlabel('Accumulation Probability (%)', fontsize=10)
    
    # Top 5 OPPORTUNITY
    plt.subplot(2, 2, 2)
    opp_df = df[(df['Status'] == 'OPPORTUNITY') & 
                (df['Market_Phase'].isin(['ACCUMULATION', 'WEAK_ACCUMULATION']))]
    if not opp_df.empty:
        top_opps = opp_df.nlargest(5, 'Phase_Probability')
        bars = plt.barh(top_opps['Symbol'], top_opps['Phase_Probability'], 
                       color='#3498db', alpha=0.8)
        for i, v in enumerate(top_opps['Phase_Probability']):
            plt.text(v + 1, i, f'{v:.1f}%', va='center')
        plt.title('Top OPPORTUNITY Stocks', pad=20, fontsize=12)
        plt.xlabel('Accumulation Probability (%)', fontsize=10)
    
    # Top 5 SELL
    plt.subplot(2, 2, 3)
    sell_df = df[df['Status'] == 'SELL'].nlargest(5, 'Phase_Probability')
    if not sell_df.empty:
        bars = plt.barh(sell_df['Symbol'], sell_df['Phase_Probability'], 
                       color='#e74c3c', alpha=0.8)
        for i, v in enumerate(sell_df['Phase_Probability']):
            plt.text(v + 1, i, f'{v:.1f}%', va='center')
        plt.title('Top SELL Stocks', pad=20, fontsize=12)
        plt.xlabel('Distribution Probability (%)', fontsize=10)
    
    # Market Phase Summary
    plt.subplot(2, 2, 4)
    phase_summary = df['Market_Phase'].value_counts()
    colors = ['#27ae60', '#2ecc71', '#c0392b', '#7f8c8d']
    plt.pie(phase_summary, labels=phase_summary.index, autopct='%1.1f%%',
            colors=colors[:len(phase_summary)],
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title('Market Phase Summary', pad=20, fontsize=12)
    
    # Add footer with summary
    plt.figtext(0.5, 0.02, 
                f'Total Stocks: {len(df)} | BUY/HOLD: {len(df[df["Status"] == "BUY/HOLD"])} | '
                f'SELL: {len(df[df["Status"] == "SELL"])} | '
                f'OPPORTUNITY: {len(df[df["Status"] == "OPPORTUNITY"])}',
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save with high DPI
    path = os.path.join(folder, f'recommendations_{date}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Send to Telegram
    message = f"📈 PSX Trading Recommendations - {date}\n\n"
    message += f"Total Stocks: {len(df)}\n"
    message += f"BUY/HOLD: {len(df[df['Status'] == 'BUY/HOLD'])}\n"
    message += f"SELL: {len(df[df['Status'] == 'SELL'])}\n"
    message += f"OPPORTUNITY: {len(df[df['Status'] == 'OPPORTUNITY'])}"
    
    send_telegram_message_with_image(path, message)
    return True

def create_category_tables(df, folder, date):
    """Create tabular dashboards for each category"""
    # BUY/HOLD stocks
    buy_df = df[df['Status'] == 'BUY/HOLD'].copy()
    if not buy_df.empty:
        buy_df = buy_df[['Symbol', 'Close', 'RSI', 'AO', 'Market_Phase', 'Phase_Probability', 'Holding_Days', 'Profit_Loss']]
        buy_df.sort_values('Phase_Probability', ascending=False, inplace=True)
        buy_df['Profit_Loss'] = buy_df['Profit_Loss'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        buy_csv = os.path.join(folder, f'buy_stocks_{date}.csv')
        buy_df.to_csv(buy_csv, index=False)
        buy_html = os.path.join(folder, f'buy_stocks_{date}.html')
        buy_df.to_html(buy_html, index=False, classes='table table-striped table-hover', border=0)
    
    # SELL stocks
    sell_df = df[df['Status'] == 'SELL'].copy()
    if not sell_df.empty:
        sell_df = sell_df[['Symbol', 'Close', 'RSI', 'AO', 'Market_Phase', 'Phase_Probability']]
        sell_df.sort_values('Phase_Probability', ascending=False, inplace=True)
        sell_csv = os.path.join(folder, f'sell_stocks_{date}.csv')
        sell_df.to_csv(sell_csv, index=False)
        sell_html = os.path.join(folder, f'sell_stocks_{date}.html')
        sell_df.to_html(sell_html, index=False, classes='table table-striped table-hover', border=0)
    
    # OPPORTUNITY stocks
    opp_df = df[(df['Status'] == 'OPPORTUNITY') & (df['Market_Phase'] == 'ACCUMULATION')].copy()
    if not opp_df.empty:
        opp_df = opp_df[['Symbol', 'Close', 'RSI', 'AO', 'Phase_Probability']]
        opp_df.sort_values('Phase_Probability', ascending=False, inplace=True)
        opp_csv = os.path.join(folder, f'opportunity_stocks_{date}.csv')
        opp_df.to_csv(opp_csv, index=False)
        opp_html = os.path.join(folder, f'opportunity_stocks_{date}.html')
        opp_df.to_html(opp_html, index=False, classes='table table-striped table-hover', border=0)

def generate_portfolio_recommendations(df):
    """Generate portfolio management recommendations"""
    if df.empty:
        return "Insufficient data for portfolio recommendations"
    accumulation_pct = len(df[df['Market_Phase'] == 'ACCUMULATION']) / len(df) * 100
    bullish_rsi_pct = len(df[df['RSI'] > 50]) / len(df) * 100
    positive_ao_pct = len(df[df['AO'] > 0]) / len(df) * 100
    market_score = (accumulation_pct + bullish_rsi_pct + positive_ao_pct) / 3
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
    value_picks = df[(df['Market_Phase'] == 'ACCUMULATION') & 
                     (df['RSI'] < 50) & 
                     (df['AO'] > 0)].nlargest(5, 'Phase_Probability')
    growth_picks = df[(df['AO'] > 0) & 
                      (df['RSI'] > 50) & 
                      (df['RSI'] < 70)].nlargest(5, 'AO')
    avoid_picks = df[(df['Market_Phase'] == 'DISTRIBUTION') & 
                     (df['Phase_Probability'] > 70)].nlargest(5, 'Phase_Probability')
    recommendations = f"🔍 PORTFOLIO RECOMMENDATIONS ({market_condition} MARKET)\n\n"
    if market_condition in ["STRONGLY BULLISH", "MODERATELY BULLISH"]:
        recommendations += "📊 POSITION SIZING: Standard to aggressive\n"
        recommendations += "🎯 TARGET ALLOCATION: 80-100% invested\n"
    elif market_condition == "NEUTRAL":
        recommendations += "📊 POSITION SIZING: Standard\n"
        recommendations += "🎯 TARGET ALLOCATION: 60-80% invested\n"
    else:
        recommendations += "📊 POSITION SIZING: Reduced\n"
        recommendations += "🎯 TARGET ALLOCATION: 30-50% invested\n"
    recommendations += f"\n💼 STRATEGY ({market_score:.1f}% bullish):\n"
    if market_condition in ["STRONGLY BULLISH", "MODERATELY BULLISH"]:
        recommendations += "✅ Focus on growth/momentum\n"
        recommendations += "✅ Pyramid profitable positions\n"
        recommendations += "✅ Use trailing stops\n"
    elif market_condition == "NEUTRAL":
        recommendations += "✅ Balance growth/value\n"
        recommendations += "✅ Tighter stops\n"
        recommendations += "✅ Take partial profits\n"
    else:
        recommendations += "✅ Capital preservation\n"
        recommendations += "✅ High conviction value setups\n"
        recommendations += "✅ Reduce/hedge positions\n"
    recommendations += "\n🔝 TOP VALUE PICKS:\n"
    if not value_picks.empty:
        for i, (_, row) in enumerate(value_picks.iterrows(), 1):
            recommendations += f"{i}. {row['Symbol']} - {row['Phase_Probability']:.1f}% acc, RSI: {row['RSI']:.1f}\n"
    else:
        recommendations += "No clear value picks\n"
    recommendations += "\n🚀 TOP GROWTH PICKS:\n"
    if not growth_picks.empty:
        for i, (_, row) in enumerate(growth_picks.iterrows(), 1):
            recommendations += f"{i}. {row['Symbol']} - AO: {row['AO']:.2f}, RSI: {row['RSI']:.1f}\n"
    else:
        recommendations += "No clear growth picks\n"
    recommendations += "\n⚠️ STOCKS TO AVOID:\n"
    if not avoid_picks.empty:
        for i, (_, row) in enumerate(avoid_picks.iterrows(), 1):
            recommendations += f"{i}. {row['Symbol']} - {row['Phase_Probability']:.1f}% dist.\n"
    else:
        recommendations += "No stocks to avoid\n"
    return recommendations

def backtest_signals(symbol, buy_signals, sell_signals):
    """Backtest trading signals"""
    trades = []
    position = None
    for date, price in sorted(buy_signals + sell_signals, key=lambda x: x[0]):
        if date in [b[0] for b in buy_signals] and not position:
            position = {'entry_date': date, 'entry_price': price}
        elif date in [s[0] for s in sell_signals] and position:
            profit_loss = ((price - position['entry_price']) / position['entry_price']) * 100
            trades.append({
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': date,
                'profit_loss': profit_loss
            })
            position = None
    return pd.DataFrame(trades)

def run_backtest(available_symbols, database_path):
    """Run backtest on all symbols"""
    results = []
    engine = create_engine(f'sqlite:///{database_path}')
    for symbol in tqdm(available_symbols, desc="Backtesting"):
        try:
            table_name = f"PSX_{symbol}_stock_data"
            df = pd.read_sql(f"SELECT Date, Close FROM {table_name}", engine)
            df['Date'] = pd.to_datetime(df['Date'])
            buy_signals, sell_signals = get_buy_sell_signals(symbol)
            trades = backtest_signals(symbol, buy_signals, sell_signals)
            results.append(trades)
        except Exception as e:
            logging.error(f"Error backtesting {symbol}: {e}")
            continue
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def generate_stock_dashboard():
    """Generate dashboards for stock analysis"""
    try:
        database_path = config['database']['main_db']
        engine = create_engine(f'sqlite:///{database_path}')
        connection = engine.connect()
        cursor = connection.connection.cursor()
        available_symbols = get_available_symbols(cursor)
        buy_df = get_latest_buy_stocks()
        buy_symbols = set(buy_df['Stock'].tolist()) if not buy_df.empty else set()
        all_results = []
        print("\nAnalyzing stocks for dashboard...")
        for symbol in tqdm(available_symbols, desc="Processing stocks"):
            try:
                table_name = f"PSX_{symbol}_stock_data"
                available_columns = fetch_column_names(engine, table_name)
                required_cols = ["Date", "Close"]
                optional_cols = {
                    "RSI_weekly_Avg": None,
                    "AO_weekly_AVG": None,
                    "MA_30": None,
                    "RSI_weekly": None,
                    "Volume": None
                }
                if not all(col in available_columns for col in required_cols):
                    logging.warning(f"Missing required columns in {table_name}")
                    continue
                select_cols = required_cols + [col for col in optional_cols if col in available_columns]
                query = f"SELECT {', '.join(select_cols)} FROM {table_name} ORDER BY Date DESC LIMIT 60"
                df = pd.read_sql(query, connection)
                if df.empty:
                    continue
                df = df.copy()
                df['Date'] = pd.to_datetime(df['Date'])
                df['pct_change'] = df['Close'].pct_change() * 100
                for col, default_val in optional_cols.items():
                    if col not in df.columns:
                        df[col] = default_val
                latest = df.iloc[0] if not df.empty else None
                if latest is None:
                    continue
                buy_signals, sell_signals = get_buy_sell_signals(symbol)
                if buy_signals and sell_signals:
                    latest_buy = max(buy_signals, key=lambda x: x[0])
                    latest_sell = max(sell_signals, key=lambda x: x[0])
                    status = "BUY/HOLD" if latest_buy[0] > latest_sell[0] else "SELL"
                elif buy_signals:
                    status = "BUY/HOLD"
                elif sell_signals:
                    status = "SELL"
                else:
                    status = "OPPORTUNITY"
                market_phase, phase_probability, _ = calculate_market_phase(df, symbol)
                holding_days = None
                profit_loss = None
                if status == "BUY/HOLD" and symbol in buy_symbols:
                    stock_info = buy_df[buy_df['Stock'] == symbol].iloc[0]
                    holding_days = int(stock_info['holding_days']) if 'holding_days' in stock_info else None
                    if 'Signal_Close' in stock_info:
                        signal_price = float(stock_info['Signal_Close'])
                        current_price = latest['Close']
                        profit_loss = ((current_price - signal_price) / signal_price) * 100
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
                logging.error(f"Error processing {symbol} for dashboard: {e}")
                print(f"Error processing {symbol} for dashboard: {str(e)}")
                continue
        dashboard_df = pd.DataFrame(all_results)
        if not dashboard_df.empty:
            current_date = datetime.now().strftime('%Y-%m-%d')
            create_market_overview_dashboard(dashboard_df, config['output']['dashboards_folder'])
            create_recommendation_dashboard(dashboard_df, config['output']['dashboards_folder'])
            create_category_tables(dashboard_df, config['output']['dashboards_folder'], current_date)
        connection.close()
        return dashboard_df
    except Exception as e:
        logging.error(f"Error generating dashboard: {e}")
        print(f"Error generating dashboard: {str(e)}")
        return pd.DataFrame()

def main():
    """Main function to run the analysis"""
    try:
        # Check if configuration is loaded
        if config is None:
            logging.error("Failed to load configuration. Using default configuration.")
            print("Error: Failed to load configuration. Using default configuration.")
            return False
            
        # Check database files
        if not check_database_files():
            logging.error("Database files check failed.")
            logging.error("Please ensure the database files exist before running the analysis.")
            print("Error: Database files check failed.")
            print("Please ensure the database files exist before running the analysis.")
            return False
            
        print("Database checks passed. Proceeding with analysis...")
        # Rest of the main function...
        # Parse command-line arguments
        args = parse_args()
        print("Parsed command-line arguments:", args)
        
        # Get available symbols
        database_path = config['database']['main_db']
        print(f"Connecting to database at: {database_path}")
        engine = create_engine(f'sqlite:///{database_path}')
        connection = engine.connect()
        cursor = connection.connection.cursor()
        print("Successfully connected to database for symbol retrieval")
        
        available_symbols = get_available_symbols(cursor)
        print(f"Found {len(available_symbols)} available symbols: {available_symbols[:5]}...")
        
        if not available_symbols:
            logging.error("No symbols available for analysis")
            print("Error: No symbols available for analysis")
            connection.close()
            return False
        
        # Handle backtest option
        if args.backtest:
            print("Running backtest...")
            backtest_results = run_backtest(available_symbols, database_path)
            if not backtest_results.empty:
                output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                backtest_results.to_csv(output_file, index=False)
                print(f"Backtest results saved to {output_file}")
            else:
                logging.error("Backtest returned no results")
                print("Error: Backtest returned no results")
            connection.close()
            return True
        
        # Handle dashboard-only option
        if args.dashboard_only:
            print("Generating dashboard only...")
            dashboard_df = generate_stock_dashboard()
            if dashboard_df.empty:
                logging.error("Failed to generate dashboard data")
                print("Error: Failed to generate dashboard data")
            else:
                print(f"Dashboard generated with {len(dashboard_df)} stocks")
            connection.close()
            return True
        
        # Handle specific symbols if provided
        symbols_to_process = args.symbols if args.symbols else available_symbols
        print(f"Processing {len(symbols_to_process)} symbols")
        
        # Get latest buy stocks
        buy_df = get_latest_buy_stocks()
        print(f"Retrieved {len(buy_df)} latest buy stocks")
        latest_buy_symbols = buy_df['Stock'].tolist() if not buy_df.empty else []
        print(f"Latest buy symbols: {latest_buy_symbols[:5]}...")
        
        # Generate charts for latest buy symbols
        if latest_buy_symbols:
            print(f"Generating charts for {len(latest_buy_symbols)} buy symbols")
            generate_charts_parallel(latest_buy_symbols, database_path, available_symbols)
        else:
            print("No buy symbols to generate charts for")
        
        # Send summary if Telegram is enabled
        if config['telegram'].get('enabled', False):
            print("Sending Telegram summary")
            sell_df = get_latest_sell_stocks()
            send_signals_and_charts_summary(buy_df, sell_df, symbols_to_process, len(available_symbols))
        else:
            print("Telegram notifications are disabled")
        
        # Generate dashboard as final step
        print("Generating final dashboard")
        dashboard_df = generate_stock_dashboard()
        if dashboard_df.empty:
            logging.error("Failed to generate final dashboard data")
            print("Error: Failed to generate final dashboard data")
        else:
            print(f"Final dashboard generated with {len(dashboard_df)} stocks")
        
        connection.close()
        print("Analysis completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        print(f"Error in main function: {str(e)}")
        return False

if __name__ == "__main__":
    # Set encoding to UTF-8 to handle Unicode characters
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    success = main()
    if not success:
        logging.error("Script execution failed. Please check the logs for details.")
        print("Error: Script execution failed. Please check the logs for details.")
        sys.exit(1)
    sys.exit(0)

def draw_trend_lines(ax, df, column, color, label):
    """Draw trend lines for a given column"""
    try:
        x = np.arange(len(df))
        y = df[column].values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:
            return
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        coeffs = np.polyfit(x_valid, y_valid, 1)
        trend_line = np.polyval(coeffs, x)
        ax.plot(df['Date'], trend_line, color=color, linestyle='--', label=label, alpha=0.7)
    except Exception as e:
        logging.error(f"Error drawing trend for {label}: {str(e)}")
        print(f"Error drawing trend for {label}: {str(e)}")

def generate_chart_for_symbol(symbol, database_path, available_symbols):
    """Generate chart for a specific symbol"""
    try:
        engine = create_engine(f'sqlite:///{database_path}')
        connection = engine.connect()
        table_name = f"PSX_{symbol}_stock_data"
        available_columns = fetch_column_names(engine, table_name)
        logging.info(f"Columns in table {table_name}: {available_columns}")
        print(f"Columns in table {table_name}: {available_columns}")
        query = f"SELECT Date, Close, RSI_weekly_Avg, AO_weekly_AVG, MA_30 FROM {table_name} ORDER BY Date DESC LIMIT 180"
        df = pd.read_sql(query, connection)
        if df.empty:
            logging.warning(f"No data found for {symbol}")
            connection.close()
            return
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        buy_signals, sell_signals = get_buy_sell_signals(symbol)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        ax1.plot(df['Date'], df['Close'], label='Close', color='black', alpha=0.8)
        if 'MA_30' in df.columns:
            ax1.plot(df['Date'], df['MA_30'], label='MA 30', color='orange', alpha=0.7)
        for date, price, _ in buy_signals:
            ax1.plot(date, price, marker='^', color='green', markersize=10, label='Buy Signal' if buy_signals.index((date, price, 'Buy')) == 0 else "")
        for date, price, _ in sell_signals:
            ax1.plot(date, price, marker='v', color='red', markersize=10, label='Sell Signal' if sell_signals.index((date, price, 'Sell')) == 0 else "")
        ax1.set_title(f"Stock Price with Signals - {symbol}")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        draw_trend_lines(ax1, df, 'Close', 'blue', 'Price Trend')
        if 'RSI_weekly_Avg' in df.columns:
            ax2.plot(df['Date'], df['RSI_weekly_Avg'], label='RSI Weekly Avg', color='purple')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.3)
            ax2.set_ylabel("RSI")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            draw_trend_lines(ax2, df, 'RSI_weekly_Avg', 'purple', 'RSI Trend')
        if 'AO_weekly_AVG' in df.columns:
            ax3.bar(df['Date'], df['AO_weekly_AVG'], label='AO Weekly AVG', color=df['AO_weekly_AVG'].apply(lambda x: 'green' if x > 0 else 'red'), alpha=0.5)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_ylabel("AO")
            ax3.set_xlabel("Date")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        chart_dir = os.path.join(config['output']['charts_folder'], datetime.now().strftime('%Y%m%d'))
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = os.path.join(chart_dir, f"{symbol}_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Generated chart for {symbol} at {chart_path}")
        connection.close()
    except Exception as e:
        logging.error(f"Error generating chart for {symbol}: {e}")
        print(f"Error generating chart for {symbol}: {str(e)}")

def generate_stock_dashboard(buy_stocks, sell_stocks, opp_stocks):
    try:
        # Calculate percentages
        total = len(buy_stocks) + len(sell_stocks) + len(opp_stocks)
        buy_pct = (len(buy_stocks) / total * 100) if total > 0 else 0
        sell_pct = (len(sell_stocks) / total * 100) if total > 0 else 0
        opp_pct = (len(opp_stocks) / total * 100) if total > 0 else 0
        
        # Create dashboard HTML
        filename = os.path.join(config['output']['dashboards_folder'], f'stock_dashboard_{datetime.now().strftime("%Y%m%d")}.html')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>PSX Stock Dashboard</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .chart-container { width: 80%; margin: 20px auto; }
                </style>
            </head>
            <body>
                <h1>PSX Stock Dashboard</h1>
                <div class="chart-container">
                    <canvas id="signalChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="breadthChart"></canvas>
                </div>
                <script>
                    // Signal Distribution Chart
                    new Chart(document.getElementById('signalChart'), {
                        type: 'pie',
                        data: {
                            labels: ['BUY/HOLD', 'SELL', 'OPPORTUNITY'],
                            datasets: [{
                                data: [""" + f"{buy_pct:.2f}" + """, """ + f"{sell_pct:.2f}" + """, """ + f"{opp_pct:.2f}" + """],
                                backgroundColor: ['rgba(0, 255, 0, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)'],
                                borderColor: ['rgba(0, 255, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(0, 0, 255, 1)'],
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: True,
                            plugins: {
                                legend: {position: 'top'},
                                title: {display: True, text: 'Stock Signal Distribution'}
                            }
                        }
                    });
                    
                    // Market Breadth Chart
                    new Chart(document.getElementById('breadthChart'), {
                        type: 'bar',
                        data: {
                            labels: ['BUY/HOLD', 'SELL', 'OPPORTUNITY'],
                            datasets: [{
                                label: 'Signal Distribution',
                                data: [""" + f"{buy_pct:.2f}" + """, """ + f"{sell_pct:.2f}" + """, """ + f"{opp_pct:.2f}" + """],
                                backgroundColor: ['rgba(0, 255, 0, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)'],
                                borderColor: ['rgba(0, 255, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(0, 0, 255, 1)'],
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: True,
                            scales: {
                                y: {
                                    beginAtZero: True,
                                    max: 100,
                                    title: {display: True, text: 'Percentage (%)'}
                                }
                            },
                            plugins: {
                                legend: {display: False},
                                title: {display: True, text: 'Market Breadth Indicator'}
                            }
                        }
                    });
                </script>
            </body>
            </html>
            """)
        logging.info(f"Dashboard generated at {filename}")
        print(f"Dashboard generated at {filename}")
        return True
    except Exception as e:
        logging.error(f"Error generating dashboard: {e}")
        print(f"Error generating dashboard: {str(e)}")
        return False

