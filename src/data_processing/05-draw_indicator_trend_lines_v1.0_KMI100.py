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
from PIL import Image

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


def get_kmi100_symbols():
    """Read KMI100 symbols from Excel file"""
    try:
        file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        print(f"Attempting to read KMI100 symbols from: {file_path}")
        if not os.path.exists(file_path):
            logging.warning(f"KMI100 symbols file not found at {file_path}")
            print(f"KMI100 symbols file not found at {file_path}")
            return []
            
        # Read symbols from Excel file
        try:
            df = pd.read_excel(file_path, sheet_name='KMI100')
            print(f"Successfully read Excel file, found {len(df)} rows")
            # Make column name search case-insensitive
            symbol_col = next((col for col in df.columns if col.strip().lower() == 'symbol'), None)
            if not symbol_col:
                logging.error("'Symbol' column not found in KMI100 sheet (case-insensitive search)")
                print("Error: 'Symbol' column not found in KMI100 sheet (case-insensitive search)")
                # Fall back to default symbols
                default_symbols = ['AICL', 'ATRL', 'BAFL', 'BAHL', 'CNERGY', 'EFERT', 'ENGRO', 'FFBL', 'FFC', 'FCCL', 'HUBC', 'HBL', 'ISL', 'ILP', 'LUCK', 'MCB', 'MARI', 'MEBL', 'MLCF', 'MTL', 'NBP', 'NML', 'OGDC', 'PAKT', 'PPL', 'PIOC', 'PSO', 'SNGP', 'SSGC', 'UBL']
                print(f"Falling back to default list of {len(default_symbols)} symbols")
                return default_symbols
            symbols = df[symbol_col].str.strip().str.upper().tolist()
            # Only return the top 100 symbols
            symbols = symbols[:100]
            logging.info(f"Successfully read {len(symbols)} KMI100 symbols from Excel file (top 100)")
            print(f"Successfully read {len(symbols)} KMI100 symbols: {symbols[:5]}...")
            return symbols
        except Exception as e:
            logging.error(f"Error reading Excel file: {str(e)}")
            print(f"Error reading Excel file: {str(e)}")
            # Fall back to default symbols
            default_symbols = ['AICL', 'ATRL', 'BAFL', 'BAHL', 'CNERGY', 'EFERT', 'ENGRO', 'FFBL', 'FFC', 'FCCL', 'HUBC', 'HBL', 'ISL', 'ILP', 'LUCK', 'MCB', 'MARI', 'MEBL', 'MLCF', 'MTL', 'NBP', 'NML', 'OGDC', 'PAKT', 'PPL', 'PIOC', 'PSO', 'SNGP', 'SSGC', 'UBL']
            print(f"Falling back to default list of {len(default_symbols)} symbols")
            return default_symbols
    except Exception as e:
        logging.error(f"Error reading KMI100 symbols: {e}")
        print(f"Error reading KMI100 symbols: {str(e)}")
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
        # First get KMI100 symbols from Excel
        KMI100_symbols = set(get_kmi100_symbols())
        print(f"Retrieved {len(KMI100_symbols)} KMI100 symbols")
        
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
            # Only include symbols that are in KMI100
            if symbol in KMI100_symbols:
                available_symbols.append(symbol)
                print(f"Added symbol {symbol} to available symbols")
                
        logging.info(f"Found {len(available_symbols)} available KMI100 symbols")
        print(f"Found {len(available_symbols)} available KMI100 symbols")
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
    try:
        if not config.get('telegram', {}).get('enabled', False):
            logging.warning("Telegram messaging is disabled in configuration.")
            return False
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        max_images_per_batch = 6          # was 10
        default_delay_between_batches = 7 # was 5
        bot_token = config['telegram']['bot_token']
        chat_id = config['telegram']['chat_id']
        prepared_images = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                continue
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    logging.info(f"Original image {image_path}: {width}x{height}")
                    if width < 320 or height < 320:
                        logging.error(f"Image {image_path} has invalid dimensions: {width}x{height}. Skipping send.")
                        continue
                    MAX_DIM = 4096
                    if width > MAX_DIM or height > MAX_DIM:
                        ratio = min(MAX_DIM / width, MAX_DIM / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        width, height = img.size
                        logging.info(f"Resized image {image_path} to {width}x{height} for Telegram (max 4096px).")
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        logging.info(f"Converted image {image_path} to RGB mode.")
                    aspect_ratio = max(width/height, height/width)
                    logging.info(f"Image {image_path} aspect ratio: {aspect_ratio:.2f}")
                    if aspect_ratio > 20:
                        logging.error(f"Image {image_path} has invalid aspect ratio: {aspect_ratio}. Skipping send.")
                        continue
                    if width < 320 or height < 320 or width > MAX_DIM or height > MAX_DIM:
                        logging.error(f"Image {image_path} still has invalid dimensions after resizing: {width}x{height}. Skipping send.")
                        continue
                    temp_path = f"{image_path}_temp.png"
                    img.save(temp_path, "PNG")
                    prepared_images.append(temp_path)
            except Exception as e:
                logging.error(f"Failed to process image {image_path}: {e}")
                continue
        i = 0
        while i < len(prepared_images):
            batch = prepared_images[i:i+max_images_per_batch]
            if not batch:
                i += max_images_per_batch
                continue
            if len(batch) == 1:
                _send_telegram_api_call(message, batch[0])
            else:
                url = f"https://api.telegram.org/bot{bot_token}/sendMediaGroup"
                media = []
                files = {}
                for idx, img_path in enumerate(batch):
                    file_key = f"photo{idx}"
                    files[file_key] = open(img_path, 'rb')
                    media.append({
                        "type": "photo",
                        "media": f"attach://{file_key}",
                        "caption": message if idx == 0 and message else None,
                        "parse_mode": "Markdown" if idx == 0 and message else None
                    })
                for m in media:
                    keys_to_remove = [k for k, v in m.items() if v is None]
                    for k in keys_to_remove:
                        del m[k]
                data = {
                    "chat_id": chat_id,
                    "media": json.dumps(media)
                }
                try:
                    response = requests.post(url, data=data, files=files)
                    if response.status_code == 429:
                        retry_after = response.headers.get('Retry-After')
                        wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 30
                        logging.warning(f"Rate limit hit, waiting {wait_time} seconds")
                        for f in files.values():
                            f.close()
                        time.sleep(wait_time)
                        continue  # retry this batch after waiting
                    response.raise_for_status()
                except Exception as e:
                    logging.error(f"Failed to send media group: {e}")
                finally:
                    for f in files.values():
                        f.close()
            i += max_images_per_batch
            time.sleep(default_delay_between_batches)
        for temp_path in prepared_images:
            try:
                os.remove(temp_path)
            except Exception:
                pass
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
def draw_indicator_trend_lines_with_signals(df, symbol, output_folder, entry_date=None):
    try:
        # Use the full range of available data for a comprehensive view of trends
        df = df.copy()  # No filtering, use all available data
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 22,
            'axes.titleweight': 'bold',
            'axes.edgecolor': '#CCCCCC',
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'text.color': '#333333',
            'axes.facecolor': '#f7f7f7',
            'figure.facecolor': '#f7f7f7'
        })

        # Create subplot with 3 rows for Price, RSI, and AO (no ATR)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot Price on first axis
        ax1.plot(df['Date'], df['Close'], label=f'{symbol} Close', color='#1f77b4', linewidth=2.5)

        # --- Add crossovers between timeframes ---
        crossover_caption = ""
        try:
            cross_df = df[['Date', 'Open', 'Close']].copy() if 'Date' in df.columns else df.reset_index()[['Date', 'Open', 'Close']]
            weekly, monthly, three_monthly, cross_wk_mo_dates, cross_mo_3mo_dates = calculate_timeframe_crosses(cross_df, print_crosses=False)
            # Plot vertical lines for weekly/monthly crossovers
            crossover_plotted = set()
            for dt in cross_wk_mo_dates:
                if 'wk_mo' not in crossover_plotted:
                    ax1.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.5, label='Weekly/Monthly Close Crossover')
                    crossover_plotted.add('wk_mo')
                else:
                    ax1.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.5)
            for dt in cross_mo_3mo_dates:
                if 'mo_3mo' not in crossover_plotted:
                    ax1.axvline(dt, color='cyan', linestyle=':', alpha=0.7, linewidth=1.5, label='Monthly/3M Close Crossover')
                    crossover_plotted.add('mo_3mo')
                else:
                    ax1.axvline(dt, color='cyan', linestyle=':', alpha=0.7, linewidth=1.5)
            # Prepare caption info for the most recent 3 cross dates of each type
            if len(cross_wk_mo_dates) > 0:
                crossover_caption += "\nWeekly/Monthly Close Crosses (latest):\n"
                for dt in list(cross_wk_mo_dates)[-3:]:
                    w = weekly.loc[weekly.index <= dt].iloc[-1]['Close'] if not weekly.loc[weekly.index <= dt].empty else None
                    m = monthly.loc[monthly.index <= dt].iloc[-1]['Close'] if not monthly.loc[monthly.index <= dt].empty else None
                    crossover_caption += f"{dt.date()}: W={w:.2f} M={m:.2f}\n"
            if len(cross_mo_3mo_dates) > 0:
                crossover_caption += "Monthly/3M Close Crosses (latest):\n"
                for dt in list(cross_mo_3mo_dates)[-3:]:
                    m = monthly.loc[monthly.index <= dt].iloc[-1]['Close'] if not monthly.loc[monthly.index <= dt].empty else None
                    q = three_monthly.loc[three_monthly.index <= dt].iloc[-1]['Close'] if not three_monthly.loc[three_monthly.index <= dt].empty else None
                    crossover_caption += f"{dt.date()}: M={m:.2f} 3M={q:.2f}\n"
        except Exception as e:
            logging.warning(f"Could not plot crossovers: {e}")

        buy_signals = get_buy_signals(symbol)
        sell_signals = get_sell_signals(symbol)
        
        # Calculate analysis metrics
        analysis = {}
        current_price = df['Close'].iloc[-1]
        
        # Get entry price from buy signals
        entry_price = None
        if entry_date:
            entry_data = df[df['Date'] == entry_date]
            if not entry_data.empty:
                entry_price = entry_data['Close'].iloc[0]
                print(f"Found entry price for {symbol}: {entry_price}")
        
        # Calculate profit/loss
        profit_loss = calculate_profit_loss(current_price, entry_price)
        print(f"Calculated P/L for {symbol}: {profit_loss}")
        
        # Calculate holding days
        holding_days = calculate_holding_days(entry_date)
        print(f"Calculated holding days for {symbol}: {holding_days}")
        
        # Calculate enhanced stop loss
        stop_loss = calculate_enhanced_stop_loss(df, current_price, analysis)
        print(f"Calculated stop loss for {symbol}: {stop_loss}")
        
        # Calculate take profit
        take_profit = calculate_take_profit(current_price, stop_loss)
        print(f"Calculated take profit for {symbol}: {take_profit}")
        
        # Store the values for the caption
        chart_values = {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': current_price,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'profit_loss': profit_loss,
            'holding_days': holding_days
        }
        
        print(f"Chart values for {symbol}: {chart_values}")
        
        # Plot buy signals
        for date, price in buy_signals:
            ax1.scatter(pd.to_datetime(date), price, marker='^', color='green', s=120, zorder=5, label='Buy Signal')
            ax1.annotate(f'BUY\n{pd.to_datetime(date).strftime("%Y-%m-%d")}\nPrice: {price:.2f}', (pd.to_datetime(date), price),
                         textcoords="offset points", xytext=(0, 20), ha='center', fontsize=12, fontweight='bold', color='green',
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', lw=2, alpha=0.8))
        
        # Plot sell signals
        for date, price in sell_signals:
            ax1.scatter(pd.to_datetime(date), price, marker='v', color='red', s=120, zorder=5, label='Sell Signal')
            ax1.annotate(f'SELL\n{pd.to_datetime(date).strftime("%Y-%m-%d")}\nPrice: {price:.2f}', (pd.to_datetime(date), price),
                         textcoords="offset points", xytext=(0, -25), ha='center', fontsize=12, fontweight='bold', color='red',
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', lw=2, alpha=0.8))

        # Plot stop loss and take profit
        if stop_loss and stop_loss > 0:
            ax1.axhline(stop_loss, color='orange', linestyle='-.', linewidth=2, label='Stop Loss')
            ax1.text(df['Date'].iloc[0], stop_loss, f'Stop Loss: {stop_loss:.2f}', 
                    color='orange', fontsize=13, va='bottom', ha='left', 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))
        
        if take_profit and take_profit > current_price:
            ax1.axhline(take_profit, color='green', linestyle='-.', linewidth=2, label='Take Profit')
            ax1.text(df['Date'].iloc[0], take_profit, f'Take Profit: {take_profit:.2f}', 
                    color='green', fontsize=13, va='bottom', ha='left', 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

        # Fibonacci Retracement
        recent_high = df['Close'].max()
        recent_low = df['Close'].min()
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        fib_prices = [recent_high - (recent_high - recent_low) * level for level in fib_levels]
        fib_labels = [f"Fib {int(level*100)}%" for level in fib_levels]
        for price, label in zip(fib_prices, fib_labels):
            ax1.axhline(price, linestyle='--', alpha=0.5, color='purple')
            ax1.text(df['Date'].iloc[-1], price, label, color='purple', fontsize=11, va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7, edgecolor='purple'))

        # Entry/Exit Suggestion Logic with AO, RSI, RSI Crossovers (no ATR)
        rsi_monthly = df['rs_monthly'] if 'rs_monthly' in df else None
        rsi_3monthly = df['RSI_3months_Avg'] if 'RSI_3months_Avg' in df else None
        rsi_weekly = df['RSI_weekly_Avg'] if 'RSI_weekly_Avg' in df else None
        rsi = rsi_monthly if rsi_monthly is not None else rsi_weekly
        if rsi is None:
            logging.warning(f"No RSI data available for {symbol}")
            print(f"No RSI data available for {symbol}")
        ao = df['AO_weekly_AVG'] if 'AO_weekly_AVG' in df else None
        volume = df['Volume'] if 'Volume' in df else None
        close = df['Close']
        entry_suggestion = exit_suggestion = take_profit = stop_loss = None
        suggestion_reason = ""
        entry_date = holding_days = profit_loss = None
        justification = ""
        rsi_crossover = False
        if rsi is not None and ao is not None and volume is not None:
            if rsi_monthly is not None and rsi_3monthly is not None:
                # Check for crossovers (bullish: monthly RSI crosses above 3-monthly RSI)
                df['RSI_crossover_bullish'] = (df['rs_monthly'].shift(1) < df['RSI_3months_Avg'].shift(1)) & (df['rs_monthly'] > df['RSI_3months_Avg'])
                df['RSI_crossover_bearish'] = (df['rs_monthly'].shift(1) > df['RSI_3months_Avg'].shift(1)) & (df['rs_monthly'] < df['RSI_3months_Avg'])
                if df['RSI_crossover_bullish'].any():
                    rsi_crossover = True
                    crossover_date = df[df['RSI_crossover_bullish']]['Date'].iloc[-1] if not df[df['RSI_crossover_bullish']].empty else None
                    if crossover_date:
                        ax2.axvline(crossover_date, color='lime', linestyle='--', alpha=0.7, label='RSI Bullish Crossover')
                if df['RSI_crossover_bearish'].any():
                    crossover_date = df[df['RSI_crossover_bearish']]['Date'].iloc[-1] if not df[df['RSI_crossover_bearish']].empty else None
                    if crossover_date:
                        ax2.axvline(crossover_date, color='darkred', linestyle='--', alpha=0.7, label='RSI Bearish Crossover')
            for fib, price in zip(fib_labels[::-1], fib_prices[::-1]):
                idx = (close - price).abs().idxmin()
                rsi_val = rsi.iloc[idx] if idx in rsi.index else None
                ao_val = ao.iloc[idx] if idx in ao.index else None
                vol_val = volume.iloc[idx] if idx in volume.index else None
                if rsi_val is not None and ao_val is not None and vol_val is not None:
                    if rsi_val < 40 and ao_val > 0 and vol_val > volume.mean() or (rsi_monthly is not None and rsi_3monthly is not None and rsi_crossover):
                        entry_suggestion = price
                        entry_date = df['Date'].iloc[idx]
                        holding_days = (df['Date'].iloc[-1] - entry_date).days
                        profit_loss = ((df['Close'].iloc[-1] - entry_suggestion) / entry_suggestion) * 100
                        # Simple stop-loss and take-profit based on recent high/low
                        stop_loss = recent_low
                        take_profit = recent_high
                        rsi_label = "RSI Monthly" if rsi_monthly is not None else "RSI Weekly"
                        suggestion_reason = (
                            f"Entry suggested at {entry_suggestion:.2f} on {entry_date.strftime('%Y-%m-%d')} due to bullish confluence: "
                            f"{rsi_label}={rsi_val:.1f} (<40), AO={ao_val:.2f} (>0), Volume spike above mean near {fib} retracement"
                            f"{' and RSI Bullish Crossover' if (rsi_monthly is not None and rsi_3monthly is not None and rsi_crossover) else ''}"
                        )
                        justification = (
                            f"Signal Date: {entry_date.strftime('%Y-%m-%d')}\n"
                            f"Reason: {rsi_label}={rsi_val:.1f} (<40), AO={ao_val:.2f} (>0), "
                            f"Volume={vol_val:.0f} (>{volume.mean():.0f}), Price near {fib} retracement"
                            f"{' + RSI Bullish Crossover' if (rsi_monthly is not None and rsi_3monthly is not None and rsi_crossover) else ''}"
                        )
                        break
            if entry_suggestion:
                exit_suggestion = take_profit
            else:
                exit_suggestion = recent_high
                take_profit = recent_high
                stop_loss = recent_low

        # Annotate suggestions
        if entry_suggestion:
            ax1.axhline(entry_suggestion, color='green', linestyle='-', linewidth=2, label='Suggested Entry')
            ax1.text(df['Date'].iloc[0], entry_suggestion, 'Entry', color='green', fontsize=13, va='bottom', ha='left', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
            ax1.axvline(entry_date, color='green', linestyle=':', alpha=0.7)
            # Removed justification annotation from chart to keep reasons only in caption
            # ax1.annotate(justification, (entry_date, entry_suggestion), textcoords="offset points", xytext=(30, 30), ha='left', fontsize=12, color='black', bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='green', lw=2, alpha=0.9))
            ax1.text(df['Date'].iloc[-1], entry_suggestion, f'Holding: {holding_days}d\nP/L: {profit_loss:.2f}%', color='blue', fontsize=12, va='bottom', ha='left', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))
        if exit_suggestion:
            ax1.axhline(exit_suggestion, color='red', linestyle='-', linewidth=2, label='Suggested Exit/TP')
            ax1.text(df['Date'].iloc[0], exit_suggestion, 'Exit/TP', color='red', fontsize=13, va='bottom', ha='left', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        if stop_loss:
            ax1.axhline(stop_loss, color='orange', linestyle='-.', linewidth=2, label='Suggested Stop')
            ax1.text(df['Date'].iloc[0], stop_loss, 'Stop', color='orange', fontsize=13, va='bottom', ha='left', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))

        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylabel('Price', fontsize=14, fontweight='bold', color='#1f77b4')
        ax1.set_title(f'{symbol} Stock Analysis with Trading Signals', fontsize=18, fontweight='bold', pad=15)
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Diagnostics after SQL query and before plotting
        print(f"Columns in DataFrame: {df.columns.tolist()}")
        # List of possible monthly and 3-month RSI column names
        monthly_rsi_candidates = ['rs_monthly', 'RSI_monthly', 'RSI_monthly_Avg']
        three_month_rsi_candidates = ['RSI_3months_Avg', 'RSI_3monthly', 'RSI_3month', 'RSI_3m']
        # Find the first available, non-empty column for each
        rsi_monthly_col = next((col for col in monthly_rsi_candidates if col in df and not df[col].isna().all()), None)
        rsi_3monthly_col = next((col for col in three_month_rsi_candidates if col in df and not df[col].isna().all()), None)
        if rsi_monthly_col:
            print(f"Using monthly RSI column: {rsi_monthly_col}, head: {df[rsi_monthly_col].head()}")
        else:
            print("No valid monthly RSI column found.")
        if rsi_3monthly_col:
            print(f"Using 3-month RSI column: {rsi_3monthly_col}, head: {df[rsi_3monthly_col].head()}")
        else:
            print("No valid 3-month RSI column found.")

        # Plot RSI on second axis
        if 'RSI_monthly' in df.columns and 'RSI_3months_Avg' in df.columns:
            ax2.plot(df['Date'], df['RSI_monthly'], color='blue', label='RSI Monthly', linewidth=2)
            ax2.plot(df['Date'], df['RSI_3months_Avg'], color='purple', label='RSI 3-Month', linewidth=2)
        elif 'RSI_weekly_Avg' in df.columns:
            ax2.plot(df['Date'], df['RSI_weekly_Avg'], color='blue', label='RSI Weekly', linewidth=2)
        else:
            logging.warning(f"No RSI data available for {symbol}")
            print(f"Warning: No RSI data available for {symbol}")
        
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if entry_date is not None:
            ax2.axvline(entry_date, color='green', linestyle=':', alpha=0.7)
        
        # Plot AO on third axis
        if ao is not None:
            # Create separate series for positive and negative values
            positive_ao = df['AO_weekly_AVG'].copy()
            negative_ao = df['AO_weekly_AVG'].copy()
            positive_ao[positive_ao <= 0] = np.nan
            negative_ao[negative_ao > 0] = np.nan
            
            ax3.bar(df['Date'], positive_ao, color='green', alpha=0.7, label='Positive AO')
            ax3.bar(df['Date'], negative_ao, color='red', alpha=0.7, label='Negative AO')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_ylabel("AO")
            ax3.set_xlabel("Date")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            if entry_date is not None:
                ax3.axvline(entry_date, color='green', linestyle=':', alpha=0.7)

        plt.tight_layout()
        lines1, labels1 = ax1.get_legend_handles_labels()
        # Remove duplicate labels
        seen = set()
        new_lines = []
        new_labels = []
        for l, lab in zip(lines1, labels1):
            if lab not in seen:
                new_lines.append(l)
                new_labels.append(lab)
                seen.add(lab)
        ax1.legend(new_lines, new_labels, loc='best', fontsize=12, frameon=True, shadow=True)
        ax2.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        ax3.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        chart_path = os.path.join(output_folder, f'{symbol}_stock_analysis.png')
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create the caption for the chart
        caption = f"{symbol} Stock Analysis Chart\n"
        
        # Add entry date if available
        if entry_date:
            entry_date_str = pd.to_datetime(entry_date).strftime('%Y-%m-%d')
            caption += f"Entry Date: {entry_date_str}\n"
            print(f"Added entry date to caption: {entry_date_str}")
        
        # Add holding days if available
        if holding_days is not None:
            caption += f"Holding Days: {holding_days}\n"
            print(f"Added holding days to caption: {holding_days}")
        
        # Add profit/loss if available
        if profit_loss is not None:
            profit_loss_str = f"{profit_loss:+.2f}%"
            caption += f"P/L: {profit_loss_str}\n"
            print(f"Added P/L to caption: {profit_loss_str}")
        
        # Add current price
        caption += f"Current Price: {current_price:.2f}\n"
        print(f"Added current price to caption: {current_price:.2f}")
        
        # Add take profit and stop loss
        if take_profit:
            caption += f"Take Profit: {take_profit:.2f}\n"
            print(f"Added take profit to caption: {take_profit:.2f}")
        if stop_loss:
            caption += f"Stop Loss: {stop_loss:.2f}\n"
            print(f"Added stop loss to caption: {stop_loss:.2f}")
        # Add crossover info to caption
        if crossover_caption:
            caption += "\n" + crossover_caption
        print(f"Final caption for {symbol}:\n{caption}")
        
        # Send the chart with caption if Telegram is enabled
        if config.get('telegram', {}).get('enabled', False):
            send_telegram_message_with_image(chart_path, caption)
            print(f"Sent chart with caption for {symbol}")
        
        return True, chart_values
        
    except Exception as e:
        logging.error(f"Error in draw_indicator_trend_lines_with_signals: {str(e)}")
        print(f"Error in draw_indicator_trend_lines_with_signals: {str(e)}")
        return False, None

def send_signals_and_charts_summary(buy_df, sell_df, symbols, total_processed):
    try:
        if not config.get('telegram', {}).get('enabled', False):
            return

        # Sort buy_df by holding days in ascending order
        if not buy_df.empty and 'Holding_Days' in buy_df.columns:
            buy_df = buy_df.sort_values('Holding_Days', ascending=True)
            buy_df = buy_df.reset_index(drop=True)

        # Get the charts folder for today
        charts_folder = os.path.join(config['output']['charts_folder'], datetime.now().strftime('%Y%m%d'))
        
        if not os.path.exists(charts_folder):
            logging.warning(f"Charts folder not found: {charts_folder}")
            return

        # Process each symbol and collect chart values
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            success, chart_values = generate_chart_for_symbol(symbol, config['database']['main_db'], get_available_symbols(cursor))
            if success and chart_values:
                print(f"Got chart values for {symbol}: {chart_values}")
                
                # Create the message for this symbol
                symbol_message = f"\n{symbol} Stock Analysis Chart"
                
                # Add entry date if available
                if chart_values['entry_date']:
                    entry_date_str = pd.to_datetime(chart_values['entry_date']).strftime('%Y-%m-%d')
                    symbol_message += f"\nEntry Date: {entry_date_str}"
                    print(f"Added entry date to message: {entry_date_str}")
                
                # Add holding days if available
                if chart_values['holding_days'] is not None:
                    symbol_message += f"\nHolding Days: {chart_values['holding_days']}"
                    print(f"Added holding days to message: {chart_values['holding_days']}")
                
                # Add profit/loss if available
                if chart_values['profit_loss'] is not None:
                    profit_loss_str = f"{chart_values['profit_loss']:+.2f}%"
                    symbol_message += f"\nP/L: {profit_loss_str}"
                
                # Add current price
                symbol_message += f"\nCurrent Price: {chart_values['current_price']:.2f}"
                
                # Add take profit and stop loss
                if chart_values['take_profit']:
                    symbol_message += f"\nTake Profit: {chart_values['take_profit']:.2f}"
                if chart_values['stop_loss']:
                    symbol_message += f"\nStop Loss: {chart_values['stop_loss']:.2f}"
                
                symbol_message += "\n"
                
                # Send the message
                send_telegram_message(symbol_message)
                
                # Log the values for verification
                logging.info(f"Caption values for {symbol}: Entry Date: {chart_values['entry_date']}, "
                           f"Holding Days: {chart_values['holding_days']}, "
                           f"P/L: {chart_values['profit_loss']:.2f}%, "
                           f"Stop Loss: {chart_values['stop_loss']:.2f}, "
                           f"Take Profit: {chart_values['take_profit']:.2f}")
        
    except Exception as e:
        logging.error(f"Error sending signals and charts summary: {e}")

def process_symbol(symbol, database_path):
    """Process a single symbol for chart generation"""
    try:
        engine = create_engine(f'sqlite:///{database_path}')
        with engine.connect() as connection:
            table_name = f"PSX_{symbol}_stock_data"
            available_columns = fetch_column_names(engine, table_name)
            logging.info(f"Columns in table {table_name}: {available_columns}")
            print(f"Columns in table {table_name}: {available_columns}")
            query = f"SELECT Date, Close, RSI_weekly_Avg, RSI_monthly, RSI_3months_Avg, AO_weekly_AVG, MA_30, Volume FROM {table_name} ORDER BY Date"
            df = pd.read_sql(query, connection)
            if df.empty:
                logging.warning(f"No data found for {symbol}")
                return
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            # Calculate a simple accumulation metric based on volume
            df['Accumulation'] = df['Volume'].rolling(window=5).mean()
            latest_accumulation = df['Accumulation'].iloc[-1] if not df['Accumulation'].isna().all() else 0
            logging.info(f"Accumulation for {symbol} (5-day avg volume): {latest_accumulation}")
            print(f"Accumulation for {symbol} (5-day avg volume): {latest_accumulation}")
            output_folder = config['output']['charts_folder']
            success = draw_indicator_trend_lines_with_signals(df, symbol, output_folder)
            return symbol, success is not None
    except Exception as e:
        logging.error(f"Error processing symbol {symbol}: {e}")
        return symbol, False

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

# Set global Matplotlib style and parameters for consistent aesthetics
plt.style.use('ggplot')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.edgecolor': '#CCCCCC',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333'
})

# Color dictionaries for consistent use
STATUS_COLORS = {'BUY/HOLD': '#2ecc71', 'SELL': '#e74c3c', 'OPPORTUNITY': '#3498db'}
PHASE_COLORS = {'ACCUMULATION': '#2ecc71', 'DISTRIBUTION': '#e74c3c', 'NEUTRAL': '#95a5a6'}
RISK_COLORS = {'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c'}

def plot_pie_chart(ax, data, title, colors, explode=None):
    """Reusable pie chart plotting function with improved aesthetics."""
    if explode is None:
        explode = [0.05] * len(data)
    wedges, texts, autotexts = ax.pie(
        data.values, labels=data.index, autopct='%.2f%%',
        colors=colors, explode=explode, shadow=True,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)

def plot_bar_chart(ax, metrics, values, colors, title, ylabel):
    """Reusable bar chart plotting function with value annotations."""
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', fontsize=10)
    return bars

def create_market_overview_dashboard(df):
    """Create a market overview dashboard with improved aesthetics and code structure."""
    if df.empty:
        logging.error("No data available for market overview dashboard")
        return False
    fig, axs = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('PSX Market Overview Dashboard', fontsize=24, y=0.98, fontweight='bold')
    plt.subplots_adjust(hspace=0.6, wspace=0.4)

    # 1. Status Distribution Pie Chart
    status_counts = df['Status'].value_counts()
    status_colors = [STATUS_COLORS.get(s, '#95a5a6') for s in status_counts.index]
    plot_pie_chart(axs[0, 0], status_counts, 'Stock Signal Distribution', status_colors)

    # 2. Market Phase Distribution Pie Chart
    phase_counts = df['Market_Phase'].value_counts()
    phase_colors_list = [PHASE_COLORS.get(p, '#3498db') for p in phase_counts.index]
    plot_pie_chart(axs[0, 1], phase_counts, 'Market Phase Distribution', phase_colors_list)

    # 3. Market Breadth Indicator
    above_ma = df['Above_MA30'].sum() / len(df) * 100
    acc_stocks = len(df[df['Market_Phase'] == 'ACCUMULATION']) / len(df) * 100
    high_rsi = len(df[df['RSI'] > 50]) / len(df) * 100
    pos_ao = len(df[df['AO'] > 0]) / len(df) * 100
    metrics = ['Above MA30', 'Accumulation', 'RSI > 50', 'AO > 0']
    values = [above_ma, acc_stocks, high_rsi, pos_ao]
    bar_colors = ['#1f77b4', '#2ecc71', '#9467bd', '#ff7f0e']
    plot_bar_chart(axs[0, 2], metrics, values, bar_colors, 'Market Breadth Indicators', 'Percentage of Stocks')
    axs[0, 2].axhline(y=50, color='#e74c3c', linestyle='--', alpha=0.5)

    # 4. Recommended Portfolio Allocation (if available)
    if 'recommended_equity' in df.columns:
        equity = df['recommended_equity'].iloc[0]
        alloc_labels = ['Equity', 'Cash']
        alloc_sizes = [equity, 100 - equity]
        alloc_colors = ['#2ecc71', '#e74c3c']
        plot_pie_chart(axs[1, 0], pd.Series(alloc_sizes, index=alloc_labels), 'Recommended Portfolio Allocation', alloc_colors, explode=[0.1, 0])

    # 5. Market Sentiment (if available)
    if 'market_sentiment' in df.columns:
        sentiment = df['market_sentiment'].iloc[0]
        axs[1, 1].bar(['Market Sentiment'], [sentiment], color='#3498db')
        axs[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[1, 1].set_title('Market Sentiment', fontsize=14, fontweight='bold')
        axs[1, 1].set_ylim(-1, 1)
        axs[1, 1].set_ylabel('Sentiment Score')

    # 6. Risk Level Distribution (if available)
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        risk_labels = risk_counts.index
        risk_sizes = risk_counts.values
        risk_colors = [RISK_COLORS.get(r, '#95a5a6') for r in risk_labels]
        explode = [0.1 if i == 0 else 0 for i in range(len(risk_labels))]
        plot_pie_chart(axs[1, 2], pd.Series(risk_sizes, index=risk_labels), 'Risk Level Distribution', risk_colors, explode=explode)

    # 7. Average Holding Days
    if 'Holding_Days' in df.columns:
        avg_holding = df['Holding_Days'].dropna().mean()
        axs[2, 0].bar(['Avg Holding Days'], [avg_holding], color='#3498db')
        axs[2, 0].set_title('Average Holding Days', fontsize=14, fontweight='bold')
        axs[2, 0].set_ylabel('Days')
        for bar in axs[2, 0].patches:
            axs[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom')
    # 8. Average Profit/Loss
    if 'Profit_Loss' in df.columns:
        avg_pl = df['Profit_Loss'].dropna().mean()
        axs[2, 1].bar(['Avg P/L %'], [avg_pl], color='#2ecc71' if avg_pl >= 0 else '#e74c3c')
        axs[2, 1].set_title('Average Profit/Loss (%)', fontsize=14, fontweight='bold')
        axs[2, 1].set_ylabel('P/L %')
        for bar in axs[2, 1].patches:
            axs[2, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}%', ha='center', va='bottom')
    # 9. Sector/Market Cap Breakdown (if available)
    if 'Sector' in df.columns:
        sector_counts = df['Sector'].value_counts()
        sector_colors = plt.cm.tab20.colors[:len(sector_counts)]
        plot_pie_chart(axs[2, 2], sector_counts, 'Sector Distribution', sector_colors)
    elif 'Market_Cap' in df.columns:
        cap_bins = pd.qcut(df['Market_Cap'], q=3, labels=['Small Cap', 'Mid Cap', 'Large Cap'])
        cap_counts = cap_bins.value_counts()
        cap_colors = ['#a3e1d4', '#f7b6d2', '#c7c7c7']
        plot_pie_chart(axs[2, 2], cap_counts, 'Market Cap Distribution', cap_colors)

    # Add footer with summary
    fig.text(0.5, 0.02, 
                f'Total Stocks: {len(df)} | Strong Accumulation: {len(df[df["Market_Phase"] == "ACCUMULATION"])} | '
                f'Weak Accumulation: {len(df[df["Market_Phase"] == "WEAK_ACCUMULATION"])} | '
                f'Distribution: {len(df[df["Market_Phase"] == "DISTRIBUTION"])}',
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e0e0e0'))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(config['output']['dashboards_folder'], f'market_overview_{datetime.now().strftime("%Y%m%d")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    message = f"📊 PSX Market Overview - {datetime.now().strftime('%Y-%m-%d')}\n\n"
    message += f"Total Stocks: {len(df)}\n"
    message += f"Strong Accumulation: {len(df[df['Market_Phase'] == 'ACCUMULATION'])}\n"
    message += f"Weak Accumulation: {len(df[df['Market_Phase'] == 'WEAK_ACCUMULATION'])}\n"
    message += f"Distribution: {len(df[df['Market_Phase'] == 'DISTRIBUTION'])}"
    send_telegram_message_with_image(path, message)
    return True

def create_recommendation_dashboard(df):
    """Create a dashboard for portfolio recommendations with improved aesthetics and code structure."""
    if df.empty:
        logging.error("No data available for recommendation dashboard")
        return False
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('PSX Portfolio Recommendations', fontsize=24, y=0.98, fontweight='bold')
    plt.subplots_adjust(hspace=0.6, wspace=0.4)

    # 1. Recommended Allocation Pie Chart
    alloc_labels = ['Equity', 'Cash']
    alloc_sizes = [df['recommended_equity'].iloc[0], 100 - df['recommended_equity'].iloc[0]]
    alloc_colors = ['#2ecc71', '#e74c3c']
    plot_pie_chart(axs[0, 0], pd.Series(alloc_sizes, index=alloc_labels), 'Recommended Portfolio Allocation', alloc_colors, explode=[0.1, 0])

    # 2. Market Sentiment Gauge
    sentiment = df['market_sentiment'].iloc[0]
    axs[0, 1].bar(['Market Sentiment'], [sentiment], color='#3498db')
    axs[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axs[0, 1].set_title('Market Sentiment', fontsize=16, pad=20)
    axs[0, 1].set_ylim(-1, 1)
    axs[0, 1].set_ylabel('Sentiment Score')

    # 3. Top Picks Table
    top_picks = df[df['is_top_pick'] == True].sort_values('Phase_Probability', ascending=False)
    axs[1, 0].axis('off')
    if not top_picks.empty:
        data = [[row['Symbol'], f"{row['Phase_Probability']:.1f}%", row['risk_level']] 
                for _, row in top_picks.iterrows()]
        table = axs[1, 0].table(cellText=data,
                         colLabels=['Symbol', 'Probability', 'Risk'],
                         loc='center',
                         cellLoc='center',
                         colWidths=[0.3, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
    axs[1, 0].set_title('Top Stock Picks', fontsize=16, pad=20)

    # 4. Stock Portfolio Allocation with Signals and Accumulation Status
    active_signals = df[df['Status'].isin(['BUY/HOLD', 'SELL'])].sort_values('Phase_Probability', ascending=False)
    axs[1, 1].axis('off')
    if not active_signals.empty:
        alloc_data = []
        for _, row in active_signals.iterrows():
            symbol = row['Symbol']
            status = row['Status']
            accum = row.get('Accumulation', 0)  # Assuming accumulation is calculated or available in df
            alloc_data.append([symbol, status, f"{accum:.0f}"])
        alloc_table = axs[1, 1].table(cellText=alloc_data,
                               colLabels=['Symbol', 'Signal', 'Accumulation'],
                               loc='center',
                               cellLoc='center',
                               colWidths=[0.3, 0.3, 0.4])
        alloc_table.auto_set_font_size(False)
        alloc_table.set_fontsize(12)
        alloc_table.scale(1, 1.5)
    axs[1, 1].set_title('Portfolio Allocation & Signals', fontsize=16, pad=20)

    # Adjust layout to accommodate new table
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    try:
        filename = os.path.join(config['output']['dashboards_folder'], 
                              f'portfolio_recommendations_{datetime.now().strftime("%Y%m%d")}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Recommendation dashboard saved to {filename}")
        print(f"Recommendation dashboard saved to {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving recommendation dashboard: {e}")
        print(f"Error saving recommendation dashboard: {str(e)}")
        plt.close()
        return False

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
    """Generate portfolio recommendations based on market analysis"""
    try:
        if df.empty:
            logging.error("No data available for portfolio recommendations")
            return df
        
        # Calculate market conditions
        total_stocks = len(df)
        buy_hold_stocks = len(df[df['Status'] == 'BUY/HOLD'])
        sell_stocks = len(df[df['Status'] == 'SELL'])
        opportunity_stocks = len(df[df['Status'] == 'OPPORTUNITY'])
        
        # Calculate market sentiment
        market_sentiment = (buy_hold_stocks - sell_stocks) / total_stocks if total_stocks > 0 else 0
        
        # Calculate recommended equity allocation
        if market_sentiment > 0.3:  # Bullish
            recommended_equity = 80
        elif market_sentiment > 0:  # Slightly bullish
            recommended_equity = 60
        elif market_sentiment > -0.3:  # Neutral
            recommended_equity = 40
        else:  # Bearish
            recommended_equity = 20
        
        # Add recommendations to DataFrame
        df['recommended_equity'] = recommended_equity
        df['market_sentiment'] = market_sentiment
        
        # Add top picks
        top_picks = df[df['Status'] == 'BUY/HOLD'].sort_values('Phase_Probability', ascending=False).head(5)
        df['is_top_pick'] = df['Symbol'].isin(top_picks['Symbol'])
        
        # Add risk assessment
        df['risk_level'] = df.apply(lambda row: 'High' if row['RSI'] > 70 or row['RSI'] < 30 else 'Medium' if row['RSI'] > 60 or row['RSI'] < 40 else 'Low', axis=1)
        
        return df
    except Exception as e:
        logging.error(f"Error generating portfolio recommendations: {e}")
        print(f"Error generating portfolio recommendations: {str(e)}")
        return df

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

def create_unified_market_dashboard(df):
    """Create a unified dashboard combining market overview and portfolio recommendations."""
    if df.empty:
        logging.error("No data available for unified market dashboard")
        return False
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('PSX Unified Market & Portfolio Dashboard', fontsize=24, y=0.98, fontweight='bold')
    plt.subplots_adjust(hspace=0.6, wspace=0.4)

    # Define subplot grid: 4 rows, 3 columns (12 total subplots)
    # Market Overview Section
    # 1. Status Distribution Pie Chart
    ax1 = plt.subplot(4, 3, 1)
    status_counts = df['Status'].value_counts()
    status_colors = [STATUS_COLORS.get(s, '#95a5a6') for s in status_counts.index]
    plot_pie_chart(ax1, status_counts, 'Stock Signal Distribution', status_colors)

    # 2. Market Phase Distribution Pie Chart
    ax2 = plt.subplot(4, 3, 2)
    phase_counts = df['Market_Phase'].value_counts()
    phase_colors_list = [PHASE_COLORS.get(p, '#3498db') for p in phase_counts.index]
    plot_pie_chart(ax2, phase_counts, 'Market Phase Distribution', phase_colors_list)

    # 3. Market Breadth Indicator
    ax3 = plt.subplot(4, 3, 3)
    above_ma = df['Above_MA30'].sum() / len(df) * 100
    acc_stocks = len(df[df['Market_Phase'] == 'ACCUMULATION']) / len(df) * 100
    high_rsi = len(df[df['RSI'] > 50]) / len(df) * 100
    pos_ao = len(df[df['AO'] > 0]) / len(df) * 100
    metrics = ['Above MA30', 'Accumulation', 'RSI > 50', 'AO > 0']
    values = [above_ma, acc_stocks, high_rsi, pos_ao]
    bar_colors = ['#1f77b4', '#2ecc71', '#9467bd', '#ff7f0e']
    plot_bar_chart(ax3, metrics, values, bar_colors, 'Market Breadth Indicators', 'Percentage of Stocks')
    ax3.axhline(y=50, color='#e74c3c', linestyle='--', alpha=0.5)

    # 4. Average Holding Days
    ax4 = plt.subplot(4, 3, 4)
    if 'Holding_Days' in df.columns:
        avg_holding = df['Holding_Days'].dropna().mean()
        ax4.bar(['Avg Holding Days'], [avg_holding], color='#3498db')
        ax4.set_title('Average Holding Days', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Days')
        for bar in ax4.patches:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom')

    # 5. Average Profit/Loss
    ax5 = plt.subplot(4, 3, 5)
    if 'Profit_Loss' in df.columns:
        avg_pl = df['Profit_Loss'].dropna().mean()
        ax5.bar(['Avg P/L %'], [avg_pl], color='#2ecc71' if avg_pl >= 0 else '#e74c3c')
        ax5.set_title('Average Profit/Loss (%)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('P/L %')
        for bar in ax5.patches:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}%', ha='center', va='bottom')

    # 6. Sector/Market Cap Breakdown (if available)
    ax6 = plt.subplot(4, 3, 6)
    if 'Sector' in df.columns:
        sector_counts = df['Sector'].value_counts()
        sector_colors = plt.cm.tab20.colors[:len(sector_counts)]
        plot_pie_chart(ax6, sector_counts, 'Sector Distribution', sector_colors)
    elif 'Market_Cap' in df.columns:
        cap_bins = pd.qcut(df['Market_Cap'], q=3, labels=['Small Cap', 'Mid Cap', 'Large Cap'])
        cap_counts = cap_bins.value_counts()
        cap_colors = ['#a3e1d4', '#f7b6d2', '#c7c7c7']
        plot_pie_chart(ax6, cap_counts, 'Market Cap Distribution', cap_colors)

    # Portfolio Recommendations Section
    # 7. Recommended Allocation Pie Chart
    ax7 = plt.subplot(4, 3, 7)
    if 'recommended_equity' in df.columns:
        equity = df['recommended_equity'].iloc[0]
        alloc_labels = ['Equity', 'Cash']
        alloc_sizes = [equity, 100 - equity]
        alloc_colors = ['#2ecc71', '#e74c3c']
        plot_pie_chart(ax7, pd.Series(alloc_sizes, index=alloc_labels), 'Portfolio Allocation', alloc_colors, explode=[0.1, 0])

    # 8. Market Sentiment Gauge
    ax8 = plt.subplot(4, 3, 8)
    if 'market_sentiment' in df.columns:
        sentiment = df['market_sentiment'].iloc[0]
        ax8.bar(['Market Sentiment'], [sentiment], color='#3498db')
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax8.set_title('Market Sentiment', fontsize=14, fontweight='bold')
        ax8.set_ylim(-1, 1)
        ax8.set_ylabel('Sentiment Score')

    # 9. Risk Level Distribution
    ax9 = plt.subplot(4, 3, 9)
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        risk_labels = risk_counts.index
        risk_sizes = risk_counts.values
        risk_colors = [RISK_COLORS.get(r, '#95a5a6') for r in risk_labels]
        explode = [0.1 if i == 0 else 0 for i in range(len(risk_labels))]
        plot_pie_chart(ax9, pd.Series(risk_sizes, index=risk_labels), 'Risk Level Distribution', risk_colors, explode=explode)

    # 10. Top Picks Table
    ax10 = plt.subplot(4, 3, (10, 12))  # Spans last row, all columns
    ax10.axis('off')
    if 'is_top_pick' in df.columns and not df[df['is_top_pick'] == True].empty:
        top_picks = df[df['is_top_pick'] == True].sort_values('Phase_Probability', ascending=False)
        data = [[row['Symbol'], f"{row['Phase_Probability']:.1f}%", row['risk_level']] 
                for _, row in top_picks.iterrows()]
        table = ax10.table(cellText=data,
                         colLabels=['Symbol', 'Probability', 'Risk'],
                         loc='center',
                         cellLoc='center',
                         colWidths=[0.3, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
    ax10.set_title('Top Stock Picks', fontsize=14, fontweight='bold', pad=20)

    # Add footer with summary
    fig.text(0.5, 0.02, 
             f'Total Stocks: {len(df)} | Strong Accumulation: {len(df[df["Market_Phase"] == "ACCUMULATION"])} | '
             f'Weak Accumulation: {len(df[df["Market_Phase"] == "WEAK_ACCUMULATION"])} | '
             f'Distribution: {len(df[df["Market_Phase"] == "DISTRIBUTION"])}',
             ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e0e0e0'))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(config['output']['dashboards_folder'], f'unified_market_dashboard_{datetime.now().strftime("%Y%m%d")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    message = f"📊 PSX Unified Market & Portfolio Dashboard - {datetime.now().strftime('%Y-%m-%d')}\n\n"
    message += f"Total Stocks: {len(df)}\n"
    message += f"Strong Accumulation: {len(df[df['Market_Phase'] == 'ACCUMULATION'])}\n"
    message += f"Weak Accumulation: {len(df[df['Market_Phase'] == 'WEAK_ACCUMULATION'])}\n"
    message += f"Distribution: {len(df[df['Market_Phase'] == 'DISTRIBUTION'])}"
    if 'recommended_equity' in df.columns:
        message += f"\nRecommended Equity Allocation: {df['recommended_equity'].iloc[0]}%\n"
    if 'market_sentiment' in df.columns:
        sentiment = df['market_sentiment'].iloc[0]
        message += f"Market Sentiment: {'Bullish' if sentiment > 0.3 else 'Slightly Bullish' if sentiment > 0 else 'Bearish' if sentiment < -0.3 else 'Neutral'}\n"
    send_telegram_message_with_image(path, message)
    return True

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
            # Generate portfolio recommendations before creating the unified dashboard
            dashboard_df = generate_portfolio_recommendations(dashboard_df)
            # Create the unified dashboard instead of separate ones
            create_unified_market_dashboard(dashboard_df)
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
    try:
        if symbol not in available_symbols:
            logging.warning(f"Symbol {symbol} not found in available symbols")
            return False, None
            
        with sqlite3.connect(database_path) as connection:
            table_name = f"KMI100_{symbol}"  # Changed from KMI100 to KMI100
            query = f"SELECT Date, Open, Close, RSI_monthly, RSI_3months_Avg, RSI_weekly_Avg, AO_weekly_AVG, MA_30 FROM {table_name} ORDER BY Date DESC LIMIT 180"
            df = pd.read_sql(query, connection)
            if df.empty:
                logging.warning(f"No data found for {symbol}")
                return False, None
                
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Get entry date from buy signals
            buy_signals, sell_signals = get_buy_sell_signals(symbol)
            entry_date = None
            if buy_signals:
                latest_buy = max(buy_signals, key=lambda x: x[0])
                entry_date = latest_buy[0]
            
            output_folder = config['output']['charts_folder']
            success, chart_values = draw_indicator_trend_lines_with_signals(df, symbol, output_folder, entry_date)
            
            # Log the values for verification
            if success and chart_values:
                logging.info(f"Generated chart values for {symbol}: {chart_values}")
            
            return success, chart_values
            
    except Exception as e:
        logging.error(f"Error generating chart for {symbol}: {e}")
        print(f"Error generating chart for {symbol}: {str(e)}")
        return False, None

def calculate_atr_stop_loss(current_price, atr, multiplier=2):
    """Calculate ATR-based stop loss"""
    return current_price - (atr * multiplier)

def calculate_support_stop_loss(current_price, support_level, buffer=0.02):
    """Calculate support-based stop loss"""
    return support_level * (1 - buffer)

def calculate_volatility_stop_loss(current_price, volatility_score, risk_percentage=0.02):
    """Calculate volatility-adjusted stop loss"""
    vol_factor = 1 + (abs(volatility_score) / 100)
    return current_price * (1 - (risk_percentage * vol_factor))

def calculate_trailing_stop(high_price, atr, multiplier=2):
    """Calculate trailing stop loss"""
    return high_price - (atr * multiplier)

def calculate_enhanced_stop_loss(df, current_price, analysis):
    """Calculate enhanced stop loss using multiple methods"""
    stop_losses = []
    
    # Validate current price
    if current_price <= 0:
        logging.error(f"Invalid current price: {current_price}")
        return None
    
    # Calculate minimum and maximum allowed stop loss percentages
    min_stop_percent = 0.05  # 5% below current price
    max_stop_percent = 0.20  # 20% below current price
    
    # 1. ATR-based stop loss
    if 'ATR' in df.columns:
        atr = df['ATR'].iloc[-1]
        if atr > 0:
            atr_stop = calculate_atr_stop_loss(current_price, atr)
            if atr_stop > current_price * (1 - max_stop_percent) and atr_stop < current_price * (1 - min_stop_percent):
                stop_losses.append(atr_stop)
    
    # 2. Support level stop loss
    if 'support_level' in analysis and analysis['support_level'] > 0:
        support_stop = calculate_support_stop_loss(current_price, analysis['support_level'])
        if support_stop > current_price * (1 - max_stop_percent) and support_stop < current_price * (1 - min_stop_percent):
            stop_losses.append(support_stop)
    
    # 3. Volatility-adjusted stop loss
    if 'volatility_score' in analysis:
        vol_stop = calculate_volatility_stop_loss(current_price, analysis['volatility_score'])
        if vol_stop > current_price * (1 - max_stop_percent) and vol_stop < current_price * (1 - min_stop_percent):
            stop_losses.append(vol_stop)
    
    # 4. BB-based stop loss
    if 'bb_lower' in analysis and analysis['bb_lower'] > 0:
        if analysis['bb_lower'] > current_price * (1 - max_stop_percent) and analysis['bb_lower'] < current_price * (1 - min_stop_percent):
            stop_losses.append(analysis['bb_lower'])
    
    # 5. SMA20-based stop loss
    if 'MA_30' in df.columns:
        sma = df['MA_30'].iloc[-1]
        if sma > 0:
            sma_stop = sma * 0.95
            if sma_stop > current_price * (1 - max_stop_percent) and sma_stop < current_price * (1 - min_stop_percent):
                stop_losses.append(sma_stop)
    
    # Validate and select the highest valid stop loss
    valid_stops = [s for s in stop_losses if s > current_price * (1 - max_stop_percent) and s < current_price * (1 - min_stop_percent)]
    
    if valid_stops:
        selected_stop = max(valid_stops)
        logging.info(f"Selected stop loss: {selected_stop:.2f} for price: {current_price:.2f}")
        return selected_stop
    else:
        # Fallback to percentage-based stop loss
        fallback_stop = current_price * (1 - min_stop_percent)  # 5% below current price
        logging.warning(f"No valid stop losses found, using fallback: {fallback_stop:.2f}")
        return fallback_stop

def calculate_take_profit(current_price, stop_loss, risk_reward_ratio=2):
    """Calculate take profit level based on stop loss and risk-reward ratio"""
    if stop_loss is None or current_price <= 0:
        return None
    
    risk = current_price - stop_loss
    if risk <= 0:
        return None
    
    # Ensure minimum risk-reward ratio
    min_risk_reward = 1.5
    if risk_reward_ratio < min_risk_reward:
        risk_reward_ratio = min_risk_reward
    
    take_profit = current_price + (risk * risk_reward_ratio)
    
    # Log the calculation
    logging.info(f"Take profit calculation: Current Price: {current_price:.2f}, Stop Loss: {stop_loss:.2f}, Risk: {risk:.2f}, Take Profit: {take_profit:.2f}")
    
    return take_profit

def calculate_profit_loss(current_price, entry_price):
    """Calculate profit/loss percentage"""
    if entry_price and entry_price > 0:
        return ((current_price - entry_price) / entry_price) * 100
    return None

def calculate_holding_days(entry_date):
    """Calculate holding days from entry date"""
    if entry_date:
        entry_date = pd.to_datetime(entry_date)
        current_date = pd.to_datetime('today')
        return (current_date - entry_date).days
    return None

def get_kmi100_symbols():
    """Get KMI100 symbols from Excel file"""
    try:
        excel_path = os.path.join(config['data']['input_folder'], 'KMI100.xlsx')
        if not os.path.exists(excel_path):
            logging.error(f"KMI100 Excel file not found at {excel_path}")
            return []
            
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Sort by rank and get top 100
        df = df.sort_values('rank')
        top_100_symbols = df.head(100)['symbol'].tolist()
        
        logging.info(f"Found {len(top_100_symbols)} KMI100 symbols")
        print(f"Found {len(top_100_symbols)} KMI100 symbols")
        
        return top_100_symbols
    except Exception as e:
        logging.error(f"Error getting KMI100 symbols: {e}")
        print(f"Error getting KMI100 symbols: {e}")
        return []

def generate_chart_for_symbol(symbol, database_path, available_symbols):
    try:
        if symbol not in available_symbols:
            logging.warning(f"Symbol {symbol} not found in available symbols")
            return False, None
            
        with sqlite3.connect(database_path) as connection:
            table_name = f"KMI100_{symbol}"  # Changed from KMI100 to KMI100
            query = f"SELECT Date, Open, Close, RSI_monthly, RSI_3months_Avg, RSI_weekly_Avg, AO_weekly_AVG, MA_30 FROM {table_name} ORDER BY Date DESC LIMIT 180"
            df = pd.read_sql(query, connection)
            if df.empty:
                logging.warning(f"No data found for {symbol}")
                return False, None
                
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Get entry date from buy signals
            buy_signals, sell_signals = get_buy_sell_signals(symbol)
            entry_date = None
            if buy_signals:
                latest_buy = max(buy_signals, key=lambda x: x[0])
                entry_date = latest_buy[0]
            
            output_folder = config['output']['charts_folder']
            success, chart_values = draw_indicator_trend_lines_with_signals(df, symbol, output_folder, entry_date)
            
            # Log the values for verification
            if success and chart_values:
                logging.info(f"Generated chart values for {symbol}: {chart_values}")
            
            return success, chart_values
            
    except Exception as e:
        logging.error(f"Error generating chart for {symbol}: {e}")
        print(f"Error generating chart for {symbol}: {str(e)}")
        return False, None
def calculate_timeframe_crosses(df, print_crosses=False):
    """
    Calculate weekly, monthly, and 3-monthly open/close prices.
    Show the dates when:
      - weekly close crosses monthly close
      - monthly close crosses 3-monthly close
    If print_crosses is True, print the cross info to the console.
    """
    import pandas as pd
    import numpy as np
    
    # Ensure Date is datetime and set as index
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')
    
    # Resample to weekly, monthly, 3-monthly
    weekly = df.resample('W-FRI').agg({'Open': 'first', 'Close': 'last'})
    monthly = df.resample('M').agg({'Open': 'first', 'Close': 'last'})
    three_monthly = df.resample('Q').agg({'Open': 'first', 'Close': 'last'})
    
    # Align all to the same index for comparison (forward fill for missing)
    combined = pd.DataFrame(index=df.index)
    combined['Weekly_Close'] = weekly['Close'].reindex(df.index, method='ffill')
    combined['Monthly_Close'] = monthly['Close'].reindex(df.index, method='ffill')
    combined['3M_Close'] = three_monthly['Close'].reindex(df.index, method='ffill')
    
    # Find where weekly close crosses monthly close
    cross_wk_mo = (np.sign(combined['Weekly_Close'] - combined['Monthly_Close']).diff() != 0)
    cross_wk_mo_dates = combined.index[cross_wk_mo & combined['Weekly_Close'].notna() & combined['Monthly_Close'].notna()]
    
    # Find where monthly close crosses 3-monthly close
    cross_mo_3mo = (np.sign(combined['Monthly_Close'] - combined['3M_Close']).diff() != 0)
    cross_mo_3mo_dates = combined.index[cross_mo_3mo & combined['Monthly_Close'].notna() & combined['3M_Close'].notna()]
    
    if print_crosses:
        print("\n--- Weekly Close crosses Monthly Close ---")
        for dt in cross_wk_mo_dates:
            print(f"{dt.date()}: Weekly Close = {combined.loc[dt, 'Weekly_Close']:.2f}, Monthly Close = {combined.loc[dt, 'Monthly_Close']:.2f}")
        print(f"Total crosses: {len(cross_wk_mo_dates)}")
        print("\n--- Monthly Close crosses 3-Monthly Close ---")
        for dt in cross_mo_3mo_dates:
            print(f"{dt.date()}: Monthly Close = {combined.loc[dt, 'Monthly_Close']:.2f}, 3M Close = {combined.loc[dt, '3M_Close']:.2f}")
        print(f"Total crosses: {len(cross_mo_3mo_dates)}")
    
    # Optionally return the calculated data
    return weekly, monthly, three_monthly, cross_wk_mo_dates, cross_mo_3mo_dates


def main():
    try:
        # Load configuration
        config = load_config()
        if not config:
            return False
            
        # Setup logging
        setup_logging(config)
        
        # Check database files
        if not check_database_files():
            return False
            
        # Get KMI100 symbols
        symbols = get_kmi100_symbols()
        if not symbols:
            logging.error("No KMI100 symbols found")
            return False
            
        # Get available symbols from database
        with sqlite3.connect(config['database']['main_db']) as connection:
            cursor = connection.cursor()
            available_symbols = get_available_symbols(cursor)
            
            # Filter symbols to only those available in database
            symbols = [s for s in symbols if s in available_symbols]
            
            if not symbols:
                logging.error("No matching symbols found in database")
                return False
                
            # Get buy and sell signals
            buy_df = get_latest_buy_stocks()
            sell_df = get_latest_sell_stocks()
            
            # Process symbols
            total_processed = 0
            for symbol in symbols:
                try:
                    table_name = f"PSX_{symbol}_stock_data"
                    df = pd.read_sql(f"SELECT Date, Open, Close FROM {table_name} ORDER BY Date", connection)
                    if not df.empty:
                        print(f"\n=== Timeframe Crosses for {symbol} ===")
                        calculate_timeframe_crosses(df, print_crosses=True)
                except Exception as e:
                    print(f"Error calculating crosses for {symbol}: {e}")
                success = process_symbol(symbol, config['database']['main_db'])
                if success:
                    total_processed += 1
                    
            # Send summary
            send_signals_and_charts_summary(buy_df, sell_df, symbols, total_processed)
            
        return True
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error in main: {e}")
        return False

