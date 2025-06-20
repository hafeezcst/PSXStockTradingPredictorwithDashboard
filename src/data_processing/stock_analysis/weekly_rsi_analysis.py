# This file was renamed from 04-List_Weekly_RSI_GT_40_BUY_SELL_KMI30_100_Weekly_v1.0_stable_Development.py for importability

import sqlite3
import pandas as pd
from tabulate import tabulate
import os
import logging
from datetime import datetime
import requests
import time
import sys
from typing import Dict, Optional

# Create test file in project root to verify script execution
test_file_path = os.path.join(os.getcwd(), 'script_execution_test.txt')
with open(test_file_path, 'w') as f:
    f.write(f"Script executed at {datetime.now().isoformat()}\n")
    f.write(f"Current working directory: {os.getcwd()}\n")
    f.write(f"Python version: {sys.version}\n")

# Set up logging with file output
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
os.system('cls' if os.name == 'nt' else 'clear')
logging.info(f"Logging to file: {log_file}")
# Telegram configuration
TELEGRAM_BOT_TOKEN = '6860197701:AAESTzERZLYbqyU6gFKfAwJQL8jJ_HNKLbM'
TELEGRAM_CHAT_ID = '-4152327824'

# Import refactored modules
from src.data_processing.stock_analysis import data_fetcher, analyzer

# Alias frequently used functions
get_freefloatratio = data_fetcher.get_freefloatratio
get_multibagger_symbols = data_fetcher.get_multibagger_symbols
fetch_table_names = data_fetcher.fetch_table_names
fetch_stock_data = data_fetcher.fetch_stock_data
get_ao_change_date = data_fetcher.get_ao_change_date
get_dividend_info = data_fetcher.get_dividend_info
identify_weekly_breakouts = analyzer.identify_weekly_breakouts
identify_monthly_breakouts = analyzer.identify_monthly_breakouts
process_stock_data = analyzer.process_stock_data

# === Begin copied functions from original analysis script ===

def update_psx_investing_db(data, table_name):
    """
    Update the buy, sell, or neutral stock tables in PSX_investing_Stocks_KMI100.db
    
    Args:
        data: DataFrame containing stock data
        table_name: Name of the table to update (buy_stocks, sell_stocks, etc.)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create database if it doesn't exist
        with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI100.db') as conn:
            # Add 'Update_Date' column to the data
            data['Update_Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check for and remove potential duplicates based on Stock and Date
            if 'Stock' in data.columns and 'Date' in data.columns:
                try:
                    existing = pd.read_sql(f"SELECT Stock, Date FROM {table_name}", conn)
                    if not existing.empty:
                        # Create a unique key for both dataframes
                        data['unique_key'] = data['Stock'] + data['Date']
                        existing['unique_key'] = existing['Stock'] + existing['Date']
                        
                        # Filter out records that already exist
                        data = data[~data['unique_key'].isin(existing['unique_key'].tolist())]
                        data = data.drop('unique_key', axis=1)
                        
                        if data.empty:
                            logging.info(f"No new records to add to {table_name}")
                            return True
                except sqlite3.OperationalError as oe:
                    logging.warning(f"Table {table_name} does not exist yet: {oe}")
                except Exception as e:
                    logging.error(f"Error checking duplicates in {table_name}: {e}")

            # Get existing table schema if it exists and align columns
            try:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                existing_columns = {row[1] for row in cursor.fetchall()}
                if existing_columns:
                    # Only keep columns that exist in the database table
                    data_columns = set(data.columns)
                    columns_to_drop = data_columns - existing_columns
                    if columns_to_drop:
                        logging.info(f"Dropping columns not in database schema: {columns_to_drop}")
                        data = data.drop(columns=columns_to_drop, errors='ignore')
            except sqlite3.OperationalError as oe:
                logging.warning(f"Could not get schema for {table_name}, likely doesn't exist: {oe}")
            except Exception as e:
                logging.error(f"Unexpected error getting schema for {table_name}: {e}")
            
            # Explicitly drop known problematic columns that might not be in schema
            known_problematic_columns = ['Breakout']
            for col in known_problematic_columns:
                if col in data.columns:
                    logging.info(f"Explicitly dropping known problematic column: {col}")
                    data = data.drop(columns=[col], errors='ignore')

            # Append new data to the specific table
            data.to_sql(table_name, conn, if_exists='append', index=False)
            logging.info(f"Added {len(data)} new records to {table_name}")
            return True
    except sqlite3.OperationalError as oe:
        logging.error(f"Database operational error updating {table_name}: {oe}")
        return False
    except sqlite3.DatabaseError as de:
        logging.error(f"Database error updating {table_name}: {de}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error updating {table_name}: {e}")
        return False

# (Repeat for all other functions: get_stock_data_with_rsi_above_40, get_KMI_symbols, get_kmi_tag, get_dividend_info, format_dividend_info, generate_buy_signal_description, format_buy_signals, generate_sell_signal_description, format_sell_signals, generate_neutral_signal_description, format_neutral_signals, send_telegram_message, format_breakout_message, handle_breakout_data, etc.)

# === End copied functions ===

# ... (rest of the code from 04-List_Weekly_RSI_GT_40_BUY_SELL_KMI30_100_Weekly_v1.0_stable_Development.py, lines 52-848) ... 