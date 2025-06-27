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

# ... (rest of the code from the original script, including all function definitions and logic, up to and including the if __name__ == "__main__": block) ... 