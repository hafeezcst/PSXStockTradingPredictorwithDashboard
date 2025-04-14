import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import sqlite3
import os
import sys
import logging
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from scripts.data_processing.telegram_message import send_telegram_message_with_image, send_telegram_message
from tabulate import tabulate
import numpy as np
from collections import Counter
import time
import argparse
from tqdm import tqdm
import concurrent.futures
import functools
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global connection pools for database access
psx_data_engine = create_engine('sqlite:///../../data/databases/production/psx_consolidated_data_indicators_PSX.db',
                               poolclass=QueuePool, pool_size=10, max_overflow=20)
kmi30_engine = create_engine('sqlite:///../../data/databases/production/PSX_investing_Stocks_KMI30.db',
                            poolclass=QueuePool, pool_size=10, max_overflow=20)

# Cache for frequently accessed data
_cache = {}

def cache_result(func):
    """Decorator to cache function results"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
        if key in _cache:
            return _cache[key]
        result = func(*args, **kwargs)
        _cache[key] = result
        return result
    return wrapper

def send_signals_and_charts_summary(buy_df, sell_df, available_symbols, total_processed):
    """Send a summary of signals and charts processed to Telegram"""
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
        message += "📱 Individual charts have been sent via Telegram\n\n"
        
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
        
        # Send the message
        send_telegram_message(message)
        return True
    except Exception as e:
        logging.error(f"Error sending signals and charts summary: {e}")
        return False

def get_latest_sell_stocks():
    """Get the latest sell stocks from the database"""
    try:
        with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI30.db') as conn:
            cursor = conn.cursor()
            
            # First check if the sell_stocks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sell_stocks'")
            if not cursor.fetchone():
                logging.warning("Table 'sell_stocks' does not exist in the database")
                return pd.DataFrame()
            
            cursor.execute("PRAGMA table_info(sell_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            logging.info(f"Actual columns in sell_stocks: {columns}")
            
            # Get a list of all unique stocks
            cursor.execute("SELECT DISTINCT Stock FROM sell_stocks WHERE Signal_Date IS NOT NULL")
            stocks = [row[0] for row in cursor.fetchall()]
            logging.info(f"Found {len(stocks)} unique stocks with sell signal dates")
            
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
        logging.error(f"Error getting latest sell stocks: {e}")
        import traceback
        logging.error(traceback.format_exc())
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
                logging.warning(f"Operation failed, retrying in {wait_time:.2f}s ({attempt+1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                raise

def check_database_files():
    """Check if required database files exist and are accessible"""
    required_dbs = {
        'data/databases/production/psx_consolidated_data_indicators_PSX.db': 'Main stock data database',
        'data/databases/production/PSX_investing_Stocks_KMI30.db': 'Signals database'
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
        print("\n⚠️ DATABASE ERROR ⚠️")
        print("The following required database files are missing or inaccessible:")
        for msg in missing_dbs:
            print(msg)
        print("\nTo use this script, please ensure:")
        print("1. You've downloaded the latest database files from the repository")
        print("2. The database files are in the same directory as this script")
        print("3. You have read/write permissions for these files")
        return False
        
    return True

def create_default_symbols_file():
    """Create a default KMI30 symbols file if it doesn't exist"""
    try:
        file_path = os.path.join(os.getcwd(), 'psxsymbols.xlsx')
        
        # Check if file already exists
        if os.path.exists(file_path):
            logging.info(f"Symbols file already exists at {file_path}")
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
            
        logging.info(f"Created default symbols file at {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating symbols file: {e}")
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
            logging.warning("No stock tables found in database")
        else:
            logging.info(f"Fetched {len(stock_tables)} stock tables")
            
        return stock_tables
    except Exception as e:
        logging.error(f"Error fetching table names: {e}")
        return []

def fetch_column_names(engine, table_name):
    """Fetch column names from the specified table"""
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
                logging.info(f"Skipping non-stock table: {table_name}")
                continue
                
            symbols.append(symbol)
            
        return symbols
    except Exception as e:
        logging.error(f"Error getting available symbols: {e}")
        print(f"\n⚠️ Database access error: {e}")
        print("Please check if your database file is valid and not corrupted.")
        return []

def get_latest_buy_stocks():
    """Get the latest buy stocks from the database"""
    try:
        with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI30.db') as conn:
            cursor = conn.cursor()
            
            cursor.execute("PRAGMA table_info(buy_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            logging.info(f"Actual columns in buy_stocks: {columns}")
            
            # Get a list of all unique stocks
            cursor.execute("SELECT DISTINCT Stock FROM buy_stocks WHERE Signal_Date IS NOT NULL")
            stocks = [row[0] for row in cursor.fetchall()]
            logging.info(f"Found {len(stocks)} unique stocks with signal dates")
            
            # For each stock, get the most recent signal
            results = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for stock in stocks:
                if 'update_date' in columns:
                    # If update_date exists, use it to find the most recent entry
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close, 
                              update_date, julianday('{current_date}') - julianday(Signal_Date) AS days_held
                        FROM buy_stocks 
                        WHERE Stock = ? AND Signal_Date IS NOT NULL
                        ORDER BY update_date DESC, Signal_Date DESC
                        LIMIT 1
                    """
                else:
                    # Otherwise just use Signal_Date
                    query = f"""
                        SELECT Stock, Date, Close, RSI_Weekly_Avg, AO_Weekly, Signal_Date, Signal_Close,
                              julianday('{current_date}') - julianday(Signal_Date) AS days_held
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
            
    except sqlite3.Error as e:
        logging.error(f"SQLite error getting latest buy stocks: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error getting latest buy stocks: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def get_latest_sell_stocks():
    """Get the latest sell stocks from the database"""
    try:
        with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI30.db') as conn:
            cursor = conn.cursor()
            
            # First check if the sell_stocks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sell_stocks'")
            if not cursor.fetchone():
                logging.warning("Table 'sell_stocks' does not exist in the database")
                return pd.DataFrame()
            
            cursor.execute("PRAGMA table_info(sell_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            logging.info(f"Actual columns in sell_stocks: {columns}")
            
            # Get a list of all unique stocks
            cursor.execute("SELECT DISTINCT Stock FROM sell_stocks WHERE Signal_Date IS NOT NULL")
            stocks = [row[0] for row in cursor.fetchall()]
            logging.info(f"Found {len(stocks)} unique stocks with sell signal dates")
            
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
        logging.error(f"Error getting latest sell stocks: {e}")
        import traceback
        logging.error(traceback.format_exc())
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
        logging.error(f"Error formatting {signal_type} signals for Telegram: {e}")
        return f"Error formatting {signal_type} signals for Telegram: {e}"
def get_buy_sell_signals(symbol):
    """Get buy and sell signals for a symbol from PSX_investing_Stocks_KMI30.db"""
    buy_signals = []
    sell_signals = []
    
    try:
        with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI30.db') as conn:
            # First check which tables exist
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            # Get buy signals if table exists
            if 'buy_stocks' in tables:
                query = f"SELECT Date, Signal_Date, Signal_Close FROM buy_stocks WHERE Stock = ? ORDER BY Date"
                buy_df = pd.read_sql_query(query, conn, params=(symbol,))
                if not buy_df.empty:
                    for _, row in buy_df.iterrows():
                        if pd.notna(row['Signal_Date']) and pd.notna(row['Signal_Close']):
                            buy_signals.append((pd.to_datetime(row['Signal_Date']), row['Signal_Close']))
            else:
                logging.info(f"Table 'buy_stocks' not found in database - no buy signals available for {symbol}")
            
            # Get sell signals if table exists
            if 'sell_stocks' in tables:
                query = f"SELECT Date, Signal_Date, Signal_Close FROM sell_stocks WHERE Stock = ? ORDER BY Date"
                sell_df = pd.read_sql_query(query, conn, params=(symbol,))
                if not sell_df.empty:
                    for _, row in sell_df.iterrows():
                        if pd.notna(row['Signal_Date']) and pd.notna(row['Signal_Close']):
                            sell_signals.append((pd.to_datetime(row['Signal_Date']), row['Signal_Close']))
            else:
                # Only log this as info since it's a normal condition in your case
                logging.info(f"Table 'sell_stocks' not found in database - no sell signals available for {symbol}")
                        
        logging.info(f"Found {len(buy_signals)} buy signals and {len(sell_signals)} sell signals for {symbol}")
        return buy_signals, sell_signals
    
    except sqlite3.Error as e:
        logging.error(f"SQLite error getting buy/sell signals for {symbol}: {e}")
        return [], []
    except Exception as e:
        logging.error(f"Unexpected error getting buy/sell signals for {symbol}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [], []

def remove_duplicate_buy_stocks():
    """Remove duplicate entries from the buy_stocks table, keeping only the most recent signal for each stock"""
    try:
        with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI30.db') as conn:
            cursor = conn.cursor()
            
            # First, check if the table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='buy_stocks'")
            if not cursor.fetchone():
                logging.warning("Table 'buy_stocks' does not exist in the database")
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
                logging.info("No duplicate stocks found in buy_stocks table")
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
                logging.info(f"Removed {removed} duplicate entries for {stock}")
            
            conn.commit()
            
            # Count total records after cleaning
            cursor.execute("SELECT COUNT(*) FROM buy_stocks")
            total_after = cursor.fetchone()[0]
            
            logging.info(f"Cleaning complete. Records before: {total_before}, after: {total_after}, removed: {total_removed}")
            
            return total_removed
            
    except Exception as e:
        logging.error(f"Error removing duplicate buy stocks: {e}")
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
        
        # Enhanced scoring with more nuanced thresholds
        rsi_score = 0
        
        # Strong oversold conditions
        if current_rsi < 30 and rsi_trend > 0.15:
            rsi_score = 3.0
        elif current_rsi < 35 and rsi_trend > 0.1:
            rsi_score = 2.5
        # Moderate oversold
        elif current_rsi < 40 and rsi_trend > 0.05:
            rsi_score = 2.0
        # Strong overbought conditions
        elif current_rsi > 70 and rsi_trend < -0.15:
            rsi_score = -3.0
        elif current_rsi > 65 and rsi_trend < -0.1:
            rsi_score = -2.5
        # Moderate overbought
        elif current_rsi > 60 and rsi_trend < -0.05:
            rsi_score = -2.0
        # Neutral zone with positive momentum
        elif current_rsi < 50 and rsi_trend > 0.03:
            rsi_score = 1.0
        # Neutral zone with negative momentum
        elif current_rsi > 50 and rsi_trend < -0.03:
            rsi_score = -1.0
        # Long-term positive trend
        elif current_rsi < 50 and longer_rsi_trend > 0.02:
            rsi_score = 0.7
        # Long-term negative trend
        elif current_rsi > 50 and longer_rsi_trend < -0.02:
            rsi_score = -0.7
            
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
        logging.error(f"Error analyzing RSI trend: {e}")
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
        logging.error(f"Error analyzing AO trend: {e}")
        return 0, {'error': str(e)}

def analyze_ao_trend(analysis_df):
    """Analyze Awesome Oscillator trends to detect momentum shifts"""
    try:
        # Get recent AO values (last 30 periods)
        recent_ao = analysis_df['AO_weekly_AVG'].tail(30).values
        
        # Current AO value and trend
        current_ao = recent_ao[-1] if len(recent_ao) > 0 else 0
        ao_trend = np.polyfit(range(len(recent_ao)), recent_ao, 1)[0] if len(recent_ao) > 1 else 0
        
        # Calculate AO momentum strength
        ao_momentum = np.mean(recent_ao[-5:]) - np.mean(recent_ao[-10:-5]) if len(recent_ao) >= 10 else 0
        
        # Check for recent crosses (last 15 periods)
        recent_crosses = recent_ao[-15:] if len(recent_ao) >= 15 else recent_ao
        ao_crosses_up = False
        ao_crosses_down = False
        consecutive_bars = 0
        
        for i in range(1, len(recent_crosses)):
            if recent_crosses[i-1] < 0 and recent_crosses[i] >= 0:
                ao_crosses_up = True
                consecutive_bars = 0
            if recent_crosses[i-1] > 0 and recent_crosses[i] <= 0:
                ao_crosses_down = True
                consecutive_bars = 0
            if (recent_crosses[i] > 0 and recent_crosses[i-1] > 0) or (recent_crosses[i] < 0 and recent_crosses[i-1] < 0):
                consecutive_bars += 1
        
        # Enhanced scoring with momentum consideration
        ao_score = 0
        
        # Strong bullish signals
        if ao_crosses_up and ao_momentum > 0.05:
            ao_score = 3.0
        elif ao_crosses_up:
            ao_score = 2.5
        # Strong bearish signals
        elif ao_crosses_down and ao_momentum < -0.05:
            ao_score = -3.0
        elif ao_crosses_down:
            ao_score = -2.5
        # Positive momentum above zero
        elif current_ao > 0 and ao_trend > 0.03 and consecutive_bars >= 3:
            ao_score = 2.0
        elif current_ao > 0 and ao_trend > 0.02:
            ao_score = 1.5
        # Negative momentum below zero
        elif current_ao < 0 and ao_trend < -0.03 and consecutive_bars >= 3:
            ao_score = -2.0
        elif current_ao < 0 and ao_trend < -0.02:
            ao_score = -1.5
        # Neutral positive
        elif current_ao > 0:
            ao_score = 0.5
        # Neutral negative
        elif current_ao < 0:
            ao_score = -0.5
            
        details = {
            'current_ao': round(current_ao, 2),
            'ao_trend': round(ao_trend, 4),
            'crosses_up': ao_crosses_up,
            'crosses_down': ao_crosses_down
        }
            
        return ao_score, details
        
    except Exception as e:
        logging.error(f"Error analyzing AO trend: {e}")
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
        logging.error(f"Error analyzing volume pattern: {e}")
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
        logging.error(f"Error analyzing price-MA relationship: {e}")
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
        logging.error(f"Error analyzing price-MA relationship: {e}")
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
        logging.error(f"Error analyzing price pattern: {e}")
        return 0, {'error': str(e)}

def calculate_final_phase_score(rsi_score, ao_score, volume_score, ma_score, pattern_score):
    """Calculate the final market phase score and determine accumulation/distribution"""
    try:
        # Calculate weighted total score with emphasis on RSI and AO
        weights = {
            'rsi': 0.35,    # Most important indicator
            'ao': 0.30,     # Second most important
            'volume': 0.15,
            'ma': 0.12,
            'pattern': 0.08
        }
        
        total_score = (
            rsi_score * weights['rsi'] +
            ao_score * weights['ao'] +
            volume_score * weights['volume'] +
            ma_score * weights['ma'] +
            pattern_score * weights['pattern']
        )
        
        # Dynamic threshold based on indicator agreement
        indicators = [rsi_score, ao_score, volume_score, ma_score, pattern_score]
        positive_indicators = sum(1 for score in indicators if score > 0)
        negative_indicators = sum(1 for score in indicators if score < 0)
        
        # Adjust threshold based on consensus
        if positive_indicators >= 4 or negative_indicators >= 4:
            neutral_threshold = 1.0  # Lower threshold for strong consensus
        else:
            neutral_threshold = 1.5  # Default threshold
            
        max_possible_score = 3.0  # Adjusted for weighted scores
        
        # Calculate probability with confidence boost for consensus
        if total_score > neutral_threshold:  # Accumulation
            confidence_boost = min(positive_indicators * 5, 15)  # Up to 15% boost
            probability = min(round(
                ((total_score - neutral_threshold) / (max_possible_score - neutral_threshold)) * 85 + confidence_boost,
                2), 100)
            phase = "ACCUMULATION"
        elif total_score < -neutral_threshold:  # Distribution
            confidence_boost = min(negative_indicators * 5, 15)  # Up to 15% boost
            probability = min(round(
                ((abs(total_score) - neutral_threshold) / (max_possible_score - neutral_threshold)) * 85 + confidence_boost,
                2), 100)
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
            'total_score': round(total_score, 2),
            'weights': weights,
            'indicator_agreement': f"{positive_indicators} positive, {negative_indicators} negative"
        }
        
    except Exception as e:
        logging.error(f"Error calculating final phase score: {e}")
        return "NEUTRAL", 50, {'error': str(e)}
            
        return phase, probability, {
            'rsi_score': rsi_score,
            'ao_score': ao_score,
            'volume_score': volume_score,
            'ma_score': ma_score,
            'pattern_score': pattern_score,
            'total_score': round(total_score, 2)
        }
        
    except Exception as e:
        logging.error(f"Error calculating final phase score: {e}")
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
def draw_indicator_trend_lines_with_signals(database_path, table_name):
    """Draw technical indicator trend lines with buy/sell signals"""
    try:
        # Extract the symbol name from the table name
        symbol_name = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()

        # Create a connection to the database
        engine = create_engine(f'sqlite:///{database_path}')
        connection = engine.connect()

        # Get available columns
        available_columns = fetch_column_names(engine, table_name)
        
        # Define the specific columns we're looking for
        monthly_column = 'RSI_monthly_Avg'
        threemonth_column = 'RSI_3months_Avg'
        
        # Base columns we always want
        base_columns = ["Date", "Close", "RSI_weekly_Avg", "AO_weekly_AVG", "MA_30", 
                         "RSI_weekly", "Volume", "pct_change"]
                         
        # Add our specific RSI columns if they exist
        columns_to_select = base_columns.copy()
        
        if monthly_column in available_columns:
            columns_to_select.append(monthly_column)
            logging.info(f"Found monthly RSI column: {monthly_column}")
            
        if threemonth_column in available_columns:
            columns_to_select.append(threemonth_column)
            logging.info(f"Found 3-month RSI column: {threemonth_column}")
        
        # Build the query
        query = f"SELECT {', '.join(columns_to_select)} FROM {table_name}"
        logging.info(f"Executing query: {query}")
        
        # Query the data from the database
        df = pd.read_sql(query, connection)
        connection.close()

        # Check if the dataframe is empty
        if df.empty:
            logging.warning(f"No data found in table: {table_name}")
            return False

        # Convert the date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter data to include only the last 10 years
        custom_years_ago = datetime.now() - timedelta(days=10*365)
        df = df[df['Date'] >= custom_years_ago]

        # Get AO change dates
        change_dates = get_ao_change_dates(df)

        # Get buy and sell signals
        buy_signals, sell_signals = get_buy_sell_signals(symbol_name)
        
        # Get holding days and profit/loss for this stock
        holding_days = None
        profit_loss_pct = None
        signal_price = None
        current_price = None
        
        try:
            with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI30.db') as conn:
                cursor = conn.cursor()
                current_date = datetime.now().strftime('%Y-%m-%d')
                
                # Check if %p/L column exists
                cursor.execute("PRAGMA table_info(buy_stocks)")
                columns = [info[1] for info in cursor.fetchall()]
                
                pl_column = None
                for col in columns:
                    if 'p/l' in col.lower() or 'profit' in col.lower() or 'loss' in col.lower() or 'gain' in col.lower():
                        pl_column = col
                        break
                
                # Get holding days and the most recent buy signal
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
                    logging.info(f"Stock {symbol_name} has been held for {holding_days} days, bought at {signal_price}")
                    
                    # If p/L column exists, get the value
                    if pl_column:
                        query = f"""
                            SELECT "{pl_column}" 
                            FROM buy_stocks 
                            WHERE Stock = ? AND Signal_Date = ?
                        """
                        cursor.execute(query, (symbol_name, signal_date))
                        pl_result = cursor.fetchone()
                        if pl_result and pl_result[0] is not None:
                            profit_loss_pct = float(pl_result[0])
                    
                    # If we couldn't get p/L from database, calculate it
                    if profit_loss_pct is None and signal_price:
                        # Get current price from the most recent data
                        query = f"""
                            SELECT Close 
                            FROM {table_name} 
                            ORDER BY Date DESC 
                            LIMIT 1
                        """
                        cursor.execute(query)
                        close_result = cursor.fetchone()
                        if close_result:
                            current_price = float(close_result[0])
                            profit_loss_pct = ((current_price - signal_price) / signal_price) * 100
                            logging.info(f"Calculated profit/loss: {profit_loss_pct:.2f}% (Current: {current_price}, Signal: {signal_price})")
        except Exception as e:
            logging.error(f"Error getting holding days and profit/loss for {symbol_name}: {e}")
            
        # Determine stock status (buy/sell/neutral)
        # Priority: most recent signal type or neutral if no signals
        stock_status = "OPPORTUNITY"  # Default status
        
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
        
        # Calculate market phase (accumulation/distribution)
        market_phase, phase_probability, phase_details = calculate_market_phase(df, symbol_name)
        
        # Plot the trend lines
        plt.figure(figsize=(14, 14))

        # Plot RSI_weekly_Avg and RSI_weekly overlapped
        plt.subplot(5, 1, 1)
        plt.plot(df['Date'], df['RSI_weekly_Avg'], label='RSI_weekly_Avg', color='blue')
        plt.plot(df['Date'], df['RSI_weekly'], label='RSI_weekly', color='purple')
        plt.axhline(y=40, color='green', linestyle='--', label='RSI 40')
        plt.axhline(y=60, color='red', linestyle='--', label='RSI 60')
        plt.title(f'{symbol_name} - RSI Weekly Average and RSI Weekly Trend Line')
        plt.xlabel('Date')
        plt.ylabel('RSI Values')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot AO_weekly_AVG with different colors for positive and negative values
        plt.subplot(5, 1, 2)
        
        # Dual color plot for AO_weekly_AVG
        positive_ao = df[df['AO_weekly_AVG'] >= 0]
        negative_ao = df[df['AO_weekly_AVG'] < 0]
        
        plt.plot(positive_ao['Date'], positive_ao['AO_weekly_AVG'], color='green', label='AO_weekly_AVG Positive')
        plt.plot(negative_ao['Date'], negative_ao['AO_weekly_AVG'], color='red', label='AO_weekly_AVG Negative')
        plt.axhline(y=0, color='gray', linestyle='--')
        
        plt.title(f'{symbol_name} - AO Weekly Average Trend Line')
        plt.xlabel('Date')
        plt.ylabel('AO_weekly_AVG')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot stars for AO change dates
        for date, close in change_dates['negative_to_positive']:
            plt.plot(date, 0, marker='*', color='green', markersize=10)
        for date, close in change_dates['positive_to_negative']:
            plt.plot(date, 0, marker='*', color='red', markersize=10)

        # Plot Close and MA_30 overlapped
        ax3 = plt.subplot(5, 1, 3)
        plt.plot(df['Date'], df['Close'], label='Close', color='orange')
        plt.plot(df['Date'], df['MA_30'], label='MA_30', color='red')
        plt.title(f'{symbol_name} - Close Price and MA30 Trend Line')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Add buy signals
        for date, price in buy_signals:
            plt.plot(date, price, marker='^', color='green', markersize=10)
            plt.annotate('Buy', (date, price), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=9, color='green')
        
        # Add sell signals
        for date, price in sell_signals:
            plt.plot(date, price, marker='v', color='red', markersize=10)
            plt.annotate('Sell', (date, price), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=9, color='red')
        
        # Format x-axis to show dates clearly
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        plt.legend()

        # Plot Volume
        plt.subplot(5, 1, 4)
        plt.bar(df['Date'], df['Volume'], color='blue', alpha=0.6)
        plt.title(f'{symbol_name} - Trading Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.grid(True, axis='y', alpha=0.3)

        # Plot RSI Monthly and RSI 3-Month
        plt.subplot(5, 1, 5)
        
        # Track if we have any monthly indicators to plot
        has_monthly_data = False
        
        # Plot monthly RSI if available
        if monthly_column in df.columns and df[monthly_column].notna().any():
            plt.plot(df['Date'], df[monthly_column], label='RSI Monthly', color='green')
            has_monthly_data = True
            logging.info(f"Successfully plotting RSI Monthly for {symbol_name}")
        
        # Plot 3-month RSI if available
        if threemonth_column in df.columns and df[threemonth_column].notna().any():
            plt.plot(df['Date'], df[threemonth_column], label='RSI 3-Month', color='blue')
            has_monthly_data = True
            logging.info(f"Successfully plotting RSI 3-Month for {symbol_name}")
        
        if not has_monthly_data:
            plt.text(0.5, 0.5, 'No monthly RSI data available', 
                     ha='center', va='center', transform=plt.gca().transAxes)
            logging.warning(f"No monthly RSI data available for {symbol_name}")
        
        plt.axhline(y=40, color='green', linestyle='--', label='RSI 40')
        plt.axhline(y=60, color='red', linestyle='--', label='RSI 60')
        plt.title(f'{symbol_name} - Monthly and 3-Month RSI')
        plt.xlabel('Date')
        plt.ylabel('RSI Values')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add a title with current date, signal status, and market phase
        current_date = datetime.now().strftime('%Y-%m-%d')
        title_text = f'{symbol_name} Technical Analysis - {stock_status}'
        if holding_days is not None and stock_status == "BUY/HOLD":
            title_text += f' - Held for {holding_days} days'
        if profit_loss_pct is not None:
            title_text += f' - P/L: {profit_loss_pct:.2f}%'
        title_text += f' - {market_phase} {phase_probability:.2f}% - Generated on {current_date}'

        # Add watermark
        watermark_color = {
            "BUY/HOLD": "green",
            "SELL": "red",
            "OPPORTUNITY": "blue"
        }.get(stock_status, "gray")
        
        # Create watermark text with holding days, profit/loss, and market phase
        watermark_text = stock_status
        
        # Add holding days for BUY/HOLD stocks
        if holding_days is not None and stock_status == "BUY/HOLD":
            watermark_text = f"{stock_status}\n{holding_days} DAYS"
            
        # Add profit/loss if available
        if profit_loss_pct is not None:
            profit_loss_sign = "+" if profit_loss_pct >= 0 else ""
            watermark_text += f"\n{profit_loss_sign}{profit_loss_pct:.2f}%"
            
        # Add market phase information
        phase_color = {
            "ACCUMULATION": "green",
            "DISTRIBUTION": "red",
            "NEUTRAL": "gray"
        }.get(market_phase, "gray")
        
        fig = plt.gcf()
        # Main watermark with status and days
        fig.text(0.5, 0.55, watermark_text, fontsize=80, color=watermark_color, 
                 ha='center', va='center', alpha=0.2, rotation=30)
                # Add market phase at the bottom of the watermark with more separation
        fig.text(0.5, 0.25, f"{market_phase} {phase_probability:.2f}%", 
                 fontsize=60, color=phase_color, 
                 ha='center', va='center', alpha=0.2, rotation=30)

        # Adjust layout and save the plot as an image file
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
        charts_folder = 'RSI_AO_CHARTS'
        os.makedirs(charts_folder, exist_ok=True)
        plot_filename = os.path.join(charts_folder, f'{symbol_name}_trend_lines_with_signals.png')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=120)

        # Send the plot to Telegram
        message = title_text
        send_telegram_message_with_image(plot_filename, message)

        # Close the figure to avoid memory issues
        plt.close()  
        
        return True
    except Exception as e:
        logging.error(f"Error drawing trend lines for table {table_name}: {e}")
        return False

def generate_stock_dashboard():
    """Generate a dashboard showing buy, sell and neutral stocks with key metrics"""
    try:
        # Get absolute path to database file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../../'))
        database_path = os.path.join(project_root, 'data/databases/production/psx_consolidated_data_indicators_PSX.db')
        
        # Verify database file exists
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Database file not found at: {database_path}")
            
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
        print("Analyzing all available stocks for dashboard...")
        for symbol in tqdm(available_symbols, desc="Processing stocks"):
            try:
                # Get stock data for this specific symbol only
                table_name = f"PSX_{symbol}_stock_data"
                
                # First, check which columns exist in this table
                available_columns = fetch_column_names(engine, table_name)
                if not available_columns:
                    logging.warning(f"No columns found for {table_name}")
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
                    logging.warning(f"Missing required columns in {table_name}")
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
                logging.error(f"Error processing {symbol} for dashboard: {e}")
                continue
        
        # Convert results to DataFrame
        dashboard_df = pd.DataFrame(all_results)
        
        # Create the dashboard
        create_dashboard_visualization(dashboard_df)
        
        # Close connection
        connection.close()
        
        return dashboard_df
        
    except Exception as e:
        logging.error(f"Error generating dashboard: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def create_dashboard_visualization(df):
    """Create visual dashboard using Matplotlib"""
    if df.empty:
        logging.error("No data available for dashboard visualization")
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
    dashboards_folder = 'PSX_DASHBOARDS'
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
    
    print(f"Dashboard saved to {dashboard_path} and sent to Telegram")
    return True

    # Add risk-adjusted performance matrix
    fig = plt.figure(figsize=(22, 18))  # Increase overall figure size
    plt.subplots_adjust(hspace=0.8, wspace=0.4)
    
    # After the existing 9 charts, add the following:
    
    # 10. Risk-Adjusted Performance Matrix
    ax10 = plt.subplot(4, 3, 10)  # Change to 4x3 grid
    ax10.set_title('Risk-Adjusted Performance Matrix', fontsize=12)
    
    # Only use BUY/HOLD stocks with valid data
    perf_df = df[(df['Status'] == 'BUY/HOLD') & df['Profit_Loss'].notna() & df['Holding_Days'].notna()].copy()
    
    if not perf_df.empty and len(perf_df) >= 3:
        # Calculate daily return and volatility
        perf_df['Daily_Return'] = perf_df['Profit_Loss'] / perf_df['Holding_Days']
        perf_df['Risk_Category'] = pd.qcut(perf_df['Profit_Loss'].abs(), 3, labels=['Low', 'Medium', 'High'])
        perf_df['Return_Category'] = pd.qcut(perf_df['Daily_Return'], 3, labels=['Low', 'Medium', 'High'])
        
        # Create scatter plot
        risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        for risk, group in perf_df.groupby('Risk_Category'):
            ax10.scatter(group['Holding_Days'], group['Profit_Loss'], 
                        label=f'{risk} Risk', color=risk_colors[risk], 
                        alpha=0.7, s=100)
            
            # Label top performers in each category
            for _, row in top.iterrows():
                ax10.annotate(row['Symbol'], 
                             (row['Holding_Days'], row['Profit_Loss']),
                             xytext=(5, 5), textcoords='offset points')
                             
        # Add optimal hold period range
        if len(perf_df) > 5:
            # Find optimal holding period range (highest avg daily returns)
            perf_df['Hold_Bucket'] = pd.cut(perf_df['Holding_Days'], 
                                          bins=[0, 10, 30, 60, 120, float('inf')],
                                          labels=['0-10d', '11-30d', '31-60d', '61-120d', '>120d'])
            best_bucket = perf_df.groupby('Hold_Bucket')['Daily_Return'].mean().idxmax()
            
            # Shade the optimal region
            bucket_ranges = {'0-10d': (0, 10), '11-30d': (11, 30), 
                           '31-60d': (31, 60), '61-120d': (61, 120), '>120d': (121, 200)}
            if best_bucket in bucket_ranges:
                min_x, max_x = bucket_ranges[best_bucket]
                ax10.axvspan(min_x, max_x, alpha=0.2, color='green')
                ax10.text((min_x + max_x)/2, ax10.get_ylim()[1]*0.9, 
                         f"Optimal Hold: {best_bucket}", ha='center',
                         bbox=dict(facecolor='white', alpha=0.8))
            
        ax10.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax10.set_xlabel('Holding Period (Days)')
        ax10.set_ylabel('Profit/Loss (%)')
        ax10.legend(title='Risk Level')
        ax10.grid(True, alpha=0.3)
    else:
        ax10.text(0.5, 0.5, 'Insufficient data for performance matrix',
                 ha='center', va='center', transform=ax10.transAxes)

    # 11. Decision Support - Action Recommendations
    ax11 = plt.subplot(4, 3, 11)
    ax11.set_title('Action Recommendations', fontsize=12)
    
    # Create decision support categories
    action_counts = {
        'Strong Buy': len(df[(df['Market_Phase'] == 'ACCUMULATION') & 
                             (df['Phase_Probability'] > 70) & 
                             (df['RSI'] < 50) & 
                             (df['AO'] > 0)]),
        'Buy': len(df[(df['Market_Phase'] == 'ACCUMULATION') & 
                      (df['Phase_Probability'] > 55)]),
        'Hold': len(df[(df['Status'] == 'BUY/HOLD') & 
                       (df['Market_Phase'] != 'DISTRIBUTION')]),
        'Take Profit': len(df[(df['Status'] == 'BUY/HOLD') & 
                             (df['Market_Phase'] == 'DISTRIBUTION') & 
                             (df['Profit_Loss'] > 0 if 'Profit_Loss' in df.columns else False)]),
        'Cut Loss': len(df[(df['Status'] == 'BUY/HOLD') & 
                          (df['Market_Phase'] == 'DISTRIBUTION') & 
                          (df['Profit_Loss'] < 0 if 'Profit_Loss' in df.columns else False)]),
        'Avoid': len(df[(df['Market_Phase'] == 'DISTRIBUTION') & 
                       (df['Phase_Probability'] > 70)])
    }
    
    # Create action guidance visualization
    actions = list(action_counts.keys())
    values = list(action_counts.values())
    colors = ['darkgreen', 'green', 'blue', 'orange', 'red', 'darkred']
    
    bars = ax11.bar(actions, values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax11.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                     str(int(height)), ha='center', va='bottom', fontsize=8)
    
    ax11.set_ylabel('Number of Stocks')
    ax11.set_xticklabels(actions, rotation=45, ha='right')
    ax11.grid(True, axis='y', alpha=0.3)
    
    # Add a note for how to use this chart
    ax11.text(0.5, -0.3, 'Focus on Strong Buy for new entries, Take Profit for overbought positions',
             transform=ax11.transAxes, ha='center', fontsize=9)

    # 12. Sector Rotation Heat Map
    ax12 = plt.subplot(4, 3, 12)
    ax12.set_title('Market Sector Performance', fontsize=12)
    
    # Create sector categories (could be based on industry or market cap)
    if 'Price_Category' in df.columns:
        # We already have price categories from previous analysis
        # Get average RSI and AO values by category
        sector_metrics = df.groupby('Price_Category').agg({
            'RSI': 'mean',
            'AO': 'mean',
            'Phase_Probability': 'mean',
            'Symbol': 'count'
        }).reset_index()

        sector_metrics = sector_metrics.rename(columns={'Symbol': 'Count'})

        if not sector_metrics.empty:
            # Create array for heatmap
            sectors = sector_metrics['Price_Category'].tolist()
            metrics = ['RSI', 'AO', 'Phase_Probability']
            data = sector_metrics[metrics].values
            
            # Create heatmap
            im = ax12.imshow(data.T, cmap='RdYlGn', aspect='auto')
            
            # Add labels
            ax12.set_xticks(np.arange(len(sectors)))
            ax12.set_yticks(np.arange(len(metrics)))
            ax12.set_xticklabels(sectors)
            ax12.set_yticklabels(metrics)
            plt.setp(ax12.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations in each cell
            for i in range(len(sectors)):
                for j in range(len(metrics)):
                    value = data[i, j]
                    text_color = 'black' if 30 < value < 70 else 'white'
                    ax12.text(i, j, f"{value:.1f}", ha="center", va="center", 
                             color=text_color, fontweight="bold")
            
            # Add count below each column
            for i, count in enumerate(sector_metrics['Count']):
                ax12.text(i, len(metrics), f"n={count}", ha="center", va="center")
                
            # Add a title explaining what we're seeing
            lead_sector = sector_metrics.loc[sector_metrics['Phase_Probability'].idxmax(), 'Price_Category']
            ax12.set_title(f'Market Sector Performance (Leader: {lead_sector})', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax12, orientation='horizontal', pad=0.2)
            cbar.set_label('Score (Higher is Better)')
        else:
            ax12.text(0.5, 0.5, 'Insufficient data for sector analysis',
                     ha='center', va='center', transform=ax12.transAxes)
            ax12.axis('off')
    else:
        ax12.text(0.5, 0.5, 'Sector data not available',
                 ha='center', va='center', transform=ax12.transAxes)
        ax12.axis('off')

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
        logging.error("No stock symbols found in the database.")
        connection.close()
        exit(1)
        
    # Add this line at the beginning of your main code
    create_default_symbols_file()
    
    # Display all buy stocks sorted by update_date (most recent first), then by holding days
    print("\n🟢 ALL BUY SIGNALS (SORTED BY UPDATE DATE) 🟢")
    print("============================================")
    
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
        
        print("\n📊 GENERATING CHARTS FOR BUY SIGNALS ONLY 📊")
        print("===========================================")
        
        success_count = 0
        fail_count = 0
        
        # Only generate charts for stocks with active buy signals
        if latest_buy_symbols:
            for symbol in tqdm(latest_buy_symbols, desc="Generating charts"):
                if symbol in available_symbols:
                    table_name = f"PSX_{symbol}_stock_data"
                    print(f"Processing {symbol}...")
                    
                    success = draw_indicator_trend_lines_with_signals(database_path, table_name)
                    if success:
                        success_count += 1
                        print(f"✅ Chart for {symbol} has been generated and sent to Telegram")
                    else:
                        fail_count += 1
                        print(f"❌ Failed to generate chart for {symbol}")
                else:
                    print(f"⚠️ Symbol {symbol} not found in available data tables")
                    fail_count += 1
            
            print(f"\nCompleted processing {len(latest_buy_symbols)} buy signal stocks.")
            print(f"✅ Successfully generated: {success_count} charts")
            print(f"❌ Failed to generate: {fail_count} charts")
            print(f"Charts saved in the RSI_AO_CHARTS folder.")
            
            # Send signals and charts summary to Telegram
            send_signals_and_charts_summary(latest_buy_df, latest_sell_df, available_symbols, success_count)
        else:
            print("No buy signals to process for chart generation.")
    else:
        print("No buy signals found in the database.")
    
    # Generate comprehensive market dashboard
    print("\n📈 GENERATING MARKET DASHBOARD 📈")
    print("================================")
    dashboard_df = generate_stock_dashboard()
    if not dashboard_df.empty:
        print(f"Dashboard generated successfully with {len(dashboard_df)} stocks analyzed")
        
        # Display summary statistics
        buy_count = len(dashboard_df[dashboard_df['Status'] == 'BUY/HOLD'])
        sell_count = len(dashboard_df[dashboard_df['Status'] == 'SELL'])
        opp_count = len(dashboard_df[dashboard_df['Status'] == 'OPPORTUNITY'])
        
        print(f"\nSummary Statistics:")
        print(f"- BUY/HOLD signals: {buy_count}")
        print(f"- SELL signals: {sell_count}")
        print(f"- OPPORTUNITY signals: {opp_count}")
        
        # Display accumulation/distribution stats
        acc_count = len(dashboard_df[dashboard_df['Market_Phase'] == 'ACCUMULATION'])
        dist_count = len(dashboard_df[dashboard_df['Market_Phase'] == 'DISTRIBUTION'])
        neut_count = len(dashboard_df[dashboard_df['Market_Phase'] == 'NEUTRAL'])
        
        print(f"\nMarket Phase Statistics:")
        print(f"- ACCUMULATION: {acc_count} stocks ({acc_count/len(dashboard_df)*100:.2f}%)")
        print(f"- DISTRIBUTION: {dist_count} stocks ({dist_count/len(dashboard_df)*100:.2f}%)")
        print(f"- NEUTRAL: {neut_count} stocks ({neut_count/len(dashboard_df)*100:.2f}%)")
        
        # Dashboard files location
        current_date = datetime.now().strftime('%Y-%m-%d')
        print(f"\nDashboard saved in PSX_DASHBOARDS folder with date {current_date}")
        
        # Define dashboards folder (this was missing)
        dashboards_folder = 'PSX_DASHBOARDS'
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
        print("Failed to generate market dashboard")
    
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
                logging.warning("Table 'sell_stocks' does not exist in the database")
                return pd.DataFrame()
            
            cursor.execute("PRAGMA table_info(sell_stocks)")
            columns = [info[1] for info in cursor.fetchall()]
            logging.info(f"Actual columns in sell_stocks: {columns}")
            
            # Get a list of all unique stocks
            cursor.execute("SELECT DISTINCT Stock FROM sell_stocks WHERE Signal_Date IS NOT NULL")
            stocks = [row[0] for row in cursor.fetchall()]
            logging.info(f"Found {len(stocks)} unique stocks with sell signal dates")
            
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
        logging.error(f"Error getting latest sell stocks: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()
# full and final code and no more changes required.



