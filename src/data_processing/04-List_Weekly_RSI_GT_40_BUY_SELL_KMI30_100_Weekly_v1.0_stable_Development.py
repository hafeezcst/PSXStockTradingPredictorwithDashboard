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


# Main function to get stock data with RSI above 40
def get_stock_data_with_rsi_above_40(db_paths):
    """
    Optimized function to fetch stock data with RSI above 40 from multiple databases.
    Uses batch processing and connection pooling to improve performance.
    """
    all_buy_stock_data = []
    all_sell_stock_data = []
    all_neutral_stock_data = []
    all_breakout_data = []

    for db_path in db_paths:
        logging.info(f"\nProcessing database: {db_path}")
        try:
            # Verify database exists and is accessible
            if not os.path.exists(db_path):
                logging.error(f"Database file not found: {db_path}")
                continue
                
            if os.path.getsize(db_path) == 0:
                logging.error(f"Database file is empty: {db_path}")
                continue

            # Single connection for all operations to reduce overhead
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Verify basic database functionality and log schema
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                if not tables:
                    logging.error(f"No tables found in database: {db_path}")
                    continue
                
                # Log database schema for debugging only once
                logging.info(f"Database schema for {db_path}:")
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    logging.info(f"Table: {table_name} - Columns: {[col[1] for col in columns]}")
                    
                    # Verify required columns exist for stock data tables
                    if table_name.endswith('_stock_data'):
                        required_columns = {'Date', 'Close', 'RSI_weekly_Avg', 'AO_weekly', 'Volume'}
                        actual_columns = {col[1] for col in columns}
                        missing = required_columns - actual_columns
                        if missing:
                            logging.error(f"Missing required columns in {table_name}: {missing}")
                            continue

                multibagger_symbols = get_multibagger_symbols()
                logging.info(f"Found {len(multibagger_symbols)} multibagger symbols")
                
                tables = fetch_table_names(cursor)
                logging.info(f"Found {len(tables)} tables to process")
                
                buy_stock_data = []
                sell_stock_data = []
                neutral_stock_data = []
                breakout_data = []
                filtered_symbols = []

                for table_name in tables:
                    try:
                        logging.info(f"\nProcessing table: {table_name}")
                        results = fetch_stock_data(cursor, table_name, limit=30)  # Fetch more data for breakout
                        if results:
                            logging.info(f"Found {len(results)} results for {table_name}")
                            buy_data, sell_data, neutral_data, breakout = process_stock_data(
                                table_name, results, cursor, multibagger_symbols, os.path.basename(db_path)
                            )
                            
                            if buy_data:
                                logging.info(f"Found {len(buy_data)} buy signals")
                                buy_stock_data.extend(buy_data)
                            if sell_data:
                                logging.info(f"Found {len(sell_data)} sell signals")
                                sell_stock_data.extend(sell_data)
                            if neutral_data:
                                logging.info(f"Found {len(neutral_data)} neutral signals")
                                neutral_stock_data.extend(neutral_data)
                            if breakout:
                                logging.info(f"Found breakout signals for {table_name}")
                                breakout_data.append((table_name, breakout))
                            
                            filtered_symbols.append(
                                table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
                            )
                        else:
                            logging.warning(f"No results found for {table_name}")
                    except sqlite3.OperationalError as e:
                        logging.error(f"Database error processing {table_name}: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"Error processing {table_name}: {e}")
                        continue

                # Use KMI100 as the data source
                data_source_name = 'KMI100'
                logging.info(f"\nData Source: {data_source_name}")
                logging.info(f"Total buy signals: {len(buy_stock_data)}")
                logging.info(f"Total sell signals: {len(sell_stock_data)}")
                logging.info(f"Total neutral signals: {len(neutral_stock_data)}")
                logging.info(f"Total breakout signals: {len(breakout_data)}")
                
                all_buy_stock_data.append((data_source_name, buy_stock_data, filtered_symbols))
                all_sell_stock_data.append((data_source_name, sell_stock_data, filtered_symbols))
                all_neutral_stock_data.append((data_source_name, neutral_stock_data, filtered_symbols))
                all_breakout_data.append((data_source_name, breakout_data, filtered_symbols))
        except Exception as e:
            logging.error(f"Error processing database {db_path}: {str(e)}", exc_info=True)
            continue
            
    return all_buy_stock_data, all_sell_stock_data, all_neutral_stock_data, all_breakout_data


def get_KMI_symbols():
    """Get list of KMI30 and KMI100 symbols from Excel file."""
    try:
        symbols_file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        
        # Read KMI30 symbols
        kmi30_df = pd.read_excel(symbols_file_path, sheet_name='KMI30')
        KMI30_symbols = set(kmi30_df.iloc[:, 0].tolist()[:30])  # Limit to top 30
        
        # Read KMI100 symbols
        kmi100_df = pd.read_excel(symbols_file_path, sheet_name='KMI100')
        KMI100_symbols = set(kmi100_df.iloc[:, 0].tolist()[:100])  # Limit to top 100
        
        # Return both sets separately
        return KMI30_symbols, KMI100_symbols
    except Exception as e:
        logging.error(f"Error reading KMI30 and KMI100 symbols: {e}")
        return set(), set()

def get_kmi_tag(symbol: str, kmi30_symbols: set, kmi100_symbols: set) -> str:
    """Generate KMI tag based on which indices the symbol belongs to."""
    in_kmi30 = symbol in kmi30_symbols
    in_kmi100 = symbol in kmi100_symbols
    
    if in_kmi30 and in_kmi100:
        return " (KMI30 & KMI100)"
    elif in_kmi30:
        return " (KMI30)"
    elif in_kmi100:
        return " (KMI100)"
    return ""

# Function to get dividend information for a symbol from the correct database
DIVIDEND_DB = 'data/databases/production/PSX_Dividend_Schedule.db'

def get_dividend_info(symbol: str) -> Optional[Dict]:
    """Get dividend information for a specific symbol if available."""
    db_path = DIVIDEND_DB
    
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        cursor.execute('''
        SELECT 
            symbol,
            company_name,
            face_value,
            dividend_amount,
            right_amount,
            bc_to,
            last_close,
            payout_text
        FROM dividend_schedule 
        WHERE bc_to >= ? 
        AND bc_to != '-'
        AND upper(symbol) = upper(?)
        ORDER BY date(bc_to)
        LIMIT 1
        ''', (today, symbol))
        
        row = cursor.fetchone()
        
        if row:
            return {
                'symbol': row[0],
                'company_name': row[1],
                'face_value': row[2],
                'dividend_amount': row[3],
                'right_amount': row[4],
                'bc_to': row[5],
                'last_close': row[6],
                'payout_text': row[7]
            }
        
        return None
        
    except sqlite3.Error as e:
        logging.error(f"Error getting dividend info: {str(e)}")
        return None
    finally:
        conn.close()

def format_dividend_info(dividend_info: Dict) -> str:
    """Format dividend information for a stock."""
    if not dividend_info:
        return ""
    
    if dividend_info['dividend_amount']:
        div_per_share = dividend_info['face_value'] * dividend_info['dividend_amount']
        div_yield = (div_per_share / dividend_info['last_close'] * 100) if dividend_info['last_close'] else 0
        return f"📅 Book Closure: {dividend_info['bc_to']}\n💰 Dividend: Rs. {div_per_share:.2f}/share ({dividend_info['payout_text']})\n📈 Yield: {div_yield:.2f}%"
    elif dividend_info['right_amount']:
        return f"📅 Book Closure: {dividend_info['bc_to']}\n🔄 Right Share: {dividend_info['payout_text']}"
    return ""

def generate_buy_signal_description(row: Dict) -> str:
    """Generate an AI description for buy signals based on technical indicators with enhanced logic"""
    strength = "Strong" if row['RSI_Weekly_Avg'] >= 60 else "Moderate"
    momentum = "increasing" if row['AO_Weekly'] > 0 else "steady"
    rsi_status = "bullish territory" if row['RSI_Weekly_Avg'] >= 50 else "neutral territory"
    volume_strength = "high" if row['Volume'] > 100000 else "moderate"
    
    # Enhanced trend analysis with additional conditions
    if row['Close'] > row['MA_30'] and row['AO_Weekly'] > 0:
        trend_message = f"Price {row['Close']:.2f} is above MA30 {row['MA_30']:.2f}, confirming strong upward trend"
    elif row['Close'] > row['MA_30']:
        trend_message = f"Price {row['Close']:.2f} is above MA30 {row['MA_30']:.2f}, indicating potential upward trend"
    else:
        trend_message = f"Price {row['Close']:.2f} is below MA30 {row['MA_30']:.2f}, showing potential support level"

    description = (
        f"Analysis: {strength} buy signal with {momentum} momentum. "
        f"RSI is in {rsi_status} at {row['RSI_Weekly_Avg']:.2f}, "
        f"showing bullish trend. "
        f"AO at {row['AO_Weekly']:.2f} indicates bullish pressure. "
        f"Trading with {volume_strength} volume of {row['Volume']:,.0f} shares. "
        f"{trend_message}."
    )
    return description

def format_buy_signals(data_source_name: str, df: pd.DataFrame) -> str:
    """Format buy signals for Telegram message."""
    analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"🟢 *{data_source_name} Buy Signals* 📈\n\n"
    
    # Get KMI symbols
    KMI30_symbols, KMI100_symbols = get_KMI_symbols()
    
    # Sort by holding days in ascending order
    df = df.sort_values(by='Holding_Days', ascending=True)
    
    for _, row in df.iterrows():
        symbol = row['Stock']
        company = row['Data Source']
        close = row['Close']
        rsi = row['RSI_Weekly_Avg']
        ao = row['AO_Weekly']
        signal_date = row.get('Signal_Date', 'N/A')
        holding_days = row.get('Holding_Days', 'N/A')
        signal_price = row.get('Signal_Close', 0)
        current_price_date = row['Date']
        
        # Get KMI tag based on indices
        KMI_tag = get_kmi_tag(symbol, KMI30_symbols, KMI100_symbols)
        
        # Calculate P/L only if we have a valid signal price
        pl_text = ""
        if signal_price and signal_price > 0:
            pl = ((close - signal_price) / signal_price * 100)
            pl_text = f"📊 P/L: {pl:+.2f}%\n"

        # Generate a unique hash for same-day signal tracking with index categorization
        index_category = "KMI30_100" if "KMI30 & KMI100" in KMI_tag else "KMI30" if "KMI30" in KMI_tag else "KMI100" if "KMI100" in KMI_tag else "Other"
        signal_hash = f"#BUY_Signal_{datetime.now().strftime('%Y%m%d')}_{index_category}"
        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"🟢 *{symbol}*{KMI_tag} - {signal_hash}\n"
        message += f"💰 Current Price: {close:.2f}\n"
        message += f"📊 RSI: {rsi:.2f}\n"
        message += f"📈 AO: {ao:.2f}\n"
        message += f"💸 Latest Volume: {row.get('Volume', 0):,.0f}\n"
        message += f"📅 Signal Date: {signal_date}\n"
        if signal_price and signal_price > 0:
            message += f"💵 Signal Price: {signal_price:.2f}\n"
            message += pl_text
        message += f"⏳ Holding Days: {holding_days}\n"
        
        # Add dividend information if available
        dividend_info = get_dividend_info(symbol)
        if dividend_info:
            message += f"\n{format_dividend_info(dividend_info)}\n"
        
        # Add AI-generated description
        message += f"\n🤖 {generate_buy_signal_description(row)}\n\n"
    
    return message

def generate_sell_signal_description(row: Dict) -> str:
    """Generate an AI description for sell signals based on technical indicators with enhanced logic"""
    if row['RSI_Weekly_Avg'] <= 30:
        condition = "oversold"
        action = "potential reversal"
    else:
        condition = "weakening"
        action = "downward pressure"
    
    momentum = "decreasing" if row['AO_Weekly'] < 0 else "mixed"
    volume_strength = "high" if row['Volume'] > 100000 else "moderate"
    
    # Enhanced trend analysis with additional conditions
    if row['Close'] < row['MA_30'] and row['AO_Weekly'] < 0:
        trend_message = f"Price {row['Close']:.2f} is below MA30 {row['MA_30']:.2f}, confirming strong downward trend"
    elif row['Close'] < row['MA_30']:
        trend_message = f"Price {row['Close']:.2f} is below MA30 {row['MA_30']:.2f}, indicating potential downward trend"
    else:
        trend_message = f"Price {row['Close']:.2f} is above MA30 {row['MA_30']:.2f}, showing potential resistance level"
    
    description = (
        f"Analysis: Stock showing {condition} conditions with {momentum} momentum. "
        f"RSI at {row['RSI_Weekly_Avg']:.2f} indicates {action}, "
        f"with bearish trend. "
        f"AO at {row['AO_Weekly']:.2f} shows bearish pressure. "
        f"Trading with {volume_strength} volume of {row['Volume']:,.0f} shares. "
        f"{trend_message}."
    )
    return description

def format_sell_signals(data_source_name: str, df: pd.DataFrame) -> str:
    """Format sell signals for Telegram message."""
    analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"🔴 *{data_source_name} Sell Signals* 📉\n\n"
    
    # Get KMI symbols
    KMI30_symbols, KMI100_symbols = get_KMI_symbols()
    
    # Sort by P/L in descending order
    df = df.sort_values(by='% P/L', ascending=False)
    
    for _, row in df.iterrows():
        symbol = row['Stock']
        close = row['Close']
        rsi = row['RSI_Weekly_Avg']
        ao = row['AO_Weekly']
        signal_date = row.get('Signal_Date', 'N/A')
        pl = row.get('% P/L', 0)
        current_price_date = row['Date']
        
        # Get KMI tag based on indices
        KMI_tag = get_kmi_tag(symbol, KMI30_symbols, KMI100_symbols)
        
        # Generate a unique hash for same-day signal tracking with index categorization
        index_category = "KMI30_100" if "KMI30 & KMI100" in KMI_tag else "KMI30" if "KMI30" in KMI_tag else "KMI100" if "KMI100" in KMI_tag else "Other"
        signal_hash = f"#SELL_Signal_{datetime.now().strftime('%Y%m%d')}_{index_category}"
        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"🔴 *{symbol}*{KMI_tag} - {signal_hash}\n"
        message += f"💰 Current Price: {close:.2f}\n"
        message += f"📊 RSI: {rsi:.2f}\n"
        message += f"📉 AO: {ao:.2f}\n"
        message += f"💸 Latest Volume: {row.get('Volume', 0):,.0f}\n"
        message += f"📅 Signal Date: {signal_date}\n"
        message += f"📊 P/L: {pl:+.2f}%\n\n"
        
        # Add dividend information if available
        dividend_info = get_dividend_info(symbol)
        if dividend_info:
            message += f"{format_dividend_info(dividend_info)}\n"
        
        # Add AI-generated description
        message += f"\n🤖 {generate_sell_signal_description(row)}\n\n"
    
    return message

def generate_neutral_signal_description(row: Dict) -> str:
    """Generate an AI description for neutral signals based on technical indicators"""
    rsi_position = "balanced" if 40 <= row['RSI_Weekly_Avg'] <= 60 else (
        "slightly oversold" if row['RSI_Weekly_Avg'] < 40 else "slightly overbought"
    )
    momentum = "mixed" if -2 < row['AO_Weekly'] < 2 else (
        "slightly bullish" if row['AO_Weekly'] >= 2 else "slightly bearish"
    )
    
    description = (
        f"Analysis: Stock in neutral zone with {momentum} momentum. "
        f"RSI at {row['RSI_Weekly_Avg']:.2f} shows {rsi_position} conditions, "
        f"with neutral trend. "
        f"AO at {row['AO_Weekly']:.2f} indicates neutral pressure. "
        f"Volume at {row['Volume']:,.0f} shares. "
        f"Price {row['Close']:.2f} relative to MA30 {row['MA_30']:.2f} suggests neutral trend."
    )
    return description

def format_neutral_signals(data_source_name: str, df: pd.DataFrame) -> str:
    """Format neutral signals for Telegram message."""
    analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"🟡 *{data_source_name} Neutral Signals* 📊\n\n"
    
    # Get KMI symbols
    KMI30_symbols, KMI100_symbols = get_KMI_symbols()
    
    # Sort by RSI in descending order, check if column exists
    if 'RSI_Weekly_Avg' in df.columns:
        df = df.sort_values(by='RSI_Weekly_Avg', ascending=False)
    else:
        logging.warning("Column 'RSI_Weekly_Avg' not found in DataFrame, skipping sort.")
        logging.info(f"Available columns: {df.columns}")
    
    for _, row in df.iterrows():
        symbol = row['Stock']
        close = row['Close']
        rsi = row['RSI_Weekly_Avg']
        ao = row['AO_Weekly']
        trend = row['Trend_Direction']
        current_price_date = row['Date']
        
        # Get KMI tag based on indices
        KMI_tag = get_kmi_tag(symbol, KMI30_symbols, KMI100_symbols)
        
        # Add trend indicator emoji with color adjustment (green for bullish, red for bearish)
        trend_emoji = "🟢" if trend == "Bullish" else "🔴"
        
        # Generate a unique hash for same-day signal tracking with index categorization
        index_category = "KMI30_100" if "KMI30 & KMI100" in KMI_tag else "KMI30" if "KMI30" in KMI_tag else "KMI100" if "KMI100" in KMI_tag else "Other"
        signal_type = "Bullish" if trend == "Bullish" else "Bearish"
        signal_hash = f"#NEUTRAL_Signal_{datetime.now().strftime('%Y%m%d')}_{signal_type}_{index_category}"
        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"🟡 {trend_emoji} 🚀 *{symbol}*{KMI_tag} - {signal_hash}\n"
        message += f"💰 Current Price: {close:.2f}\n"
        message += f"📊 RSI: {rsi:.2f}\n"
        message += f"📈 AO: {ao:.2f}\n"
        message += f"💸 Latest Volume: {row.get('Volume', 0):,.0f}\n"
        message += f"📈 Trend: {trend}\n\n"
        
        # Add dividend information if available
        dividend_info = get_dividend_info(symbol)
        if dividend_info:
            message += f"{format_dividend_info(dividend_info)}\n"
        
        # Add AI-generated description
        message += f"\n🤖 {generate_neutral_signal_description(row)}\n\n"
    
    return message



def send_telegram_message(message: str) -> bool:
    """Send message to Telegram channel with optimized rate limit handling and batch processing."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram bot token or chat ID not configured")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Escape special characters for MarkdownV2 efficiently
    special_chars = {'_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'}
    escaped_message = ''.join(f'\\{char}' if char in special_chars else char for char in message)
    
    # Split message into chunks if too long
    max_length = 4096
    messages = [escaped_message[i:i+max_length] for i in range(0, len(escaped_message), max_length)]
    
    success = True
    base_delay = 2  # Base delay in seconds
    max_retries = 5  # Maximum number of retries per message
    timeout = (5, 15)  # Reduced timeouts: (connect timeout, read timeout) in seconds
    
    # Use a single session for all requests to reuse connections
    with requests.Session() as session:
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        for chunk in messages:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    payload = {
                        'chat_id': TELEGRAM_CHAT_ID,
                        'text': chunk,
                        'parse_mode': 'MarkdownV2',
                        'disable_web_page_preview': True
                    }
                    
                    response = session.post(url, json=payload, timeout=timeout)
                    
                    if response.status_code == 429:  # Rate limit hit
                        retry_after = int(response.headers.get('Retry-After', base_delay))
                        wait_time = retry_after * (2 ** retry_count)  # Exponential backoff
                        logging.info(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                        
                    response.raise_for_status()
                    logging.info(f"Successfully sent chunk of length {len(chunk)}")
                    
                    # Add delay between messages to avoid rate limiting
                    if len(messages) > 1:
                        time.sleep(base_delay)
                    break
                    
                except requests.exceptions.Timeout:
                    retry_count += 1
                    wait_time = base_delay * (2 ** retry_count)  # Exponential backoff
                    logging.warning(f"Timeout occurred. Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                    if retry_count >= max_retries:
                        logging.error("Max retries reached for timeout")
                        success = False
                        break
                    continue
                    
                except requests.exceptions.ConnectionError as e:
                    retry_count += 1
                    wait_time = base_delay * (2 ** retry_count)  # Exponential backoff
                    logging.warning(f"Connection error occurred: {e}. Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                    if retry_count >= max_retries:
                        logging.error("Max retries reached for connection error")
                        success = False
                        break
                    continue
                    
                except requests.RequestException as e:
                    logging.error(f"Error sending message to Telegram: {e}")
                    if 'response' in locals() and response is not None:
                        logging.error(f"Response content: {response.text}")
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = base_delay * (2 ** retry_count)  # Exponential backoff
                        logging.info(f"Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    success = False
                    break
    
    if success:
        logging.info("Successfully sent all message chunks to Telegram")
    else:
        logging.error("Failed to send some message chunks to Telegram")
    
    return success

def format_breakout_message(breakout_data: Dict, symbol: str) -> str:
    """Format breakout signals for Telegram message, including weekly and monthly comparisons."""
    analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Generate a distinctive hash for same-day signal tracking using date component, signal type, and timeframe
    bullish_breakouts_weekly = breakout_data.get('bullish_breakout_weekly', [])
    bullish_breakouts_monthly = breakout_data.get('bullish_breakout_monthly', [])
    bearish_breakouts_weekly = breakout_data.get('bearish_breakout_weekly', [])
    bearish_breakouts_monthly = breakout_data.get('bearish_breakout_monthly', [])
    index_category = "KMI30_100" if "KMI30 & KMI100" in get_kmi_tag(symbol, get_KMI_symbols()[0], get_KMI_symbols()[1]) else "KMI30" if "KMI30" in get_kmi_tag(symbol, get_KMI_symbols()[0], get_KMI_symbols()[1]) else "KMI100" if "KMI100" in get_kmi_tag(symbol, get_KMI_symbols()[0], get_KMI_symbols()[1]) else "Other"
    
    message = f"🚀 *Breakout Alert for {symbol}* ⚡\n\n"
    message += f"🕒 Analysis Time: {analysis_time}\n\n"
    
    # Weekly Breakout Analysis
    bullish_breakouts_weekly = breakout_data.get('bullish_breakout_weekly', [])
    bearish_breakouts_weekly = breakout_data.get('bearish_breakout_weekly', [])
    
    if bullish_breakouts_weekly:
        signal_hash_weekly_bullish = f"#Breakout_Weekly_Bullish_{datetime.now().strftime('%Y%m%d')}_{index_category}"
        message += f"📈 *Weekly Bullish Breakouts Detected: {len(bullish_breakouts_weekly)}* {signal_hash_weekly_bullish}\n"
        for i, breakout in enumerate(bullish_breakouts_weekly, 1):
            message += f"{i}. Date: {breakout['date']}\n"
            message += f"   Current Price: {breakout['current_price']:.2f} vs Weekly Close: {breakout['weekly_close']:.2f}\n"
            message += f"   Volume: {breakout['volume']:,.0f}\n"
            message += f"   RSI: {breakout['rsi']:.2f}\n"
            message += f"   AO: {breakout['ao']:.2f}\n"
            message += f"   Conditions: {', '.join(breakout['conditions_met'])}\n\n"
    
    if bearish_breakouts_weekly:
        signal_hash_weekly_bearish = f"#Breakout_Weekly_Bearish_{datetime.now().strftime('%Y%m%d')}_{index_category}"
        message += f"📉 *Weekly Bearish Breakouts Detected: {len(bearish_breakouts_weekly)}* {signal_hash_weekly_bearish}\n"
        for i, breakout in enumerate(bearish_breakouts_weekly, 1):
            message += f"{i}. Date: {breakout['date']}\n"
            message += f"   Current Price: {breakout['current_price']:.2f} vs Weekly Close: {breakout['weekly_close']:.2f}\n"
            message += f"   Volume: {breakout['volume']:,.0f}\n"
            message += f"   RSI: {breakout['rsi']:.2f}\n"
            message += f"   AO: {breakout['ao']:.2f}\n"
            message += f"   Conditions: {', '.join(breakout['conditions_met'])}\n\n"
    
    # Monthly Breakout Analysis (Placeholder for future implementation if not already in analyzer)
    bullish_breakouts_monthly = breakout_data.get('bullish_breakout_monthly', [])
    bearish_breakouts_monthly = breakout_data.get('bearish_breakout_monthly', [])
    
    if bullish_breakouts_monthly:
        signal_hash_monthly_bullish = f"#Breakout_Monthly_Bullish_{datetime.now().strftime('%Y%m%d')}_{index_category}"
        message += f"📈 *Monthly Bullish Breakouts Detected: {len(bullish_breakouts_monthly)}* {signal_hash_monthly_bullish}\n"
        for i, breakout in enumerate(bullish_breakouts_monthly, 1):
            message += f"{i}. Date: {breakout['date']}\n"
            message += f"   Current Price: {breakout['current_price']:.2f} vs Monthly Close: {breakout['monthly_close']:.2f}\n"
            message += f"   Volume: {breakout['volume']:,.0f}\n"
            message += f"   RSI: {breakout['rsi']:.2f}\n"
            message += f"   AO: {breakout['ao']:.2f}\n"
            message += f"   Conditions: {', '.join(breakout['conditions_met'])}\n\n"
    
    if bearish_breakouts_monthly:
        signal_hash_monthly_bearish = f"#Breakout_Monthly_Bearish_{datetime.now().strftime('%Y%m%d')}_{index_category}"
        message += f"📉 *Monthly Bearish Breakouts Detected: {len(bearish_breakouts_monthly)}* {signal_hash_monthly_bearish}\n"
        for i, breakout in enumerate(bearish_breakouts_monthly, 1):
            message += f"{i}. Date: {breakout['date']}\n"
            message += f"   Current Price: {breakout['current_price']:.2f} vs Monthly Close: {breakout['monthly_close']:.2f}\n"
            message += f"   Volume: {breakout['volume']:,.0f}\n"
            message += f"   RSI: {breakout['rsi']:.2f}\n"
            message += f"   AO: {breakout['ao']:.2f}\n"
            message += f"   Conditions: {', '.join(breakout['conditions_met'])}\n\n"
    
    if not bullish_breakouts_weekly and not bearish_breakouts_weekly and not bullish_breakouts_monthly and not bearish_breakouts_monthly:
        return ""
    
    return message

def handle_breakout_data(data_tuple):
    data_source_name, breakout_data, filtered_symbols = data_tuple
    
    if not breakout_data:
        logging.info("No breakout signals found")
        return
        
    logging.info(f"\nProcessing breakout signals for {data_source_name}...")
    
    for table_name, breakout in breakout_data:
        symbol = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
        
        # Format breakout message with weekly and monthly analysis
        message = format_breakout_message(breakout, symbol)
        
        if message:
            logging.info(f"Sending breakout message for {symbol}")
            try:
                send_telegram_message(message)
                logging.info(f"Breakout message sent successfully for {symbol}")
            except Exception as e:
                logging.error(f"Error sending breakout message for {symbol}: {e}")
                
    # Note: Ensure analyzer.identify_weekly_breakouts and a potential identify_monthly_breakouts
    # are implemented to compare current closing price with weekly and monthly closing prices.
    logging.info("Ensure breakout analysis includes weekly and monthly closing price comparisons in analyzer module.")

# Main execution
if __name__ == "__main__":
    try:
        logging.info("\nRunning stock analysis...")
        db_paths = ['data/databases/production/psx_consolidated_data_indicators_PSX.db']
        all_buy_stock_data, all_sell_stock_data, all_neutral_stock_data, all_breakout_data = get_stock_data_with_rsi_above_40(db_paths)

        # Function to handle stock data processing and database updates
        def handle_stock_data(data_tuple, stock_type, format_func):
            data_source_name, stock_data, filtered_symbols = data_tuple
            df = pd.DataFrame(stock_data)
            
            logging.info(f"\nProcessing {stock_type} signals...")
            logging.info(f"DataFrame empty: {df.empty}")
            if not df.empty:
                logging.info(f"DataFrame columns: {df.columns}")
            
            if not df.empty and 'Volume' in df.columns:
                # Log data
                logging.info(f"\nData Source: {data_source_name}")
                logging.info(f"Found {len(df)} {stock_type} signals")
                logging.info(tabulate(df, headers='keys', tablefmt='dash', showindex=True))

                # Format and send message
                logging.info("Formatting message...")
                message = format_func(data_source_name, df)
                logging.info(f"Message length: {len(message)}")
                logging.info("Sample of message:")
                logging.info(message[:500] + "...")  # Show first 500 chars
                
                # Send message in chunks
                max_message_length = 4096
                for i in range(0, len(message), max_message_length):
                    chunk = message[i:i + max_message_length]
                    if chunk.strip():
                        logging.info(f"Sending chunk of length {len(chunk)}")
                        try:
                            send_telegram_message(chunk)
                            logging.info("Chunk sent successfully")
                        except Exception as e:
                            logging.error(f"Error sending telegram message: {e}")
                logging.info(f"{stock_type} message sent successfully")

                # Update database
                try:
                    update_psx_investing_db(df, f'{stock_type.lower()}_stocks')
                    logging.info(f"{stock_type} stocks updated in database")
                except Exception as e:
                    logging.error(f"Error updating database: {e}")
            else:
                if df.empty:
                    logging.info(f"No {stock_type} signals found (DataFrame is empty)")
                else:
                    logging.info(f"No {stock_type} signals found (Volume column missing)")
                    logging.info(f"Available columns: {df.columns}")
                    
        # Handle all signal types
        for data_tuple in all_buy_stock_data:
            handle_stock_data(data_tuple, "Buy", format_buy_signals)
            
        for data_tuple in all_sell_stock_data:
            handle_stock_data(data_tuple, "Sell", format_sell_signals)
            
        for data_tuple in all_neutral_stock_data:
            handle_stock_data(data_tuple, "Neutral", format_neutral_signals)
            
        # Handle breakout signals
        for data_tuple in all_breakout_data:
            handle_breakout_data(data_tuple)

        logging.info("Analysis complete")
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
