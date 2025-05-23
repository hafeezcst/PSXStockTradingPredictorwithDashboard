import sqlite3
import pandas as pd
from tabulate import tabulate
import os
import logging
from datetime import datetime
import requests
import time
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.system('cls' if os.name == 'nt' else 'clear')
# Telegram configuration
TELEGRAM_BOT_TOKEN = '6860197701:AAESTzERZLYbqyU6gFKfAwJQL8jJ_HNKLbM'
TELEGRAM_CHAT_ID = '-4152327824'

# Function to get the free float ratio from psxsymbols.db
def get_freefloatratio(symbol):
    try:
        with sqlite3.connect('data/databases/production/psxsymbols.db') as conn:
            cursor = conn.cursor()
            query = "SELECT freefloatratio FROM KMIALL WHERE symbol = ?;"
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
    except sqlite3.OperationalError as e:
        logging.error(f"Error fetching freefloatratio for {symbol}: {e}")
        return None


# Function to get the date when AO changed from negative to positive
def get_ao_change_date(cursor, table_name):
    query = f"SELECT Date, Close, AO_weekly FROM {table_name} ORDER BY Date DESC;"
    cursor.execute(query)
    results = cursor.fetchall()

    previous_ao = None
    for date, close, ao_weekly in results:
        if previous_ao is not None and ao_weekly < 0 <= previous_ao:
            return date.split(' ')[0], close  # Only fetch date part
        previous_ao = ao_weekly
    return None, None


# Function to fetch multibagger symbols from the database
def get_multibagger_symbols():
    try:
        with sqlite3.connect('data/databases/production/psxsymbols.db') as conn:
            cursor = conn.cursor()
            query = "SELECT symbol FROM ROIC_GT_25;"
            cursor.execute(query)
            results = cursor.fetchall()
            return [row[0].strip().upper() for row in results]
    except sqlite3.OperationalError as e:
        logging.error(f"Error fetching multibagger symbols: {e}")
        return []


# Function to fetch table names from the database
def fetch_table_names(cursor):
    """Fetch only KMI100 tables from the database"""
    try:
        # Read KMI30 symbols from Excel file
        symbols_file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        symbols_df = pd.read_excel(symbols_file_path, sheet_name='KMIALL')
        KMI30_symbols = set(symbols_df.iloc[:, 0].tolist())  # Convert to set for faster lookups
        
        # Get all tables from database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Filter tables that match KMI30 symbols
        return [table[0] for table in tables 
                if table[0].startswith('PSX_') 
                and table[0].endswith('_stock_data')
                and table[0].replace('PSX_', '').replace('_stock_data', '').strip().upper() in KMI30_symbols]
    except Exception as e:
        logging.error(f"Error fetching KMI30 table names: {e}")
        return []


# Function to fetch stock data for the last two dates
def fetch_stock_data(cursor, table_name):
    """Fetch stock data for the last two dates"""
    try:
        query = f"""
            SELECT Date, Close, Volume, RSI_Weekly_Avg, RSI_Monthly, RSI_3Months_Avg, RSI_Monthly_Avg, AO_weekly, MA_30, pct_change 
            FROM {table_name} 
            WHERE Date IN (
                SELECT Date 
                FROM {table_name} 
                ORDER BY Date DESC 
                LIMIT 2
            )
            ORDER BY Date DESC;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        if len(results) != 2:
            return None
            
        return results
        
    except Exception as e:
        logging.error(f"Error fetching stock data for {table_name}: {e}")
        return None


# Function to process stock data and filter based on specific criteria for buy, sell, and neutral conditions
def process_stock_data(table_name, results, cursor, multibagger_symbols, data_source):
    stock_data = []
    sell_stock_data = []
    neutral_stock_data = []
    
    if not results or len(results) != 2:
        return [], [], []
        
    try:
        latest, previous = results
        
        # Unpack latest row
        (date_latest, close_latest, volume_latest, rsi_weekly_latest, 
         rsi_monthly_latest, rsi_3months_latest, rsi_monthly_avg_latest, 
         ao_weekly_latest, ma_30_latest, pct_change_latest) = latest
        
        # Unpack previous row
        (_, _, _, _, rsi_monthly_previous, rsi_3months_previous, 
         _, ao_weekly_previous, ma_30_previous, _) = previous

        # Extract stock name and common data
        stock_name = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
        ao_change_date, ao_change_close = get_ao_change_date(cursor, table_name)
        freefloatratio = get_freefloatratio(stock_name)
        multibagger = stock_name in multibagger_symbols
        truncated_data_source = data_source.split('_')[4].split('.')[0]
        
        # Calculate P/L and holding days
        p_l = 0.0
        holding_days = 0
        if ao_change_date and ao_change_close:
            p_l = round(((close_latest - ao_change_close) / ao_change_close) * 100, 2)
            holding_days = (pd.to_datetime(date_latest.split(' ')[0]) - pd.to_datetime(ao_change_date)).days
            
        # Base data dictionary
        base_data = {
            'Stock': stock_name,
            'Data Source': truncated_data_source,
            'Date': date_latest.split(' ')[0],
            'Close': close_latest,
            'Volume': volume_latest,
            'RSI_Weekly_Avg': rsi_weekly_latest,
            'RSI_3Months_Avg_Recent': rsi_3months_latest,
            'AO_Weekly': ao_weekly_latest,
            'MA_30': ma_30_latest,
            'Multibagger': 'Yes' if multibagger else 'No',
            'FreeFloatRatio': freefloatratio
        }
        
        # Buy Condition
        if (rsi_3months_latest is not None and rsi_3months_latest >= 40 and
            rsi_weekly_latest is not None and rsi_weekly_latest >= 40 and
            ao_weekly_latest is not None and ao_weekly_latest >= 0 and
            volume_latest is not None and volume_latest > 500):
            
            buy_data = base_data.copy()
            buy_data.update({
                'Success': 'Yes' if close_latest >= ao_change_close else 'No',
                '% P/L': p_l,
                'Signal_Date': ao_change_date,
                'Signal_Close': ao_change_close,
                'Holding_Days': holding_days,
                'Status': 'Buy'
            })
            stock_data.append(buy_data)
            
        # Sell Condition
        elif (rsi_monthly_latest is not None and rsi_monthly_latest <= 50 and
              rsi_weekly_latest is not None and rsi_weekly_latest <= 50 and
              ao_weekly_latest is not None and ao_weekly_latest <= 0 and 
              ma_30_latest is not None and
              # Proper sell condition: price below moving average
              close_latest <= ma_30_latest and
              # More selective volume filter
              volume_latest is not None and volume_latest > 500):
              
            sell_data = base_data.copy()
            sell_data.update({
                'Success': 'Yes' if close_latest < ao_change_close else 'No',
                '% P/L': round(((ao_change_close - close_latest) / close_latest) * 100, 2),
                'Signal_Date': ao_change_date,
                'Signal_Close': ao_change_close,
                'Holding_Days': holding_days,
                'Status': 'Sell'
            })
            sell_stock_data.append(sell_data)
            
        # Neutral Condition
        else:
            neutral_data = base_data.copy()
            neutral_data.update({
                'Trend_Direction': 'Bullish' if ma_30_latest > ma_30_previous and ao_weekly_latest > ao_weekly_previous else 'Bearish',
                'Status': 'Neutral'
            })
            neutral_stock_data.append(neutral_data)
            
        return stock_data, sell_stock_data, neutral_stock_data
        
    except Exception as e:
        logging.error(f"Error processing {table_name}: {e}")
        return [], [], []


def update_psx_investing_db(data, table_name):
    """
    Update the buy, sell, or neutral stock tables in PSX_investing_Stocks_KMI30.db
    
    Args:
        data: DataFrame containing stock data
        table_name: Name of the table to update (buy_stocks, sell_stocks, etc.)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create database if it doesn't exist
        with sqlite3.connect('data/databases/production/PSX_investing_Stocks_KMI30.db') as conn:
            # Add 'Update_Date' column to the data
            data['Update_Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check for and remove potential duplicates based on Stock and Date
            if 'Stock' in data.columns and 'Date' in data.columns:
                # Get existing data
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
                except:
                    # Table might not exist yet
                    pass

            # Append new data to the specific table
            data.to_sql(table_name, conn, if_exists='append', index=False)
            logging.info(f"Added {len(data)} new records to {table_name}")
            return True
    except Exception as e:
        logging.error(f"Error updating {table_name}: {e}")
        return False


# Main function to get stock data with RSI above 40
def get_stock_data_with_rsi_above_40(db_paths):
    all_buy_stock_data = []
    all_sell_stock_data = []
    all_neutral_stock_data = []

    for db_path in db_paths:
        logging.info(f"\nProcessing database: {db_path}")
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                multibagger_symbols = get_multibagger_symbols()
                logging.info(f"Found {len(multibagger_symbols)} multibagger symbols")
                
                tables = fetch_table_names(cursor)
                logging.info(f"Found {len(tables)} tables to process")
                
                buy_stock_data = []
                sell_stock_data = []
                neutral_stock_data = []
                filtered_symbols = []

                for table_name in tables:
                    try:
                        logging.info(f"\nProcessing table: {table_name}")
                        results = fetch_stock_data(cursor, table_name)
                        if results:
                            logging.info(f"Found {len(results)} results for {table_name}")
                            buy_data, sell_data, neutral_data = process_stock_data(
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
                            
                            filtered_symbols.append(
                                table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
                            )
                        else:
                            logging.warning(f"No results found for {table_name}")
                    except sqlite3.OperationalError as e:
                        logging.error(f"Database error processing {table_name}: {e}")
                    except Exception as e:
                        logging.error(f"Error processing {table_name}: {e}")

                # Use KMI30 as the data source
                data_source_name = 'KMI30'
                logging.info(f"\nData Source: {data_source_name}")
                logging.info(f"Total buy signals: {len(buy_stock_data)}")
                logging.info(f"Total sell signals: {len(sell_stock_data)}")
                logging.info(f"Total neutral signals: {len(neutral_stock_data)}")
                
                all_buy_stock_data.append((data_source_name, buy_stock_data, filtered_symbols))
                all_sell_stock_data.append((data_source_name, sell_stock_data, filtered_symbols))
                all_neutral_stock_data.append((data_source_name, neutral_stock_data, filtered_symbols))
        except Exception as e:
            logging.error(f"Error processing database {db_path}: {e}")
            
    return all_buy_stock_data, all_sell_stock_data, all_neutral_stock_data


def get_KMI30_symbols():
    """Get list of KMI30 symbols from Excel file."""
    try:
        symbols_file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        symbols_df = pd.read_excel(symbols_file_path, sheet_name='KMIALL')
        return set(symbols_df.iloc[:, 0].tolist())
    except Exception as e:
        logging.error(f"Error reading KMI30 symbols: {e}")
        return set()

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
    """Generate an AI description for buy signals based on technical indicators"""
    strength = "Strong" if row['RSI_Weekly_Avg'] >= 60 else "Moderate"
    momentum = "increasing" if row['AO_Weekly'] > 0 else "steady"
    rsi_status = "bullish territory" if row['RSI_Weekly_Avg'] >= 50 else "neutral territory"
    volume_strength = "high" if row['Volume'] > 100000 else "moderate"
    
    if row['Close'] > row['MA_30']:
        trend_message = f"Price {row['Close']:.2f} is above MA30 {row['MA_30']:.2f}, confirming upward trend"
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
    
    # Get KMI30 symbols
    KMI30_symbols = get_KMI30_symbols()
    
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
        
        # Check if stock is in KMI30
        KMI30_tag = " (KMI30)" if symbol in KMI30_symbols else ""
        
        # Calculate P/L only if we have a valid signal price
        pl_text = ""
        if signal_price and signal_price > 0:
            pl = ((close - signal_price) / signal_price * 100)
            pl_text = f"📊 P/L: {pl:+.2f}%\n"

        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"*{symbol}*{KMI30_tag} - {company}\n"
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
    """Generate an AI description for sell signals based on technical indicators"""
    if row['RSI_Weekly_Avg'] <= 30:
        condition = "oversold"
        action = "potential reversal"
    else:
        condition = "weakening"
        action = "downward pressure"
    
    momentum = "decreasing" if row['AO_Weekly'] < 0 else "mixed"
    volume_strength = "high" if row['Volume'] > 100000 else "moderate"
    if row['Close'] > row['MA_30']:
        trend_message = f"Price {row['Close']:.2f} is above MA30 {row['MA_30']:.2f}, confirming upward trend"
    else:
        trend_message = f"Price {row['Close']:.2f} is below MA30 {row['MA_30']:.2f}, showing potential support level"    
    
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
    
    # Get KMI30 symbols
    KMI30_symbols = get_KMI30_symbols()
    
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
        
        # Check if stock is in KMI30
        KMI30_tag = " (KMI30)" if symbol in KMI30_symbols else ""
        
        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"*{symbol}*{KMI30_tag}\n"
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
    message = f"⚪️ *{data_source_name} Neutral Signals* 📊\n\n"
    
    # Get KMI30 symbols
    KMI30_symbols = get_KMI30_symbols()
    
    # Sort by RSI in descending order
    df = df.sort_values(by='RSI_Weekly_Avg', ascending=False)
    
    for _, row in df.iterrows():
        symbol = row['Stock']
        close = row['Close']
        rsi = row['RSI_Weekly_Avg']
        ao = row['AO_Weekly']
        trend = row['Trend_Direction']
        current_price_date = row['Date']
        
        # Check if stock is in KMI30
        KMI30_tag = " (KMI30)" if symbol in KMI30_symbols else ""
        
        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"*{symbol}*{KMI30_tag}\n"
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
    """Send message to Telegram channel with rate limit handling."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Telegram bot token or chat ID not configured")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Escape special characters for MarkdownV2
    escaped_message = (message
        .replace('_', '\\_')
        .replace('*', '\\*')
        .replace('[', '\\[')
        .replace(']', '\\]')
        .replace('(', '\\(')
        .replace(')', '\\)')
        .replace('~', '\\~')
        .replace('`', '\\`')
        .replace('>', '\\>')
        .replace('#', '\\#')
        .replace('+', '\\+')
        .replace('-', '\\-')
        .replace('=', '\\=')
        .replace('|', '\\|')
        .replace('{', '\\{')
        .replace('}', '\\}')
        .replace('.', '\\.')
        .replace('!', '\\!')
    )
    
    # Split message into chunks if too long
    max_length = 4096
    messages = [escaped_message[i:i+max_length] for i in range(0, len(escaped_message), max_length)]
    
    success = True
    base_delay = 2  # Base delay in seconds
    max_retries = 5  # Maximum number of retries per message
    
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
                
                response = requests.post(url, json=payload)
                
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
                
            except requests.RequestException as e:
                if response and response.status_code == 429:
                    retry_count += 1
                    if retry_count < max_retries:
                        continue
                logging.error(f"Error sending message to Telegram: {e}")
                if response:
                    logging.error(f"Response content: {response.text}")
                success = False
                break
    
    if success:
        logging.info("Successfully sent all message chunks to Telegram")
    else:
        logging.error("Failed to send some message chunks to Telegram")
    
    return success

# Main execution
if __name__ == "__main__":
    try:
        logging.info("\nRunning stock analysis...")
        db_paths = ['data/databases/production/psx_consolidated_data_indicators_PSX.db']  # Fixed case sensitivity
        all_buy_stock_data, all_sell_stock_data, all_neutral_stock_data = get_stock_data_with_rsi_above_40(db_paths)

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

        logging.info("Analysis complete")
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")