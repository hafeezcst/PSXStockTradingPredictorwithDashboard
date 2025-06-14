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
    """Fetch both KMI30 and KMI100 tables from the database"""
    try:
        # Read KMI30 and KMI100 symbols from Excel file
        symbols_file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        
        # Read KMI30 symbols
        kmi30_df = pd.read_excel(symbols_file_path, sheet_name='KMI30')
        KMI30_symbols = set(kmi30_df.iloc[:, 0].tolist()[:30])  # Convert to set for faster lookups and limit to top 30
        
        # Read KMI100 symbols
        kmi100_df = pd.read_excel(symbols_file_path, sheet_name='KMI100')
        KMI100_symbols = set(kmi100_df.iloc[:, 0].tolist()[:100])  # Convert to set for faster lookups and limit to top 100
        
        # Combine both sets of symbols
        all_symbols = KMI30_symbols.union(KMI100_symbols)
        
        # Get all tables from database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Filter tables that match either KMI30 or KMI100 symbols
        return [table[0] for table in tables 
                if table[0].startswith('PSX_') 
                and table[0].endswith('_stock_data')
                and table[0].replace('PSX_', '').replace('_stock_data', '').strip().upper() in all_symbols]
    except Exception as e:
        logging.error(f"Error fetching KMI30 and KMI100 table names: {e}")
        return []


# Function to fetch stock data for the last N dates (default 30 for breakout detection)
def fetch_stock_data(cursor, table_name, limit=30):
    """Fetch stock data for the last N dates (default 30 for breakout detection)"""
    try:
        query = f"""
            SELECT Date, Close, Volume, RSI_Weekly_Avg, RSI_Monthly, RSI_3Months_Avg, RSI_Monthly_Avg, AO_weekly, MA_30, pct_change 
            FROM {table_name} 
            ORDER BY Date DESC 
            LIMIT {limit};
        """
        cursor.execute(query)
        results = cursor.fetchall()
        if len(results) < 2:
            return None
        return results
    except Exception as e:
        logging.error(f"Error fetching stock data for {table_name}: {e}")
        return None


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) using Close prices."""
    close = df['Close']
    
    # Calculate price changes
    price_changes = close.diff().abs()
    
    # Calculate ATR using price changes
    atr = price_changes.rolling(window=period).mean()
    
    return atr

def identify_weekly_breakouts(df: pd.DataFrame) -> Dict:
    """
    Identify bullish and bearish breakouts on weekly timeframe with improved logic.
    """
    breakout_signals = {
        'bullish_breakout': [],
        'bearish_breakout': []
    }
    
    try:
        # Log the input data size
        logging.info(f"Running breakout detection with {len(df)} data points")
        
        if len(df) < 20:  # Increased minimum data points for better trend analysis
            logging.warning(f"Not enough data points for reliable breakout detection. Need at least 20, got {len(df)}")
            return {'bullish_breakout': [], 'bearish_breakout': []}
        
        # Calculate additional indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['ATR'] = calculate_atr(df, period=14)
        df['Upper_Band'] = df['MA_30'] + (2 * df['ATR'])  # Increased band width for stronger signals
        df['Lower_Band'] = df['MA_30'] - (2 * df['ATR'])
        
        # Calculate price momentum and trend indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_MA_5'] = df['Close'].rolling(window=5).mean()
        df['Price_MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        
        # Calculate trend strength
        df['Trend_Strength'] = (df['Price_MA_5'] - df['Price_MA_20']) / df['Price_MA_20']
        
        # Log after calculations to check for NaN values
        logging.info(f"NaN in Volume_MA: {df['Volume_MA'].isna().sum()}")
        logging.info(f"NaN in ATR: {df['ATR'].isna().sum()}")
        logging.info(f"NaN in Price_Change: {df['Price_Change'].isna().sum()}")
        
        # Bullish Breakout Conditions
        def check_bullish_breakout(row, prev_row):
            try:
                if pd.isna(row['Close']) or pd.isna(row['Volume']) or pd.isna(row['RSI_Weekly_Avg']):
                    return {'is_breakout': False, 'conditions': {}, 'strength': 0}
                
                # Price momentum conditions
                price_momentum = row['Price_Change'] > 0.02  # Increased to 2% price increase
                price_above_ma = row['Close'] > row['Price_MA_20']  # Price above 20-period MA
                price_trend = row['Trend_Strength'] > 0.01  # Positive trend strength
                
                # Volume conditions
                volume_momentum = row['Volume_Change'] > 0.5  # Increased to 50% volume increase
                volume_above_ma = row['Volume'] > row['Volume_MA_5']  # Volume above 5-period MA
                
                # Technical indicator conditions
                rsi_momentum = row['RSI_Weekly_Avg'] > 50  # RSI above 50
                ao_momentum = row['AO_weekly'] > 0  # Positive AO
                
                # Breakout confirmation
                price_breakout = row['Close'] > row['Upper_Band']  # Price breaks above upper band
                
                conditions = {
                    'price_momentum': price_momentum,
                    'price_above_ma': price_above_ma,
                    'price_trend': price_trend,
                    'volume_momentum': volume_momentum,
                    'volume_above_ma': volume_above_ma,
                    'rsi_momentum': rsi_momentum,
                    'ao_momentum': ao_momentum,
                    'price_breakout': price_breakout
                }
                
                # Count how many conditions are met
                conditions_met = sum(conditions.values())
                
                # Require at least 5 conditions to be met for a valid breakout
                return {
                    'is_breakout': conditions_met >= 5,
                    'conditions': conditions,
                    'strength': conditions_met / len(conditions)
                }
            except Exception as e:
                logging.error(f"Error in bullish breakout check: {e}")
                return {'is_breakout': False, 'conditions': {}, 'strength': 0}
        
        # Bearish Breakout Conditions
        def check_bearish_breakout(row, prev_row):
            try:
                if pd.isna(row['Close']) or pd.isna(row['Volume']) or pd.isna(row['RSI_Weekly_Avg']):
                    return {'is_breakout': False, 'conditions': {}, 'strength': 0}
                
                # Price momentum conditions
                price_momentum = row['Price_Change'] < -0.02  # Increased to 2% price decrease
                price_below_ma = row['Close'] < row['Price_MA_20']  # Price below 20-period MA
                price_trend = row['Trend_Strength'] < -0.01  # Negative trend strength
                
                # Volume conditions
                volume_momentum = row['Volume_Change'] > 0.5  # Increased to 50% volume increase
                volume_above_ma = row['Volume'] > row['Volume_MA_5']  # Volume above 5-period MA
                
                # Technical indicator conditions
                rsi_momentum = row['RSI_Weekly_Avg'] < 50  # RSI below 50
                ao_momentum = row['AO_weekly'] < 0  # Negative AO
                
                # Breakout confirmation
                price_breakout = row['Close'] < row['Lower_Band']  # Price breaks below lower band
                
                conditions = {
                    'price_momentum': price_momentum,
                    'price_below_ma': price_below_ma,
                    'price_trend': price_trend,
                    'volume_momentum': volume_momentum,
                    'volume_above_ma': volume_above_ma,
                    'rsi_momentum': rsi_momentum,
                    'ao_momentum': ao_momentum,
                    'price_breakout': price_breakout
                }
                
                # Count how many conditions are met
                conditions_met = sum(conditions.values())
                
                # Require at least 5 conditions to be met for a valid breakout
                return {
                    'is_breakout': conditions_met >= 5,
                    'conditions': conditions,
                    'strength': conditions_met / len(conditions)
                }
            except Exception as e:
                logging.error(f"Error in bearish breakout check: {e}")
                return {'is_breakout': False, 'conditions': {}, 'strength': 0}
        
        # Process each week's data - skip first 20 rows for accurate rolling calculations
        breakout_count_bull = 0
        breakout_count_bear = 0
        
        # Skip the first 20 rows if we have enough data
        start_idx = 20 if len(df) > 20 else 1
        
        for i in range(start_idx, len(df)):
            current_row = df.iloc[i]
            previous_row = df.iloc[i-1]
            
            # Check for breakouts
            bullish_signal = check_bullish_breakout(current_row, previous_row)
            bearish_signal = check_bearish_breakout(current_row, previous_row)
            
            # Log key info for debugging (every 5th row)
            if i % 5 == 0:
                bull_conditions = sum(bullish_signal['conditions'].values())
                bear_conditions = sum(bearish_signal['conditions'].values())
                logging.info(f"Row {i}: Bull={bull_conditions}, Bear={bear_conditions}, RSI={current_row['RSI_Weekly_Avg']:.1f}, AO={current_row['AO_weekly']:.1f}")
            
            if bullish_signal['is_breakout']:
                breakout_count_bull += 1
                breakout_signals['bullish_breakout'].append({
                    'date': current_row['Date'],
                    'price': current_row['Close'],
                    'volume': current_row['Volume'],
                    'rsi': current_row['RSI_Weekly_Avg'],
                    'ao': current_row['AO_weekly'],
                    'strength': bullish_signal['strength'],
                    'conditions_met': [k for k, v in bullish_signal['conditions'].items() if v],
                    'price_change': current_row['Price_Change'] * 100 if not pd.isna(current_row['Price_Change']) else 0,
                    'volume_change': current_row['Volume_Change'] * 100 if not pd.isna(current_row['Volume_Change']) else 0,
                    'trend_strength': current_row['Trend_Strength'] * 100 if not pd.isna(current_row['Trend_Strength']) else 0
                })
                
            if bearish_signal['is_breakout']:
                breakout_count_bear += 1
                breakout_signals['bearish_breakout'].append({
                    'date': current_row['Date'],
                    'price': current_row['Close'],
                    'volume': current_row['Volume'],
                    'rsi': current_row['RSI_Weekly_Avg'],
                    'ao': current_row['AO_weekly'],
                    'strength': bearish_signal['strength'],
                    'conditions_met': [k for k, v in bearish_signal['conditions'].items() if v],
                    'price_change': current_row['Price_Change'] * 100 if not pd.isna(current_row['Price_Change']) else 0,
                    'volume_change': current_row['Volume_Change'] * 100 if not pd.isna(current_row['Volume_Change']) else 0,
                    'trend_strength': current_row['Trend_Strength'] * 100 if not pd.isna(current_row['Trend_Strength']) else 0
                })
        
        logging.info(f"Breakout detection complete. Found {breakout_count_bull} bullish and {breakout_count_bear} bearish breakouts")
        return breakout_signals
        
    except Exception as e:
        logging.error(f"Error in identify_weekly_breakouts: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {'bullish_breakout': [], 'bearish_breakout': []}

def format_breakout_message(breakout_data: Dict, symbol: str) -> str:
    """Format breakout signals for Telegram message."""
    message = ""
    
    if breakout_data['bullish_breakout']:
        message += f"🚀 *{symbol} - BULLISH BREAKOUT DETECTED* 🚀\n\n"
        for breakout in breakout_data['bullish_breakout']:
            message += f"📅 Date: {breakout['date']}\n"
            message += f"💰 Price: {breakout['price']:.2f}\n"
            message += f"📈 Price Change: {breakout['price_change']:+.2f}%\n"
            message += f"📊 RSI: {breakout['rsi']:.2f}\n"
            message += f"📈 AO: {breakout['ao']:.2f}\n"
            message += f"💸 Volume: {breakout['volume']:,.0f}\n"
            message += f"📊 Volume Change: {breakout['volume_change']:+.2f}%\n"
            message += f"💪 Strength: {breakout['strength']*100:.1f}%\n"
            message += f"✅ Conditions Met:\n"
            for condition in breakout['conditions_met']:
                message += f"  • {condition.replace('_', ' ').title()}\n"
            message += "\n"
    
    if breakout_data['bearish_breakout']:
        message += f"📉 *{symbol} - BEARISH BREAKOUT DETECTED* 📉\n\n"
        for breakout in breakout_data['bearish_breakout']:
            message += f"📅 Date: {breakout['date']}\n"
            message += f"💰 Price: {breakout['price']:.2f}\n"
            message += f"📉 Price Change: {breakout['price_change']:+.2f}%\n"
            message += f"📊 RSI: {breakout['rsi']:.2f}\n"
            message += f"📉 AO: {breakout['ao']:.2f}\n"
            message += f"💸 Volume: {breakout['volume']:,.0f}\n"
            message += f"📊 Volume Change: {breakout['volume_change']:+.2f}%\n"
            message += f"💪 Strength: {breakout['strength']*100:.1f}%\n"
            message += f"✅ Conditions Met:\n"
            for condition in breakout['conditions_met']:
                message += f"  • {condition.replace('_', ' ').title()}\n"
            message += "\n"
    
    return message

def process_stock_data(table_name, results, cursor, multibagger_symbols, data_source):
    stock_data = []
    sell_stock_data = []
    neutral_stock_data = []
    breakout_data = None
    if not results or len(results) < 2:
        return [], [], [], None
    try:
        # Use the latest two rows for buy/sell/neutral logic
        latest = results[0]
        previous = results[1]
        (date_latest, close_latest, volume_latest, rsi_weekly_latest, 
         rsi_monthly_latest, rsi_3months_latest, rsi_monthly_avg_latest, 
         ao_weekly_latest, ma_30_latest, pct_change_latest) = latest
        (_, _, _, _, rsi_monthly_previous, rsi_3months_previous, 
         _, ao_weekly_previous, ma_30_previous, _) = previous
        stock_name = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
        ao_change_date, ao_change_close = get_ao_change_date(cursor, table_name)
        freefloatratio = get_freefloatratio(stock_name)
        multibagger = stock_name in multibagger_symbols
        truncated_data_source = data_source.split('_')[4].split('.')[0]
        p_l = 0.0
        holding_days = 0
        if ao_change_date and ao_change_close:
            p_l = round(((close_latest - ao_change_close) / ao_change_close) * 100, 2)
            holding_days = (pd.to_datetime(date_latest.split(' ')[0]) - pd.to_datetime(ao_change_date)).days
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
        # Use all available rows for breakout detection
        df_columns = ['Date', 'Close', 'Volume', 'RSI_Weekly_Avg', 
                      'RSI_Monthly', 'RSI_3Months_Avg', 'RSI_Monthly_Avg', 
                      'AO_weekly', 'MA_30', 'pct_change']
        df = pd.DataFrame(results, columns=df_columns)
        df = df.iloc[::-1].reset_index(drop=True)  # Chronological order
        try:
            breakout_data = identify_weekly_breakouts(df)
            if not breakout_data['bullish_breakout'] and not breakout_data['bearish_breakout']:
                breakout_data = None
        except Exception as e:
            logging.error(f"Error calculating breakouts for {stock_name}: {e}")
            breakout_data = None
        # Buy Condition
        if (rsi_3months_latest is not None and rsi_3months_latest >= 40 and
            rsi_weekly_latest is not None and rsi_weekly_latest >= 40 and
            ao_weekly_latest is not None and ao_weekly_latest >= 0 and
            volume_latest is not None and volume_latest > 5000):
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
              close_latest <= ma_30_latest and
              volume_latest is not None and volume_latest > 0):
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
        return stock_data, sell_stock_data, neutral_stock_data, breakout_data
    except Exception as e:
        logging.error(f"Error processing {table_name}: {e}")
        return [], [], [], None


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
    all_breakout_data = []

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
                    except Exception as e:
                        logging.error(f"Error processing {table_name}: {e}")

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
            logging.error(f"Error processing database {db_path}: {e}")
            
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

        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"🟢 *{symbol}*{KMI_tag} - BUY_SIGNAL\n"
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
        
        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"🔴 *{symbol}*{KMI_tag} - SELL_SIGNAL\n"
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
    
    # Sort by RSI in descending order
    df = df.sort_values(by='RSI_Weekly_Avg', ascending=False)
    
    for _, row in df.iterrows():
        symbol = row['Stock']
        close = row['Close']
        rsi = row['RSI_Weekly_Avg']
        ao = row['AO_Weekly']
        trend = row['Trend_Direction']
        current_price_date = row['Date']
        
        # Get KMI tag based on indices
        KMI_tag = get_kmi_tag(symbol, KMI30_symbols, KMI100_symbols)
        
        # Add trend indicator emoji
        trend_emoji = "🟢" if trend == "Bullish" else "🔴"
        
        message += f"🕒 Analysis Time: {analysis_time}\n"
        message += f"📅 DataBase Update Date: {current_price_date}\n\n"
        message += f"🟡 {trend_emoji} *{symbol}*{KMI_tag} - NEUTRAL_SIGNAL\n"
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
    timeout = (5, 15)  # Reduced timeouts: (connect timeout, read timeout) in seconds
    
    for chunk in messages:
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Create a new session for each attempt
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                
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
            finally:
                if 'session' in locals():
                    session.close()
    
    if success:
        logging.info("Successfully sent all message chunks to Telegram")
    else:
        logging.error("Failed to send some message chunks to Telegram")
    
    return success

def handle_breakout_data(data_tuple):
    data_source_name, breakout_data, filtered_symbols = data_tuple
    
    if not breakout_data:
        logging.info("No breakout signals found")
        return
        
    logging.info(f"\nProcessing breakout signals for {data_source_name}...")
    
    for table_name, breakout in breakout_data:
        symbol = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
        
        # Format breakout message
        message = format_breakout_message(breakout, symbol)
        
        if message:
            logging.info(f"Sending breakout message for {symbol}")
            try:
                send_telegram_message(message)
                logging.info(f"Breakout message sent successfully for {symbol}")
            except Exception as e:
                logging.error(f"Error sending breakout message for {symbol}: {e}")

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