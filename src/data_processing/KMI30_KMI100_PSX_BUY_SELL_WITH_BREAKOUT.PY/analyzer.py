import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from .data_fetcher import get_dividend_info, get_ao_change_date, get_freefloatratio
from . import db_handler

logger = logging.getLogger(__name__)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) using Close prices."""
    close = df['Close']
    price_changes = close.diff().abs()
    atr = price_changes.rolling(window=period).mean()
    return atr

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Calculate dynamic support and resistance levels using rolling min and max."""
    support = df['Close'].rolling(window=window).min()
    resistance = df['Close'].rolling(window=window).max()
    return support, resistance

def identify_weekly_breakouts(df: pd.DataFrame) -> Dict:
    """Identify bullish and bearish breakouts with enhanced robust logic for weekly timeframe."""
    breakout_signals = {'bullish_breakout_weekly': [], 'bearish_breakout_weekly': []}
    
    try:
        if len(df) < 25:
            logger.warning(f"Insufficient data points ({len(df)}) for breakout detection")
            return breakout_signals
        
        # Calculate indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['ATR'] = calculate_atr(df, period=14)
        df['Upper_Band'] = df['MA_30'] + (2.5 * df['ATR'])
        df['Lower_Band'] = df['MA_30'] - (2.5 * df['ATR'])
        df['Support'], df['Resistance'] = calculate_support_resistance(df, window=20)
        
        # Momentum indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_MA_5'] = df['Close'].rolling(window=5).mean()
        df['Price_MA_20'] = df['Close'].rolling(window=20).mean()
        df['Price_MA_50'] = df['Close'].rolling(window=50).mean()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Trend_Strength'] = (df['Price_MA_5'] - df['Price_MA_20']) / df['Price_MA_20']
        df['Long_Term_Trend'] = (df['Price_MA_20'] - df['Price_MA_50']) / df['Price_MA_50']
        
        # Weekly closing price comparison (designating Friday as the standard last trading day of the week)
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
        df['Week'] = df['Date'].dt.to_period('W-FRI')  # Week ending on Friday
        # Group by week and get the last trading day (Friday or the last available day before Friday if Friday is a holiday)
        weekly_closes = df.groupby('Week').apply(
            lambda x: x.loc[x['DayOfWeek'].idxmax(), 'Close'] if not x.empty else np.nan
        ).shift(1)
        df['Weekly_Close'] = df['Week'].map(weekly_closes).ffill()
        
        # Bullish Breakout Conditions for Weekly Timeframe
        def check_bullish_breakout(row):
            conditions = {
                'price_momentum': row['Price_Change'] > 0.025,
                'price_above_ma': row['Close'] > row['Price_MA_20'],
                'price_trend': row['Trend_Strength'] > 0.015,
                'long_term_trend': row['Long_Term_Trend'] > 0.01,
                'volume_momentum': row['Volume_Change'] > 0.6,
                'volume_above_ma': row['Volume'] > row['Volume_MA_5'] * 1.2,
                'rsi_momentum': row['RSI_Weekly_Avg'] > 55,
                'ao_momentum': row['AO_weekly'] > 0,
                'resistance_break': row['Close'] > row['Resistance'] if not pd.isna(row['Resistance']) else False,
                'price_breakout': row['Close'] > row['Upper_Band'],
                'weekly_price_break': row['Close'] > row['Weekly_Close'] if not pd.isna(row['Weekly_Close']) else False
            }
            conditions_met = sum(conditions.values())
            return conditions_met >= 7, conditions
        
        # Bearish Breakout Conditions for Weekly Timeframe
        def check_bearish_breakout(row):
            conditions = {
                'price_momentum': row['Price_Change'] < -0.025,
                'price_below_ma': row['Close'] < row['Price_MA_20'],
                'price_trend': row['Trend_Strength'] < -0.015,
                'long_term_trend': row['Long_Term_Trend'] < -0.01,
                'volume_momentum': row['Volume_Change'] > 0.6,
                'volume_above_ma': row['Volume'] > row['Volume_MA_5'] * 1.2,
                'rsi_momentum': row['RSI_Weekly_Avg'] < 45,
                'ao_momentum': row['AO_weekly'] < 0,
                'support_break': row['Close'] < row['Support'] if not pd.isna(row['Support']) else False,
                'price_breakout': row['Close'] < row['Lower_Band'],
                'weekly_price_break': row['Close'] < row['Weekly_Close'] if not pd.isna(row['Weekly_Close']) else False
            }
            conditions_met = sum(conditions.values())
            return conditions_met >= 7, conditions
        
        # Process each row
        start_idx = 25 if len(df) > 25 else 1
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            
            bullish, bull_conditions = check_bullish_breakout(row)
            bearish, bear_conditions = check_bearish_breakout(row)
            
            if bullish:
                breakout_signals['bullish_breakout_weekly'].append({
                    'date': row['Date'],
                    'current_price': row['Close'],
                    'weekly_close': row['Weekly_Close'] if not pd.isna(row['Weekly_Close']) else row['Close'],
                    'volume': row['Volume'],
                    'rsi': row['RSI_Weekly_Avg'],
                    'ao': row['AO_weekly'],
                    'conditions_met': [k for k, v in bull_conditions.items() if v]
                })
                
            if bearish:
                breakout_signals['bearish_breakout_weekly'].append({
                    'date': row['Date'],
                    'current_price': row['Close'],
                    'weekly_close': row['Weekly_Close'] if not pd.isna(row['Weekly_Close']) else row['Close'],
                    'volume': row['Volume'],
                    'rsi': row['RSI_Weekly_Avg'],
                    'ao': row['AO_weekly'],
                    'conditions_met': [k for k, v in bear_conditions.items() if v]
                })
        
        logger.info(f"Found {len(breakout_signals['bullish_breakout_weekly'])} weekly bullish and {len(breakout_signals['bearish_breakout_weekly'])} weekly bearish breakouts")
        return breakout_signals
        
    except Exception as e:
        logger.error(f"Error in identify_weekly_breakouts: {e}", exc_info=True)
        return {'bullish_breakout_weekly': [], 'bearish_breakout_weekly': []}

def identify_monthly_breakouts(df: pd.DataFrame) -> Dict:
    """Identify bullish and bearish breakouts with enhanced robust logic for monthly timeframe."""
    breakout_signals = {'bullish_breakout_monthly': [], 'bearish_breakout_monthly': []}
    
    try:
        if len(df) < 30:
            logger.warning(f"Insufficient data points ({len(df)}) for monthly breakout detection")
            return breakout_signals
        
        # Calculate indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['ATR'] = calculate_atr(df, period=14)
        df['Upper_Band'] = df['MA_30'] + (2.5 * df['ATR'])
        df['Lower_Band'] = df['MA_30'] - (2.5 * df['ATR'])
        df['Support'], df['Resistance'] = calculate_support_resistance(df, window=30)
        
        # Momentum indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_MA_5'] = df['Close'].rolling(window=5).mean()
        df['Price_MA_20'] = df['Close'].rolling(window=20).mean()
        df['Price_MA_50'] = df['Close'].rolling(window=50).mean()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Trend_Strength'] = (df['Price_MA_5'] - df['Price_MA_20']) / df['Price_MA_20']
        df['Long_Term_Trend'] = (df['Price_MA_20'] - df['Price_MA_50']) / df['Price_MA_50']
        
        # Monthly closing price comparison (using the closing price of the last trading day of each month)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M')
        df['Monthly_Close'] = df.groupby('Month')['Close'].transform(lambda x: x.iloc[-1]).shift(1)
        
        # Bullish Breakout Conditions for Monthly Timeframe
        def check_bullish_breakout(row):
            conditions = {
                'price_momentum': row['Price_Change'] > 0.05,
                'price_above_ma': row['Close'] > row['Price_MA_20'],
                'price_trend': row['Trend_Strength'] > 0.03,
                'long_term_trend': row['Long_Term_Trend'] > 0.02,
                'volume_momentum': row['Volume_Change'] > 0.8,
                'volume_above_ma': row['Volume'] > row['Volume_MA_5'] * 1.5,
                'rsi_momentum': row['RSI_Weekly_Avg'] > 60,
                'ao_momentum': row['AO_weekly'] > 0,
                'resistance_break': row['Close'] > row['Resistance'] if not pd.isna(row['Resistance']) else False,
                'price_breakout': row['Close'] > row['Upper_Band'],
                'monthly_price_break': row['Close'] > row['Monthly_Close'] if not pd.isna(row['Monthly_Close']) else False
            }
            conditions_met = sum(conditions.values())
            return conditions_met >= 7, conditions
        
        # Bearish Breakout Conditions for Monthly Timeframe
        def check_bearish_breakout(row):
            conditions = {
                'price_momentum': row['Price_Change'] < -0.05,
                'price_below_ma': row['Close'] < row['Price_MA_20'],
                'price_trend': row['Trend_Strength'] < -0.03,
                'long_term_trend': row['Long_Term_Trend'] < -0.02,
                'volume_momentum': row['Volume_Change'] > 0.8,
                'volume_above_ma': row['Volume'] > row['Volume_MA_5'] * 1.5,
                'rsi_momentum': row['RSI_Weekly_Avg'] < 40,
                'ao_momentum': row['AO_weekly'] < 0,
                'support_break': row['Close'] < row['Support'] if not pd.isna(row['Support']) else False,
                'price_breakout': row['Close'] < row['Lower_Band'],
                'monthly_price_break': row['Close'] < row['Monthly_Close'] if not pd.isna(row['Monthly_Close']) else False
            }
            conditions_met = sum(conditions.values())
            return conditions_met >= 7, conditions
        
        # Process each row
        start_idx = 30 if len(df) > 30 else 1
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            
            bullish, bull_conditions = check_bullish_breakout(row)
            bearish, bear_conditions = check_bearish_breakout(row)
            
            if bullish:
                breakout_signals['bullish_breakout_monthly'].append({
                    'date': row['Date'],
                    'current_price': row['Close'],
                    'monthly_close': row['Monthly_Close'] if not pd.isna(row['Monthly_Close']) else row['Close'],
                    'volume': row['Volume'],
                    'rsi': row['RSI_Weekly_Avg'],
                    'ao': row['AO_weekly'],
                    'conditions_met': [k for k, v in bull_conditions.items() if v]
                })
                
            if bearish:
                breakout_signals['bearish_breakout_monthly'].append({
                    'date': row['Date'],
                    'current_price': row['Close'],
                    'monthly_close': row['Monthly_Close'] if not pd.isna(row['Monthly_Close']) else row['Close'],
                    'volume': row['Volume'],
                    'rsi': row['RSI_Weekly_Avg'],
                    'ao': row['AO_weekly'],
                    'conditions_met': [k for k, v in bear_conditions.items() if v]
                })
        
        logger.info(f"Found {len(breakout_signals['bullish_breakout_monthly'])} monthly bullish and {len(breakout_signals['bearish_breakout_monthly'])} monthly bearish breakouts")
        return breakout_signals
        
    except Exception as e:
        logger.error(f"Error in identify_monthly_breakouts: {e}", exc_info=True)
        return {'bullish_breakout_monthly': [], 'bearish_breakout_monthly': []}

def process_stock_data(table_name: str, results: List[Tuple], cursor, multibagger_symbols: List[str], data_source: str) -> Tuple:
    """Process stock data to generate buy/sell/neutral signals"""
    if not results or len(results) < 2:
        return [], [], [], None
    
    try:
        # Extract latest data
        latest = results[0]
        previous = results[1]
        (date_latest, close_latest, volume_latest, rsi_weekly_latest, 
         rsi_monthly_latest, rsi_3months_latest, _, 
         ao_weekly_latest, ma_30_latest, _) = latest
        (_, _, _, _, _, _, _, 
         ao_weekly_previous, ma_30_previous, _) = previous
        
        stock_name = table_name.replace('PSX_', '').replace('_stock_data', '').strip().upper()
        ao_change_date, ao_change_close = get_ao_change_date(cursor, table_name)
        freefloatratio = get_freefloatratio(stock_name)
        multibagger = stock_name in multibagger_symbols
        truncated_data_source = data_source.split('_')[4].split('.')[0]
        
        # Calculate P/L and holding days
        p_l, holding_days = 0.0, 0
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
        
        # Prepare DataFrame for breakout detection
        df_columns = ['Date', 'Close', 'Volume', 'RSI_Weekly_Avg', 'RSI_Monthly', 
                      'RSI_3Months_Avg', 'RSI_Monthly_Avg', 'AO_weekly', 'MA_30', 'pct_change']
        df = pd.DataFrame(results, columns=df_columns)
        df = df.iloc[::-1].reset_index(drop=True)  # Chronological order
        
        # Detect breakouts for both weekly and monthly timeframes
        weekly_breakout_data = identify_weekly_breakouts(df)
        monthly_breakout_data = identify_monthly_breakouts(df)
        # Combine breakout data
        breakout_data = {
            'bullish_breakout_weekly': weekly_breakout_data['bullish_breakout_weekly'],
            'bearish_breakout_weekly': weekly_breakout_data['bearish_breakout_weekly'],
            'bullish_breakout_monthly': monthly_breakout_data['bullish_breakout_monthly'],
            'bearish_breakout_monthly': monthly_breakout_data['bearish_breakout_monthly']
        }
        
        # Determine signals
        buy_data, sell_data, neutral_data = [], [], []
        # Handle None values by defaulting to 0 for comparisons
        rsi_3months_val = rsi_3months_latest if rsi_3months_latest is not None else 0
        rsi_weekly_val = rsi_weekly_latest if rsi_weekly_latest is not None else 0
        rsi_monthly_val = rsi_monthly_latest if rsi_monthly_latest is not None else 0
        ao_weekly_val = ao_weekly_latest if ao_weekly_latest is not None else 0
        volume_val = volume_latest if volume_latest is not None else 0
        
        buy_signal = (rsi_3months_val >= 40 and rsi_weekly_val >= 40 and 
                      ao_weekly_val >= 0 and volume_val > 5000)
        
        sell_signal = (rsi_monthly_val <= 50 and rsi_weekly_val <= 50 and 
                       ao_weekly_val <= 0 and close_latest <= ma_30_latest and 
                       volume_val > 0)
        
        if buy_signal:
            signal_data = base_data.copy()
            signal_data.update({
                'Success': 'Yes' if close_latest >= ao_change_close else 'No',
                '% P/L': p_l,
                'Signal_Date': ao_change_date,
                'Signal_Close': ao_change_close,
                'Holding_Days': holding_days,
                'Status': 'Buy'
            })
            if breakout_data and (breakout_data['bullish_breakout_weekly'] or breakout_data['bullish_breakout_monthly']):
                signal_data['Breakout'] = 'Bullish Breakout Detected (Weekly)' if breakout_data['bullish_breakout_weekly'] else 'Bullish Breakout Detected (Monthly)'
            buy_data.append(signal_data)
            
        elif sell_signal:
            signal_data = base_data.copy()
            signal_data.update({
                'Success': 'Yes' if close_latest < ao_change_close else 'No',
                '% P/L': round(((ao_change_close - close_latest) / close_latest) * 100, 2),
                'Signal_Date': ao_change_date,
                'Signal_Close': ao_change_close,
                'Holding_Days': holding_days,
                'Status': 'Sell'
            })
            if breakout_data and (breakout_data['bearish_breakout_weekly'] or breakout_data['bearish_breakout_monthly']):
                signal_data['Breakout'] = 'Bearish Breakout Detected (Weekly)' if breakout_data['bearish_breakout_weekly'] else 'Bearish Breakout Detected (Monthly)'
            sell_data.append(signal_data)
            
        else:
            signal_data = base_data.copy()
            signal_data.update({
                'Trend_Direction': 'Bullish' if ma_30_latest > ma_30_previous and ao_weekly_latest > ao_weekly_previous else 'Bearish',
                'Status': 'Neutral'
            })
            neutral_data.append(signal_data)
            
        return buy_data, sell_data, neutral_data, breakout_data
        
    except Exception as e:
        logger.error(f"Error processing {table_name}: {e}", exc_info=True)
        return [], [], [], None
