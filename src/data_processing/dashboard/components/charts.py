"""
Charts component for the PSX dashboard.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlite3
from PIL import Image
from src.data_processing.dashboard.components.shared_styles import (
    apply_shared_styles,
    create_custom_header,
    create_custom_subheader,
    create_custom_divider,
    create_chart_container,
    create_metric_card
)
from typing import Optional, Dict, Any
import talib
from scipy import stats
import plotly.express as px
import time

# Constants for database paths
INDICATORS_DB_PATH = "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/psx_consolidated_data_indicators_PSX.db"
SIGNALS_DB_PATH = "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_investing_Stocks_KMI30.db"

# Enhanced Technical indicator categories
INDICATOR_CATEGORIES = {
    "Trend Indicators": [
        "MA_30", "MA_50", "MA_100", "MA_200",
        "EMA_9", "EMA_21", "EMA_50", "EMA_200",
        "Ichimoku_Conversion", "Ichimoku_Base", "Ichimoku_SpanA", "Ichimoku_SpanB",
        "Parabolic_SAR", "ADX", "ADX_Positive", "ADX_Negative"
    ],
    "Momentum Indicators": [
        "RSI_14", "RSI_9", "RSI_26",
        "Stochastic_K", "Stochastic_D",
        "MACD", "MACD_Signal", "MACD_Hist",
        "AO", "AO_AVG",
        "CCI", "ROC", "Williams_%R"
    ],
    "Volume Indicators": [
        "Volume_MA_20", "Volume_MA_50",
        "OBV", "CMF", "VWAP",
        "Volume_Profile", "Market_Profile"
    ],
    "Volatility Indicators": [
        "BB_Upper", "BB_Middle", "BB_Lower",
        "ATR", "Keltner_Upper", "Keltner_Middle", "Keltner_Lower",
        "Donchian_Upper", "Donchian_Middle", "Donchian_Lower"
    ],
    "Pattern Recognition": [
        "Doji", "Engulfing", "Hammer", "Shooting_Star",
        "Morning_Star", "Evening_Star", "Three_White_Soldiers", "Three_Black_Crows",
        "Head_Shoulders", "Inverse_Head_Shoulders", "Double_Top", "Double_Bottom"
    ]
}

# Advanced indicator categories
ADVANCED_INDICATORS = {
    "Market Structure": [
        "Support_Levels", "Resistance_Levels",
        "Trend_Lines", "Channels",
        "Price_Action_Zones", "Volume_Clusters"
    ],
    "Fibonacci Analysis": [
        "Fibo_236", "Fibo_382", "Fibo_500", "Fibo_618", "Fibo_786",
        "Fibo_Extensions", "Fibo_Time_Zones"
    ],
    "Market Profile": [
        "Value_Area_High", "Value_Area_Low", "Point_of_Control",
        "Volume_Profile", "Volume_Clusters", "Volume_Delta"
    ],
    "Pattern Recognition": [
        "Doji", "Engulfing", "Hammer", "EveningStar", "MorningStar",
        "Head_Shoulders", "Inverse_Head_Shoulders", "Double_Top", "Double_Bottom"
    ],
    "Advanced Patterns": [
        "Harmonic_Patterns", "Elliott_Waves",
        "Gartley", "Butterfly", "Bat", "Crab",
        "ABCD_Pattern", "Three_Drives"
    ]
}

def connect_to_database(db_path: str) -> Optional[sqlite3.Connection]:
    """Connect to SQLite database with error handling."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def get_available_stocks(conn: sqlite3.Connection) -> list:
    """Get list of available stock symbols from the database."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'PSX_%_stock_data'")
        tables = cursor.fetchall()
        return [table[0].replace('PSX_', '').replace('_stock_data', '') for table in tables]
    except sqlite3.Error as e:
        st.error(f"Error getting stock list: {str(e)}")
        return []

def get_stock_data(conn: sqlite3.Connection, symbol: str, days_back: int = 180) -> pd.DataFrame:
    """Get stock data with indicators."""
    try:
        table_name = f"PSX_{symbol}_stock_data"
        query = f"SELECT * FROM {table_name} ORDER BY Date DESC LIMIT {days_back}"
        df = pd.read_sql_query(query, conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except sqlite3.Error as e:
        st.error(f"Error getting stock data: {str(e)}")
        return pd.DataFrame()

def calculate_awesome_oscillator(high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """Calculate Awesome Oscillator manually.
    
    The Awesome Oscillator is calculated as:
    AO = SMA(Median Price, fast_period) - SMA(Median Price, slow_period)
    where Median Price = (High + Low) / 2
    """
    # Calculate median price
    median_price = (high + low) / 2
    
    # Calculate fast and slow SMAs
    fast_sma = median_price.rolling(window=fast_period).mean()
    slow_sma = median_price.rolling(window=slow_period).mean()
    
    # Calculate AO
    ao = fast_sma - slow_sma
    
    return ao

def create_ao_zones(ao_series):
    """Create AO zones using quantiles to ensure monotonic bins.
    
    Args:
        ao_series: Pandas Series containing AO values
        
    Returns:
        Pandas Series with zone labels
    """
    # Handle empty or all-NaN series
    if ao_series.empty or ao_series.isna().all():
        return pd.Series(['Neutral'] * len(ao_series), index=ao_series.index)
    
    # Remove NaN values for quantile calculation
    valid_series = ao_series.dropna()
    if len(valid_series) < 2:
        return pd.Series(['Neutral'] * len(ao_series), index=ao_series.index)
    
    # Calculate quantiles
    q1, q3 = valid_series.quantile([0.25, 0.75])
    
    # Handle cases where quantiles are equal or invalid
    if pd.isna(q1) or pd.isna(q3) or q1 >= q3:
        # Use standard deviation as fallback
        mean = valid_series.mean()
        std = valid_series.std()
        if pd.isna(std) or std == 0:
            return pd.Series(['Neutral'] * len(ao_series), index=ao_series.index)
        q1 = mean - std
        q3 = mean + std
    
    # Create monotonically increasing bins
    bins = [-np.inf, q1, q3, np.inf]
    labels = ['Oversold', 'Neutral', 'Overbought']
    
    try:
        # Attempt to create zones
        zones = pd.cut(ao_series, bins=bins, labels=labels)
        # Fill NaN values directly instead of using fillna with 'Neutral'
        zones = zones.fillna('Neutral')
        return zones
    except ValueError:
        # Fallback to simple mean-based zones if cut fails
        mean = valid_series.mean()
        return pd.Series(['Oversold' if x < mean else 'Overbought' if x > mean else 'Neutral' 
                         for x in ao_series], index=ao_series.index)

def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced technical indicators."""
    # Ensure data is valid
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fix: Replace fillna(method='ffill/bfill') with ffill()/bfill()
    df = df.ffill().bfill()
    
    # Calculate Awesome Oscillator (AO) with multiple timeframes
    # Daily AO
    df['AO'] = calculate_awesome_oscillator(df['High'], df['Low'])
    df['AO_AVG'] = df['AO'].rolling(window=5).mean()
    
    # Weekly AO
    weekly_high = df['High'].resample('W').max()
    weekly_low = df['Low'].resample('W').min()
    weekly_ao = calculate_awesome_oscillator(weekly_high, weekly_low)
    # Fix: Use ffill() instead of reindex with method='ffill'
    df['AO_weekly'] = weekly_ao.reindex(df.index).ffill()
    df['AO_weekly_AVG'] = df['AO_weekly'].rolling(window=3).mean()
    
    # Monthly AO
    # Fix: Replace 'M' with 'ME' for month end frequency
    monthly_high = df['High'].resample('ME').max()
    monthly_low = df['Low'].resample('ME').min()
    monthly_ao = calculate_awesome_oscillator(monthly_high, monthly_low)
    # Fix: Use ffill() instead of reindex with method='ffill'
    df['AO_monthly'] = monthly_ao.reindex(df.index).ffill()
    df['AO_monthly_AVG'] = df['AO_monthly'].rolling(window=3).mean()
    
    # AO Momentum
    df['AO_Momentum'] = df['AO'].diff(5)
    df['AO_weekly_Momentum'] = df['AO_weekly'].diff(3)
    df['AO_monthly_Momentum'] = df['AO_monthly'].diff(3)
    
    # AO Trend Strength
    df['AO_Trend_Strength'] = df['AO'].rolling(window=10).std()
    df['AO_weekly_Trend_Strength'] = df['AO_weekly'].rolling(window=5).std()
    df['AO_monthly_Trend_Strength'] = df['AO_monthly'].rolling(window=3).std()
    
    # AO Crossovers
    df['AO_Crossover'] = np.where(df['AO'] > df['AO_AVG'], 1, -1)
    df['AO_weekly_Crossover'] = np.where(df['AO_weekly'] > df['AO_weekly_AVG'], 1, -1)
    df['AO_monthly_Crossover'] = np.where(df['AO_monthly'] > df['AO_monthly_AVG'], 1, -1)
    
    # AO Divergence
    df['AO_Divergence'] = df['AO'] - df['AO'].shift(5)
    df['AO_weekly_Divergence'] = df['AO_weekly'] - df['AO_weekly'].shift(3)
    df['AO_monthly_Divergence'] = df['AO_monthly'] - df['AO_monthly'].shift(3)
    
    # AO Signal Strength - Fix: Add safe division to avoid divide by zero warnings
    df['AO_Signal_Strength'] = df['AO'].rolling(window=10).mean() / np.maximum(df['AO'].rolling(window=10).std(), 0.0001)
    df['AO_weekly_Signal_Strength'] = df['AO_weekly'].rolling(window=5).mean() / np.maximum(df['AO_weekly'].rolling(window=5).std(), 0.0001)
    df['AO_monthly_Signal_Strength'] = df['AO_monthly'].rolling(window=3).mean() / np.maximum(df['AO_monthly'].rolling(window=3).std(), 0.0001)
    
    # AO Histogram
    df['AO_Histogram'] = df['AO'] - df['AO_AVG']
    df['AO_weekly_Histogram'] = df['AO_weekly'] - df['AO_weekly_AVG']
    df['AO_monthly_Histogram'] = df['AO_monthly'] - df['AO_monthly_AVG']
    
    # AO Zones using quantiles
    df['AO_Zone'] = create_ao_zones(df['AO'])
    df['AO_weekly_Zone'] = create_ao_zones(df['AO_weekly'])
    df['AO_monthly_Zone'] = create_ao_zones(df['AO_monthly'])
    
    # Ichimoku Cloud
    df['Ichimoku_Conversion'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Ichimoku_Base'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Ichimoku_SpanA'] = (df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2
    df['Ichimoku_SpanB'] = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2
    
    # ADX (Average Directional Index)
    df['ADX'] = safe_talib_call('ADX', df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ADX_Positive'] = safe_talib_call('PLUS_DI', df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ADX_Negative'] = safe_talib_call('MINUS_DI', df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Parabolic SAR
    df['Parabolic_SAR'] = safe_talib_call('SAR', df['High'], df['Low'], acceleration=0.02, maximum=0.2)
    
    # Volume Indicators
    df['OBV'] = safe_talib_call('OBV', df['Close'], df['Volume'])
    df['CMF'] = safe_talib_call('ADOSC', df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
    
    # Volatility Indicators
    df['ATR'] = safe_talib_call('ATR', df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Keltner Channels
    middle = df['Close'].rolling(window=20).mean()
    atr = df['ATR']
    df['Keltner_Upper'] = middle + (2 * atr)
    df['Keltner_Middle'] = middle
    df['Keltner_Lower'] = middle - (2 * atr)
    
    # Donchian Channels
    df['Donchian_Upper'] = df['High'].rolling(window=20).max()
    df['Donchian_Middle'] = (df['High'].rolling(window=20).max() + df['Low'].rolling(window=20).min()) / 2
    df['Donchian_Lower'] = df['Low'].rolling(window=20).min()
    
    # Market Profile
    df['Value_Area_High'] = df['Close'].rolling(window=20).quantile(0.7)
    df['Value_Area_Low'] = df['Close'].rolling(window=20).quantile(0.3)
    df['Point_of_Control'] = df['Close'].rolling(window=20).median()
    
    # Volume Profile
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), 20)
    volume_profile = np.zeros(len(price_bins[:-1]))
    
    for i in range(len(price_bins) - 1):
        mask = (df['Close'] >= price_bins[i]) & (df['Close'] < price_bins[i + 1])
        volume_profile[i] = df.loc[mask, 'Volume'].sum()
    
    # Create a column for each price bin's volume
    for i in range(len(volume_profile)):
        df[f'Volume_Profile_Bin_{i}'] = volume_profile[i]
    
    # Use the middle bin as the main Volume_Profile indicator
    middle_bin = len(volume_profile) // 2
    df['Volume_Profile'] = df[f'Volume_Profile_Bin_{middle_bin}']
    
    # Advanced Pattern Recognition - using safe calls
    df['Doji'] = safe_talib_call('CDLDOJI', df['Open'], df['High'], df['Low'], df['Close'])
    df['Hammer'] = safe_talib_call('CDLHAMMER', df['Open'], df['High'], df['Low'], df['Close'])
    df['Engulfing'] = safe_talib_call('CDLENGULFING', df['Open'], df['High'], df['Low'], df['Close'])
    df['EveningStar'] = safe_talib_call('CDLEVENINGSTAR', df['Open'], df['High'], df['Low'], df['Close'])
    df['MorningStar'] = safe_talib_call('CDLMORNINGSTAR', df['Open'], df['High'], df['Low'], df['Close'])
    
    # Placeholder for complex patterns
    df['Head_Shoulders'] = 0
    df['Inverse_Head_Shoulders'] = 0
    df['Double_Top'] = 0
    df['Double_Bottom'] = 0
    
    # Support and Resistance Levels
    df['Support_Levels'] = df['Low'].rolling(window=20).min()
    df['Resistance_Levels'] = df['High'].rolling(window=20).max()
    
    # Trend Lines
    df['Trend_Lines'] = df['Close'].rolling(window=20).mean()
    
    # Price Action Zones
    df['Price_Action_Zones'] = df['Close'].rolling(window=20).std()
    
    # Volume Clusters
    df['Volume_Clusters'] = df['Volume'].rolling(window=20).mean()
    
    # Add structure break indicators
    # Price high breaks
    df['Price_High_Break'] = 0
    highest_20 = df['High'].rolling(window=20).max()
    df.loc[df['High'] > highest_20.shift(1), 'Price_High_Break'] = 1
    
    # Price low breaks
    df['Price_Low_Break'] = 0
    lowest_20 = df['Low'].rolling(window=20).min()
    df.loc[df['Low'] < lowest_20.shift(1), 'Price_Low_Break'] = 1
    
    # Volume breaks
    df['Volume_Break'] = 0
    vol_avg = df['Volume'].rolling(window=20).mean()
    df.loc[df['Volume'] > vol_avg * 2, 'Volume_Break'] = 1
    
    # Moving average breaks
    df['MA_Break'] = 0
    if 'MA_50' in df.columns and 'MA_200' in df.columns:
        # MA crossover (golden cross / death cross)
        df.loc[(df['MA_50'] > df['MA_200']) & (df['MA_50'].shift(1) <= df['MA_200'].shift(1)), 'MA_Break'] = 1
        df.loc[(df['MA_50'] < df['MA_200']) & (df['MA_50'].shift(1) >= df['MA_200'].shift(1)), 'MA_Break'] = -1
    
    # Trend line breaks
    df['Trend_Line_Break'] = 0
    if 'EMA_50' in df.columns:
        df.loc[(df['Close'] > df['EMA_50']) & (df['Close'].shift(1) <= df['EMA_50'].shift(1)), 'Trend_Line_Break'] = 1
        df.loc[(df['Close'] < df['EMA_50']) & (df['Close'].shift(1) >= df['EMA_50'].shift(1)), 'Trend_Line_Break'] = -1
    
    # Channel breaks
    df['Channel_Break'] = 0
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        df.loc[df['Close'] > df['BB_Upper'], 'Channel_Break'] = 1
        df.loc[df['Close'] < df['BB_Lower'], 'Channel_Break'] = -1
    
    return df

def calculate_market_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate market sentiment indicators with validation."""
    # Ensure data is valid
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fix: Replace fillna(method='ffill/bfill') with ffill()/bfill()
    df = df.ffill().bfill()
    
    # Calculate sentiment with validation
    rsi_sentiment = 1 if df['RSI_14'].iloc[-1] > 50 else -1 if not np.isnan(df['RSI_14'].iloc[-1]) else 0
    ma_sentiment = 1 if df['Close'].iloc[-1] > df['MA_50'].iloc[-1] else -1 if not np.isnan(df['MA_50'].iloc[-1]) else 0
    volume_sentiment = 1 if df['Volume'].iloc[-1] > df['Volume_MA_20'].iloc[-1] else -1 if not np.isnan(df['Volume_MA_20'].iloc[-1]) else 0
    ao_sentiment = 1 if df['AO'].iloc[-1] > 0 else -1 if not np.isnan(df['AO'].iloc[-1]) else 0
    
    sentiment = {
        'RSI_Sentiment': rsi_sentiment,
        'MA_Sentiment': ma_sentiment,
        'Volume_Sentiment': volume_sentiment,
        'AO_Sentiment': ao_sentiment
    }
    return sentiment

def calculate_correlation(df: pd.DataFrame, other_symbols: list) -> pd.DataFrame:
    """Calculate correlation with other symbols."""
    correlations = {}
    for symbol in other_symbols:
        try:
            other_df = get_stock_data(connect_to_database(INDICATORS_DB_PATH), symbol)
            if not other_df.empty:
                correlation = df['Close'].corr(other_df['Close'])
                correlations[symbol] = correlation
        except:
            continue
    return pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])

def create_stock_header(stock_data: pd.DataFrame, selected_stock: str) -> None:
    """Create enhanced stock header with key metrics."""
    # Ensure data is valid
    stock_data = stock_data.replace([np.inf, -np.inf], np.nan)
    stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
    
    latest_price = stock_data['Close'].iloc[-1]
    prev_close = stock_data['Close'].iloc[-2]
    price_change = latest_price - prev_close
    percent_change = (price_change / prev_close * 100) if prev_close != 0 else 0
    
    # Calculate additional metrics with validation
    day_high = stock_data['High'].iloc[-1]
    day_low = stock_data['Low'].iloc[-1]
    volume = stock_data['Volume'].iloc[-1]
    avg_volume = stock_data['Volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
    
    # Create header container with tabs
    header_tabs = st.tabs(["Overview", "Performance", "Technical", "Market Context"])
    
    with header_tabs[0]:  # Overview
        st.markdown(f"### {selected_stock}")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Price",
                f"₨ {latest_price:,.2f}",
                f"{price_change:+,.2f} ({percent_change:+.2f}%)",
                delta_color="normal"
            )
        
        with col2:
            day_range_pct = ((day_high - day_low) / day_low * 100) if day_low != 0 else 0
            st.metric(
                "Day Range",
                f"₨ {day_low:,.2f} - {day_high:,.2f}",
                f"{day_range_pct:.1f}%"
            )
        
        with col3:
            volume_vs_avg = ((volume / avg_volume) - 1) * 100 if avg_volume != 0 else 0
            st.metric(
                "Volume",
                f"{volume:,.0f}",
                f"{volume_vs_avg:+.1f}% vs Avg"
            )
        
        with col4:
            volatility = stock_data['Close'].pct_change().std() * 100
            volatility = 0 if np.isnan(volatility) else volatility
            st.metric(
                "Volatility",
                f"{volatility:.1f}%",
                "Daily"
            )
    
    with header_tabs[1]:  # Performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Weekly performance with validation
            weekly_price = stock_data['Close'].iloc[-5] if len(stock_data) >= 5 else stock_data['Close'].iloc[0]
            weekly_change = ((latest_price / weekly_price) - 1) * 100 if weekly_price != 0 else 0
            st.metric(
                "Weekly",
                f"{weekly_change:+.1f}%",
                delta_color="normal"
            )
        
        with col2:
            # Monthly performance with validation
            monthly_price = stock_data['Close'].iloc[-20] if len(stock_data) >= 20 else stock_data['Close'].iloc[0]
            monthly_change = ((latest_price / monthly_price) - 1) * 100 if monthly_price != 0 else 0
            st.metric(
                "Monthly",
                f"{monthly_change:+.1f}%",
                delta_color="normal"
            )
        
        with col3:
            # YTD performance with validation
            ytd_price = stock_data['Close'].iloc[0]
            ytd_change = ((latest_price / ytd_price) - 1) * 100 if ytd_price != 0 else 0
            st.metric(
                "YTD",
                f"{ytd_change:+.1f}%",
                delta_color="normal"
            )
    
    with header_tabs[2]:  # Technical
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # RSI with validation
            rsi = stock_data['RSI_14'].iloc[-1] if 'RSI_14' in stock_data.columns else 50
            rsi = 50 if np.isnan(rsi) else rsi
            st.metric(
                "RSI(14)",
                f"{rsi:.1f}",
                "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
                delta_color="normal"
            )
        
        with col2:
            # MACD with validation
            macd = stock_data['MACD'].iloc[-1] if 'MACD' in stock_data.columns else 0
            macd = 0 if np.isnan(macd) else macd
            st.metric(
                "MACD",
                f"{macd:.2f}",
                "Bullish" if macd > 0 else "Bearish",
                delta_color="normal"
            )
        
        with col3:
            # Trend with validation
            ma_50 = stock_data['MA_50'].iloc[-1] if 'MA_50' in stock_data.columns else latest_price
            ma_50 = latest_price if np.isnan(ma_50) else ma_50
            trend = "Bullish" if latest_price > ma_50 else "Bearish"
            st.metric(
                "Trend",
                trend,
                f"vs MA(50): {ma_50:,.2f}",
                delta_color="normal"
            )
    
    with header_tabs[3]:  # Market Context
        col1, col2 = st.columns(2)
        
        with col1:
            # Market Cap with validation
            if 'Market_Cap' in stock_data.columns:
                market_cap = stock_data['Market_Cap'].iloc[-1]
                market_cap = 0 if np.isnan(market_cap) else market_cap
                st.metric(
                    "Market Cap",
                    f"₨ {market_cap:,.0f}",
                    "Large Cap" if market_cap > 1e12 else "Mid Cap" if market_cap > 1e11 else "Small Cap"
                )
        
        with col2:
            # Beta with validation
            if 'Beta' in stock_data.columns:
                beta = stock_data['Beta'].iloc[-1]
                beta = 1 if np.isnan(beta) else beta
                st.metric(
                    "Beta",
                    f"{beta:.2f}",
                    "High Risk" if beta > 1.5 else "Moderate Risk" if beta > 1 else "Low Risk"
                )

def create_price_indicators_layout(stock_data: pd.DataFrame, selected_stock: str) -> None:
    """Create an enhanced layout for price and indicators section."""
    # Create main layout columns
    chart_col, sidebar_col = st.columns([4, 1])
    
    with chart_col:
        # Stock Header Section
        create_stock_header(stock_data, selected_stock)
        
        # Main Chart Container
        with st.container():
            # Enhanced Chart Controls
            controls_col1, controls_col2, controls_col3, controls_col4, controls_col5 = st.columns([2, 2, 1, 1, 1])
            
            with controls_col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Candlestick", "OHLC", "Line", "Area", "Heikin-Ashi"],
                    key="chart_type_selector"
                )
            
            with controls_col2:
                timeframe = st.selectbox(
                    "Timeframe",
                    ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "ALL"],
                    index=3,
                    key="timeframe_selector"
                )
            
            with controls_col3:
                show_patterns = st.checkbox("Show Patterns", value=True, key="show_patterns_checkbox")
            
            with controls_col4:
                show_fibonacci = st.checkbox("Show Fibonacci", value=False, key="show_fibonacci_checkbox")
            
            with controls_col5:
                show_structure_breaks = st.checkbox("Show Structure Breaks", value=True, key="show_structure_breaks_checkbox")
            
            # Create and display main price chart
            price_chart = create_enhanced_price_chart(
                stock_data, 
                chart_type,
                show_patterns=show_patterns,
                show_fibonacci=show_fibonacci,
                show_structure_breaks=show_structure_breaks
            )
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Volume and Indicators Section
            vol_ind_tabs = st.tabs(["Volume", "Technical Indicators", "Patterns", "Structure"])
            
            with vol_ind_tabs[0]:  # Volume
                volume_chart = create_volume_chart(stock_data)
                st.plotly_chart(volume_chart, use_container_width=True)
                
                # Volume Analysis
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Volume",
                        f"{stock_data['Volume'].iloc[-1]:,.0f}",
                        f"{((stock_data['Volume'].iloc[-1] / stock_data['Volume'].rolling(20).mean().iloc[-1]) - 1) * 100:+.1f}% vs Avg"
                    )
                with col2:
                    st.metric(
                        "Volume Trend",
                        "Increasing" if stock_data['Volume'].iloc[-1] > stock_data['Volume'].iloc[-2] else "Decreasing",
                        f"{((stock_data['Volume'].iloc[-1] / stock_data['Volume'].iloc[-2]) - 1) * 100:+.1f}%"
                    )
                with col3:
                    st.metric(
                        "Volume Profile",
                        "Accumulation" if stock_data['Close'].iloc[-1] > stock_data['Open'].iloc[-1] else "Distribution",
                        f"{stock_data['Volume'].iloc[-1] / stock_data['Volume'].rolling(20).mean().iloc[-1]:.1f}x Avg"
                    )
            
            with vol_ind_tabs[1]:  # Technical Indicators
                # RSI
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['RSI_14'],
                    name='RSI(14)',
                    line=dict(color='blue')
                ))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                rsi_fig.update_layout(height=200, title="RSI(14)")
                st.plotly_chart(rsi_fig, use_container_width=True)
                
                # MACD
                if all(col in stock_data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['MACD'],
                        name='MACD',
                        line=dict(color='blue')
                    ))
                    macd_fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['MACD_Signal'],
                        name='Signal',
                        line=dict(color='orange')
                    ))
                    macd_fig.add_trace(go.Bar(
                        x=stock_data.index,
                        y=stock_data['MACD_Hist'],
                        name='Histogram'
                    ))
                    macd_fig.update_layout(height=200, title="MACD")
                    st.plotly_chart(macd_fig, use_container_width=True)
            
            with vol_ind_tabs[2]:  # Patterns
                # Pattern Recognition
                pattern_fig = go.Figure()
                pattern_fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name='Price'
                ))
                add_pattern_recognition(pattern_fig, stock_data)
                pattern_fig.update_layout(height=300, title="Pattern Recognition")
                st.plotly_chart(pattern_fig, use_container_width=True)
            
            with vol_ind_tabs[3]:  # Structure
                # Structure Analysis
                structure_fig = go.Figure()
                structure_fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name='Price'
                ))
                add_structure_break_analysis(structure_fig, stock_data)
                structure_fig.update_layout(height=300, title="Structure Analysis")
                st.plotly_chart(structure_fig, use_container_width=True)
            
            # Market Sentiment Analysis
            sentiment = calculate_market_sentiment(stock_data)
            display_market_sentiment(sentiment)
    
    with sidebar_col:
        create_indicator_sidebar(stock_data)

def create_enhanced_price_chart(
    data: pd.DataFrame, 
    chart_type: str = "Candlestick",
    show_patterns: bool = True,
    show_fibonacci: bool = False,
    show_structure_breaks: bool = True
) -> go.Figure:
    """Create an enhanced price chart with advanced features."""
    fig = go.Figure()
    
    # Add price data based on chart type
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price",
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ))
    elif chart_type == "Heikin-Ashi":
        ha_data = calculate_heikin_ashi(data)
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=ha_data['HA_Open'],
            high=ha_data['HA_High'],
            low=ha_data['HA_Low'],
            close=ha_data['HA_Close'],
            name="Heikin-Ashi",
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ))
    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ))
    else:  # Line or Area
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name="Price",
            fill='tonexty' if chart_type == "Area" else None,
            line=dict(color='#2196F3')
        ))
    
    # Add technical overlays
    add_technical_overlays(fig, data)
    
    # Add pattern recognition if enabled
    if show_patterns:
        add_pattern_recognition(fig, data)
    
    # Add Fibonacci levels if enabled
    if show_fibonacci:
        add_fibonacci_levels(fig, data)
    
    # Add structure breaks if enabled
    if show_structure_breaks:
        add_structure_break_analysis(fig, data)
    
    # Update layout
    fig.update_layout(
        title=None,
        height=600,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        margin=dict(t=20, b=20, l=50, r=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def calculate_heikin_ashi(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Heikin-Ashi candlesticks."""
    ha_data = pd.DataFrame(index=data.index)
    ha_data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    ha_data['HA_Open'] = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
    ha_data['HA_High'] = data[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_data['HA_Low'] = data[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    return ha_data

def add_pattern_recognition(fig: go.Figure, data: pd.DataFrame) -> None:
    """Add pattern recognition markers to the chart."""
    # Add markers for Doji patterns
    doji_points = data[data['Doji'] != 0]
    if not doji_points.empty:
        fig.add_trace(go.Scatter(
            x=doji_points.index,
            y=doji_points['Close'],
            mode='markers',
            name='Doji',
            marker=dict(
                symbol='diamond',
                size=10,
                color='purple'
            )
        ))
    
    # Add markers for Engulfing patterns
    engulfing_points = data[data['Engulfing'] != 0]
    if not engulfing_points.empty:
        fig.add_trace(go.Scatter(
            x=engulfing_points.index,
            y=engulfing_points['Close'],
            mode='markers',
            name='Engulfing',
            marker=dict(
                symbol='star',
                size=12,
                color='orange'
            )
        ))

def add_fibonacci_levels(fig: go.Figure, data: pd.DataFrame) -> None:
    """Add Fibonacci retracement levels to the chart."""
    # Calculate Fibonacci levels if they don't exist
    if not all(level in data.columns for level in ['Fibo_236', 'Fibo_382', 'Fibo_500', 'Fibo_618', 'Fibo_786']):
        # Calculate high and low points for the current view
        high = data['High'].max()
        low = data['Low'].min()
        range_size = high - low
        
        # Calculate Fibonacci levels
        fibo_levels = {
            'Fibo_236': high - (range_size * 0.236),
            'Fibo_382': high - (range_size * 0.382),
            'Fibo_500': high - (range_size * 0.500),
            'Fibo_618': high - (range_size * 0.618),
            'Fibo_786': high - (range_size * 0.786)
        }
    else:
        # Use existing Fibonacci levels
        fibo_levels = {
            'Fibo_236': data['Fibo_236'].iloc[-1],
            'Fibo_382': data['Fibo_382'].iloc[-1],
            'Fibo_500': data['Fibo_500'].iloc[-1],
            'Fibo_618': data['Fibo_618'].iloc[-1],
            'Fibo_786': data['Fibo_786'].iloc[-1]
        }
    
    colors = ['#FF9800', '#2196F3', '#4CAF50', '#9C27B0', '#F44336']
    
    for level, color in zip(fibo_levels.keys(), colors):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=[fibo_levels[level]] * len(data),
            name=level,
            line=dict(color=color, dash='dash'),
            visible='legendonly'
        ))

def add_structure_break_analysis(fig: go.Figure, data: pd.DataFrame) -> None:
    """Add structure break markers to the chart."""
    # Add safety to check if columns exist
    required_columns = ['Price_High_Break', 'Price_Low_Break', 'Volume_Break', 
                        'MA_Break', 'Trend_Line_Break', 'Channel_Break']
    
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0
    
    # Price High Breaks
    high_breaks = data[data['Price_High_Break'] == 1]
    if not high_breaks.empty:
        fig.add_trace(go.Scatter(
            x=high_breaks.index,
            y=high_breaks['High'],
            mode='markers',
            name='Price High Break',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green'
            )
        ))
    
    # Price Low Breaks
    low_breaks = data[data['Price_Low_Break'] == 1]
    if not low_breaks.empty:
        fig.add_trace(go.Scatter(
            x=low_breaks.index,
            y=low_breaks['Low'],
            mode='markers',
            name='Price Low Break',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red'
            )
        ))
    
    # Volume Breaks
    volume_breaks = data[data['Volume_Break'] == 1]
    if not volume_breaks.empty:
        fig.add_trace(go.Scatter(
            x=volume_breaks.index,
            y=volume_breaks['Close'],
            mode='markers',
            name='Volume Break',
            marker=dict(
                symbol='diamond',
                size=10,
                color='blue'
            )
        ))
    
    # MA Breaks
    ma_breaks = data[data['MA_Break'] == 1]
    if not ma_breaks.empty:
        fig.add_trace(go.Scatter(
            x=ma_breaks.index,
            y=ma_breaks['Close'],
            mode='markers',
            name='MA Break',
            marker=dict(
                symbol='star',
                size=10,
                color='purple'
            )
        ))
    
    # Trend Line Breaks
    trend_breaks = data[data['Trend_Line_Break'] != 0]
    if not trend_breaks.empty:
        fig.add_trace(go.Scatter(
            x=trend_breaks.index,
            y=trend_breaks['Close'],
            mode='markers',
            name='Trend Line Break',
            marker=dict(
                symbol='circle',
                size=10,
                color='orange'
            )
        ))
    
    # Channel Breaks
    channel_breaks = data[data['Channel_Break'] != 0]
    if not channel_breaks.empty:
        fig.add_trace(go.Scatter(
            x=channel_breaks.index,
            y=channel_breaks['Close'],
            mode='markers',
            name='Channel Break',
            marker=dict(
                symbol='square',
                size=10,
                color='black'
            )
        ))

def add_technical_overlays(fig: go.Figure, data: pd.DataFrame) -> None:
    """Add technical overlays to the chart."""
    # Add Ichimoku Cloud
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Ichimoku_SpanA'],
        name='Ichimoku Span A',
        line=dict(color='blue', width=1),
        visible='legendonly'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Ichimoku_SpanB'],
        name='Ichimoku Span B',
        line=dict(color='red', width=1),
        visible='legendonly'
    ))
    
    # Add Keltner Channels
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Keltner_Upper'],
        name='Keltner Upper',
        line=dict(color='gray', dash='dash'),
        visible='legendonly'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Keltner_Middle'],
        name='Keltner Middle',
        line=dict(color='gray'),
        visible='legendonly'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Keltner_Lower'],
        name='Keltner Lower',
        line=dict(color='gray', dash='dash'),
        visible='legendonly'
    ))
    
    # Add Donchian Channels
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Donchian_Upper'],
        name='Donchian Upper',
        line=dict(color='purple', dash='dash'),
        visible='legendonly'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Donchian_Middle'],
        name='Donchian Middle',
        line=dict(color='purple'),
        visible='legendonly'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Donchian_Lower'],
        name='Donchian Lower',
        line=dict(color='purple', dash='dash'),
        visible='legendonly'
    ))
    
    # Add Parabolic SAR
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Parabolic_SAR'],
        name='Parabolic SAR',
        mode='markers',
        marker=dict(
            symbol='circle',
            size=6,
            color='red'
        ),
        visible='legendonly'
    ))
    
    # Add Value Area
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Value_Area_High'],
        name='Value Area High',
        line=dict(color='green', dash='dot'),
        visible='legendonly'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Value_Area_Low'],
        name='Value Area Low',
        line=dict(color='red', dash='dot'),
        visible='legendonly'
    ))
    
    # Add Point of Control
    fig.add_trace(go.Scatter(
        x=data.index,
        y=[data['Point_of_Control'].iloc[-1]] * len(data),
        name='Point of Control',
        line=dict(color='blue', dash='dot'),
        visible='legendonly'
    ))

def display_market_sentiment(sentiment: Dict[str, float]) -> None:
    """Display market sentiment analysis."""
    st.markdown("### Market Sentiment Analysis")
    
    # Calculate overall sentiment
    overall_sentiment = sum(sentiment.values()) / len(sentiment)
    sentiment_direction = "normal" if overall_sentiment > 0 else "inverse"
    
    # Display sentiment indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "RSI Sentiment",
            "Bullish" if sentiment['RSI_Sentiment'] > 0 else "Bearish",
            delta_color=sentiment_direction
        )
    
    with col2:
        st.metric(
            "MA Sentiment",
            "Bullish" if sentiment['MA_Sentiment'] > 0 else "Bearish",
            delta_color=sentiment_direction
        )
    
    with col3:
        st.metric(
            "Volume Sentiment",
            "Bullish" if sentiment['Volume_Sentiment'] > 0 else "Bearish",
            delta_color=sentiment_direction
        )
    
    with col4:
        st.metric(
            "AO Sentiment",
            "Bullish" if sentiment['AO_Sentiment'] > 0 else "Bearish",
            delta_color=sentiment_direction
        )

def create_volume_chart(data: pd.DataFrame) -> go.Figure:
    """Create enhanced volume chart with profile."""
    fig = go.Figure()
    
    # Add volume bars
    colors = ['red' if close < open else 'green' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.5
    ))
    
    # Add volume MA if available
    if 'Volume_MA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_MA_20'],
            name='Volume MA(20)',
            line=dict(color='blue', width=1)
        ))
    
    # Add volume profile
    price_bins = np.linspace(data['Low'].min(), data['High'].max(), 20)
    volume_profile = np.zeros(len(price_bins[:-1]))
    
    for i in range(len(price_bins) - 1):
        mask = (data['Close'] >= price_bins[i]) & (data['Close'] < price_bins[i + 1])
        volume_profile[i] = data.loc[mask, 'Volume'].sum()
    
    # Create a column for each price bin's volume
    for i in range(len(volume_profile)):
        data[f'Volume_Profile_Bin_{i}'] = volume_profile[i]
    
    # Use the middle bin as the main Volume_Profile indicator
    middle_bin = len(volume_profile) // 2
    data['Volume_Profile'] = data[f'Volume_Profile_Bin_{middle_bin}']
    
    fig.add_trace(go.Bar(
        x=volume_profile,
        y=[(price_bins[i] + price_bins[i+1])/2 for i in range(len(price_bins)-1)],
        name='Volume Profile',
        orientation='h',
        opacity=0.3,
        visible='legendonly'
    ))
    
    # Add volume delta
    if 'Volume_Delta' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_Delta'],
            name='Volume Delta',
            line=dict(color='purple', width=1),
            visible='legendonly'
        ))
    
    # Add volume clusters
    if 'Volume_Clusters' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_Clusters'],
            name='Volume Clusters',
            mode='markers',
            marker=dict(
                size=8,
                color='orange',
                symbol='circle'
            ),
            visible='legendonly'
        ))
    
    # Add volume trend
    if 'Volume_Trend' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_Trend'],
            name='Volume Trend',
            line=dict(color='green', width=2),
            visible='legendonly'
        ))
    
    # Update layout
    fig.update_layout(
        height=200,
        template='plotly_white',
        margin=dict(t=0, b=20, l=50, r=50),
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_indicator_sidebar(data: pd.DataFrame) -> None:
    """Create enhanced indicator selection sidebar."""
    st.markdown("### Technical Indicators")
    
    # Create tabs for each category
    indicator_tabs = st.tabs(list(INDICATOR_CATEGORIES.keys()) + list(ADVANCED_INDICATORS.keys()))
    
    # Combine both indicator dictionaries for easier access
    all_indicators = {**INDICATOR_CATEGORIES, **ADVANCED_INDICATORS}
    
    # Create a unique identifier for each category
    category_index = 0
    
    for tab, category_name in zip(indicator_tabs, list(INDICATOR_CATEGORIES.keys()) + list(ADVANCED_INDICATORS.keys())):
        with tab:
            st.markdown(f"#### {category_name}")
            
            # Get indicators for this category (with safe fallback)
            category_indicators = all_indicators.get(category_name, [])
            
            # Filter to only indicators that exist in the data
            available_indicators = [ind for ind in category_indicators if ind in data.columns]
            
            if not available_indicators:
                st.info(f"No {category_name} indicators available")
                continue
                
            # Create multiselect for indicators in this category with unique key
            selected_indicators = st.multiselect(
                f"Select {category_name}",
                options=available_indicators,
                default=available_indicators[:2] if len(available_indicators) >= 2 else available_indicators,
                key=f"multiselect_{category_name}_{category_index}"
            )
            
            # Increment the category index for the next iteration
            category_index += 1
            
            # Display selected indicators
            for indicator in selected_indicators:
                with st.expander(indicator):
                    # Show current value and trend
                    current_value = data[indicator].iloc[-1]
                    prev_value = data[indicator].iloc[-2]
                    change = current_value - prev_value
                    
                    st.metric(
                        indicator,
                        f"{current_value:.2f}",
                        f"{change:+.2f}",
                        delta_color="normal"
                    )
                    
                    # Add indicator-specific analysis
                    if indicator.startswith('RSI'):
                        add_rsi_analysis(indicator, data)
                    elif indicator.startswith('MA'):
                        add_ma_analysis(indicator, data)
                    elif indicator.startswith('AO'):
                        add_ao_analysis(indicator, data)
                    elif indicator == 'Volume_MA_20':
                        add_volume_analysis(indicator, data)
                    elif indicator in INDICATOR_CATEGORIES.get("Pattern Recognition", []):
                        add_pattern_analysis(indicator, data)
                    elif indicator in ADVANCED_INDICATORS.get("Fibonacci Analysis", []):
                        add_fibonacci_analysis(indicator, data)
                    elif indicator in ADVANCED_INDICATORS.get("Market Structure", []):
                        add_market_structure_analysis(indicator, data)
                    elif indicator in ADVANCED_INDICATORS.get("Market Profile", []):
                        add_market_profile_analysis(indicator, data)
                    elif indicator in ADVANCED_INDICATORS.get("Advanced Patterns", []):
                        add_advanced_pattern_analysis(indicator, data)

def add_pattern_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add pattern analysis with enhanced pattern detection and visualization."""
    st.write(f"Pattern analysis for {indicator}")
    
    # Create the chart
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Detect patterns
    patterns = detect_patterns(data)
    
    # Add pattern markers
    for pattern_name, pattern_data in patterns.items():
        if pattern_data['occurrences']:
            fig.add_trace(go.Scatter(
                x=pattern_data['dates'],
                y=pattern_data['prices'],
                mode='markers',
                name=pattern_name,
                marker=dict(
                    symbol='star',
                    size=12,
                    color=pattern_data['color'],
                    line=dict(width=2, color='black')
                )
            ))
    
    # Update layout
    fig.update_layout(
        title='Pattern Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
        height=600
    )
    
    # Generate a unique key using timestamp and indicator name
    unique_key = f"pattern_{indicator}_{int(time.time() * 1000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)
    
    # Display pattern statistics
    st.subheader("Pattern Statistics")
    for pattern_name, pattern_data in patterns.items():
        if pattern_data['occurrences']:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"{pattern_name} Occurrences",
                    len(pattern_data['occurrences'])
                )
            with col2:
                st.metric(
                    f"Last {pattern_name}",
                    pattern_data['last_occurrence'].strftime('%Y-%m-%d')
                )

def detect_patterns(data: pd.DataFrame) -> dict:
    """Detect various chart patterns in the price data."""
    patterns = {
        'Head and Shoulders': {'occurrences': [], 'dates': [], 'prices': [], 'color': 'red'},
        'Double Top': {'occurrences': [], 'dates': [], 'prices': [], 'color': 'orange'},
        'Double Bottom': {'occurrences': [], 'dates': [], 'prices': [], 'color': 'green'},
        'Triple Top': {'occurrences': [], 'dates': [], 'prices': [], 'color': 'purple'},
        'Triple Bottom': {'occurrences': [], 'dates': [], 'prices': [], 'color': 'blue'},
        'Ascending Triangle': {'occurrences': [], 'dates': [], 'prices': [], 'color': 'cyan'},
        'Descending Triangle': {'occurrences': [], 'dates': [], 'prices': [], 'color': 'magenta'}
    }
    
    # Lookback window for pattern detection
    window = 20
    
    for i in range(window, len(data)):
        # Get the window of data
        window_data = data.iloc[i-window:i+1]
        
        # Detect Head and Shoulders
        if detect_head_and_shoulders(window_data):
            patterns['Head and Shoulders']['occurrences'].append(i)
            patterns['Head and Shoulders']['dates'].append(data.index[i])
            patterns['Head and Shoulders']['prices'].append(data['High'].iloc[i])
        
        # Detect Double Top
        if detect_double_top(window_data):
            patterns['Double Top']['occurrences'].append(i)
            patterns['Double Top']['dates'].append(data.index[i])
            patterns['Double Top']['prices'].append(data['High'].iloc[i])
        
        # Detect Double Bottom
        if detect_double_bottom(window_data):
            patterns['Double Bottom']['occurrences'].append(i)
            patterns['Double Bottom']['dates'].append(data.index[i])
            patterns['Double Bottom']['prices'].append(data['Low'].iloc[i])
        
        # Detect Triple Top
        if detect_triple_top(window_data):
            patterns['Triple Top']['occurrences'].append(i)
            patterns['Triple Top']['dates'].append(data.index[i])
            patterns['Triple Top']['prices'].append(data['High'].iloc[i])
        
        # Detect Triple Bottom
        if detect_triple_bottom(window_data):
            patterns['Triple Bottom']['occurrences'].append(i)
            patterns['Triple Bottom']['dates'].append(data.index[i])
            patterns['Triple Bottom']['prices'].append(data['Low'].iloc[i])
        
        # Detect Ascending Triangle
        if detect_ascending_triangle(window_data):
            patterns['Ascending Triangle']['occurrences'].append(i)
            patterns['Ascending Triangle']['dates'].append(data.index[i])
            patterns['Ascending Triangle']['prices'].append(data['High'].iloc[i])
        
        # Detect Descending Triangle
        if detect_descending_triangle(window_data):
            patterns['Descending Triangle']['occurrences'].append(i)
            patterns['Descending Triangle']['dates'].append(data.index[i])
            patterns['Descending Triangle']['prices'].append(data['Low'].iloc[i])
    
    # Add last occurrence dates
    for pattern in patterns.values():
        if pattern['occurrences']:
            pattern['last_occurrence'] = data.index[pattern['occurrences'][-1]]
    
    return patterns

def detect_head_and_shoulders(data: pd.DataFrame) -> bool:
    """Detect head and shoulders pattern."""
    if len(data) < 5:
        return False
    
    # Find local maxima
    highs = data['High']
    left_shoulder = highs.iloc[0]
    head = highs.iloc[1]
    right_shoulder = highs.iloc[2]
    
    # Check if pattern conditions are met
    return (left_shoulder < head and right_shoulder < head and
            abs(left_shoulder - right_shoulder) < 0.02 * head)

def detect_double_top(data: pd.DataFrame) -> bool:
    """Detect double top pattern."""
    if len(data) < 3:
        return False
    
    highs = data['High']
    first_top = highs.iloc[0]
    second_top = highs.iloc[1]
    
    return abs(first_top - second_top) < 0.02 * first_top

def detect_double_bottom(data: pd.DataFrame) -> bool:
    """Detect double bottom pattern."""
    if len(data) < 3:
        return False
    
    lows = data['Low']
    first_bottom = lows.iloc[0]
    second_bottom = lows.iloc[1]
    
    return abs(first_bottom - second_bottom) < 0.02 * first_bottom

def detect_triple_top(data: pd.DataFrame) -> bool:
    """Detect triple top pattern."""
    if len(data) < 4:
        return False
    
    highs = data['High']
    first_top = highs.iloc[0]
    second_top = highs.iloc[1]
    third_top = highs.iloc[2]
    
    return (abs(first_top - second_top) < 0.02 * first_top and
            abs(second_top - third_top) < 0.02 * first_top)

def detect_triple_bottom(data: pd.DataFrame) -> bool:
    """Detect triple bottom pattern."""
    if len(data) < 4:
        return False
    
    lows = data['Low']
    first_bottom = lows.iloc[0]
    second_bottom = lows.iloc[1]
    third_bottom = lows.iloc[2]
    
    return (abs(first_bottom - second_bottom) < 0.02 * first_bottom and
            abs(second_bottom - third_bottom) < 0.02 * first_bottom)

def detect_ascending_triangle(data: pd.DataFrame) -> bool:
    """Detect ascending triangle pattern."""
    if len(data) < 5:
        return False
    
    # Check for ascending triangle pattern
    if not (data['High'].iloc[0] < data['High'].iloc[1] < data['High'].iloc[2] < data['High'].iloc[3] < data['High'].iloc[4]):
        return False
    
    # Check for descending volume
    if not (data['Volume'].iloc[0] > data['Volume'].iloc[1] > data['Volume'].iloc[2] > data['Volume'].iloc[3] > data['Volume'].iloc[4]):
        return False
    
    return True

def detect_descending_triangle(data: pd.DataFrame) -> bool:
    """Detect descending triangle pattern."""
    if len(data) < 5:
        return False
    
    # Check for descending triangle pattern
    if not (data['Low'].iloc[0] > data['Low'].iloc[1] > data['Low'].iloc[2] > data['Low'].iloc[3] > data['Low'].iloc[4]):
        return False
    
    # Check for ascending volume
    if not (data['Volume'].iloc[0] < data['Volume'].iloc[1] < data['Volume'].iloc[2] < data['Volume'].iloc[3] < data['Volume'].iloc[4]):
        return False
    
    return True

def display_charts(data: Dict[str, Any], title: str = "Stock Analysis") -> None:
    """Main function to display charts with enhanced layout."""
    apply_shared_styles()
    create_custom_header(title)
    create_custom_divider()
    
    if isinstance(data, dict):
        # Database connection and stock selection
        conn = connect_to_database(data.get("indicator_db", INDICATORS_DB_PATH))
        if conn is None:
            return
        
        stock_list = get_available_stocks(conn)
        if not stock_list:
            st.error("No stock data available.")
            conn.close()
            return
        
        # Stock selection and data loading
        selected_stock = st.selectbox("Select Stock", stock_list)
        time_range = st.slider("Time Range (days)", 30, 365, 180)
        stock_data = get_stock_data(conn, selected_stock, time_range)
        
        if stock_data.empty:
            st.error(f"No data available for {selected_stock}")
            conn.close()
            return
        
        # Calculate advanced indicators
        stock_data = calculate_advanced_indicators(stock_data)
        
        # Create enhanced price and indicators layout
        create_price_indicators_layout(stock_data, selected_stock)
        
        # Add correlation analysis
        st.markdown("### Correlation Analysis")
        correlation_df = calculate_correlation(stock_data, stock_list)
        st.dataframe(correlation_df.sort_values('Correlation', ascending=False))
        
        conn.close()
    elif not isinstance(data, pd.DataFrame):
        st.error("Invalid data type. Expected DataFrame or config dictionary.")
        return
    else:
        # If a DataFrame is provided directly, create a basic chart
        price_chart = create_enhanced_price_chart(data)
        create_chart_container(price_chart, "Price Chart")

def main() -> None:
    """Main function to display the charts component."""
    apply_shared_styles()
    st.set_page_config(page_title="PSX Advanced Charts", layout="wide")
    
    # Add sidebar navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/stocks.png", width=80)
    st.sidebar.title("Chart Navigator")
    
    # Navigation options
    page = st.sidebar.radio(
        "Select View",
        ["Chart Dashboard", "Indicator Explorer", "Market Analysis"]
    )
    
    if page == "Chart Dashboard":
        # Use the config dictionary to load data from the database
        config = {
            "indicator_db": INDICATORS_DB_PATH
        }
        display_charts(config, "PSX Stock Analysis Dashboard")
    
    elif page == "Indicator Explorer":
        create_custom_header("Technical Indicator Explorer")
        create_custom_divider()
        
        # Connect to database
        conn = connect_to_database(INDICATORS_DB_PATH)
        if conn is None:
            st.error("Could not connect to database.")
            return
            
        # Get available stocks
        stock_list = get_available_stocks(conn)
        if not stock_list:
            st.error("No stock data available in the database.")
            conn.close()
            return
            
        # Let user select a stock
        selected_stock = st.selectbox("Select Stock", stock_list)
        
        # Get stock data
        time_range = st.slider("Time Range (days)", 30, 365, 180)
        stock_data = get_stock_data(conn, selected_stock, time_range)
        
        if stock_data.empty:
            st.error(f"No data available for {selected_stock}.")
            conn.close()
            return
        
        # Calculate advanced indicators
        stock_data = calculate_advanced_indicators(stock_data)
        
        # Create enhanced price and indicators layout
        create_price_indicators_layout(stock_data, selected_stock)
        
        conn.close()
    
    elif page == "Market Analysis":
        create_custom_header("Market Analysis")
        create_custom_divider()
        
        # Connect to database
        conn = connect_to_database(INDICATORS_DB_PATH)
        if conn is None:
            st.error("Could not connect to database.")
            return
            
        # Get available stocks
        stock_list = get_available_stocks(conn)
        if not stock_list:
            st.error("No stock data available in the database.")
            conn.close()
            return
        
        # Market sentiment analysis
        st.markdown("### Market Sentiment Analysis")
        sentiment_data = {}
        for stock in stock_list[:10]:  # Limit to first 10 stocks for performance
            stock_data = get_stock_data(conn, stock, 30)  # Last 30 days
            if not stock_data.empty:
                sentiment_data[stock] = calculate_market_sentiment(stock_data)
        
        # Display market sentiment heatmap
        sentiment_df = pd.DataFrame(sentiment_data).T
        fig = px.imshow(
            sentiment_df,
            labels=dict(x="Indicator", y="Stock", color="Sentiment"),
            x=sentiment_df.columns,
            y=sentiment_df.index,
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        conn.close()

def safe_talib_call(func_name, *args, default_value=0, **kwargs):
    """Safely call a TA-Lib function, returning a default value if the function doesn't exist.
    
    Args:
        func_name: Name of the TA-Lib function to call
        *args: Positional arguments to pass to the function
        default_value: Value to return if function fails
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the TA-Lib function or default value if function fails
    """
    try:
        func = getattr(talib, func_name)
        return func(*args, **kwargs)
    except (AttributeError, TypeError):
        # Return a pandas Series or numpy array of the same shape as the first argument
        if len(args) > 0 and hasattr(args[0], 'shape'):
            return pd.Series(default_value, index=args[0].index)
        return default_value

def add_ma_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add Moving Average analysis."""
    current_ma = data[indicator].iloc[-1]
    prev_ma = data[indicator].iloc[-2]
    current_price = data['Close'].iloc[-1]
    
    # MA levels
    st.write("MA Levels:")
    st.write(f"- Current MA: {current_ma:.2f}")
    st.write(f"- Previous MA: {prev_ma:.2f}")
    st.write(f"- Current Price: {current_price:.2f}")
    
    # Price vs MA
    if current_price > current_ma:
        st.success("Price above MA - Bullish")
    else:
        st.warning("Price below MA - Bearish")
    
    # MA trend
    if current_ma > prev_ma:
        st.info("MA trending up")
    else:
        st.info("MA trending down")
    
    # Add MA chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[indicator],
        name=indicator,
        line=dict(color='orange')
    ))
    fig.update_layout(
        title=f"{indicator} Analysis",
        height=200,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def add_rsi_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add RSI-specific analysis."""
    current_rsi = data[indicator].iloc[-1]
    prev_rsi = data[indicator].iloc[-2]
    
    # RSI levels
    st.write("RSI Levels:")
    st.write(f"- Current: {current_rsi:.2f}")
    st.write(f"- Previous: {prev_rsi:.2f}")
    st.write(f"- Change: {current_rsi - prev_rsi:+.2f}")
    
    # RSI interpretation
    if current_rsi > 70:
        st.warning("RSI indicates overbought conditions")
    elif current_rsi < 30:
        st.warning("RSI indicates oversold conditions")
    else:
        st.success("RSI in normal range")
    
    # RSI trend
    if current_rsi > prev_rsi:
        st.info("RSI trending up")
    else:
        st.info("RSI trending down")
    
    # Add RSI chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[indicator],
        name=indicator,
        line=dict(color='blue')
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(
        title=f"{indicator} Analysis",
        height=200,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def add_volume_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add Volume analysis."""
    current_volume = data['Volume'].iloc[-1]
    current_volume_ma = data[indicator].iloc[-1]
    prev_volume = data['Volume'].iloc[-2]
    
    # Volume levels
    st.write("Volume Analysis:")
    st.write(f"- Current Volume: {current_volume:,.0f}")
    st.write(f"- Volume MA: {current_volume_ma:,.0f}")
    st.write(f"- Previous Volume: {prev_volume:,.0f}")
    
    # Volume vs MA
    if current_volume > current_volume_ma:
        st.success("Volume above MA - Strong activity")
    else:
        st.warning("Volume below MA - Weak activity")
    
    # Volume trend
    if current_volume > prev_volume:
        st.info("Volume increasing")
    else:
        st.info("Volume decreasing")
    
    # Add Volume chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color='blue'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[indicator],
        name=indicator,
        line=dict(color='orange')
    ))
    fig.update_layout(
        title="Volume Analysis",
        height=200,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def add_ao_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add detailed Awesome Oscillator analysis."""
    st.markdown("### Awesome Oscillator Analysis")
    
    # Create tabs for different timeframes
    ao_tabs = st.tabs(["Daily", "Weekly", "Monthly"])
    
    with ao_tabs[0]:  # Daily AO
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Daily AO",
                f"{data['AO'].iloc[-1]:.2f}",
                f"{data['AO'].iloc[-1] - data['AO'].iloc[-2]:+.2f}",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "AO Momentum",
                f"{data['AO_Momentum'].iloc[-1]:.2f}",
                "Increasing" if data['AO_Momentum'].iloc[-1] > 0 else "Decreasing"
            )
        
        with col3:
            st.metric(
                "AO Zone",
                data['AO_Zone'].iloc[-1],
                "Strong" if abs(data['AO'].iloc[-1]) > data['AO'].std() else "Weak"
            )
        
        # AO Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['AO_Histogram'],
            name='AO Histogram',
            marker_color=np.where(data['AO_Histogram'] >= 0, 'green', 'red')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['AO'],
            name='AO',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['AO_AVG'],
            name='AO Average',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title='Daily Awesome Oscillator',
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with ao_tabs[1]:  # Weekly AO
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Weekly AO",
                f"{data['AO_weekly'].iloc[-1]:.2f}",
                f"{data['AO_weekly'].iloc[-1] - data['AO_weekly'].iloc[-2]:+.2f}",
            )
        
        with col2:
            st.metric(
                "Weekly AO Momentum",
                f"{data['AO_weekly_Momentum'].iloc[-1]:.2f}",
                "Increasing" if data['AO_weekly_Momentum'].iloc[-1] > 0 else "Decreasing"
            )
        
        with col3:
            st.metric(
                "Weekly AO Zone",
                data['AO_weekly_Zone'].iloc[-1],
                "Strong" if abs(data['AO_weekly'].iloc[-1]) > data['AO_weekly'].std() else "Weak"
            )
        
        # Weekly AO Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['AO_weekly_Histogram'],
            name='Weekly AO Histogram',
            marker_color=np.where(data['AO_weekly_Histogram'] >= 0, 'green', 'red')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['AO_weekly'],
            name='Weekly AO',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['AO_weekly_AVG'],
            name='Weekly AO Average',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title='Weekly Awesome Oscillator',
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with ao_tabs[2]:  # Monthly AO
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Monthly AO",
                f"{data['AO_monthly'].iloc[-1]:.2f}",
                f"{data['AO_monthly'].iloc[-1] - data['AO_monthly'].iloc[-2]:+.2f}",
            )
        
        with col2:
            st.metric(
                "Monthly AO Momentum",
                f"{data['AO_monthly_Momentum'].iloc[-1]:.2f}",
                "Increasing" if data['AO_monthly_Momentum'].iloc[-1] > 0 else "Decreasing"
            )
        
        with col3:
            st.metric(
                "Monthly AO Zone",
                data['AO_monthly_Zone'].iloc[-1],
                "Strong" if abs(data['AO_monthly'].iloc[-1]) > data['AO_monthly'].std() else "Weak"
            )
        
        # Monthly AO Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['AO_monthly_Histogram'],
            name='Monthly AO Histogram',
            marker_color=np.where(data['AO_monthly_Histogram'] >= 0, 'green', 'red')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['AO_monthly'],
            name='Monthly AO',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['AO_monthly_AVG'],
            name='Monthly AO Average',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title='Monthly Awesome Oscillator',
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # AO Analysis Summary
    st.markdown("### AO Analysis Summary")
    
    # Signal Analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Signal Strength**")
        st.write(f"Daily: {data['AO_Signal_Strength'].iloc[-1]:.2f}")
        st.write(f"Weekly: {data['AO_weekly_Signal_Strength'].iloc[-1]:.2f}")
        st.write(f"Monthly: {data['AO_monthly_Signal_Strength'].iloc[-1]:.2f}")
    
    with col2:
        st.write("**Trend Strength**")
        st.write(f"Daily: {data['AO_Trend_Strength'].iloc[-1]:.2f}")
        st.write(f"Weekly: {data['AO_weekly_Trend_Strength'].iloc[-1]:.2f}")
        st.write(f"Monthly: {data['AO_monthly_Trend_Strength'].iloc[-1]:.2f}")
    
    with col3:
        st.write("**Divergence**")
        st.write(f"Daily: {data['AO_Divergence'].iloc[-1]:.2f}")
        st.write(f"Weekly: {data['AO_weekly_Divergence'].iloc[-1]:.2f}")
        st.write(f"Monthly: {data['AO_monthly_Divergence'].iloc[-1]:.2f}")
    
    # Trading Signals
    st.markdown("### Trading Signals")
    
    # Daily Signals
    if data['AO_Crossover'].iloc[-1] == 1 and data['AO_Crossover'].iloc[-2] == -1:
        st.success("Daily AO: Bullish Crossover Signal")
    elif data['AO_Crossover'].iloc[-1] == -1 and data['AO_Crossover'].iloc[-2] == 1:
        st.warning("Daily AO: Bearish Crossover Signal")
    
    # Weekly Signals
    if data['AO_weekly_Crossover'].iloc[-1] == 1 and data['AO_weekly_Crossover'].iloc[-2] == -1:
        st.success("Weekly AO: Bullish Crossover Signal")
    elif data['AO_weekly_Crossover'].iloc[-1] == -1 and data['AO_weekly_Crossover'].iloc[-2] == 1:
        st.warning("Weekly AO: Bearish Crossover Signal")
    
    # Monthly Signals
    if data['AO_monthly_Crossover'].iloc[-1] == 1 and data['AO_monthly_Crossover'].iloc[-2] == -1:
        st.success("Monthly AO: Bullish Crossover Signal")
    elif data['AO_monthly_Crossover'].iloc[-1] == -1 and data['AO_monthly_Crossover'].iloc[-2] == 1:
        st.warning("Monthly AO: Bearish Crossover Signal")

if __name__ == "__main__":
    main() 