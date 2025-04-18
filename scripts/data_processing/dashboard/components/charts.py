"""
Charts component for the PSX dashboard.
"""

import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlite3
from PIL import Image
from scripts.data_processing.dashboard.components.shared_styles import (
    apply_shared_styles,
    create_custom_header,
    create_custom_subheader,
    create_custom_divider,
    create_chart_container
)
from typing import Optional, Dict, Any

# Constants for database paths
INDICATORS_DB_PATH = "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/psx_consolidated_data_indicators_PSX.db"
SIGNALS_DB_PATH = "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_investing_Stocks_KMI30_tracking.db"

# Technical indicator categories
INDICATOR_CATEGORIES = {
    "RSI Indicators": [
        "RSI_14", "RSI_14_Avg",
        "RSI_weekly", "RSI_weekly_Avg",
        "RSI_monthly", "RSI_monthly_Avg",
        "RSI_3months", "RSI_3months_Avg",
        "RSI_6months", "RSI_6months_Avg",
        "RSI_annual", "RSI_annual_Avg",
        "RSI_9", "RSI_26"
    ],
    "Moving Averages": [
        "MA_30", "MA_30_weekly", "MA_30_weekly_Avg",
        "MA_50", "MA_50_weekly", "MA_50_weekly_Avg",
        "MA_100", "MA_200"
    ],
    "Volume": [
        "Volume_MA_20"
    ],
    "Other Indicators": [
        "AO", "AO_AVG",
        "AO_weekly", "AO_weekly_AVG",
        "Pct_Change",
        "Daily_Fluctuation"
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

def create_price_indicators_layout(stock_data: pd.DataFrame, selected_stock: str) -> None:
    """Create an enhanced layout for price and indicators section."""
    # Create main layout columns
    chart_col, sidebar_col = st.columns([4, 1])
    
    with chart_col:
        # Stock Header Section
        create_stock_header(stock_data, selected_stock)
        
        # Main Chart Container
        with st.container():
            # Chart Controls
            controls_col1, controls_col2, controls_col3 = st.columns([2, 2, 1])
            with controls_col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Candlestick", "OHLC", "Line", "Area"],
                    key="chart_type"
                )
            with controls_col2:
                timeframe = st.selectbox(
                    "Timeframe",
                    ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "ALL"],
                    index=3,
                    key="timeframe"
                )
            with controls_col3:
                st.button("Refresh", key="refresh_chart")
            
            # Create and display main price chart
            price_chart = create_enhanced_price_chart(stock_data, chart_type)
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Volume Chart
            volume_chart = create_volume_chart(stock_data)
            st.plotly_chart(volume_chart, use_container_width=True)
    
    with sidebar_col:
        create_indicator_sidebar(stock_data)

def create_stock_header(stock_data: pd.DataFrame, selected_stock: str) -> None:
    """Create enhanced stock header with key metrics."""
    latest_price = stock_data['Close'].iloc[-1]
    prev_close = stock_data['Close'].iloc[-2]
    price_change = latest_price - prev_close
    percent_change = (price_change / prev_close * 100)
    
    # Header container
    header_container = st.container()
    with header_container:
        # Stock symbol and price
        st.markdown(f"### {selected_stock}")
        
        # Price metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(
                "Price",
                f"₨ {latest_price:,.2f}",
                f"{price_change:+,.2f} ({percent_change:+.2f}%)",
                delta_color="normal"
            )
        with metrics_col2:
            st.metric(
                "Day Range",
                f"₨ {stock_data['Low'].iloc[-1]:,.2f} - {stock_data['High'].iloc[-1]:,.2f}"
            )
        with metrics_col3:
            st.metric(
                "Volume",
                f"{stock_data['Volume'].iloc[-1]:,.0f}",
                f"{((stock_data['Volume'].iloc[-1] / stock_data['Volume'].iloc[-2]) - 1) * 100:+.1f}%"
            )

def create_enhanced_price_chart(data: pd.DataFrame, chart_type: str = "Candlestick") -> go.Figure:
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

def create_volume_chart(data: pd.DataFrame) -> go.Figure:
    """Create enhanced volume chart."""
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
    
    # Update layout
    fig.update_layout(
        height=200,
        template='plotly_white',
        margin=dict(t=0, b=20, l=50, r=50),
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_indicator_sidebar(data: pd.DataFrame) -> None:
    """Create indicator selection sidebar."""
    st.markdown("### Technical Indicators")
    
    # Create tabs for each category
    indicator_tabs = st.tabs(list(INDICATOR_CATEGORIES.keys()))
    
    for tab, category in zip(indicator_tabs, INDICATOR_CATEGORIES.items()):
        with tab:
            st.markdown(f"#### {category[0]}")
            
            # Filter to only indicators that exist in the data
            available_indicators = [ind for ind in category[1] if ind in data.columns]
            
            if not available_indicators:
                st.info(f"No {category[0]} indicators available")
                continue
                
            # Create multiselect for indicators in this category
            selected_indicators = st.multiselect(
                f"Select {category[0]}",
                options=available_indicators,
                default=available_indicators[:2] if len(available_indicators) >= 2 else available_indicators
            )
            
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

def add_technical_overlays(fig: go.Figure, data: pd.DataFrame) -> None:
    """Add technical analysis overlays to the chart."""
    # Add Moving Averages
    ma_colors = {'MA_30': '#FF9800', 'MA_50': '#2196F3', 'MA_200': '#4CAF50'}
    for ma, color in ma_colors.items():
        if ma in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[ma],
                name=ma,
                line=dict(color=color, width=1),
                visible='legendonly'
            ))

def add_rsi_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add RSI-specific analysis."""
    current_rsi = data[indicator].iloc[-1]
    
    if current_rsi > 70:
        st.warning("Overbought condition (RSI > 70)")
    elif current_rsi < 30:
        st.warning("Oversold condition (RSI < 30)")
    else:
        st.info("Neutral condition")
        
    # Add RSI chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[indicator],
        name=indicator
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)

def add_ma_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add Moving Average analysis."""
    # Add MA chart with price
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[indicator],
        name=indicator,
        line=dict(color='blue')
    ))
    st.plotly_chart(fig, use_container_width=True)

def add_ao_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add Awesome Oscillator analysis."""
    current_ao = data[indicator].iloc[-1]
    
    if current_ao > 0:
        st.success("Bullish signal (AO > 0)")
    else:
        st.error("Bearish signal (AO < 0)")
        
    # Add AO chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[indicator],
        name=indicator
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

def add_volume_analysis(indicator: str, data: pd.DataFrame) -> None:
    """Add Volume analysis."""
    # Add volume chart with MA
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[indicator],
        name=indicator,
        line=dict(color='blue')
    ))
    st.plotly_chart(fig, use_container_width=True)

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
        
        # Create enhanced price and indicators layout
        create_price_indicators_layout(stock_data, selected_stock)
        
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
        ["Chart Dashboard", "Indicator Explorer"]
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
        
        # Create enhanced price and indicators layout
        create_price_indicators_layout(stock_data, selected_stock)
        
        conn.close()

if __name__ == "__main__":
    main() 