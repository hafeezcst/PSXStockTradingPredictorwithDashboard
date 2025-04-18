"""
Technical indicator analysis component for the PSX dashboard.
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple
import os
from datetime import datetime, timedelta

from src.data_processing.dashboard.components.shared_styles import (
    apply_shared_styles,
    create_custom_header,
    create_custom_subheader,
    create_custom_divider,
    create_chart_container,
    create_metric_card
)

# Define indicator categories
INDICATOR_CATEGORIES = {
    "RSI Indicators": ["RSI_14", "RSI_9", "RSI_26", "RSI_weekly", "RSI_monthly"],
    "Moving Averages": ["MA_30", "MA_50", "MA_100", "MA_200", "MA_30_weekly", "MA_50_weekly"],
    "Oscillators": ["AO", "AO_weekly", "AO_monthly", "AO_3Months", "AO_6Months"],
    "Volatility": ["ATR_weekly", "Daily_Fluctuation"],
    "Price Extremes": ["weekly_high", "weekly_low", "monthly_high", "monthly_low"]
}

def connect_to_database(db_path: str) -> sqlite3.Connection:
    """
    Connect to the SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        SQLite connection object
    """
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

def get_available_stocks(conn: sqlite3.Connection) -> List[str]:
    """
    Get list of available stock symbols from the database.
    
    Args:
        conn: SQLite connection
        
    Returns:
        List of stock symbols
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'PSX_%_stock_data'")
        tables = cursor.fetchall()
        # Extract stock symbols from table names
        stock_symbols = [table[0].replace('PSX_', '').replace('_stock_data', '') for table in tables]
        return sorted(stock_symbols)
    except Exception as e:
        st.error(f"Error getting stock list: {str(e)}")
        return []

def get_indicator_data(conn: sqlite3.Connection, stock_symbol: str, days_back: int = 180) -> pd.DataFrame:
    """
    Get technical indicator data for a specific stock.
    
    Args:
        conn: SQLite connection
        stock_symbol: Stock symbol to fetch data for
        days_back: Number of days to fetch
        
    Returns:
        DataFrame containing the indicator data
    """
    try:
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        table_name = f"PSX_{stock_symbol}_stock_data"
        query = f"""
        SELECT * FROM {table_name}
        WHERE Date >= '{cutoff_date}'
        ORDER BY Date
        """
        df = pd.read_sql_query(query, conn)
        
        # Convert Date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    except Exception as e:
        st.error(f"Error getting indicator data for {stock_symbol}: {str(e)}")
        return pd.DataFrame()

def get_market_conditions(conn: sqlite3.Connection, days_back: int = 180) -> pd.DataFrame:
    """
    Get market conditions data.
    
    Args:
        conn: SQLite connection
        days_back: Number of days to fetch
        
    Returns:
        DataFrame containing market condition data
    """
    try:
        # Check if market_conditions table exists, if not create it
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_conditions'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # Create market_conditions table
            cursor.execute("""
                CREATE TABLE market_conditions (
                    date TEXT PRIMARY KEY,
                    score REAL,
                    kmi30_score REAL,
                    kmi100_score REAL,
                    advancing INTEGER,
                    declining INTEGER,
                    unchanged INTEGER
                )
            """)
            
            # Generate some sample market condition data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            current_date = start_date
            
            # Insert sample data
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Generate random scores between 0 and 100
                score = np.random.uniform(30, 70)
                kmi30_score = np.random.uniform(30, 70)
                kmi100_score = np.random.uniform(30, 70)
                
                # Generate random market breadth data
                total_stocks = np.random.randint(100, 200)
                advancing = np.random.randint(30, 70)
                declining = np.random.randint(30, 70)
                unchanged = total_stocks - advancing - declining
                
                cursor.execute("""
                    INSERT INTO market_conditions 
                    (date, score, kmi30_score, kmi100_score, advancing, declining, unchanged)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (date_str, score, kmi30_score, kmi100_score, advancing, declining, unchanged))
                
                current_date += timedelta(days=1)
            
            conn.commit()
            st.info("Created market_conditions table with sample data")
        
        # Now query the table
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        query = f"""
        SELECT * FROM market_conditions
        WHERE date >= '{cutoff_date}'
        ORDER BY date
        """
        df = pd.read_sql_query(query, conn)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    except Exception as e:
        st.error(f"Error getting market conditions: {str(e)}")
        return pd.DataFrame()

def create_price_chart(df: pd.DataFrame, selected_indicators: List[str] = None) -> go.Figure:
    """
    Create price chart with selected indicators.
    
    Args:
        df: DataFrame containing price and indicator data
        selected_indicators: List of indicators to include
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        showlegend=True
    ))
    
    # Add selected indicators as overlays
    if selected_indicators:
        for indicator in selected_indicators:
            if indicator in df.columns:
                # Skip indicators with too many NaN values
                if df[indicator].isna().sum() < len(df) * 0.5:
                    # Use different color for each indicator
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df[indicator],
                        name=indicator,
                        line=dict(width=1.5),
                        visible='legendonly' if 'RSI' in indicator or 'AO' in indicator else True
                    ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker_color='rgba(0, 0, 128, 0.3)',
        yaxis='y2',
        visible='legendonly'
    ))
    
    # Update layout
    fig.update_layout(
        title='Price Chart with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

def create_indicator_chart(df: pd.DataFrame, indicator: str) -> go.Figure:
    """
    Create chart for a specific indicator.
    
    Args:
        df: DataFrame containing indicator data
        indicator: Indicator to plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add indicator line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[indicator],
        name=indicator,
        line=dict(color='blue', width=1.5)
    ))
    
    # Add average line if available
    avg_col = f"{indicator}_Avg"
    if avg_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df[avg_col],
            name=f"{indicator} Avg",
            line=dict(color='orange', width=1.5, dash='dash')
        ))
    
    # Add reference lines based on indicator type
    if 'RSI' in indicator:
        # Add overbought/oversold levels for RSI
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="gray")
        y_range = [0, 100]
    elif 'AO' in indicator:
        # Add zero line for oscillators
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        max_val = df[indicator].abs().max() * 1.1
        y_range = [-max_val, max_val]
    else:
        y_range = None
    
    # Update layout
    fig.update_layout(
        title=f"{indicator} Analysis",
        xaxis_title='Date',
        yaxis_title=indicator,
        yaxis=dict(range=y_range) if y_range else {},
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame, selected_indicators: List[str]) -> go.Figure:
    """
    Create correlation heatmap for selected indicators.
    
    Args:
        df: DataFrame containing indicator data
        selected_indicators: List of indicators to include
        
    Returns:
        Plotly figure object
    """
    # Add price to correlation analysis
    columns_to_include = ['Close'] + selected_indicators
    
    # Filter out columns that don't exist
    available_columns = [col for col in columns_to_include if col in df.columns]
    
    # Calculate correlation matrix
    corr_df = df[available_columns].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="Indicator Correlation Matrix"
    )
    
    fig.update_layout(height=500, width=700)
    
    return fig

def create_indicator_distribution(df: pd.DataFrame, indicator: str) -> go.Figure:
    """
    Create distribution histogram for an indicator.
    
    Args:
        df: DataFrame containing indicator data
        indicator: Indicator to analyze
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[indicator],
        name=indicator,
        nbinsx=30,
        marker_color='royalblue'
    ))
    
    # Add vertical lines for current value and mean
    current_value = df[indicator].iloc[-1]
    mean_value = df[indicator].mean()
    
    fig.add_vline(x=current_value, line_width=2, line_dash="solid", line_color="red",
                 annotation_text=f"Current: {current_value:.2f}")
    fig.add_vline(x=mean_value, line_width=2, line_dash="dash", line_color="green",
                 annotation_text=f"Mean: {mean_value:.2f}")
    
    fig.update_layout(
        title=f"{indicator} Distribution",
        xaxis_title=indicator,
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def calculate_final_signal(rsi_signal: str, ao_signal: str, ma_signal: str) -> str:
    """
    Calculate final trading signal based on RSI, AO, and MA signals.
    
    Args:
        rsi_signal: RSI indicator signal
        ao_signal: Awesome Oscillator signal
        ma_signal: Moving Average signal
        
    Returns:
        Final combined trading signal
    """
    buy_count = 0
    sell_count = 0
    
    # Count buy/sell signals
    if rsi_signal == 'Oversold':
        buy_count += 1
    elif rsi_signal == 'Overbought':
        sell_count += 1
    
    if ao_signal == 'Buy':
        buy_count += 1
    elif ao_signal == 'Sell':
        sell_count += 1
    
    if ma_signal == 'Bullish':
        buy_count += 1
    elif ma_signal == 'Bearish':
        sell_count += 1
    
    # Determine final signal based on majority
    if buy_count > sell_count:
        return 'Buy'
    elif sell_count > buy_count:
        return 'Sell'
    else:
        return 'Neutral'

def analyze_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze technical indicators."""
    # Create explicit copy to avoid warnings
    result_df = df.copy()
    
    # Calculate indicators using .loc
    result_df.loc[:, 'RSI_Signal'] = result_df.apply(lambda row: 
        'Oversold' if row['RSI_14'] < 30 else 
        'Overbought' if row['RSI_14'] > 70 else 'Neutral', axis=1)
    
    result_df.loc[:, 'AO_Signal'] = result_df.apply(lambda row:
        'Buy' if row['AO'] > 0 else 'Sell', axis=1)
    
    result_df.loc[:, 'MA_Signal'] = result_df.apply(lambda row:
        'Bullish' if row['Close'] > row['MA_50'] else 'Bearish', axis=1)
    
    # Create final signal column
    result_df.loc[:, 'Final_Signal'] = result_df.apply(lambda row:
        calculate_final_signal(row['RSI_Signal'], row['AO_Signal'], row['MA_Signal']), axis=1)
    
    return result_df

def display_indicator_analysis(config: Dict[str, Any]):
    """Display technical indicator analysis."""
    apply_shared_styles()
    create_custom_header("Technical Indicator Analysis")
    create_custom_divider()
    
    # Get database path from config or use default
    db_path = config.get("indicator_db", 
                        "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/psx_consolidated_data_indicators_PSX.db")
    
    # Check if database exists
    if not os.path.exists(db_path):
        st.error(f"Database not found: {db_path}")
        return
    
    # Connect to the database
    conn = connect_to_database(db_path)
    if conn is None:
        return
    
    # Sidebar filters
    st.sidebar.subheader("Indicator Analysis Settings")
    
    # Get available stocks
    stock_list = get_available_stocks(conn)
    if not stock_list:
        st.error("No stock data available in the database.")
        conn.close()
        return
    
    # Stock selection
    selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
    
    # Time range selection
    time_range = st.sidebar.slider("Time Range (days)", 30, 365, 180)
    
    # Fetch data
    stock_data = get_indicator_data(conn, selected_stock, time_range)
    if stock_data.empty:
        st.error(f"No data available for {selected_stock}.")
        conn.close()
        return
    
    # Fetch market conditions
    market_conditions = get_market_conditions(conn, time_range)
    
    # Display stock information header
    latest_data = stock_data.iloc[-1]
    prev_data = stock_data.iloc[-2] if len(stock_data) > 1 else None
    
    # Price information in a row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = latest_data['Close']
        price_change = (latest_data['Close'] - prev_data['Close']) / prev_data['Close'] * 100 if prev_data is not None else 0
        create_metric_card("Price", f"{price:.2f}", f"{price_change:.2f}%")
    
    with col2:
        volume = latest_data['Volume']
        volume_change = (latest_data['Volume'] - prev_data['Volume']) / prev_data['Volume'] * 100 if prev_data is not None else 0
        create_metric_card("Volume", f"{volume:,.0f}", f"{volume_change:.2f}%")
    
    with col3:
        rsi = latest_data['RSI_14']
        rsi_change = latest_data['RSI_14'] - prev_data['RSI_14'] if prev_data is not None else 0
        condition = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        create_metric_card("RSI (14)", f"{rsi:.2f} ({condition})", f"{rsi_change:.2f}")
    
    with col4:
        ao = latest_data['AO']
        ao_change = latest_data['AO'] - prev_data['AO'] if prev_data is not None else 0
        create_metric_card("Awesome Oscillator", f"{ao:.2f}", f"{ao_change:.2f}")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Indicators", "Market Context", "Advanced Analysis"])
    
    with tab1:
        # Indicator selection for price chart
        create_custom_subheader("Price Chart with Indicators")
        
        # Group indicators by category for easier selection
        selected_indicators = []
        for category, indicators in INDICATOR_CATEGORIES.items():
            st.markdown(f"**{category}**")
            # Filter indicators that are in the dataframe
            available_indicators = [ind for ind in indicators if ind in stock_data.columns]
            if available_indicators:
                category_selection = st.multiselect(
                    f"Select {category}",
                    available_indicators,
                    key=f"ms_{category}"
                )
                selected_indicators.extend(category_selection)
        
        # Create and display price chart
        price_chart = create_price_chart(stock_data, selected_indicators)
        create_chart_container(price_chart, "Price and Indicators")
    
    with tab2:
        create_custom_subheader("Detailed Indicator Analysis")
        
        # Create indicator categories with expanders
        for category, indicators in INDICATOR_CATEGORIES.items():
            with st.expander(category, expanded=(category == "RSI Indicators")):
                # Get available indicators in this category
                available_indicators = [ind for ind in indicators if ind in stock_data.columns]
                
                if available_indicators:
                    for indicator in available_indicators:
                        # Create indicator chart
                        indicator_chart = create_indicator_chart(stock_data, indicator)
                        create_chart_container(indicator_chart, f"{indicator} Analysis")
                        
                        # Show distribution for this indicator
                        show_dist = st.checkbox(f"Show {indicator} Distribution", key=f"dist_{indicator}")
                        if show_dist:
                            dist_chart = create_indicator_distribution(stock_data, indicator)
                            create_chart_container(dist_chart, f"{indicator} Distribution")
                else:
                    st.info(f"No {category} indicators available for {selected_stock}")
    
    with tab3:
        create_custom_subheader("Market Context")
        
        # Display market conditions if available
        if not market_conditions.empty:
            # Create market conditions timeline
            market_fig = px.line(
                market_conditions, 
                x='date', 
                y=['score', 'kmi30_score', 'kmi100_score'],
                title='Market Condition Scores',
                labels={'value': 'Score', 'variable': 'Index'}
            )
            create_chart_container(market_fig, "Market Condition Scores")
            
            # Market breadth
            breadth_df = market_conditions[['date', 'advancing', 'declining', 'unchanged']]
            breadth_df['advance_decline_ratio'] = breadth_df['advancing'] / breadth_df['declining']
            
            breadth_fig = px.bar(
                breadth_df,
                x='date',
                y=['advancing', 'declining', 'unchanged'],
                title='Market Breadth',
                labels={'value': 'Number of Stocks', 'variable': 'Direction'}
            )
            create_chart_container(breadth_fig, "Market Breadth")
            
            # Show correlation with market
            if 'Close' in stock_data.columns and 'score' in market_conditions.columns:
                # Merge stock data with market conditions
                merged_df = pd.merge_asof(
                    stock_data[['Date', 'Close', 'RSI_14']], 
                    market_conditions[['date', 'score']].rename(columns={'date': 'Date'}),
                    on='Date'
                )
                
                if not merged_df.empty:
                    # Calculate correlation
                    corr = merged_df[['Close', 'score']].corr().iloc[0, 1]
                    st.info(f"Correlation between {selected_stock} price and market conditions: {corr:.2f}")
                    
                    # Plot correlation
                    fig = px.scatter(
                        merged_df,
                        x='score',
                        y='Close',
                        title=f"{selected_stock} Price vs Market Conditions",
                        trendline="ols"
                    )
                    create_chart_container(fig, "Price-Market Correlation")
        else:
            st.info("No market condition data available.")
    
    with tab4:
        create_custom_subheader("Advanced Technical Analysis")
        
        # Correlation heatmap
        st.markdown("### Indicator Correlations")
        st.info("This analysis helps identify which indicators are providing similar or contrasting signals.")
        
        # Select indicators for correlation analysis
        flat_indicators = [ind for sublist in INDICATOR_CATEGORIES.values() for ind in sublist]
        available_indicators = [ind for ind in flat_indicators if ind in stock_data.columns]
        
        selected_corr_indicators = st.multiselect(
            "Select Indicators for Correlation Analysis",
            available_indicators,
            default=available_indicators[:5] if len(available_indicators) >= 5 else available_indicators
        )
        
        if selected_corr_indicators:
            corr_heatmap = create_correlation_heatmap(stock_data, selected_corr_indicators)
            create_chart_container(corr_heatmap, "Indicator Correlation Matrix")
        
        # Trading signals
        create_custom_subheader("Trading Signals Analysis")
        st.info("This analysis identifies potential trading signals based on technical indicators.")
        
        # Generate simple trading signals based on indicators
        signals = pd.DataFrame(index=stock_data.index)
        signals['Date'] = stock_data['Date']
        signals['Price'] = stock_data['Close']
        
        # RSI signals
        if 'RSI_14' in stock_data.columns:
            signals['RSI_Signal'] = 'Neutral'
            signals.loc[stock_data['RSI_14'] < 30, 'RSI_Signal'] = 'Oversold (Buy)'
            signals.loc[stock_data['RSI_14'] > 70, 'RSI_Signal'] = 'Overbought (Sell)'
        
        # Moving Average signals
        if 'MA_50' in stock_data.columns and 'MA_200' in stock_data.columns:
            signals['MA_Signal'] = 'Neutral'
            signals.loc[stock_data['MA_50'] > stock_data['MA_200'], 'MA_Signal'] = 'Golden Cross (Bullish)'
            signals.loc[stock_data['MA_50'] < stock_data['MA_200'], 'MA_Signal'] = 'Death Cross (Bearish)'
        
        # Awesome Oscillator signals
        if 'AO' in stock_data.columns:
            signals['AO_Signal'] = 'Neutral'
            signals.loc[stock_data['AO'] > 0, 'AO_Signal'] = 'Bullish'
            signals.loc[stock_data['AO'] < 0, 'AO_Signal'] = 'Bearish'
            # Find zero crossings (more significant signals)
            signals['AO_Prev'] = stock_data['AO'].shift(1)
            crossover_up = (signals['AO_Prev'] < 0) & (stock_data['AO'] > 0)
            crossover_down = (signals['AO_Prev'] > 0) & (stock_data['AO'] < 0)
            signals.loc[crossover_up, 'AO_Signal'] = 'Zero Crossover Up (Strong Buy)'
            signals.loc[crossover_down, 'AO_Signal'] = 'Zero Crossover Down (Strong Sell)'
            signals.drop('AO_Prev', axis=1, inplace=True)
        
        # Display recent signals
        st.dataframe(signals.tail(20).sort_values('Date', ascending=False), use_container_width=True)
        
        # Composite signal
        create_custom_subheader("Composite Signal Analysis")
        st.info("This combines multiple indicators to generate a more robust trading signal.")
        
        # Create composite score
        composite_signals = pd.DataFrame(index=stock_data.index)
        composite_signals['Date'] = stock_data['Date']
        composite_signals['Price'] = stock_data['Close']
        
        # Initialize score
        composite_signals['Score'] = 0
        
        # RSI component (0-100 scale to -1 to 1 scale)
        if 'RSI_14' in stock_data.columns:
            # Convert RSI to -1 to 1 scale (30=bearish, 50=neutral, 70=bullish)
            composite_signals['RSI_Component'] = (stock_data['RSI_14'] - 50) / 20
            composite_signals['RSI_Component'] = composite_signals['RSI_Component'].clip(-1, 1)
            composite_signals['Score'] += composite_signals['RSI_Component']
        
        # AO component
        if 'AO' in stock_data.columns and 'AO_AVG' in stock_data.columns:
            # Normalize AO by its average
            max_ao = stock_data['AO'].abs().max()
            if max_ao > 0:
                composite_signals['AO_Component'] = stock_data['AO'] / max_ao
                composite_signals['Score'] += composite_signals['AO_Component']
        
        # MA component (price vs MA)
        if 'MA_50' in stock_data.columns:
            # Above MA is bullish (max +1), below is bearish (max -1)
            composite_signals['MA_Component'] = (stock_data['Close'] / stock_data['MA_50'] - 1).clip(-1, 1)
            composite_signals['Score'] += composite_signals['MA_Component']
        
        # Normalize final score
        if 'Score' in composite_signals.columns:
            # Count number of components
            n_components = sum(['_Component' in col for col in composite_signals.columns])
            if n_components > 0:
                composite_signals['Score'] = composite_signals['Score'] / n_components
            
            # Add signal interpretation
            composite_signals['Signal'] = 'Neutral'
            composite_signals.loc[composite_signals['Score'] > 0.5, 'Signal'] = 'Strong Buy'
            composite_signals.loc[(composite_signals['Score'] <= 0.5) & (composite_signals['Score'] > 0.2), 'Signal'] = 'Buy'
            composite_signals.loc[(composite_signals['Score'] <= 0.2) & (composite_signals['Score'] > -0.2), 'Signal'] = 'Neutral'
            composite_signals.loc[(composite_signals['Score'] <= -0.2) & (composite_signals['Score'] > -0.5), 'Signal'] = 'Sell'
            composite_signals.loc[composite_signals['Score'] <= -0.5, 'Signal'] = 'Strong Sell'
            
            # Create composite score chart
            fig = go.Figure()
            
            # Add price
            fig.add_trace(go.Scatter(
                x=composite_signals['Date'],
                y=composite_signals['Price'],
                name='Price',
                yaxis='y2'
            ))
            
            # Add score
            fig.add_trace(go.Bar(
                x=composite_signals['Date'],
                y=composite_signals['Score'],
                name='Composite Score',
                marker_color=composite_signals['Score'].apply(
                    lambda x: 'green' if x > 0.2 else 'red' if x < -0.2 else 'gray'
                )
            ))
            
            fig.update_layout(
                title='Composite Technical Score',
                xaxis_title='Date',
                yaxis_title='Score (-1 to 1)',
                yaxis2=dict(
                    title='Price',
                    overlaying='y',
                    side='right'
                ),
                height=500,
                hovermode='x unified'
            )
            
            create_chart_container(fig, "Composite Score Analysis")
            
            # Display the latest signal
            latest_score = composite_signals['Score'].iloc[-1]
            latest_signal = composite_signals['Signal'].iloc[-1]
            
            # Color the signal
            color = "green" if latest_score > 0.2 else "red" if latest_score < -0.2 else "gray"
            st.markdown(f"### Current Signal: <span style='color:{color}'>{latest_signal}</span> (Score: {latest_score:.2f})", unsafe_allow_html=True)
            
            # Display breakdown of components
            component_cols = [col for col in composite_signals.columns if '_Component' in col]
            if component_cols:
                create_custom_subheader("Signal Components")
                component_values = {col.replace('_Component', ''): composite_signals[col].iloc[-1] for col in component_cols}
                
                # Create component chart
                comp_fig = go.Figure()
                for comp, value in component_values.items():
                    comp_fig.add_trace(go.Bar(
                        x=[comp],
                        y=[value],
                        name=comp,
                        marker_color='green' if value > 0 else 'red'
                    ))
                
                comp_fig.update_layout(
                    title='Signal Component Breakdown',
                    yaxis_title='Component Contribution (-1 to 1)',
                    height=400
                )
                
                create_chart_container(comp_fig, "Component Breakdown")
        
    # Close database connection
    conn.close()