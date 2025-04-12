"""
Dividend analysis component for the PSX dashboard.
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple
import os
from datetime import datetime, timedelta

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
        cursor.execute("SELECT DISTINCT symbol FROM dividend_schedule")
        stocks = cursor.fetchall()
        return sorted([stock[0] for stock in stocks])
    except Exception as e:
        st.error(f"Error getting stock list: {str(e)}")
        return []

def get_dividend_history(conn: sqlite3.Connection, stock_symbol: str) -> pd.DataFrame:
    """
    Get dividend history for a specific stock.
    
    Args:
        conn: SQLite connection
        stock_symbol: Stock symbol to fetch data for
        
    Returns:
        DataFrame containing the dividend history
    """
    try:
        query = f"""
        SELECT * FROM dividend_schedule 
        WHERE symbol = '{stock_symbol}'
        ORDER BY bc_from DESC
        """
        df = pd.read_sql_query(query, conn)
        
        # Convert date columns to datetime
        date_columns = ['bc_from', 'bc_to', 'created_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error getting dividend history for {stock_symbol}: {str(e)}")
        return pd.DataFrame()

def get_upcoming_dividends(conn: sqlite3.Connection, days_ahead: int = 30) -> pd.DataFrame:
    """
    Get upcoming dividend payments.
    
    Args:
        conn: SQLite connection
        days_ahead: Number of days to look ahead
        
    Returns:
        DataFrame containing upcoming dividends
    """
    try:
        cutoff_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        query = f"""
        SELECT * FROM dividend_schedule 
        WHERE bc_from >= date('now') 
        AND bc_from <= '{cutoff_date}'
        ORDER BY bc_from
        """
        df = pd.read_sql_query(query, conn)
        
        # Convert date columns to datetime
        date_columns = ['bc_from', 'bc_to', 'created_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error getting upcoming dividends: {str(e)}")
        return pd.DataFrame()

def create_dividend_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Create timeline visualization of dividend payments.
    
    Args:
        df: DataFrame containing dividend data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add book closure start dates
    fig.add_trace(go.Scatter(
        x=df['bc_from'],
        y=[1] * len(df),
        mode='markers+text',
        name='Book Closure Start',
        text=df['symbol'],
        textposition='top center',
        marker=dict(size=10, color='blue')
    ))
    
    # Add book closure end dates
    fig.add_trace(go.Scatter(
        x=df['bc_to'],
        y=[0.8] * len(df),
        mode='markers+text',
        name='Book Closure End',
        text=df['symbol'],
        textposition='bottom center',
        marker=dict(size=10, color='red')
    ))
    
    fig.update_layout(
        title='Dividend Timeline',
        xaxis_title='Date',
        yaxis=dict(
            showticklabels=False,
            range=[0, 1.2]
        ),
        height=400,
        showlegend=True
    )
    
    return fig

def create_dividend_yield_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create chart showing dividend yield over time.
    
    Args:
        df: DataFrame containing dividend data
        
    Returns:
        Plotly figure object
    """
    # Calculate dividend yield using last_close price
    df['DividendYield'] = df['dividend_amount'] / df['last_close'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['bc_from'],
        y=df['DividendYield'],
        name='Dividend Yield',
        marker_color='gold'
    ))
    
    fig.update_layout(
        title='Dividend Yield History',
        xaxis_title='Book Closure Date',
        yaxis_title='Dividend Yield (%)',
        height=400
    )
    
    return fig

def display_dividend_analysis(config: Dict[str, Any]):
    """Display dividend analysis."""
    st.subheader("Dividend Analysis")
    
    # Get database path from config or use default
    db_path = config.get("dividend_db", 
                        "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_Dividend_Schedule.db")
    
    # Check if database exists
    if not os.path.exists(db_path):
        st.error(f"Database not found: {db_path}")
        return
    
    # Connect to the database
    conn = connect_to_database(db_path)
    if conn is None:
        return
    
    # Sidebar filters
    st.sidebar.subheader("Dividend Analysis Settings")
    
    # Get available stocks
    stock_list = get_available_stocks(conn)
    if not stock_list:
        st.error("No dividend data available in the database.")
        conn.close()
        return
    
    # Stock selection
    selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
    
    # Fetch dividend history
    dividend_history = get_dividend_history(conn, selected_stock)
    if dividend_history.empty:
        st.error(f"No dividend history available for {selected_stock}.")
        conn.close()
        return
    
    # Display stock information header
    latest_dividend = dividend_history.iloc[0]
    
    # Dividend information in a row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Latest Dividend", f"Rs. {latest_dividend['dividend_amount']:.2f}")
    
    with col2:
        st.metric("Face Value", f"Rs. {latest_dividend['face_value']:.2f}")
    
    with col3:
        st.metric("Book Closure From", latest_dividend['bc_from'].strftime('%Y-%m-%d'))
    
    with col4:
        st.metric("Book Closure To", latest_dividend['bc_to'].strftime('%Y-%m-%d'))
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Dividend History", "Upcoming Dividends", "Dividend Analysis"])
    
    with tab1:
        st.subheader("Dividend History")
        
        # Display dividend timeline
        timeline = create_dividend_timeline(dividend_history)
        st.plotly_chart(timeline, use_container_width=True)
        
        # Display dividend history table
        st.dataframe(
            dividend_history[[
                'bc_from', 'bc_to', 'dividend_amount', 'right_amount',
                'payout_text', 'data_type'
            ]].sort_values('bc_from', ascending=False),
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Upcoming Dividends")
        
        # Get upcoming dividends
        upcoming_dividends = get_upcoming_dividends(conn)
        
        if not upcoming_dividends.empty:
            # Display upcoming dividends timeline
            upcoming_timeline = create_dividend_timeline(upcoming_dividends)
            st.plotly_chart(upcoming_timeline, use_container_width=True)
            
            # Display upcoming dividends table
            st.dataframe(
                upcoming_dividends[[
                    'symbol', 'company_name', 'bc_from', 'bc_to',
                    'dividend_amount', 'right_amount', 'payout_text'
                ]].sort_values('bc_from'),
                use_container_width=True
            )
        else:
            st.info("No upcoming dividend payments in the next 30 days.")
    
    with tab3:
        st.subheader("Dividend Analysis")
        
        # Calculate dividend statistics
        total_dividends = len(dividend_history)
        avg_dividend = dividend_history['dividend_amount'].mean()
        max_dividend = dividend_history['dividend_amount'].max()
        min_dividend = dividend_history['dividend_amount'].min()
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Dividends", total_dividends)
        
        with col2:
            st.metric("Average Dividend", f"Rs. {avg_dividend:.2f}")
        
        with col3:
            st.metric("Highest Dividend", f"Rs. {max_dividend:.2f}")
        
        with col4:
            st.metric("Lowest Dividend", f"Rs. {min_dividend:.2f}")
        
        # Display dividend yield chart if price data is available
        if 'last_close' in dividend_history.columns and not dividend_history['last_close'].isna().all():
            yield_chart = create_dividend_yield_chart(dividend_history)
            st.plotly_chart(yield_chart, use_container_width=True)
        
        # Dividend frequency analysis
        st.subheader("Dividend Frequency")
        
        # Calculate months between payments
        dividend_history['MonthsBetween'] = dividend_history['bc_from'].diff().dt.days / 30
        
        # Create frequency distribution
        fig = px.histogram(
            dividend_history,
            x='MonthsBetween',
            title='Dividend Payment Frequency',
            labels={'MonthsBetween': 'Months Between Payments'},
            nbins=20
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display dividend type distribution
        if 'data_type' in dividend_history.columns:
            type_counts = dividend_history['data_type'].value_counts()
            
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title='Dividend Type Distribution'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Close database connection
    conn.close() 