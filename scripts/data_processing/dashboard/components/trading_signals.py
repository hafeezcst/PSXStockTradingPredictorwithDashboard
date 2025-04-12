"""
Trading signals component for the PSX dashboard.
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
import os
from datetime import datetime, timedelta
import numpy as np

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

def get_table_data(conn: sqlite3.Connection, table_name: str, limit: int = 100) -> pd.DataFrame:
    """
    Query data from a specific table.
    
    Args:
        conn: SQLite connection
        table_name: Name of the table to query
        limit: Maximum number of rows to return
        
    Returns:
        DataFrame containing the query results
    """
    try:
        query = f"SELECT * FROM {table_name} ORDER BY Date DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error querying {table_name}: {str(e)}")
        return pd.DataFrame()

def get_signal_counts(conn: sqlite3.Connection, days_back: int = 7) -> pd.DataFrame:
    """
    Get daily counts of buy, sell, and neutral signals for the past N days.
    
    Args:
        conn: SQLite connection
        days_back: Number of days to look back
        
    Returns:
        DataFrame with daily signal counts
    """
    try:
        # Get the count of signals for each day in the past N days
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Query for buy signals
        buy_query = f"""
        SELECT Date, COUNT(*) as Buy_Count 
        FROM buy_stocks 
        WHERE Date >= '{cutoff_date}' 
        GROUP BY Date 
        ORDER BY Date
        """
        buy_df = pd.read_sql_query(buy_query, conn)
        
        # Query for sell signals
        sell_query = f"""
        SELECT Date, COUNT(*) as Sell_Count 
        FROM sell_stocks 
        WHERE Date >= '{cutoff_date}' 
        GROUP BY Date 
        ORDER BY Date
        """
        sell_df = pd.read_sql_query(sell_query, conn)
        
        # Query for neutral signals
        neutral_query = f"""
        SELECT Date, COUNT(*) as Neutral_Count 
        FROM neutral_stocks 
        WHERE Date >= '{cutoff_date}' 
        GROUP BY Date 
        ORDER BY Date
        """
        neutral_df = pd.read_sql_query(neutral_query, conn)
        
        # Merge the dataframes
        result = pd.merge(buy_df, sell_df, on='Date', how='outer')
        result = pd.merge(result, neutral_df, on='Date', how='outer')
        
        # Fill NaN values with 0
        result = result.fillna(0)
        
        # Convert counts to integers
        for col in ['Buy_Count', 'Sell_Count', 'Neutral_Count']:
            if col in result.columns:
                result[col] = result[col].astype(int)
        
        return result
    except Exception as e:
        st.error(f"Error getting signal counts: {str(e)}")
        return pd.DataFrame()

def get_signal_performance(conn: sqlite3.Connection, signal_type: str) -> pd.DataFrame:
    """
    Get performance metrics for signals.
    
    Args:
        conn: SQLite connection
        signal_type: Type of signal ('buy', 'sell', or 'neutral')
        
    Returns:
        DataFrame with performance metrics
    """
    try:
        table_name = f"{signal_type}_stocks"
        
        # Query for performance metrics - don't rename the column
        query = f"""
        SELECT 
            Stock,
            Date,
            Signal_Date,
            Signal_Close,
            Close,
            Holding_Days,
            "% P/L",
            Success,
            Status,
            Update_Date
        FROM {table_name}
        WHERE "% P/L" IS NOT NULL
        ORDER BY Date DESC
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Convert date columns to datetime
        date_columns = ['Date', 'Signal_Date', 'Update_Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error getting signal performance: {str(e)}")
        return pd.DataFrame()

def get_signal_details(conn: sqlite3.Connection, signal_type: str, stock: str = None, date: str = None) -> pd.DataFrame:
    """
    Get detailed information for signals with optional filtering.
    
    Args:
        conn: SQLite connection
        signal_type: Type of signal ('buy', 'sell', or 'neutral')
        stock: Stock symbol to filter by
        date: Date to filter by
        
    Returns:
        DataFrame with signal details
    """
    try:
        table_name = f"{signal_type}_stocks"
        
        # Build query with optional filters
        query = f"SELECT * FROM {table_name}"
        conditions = []
        
        if stock:
            conditions.append(f"Stock = '{stock}'")
        
        if date:
            conditions.append(f"Date = '{date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY Date DESC"
        
        df = pd.read_sql_query(query, conn)
        
        # Convert date columns to datetime
        date_columns = ['Date', 'Signal_Date', 'Update_Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error getting signal details: {str(e)}")
        return pd.DataFrame()

def create_signal_trend_chart(signal_counts: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing the trend of buy/sell/neutral signals.
    
    Args:
        signal_counts: DataFrame with daily signal counts
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if 'Buy_Count' in signal_counts.columns:
        fig.add_trace(go.Scatter(
            x=signal_counts['Date'],
            y=signal_counts['Buy_Count'],
            name='Buy Signals',
            line=dict(color='green', width=2)
        ))
    
    if 'Sell_Count' in signal_counts.columns:
        fig.add_trace(go.Scatter(
            x=signal_counts['Date'],
            y=signal_counts['Sell_Count'],
            name='Sell Signals',
            line=dict(color='red', width=2)
        ))
    
    if 'Neutral_Count' in signal_counts.columns:
        fig.add_trace(go.Scatter(
            x=signal_counts['Date'],
            y=signal_counts['Neutral_Count'],
            name='Neutral Signals',
            line=dict(color='gray', width=2)
        ))
    
    fig.update_layout(
        title='Trading Signal Trends',
        xaxis_title='Date',
        yaxis_title='Number of Signals',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_performance_chart(performance_df: pd.DataFrame, metric: str = 'P_L') -> go.Figure:
    """
    Create a chart showing signal performance.
    
    Args:
        performance_df: DataFrame with performance metrics
        metric: Metric to display ('P_L' or 'Holding_Days')
        
    Returns:
        Plotly figure object
    """
    if performance_df.empty:
        return None
    
    if metric == 'P_L':
        title = 'Profit/Loss Distribution'
        y_label = 'Profit/Loss (%)'
        # Use the correct column name
        color = '% P/L'
        color_scale = 'RdYlGn'  # Red for negative, yellow for zero, green for positive
    else:
        title = 'Holding Period Distribution'
        y_label = 'Holding Days'
        color = 'Holding_Days'
        color_scale = 'Blues'
    
    # Determine hover data based on available columns
    hover_data = ['Stock', 'Signal_Date', 'Signal_Close', 'Close', 'Holding_Days', 'Success']
    if '% P/L' in performance_df.columns:
        hover_data.append('% P/L')
    
    fig = px.scatter(
        performance_df,
        x='Date',
        y=color,
        color=color,
        color_continuous_scale=color_scale,
        hover_data=hover_data,
        title=title,
        labels={color: y_label}
    )
    
    fig.update_layout(
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_success_rate_chart(performance_df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing success rate of signals.
    
    Args:
        performance_df: DataFrame with performance metrics
        
    Returns:
        Plotly figure object
    """
    if performance_df.empty or 'Success' not in performance_df.columns:
        return None
    
    # Count success and failure
    success_counts = performance_df['Success'].value_counts()
    
    fig = px.pie(
        values=success_counts.values,
        names=success_counts.index,
        title='Signal Success Rate',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def get_latest_signals(conn: sqlite3.Connection, signal_type: str, limit: int = 5) -> pd.DataFrame:
    """
    Get the latest signals of a specific type.
    
    Args:
        conn: SQLite connection
        signal_type: Type of signal ('buy', 'sell', or 'neutral')
        limit: Maximum number of signals to return
        
    Returns:
        DataFrame with the latest signals
    """
    try:
        table_name = f"{signal_type}_stocks"
        
        # Query for the latest signals
        query = f"""
        SELECT 
            Stock,
            Date,
            Signal_Date,
            Signal_Close,
            Close,
            Holding_Days,
            "% P/L",
            Success,
            Status,
            Update_Date
        FROM {table_name}
        ORDER BY Signal_Date DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Convert date columns to datetime
        date_columns = ['Date', 'Signal_Date', 'Update_Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error getting latest {signal_type} signals: {str(e)}")
        return pd.DataFrame()

def display_trading_signals(config: Dict[str, Any]):
    """Display trading signals analysis."""
    st.subheader("Trading Signals Analysis")
    
    # Get database path from config or use default
    db_path = config.get("trading_signals_db", 
                        "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_investing_Stocks_KMI30.db")
    
    # Check if database exists
    if not os.path.exists(db_path):
        st.error(f"Database not found: {db_path}")
        return
    
    # Connect to the database
    conn = connect_to_database(db_path)
    if conn is None:
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Summary", "Latest Signals", "Buy Signals", "Sell Signals", "Neutral Signals", "Performance Analysis"])
    
    with tab1:
        # Get signal counts for trend chart
        days_back = st.slider("Number of days to analyze", 3, 30, 7, key="summary_days_back")
        signal_counts = get_signal_counts(conn, days_back)
        
        if not signal_counts.empty:
            # Get the latest date in the data
            latest_date = signal_counts['Date'].max()
            
            # Display summary metrics for the latest date
            latest_data = signal_counts[signal_counts['Date'] == latest_date]
            if not latest_data.empty:
                st.markdown(f"### Latest Signals: {latest_date}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    buy_count = latest_data['Buy_Count'].iloc[0] if 'Buy_Count' in latest_data.columns else 0
                    st.metric("Buy Signals", int(buy_count))
                
                with col2:
                    sell_count = latest_data['Sell_Count'].iloc[0] if 'Sell_Count' in latest_data.columns else 0
                    st.metric("Sell Signals", int(sell_count))
                
                with col3:
                    neutral_count = latest_data['Neutral_Count'].iloc[0] if 'Neutral_Count' in latest_data.columns else 0
                    st.metric("Neutral Signals", int(neutral_count))
                
                # Calculate market sentiment
                total = buy_count + sell_count + neutral_count
                if total > 0:
                    buy_pct = (buy_count / total) * 100
                    sell_pct = (sell_count / total) * 100
                    
                    # Determine sentiment
                    if buy_pct > 60:
                        sentiment = "Bullish"
                        color = "green"
                    elif sell_pct > 60:
                        sentiment = "Bearish"
                        color = "red"
                    elif buy_pct > sell_pct:
                        sentiment = "Mildly Bullish"
                        color = "lightgreen"
                    elif sell_pct > buy_pct:
                        sentiment = "Mildly Bearish"
                        color = "lightcoral"
                    else:
                        sentiment = "Neutral"
                        color = "gray"
                    
                    st.markdown(f"**Market Sentiment:** <span style='color:{color}'>{sentiment}</span> (Buy: {buy_pct:.1f}%, Sell: {sell_pct:.1f}%)", unsafe_allow_html=True)
            
            # Display trend chart
            fig = create_signal_trend_chart(signal_counts)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display ratio chart
            signal_counts['Total'] = signal_counts['Buy_Count'] + signal_counts['Sell_Count'] + signal_counts['Neutral_Count']
            signal_counts['Buy_Ratio'] = signal_counts['Buy_Count'] / signal_counts['Total'] * 100
            signal_counts['Sell_Ratio'] = signal_counts['Sell_Count'] / signal_counts['Total'] * 100
            signal_counts['Neutral_Ratio'] = signal_counts['Neutral_Count'] / signal_counts['Total'] * 100
            
            ratio_fig = px.area(
                signal_counts, 
                x='Date', 
                y=['Buy_Ratio', 'Sell_Ratio', 'Neutral_Ratio'],
                title='Signal Composition (%)',
                labels={'value': 'Percentage', 'variable': 'Signal Type'},
                color_discrete_map={
                    'Buy_Ratio': 'green',
                    'Sell_Ratio': 'red',
                    'Neutral_Ratio': 'gray'
                }
            )
            st.plotly_chart(ratio_fig, use_container_width=True)
            
            # Display signal performance summary
            st.markdown("### Signal Performance Summary")
            
            # Get performance data for buy and sell signals
            buy_performance = get_signal_performance(conn, "buy")
            sell_performance = get_signal_performance(conn, "sell")
            
            # Calculate performance metrics
            if not buy_performance.empty:
                buy_success_rate = (buy_performance['Success'] == 'Yes').mean() * 100
                # Use the correct column name
                buy_avg_pl = buy_performance['% P/L'].mean() if '% P/L' in buy_performance.columns else 0
                buy_avg_holding = buy_performance['Holding_Days'].mean()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Buy Success Rate", f"{buy_success_rate:.1f}%")
                
                with col2:
                    st.metric("Avg. Buy P/L", f"{buy_avg_pl:.1f}%")
                
                with col3:
                    st.metric("Avg. Holding Days", f"{buy_avg_holding:.1f}")
            
            if not sell_performance.empty:
                sell_success_rate = (sell_performance['Success'] == 'Yes').mean() * 100
                # Use the correct column name
                sell_avg_pl = sell_performance['% P/L'].mean() if '% P/L' in sell_performance.columns else 0
                sell_avg_holding = sell_performance['Holding_Days'].mean()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sell Success Rate", f"{sell_success_rate:.1f}%")
                
                with col2:
                    st.metric("Avg. Sell P/L", f"{sell_avg_pl:.1f}%")
                
                with col3:
                    st.metric("Avg. Holding Days", f"{sell_avg_holding:.1f}")
    
    with tab2:
        st.markdown("### Latest Trading Signals")
        
        # Get the latest buy and sell signals
        latest_buy_signals = get_latest_signals(conn, "buy", 5)
        latest_sell_signals = get_latest_signals(conn, "sell", 5)
        
        # Display latest buy signals
        st.markdown("#### Latest Buy Signals")
        if not latest_buy_signals.empty:
            # Format the dataframe for display
            display_df = latest_buy_signals.copy()
            
            # Format date columns
            if 'Signal_Date' in display_df.columns:
                display_df['Signal_Date'] = display_df['Signal_Date'].dt.strftime('%Y-%m-%d')
            
            if 'Date' in display_df.columns:
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Format numeric columns
            if 'Signal_Close' in display_df.columns:
                display_df['Signal_Close'] = display_df['Signal_Close'].round(2)
            
            if 'Close' in display_df.columns:
                display_df['Close'] = display_df['Close'].round(2)
            
            if '% P/L' in display_df.columns:
                display_df['% P/L'] = display_df['% P/L'].round(2)
            
            # Select and rename columns for display
            display_columns = {
                'Stock': 'Stock',
                'Signal_Date': 'Signal Date',
                'Signal_Close': 'Signal Price',
                'Close': 'Current Price',
                '% P/L': 'P/L %',
                'Holding_Days': 'Holding Days',
                'Success': 'Success'
            }
            
            # Filter columns that exist in the dataframe
            display_columns = {k: v for k, v in display_columns.items() if k in display_df.columns}
            
            # Rename columns
            display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
            
            # Display the dataframe
            st.dataframe(display_df, use_container_width=True)
            
            # Create a chart showing the latest buy signals
            if 'Signal_Close' in latest_buy_signals.columns and 'Close' in latest_buy_signals.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=latest_buy_signals['Stock'],
                    y=latest_buy_signals['Signal_Close'],
                    name='Signal Price',
                    marker_color='green',
                    text=latest_buy_signals['Signal_Close'].round(2),
                    textposition='auto',
                ))
                
                fig.add_trace(go.Bar(
                    x=latest_buy_signals['Stock'],
                    y=latest_buy_signals['Close'],
                    name='Current Price',
                    marker_color='lightgreen',
                    text=latest_buy_signals['Close'].round(2),
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title='Latest Buy Signals - Price Comparison',
                    xaxis_title='Stock',
                    yaxis_title='Price',
                    barmode='group',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No buy signals available.")
        
        # Display latest sell signals
        st.markdown("#### Latest Sell Signals")
        if not latest_sell_signals.empty:
            # Format the dataframe for display
            display_df = latest_sell_signals.copy()
            
            # Format date columns
            if 'Signal_Date' in display_df.columns:
                display_df['Signal_Date'] = display_df['Signal_Date'].dt.strftime('%Y-%m-%d')
            
            if 'Date' in display_df.columns:
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Format numeric columns
            if 'Signal_Close' in display_df.columns:
                display_df['Signal_Close'] = display_df['Signal_Close'].round(2)
            
            if 'Close' in display_df.columns:
                display_df['Close'] = display_df['Close'].round(2)
            
            if '% P/L' in display_df.columns:
                display_df['% P/L'] = display_df['% P/L'].round(2)
            
            # Select and rename columns for display
            display_columns = {
                'Stock': 'Stock',
                'Signal_Date': 'Signal Date',
                'Signal_Close': 'Signal Price',
                'Close': 'Current Price',
                '% P/L': 'P/L %',
                'Holding_Days': 'Holding Days',
                'Success': 'Success'
            }
            
            # Filter columns that exist in the dataframe
            display_columns = {k: v for k, v in display_columns.items() if k in display_df.columns}
            
            # Rename columns
            display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
            
            # Display the dataframe
            st.dataframe(display_df, use_container_width=True)
            
            # Create a chart showing the latest sell signals
            if 'Signal_Close' in latest_sell_signals.columns and 'Close' in latest_sell_signals.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=latest_sell_signals['Stock'],
                    y=latest_sell_signals['Signal_Close'],
                    name='Signal Price',
                    marker_color='red',
                    text=latest_sell_signals['Signal_Close'].round(2),
                    textposition='auto',
                ))
                
                fig.add_trace(go.Bar(
                    x=latest_sell_signals['Stock'],
                    y=latest_sell_signals['Close'],
                    name='Current Price',
                    marker_color='lightcoral',
                    text=latest_sell_signals['Close'].round(2),
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title='Latest Sell Signals - Price Comparison',
                    xaxis_title='Stock',
                    yaxis_title='Price',
                    barmode='group',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sell signals available.")
    
    with tab3:
        st.markdown("### Buy Signals")
        
        # Get buy signals data
        buy_df = get_table_data(conn, "buy_stocks")
        
        if not buy_df.empty:
            # Create filters
            col1, col2 = st.columns(2)
            
            with col1:
                # Date filter
                dates = sorted(buy_df['Date'].unique(), reverse=True)
                selected_date = st.selectbox("Select Date for Buy Signals:", dates, key="buy_date_filter")
            
            with col2:
                # Stock filter
                stocks = sorted(buy_df['Stock'].unique())
                selected_stock = st.selectbox("Select Stock:", ["All"] + stocks, key="buy_stock_filter")
            
            # Apply filters
            filtered_df = buy_df[buy_df['Date'] == selected_date]
            if selected_stock != "All":
                filtered_df = filtered_df[filtered_df['Stock'] == selected_stock]
            
            if not filtered_df.empty:
                # Display signal details
                st.markdown(f"#### Signal Details for {selected_date}")
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_pl = filtered_df['% P/L'].mean() if '% P/L' in filtered_df.columns else 0
                    st.metric("Average P/L", f"{avg_pl:.1f}%")
                
                with col2:
                    success_rate = (filtered_df['Success'] == 'Yes').mean() * 100 if 'Success' in filtered_df.columns else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col3:
                    avg_holding = filtered_df['Holding_Days'].mean() if 'Holding_Days' in filtered_df.columns else 0
                    st.metric("Avg. Holding Days", f"{avg_holding:.1f}")
                
                with col4:
                    multibagger_count = (filtered_df['Multibagger'] == 'Yes').sum() if 'Multibagger' in filtered_df.columns else 0
                    st.metric("Multibagger Signals", multibagger_count)
                
                # Display detailed dataframe
                st.dataframe(filtered_df, use_container_width=True)
                
                # Display performance chart if available
                if '% P/L' in filtered_df.columns and 'Holding_Days' in filtered_df.columns:
                    st.markdown("#### Performance Visualization")
                    
                    metric = st.radio("Select Metric:", ["P/L", "Holding Days"], key="buy_metric_radio")
                    
                    if metric == "P/L":
                        fig = create_performance_chart(filtered_df, 'P_L')
                    else:
                        fig = create_performance_chart(filtered_df, 'Holding_Days')
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No buy signals for the selected filters.")
        else:
            st.info("No buy signals data available.")
    
    with tab4:
        st.markdown("### Sell Signals")
        
        # Get sell signals data
        sell_df = get_table_data(conn, "sell_stocks")
        
        if not sell_df.empty:
            # Create filters
            col1, col2 = st.columns(2)
            
            with col1:
                # Date filter
                dates = sorted(sell_df['Date'].unique(), reverse=True)
                selected_date = st.selectbox("Select Date for Sell Signals:", dates, key="sell_date_filter")
            
            with col2:
                # Stock filter
                stocks = sorted(sell_df['Stock'].unique())
                selected_stock = st.selectbox("Select Stock:", ["All"] + stocks, key="sell_stock_filter")
            
            # Apply filters
            filtered_df = sell_df[sell_df['Date'] == selected_date]
            if selected_stock != "All":
                filtered_df = filtered_df[filtered_df['Stock'] == selected_stock]
            
            if not filtered_df.empty:
                # Display signal details
                st.markdown(f"#### Signal Details for {selected_date}")
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_pl = filtered_df['% P/L'].mean() if '% P/L' in filtered_df.columns else 0
                    st.metric("Average P/L", f"{avg_pl:.1f}%")
                
                with col2:
                    success_rate = (filtered_df['Success'] == 'Yes').mean() * 100 if 'Success' in filtered_df.columns else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col3:
                    avg_holding = filtered_df['Holding_Days'].mean() if 'Holding_Days' in filtered_df.columns else 0
                    st.metric("Avg. Holding Days", f"{avg_holding:.1f}")
                
                with col4:
                    multibagger_count = (filtered_df['Multibagger'] == 'Yes').sum() if 'Multibagger' in filtered_df.columns else 0
                    st.metric("Multibagger Signals", multibagger_count)
                
                # Display detailed dataframe
                st.dataframe(filtered_df, use_container_width=True)
                
                # Display performance chart if available
                if '% P/L' in filtered_df.columns and 'Holding_Days' in filtered_df.columns:
                    st.markdown("#### Performance Visualization")
                    
                    metric = st.radio("Select Metric:", ["P/L", "Holding Days"], key="sell_metric_radio")
                    
                    if metric == "P/L":
                        fig = create_performance_chart(filtered_df, 'P_L')
                    else:
                        fig = create_performance_chart(filtered_df, 'Holding_Days')
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sell signals for the selected filters.")
        else:
            st.info("No sell signals data available.")
    
    with tab5:
        st.markdown("### Neutral Signals")
        
        # Get neutral signals data
        neutral_df = get_table_data(conn, "neutral_stocks")
        
        if not neutral_df.empty:
            # Create filters
            col1, col2 = st.columns(2)
            
            with col1:
                # Date filter
                dates = sorted(neutral_df['Date'].unique(), reverse=True)
                selected_date = st.selectbox("Select Date for Neutral Signals:", dates, key="neutral_date_filter")
            
            with col2:
                # Stock filter
                stocks = sorted(neutral_df['Stock'].unique())
                selected_stock = st.selectbox("Select Stock:", ["All"] + stocks, key="neutral_stock_filter")
            
            # Apply filters
            filtered_df = neutral_df[neutral_df['Date'] == selected_date]
            if selected_stock != "All":
                filtered_df = filtered_df[filtered_df['Stock'] == selected_stock]
            
            if not filtered_df.empty:
                # Display signal details
                st.markdown(f"#### Signal Details for {selected_date}")
                
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trend_direction = filtered_df['Trend_Direction'].value_counts().index[0] if 'Trend_Direction' in filtered_df.columns else "Unknown"
                    st.metric("Primary Trend", trend_direction)
                
                with col2:
                    avg_rsi = filtered_df['RSI_Weekly_Avg'].mean() if 'RSI_Weekly_Avg' in filtered_df.columns else 0
                    st.metric("Avg. RSI", f"{avg_rsi:.1f}")
                
                with col3:
                    multibagger_count = (filtered_df['Multibagger'] == 'Yes').sum() if 'Multibagger' in filtered_df.columns else 0
                    st.metric("Multibagger Signals", multibagger_count)
                
                # Display detailed dataframe
                st.dataframe(filtered_df, use_container_width=True)
                
                # Display trend direction chart if available
                if 'Trend_Direction' in filtered_df.columns:
                    st.markdown("#### Trend Direction Distribution")
                    
                    trend_counts = filtered_df['Trend_Direction'].value_counts()
                    
                    fig = px.pie(
                        values=trend_counts.values,
                        names=trend_counts.index,
                        title='Trend Direction Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No neutral signals for the selected filters.")
        else:
            st.info("No neutral signals data available.")
    
    with tab6:
        st.markdown("### Performance Analysis")
        
        # Get performance data for buy and sell signals
        buy_performance = get_signal_performance(conn, "buy")
        sell_performance = get_signal_performance(conn, "sell")
        
        # Create tabs for buy and sell performance
        perf_tab1, perf_tab2 = st.tabs(["Buy Performance", "Sell Performance"])
        
        with perf_tab1:
            if not buy_performance.empty:
                st.markdown("#### Buy Signal Performance")
                
                # Display performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    success_rate = (buy_performance['Success'] == 'Yes').mean() * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col2:
                    # Use the correct column name
                    avg_pl = buy_performance['% P/L'].mean() if '% P/L' in buy_performance.columns else 0
                    st.metric("Average P/L", f"{avg_pl:.1f}%")
                
                with col3:
                    avg_holding = buy_performance['Holding_Days'].mean()
                    st.metric("Avg. Holding Days", f"{avg_holding:.1f}")
                
                with col4:
                    # Use the correct column name
                    profitable_signals = (buy_performance['% P/L'] > 0).mean() * 100 if '% P/L' in buy_performance.columns else 0
                    st.metric("Profitable Signals", f"{profitable_signals:.1f}%")
                
                # Display performance charts
                st.markdown("#### Performance Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # P/L distribution
                    fig1 = create_performance_chart(buy_performance, 'P_L')
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Success rate
                    fig2 = create_success_rate_chart(buy_performance)
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Display detailed performance data
                st.markdown("#### Detailed Performance Data")
                st.dataframe(buy_performance, use_container_width=True)
            else:
                st.info("No buy performance data available.")
        
        with perf_tab2:
            if not sell_performance.empty:
                st.markdown("#### Sell Signal Performance")
                
                # Display performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    success_rate = (sell_performance['Success'] == 'Yes').mean() * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col2:
                    # Use the correct column name
                    avg_pl = sell_performance['% P/L'].mean() if '% P/L' in sell_performance.columns else 0
                    st.metric("Average P/L", f"{avg_pl:.1f}%")
                
                with col3:
                    avg_holding = sell_performance['Holding_Days'].mean()
                    st.metric("Avg. Holding Days", f"{avg_holding:.1f}")
                
                with col4:
                    # Use the correct column name
                    profitable_signals = (sell_performance['% P/L'] > 0).mean() * 100 if '% P/L' in sell_performance.columns else 0
                    st.metric("Profitable Signals", f"{profitable_signals:.1f}%")
                
                # Display performance charts
                st.markdown("#### Performance Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # P/L distribution
                    fig1 = create_performance_chart(sell_performance, 'P_L')
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Success rate
                    fig2 = create_success_rate_chart(sell_performance)
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Display detailed performance data
                st.markdown("#### Detailed Performance Data")
                st.dataframe(sell_performance, use_container_width=True)
            else:
                st.info("No sell performance data available.")
    
    # Close the database connection
    conn.close() 