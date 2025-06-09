"""
Enhanced Signal Analysis Component for PSX Stock Trading Predictor Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from pathlib import Path
from datetime import datetime, timedelta
import os
import sys
import sqlite3
import importlib.util

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def calculate_signal_metrics(df):
    """Calculate various signal metrics."""
    metrics = {
        'total_signals': len(df),
        'unique_symbols': df['symbol'].nunique(),
        'signal_distribution': df['signal'].value_counts().to_dict(),
        'avg_confidence': df['confidence'].mean(),
        'avg_score': df['score'].mean(),
        'signal_strength': df.groupby('signal')['confidence'].mean().to_dict(),
        'signal_reliability': df.groupby('signal')['score'].mean().to_dict()
    }
    return metrics

def calculate_signal_correlation(df):
    """Calculate correlation between signals and confidence/score."""
    # Create dummy variables for signals
    signal_dummies = pd.get_dummies(df['signal'])
    correlation = pd.concat([signal_dummies, df[['confidence', 'score']]], axis=1).corr()
    return correlation

def analyze_signal_changes(df):
    """Analyze signal changes and their impact."""
    df = df.sort_values(['symbol', 'date'])
    df['prev_signal'] = df.groupby('symbol')['signal'].shift(1)
    df['signal_change'] = df['signal'] != df['prev_signal']
    
    changes = df[df['signal_change']].copy()
    changes['days_since_change'] = (pd.to_datetime(changes['date']) - 
                                  pd.to_datetime(changes.groupby('symbol')['date'].shift(1))).dt.days
    
    return changes

def safe_metric_difference(current, previous):
    """Safely calculate the difference between current and previous values."""
    if pd.isna(current) or pd.isna(previous):
        return None
    return current - previous

def display_signal_analysis(config):
    """
    Display enhanced signal analysis from the fairvalue database.
    
    Args:
        config (dict): Configuration dictionary containing database paths
    """
    st.header("Enhanced Signal Analysis")
    
    # Create database connection with proper path handling
    db_path = Path(config['database_path']).resolve()
    if not db_path.exists():
        st.error(f"Database file not found at: {db_path}")
        st.info("Please check if the database file exists and the path is correct.")
        return
        
    engine = create_engine(f'sqlite:///{db_path}')
    
    try:
        # First check if the table exists
        with engine.connect() as conn:
            # Get list of all tables
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            st.sidebar.write("Available tables:", tables['name'].tolist())
            
            # Check for stock_signals table
            if 'stock_signals' not in tables['name'].values:
                st.warning("The 'stock_signals' table is not found in the database. Attempting to create it...")
                
                # Import the create_stock_signals module dynamically
                module_path = os.path.join(Path(__file__).parent.parent, "utils", "create_stock_signals.py")
                spec = importlib.util.spec_from_file_location("create_stock_signals", module_path)
                create_stock_signals_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(create_stock_signals_module)
                
                # Try to create the table
                if create_stock_signals_module.create_stock_signals_table(db_path):
                    st.success("Successfully created 'stock_signals' table!")
                    # Refresh tables list
                    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                else:
                    st.error("Failed to create 'stock_signals' table. Using available tables instead.")
            
            # Check for required tables - using available tables in the database
            available_tables = ['buy_stocks', 'sell_stocks', 'neutral_stocks', 'signal_transition_history']
            missing_tables = [table for table in available_tables if table not in tables['name'].values]
            
            if missing_tables:
                st.error(f"Required tables not found: {', '.join(missing_tables)}")
                st.sidebar.error("Please check if the required tables exist in the database.")
                return
            
            # Get table structure for buy_stocks
            result = conn.execute(text("PRAGMA table_info(buy_stocks)"))
            buy_columns = [row[1] for row in result]
            st.sidebar.write("buy_stocks columns:", buy_columns)
            
            # Get table structure for sell_stocks
            result = conn.execute(text("PRAGMA table_info(sell_stocks)"))
            sell_columns = [row[1] for row in result]
            st.sidebar.write("sell_stocks columns:", sell_columns)
            
            # Get a sample row from buy_stocks
            buy_sample = conn.execute(text("SELECT * FROM buy_stocks LIMIT 1")).fetchone()
            if buy_sample:
                st.sidebar.write("Sample buy_stocks data:", dict(zip(buy_columns, buy_sample)))
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Signal Overview", 
                "Signal Changes", 
                "Signal Details",
                "Signal Analysis",
                "Performance Metrics"
            ])
            
            with tab1:
                st.subheader("Signal Overview")
                
                # Get signal distribution
                df_signals = pd.read_sql_query("""
                    WITH all_signals AS (
                        SELECT
                            Stock as symbol, 
                            'BUY' as signal,
                            RSI_Weekly_Avg as confidence,
                            MA_30 as score,
                            COUNT(*) as count
                        FROM buy_stocks
                        GROUP BY Stock
                        
                        UNION ALL
                        
                        SELECT
                            Stock as symbol,
                            'SELL' as signal,
                            RSI_Weekly_Avg as confidence,
                            MA_30 as score,
                            COUNT(*) as count
                        FROM sell_stocks
                        GROUP BY Stock
                        
                        UNION ALL
                        
                        SELECT
                            Stock as symbol,
                            'NEUTRAL' as signal,
                            RSI_Weekly_Avg as confidence,
                            MA_30 as score,
                            COUNT(*) as count
                        FROM neutral_stocks
                        GROUP BY Stock
                    )
                    SELECT
                        signal,
                        SUM(count) as count,
                        AVG(COALESCE(confidence, 0)) as avg_confidence,
                        AVG(COALESCE(score, 0)) as avg_score,
                        COUNT(DISTINCT symbol) as unique_symbols
                    FROM all_signals
                    GROUP BY signal
                    ORDER BY count DESC
                """, engine)
                
                if not df_signals.empty:
                    # Display signal distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create pie chart for signal distribution
                        fig = go.Figure(data=[go.Pie(
                            labels=df_signals['signal'],
                            values=df_signals['count'],
                            hole=.3,
                            textinfo='label+percent+value'
                        )])
                        fig.update_layout(
                            title="Signal Distribution",
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Display signal statistics
                        st.metric("Total Signals", df_signals['count'].sum())
                        st.metric("Unique Symbols", df_signals['unique_symbols'].sum())
                        st.metric("Average Confidence", f"{df_signals['avg_confidence'].mean():.2f}")
                        st.metric("Average Score", f"{df_signals['avg_score'].mean():.2f}")
                    
                    # Signal strength analysis
                    st.subheader("Signal Strength Analysis")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_signals['signal'],
                        y=df_signals['avg_confidence'],
                        name='Confidence'
                    ))
                    fig.add_trace(go.Bar(
                        x=df_signals['signal'],
                        y=df_signals['avg_score'],
                        name='Technical Score'
                    ))
                    fig.update_layout(
                        title="Signal Strength by Type",
                        barmode='group',
                        yaxis_title="Score",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No signals found in the database.")
            
            with tab2:
                st.subheader("Recent Signal Changes")
                
                # Get recent signal changes with enhanced analysis
                df_changes = pd.read_sql_query("""
                    SELECT 
                        Stock as symbol,
                        transition_date as date,
                        Current_Signal as new_signal,
                        Previous_Signal as previous_signal,
                        Current_Close as current_close,
                        Previous_Close as previous_close,
                        Profit_Loss_Pct as profit_loss,
                        Days_In_Signal as days_in_signal,
                        Notes as reasons,
                        CASE 
                            WHEN Profit_Loss_Pct > 0 THEN 'Increasing'
                            WHEN Profit_Loss_Pct < 0 THEN 'Decreasing'
                            ELSE 'Stable'
                        END as confidence_trend
                    FROM signal_transition_history
                    ORDER BY transition_date DESC
                    LIMIT 10
                """, engine)
                
                if not df_changes.empty:
                    for _, row in df_changes.iterrows():
                        with st.expander(f"{row['symbol']} - {row['new_signal']} (from {row['previous_signal']})"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                price_diff = safe_metric_difference(row['current_close'], row['previous_close'])
                                profit_loss = row.get('profit_loss', 0)
                                
                                st.metric("Current Close", 
                                         f"{row['current_close']:.2f}",
                                         f"{price_diff:.2f}" if price_diff is not None else None)
                                st.metric("Profit/Loss %", 
                                         f"{profit_loss:.2f}%" if profit_loss is not None else "N/A")
                            with col2:
                                st.metric("Signal Change", f"{row['previous_signal']} → {row['new_signal']}")
                                st.metric("Days in Signal", row.get('days_in_signal', 'N/A'))
                            with col3:
                                if 'reasons' in row and row['reasons']:
                                    st.write("Notes:", row['reasons'])
                else:
                    st.info("No recent signal changes found.")
            
            with tab3:
                st.subheader("Signal Details")
                
                # Symbol selector with search
                symbols = pd.read_sql_query("""
                    SELECT DISTINCT Stock as symbol 
                    FROM (
                        SELECT Stock FROM buy_stocks
                        UNION 
                        SELECT Stock FROM sell_stocks
                        UNION
                        SELECT Stock FROM neutral_stocks
                    )
                    ORDER BY symbol
                """, engine)['symbol'].tolist()
                
                selected_symbol = st.selectbox("Select Symbol", symbols)
                
                if selected_symbol:
                    # Get signal history for selected symbol with enhanced metrics
                    df_history = pd.read_sql_query(f"""
                        WITH all_signals AS (
                            -- Buy signals
                            SELECT
                                Stock as symbol,
                                Date as date,
                                'BUY' as signal,
                                RSI_Weekly_Avg as confidence,
                                MA_30 as score,
                                'Identified as BUY based on RSI and MA indicators' as reasons
                            FROM buy_stocks
                            WHERE Stock = '{selected_symbol}'
                            
                            UNION ALL
                            
                            -- Sell signals
                            SELECT
                                Stock as symbol,
                                Date as date,
                                'SELL' as signal,
                                RSI_Weekly_Avg as confidence,
                                MA_30 as score,
                                'Identified as SELL based on RSI and MA indicators' as reasons
                            FROM sell_stocks
                            WHERE Stock = '{selected_symbol}'
                            
                            UNION ALL
                            
                            -- Neutral signals
                            SELECT
                                Stock as symbol,
                                Date as date,
                                'NEUTRAL' as signal,
                                RSI_Weekly_Avg as confidence,
                                MA_30 as score,
                                'Identified as NEUTRAL based on RSI and MA indicators' as reasons
                            FROM neutral_stocks
                            WHERE Stock = '{selected_symbol}'
                        ),
                        signal_history AS (
                            SELECT 
                                date,
                                signal,
                                COALESCE(confidence, 0) as confidence,
                                COALESCE(score, 0) as score,
                                reasons,
                                LAG(signal) OVER (ORDER BY date) as prev_signal,
                                LAG(COALESCE(confidence, 0)) OVER (ORDER BY date) as prev_confidence,
                                LAG(COALESCE(score, 0)) OVER (ORDER BY date) as prev_score
                            FROM all_signals
                        )
                        SELECT 
                            *,
                            CASE 
                                WHEN confidence > prev_confidence THEN 'Increasing'
                                WHEN confidence < prev_confidence THEN 'Decreasing'
                                ELSE 'Stable'
                            END as confidence_trend,
                            CASE 
                                WHEN score > prev_score THEN 'Increasing'
                                WHEN score < prev_score THEN 'Decreasing'
                                ELSE 'Stable'
                            END as score_trend
                        FROM signal_history
                        ORDER BY date DESC
                        LIMIT 30
                    """, engine)
                    
                    if not df_history.empty:
                        # Display latest signal with trends
                        latest = df_history.iloc[0]
                        st.subheader(f"Latest Signal for {selected_symbol}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Signal", latest['signal'])
                            confidence_diff = safe_metric_difference(latest['confidence'], latest['prev_confidence'])
                            st.metric("Confidence", 
                                     f"{latest['confidence']:.2f}",
                                     f"{confidence_diff:.2f}" if confidence_diff is not None else None)
                        with col2:
                            score_diff = safe_metric_difference(latest['score'], latest['prev_score'])
                            st.metric("Technical Score", 
                                     f"{latest['score']:.2f}",
                                     f"{score_diff:.2f}" if score_diff is not None else None)
                            st.metric("Confidence Trend", latest['confidence_trend'])
                        with col3:
                            st.metric("Score Trend", latest['score_trend'])
                            if latest['reasons']:
                                st.write("Reasons:", latest['reasons'])
                        
                        # Create enhanced signal history chart
                        fig = make_subplots(rows=3, cols=1, 
                                          shared_xaxes=True,
                                          vertical_spacing=0.05,
                                          subplot_titles=('Confidence History', 
                                                        'Technical Score History',
                                                        'Signal History'),
                                          row_heights=[0.4, 0.4, 0.2])
                        
                        # Add confidence line with trend
                        fig.add_trace(go.Scatter(
                            x=df_history['date'],
                            y=df_history['confidence'],
                            mode='lines+markers',
                            name='Confidence',
                            line=dict(color='blue')
                        ), row=1, col=1)
                        
                        # Add score line with trend
                        fig.add_trace(go.Scatter(
                            x=df_history['date'],
                            y=df_history['score'],
                            mode='lines+markers',
                            name='Technical Score',
                            line=dict(color='green')
                        ), row=2, col=1)
                        
                        # Add signal history with color coding
                        signal_colors = {
                            'STRONG_BUY': 'green',
                            'BUY': 'lightgreen',
                            'NEUTRAL': 'gray',
                            'SELL': 'orange',
                            'STRONG_SELL': 'red'
                        }
                        
                        for signal_type in signal_colors.keys():
                            mask = df_history['signal'] == signal_type
                            if mask.any():
                                fig.add_trace(go.Scatter(
                                    x=df_history[mask]['date'],
                                    y=[signal_type] * mask.sum(),
                                    mode='markers',
                                    name=signal_type,
                                    marker=dict(color=signal_colors[signal_type])
                                ), row=3, col=1)
                        
                        fig.update_layout(
                            title=f"{selected_symbol} Signal History",
                            yaxis_title="Confidence",
                            yaxis2_title="Score",
                            yaxis3_title="Signal",
                            showlegend=True,
                            height=800
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No signal history found for {selected_symbol}")
            
            with tab4:
                st.subheader("Signal Analysis")
                
                # Get all signals for analysis
                df_analysis = pd.read_sql_query("""
                    WITH all_signals AS (
                        -- Buy signals
                        SELECT
                            Stock as symbol,
                            Date as date,
                            'BUY' as signal,
                            RSI_Weekly_Avg as confidence,
                            MA_30 as score,
                            'Identified as BUY based on RSI and MA indicators' as reasons
                        FROM buy_stocks
                        
                        UNION ALL
                        
                        -- Sell signals
                        SELECT
                            Stock as symbol,
                            Date as date,
                            'SELL' as signal,
                            RSI_Weekly_Avg as confidence,
                            MA_30 as score,
                            'Identified as SELL based on RSI and MA indicators' as reasons
                        FROM sell_stocks
                        
                        UNION ALL
                        
                        -- Neutral signals
                        SELECT
                            Stock as symbol,
                            Date as date,
                            'NEUTRAL' as signal,
                            RSI_Weekly_Avg as confidence,
                            MA_30 as score,
                            'Identified as NEUTRAL based on RSI and MA indicators' as reasons
                        FROM neutral_stocks
                    )
                    
                    SELECT 
                        symbol,
                        date,
                        signal,
                        COALESCE(confidence, 0) as confidence,
                        COALESCE(score, 0) as score,
                        reasons
                    FROM all_signals
                    ORDER BY date DESC
                """, engine)
                
                if not df_analysis.empty:
                    # Calculate metrics
                    metrics = calculate_signal_metrics(df_analysis)
                    
                    # Display overall metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Signals", metrics['total_signals'])
                        st.metric("Unique Symbols", metrics['unique_symbols'])
                    with col2:
                        st.metric("Average Confidence", f"{metrics['avg_confidence']:.2f}")
                        st.metric("Average Score", f"{metrics['avg_score']:.2f}")
                    
                    # Signal correlation analysis
                    st.subheader("Signal Correlation Analysis")
                    correlation = calculate_signal_correlation(df_analysis)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation.values,
                        x=correlation.columns,
                        y=correlation.index,
                        colorscale='RdBu',
                        zmin=-1,
                        zmax=1
                    ))
                    fig.update_layout(
                        title="Signal Correlation Matrix",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Signal reliability analysis
                    st.subheader("Signal Reliability Analysis")
                    reliability_df = pd.DataFrame({
                        'Signal': list(metrics['signal_reliability'].keys()),
                        'Reliability': list(metrics['signal_reliability'].values())
                    })
                    
                    fig = go.Figure(data=go.Bar(
                        x=reliability_df['Signal'],
                        y=reliability_df['Reliability']
                    ))
                    fig.update_layout(
                        title="Signal Reliability by Type",
                        yaxis_title="Reliability Score",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for analysis.")
            
            with tab5:
                st.subheader("Performance Metrics")
                
                # Get signal changes for performance analysis
                df_performance = pd.read_sql_query("""
                    SELECT 
                        Stock as symbol,
                        transition_date as date,
                        Current_Signal as signal,
                        Current_Close as confidence,
                        COALESCE(Profit_Loss_Pct, 0) as score,
                        Previous_Signal as prev_signal,
                        Previous_Close as prev_confidence,
                        0 as prev_score
                    FROM signal_transition_history
                    ORDER BY transition_date DESC
                """, engine)
                
                if not df_performance.empty:
                    # Calculate performance metrics
                    performance_metrics = {
                        'total_changes': len(df_performance),
                        'avg_confidence_change': (df_performance['confidence'] - 
                                                df_performance['prev_confidence']).mean(),
                        'avg_score_change': (df_performance['score'] - 
                                           df_performance['prev_score']).mean(),
                        'signal_transitions': df_performance.groupby(['prev_signal', 'signal']).size().unstack()
                    }
                    
                    # Display performance metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Signal Changes", performance_metrics['total_changes'])
                        st.metric("Average Confidence Change", 
                                 f"{performance_metrics['avg_confidence_change']:.2f}")
                    with col2:
                        st.metric("Average Score Change", 
                                 f"{performance_metrics['avg_score_change']:.2f}")
                    
                    # Signal transition matrix
                    st.subheader("Signal Transition Matrix")
                    if 'signal_transitions' in performance_metrics:
                        fig = go.Figure(data=go.Heatmap(
                            z=performance_metrics['signal_transitions'].values,
                            x=performance_metrics['signal_transitions'].columns,
                            y=performance_metrics['signal_transitions'].index,
                            colorscale='Viridis'
                        ))
                        fig.update_layout(
                            title="Signal Transition Matrix",
                            xaxis_title="New Signal",
                            yaxis_title="Previous Signal",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No performance data available.")
    
    except Exception as e:
        st.error(f"Error loading signal analysis: {str(e)}")
        st.info("Please check the database connection and table structure.") 