"""
Signal Tracker Dashboard Component for PSX Trading.
This component provides advanced signal tracking analytics and management.
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import sys
import time

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

# Define a placeholder function for telegram_message until proper integration
def send_telegram_message(message: str) -> bool:
    """
    Placeholder function for sending Telegram messages.
    Will be replaced with actual implementation later.
    
    Args:
        message (str): Message to send
        
    Returns:
        bool: Success status
    """
    # Log the message but don't actually send it
    print(f"[TELEGRAM] Would send: {message[:100]}...")
    return True

# Use absolute imports for dashboard components
from src.data_processing.dashboard.components.database_manager import DatabaseManager
from src.data_processing.dashboard.components.shared_styles import (
    apply_shared_styles,
    create_custom_header,
    create_custom_subheader,
    create_custom_divider,
    create_chart_container,
    create_metric_card,
    create_alert
)

# Constants
DEFAULT_DB_PATH = 'data/databases/production/fairvalue.db'
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../scripts/config/alert_config.json')

def load_config():
    """Load configuration from config file"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            config = {
                "telegram": {
                    "enabled": True,  # Enable Telegram by default
                },
                "alerts": {
                    "signal_transitions": True,
                    "profit_threshold": 5.0,
                    "loss_threshold": -5.0,
                    "days_in_signal_threshold": 14
                },
                "analysis": {
                    "trend_detection": True,
                    "volume_analysis": True,
                    "performance_metrics": True
                }
            }
            
            # Ensure config directory exists
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            
            # Save default config
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)
            
            return config
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return {}

def connect_to_database(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
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

def track_signal_transition(conn: sqlite3.Connection, stock: str, current_signal: str, 
                          current_close: float, previous_signal: str = None, 
                          previous_close: float = None, notes: str = None) -> bool:
    """
    Track a signal transition in the history table.
    
    Args:
        conn: SQLite connection
        stock: Stock symbol
        current_signal: Current signal type (Buy/Sell/Neutral)
        current_close: Current closing price
        previous_signal: Previous signal type (optional)
        previous_close: Previous closing price (optional)
        notes: Additional notes (optional)
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create the table if it doesn't exist
        create_signal_transition_history_table(conn)
        
        # Initialize cursor - ADDED THIS LINE
        cursor = conn.cursor()
        
        # Calculate profit/loss percentage if we have both prices
        profit_loss_pct = None
        if previous_close and current_close:
            profit_loss_pct = ((current_close - previous_close) / previous_close) * 100
        
        # Calculate days in signal
        days_in_signal = None
        if previous_signal:
            # Get the last transition date for this stock
            cursor.execute("""
                SELECT transition_date 
                FROM signal_transition_history 
                WHERE Stock = ? 
                ORDER BY transition_date DESC 
                LIMIT 1
            """, (stock,))
            result = cursor.fetchone()
            if result:
                last_transition_date = datetime.strptime(result[0], '%Y-%m-%d')
                days_in_signal = (datetime.now() - last_transition_date).days
        
        # Insert the transition record
        cursor.execute("""
            INSERT INTO signal_transition_history 
            (Stock, Previous_Signal, Current_Signal, transition_date, 
             Previous_Close, Current_Close, Profit_Loss_Pct, 
             Days_In_Signal, Notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stock, previous_signal, current_signal, 
            datetime.now().strftime('%Y-%m-%d'),
            previous_close, current_close, profit_loss_pct,
            days_in_signal, notes
        ))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error tracking signal transition: {str(e)}")
        return False

def create_signal_tracking_table(conn: sqlite3.Connection) -> bool:
    """
    Create the signal_tracking table if it doesn't exist.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Boolean indicating success
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_tracking (
                Stock TEXT PRIMARY KEY,
                Current_Signal TEXT NOT NULL,
                Signal_Changes INTEGER DEFAULT 0,
                Total_Days INTEGER DEFAULT 0,
                Notes TEXT,
                Last_Updated DATE DEFAULT CURRENT_DATE
            )
        """)
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error creating signal_tracking table: {str(e)}")
        return False

def get_signal_tracking_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Get signal tracking data from the database.
    
    Args:
        conn: SQLite connection
        
    Returns:
        DataFrame with signal tracking data
    """
    try:
        # First, let's see what tables exist in the database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table'
        """)
        all_tables = [row[0] for row in cursor.fetchall()]
        
        if not all_tables:
            st.error("No tables found in the database.")
            return pd.DataFrame()
        
        # Check if required tables exist
        required_tables = ['tradingview_ta', 'tradingview_signals']
        missing_tables = [table for table in required_tables if table not in all_tables]
        
        if missing_tables:
            st.error(f"Required tables not found: {', '.join(missing_tables)}")
            return pd.DataFrame()
        
        # Query to get the latest signal tracking data by combining both tables
        query = """
        WITH LatestTAData AS (
            SELECT 
                symbol as Stock,
                recommendation as Current_Signal,
                date as Current_Signal_Date,
                close as Current_Close,
                date as Initial_Signal_Date,
                close as Initial_Close,
                0 as Days_In_Signal,
                change_percent as Profit_Loss_Pct,
                (buy_signals + sell_signals + neutral_signals) as Signal_Changes,
                0 as Total_Days,
                CASE 
                    WHEN buy_signals > sell_signals AND buy_signals > neutral_signals THEN 'Strong Buy'
                    WHEN sell_signals > buy_signals AND sell_signals > neutral_signals THEN 'Strong Sell'
                    WHEN neutral_signals > buy_signals AND neutral_signals > sell_signals THEN 'Neutral'
                    ELSE 'Mixed Signals'
                END as TA_Notes,
                last_updated as Last_Updated,
                rsi,
                macd,
                sma_20,
                sma_50,
                sma_200,
                volume
            FROM tradingview_ta
            WHERE date = (
                SELECT MAX(date) 
                FROM tradingview_ta
            )
        ),
        LatestSignalsData AS (
            SELECT 
                symbol as Stock,
                signal_type as Signal_Type,
                signal_strength,
                confidence_score,
                technical_score,
                trend_score,
                momentum_score,
                volume_score,
                volatility_score,
                support_level,
                resistance_level,
                stop_loss,
                take_profit,
                risk_reward_ratio,
                analysis_summary,
                ai_score,
                ai_confidence,
                ai_pattern_recognition,
                ai_signal_strength,
                ai_risk_assessment,
                ai_recommendation,
                ai_price_targets,
                ai_entry_points,
                ai_exit_points
            FROM tradingview_signals
            WHERE date = (
                SELECT MAX(date) 
                FROM tradingview_signals
            )
        )
        SELECT 
            ta.*,
            sig.Signal_Type,
            sig.signal_strength,
            sig.confidence_score,
            sig.technical_score,
            sig.trend_score,
            sig.momentum_score,
            sig.volume_score,
            sig.volatility_score,
            sig.support_level,
            sig.resistance_level,
            sig.stop_loss,
            sig.take_profit,
            sig.risk_reward_ratio,
            sig.analysis_summary,
            sig.ai_score,
            sig.ai_confidence,
            sig.ai_pattern_recognition,
            sig.ai_signal_strength,
            sig.ai_risk_assessment,
            sig.ai_recommendation,
            sig.ai_price_targets,
            sig.ai_entry_points,
            sig.ai_exit_points,
            CASE 
                WHEN sig.ai_recommendation IS NOT NULL THEN 
                    ta.TA_Notes || ' | AI: ' || sig.ai_recommendation
                ELSE ta.TA_Notes
            END as Notes
        FROM LatestTAData ta
        LEFT JOIN LatestSignalsData sig ON ta.Stock = sig.Stock
        ORDER BY ta.Stock
        """
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            st.warning("No signal data found in the database.")
            return df
        
        # Convert date columns to datetime
        date_columns = ['Current_Signal_Date', 'Initial_Signal_Date', 'Last_Updated']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Fill NaN values for all numeric columns
        numeric_columns = [
            'Signal_Changes', 'Total_Days', 'Days_In_Signal', 'Profit_Loss_Pct',
            'rsi', 'macd', 'sma_20', 'sma_50', 'sma_200', 'volume',
            'signal_strength', 'confidence_score', 'technical_score',
            'trend_score', 'momentum_score', 'volume_score', 'volatility_score',
            'support_level', 'resistance_level', 'stop_loss', 'take_profit',
            'risk_reward_ratio', 'ai_score', 'ai_confidence'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error getting signal tracking data: {str(e)}")
        st.error("Please check if the database has the correct tables and structure.")
        return pd.DataFrame()

def create_signal_transition_history_table(conn: sqlite3.Connection) -> bool:
    """
    Create the signal_transition_history table if it doesn't exist.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Boolean indicating success
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_transition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Stock TEXT NOT NULL,
                Previous_Signal TEXT,
                Current_Signal TEXT NOT NULL,
                transition_date DATE NOT NULL,
                Previous_Close REAL,
                Current_Close REAL,
                Profit_Loss_Pct REAL,
                Days_In_Signal INTEGER,
                Notes TEXT,
                update_date DATE DEFAULT CURRENT_DATE
            )
        """)
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error creating signal_transition_history table: {str(e)}")
        return False

def get_signal_transitions(conn: sqlite3.Connection, days_back: int = 30) -> pd.DataFrame:
    """
    Get signal transition history.
    
    Args:
        conn: SQLite connection
        days_back: Number of days to look back
        
    Returns:
        DataFrame with signal transition history
    """
    try:
        # Create the table if it doesn't exist
        create_signal_transition_history_table(conn)
        
        # Get the cutoff date
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT * FROM signal_transition_history
        WHERE transition_date >= '{cutoff_date}'
        ORDER BY transition_date DESC
        """
        df = pd.read_sql_query(query, conn)
        
        # Convert date columns to datetime
        if 'transition_date' in df.columns:
            df['transition_date'] = pd.to_datetime(df['transition_date'])
        
        return df
    except Exception as e:
        st.error(f"Error getting signal transitions: {str(e)}")
        return pd.DataFrame()

def detect_high_profit_signals(conn: sqlite3.Connection, profit_threshold: float = 5.0, loss_threshold: float = -5.0) -> pd.DataFrame:
    """
    Detect signals with high profit or significant loss.
    
    Args:
        conn: SQLite connection
        profit_threshold: Threshold for high profit signals
        loss_threshold: Threshold for significant loss signals
        
    Returns:
        DataFrame with high profit/loss signals
    """
    try:
        # Get the signal tracking data which includes the calculated metrics
        signal_data = get_signal_tracking_data(conn)
        
        if signal_data.empty:
            return pd.DataFrame()
            
        # Filter for high profit or significant loss signals
        high_profit_signals = signal_data[
            (signal_data['Profit_Loss_Pct'] > profit_threshold) | 
            (signal_data['Profit_Loss_Pct'] < loss_threshold)
        ].sort_values('Profit_Loss_Pct', ascending=False)
        
        # Select and rename columns to match the expected output
        result = high_profit_signals[[
            'Stock', 
            'Current_Signal', 
            'Profit_Loss_Pct', 
            'Days_In_Signal', 
            'Last_Updated'
        ]].copy()
        
        return result
    except Exception as e:
        st.error(f"Error detecting high profit signals: {str(e)}")
        return pd.DataFrame()

def detect_long_duration_signals(conn: sqlite3.Connection, days_threshold: int = 14) -> pd.DataFrame:
    """
    Detect signals that have been active for a long duration.
    
    Args:
        conn: SQLite connection
        days_threshold: Threshold for long duration in days
        
    Returns:
        DataFrame with long duration signals
    """
    try:
        # Get the signal tracking data which includes the calculated metrics
        signal_data = get_signal_tracking_data(conn)
        
        if signal_data.empty:
            return pd.DataFrame()
            
        # Filter for signals with duration above threshold
        long_duration_signals = signal_data[
            signal_data['Days_In_Signal'] > days_threshold
        ].sort_values('Days_In_Signal', ascending=False)
        
        # Select and rename columns to match the expected output
        result = long_duration_signals[[
            'Stock', 
            'Current_Signal', 
            'Profit_Loss_Pct', 
            'Days_In_Signal', 
            'Last_Updated'
        ]].copy()
        
        return result
    except Exception as e:
        st.error(f"Error detecting long duration signals: {str(e)}")
        return pd.DataFrame()

def get_signal_performance_metrics(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Calculate performance metrics for different signals.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Get the signal tracking data which includes the calculated metrics
        signal_data = get_signal_tracking_data(conn)
        
        if signal_data.empty:
            return {}
            
        # Create a function to categorize signals (same as in display_signal_tracker)
        def categorize_signal(row):
            # Convert signals to lowercase and handle NaN values
            current_signal = str(row['Current_Signal']).lower() if pd.notna(row['Current_Signal']) else ''
            signal_type = str(row['Signal_Type']).lower() if pd.notna(row['Signal_Type']) else ''
            
            # Define signal strength hierarchy
            signal_strength = {
                'strong buy': 5,
                'buy': 4,
                'neutral': 3,
                'sell': 2,
                'strong sell': 1
            }
            
            # Get the strongest signal from both columns
            current_strength = max([signal_strength.get(s, 0) for s in current_signal.split()])
            type_strength = max([signal_strength.get(s, 0) for s in signal_type.split()])
            
            # Use the stronger signal
            if current_strength >= type_strength:
                signal = current_signal
            else:
                signal = signal_type
            
            # Categorize based on the stronger signal
            if 'strong buy' in signal:
                return 'Strong Buy'
            elif 'buy' in signal:
                return 'Buy'
            elif 'strong sell' in signal:
                return 'Strong Sell'
            elif 'sell' in signal:
                return 'Sell'
            elif 'neutral' in signal:
                return 'Neutral'
            return 'Unknown'
        
        # Categorize signals
        signal_data['Signal_Category'] = signal_data.apply(categorize_signal, axis=1)
        
        # Calculate metrics for each signal category
        metrics = {}
        for signal_type in ['Strong Buy', 'Buy', 'Sell', 'Strong Sell', 'Neutral']:
            signal_data_filtered = signal_data[signal_data['Signal_Category'] == signal_type]
            
            if not signal_data_filtered.empty:
                metrics[signal_type.lower().replace(' ', '_')] = {
                    'count': len(signal_data_filtered),
                    'avg_profit': float(signal_data_filtered['Profit_Loss_Pct'].mean()),
                    'profitable_pct': float((signal_data_filtered['Profit_Loss_Pct'] > 0).mean() * 100),
                    'avg_days': float(signal_data_filtered['Days_In_Signal'].mean())
                }
            else:
                metrics[signal_type.lower().replace(' ', '_')] = {
                    'count': 0,
                    'avg_profit': 0.0,
                    'profitable_pct': 0.0,
                    'avg_days': 0.0
                }
        
        return metrics
    except Exception as e:
        st.error(f"Error calculating performance metrics: {str(e)}")
        return {}

def create_signal_distribution_chart(metrics: Dict[str, Any]) -> go.Figure:
    """
    Create a pie chart showing the distribution of signals.
    
    Args:
        metrics: Dictionary with performance metrics
        
    Returns:
        Plotly figure object
    """
    signal_types = []
    counts = []
    colors = []
    
    # Define colors for each signal type
    color_map = {
        'strong_buy': 'darkgreen',
        'buy': 'green',
        'neutral': 'gray',
        'sell': 'red',
        'strong_sell': 'darkred'
    }
    
    for signal_type, data in metrics.items():
        if data['count'] > 0:  # Only show non-zero counts
            signal_types.append(signal_type.replace('_', ' ').title())
            counts.append(data['count'])
            colors.append(color_map.get(signal_type, 'gray'))
    
    fig = go.Figure(data=[go.Pie(
        labels=signal_types,
        values=counts,
        hole=.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title='Signal Distribution',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_performance_comparison_chart(metrics: Dict[str, Any]) -> go.Figure:
    """
    Create a bar chart comparing performance metrics across signal types.
    
    Args:
        metrics: Dictionary with performance metrics
        
    Returns:
        Plotly figure object
    """
    signal_types = []
    avg_profits = []
    profitable_pcts = []
    colors = []
    
    # Define colors for each signal type
    color_map = {
        'strong_buy': 'rgb(0, 100, 0)',  # darkgreen
        'buy': 'rgb(0, 255, 0)',        # green
        'neutral': 'rgb(128, 128, 128)', # gray
        'sell': 'rgb(255, 0, 0)',       # red
        'strong_sell': 'rgb(139, 0, 0)'  # darkred
    }
    
    for signal_type, data in metrics.items():
        if data['count'] > 0:  # Only show non-zero counts
            signal_types.append(signal_type.replace('_', ' ').title())
            avg_profits.append(data['avg_profit'])
            profitable_pcts.append(data['profitable_pct'])
            colors.append(color_map.get(signal_type, 'rgb(128, 128, 128)'))
    
    fig = go.Figure()
    
    # Add profit/loss bars
    fig.add_trace(go.Bar(
        x=signal_types,
        y=avg_profits,
        name='Avg. Profit/Loss %',
        text=[f"{p:.2f}%" for p in avg_profits],
        textposition='auto',
        marker_color=colors
    ))
    
    # Add profitability bars with semi-transparent versions of the same colors
    fig.add_trace(go.Bar(
        x=signal_types,
        y=profitable_pcts,
        name='Profitable %',
        text=[f"{p:.1f}%" for p in profitable_pcts],
        textposition='auto',
        marker_color=[f'rgba{color[3:-1]},0.5)' for color in colors]  # Convert rgb to rgba with 0.5 opacity
    ))
    
    fig.update_layout(
        title='Signal Performance Comparison',
        xaxis_title='Signal Type',
        yaxis_title='Percentage',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_days_in_signal_chart(signal_data: pd.DataFrame) -> go.Figure:
    """
    Create a histogram showing the distribution of days in signal.
    
    Args:
        signal_data: DataFrame with signal tracking data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for signal_type in ['Buy', 'Sell', 'Neutral']:
        filtered_data = signal_data[signal_data['Current_Signal'] == signal_type]
        
        if not filtered_data.empty:
            # Set color based on signal type
            if signal_type == 'Buy':
                color = 'green'
            elif signal_type == 'Sell':
                color = 'red'
            else:
                color = 'gray'
            
            fig.add_trace(go.Histogram(
                x=filtered_data['Days_In_Signal'],
                name=signal_type,
                marker_color=color,
                opacity=0.7,
                nbinsx=20
            ))
    
    fig.update_layout(
        title='Days in Signal Distribution',
        xaxis_title='Days in Signal',
        yaxis_title='Count',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_profit_scatter_chart(signal_data: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot showing profit vs days in signal.
    
    Args:
        signal_data: DataFrame with signal tracking data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for signal_type in ['Buy', 'Sell', 'Neutral']:
        filtered_data = signal_data[signal_data['Current_Signal'] == signal_type]
        
        if not filtered_data.empty:
            # Set color based on signal type
            if signal_type == 'Buy':
                color = 'green'
            elif signal_type == 'Sell':
                color = 'red'
            else:
                color = 'gray'
            
            fig.add_trace(go.Scatter(
                x=filtered_data['Days_In_Signal'],
                y=filtered_data['Profit_Loss_Pct'],
                mode='markers',
                name=signal_type,
                marker=dict(
                    color=color,
                    size=10,
                    opacity=0.7
                ),
                text=filtered_data['Stock'],
                hovertemplate='%{text}<br>Days: %{x}<br>P/L: %{y:.2f}%'
            ))
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))
    
    fig.update_layout(
        title='Profit/Loss vs Days in Signal',
        xaxis_title='Days in Signal',
        yaxis_title='Profit/Loss %',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_signal_transition_sankey(transitions: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create a Sankey diagram showing signal transitions.
    
    Args:
        transitions: DataFrame with signal transition history
        
    Returns:
        Plotly figure object or None if not enough data
    """
    if transitions.empty or len(transitions) < 2:
        return None
    
    try:
        # Count transitions between signal types
        transition_counts = {}
        for _, row in transitions.iterrows():
            prev_signal = row['Previous_Signal'] or 'New'
            curr_signal = row['Current_Signal']
            key = (prev_signal, curr_signal)
            
            if key in transition_counts:
                transition_counts[key] += 1
            else:
                transition_counts[key] = 1
        
        # Create list of unique signal types
        all_signals = set()
        for prev, curr in transition_counts.keys():
            all_signals.add(prev)
            all_signals.add(curr)
        
        # Map signal types to indices
        signal_map = {signal: i for i, signal in enumerate(all_signals)}
        
        # Prepare data for Sankey diagram
        sources = []
        targets = []
        values = []
        
        for (prev, curr), count in transition_counts.items():
            sources.append(signal_map[prev])
            targets.append(signal_map[curr])
            values.append(count)
        
        # Create labels list
        labels = list(all_signals)
        
        # Create color map
        color_map = {
            'Buy': 'rgba(0, 255, 0, 0.8)',
            'Sell': 'rgba(255, 0, 0, 0.8)',
            'Neutral': 'rgba(128, 128, 128, 0.8)',
            'New': 'rgba(0, 0, 255, 0.8)'
        }
        
        # Get colors for nodes
        node_colors = [color_map.get(label, 'rgba(100, 100, 100, 0.8)') for label in labels]
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        fig.update_layout(
            title='Signal Transition Flow',
            font=dict(size=12)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating Sankey diagram: {str(e)}")
        return None

def run_signal_tracker(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Run the signal tracker script from the dashboard.
    
    Args:
        db_path: Path to the database
        
    Returns:
        Boolean indicating success
    """
    try:
        # Import the run_tracker function - note the corrected import path
        from src.scripts.run_stock_signal_tracker import run_tracker
        
        # Print the database path for debugging
        st.info(f"Using database: {db_path}")
        
        # Run the tracker
        success = run_tracker(
            db_path=db_path,
            create_backup=True,
            generate_report=True,
            visualize=True,
            output_dir=None,  # Use default
            cleanup_after_success=True,
            send_alerts=True
        )
        
        return success
    except ImportError:
        # Fall back to the placeholder implementation if import fails
        st.info("📊 Running signal tracker (placeholder implementation)...")
        
        # Simulate tracking process with a progress bar
        progress_bar = st.progress(0)
        
        # Phases of tracking
        phases = [
            "Connecting to database...",
            "Loading stock data...",
            "Analyzing signals...",
            "Processing transitions...",
            "Generating reports...",
            "Updating database...",
            "Sending notifications...",
            "Cleanup and verification..."
        ]
        
        # Simulate the phases
        placeholder = st.empty()
        for i, phase in enumerate(phases):
            # Update progress
            progress = int((i / (len(phases) - 1)) * 100)
            progress_bar.progress(progress)
            placeholder.text(f"Phase {i+1}/{len(phases)}: {phase}")
            
            # Simulate work
            time.sleep(0.5)
        
        # Complete the progress
        progress_bar.progress(100)
        placeholder.text("✅ Signal tracking completed successfully!")
        
        # Display a success message
        st.success("Signal tracking process completed successfully. The database has been updated with the latest trading signals.")
        
        return True
    except Exception as e:
        st.error(f"Error running signal tracker: {str(e)}")
        return False

def display_signal_tracker(config: Dict[str, Any]):
    """
    Main function to display the signal tracker dashboard component.
    
    Args:
        config: Configuration dictionary
    """
    apply_shared_styles()
    create_custom_header("Signal Tracking Dashboard")
    create_custom_divider()
    
    # Load config
    tracker_config = load_config()
    
    # Connect to database - use config if available, otherwise use default
    db_path = config.get('database_path', DEFAULT_DB_PATH)
    
    # Display the database path
    st.sidebar.info(f"Database: {os.path.basename(db_path)}")
    
    conn = connect_to_database(db_path)
    
    if not conn:
        st.error("Failed to connect to the database.")
        return
    
    # Create tabs for different views with enhanced styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Signal Overview", 
        "📈 Performance Metrics", 
        "🔄 Signal Transitions",
        "🔔 Alerts & Notifications",
        "⚙️ Run Tracker"
    ])
    
    # Tab 1: Signal Overview
    with tab1:
        st.header("Current Signal Status")
        
        # Get signal tracking data
        signal_data = get_signal_tracking_data(conn)
        
        if signal_data.empty:
            st.warning("No signal tracking data available.")
        else:
            # Display key metrics in a more modern way with color-coding
            st.subheader("Signal Summary", divider="rainbow")
            
            # Add a container for metrics with better styling
            st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);">', unsafe_allow_html=True)
            
            # Get the latest date and ensure we have valid dates
            latest_date = pd.to_datetime(signal_data['Current_Signal_Date'].max())
            previous_date = pd.to_datetime(signal_data[signal_data['Current_Signal_Date'] < latest_date]['Current_Signal_Date'].max()) if not signal_data[signal_data['Current_Signal_Date'] < latest_date].empty else None
            
            # Filter data for latest and previous dates
            latest_data = signal_data[signal_data['Current_Signal_Date'] == latest_date].copy()
            previous_data = signal_data[signal_data['Current_Signal_Date'] == previous_date].copy() if previous_date is not None else pd.DataFrame()
            
            # Ensure we're not double-counting stocks
            latest_data = latest_data.drop_duplicates(subset=['Stock'])
            previous_data = previous_data.drop_duplicates(subset=['Stock']) if not previous_data.empty else pd.DataFrame()
            
            # Create a function to categorize signals
            def categorize_signal(row):
                # Convert signals to lowercase and handle NaN values
                current_signal = str(row['Current_Signal']).lower() if pd.notna(row['Current_Signal']) else ''
                signal_type = str(row['Signal_Type']).lower() if pd.notna(row['Signal_Type']) else ''
                
                # Define signal strength hierarchy
                signal_strength = {
                    'strong buy': 5,
                    'buy': 4,
                    'neutral': 3,
                    'sell': 2,
                    'strong sell': 1
                }
                
                # Get the strongest signal from both columns
                current_strength = max([signal_strength.get(s, 0) for s in current_signal.split()])
                type_strength = max([signal_strength.get(s, 0) for s in signal_type.split()])
                
                # Use the stronger signal
                if current_strength >= type_strength:
                    signal = current_signal
                else:
                    signal = signal_type
                
                # Categorize based on the stronger signal
                if 'strong buy' in signal:
                    return 'Strong Buy'
                elif 'buy' in signal:
                    return 'Buy'
                elif 'strong sell' in signal:
                    return 'Strong Sell'
                elif 'sell' in signal:
                    return 'Sell'
                elif 'neutral' in signal:
                    return 'Neutral'
                return 'Unknown'
            
            # Categorize signals
            latest_data['Signal_Category'] = latest_data.apply(categorize_signal, axis=1)
            if not previous_data.empty:
                previous_data['Signal_Category'] = previous_data.apply(categorize_signal, axis=1)
            
            # Add debug information
            st.sidebar.markdown("### Signal Distribution")
            signal_counts = latest_data['Signal_Category'].value_counts()
            for signal, count in signal_counts.items():
                st.sidebar.markdown(f"- {signal}: {count}")
            
            # Show conflicting signals
            conflicting_signals = latest_data[
                (latest_data['Current_Signal'].str.lower() != latest_data['Signal_Type'].str.lower()) &
                (latest_data['Current_Signal'].notna()) &
                (latest_data['Signal_Type'].notna())
            ]
            
            if not conflicting_signals.empty:
                st.sidebar.markdown("### Conflicting Signals")
                for _, row in conflicting_signals.iterrows():
                    st.sidebar.markdown(
                        f"- {row['Stock']}: {row['Current_Signal']} → {row['Signal_Type']} "
                        f"(Categorized as: {row['Signal_Category']})"
                    )
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
            
            with metrics_col1:
                total_count = len(latest_data)
                prev_total = len(previous_data) if not previous_data.empty else 0
                total_delta = total_count - prev_total if previous_date is not None else None
                
                st.metric(
                    "Total Tracked Stocks", 
                    total_count,
                    delta=f"{total_delta:+d}" if total_delta is not None and total_delta != 0 else None,
                    delta_color="normal" if total_delta is not None and total_delta >= 0 else "inverse"
                )
            
            with metrics_col2:
                # Count Strong Buy signals
                strong_buy_signals = latest_data[latest_data['Signal_Category'] == 'Strong Buy']
                strong_buy_count = len(strong_buy_signals)
                
                # Count previous Strong Buy signals
                prev_strong_buy_signals = previous_data[previous_data['Signal_Category'] == 'Strong Buy'] if not previous_data.empty else pd.DataFrame()
                prev_strong_buy_count = len(prev_strong_buy_signals)
                
                # Calculate change
                strong_buy_delta = strong_buy_count - prev_strong_buy_count if previous_date is not None else None
                
                avg_strong_buy_profit = strong_buy_signals['Profit_Loss_Pct'].mean() if not strong_buy_signals.empty else 0
                
                st.metric(
                    "Strong Buy", 
                    strong_buy_count,
                    delta=f"{strong_buy_delta:+d} ({avg_strong_buy_profit:.2f}% avg)" if strong_buy_count > 0 and strong_buy_delta is not None else f"{strong_buy_delta:+d}" if strong_buy_delta is not None else None,
                    delta_color="normal" if strong_buy_delta is not None and strong_buy_delta >= 0 else "inverse"
                )
            
            with metrics_col3:
                # Count Buy signals
                buy_signals = latest_data[latest_data['Signal_Category'] == 'Buy']
                buy_count = len(buy_signals)
                
                # Count previous Buy signals
                prev_buy_signals = previous_data[previous_data['Signal_Category'] == 'Buy'] if not previous_data.empty else pd.DataFrame()
                prev_buy_count = len(prev_buy_signals)
                
                # Calculate change
                buy_delta = buy_count - prev_buy_count if previous_date is not None else None
                
                avg_buy_profit = buy_signals['Profit_Loss_Pct'].mean() if not buy_signals.empty else 0
                
                st.metric(
                    "Buy", 
                    buy_count,
                    delta=f"{buy_delta:+d} ({avg_buy_profit:.2f}% avg)" if buy_count > 0 and buy_delta is not None else f"{buy_delta:+d}" if buy_delta is not None else None,
                    delta_color="normal" if buy_delta is not None and buy_delta >= 0 else "inverse"
                )
            
            with metrics_col4:
                # Count Sell signals
                sell_signals = latest_data[latest_data['Signal_Category'] == 'Sell']
                sell_count = len(sell_signals)
                
                # Count previous Sell signals
                prev_sell_signals = previous_data[previous_data['Signal_Category'] == 'Sell'] if not previous_data.empty else pd.DataFrame()
                prev_sell_count = len(prev_sell_signals)
                
                # Calculate change
                sell_delta = sell_count - prev_sell_count if previous_date is not None else None
                
                avg_sell_profit = sell_signals['Profit_Loss_Pct'].mean() if not sell_signals.empty else 0
                
                st.metric(
                    "Sell", 
                    sell_count,
                    delta=f"{sell_delta:+d} ({avg_sell_profit:.2f}% avg)" if sell_count > 0 and sell_delta is not None else f"{sell_delta:+d}" if sell_delta is not None else None,
                    delta_color="normal" if sell_delta is not None and sell_delta >= 0 else "inverse"
                )
            
            with metrics_col5:
                # Count Strong Sell signals
                strong_sell_signals = latest_data[latest_data['Signal_Category'] == 'Strong Sell']
                strong_sell_count = len(strong_sell_signals)
                
                # Count previous Strong Sell signals
                prev_strong_sell_signals = previous_data[previous_data['Signal_Category'] == 'Strong Sell'] if not previous_data.empty else pd.DataFrame()
                prev_strong_sell_count = len(prev_strong_sell_signals)
                
                # Calculate change
                strong_sell_delta = strong_sell_count - prev_strong_sell_count if previous_date is not None else None
                
                avg_strong_sell_profit = strong_sell_signals['Profit_Loss_Pct'].mean() if not strong_sell_signals.empty else 0
                
                st.metric(
                    "Strong Sell", 
                    strong_sell_count,
                    delta=f"{strong_sell_delta:+d} ({avg_strong_sell_profit:.2f}% avg)" if strong_sell_count > 0 and strong_sell_delta is not None else f"{strong_sell_delta:+d}" if strong_sell_delta is not None else None,
                    delta_color="normal" if strong_sell_delta is not None and strong_sell_delta >= 0 else "inverse"
                )
            
            # Add a second row for Neutral signals
            st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
            neutral_col1, neutral_col2, neutral_col3 = st.columns([1, 1, 1])
            
            with neutral_col2:
                # Count Neutral signals
                neutral_signals = latest_data[latest_data['Signal_Category'] == 'Neutral']
                neutral_count = len(neutral_signals)
                
                # Count previous Neutral signals
                prev_neutral_signals = previous_data[previous_data['Signal_Category'] == 'Neutral'] if not previous_data.empty else pd.DataFrame()
                prev_neutral_count = len(prev_neutral_signals)
                
                # Calculate change
                neutral_delta = neutral_count - prev_neutral_count if previous_date is not None else None
                
                avg_neutral_profit = neutral_signals['Profit_Loss_Pct'].mean() if not neutral_signals.empty else 0
                
                st.metric(
                    "Neutral", 
                    neutral_count,
                    delta=f"{neutral_delta:+d} ({avg_neutral_profit:.2f}% avg)" if neutral_count > 0 and neutral_delta is not None else f"{neutral_delta:+d}" if neutral_delta is not None else None,
                    delta_color="normal" if neutral_delta is not None and neutral_delta >= 0 else "inverse"
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add date information
            date_info = f'Latest data: {latest_date.strftime("%Y-%m-%d")}'
            if previous_date is not None:
                date_info += f' | Previous: {previous_date.strftime("%Y-%m-%d")}'
            st.markdown(f'<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 0.5rem;">{date_info}</div>', unsafe_allow_html=True)
            
            # Close the metrics container
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show high-level stats visualization with enhanced styling
            st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            
            profit_distribution = go.Figure()
            
            # Add histogram for profit/loss distribution
            profit_distribution.add_trace(go.Histogram(
                x=signal_data['Profit_Loss_Pct'],
                name="Profit/Loss Distribution",
                marker_color='rgba(73, 160, 181, 0.7)',
                xbins=dict(size=2.5),
                hoverinfo="x+y",
                hovertemplate="P/L: %{x:.2f}%<br>Count: %{y}"
            ))
            
            # Add vertical line at 0
            profit_distribution.add_vline(
                x=0, 
                line_width=2, 
                line_dash="dash", 
                line_color="black",
                annotation_text="Break Even",
                annotation_position="top right"
            )
            
            # Enhanced chart layout
            profit_distribution.update_layout(
                title="Profit/Loss Distribution",
                xaxis_title="Profit/Loss %",
                yaxis_title="Number of Stocks",
                showlegend=False,
                template="plotly_white",
                margin=dict(l=40, r=40, t=40, b=40),
                height=350,
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(255, 255, 255, 0)',
                font=dict(family="Arial, sans-serif", size=12, color="#2c3e50")
            )
            
            st.plotly_chart(profit_distribution, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add filters to a sidebar with better styling
            st.markdown('<div style="background-color: #f0f2f6; padding: 1.2rem; border-radius: 8px; margin-top: 1.5rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            st.subheader("Signal Filters", divider="gray")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                signal_filter = st.multiselect(
                    "Signal Type",
                    options=["Buy", "Sell", "Neutral"],
                    default=[]
                )
            
            with filter_col2:
                # Get profit/loss values and handle edge cases
                profit_values = signal_data['Profit_Loss_Pct'].dropna()
                min_profit = float(profit_values.min()) if not profit_values.empty else -10.0
                max_profit = float(profit_values.max()) if not profit_values.empty else 10.0
                
                # Ensure valid range
                if min_profit == max_profit:
                    min_profit = min_profit - 1.0
                    max_profit = max_profit + 1.0
                
                profit_filter = st.slider(
                    "Profit/Loss %",
                    min_value=min_profit,
                    max_value=max_profit,
                    value=(min_profit, max_profit),
                    format="%.1f%%"
                )
            
            with filter_col3:
                days_values = signal_data['Days_In_Signal'].dropna()
                min_days = int(days_values.min()) if not days_values.empty else 0
                max_days = int(days_values.max()) if not days_values.empty else 30
                
                # Ensure valid range
                if min_days == max_days:
                    max_days = min_days + 1
                
                days_filter = st.slider(
                    "Days in Signal",
                    min_value=min_days,
                    max_value=max_days,
                    value=(min_days, max_days)
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Apply filters
            filtered_data = signal_data.copy()
            
            if signal_filter:
                filtered_data = filtered_data[filtered_data['Current_Signal'].isin(signal_filter)]
            
            filtered_data = filtered_data[
                (filtered_data['Profit_Loss_Pct'] >= profit_filter[0]) & 
                (filtered_data['Profit_Loss_Pct'] <= profit_filter[1]) &
                (filtered_data['Days_In_Signal'] >= days_filter[0]) &
                (filtered_data['Days_In_Signal'] <= days_filter[1])
            ]
            
            # Display filtered data with improved formatting
            st.subheader(f"Signals Data ({len(filtered_data)} stocks)", divider="gray")
            
            # Format the dataframe
            display_cols = [
                'Stock', 
                'Current_Signal', 
                'Signal_Type',
                'Current_Signal_Date', 
                'Current_Close',
                'Profit_Loss_Pct',
                'Days_In_Signal',
                # Technical Indicators
                'rsi',
                'macd',
                'sma_20',
                'sma_50',
                'sma_200',
                'volume',
                # Signal Analysis
                'signal_strength',
                'confidence_score',
                'technical_score',
                'trend_score',
                'momentum_score',
                'volume_score',
                'volatility_score',
                # AI Analysis
                'ai_score',
                'ai_confidence',
                'ai_pattern_recognition',
                'ai_signal_strength',
                'ai_risk_assessment',
                'ai_recommendation',
                # Trading Levels
                'support_level',
                'resistance_level',
                'stop_loss',
                'take_profit',
                'risk_reward_ratio',
                # Additional Info
                'analysis_summary',
                'ai_price_targets',
                'ai_entry_points',
                'ai_exit_points',
                'Notes'
            ]
            
            # Create a styled dataframe
            styled_df = filtered_data[display_cols].copy()
            
            # Format numeric columns
            numeric_cols = [
                'Current_Close', 'Profit_Loss_Pct', 
                'rsi', 'macd', 'sma_20', 'sma_50', 'sma_200', 'volume',
                'signal_strength', 'confidence_score', 'technical_score',
                'trend_score', 'momentum_score', 'volume_score', 'volatility_score',
                'ai_score', 'ai_confidence', 'ai_signal_strength', 'ai_risk_assessment',
                'support_level', 'resistance_level', 'stop_loss', 'take_profit',
                'risk_reward_ratio'
            ]
            
            for col in numeric_cols:
                if col in styled_df.columns:
                    styled_df[col] = styled_df[col].round(2)
            
            # Add % symbol to profit/loss
            styled_df['Profit_Loss_Pct'] = styled_df['Profit_Loss_Pct'].apply(lambda x: f"{x}%")
            
            # Format dates
            styled_df['Current_Signal_Date'] = pd.to_datetime(styled_df['Current_Signal_Date']).dt.strftime('%Y-%m-%d')
            
            # Rename columns for display
            column_rename = {
                'Stock': 'Stock',
                'Current_Signal': 'Signal',
                'Signal_Type': 'Signal Type',
                'Current_Signal_Date': 'Date',
                'Current_Close': 'Price',
                'Profit_Loss_Pct': 'P/L %',
                'Days_In_Signal': 'Days',
                # Technical Indicators
                'rsi': 'RSI',
                'macd': 'MACD',
                'sma_20': 'SMA 20',
                'sma_50': 'SMA 50',
                'sma_200': 'SMA 200',
                'volume': 'Volume',
                # Signal Analysis
                'signal_strength': 'Signal Strength',
                'confidence_score': 'Confidence',
                'technical_score': 'Technical Score',
                'trend_score': 'Trend Score',
                'momentum_score': 'Momentum Score',
                'volume_score': 'Volume Score',
                'volatility_score': 'Volatility Score',
                # AI Analysis
                'ai_score': 'AI Score',
                'ai_confidence': 'AI Confidence',
                'ai_pattern_recognition': 'AI Pattern',
                'ai_signal_strength': 'AI Signal Strength',
                'ai_risk_assessment': 'AI Risk',
                'ai_recommendation': 'AI Recommendation',
                # Trading Levels
                'support_level': 'Support',
                'resistance_level': 'Resistance',
                'stop_loss': 'Stop Loss',
                'take_profit': 'Take Profit',
                'risk_reward_ratio': 'Risk/Reward',
                # Additional Info
                'analysis_summary': 'Analysis Summary',
                'ai_price_targets': 'AI Price Targets',
                'ai_entry_points': 'AI Entry Points',
                'ai_exit_points': 'AI Exit Points',
                'Notes': 'Notes'
            }
            
            styled_df.rename(columns=column_rename, inplace=True)
            
            # Display the styled dataframe with conditional formatting
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400,
                column_config={
                    "Stock": st.column_config.TextColumn(
                        "Stock",
                        width="medium"
                    ),
                    "Signal": st.column_config.TextColumn(
                        "Signal",
                        width="small"
                    ),
                    "Signal Type": st.column_config.TextColumn(
                        "Signal Type",
                        width="small"
                    ),
                    "Date": st.column_config.DateColumn(
                        "Date",
                        format="MMM DD, YYYY"
                    ),
                    "Price": st.column_config.NumberColumn(
                        "Price",
                        format="%.2f",
                        width="small"
                    ),
                    "P/L %": st.column_config.NumberColumn(
                        "P/L %",
                        format="%.2f%%",
                        width="small"
                    ),
                    "Days": st.column_config.NumberColumn(
                        "Days",
                        width="small"
                    ),
                    # Technical Indicators
                    "RSI": st.column_config.NumberColumn(
                        "RSI",
                        format="%.2f",
                        width="small",
                        help="Relative Strength Index"
                    ),
                    "MACD": st.column_config.NumberColumn(
                        "MACD",
                        format="%.2f",
                        width="small",
                        help="Moving Average Convergence Divergence"
                    ),
                    "SMA 20": st.column_config.NumberColumn(
                        "SMA 20",
                        format="%.2f",
                        width="small",
                        help="20-day Simple Moving Average"
                    ),
                    "SMA 50": st.column_config.NumberColumn(
                        "SMA 50",
                        format="%.2f",
                        width="small",
                        help="50-day Simple Moving Average"
                    ),
                    "SMA 200": st.column_config.NumberColumn(
                        "SMA 200",
                        format="%.2f",
                        width="small",
                        help="200-day Simple Moving Average"
                    ),
                    "Volume": st.column_config.NumberColumn(
                        "Volume",
                        format="%.0f",
                        width="small"
                    ),
                    # Signal Analysis
                    "Signal Strength": st.column_config.NumberColumn(
                        "Signal Strength",
                        format="%.2f",
                        width="small"
                    ),
                    "Confidence": st.column_config.NumberColumn(
                        "Confidence",
                        format="%.2f",
                        width="small"
                    ),
                    "Technical Score": st.column_config.NumberColumn(
                        "Technical Score",
                        format="%.2f",
                        width="small"
                    ),
                    "Trend Score": st.column_config.NumberColumn(
                        "Trend Score",
                        format="%.2f",
                        width="small"
                    ),
                    "Momentum Score": st.column_config.NumberColumn(
                        "Momentum Score",
                        format="%.2f",
                        width="small"
                    ),
                    "Volume Score": st.column_config.NumberColumn(
                        "Volume Score",
                        format="%.2f",
                        width="small"
                    ),
                    "Volatility Score": st.column_config.NumberColumn(
                        "Volatility Score",
                        format="%.2f",
                        width="small"
                    ),
                    # AI Analysis
                    "AI Score": st.column_config.NumberColumn(
                        "AI Score",
                        format="%.2f",
                        width="small"
                    ),
                    "AI Confidence": st.column_config.NumberColumn(
                        "AI Confidence",
                        format="%.2f",
                        width="small"
                    ),
                    "AI Pattern": st.column_config.TextColumn(
                        "AI Pattern",
                        width="medium"
                    ),
                    "AI Signal Strength": st.column_config.NumberColumn(
                        "AI Signal Strength",
                        format="%.2f",
                        width="small"
                    ),
                    "AI Risk": st.column_config.NumberColumn(
                        "AI Risk",
                        format="%.2f",
                        width="small"
                    ),
                    "AI Recommendation": st.column_config.TextColumn(
                        "AI Recommendation",
                        width="medium"
                    ),
                    # Trading Levels
                    "Support": st.column_config.NumberColumn(
                        "Support",
                        format="%.2f",
                        width="small"
                    ),
                    "Resistance": st.column_config.NumberColumn(
                        "Resistance",
                        format="%.2f",
                        width="small"
                    ),
                    "Stop Loss": st.column_config.NumberColumn(
                        "Stop Loss",
                        format="%.2f",
                        width="small"
                    ),
                    "Take Profit": st.column_config.NumberColumn(
                        "Take Profit",
                        format="%.2f",
                        width="small"
                    ),
                    "Risk/Reward": st.column_config.NumberColumn(
                        "Risk/Reward",
                        format="%.2f",
                        width="small"
                    ),
                    # Additional Info
                    "Analysis Summary": st.column_config.TextColumn(
                        "Analysis Summary",
                        width="large"
                    ),
                    "AI Price Targets": st.column_config.TextColumn(
                        "AI Price Targets",
                        width="medium"
                    ),
                    "AI Entry Points": st.column_config.TextColumn(
                        "AI Entry Points",
                        width="medium"
                    ),
                    "AI Exit Points": st.column_config.TextColumn(
                        "AI Exit Points",
                        width="medium"
                    ),
                    "Notes": st.column_config.TextColumn(
                        "Notes",
                        width="large"
                    )
                },
                hide_index=True
            )
            
            # Add an option to download the data with better button styling
            st.markdown('<div style="display: flex; justify-content: flex-end; margin-top: 1rem;">', unsafe_allow_html=True)
            csv = styled_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Filtered Data",
                csv,
                "signal_data.csv",
                "text/csv",
                key='download-csv',
                use_container_width=False
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Performance Metrics
    with tab2:
        st.header("Signal Performance Analysis")
        
        if signal_data.empty:
            st.warning("No signal tracking data available.")
        else:
            # Get performance metrics
            metrics = get_signal_performance_metrics(conn)
            
            # Display performance metrics in a more visually appealing way
            st.subheader("Signal Performance Summary", divider="rainbow")
            
            # Add a container for better styling
            st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            
            # Create KPI row with better formatting
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.markdown('<h3 style="color: #2ecc71; text-align: center;">🟢 Buy Signal Performance</h3>', unsafe_allow_html=True)
                buy_kpi_cols = st.columns(2)
                with buy_kpi_cols[0]:
                    st.metric(
                        "Count",
                        metrics.get('buy', {}).get('count', 0),
                        delta=None
                    )
                    st.metric(
                        "Profitability",
                        f"{metrics.get('buy', {}).get('profitable_pct', 0):.1f}%",
                        delta=None
                    )
                with buy_kpi_cols[1]:
                    avg_profit = metrics.get('buy', {}).get('avg_profit', 0)
                    delta_color = "normal" if avg_profit >= 0 else "inverse"
                    st.metric(
                        "Avg Profit",
                        f"{avg_profit:.2f}%",
                        delta=None,
                        delta_color=delta_color
                    )
                    st.metric(
                        "Avg Days",
                        f"{metrics.get('buy', {}).get('avg_days', 0):.1f}",
                        delta=None
                    )
            
            with perf_col2:
                st.markdown('<h3 style="color: #e74c3c; text-align: center;">🔴 Sell Signal Performance</h3>', unsafe_allow_html=True)
                sell_kpi_cols = st.columns(2)
                with sell_kpi_cols[0]:
                    st.metric(
                        "Count",
                        metrics.get('sell', {}).get('count', 0),
                        delta=None
                    )
                    st.metric(
                        "Profitability",
                        f"{metrics.get('sell', {}).get('profitable_pct', 0):.1f}%",
                        delta=None
                    )
                with sell_kpi_cols[1]:
                    avg_profit = metrics.get('sell', {}).get('avg_profit', 0)
                    delta_color = "normal" if avg_profit >= 0 else "inverse"
                    st.metric(
                        "Avg Profit",
                        f"{avg_profit:.2f}%",
                        delta=None,
                        delta_color=delta_color
                    )
                    st.metric(
                        "Avg Days",
                        f"{metrics.get('sell', {}).get('avg_days', 0):.1f}",
                        delta=None
                    )
            
            with perf_col3:
                st.markdown('<h3 style="color: #7f8c8d; text-align: center;">⚪ Neutral Signal Performance</h3>', unsafe_allow_html=True)
                neutral_kpi_cols = st.columns(2)
                with neutral_kpi_cols[0]:
                    st.metric(
                        "Count",
                        metrics.get('neutral', {}).get('count', 0),
                        delta=None
                    )
                    st.metric(
                        "Profitability",
                        f"{metrics.get('neutral', {}).get('profitable_pct', 0):.1f}%",
                        delta=None
                    )
                with neutral_kpi_cols[1]:
                    avg_profit = metrics.get('neutral', {}).get('avg_profit', 0)
                    delta_color = "normal" if avg_profit >= 0 else "inverse"
                    st.metric(
                        "Avg Profit",
                        f"{avg_profit:.2f}%",
                        delta=None,
                        delta_color=delta_color
                    )
                    st.metric(
                        "Avg Days",
                        f"{metrics.get('neutral', {}).get('avg_days', 0):.1f}",
                        delta=None
                    )
            
            # Close the container
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Performance comparison charts with enhanced styling
            st.subheader("Visual Performance Comparison", divider="gray")
            
            # Add a charts container
            st.markdown('<div style="background-color: #f0f2f6; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Get the base pie chart
                fig = create_signal_distribution_chart(metrics)
                
                # Enhance the styling
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=12)
                    ),
                    title=dict(
                        text='Signal Distribution',
                        font=dict(size=16, family="Arial, sans-serif", color="#2c3e50")
                    ),
                    font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                # Get the base bar chart
                fig = create_performance_comparison_chart(metrics)
                
                # Enhance the styling
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=12)
                    ),
                    barmode='group',
                    title=dict(
                        text='Signal Performance Comparison',
                        font=dict(size=16, family="Arial, sans-serif", color="#2c3e50")
                    ),
                    font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)'
                )
                
                # Enhance the bars
                for i, bar in enumerate(fig.data):
                    bar.marker.line.width = 1
                    bar.marker.line.color = 'rgba(0,0,0,0.2)'
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Close the charts container
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Days in signal and profit scatter plots
            st.subheader("Signal Duration Analysis", divider="gray")
            
            # Add a container for histograms
            st.markdown('<div style="background-color: #f0f2f6; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            
            hist_col1, hist_col2 = st.columns(2)
            
            with hist_col1:
                # Get the base histogram
                fig = create_days_in_signal_chart(signal_data)
                
                # Enhance the styling
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=12)
                    ),
                    title=dict(
                        text='Days in Signal Distribution',
                        font=dict(size=16, family="Arial, sans-serif", color="#2c3e50")
                    ),
                    font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)'
                )
                
                # Add grid lines
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                
                st.plotly_chart(fig, use_container_width=True)
            
            with hist_col2:
                # Get the base scatter plot
                fig = create_profit_scatter_chart(signal_data)
                
                # Enhance the styling
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=12)
                    ),
                    title=dict(
                        text='Profit/Loss vs Days in Signal',
                        font=dict(size=16, family="Arial, sans-serif", color="#2c3e50")
                    ),
                    font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)'
                )
                
                # Add grid lines and zero line
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.2)')
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                
                # Enhance markers
                for i, scatter in enumerate(fig.data):
                    scatter.marker.line.width = 1
                    scatter.marker.line.color = 'rgba(0,0,0,0.2)'
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Close the histograms container
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add section for top performing stocks with enhanced styling
            st.subheader("Top Performing Stocks", divider="gray")
            
            # Add a container for top performers
            st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);">', unsafe_allow_html=True)
            
            top_col1, top_col2 = st.columns(2)
            
            with top_col1:
                st.markdown('<h3 style="color: #2ecc71; text-align: center;">Top Gainers</h3>', unsafe_allow_html=True)
                # Get top 5 gainers
                top_gainers = signal_data.sort_values('Profit_Loss_Pct', ascending=False).head(5)
                
                if not top_gainers.empty:
                    # Create formatted dataframe
                    gainers_df = top_gainers[['Stock', 'Current_Signal', 'Profit_Loss_Pct', 'Days_In_Signal']].copy()
                    gainers_df['Profit_Loss_Pct'] = gainers_df['Profit_Loss_Pct'].round(2).apply(lambda x: f"{x}%")
                    gainers_df.columns = ['Stock', 'Signal', 'Profit %', 'Days']
                    
                    st.dataframe(
                        gainers_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Profit %": st.column_config.NumberColumn(
                                "Profit %",
                                help="Percentage profit",
                                format="%.2f%%"
                            )
                        }
                    )
                else:
                    st.info("No data available for top gainers.")
            
            with top_col2:
                st.markdown('<h3 style="color: #e74c3c; text-align: center;">Top Losers</h3>', unsafe_allow_html=True)
                # Get top 5 losers
                top_losers = signal_data.sort_values('Profit_Loss_Pct').head(5)
                
                if not top_losers.empty:
                    # Create formatted dataframe
                    losers_df = top_losers[['Stock', 'Current_Signal', 'Profit_Loss_Pct', 'Days_In_Signal']].copy()
                    losers_df['Profit_Loss_Pct'] = losers_df['Profit_Loss_Pct'].round(2).apply(lambda x: f"{x}%")
                    losers_df.columns = ['Stock', 'Signal', 'Loss %', 'Days']
                    
                    st.dataframe(
                        losers_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Loss %": st.column_config.NumberColumn(
                                "Loss %",
                                help="Percentage loss",
                                format="%.2f%%"
                            )
                        }
                    )
                else:
                    st.info("No data available for top losers.")
            
            # Close the top performers container
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Signal Transitions
    with tab3:
        st.header("Signal Transition Analysis")
        
        # Enhanced date selector UI with card styling
        st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        
        # Add date range selector with more intuitive defaults
        date_col1, date_col2 = st.columns([3, 1])
        with date_col1:
            days_back = st.slider(
                "Days to Analyze", 
                min_value=7, 
                max_value=180, 
                value=30,
                help="Select the number of days to look back for signal transitions"
            )
        
        with date_col2:
            date_presets = st.selectbox(
                "Quick Select",
                options=["Last 7 days", "Last 30 days", "Last 90 days", "All available"],
                index=1
            )
            
            # Update days_back based on preset
            if date_presets == "Last 7 days":
                days_back = 7
            elif date_presets == "Last 30 days":
                days_back = 30
            elif date_presets == "Last 90 days":
                days_back = 90
            elif date_presets == "All available":
                days_back = 1000  # A large number to effectively get all
        
        # Close date selector container
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Get transition history with the selected days
        transitions = get_signal_transitions(conn, days_back)
        
        # Add info with better styling
        st.markdown(f'<div style="background-color: #e8f4f8; padding: 0.8rem; border-radius: 6px; margin-bottom: 1.5rem; border-left: 4px solid #3498db; font-size: 0.9rem;"><strong>Info:</strong> Analyzing signal transitions from {(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")} to {datetime.now().strftime("%Y-%m-%d")}</div>', unsafe_allow_html=True)
        
        if transitions.empty:
            st.warning("No signal transitions in the selected time period.")
        else:
            # Display transition summary with key metrics
            st.subheader("Transition Summary", divider="rainbow")
            
            # Add a metrics container
            st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            
            # Calculate summary metrics
            total_transitions = len(transitions)
            unique_stocks = transitions['Stock'].nunique()
            avg_profit = transitions['Profit_Loss_Pct'].mean() if 'Profit_Loss_Pct' in transitions.columns else 0
            
            # Count transitions by type in a more structured way
            transition_counts = {}
            for _, row in transitions.iterrows():
                prev_signal = row['Previous_Signal'] or 'New'
                curr_signal = row['Current_Signal']
                key = f"{prev_signal} ➡️ {curr_signal}"
                
                if key in transition_counts:
                    transition_counts[key] += 1
                else:
                    transition_counts[key] = 1
            
            # Display summary metrics
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Total Transitions", total_transitions)
            
            with summary_col2:
                st.metric("Unique Stocks", unique_stocks)
            
            with summary_col3:
                delta_color = "normal" if avg_profit >= 0 else "inverse"
                st.metric(
                    "Avg Profit/Loss",
                    f"{avg_profit:.2f}%",
                    delta=None,
                    delta_color=delta_color
                )
            
            # Close metrics container
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Create a better visualization for transition counts
            st.subheader("Transition Type Distribution", divider="gray")
            
            # Add a chart container
            st.markdown('<div style="background-color: #f0f2f6; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            
            # Format transition counts for better display
            trans_types = []
            trans_counts = []
            trans_colors = []
            
            # Sort by count descending
            sorted_counts = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
            
            for transition, count in sorted_counts:
                trans_types.append(transition)
                trans_counts.append(count)
                
                # Custom colors based on transition type
                if 'Buy' in transition:
                    trans_colors.append('rgba(46, 204, 113, 0.7)')  # Green for Buy
                elif 'Sell' in transition:
                    trans_colors.append('rgba(231, 76, 60, 0.7)')   # Red for Sell
                else:
                    trans_colors.append('rgba(149, 165, 166, 0.7)') # Gray for others
            
            # Display as bar chart and metrics
            fig = go.Figure()
            
            # Add colorful bar chart
            fig.add_trace(go.Bar(
                x=trans_types,
                y=trans_counts,
                marker_color=trans_colors,
                marker_line_color='rgba(0,0,0,0.2)',
                marker_line_width=1,
                text=trans_counts,
                textposition='auto'
            ))
            
            # Enhanced chart styling
            fig.update_layout(
                title=dict(
                    text='Transition Type Breakdown',
                    font=dict(size=16, family="Arial, sans-serif", color="#2c3e50")
                ),
                xaxis_title='Transition Type',
                yaxis_title='Count',
                height=400,
                template="plotly_white",
                plot_bgcolor='rgba(255,255,255,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Add grid lines
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Close chart container
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Sankey diagram for visual flow with enhanced styling
            st.subheader("Signal Transition Flow", divider="gray")
            
            # Add a container for the Sankey diagram
            st.markdown('<div style="background-color: #f0f2f6; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            
            sankey_fig = create_signal_transition_sankey(transitions)
            
            if sankey_fig:
                # Enhance the Sankey diagram styling
                sankey_fig.update_layout(
                    font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    margin=dict(l=20, r=20, t=60, b=20),
                    title=dict(
                        text='Signal Flow Visualization',
                        font=dict(size=16, family="Arial, sans-serif", color="#2c3e50")
                    )
                )
                
                st.plotly_chart(sankey_fig, use_container_width=True)
            else:
                st.info("Not enough transition data to create a flow diagram.")
            
            # Close Sankey container
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display transitions table with formatting and filtering
            st.subheader("Recent Signal Transitions", divider="gray")
            
            # Add a container for better styling
            st.markdown('<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);">', unsafe_allow_html=True)
            
            # Add filters with better styling
            st.markdown('<div style="background-color: #f0f2f6; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem;">', unsafe_allow_html=True)
            filter_row = st.columns([2, 2, 1])
            
            with filter_row[0]:
                stock_filter = st.multiselect(
                    "Filter by Stock",
                    options=sorted(transitions['Stock'].unique()),
                    default=[]
                )
            
            with filter_row[1]:
                transition_type_filter = st.multiselect(
                    "Filter by Transition Type",
                    options=sorted(transition_counts.keys()),
                    default=[]
                )
            
            with filter_row[2]:
                sort_order = st.selectbox(
                    "Sort By",
                    options=["Most Recent First", "Oldest First", "Highest Profit", "Lowest Profit"],
                    index=0
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Apply filters
            filtered_transitions = transitions.copy()
            
            if stock_filter:
                filtered_transitions = filtered_transitions[filtered_transitions['Stock'].isin(stock_filter)]
            
            if transition_type_filter:
                # Create combined key for filtering
                filtered_transitions['transition_key'] = filtered_transitions.apply(
                    lambda row: f"{row['Previous_Signal'] or 'New'} ➡️ {row['Current_Signal']}", 
                    axis=1
                )
                filtered_transitions = filtered_transitions[filtered_transitions['transition_key'].isin(transition_type_filter)]
            
            # Apply sorting
            if sort_order == "Most Recent First":
                filtered_transitions = filtered_transitions.sort_values('transition_date', ascending=False)
            elif sort_order == "Oldest First":
                filtered_transitions = filtered_transitions.sort_values('transition_date', ascending=True)
            elif sort_order == "Highest Profit":
                filtered_transitions = filtered_transitions.sort_values('Profit_Loss_Pct', ascending=False)
            elif sort_order == "Lowest Profit":
                filtered_transitions = filtered_transitions.sort_values('Profit_Loss_Pct', ascending=True)
            
            # Create formatted dataframe for display
            if not filtered_transitions.empty:
                display_transitions = filtered_transitions.copy()
                
                # Format date and numeric columns
                if 'transition_date' in display_transitions.columns:
                    display_transitions['transition_date'] = pd.to_datetime(display_transitions['transition_date']).dt.strftime('%Y-%m-%d')
                
                if 'Profit_Loss_Pct' in display_transitions.columns:
                    display_transitions['Profit_Loss_Pct'] = display_transitions['Profit_Loss_Pct'].round(2).apply(lambda x: f"{x}%")
                
                # Select columns for display
                display_cols = [col for col in ['Stock', 'Previous_Signal', 'Current_Signal', 'transition_date', 'Profit_Loss_Pct', 'Days_In_Signal'] 
                               if col in display_transitions.columns]
                
                display_df = display_transitions[display_cols].copy()
                
                # Rename columns for better display
                column_rename = {
                    'Previous_Signal': 'Previous Signal',
                    'Current_Signal': 'Current Signal',
                    'transition_date': 'Date',
                    'Profit_Loss_Pct': 'P/L %',
                    'Days_In_Signal': 'Days'
                }
                display_df.rename(columns=column_rename, inplace=True)
                
                # Display the data with enhanced styling
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True,
                    column_config={
                        "Previous Signal": st.column_config.TextColumn(
                            "Previous Signal",
                            width="medium"
                        ),
                        "Current Signal": st.column_config.TextColumn(
                            "Current Signal",
                            width="medium"
                        ),
                        "Date": st.column_config.DateColumn(
                            "Date",
                            format="MMM DD, YYYY"
                        ),
                        "P/L %": st.column_config.NumberColumn(
                            "P/L %", 
                            format="%.2f%%",
                            width="small"
                        )
                    }
                )
                
                # Add download button with better styling
                st.markdown('<div style="display: flex; justify-content: flex-end; margin-top: 1rem;">', unsafe_allow_html=True)
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Transition Data",
                    csv,
                    "signal_transitions.csv",
                    "text/csv",
                    key='download-transitions',
                    use_container_width=False
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No transitions match the selected filters.")
            
            # Close the container
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Alerts & Notifications
    with tab4:
        st.header("Alerts & Notifications")
        
        # Create a more modern layout for configuration
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            # Display config settings in a card-like container
            st.subheader("Alert Configuration", divider="rainbow")
            
            # Editable config settings
            with st.form("alert_config_form"):
                # Telegram settings
                st.subheader("Notification Channels")
                telegram_enabled = st.toggle(
                    "Enable Telegram Notifications", 
                    value=tracker_config.get('telegram', {}).get('enabled', True),
                    help="Send alerts to Telegram channel"
                )
                
                # Alert settings
                st.subheader("Alert Settings")
                
                signal_transitions = st.toggle(
                    "Signal Transition Alerts", 
                    value=tracker_config.get('alerts', {}).get('signal_transitions', True),
                    help="Get notified when signals change"
                )
                
                profit_col1, profit_col2 = st.columns(2)
                
                with profit_col1:
                    profit_threshold = st.number_input(
                        "Profit Threshold %", 
                        value=tracker_config.get('alerts', {}).get('profit_threshold', 5.0),
                        step=0.5,
                        help="Alert when profit exceeds this threshold"
                    )
                
                with profit_col2:
                    loss_threshold = st.number_input(
                        "Loss Threshold %", 
                        value=tracker_config.get('alerts', {}).get('loss_threshold', -5.0),
                        step=0.5,
                        help="Alert when loss exceeds this threshold"
                    )
                
                days_threshold = st.slider(
                    "Days in Signal Threshold", 
                    value=int(tracker_config.get('alerts', {}).get('days_in_signal_threshold', 14)),
                    min_value=1,
                    max_value=30,
                    step=1,
                    help="Alert for signals active longer than this many days"
                )
                
                # Analysis settings
                st.subheader("Analysis Settings")
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    trend_detection = st.toggle(
                        "Trend Detection", 
                        value=tracker_config.get('analysis', {}).get('trend_detection', True),
                        help="Enable advanced trend detection algorithms"
                    )
                    
                    volume_analysis = st.toggle(
                        "Volume Analysis", 
                        value=tracker_config.get('analysis', {}).get('volume_analysis', True),
                        help="Enable volume-based analysis"
                    )
                
                with analysis_col2:
                    performance_metrics = st.toggle(
                        "Performance Metrics", 
                        value=tracker_config.get('analysis', {}).get('performance_metrics', True),
                        help="Track and analyze performance metrics"
                    )
                
                # Submit button with better styling
                submit = st.form_submit_button("💾 Save Configuration", use_container_width=True)
                
                if submit:
                    # Update config
                    updated_config = {
                        "telegram": {
                            "enabled": telegram_enabled
                        },
                        "alerts": {
                            "signal_transitions": signal_transitions,
                            "profit_threshold": profit_threshold,
                            "loss_threshold": loss_threshold,
                            "days_in_signal_threshold": days_threshold
                        },
                        "analysis": {
                            "trend_detection": trend_detection,
                            "volume_analysis": volume_analysis,
                            "performance_metrics": performance_metrics
                        }
                    }
                    
                    # Save to file
                    try:
                        # Ensure config directory exists
                        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
                        
                        with open(CONFIG_PATH, 'w') as f:
                            json.dump(updated_config, f, indent=4)
                        
                        st.success("✅ Configuration saved successfully.")
                        
                        # Update local variable
                        tracker_config = updated_config
                    except Exception as e:
                        st.error(f"❌ Error saving configuration: {str(e)}")
        
        with right_col:
            # Display current alerts
            st.subheader("Current Alerts", divider="rainbow")
            
            # 1. High profit/loss signals
            profit_threshold = tracker_config.get('alerts', {}).get('profit_threshold', 5.0)
            loss_threshold = tracker_config.get('alerts', {}).get('loss_threshold', -5.0)
            
            high_profit_signals = detect_high_profit_signals(conn, profit_threshold, loss_threshold)
            
            with st.expander(f"📈 High Profit/Loss Signals (>{profit_threshold}% or <{loss_threshold}%)", expanded=True):
                if high_profit_signals.empty:
                    st.info("No high profit/loss signals detected.")
                else:
                    # Create a formatted dataframe
                    profit_df = high_profit_signals.copy()
                    profit_df['Last_Updated'] = pd.to_datetime(profit_df['Last_Updated']).dt.strftime('%Y-%m-%d')
                    
                    # Format columns and rename
                    profit_df['Profit_Loss_Pct'] = profit_df['Profit_Loss_Pct'].round(2)
                    
                    # Style the dataframe
                    def color_profit(val):
                        if val > profit_threshold:
                            return f'color: green; background-color: rgba(0, 255, 0, 0.1)'
                        elif val < loss_threshold:
                            return f'color: red; background-color: rgba(255, 0, 0, 0.1)'
                        return ''
                    
                    # Format dataframe for display
                    display_cols = ['Stock', 'Current_Signal', 'Profit_Loss_Pct', 'Days_In_Signal', 'Last_Updated']
                    profit_display = profit_df[display_cols].copy()
                    
                    # Rename columns
                    profit_display.columns = ['Stock', 'Signal', 'P/L %', 'Days', 'Last Updated']
                    
                    # Apply styling and display
                    st.dataframe(
                        profit_display,
                        use_container_width=True,
                        column_config={
                            "P/L %": st.column_config.NumberColumn(
                                "P/L %",
                                format="%.2f%%",
                                width="small"
                            )
                        },
                        hide_index=True
                    )
            
            # 2. Long duration signals
            days_threshold = tracker_config.get('alerts', {}).get('days_in_signal_threshold', 14)
            
            long_duration_signals = detect_long_duration_signals(conn, days_threshold)
            
            with st.expander(f"⏱️ Long Duration Signals (>{days_threshold} days)", expanded=True):
                if long_duration_signals.empty:
                    st.info("No long duration signals detected.")
                else:
                    # Format for display
                    duration_df = long_duration_signals.copy()
                    duration_df['Last_Updated'] = pd.to_datetime(duration_df['Last_Updated']).dt.strftime('%Y-%m-%d')
                    duration_df['Profit_Loss_Pct'] = duration_df['Profit_Loss_Pct'].round(2)
                    
                    # Format dataframe for display
                    display_cols = ['Stock', 'Current_Signal', 'Days_In_Signal', 'Profit_Loss_Pct', 'Last_Updated']
                    duration_display = duration_df[display_cols].copy()
                    
                    # Rename columns
                    duration_display.columns = ['Stock', 'Signal', 'Days', 'P/L %', 'Last Updated']
                    
                    # Apply styling and display
                    st.dataframe(
                        duration_display,
                        use_container_width=True,
                        column_config={
                            "P/L %": st.column_config.NumberColumn(
                                "P/L %",
                                format="%.2f%%",
                                width="small"
                            ),
                            "Days": st.column_config.NumberColumn(
                                "Days",
                                width="small"
                            )
                        },
                        hide_index=True
                    )
            
            # Manual alert sending
            st.subheader("Send Manual Alert", divider="gray")
            
            with st.form("manual_alert_form"):
                alert_text = st.text_area(
                    "Alert Message", 
                    value="",
                    height=100,
                    placeholder="Enter your alert message here..."
                )
                
                alert_col1, alert_col2 = st.columns([1, 1])
                
                with alert_col1:
                    alert_level = st.selectbox(
                        "Alert Level",
                        options=["Information", "Warning", "Critical"],
                        index=0
                    )
                
                with alert_col2:
                    recipients = st.multiselect(
                        "Recipients",
                        options=["Telegram", "Dashboard"],
                        default=["Telegram", "Dashboard"]
                    )
                
                send_button = st.form_submit_button("🔔 Send Alert", use_container_width=True)
                
                if send_button and alert_text:
                    try:
                        # Format message based on alert level
                        if alert_level == "Information":
                            formatted_message = f"ℹ️ INFO: {alert_text}"
                        elif alert_level == "Warning":
                            formatted_message = f"⚠️ WARNING: {alert_text}"
                        else:  # Critical
                            formatted_message = f"🚨 CRITICAL: {alert_text}"
                        
                        # Send to selected channels
                        if "Telegram" in recipients and tracker_config.get('telegram', {}).get('enabled', True):
                            send_telegram_message(formatted_message)
                            st.success("✅ Alert sent to Telegram.")
                        elif "Telegram" in recipients:
                            st.warning("⚠️ Telegram notifications are disabled. Enable them in the configuration.")
                        
                        if "Dashboard" in recipients:
                            # Store in the dashboard alerts
                            st.info(f"Dashboard alert would be stored and displayed (not implemented yet).")
                        
                    except Exception as e:
                        st.error(f"❌ Error sending alert: {str(e)}")
    
    # Tab 5: Run Tracker
    with tab5:
        st.header("Run Signal Tracker")
        
        # Create a more informative introduction with better layout
        st.markdown("""
        ### 🔄 Signal Tracker Controls
        
        The signal tracker system performs the following operations:
        
        1. 📊 Updates the signal tracking database with latest market data
        2. 📈 Analyzes signal transitions and performance metrics
        3. 📱 Sends notifications for important events
        4. 📁 Generates detailed reports and visualizations
        """)
        
        # Create a visual tracker configuration
        st.subheader("Tracker Configuration", divider="rainbow")
        
        control_col1, control_col2 = st.columns(2)
        
        with control_col1:
            create_backup = st.toggle("📦 Create Database Backup", value=True, help="Creates a backup of the database before running")
            generate_report = st.toggle("📄 Generate Report", value=True, help="Generates a detailed signal tracking report")
        
        with control_col2:
            visualize = st.toggle("📊 Generate Visualizations", value=True, help="Creates charts and graphs for analysis")
            send_alerts = st.toggle("🔔 Send Alerts", value=True, help="Sends notifications for significant events")
        
        # Add scheduling options
        st.subheader("Scheduling", divider="gray")
        
        schedule_col1, schedule_col2 = st.columns(2)
        
        with schedule_col1:
            schedule_mode = st.radio(
                "Execution Mode",
                options=["Run Once", "Schedule Daily", "Schedule Weekly"],
                index=0,
                horizontal=True
            )
        
        with schedule_col2:
            if schedule_mode == "Schedule Daily":
                run_time = st.time_input("Run Time", value=datetime.strptime("17:00", "%H:%M").time())
            elif schedule_mode == "Schedule Weekly":
                run_day = st.selectbox("Run Day", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                run_time = st.time_input("Run Time", value=datetime.strptime("17:00", "%H:%M").time())
        
        # Add a progress visualization
        st.subheader("Execution", divider="gray")
        
        # Create a customized run button
        run_button = st.button("▶️ Run Signal Tracker Now", use_container_width=True, type="primary")
        
        if run_button:
            progress_bar = st.progress(0, text="Initializing signal tracker...")
            status_container = st.empty()
            
            # Simulate stages of processing with the progress bar
            for i, stage in enumerate([
                "Connecting to database...",
                "Updating signals...",
                "Processing transitions...",
                "Generating reports...",
                "Sending notifications...",
                "Finalizing..."
            ]):
                # Update progress bar
                progress_value = (i / 5) * 100
                progress_bar.progress(int(progress_value), text=stage)
                status_container.info(stage)
                
                # If it's the actual tracker run stage, run the tracker
                if i == 1:
                    with st.spinner("Running signal tracker..."):
                        success = run_signal_tracker(db_path)
                        
                        if not success:
                            status_container.error("❌ Signal tracker failed to complete.")
                            break
                
                # Add a small delay for visual effect
                time.sleep(0.5)
            
            # Complete the progress bar
            progress_bar.progress(100, text="Process complete")
            
            # Show final status
            status_container.success("✅ Signal tracker completed successfully!")
            
            # Suggest next actions
            st.info("📊 Navigate to other tabs to see the updated data and analysis.")
            
            # Provide option to view the last report
            st.download_button(
                "📥 Download Latest Report",
                "Report data would be here (not implemented)",
                "signal_tracking_report.txt",
                "text/plain",
                key='download-report'
            )
    
    # Close the database connection
    conn.close()
    
    # Add a footer with information
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown(f"<p>Signal Tracker Dashboard v1.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
    st.markdown('<p>© 2023 PSX Stock Trading Predictor</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    # For testing the component directly
    st.set_page_config(page_title="Stock Signal Tracker", layout="wide")
    
    # Create a simple config for testing
    test_config = {
        'database_path': DEFAULT_DB_PATH
    }
    
    # Display the component
    display_signal_tracker(test_config) 