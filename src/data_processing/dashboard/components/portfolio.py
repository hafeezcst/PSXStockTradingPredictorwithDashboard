"""
Portfolio component for the PSX dashboard.
Combines portfolio data with technical indicators for comprehensive analysis
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import logging
import numpy as np
from scipy import stats
import requests
from bs4 import BeautifulSoup
import time
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Use relative imports for local modules
from .shared_styles import (
    apply_shared_styles,
    create_custom_header,
    create_custom_subheader,
    create_custom_divider,
    create_chart_container,
    create_metric_card,
    create_alert
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with comma as decimal point"""
    return f"{value:,.{decimals}f}".replace(".", ",")

def convert_to_float(value) -> float:
    """Convert value to float, handling both string and numeric inputs"""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

def save_portfolio_data(portfolio_data: Dict) -> None:
    """Save portfolio data to a JSON file."""
    save_path = "data/portfolio_data.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Add timestamp to the data
    portfolio_data['last_updated'] = datetime.now().isoformat()
    
    with open(save_path, 'w') as f:
        json.dump(portfolio_data, f, indent=4)

def load_portfolio_data() -> Dict:
    """Load portfolio data from JSON file."""
    save_path = "data/portfolio_data.json"
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            data = json.load(f)
        return data
    return None

def initialize_portfolio_data() -> Dict:
    """Initialize portfolio data with actual holdings."""
    default_portfolio = {
        'Name': [
            'Fast Cables', 'Engro Polymer & Chemicals', 'Meezan Bank', 'Meezan Bank',
            'Millat Tractors', 'Pak Datacom Ltd', 'Waves Singer', 'Unity Foods',
            'Treet Corporation', 'Telecard Ltd', 'Pakistan Refinery', 'Oilboy Energy',
            'National Refinery', 'Lotte Chemical Pakistan', 'ITTEFAQ Iron',
            'International Steels', 'International Industries', 'Ghandhara Tyre Rubber',
            'Ghani Glass Ltd', 'Cnergyico PK', 'Clover Pakistan',
            'Century Paper & Board Mills', 'Avanceon', 'Azgard Nine Ltd',
            'IBL HealthCare', 'Ghani Gases Ltd', 'Ghani Global Glass',
            'Bank Islami Pakistan', 'Siddiqsons Tin Plate', 'NetSol Technologies',
            'Ghani Value Glass', 'WorldCall Telecom', 'Aisha Steel Mills',
            'Archroma Pakistan', 'Pakistan Intl Bulk Terminal Private', 'Sitara Peroxide',
            'Nishat Mills', 'K-Electric', 'Karachi 100', 'Shabbir Tiles & Ceramics',
            'Dewan Cement Ltd', 'Balochistan Glass', 'Mughal Iron & Steel Industries'
        ],
        'Sector': [
            'Technology', 'Chemicals', 'Banking', 'Banking', 'Automotive', 'Technology',
            'Consumer Goods', 'Food', 'Consumer Goods', 'Technology', 'Energy', 'Energy',
            'Energy', 'Chemicals', 'Steel', 'Steel', 'Steel', 'Automotive', 'Glass',
            'Energy', 'Consumer Goods', 'Paper', 'Technology', 'Textile', 'Healthcare',
            'Chemicals', 'Glass', 'Banking', 'Steel', 'Technology', 'Glass', 'Telecom',
            'Steel', 'Chemicals', 'Transport', 'Chemicals', 'Textile', 'Energy', 'Index',
            'Construction', 'Cement', 'Glass', 'Steel'
        ],
        'Market Value': [
            776320, 1460430, 255650, 1014930, 724055, 189520, 2636400, 164160,
            282555, 324450, 342700, 436400, 614707, 356400, 380500, 337815,
            555415, 316160, 474740, 512400, 547375, 454730, 848300, 369500,
            18645, 201150, 223000, 839600, 448000, 271120, 44000, 3880800,
            2266000, 390852, 2493350, 378820, 2492280, 453000, 117316, 729210,
            92700, 146550, 992922
        ],
        'Net P/L': [
            0, -361540, 1840, 57990, -99990, -10480, -2060000, -33440, -34690,
            -91800, -88800, 40800, -160610, -95600, -47500, -100890, -46970,
            -93440, 45640, 93000, -132500, -163020, -225320, -43000, -3480,
            -29550, 5500, -126100, -83700, -57880, -6000, -3520000, -1510000,
            9252, 823470, -231720, 822200, -123000, 42380, -364590, -9100,
            -103800, -351900
        ],
        'Daily P/L': [
            0, -85570, -2730, 10830, 5450, -420, -13680, -2580, -1350, -3150,
            -2500, 40000, -13330, -3800, 0, 360, 20610, -1440, 2100, 600, 20000,
            -10730, -2210, 500, 150, -3900, 1500, 800, -800, -960, 510, 0,
            -12870, 8325, 42300, 6200, 31390, 9000, 414.45, 35970, -1700,
            -2400, 6345
        ],
        'Net P/L%': [
            0.0, -19.82, 0.71, 6.06, -12.13, -5.24, -43.88, -16.92, -10.93,
            -22.05, -20.57, 10.31, -20.71, -21.15, -11.09, -22.99, -7.79,
            -22.81, 10.63, 22.17, -19.48, -26.38, -20.98, -10.42, -15.72,
            -12.80, 2.52, -13.05, -15.74, -17.59, -12.00, -47.66, -40.05,
            2.42, 49.40, -37.95, 49.23, -21.35, 56.56, -33.33, -8.93,
            -41.46, -26.16
        ],
        'Daily P/L%': [
            0.0, -5.52, -1.07, 1.07, 0.75, -0.22, -0.51, -1.54, -0.47, -0.96,
            -0.72, 10.09, -2.12, -1.05, 0.0, 0.10, 3.85, -0.45, 0.44, 0.11,
            3.79, -2.30, -0.25, 0.13, 0.81, -1.90, 0.67, 0.09, -0.17, -0.35,
            1.17, 0.0, -0.56, 2.17, 1.72, 1.66, 1.27, 2.02, 0.35, 5.18,
            -1.80, -1.61, 0.64
        ],
        'RSI': [
            50.0, 45.0, 50.0, 50.0, 45.0, 50.0, 40.0, 45.0, 45.0, 40.0, 45.0,
            55.0, 45.0, 45.0, 50.0, 45.0, 55.0, 45.0, 55.0, 55.0, 45.0, 45.0,
            45.0, 45.0, 50.0, 45.0, 55.0, 45.0, 45.0, 45.0, 50.0, 40.0, 40.0,
            55.0, 55.0, 45.0, 55.0, 45.0, 55.0, 55.0, 45.0, 45.0, 45.0
        ]
    }
    
    # Calculate scaling factor for new total portfolio value
    current_total = sum(default_portfolio['Market Value'])
    new_total = 45000000  # New total portfolio value
    scaling_factor = new_total / current_total
    
    # Scale market values and P/L amounts
    default_portfolio['Market Value'] = [int(value * scaling_factor) for value in default_portfolio['Market Value']]
    default_portfolio['Daily P/L'] = [int(value * scaling_factor) for value in default_portfolio['Daily P/L']]
    default_portfolio['Net P/L'] = [int(value * scaling_factor) for value in default_portfolio['Net P/L']]
    
    # Save the portfolio data
    save_portfolio_data(default_portfolio)
    return default_portfolio

def display_portfolio_analysis(db_path=None):
    """
    Display portfolio analysis in the Streamlit dashboard
    """
    apply_shared_styles()
    create_custom_header("Portfolio Analysis")
    create_custom_divider()
    
    # Set default database path if none provided
    if db_path is None:
        # Get project root path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        db_path = os.path.join(
            project_root, 
            'data', 
            'databases', 
            'production',
            'PSX_investing_Stocks_KMI30_tracking.db'
        )
        logging.info(f"Using tracking database at: {db_path}")
    
    # Load portfolio data
    portfolio_data = load_portfolio_data()
    if not portfolio_data:
        st.error("No portfolio data found. Please update your portfolio first.")
        st.info("To update your portfolio, please use the portfolio update feature in the dashboard.")
        return
    
    # Convert to DataFrame and handle numeric values
    portfolio_df = pd.DataFrame(portfolio_data)
    numeric_columns = ['Market Value', 'Net P/L', 'Daily P/L', 'Net P/L%', 'Daily P/L%', 'RSI']
    for col in numeric_columns:
        if col in portfolio_df.columns:
            portfolio_df[col] = portfolio_df[col].apply(convert_to_float)
    
    # Portfolio Summary
    st.header("Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    
    total_value = portfolio_df['Market Value'].sum()
    total_pl = portfolio_df['Net P/L'].sum()
    daily_pl = portfolio_df['Daily P/L'].sum()
    
    with col1:
        st.metric("Total Portfolio Value", f"₨{format_number(total_value, 2)}")
    with col2:
        st.metric("Total P/L", f"₨{format_number(total_pl, 2)}", f"{format_number(total_pl/total_value*100, 2)}%")
    with col3:
        st.metric("Daily P/L", f"₨{format_number(daily_pl, 2)}", f"{format_number(daily_pl/total_value*100, 2)}%")
    
    # Performance Analysis
    st.header("Performance Analysis")
    
    # Top Performers
    st.subheader("Top Performers")
    top_performers = portfolio_df.nlargest(5, 'Net P/L%')
    st.dataframe(top_performers[['Name', 'Sector', 'Market Value', 'Net P/L%', 'Daily P/L%']])
    
    # Underperformers
    st.subheader("Underperformers")
    underperformers = portfolio_df.nsmallest(5, 'Net P/L%')
    st.dataframe(underperformers[['Name', 'Sector', 'Market Value', 'Net P/L%', 'Daily P/L%']])
    
    # Sector Analysis
    st.header("Sector Analysis")
    sector_data = portfolio_df.groupby('Sector').agg({
        'Market Value': 'sum',
        'Net P/L': 'sum',
        'Daily P/L': 'sum'
    }).reset_index()
    
    sector_data['Weight'] = sector_data['Market Value'] / total_value * 100
    sector_data['P/L%'] = sector_data['Net P/L'] / sector_data['Market Value'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sector = go.Figure(data=[go.Pie(
            labels=sector_data['Sector'],
            values=sector_data['Market Value'],
            hole=.3
        )])
        fig_sector.update_layout(title='Sector Distribution')
        st.plotly_chart(fig_sector)
    
    with col2:
        fig_performance = go.Figure(data=[go.Bar(
            x=sector_data['Sector'],
            y=sector_data['P/L%']
        )])
        fig_performance.update_layout(title='Sector Performance')
        st.plotly_chart(fig_performance)
    
    # Risk Analysis
    st.header("Risk Analysis")
    
    # Volatility Analysis
    st.subheader("Volatility Analysis")
    portfolio_df['Volatility'] = portfolio_df['Daily P/L%'].rolling(window=20).std().fillna(0)
    high_volatility = portfolio_df.nlargest(5, 'Volatility')
    st.dataframe(high_volatility[['Name', 'Sector', 'Market Value', 'Volatility', 'Daily P/L%']])
    
    # Position Sizing
    st.subheader("Position Sizing")
    portfolio_df['Position_Size'] = portfolio_df['Market Value'] / total_value * 100
    position_sizing = portfolio_df[['Name', 'Sector', 'Market Value', 'Position_Size']].sort_values('Position_Size', ascending=False)
    st.dataframe(position_sizing)
    
    # Recommendations
    st.header("Recommendations")
    
    # Take Profit
    st.subheader("Take Profit Opportunities")
    take_profit = portfolio_df[portfolio_df['Net P/L%'] > 20].sort_values('Net P/L%', ascending=False)
    if not take_profit.empty:
        st.dataframe(take_profit[['Name', 'Sector', 'Market Value', 'Net P/L%']])
    else:
        st.info("No significant take profit opportunities identified.")
    
    # Add to Positions
    st.subheader("Add to Positions")
    add_positions = portfolio_df[
        (portfolio_df['Net P/L%'] < 0) & 
        (portfolio_df['Daily P/L%'] > 0)
    ].sort_values('Daily P/L%', ascending=False)
    if not add_positions.empty:
        st.dataframe(add_positions[['Name', 'Sector', 'Market Value', 'Net P/L%', 'Daily P/L%']])
    else:
        st.info("No positions identified for adding.")
    
    # Exit Strategy
    st.subheader("Exit Strategy")
    exit_strategy = portfolio_df[
        (portfolio_df['Net P/L%'] < -10) & 
        (portfolio_df['Daily P/L%'] < 0)
    ].sort_values('Net P/L%')
    if not exit_strategy.empty:
        st.dataframe(exit_strategy[['Name', 'Sector', 'Market Value', 'Net P/L%', 'Daily P/L%']])
    else:
        st.info("No positions identified for exit.")

if __name__ == "__main__":
    display_portfolio_analysis() 