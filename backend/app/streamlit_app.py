import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from typing import Dict, Any
import json

# Configure the page
st.set_page_config(
    page_title="PSX Investment Application",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("PSX Investment App")
st.sidebar.image("https://www.psx.com.pk/assets/images/psx-logo.png", width=200)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Portfolio", "Signals", "Analysis", "Settings"]
)

# API Configuration
API_URL = "http://localhost:8000/api/v1"

def fetch_data(endpoint: str) -> Dict[str, Any]:
    """Fetch data from the FastAPI backend"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return {}

# Dashboard Page
if page == "Dashboard":
    st.title("📊 Investment Dashboard")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Portfolio Value",
            value="₨32.0M",
            delta="+2.5%"
        )
    
    with col2:
        st.metric(
            label="Active Signals",
            value="8",
            delta="+2"
        )
    
    with col3:
        st.metric(
            label="Dividend Yield",
            value="4.2%",
            delta="+0.3%"
        )
    
    with col4:
        st.metric(
            label="Risk Score",
            value="Low",
            delta="-0.2"
        )
    
    # Portfolio Performance Chart
    st.subheader("Portfolio Performance")
    
    # Sample data - replace with actual data from API
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    portfolio_values = [32.0 + i * 0.5 for i in range(len(dates))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Date",
        yaxis_title="Value (Million PKR)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent Signals
    st.subheader("Recent Signals")
    
    # Sample data - replace with actual data from API
    signals_data = {
        "Symbol": ["OGDC", "PPL", "ENGRO", "HBL", "LUCK"],
        "Type": ["Buy", "Buy", "Sell", "Neutral", "Buy"],
        "Date": ["2024-02-15", "2024-02-14", "2024-02-13", "2024-02-12", "2024-02-11"],
        "Score": [8.5, 7.8, 6.2, 5.5, 8.9]
    }
    
    signals_df = pd.DataFrame(signals_data)
    st.dataframe(signals_df, use_container_width=True)

# Portfolio Page
elif page == "Portfolio":
    st.title("💼 Portfolio Management")
    
    # Portfolio Selection
    portfolio_name = st.selectbox(
        "Select Portfolio",
        ["Main Portfolio", "Conservative", "Aggressive"]
    )
    
    # Portfolio Holdings
    st.subheader("Current Holdings")
    
    # Sample data - replace with actual data from API
    holdings_data = {
        "Symbol": ["OGDC", "PPL", "ENGRO", "HBL", "LUCK"],
        "Shares": [1000, 500, 200, 300, 150],
        "Avg. Price": [85.5, 92.3, 350.0, 120.5, 450.0],
        "Current Price": [88.2, 94.5, 345.0, 122.0, 460.0],
        "Value": [88200, 47250, 69000, 36600, 69000],
        "Gain/Loss": ["+3.2%", "+2.4%", "-1.4%", "+1.2%", "+2.2%"]
    }
    
    holdings_df = pd.DataFrame(holdings_data)
    st.dataframe(holdings_df, use_container_width=True)
    
    # Add New Position
    st.subheader("Add New Position")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox("Stock Symbol", ["OGDC", "PPL", "ENGRO", "HBL", "LUCK"])
        shares = st.number_input("Number of Shares", min_value=1, value=100)
    
    with col2:
        price = st.number_input("Purchase Price", min_value=0.0, value=0.0)
        date = st.date_input("Purchase Date", value=datetime.now())
    
    if st.button("Add Position"):
        st.success("Position added successfully!")

# Signals Page
elif page == "Signals":
    st.title("🔔 Trading Signals")
    
    # Signal Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_type = st.selectbox(
            "Signal Type",
            ["All", "Buy", "Sell", "Neutral"]
        )
    
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now())
        )
    
    with col3:
        min_score = st.slider("Minimum Score", 0.0, 10.0, 5.0)
    
    # Signals Table
    st.subheader("Active Signals")
    
    # Sample data - replace with actual data from API
    signals_data = {
        "Symbol": ["OGDC", "PPL", "ENGRO", "HBL", "LUCK"],
        "Type": ["Buy", "Buy", "Sell", "Neutral", "Buy"],
        "Date": ["2024-02-15", "2024-02-14", "2024-02-13", "2024-02-12", "2024-02-11"],
        "Technical": [8.5, 7.8, 6.2, 5.5, 8.9],
        "Fundamental": [7.8, 8.2, 6.5, 5.8, 8.5],
        "Volume": ["1.2M", "0.8M", "1.5M", "0.9M", "1.1M"]
    }
    
    signals_df = pd.DataFrame(signals_data)
    st.dataframe(signals_df, use_container_width=True)

# Analysis Page
elif page == "Analysis":
    st.title("📊 Market Analysis")
    
    # Stock Selection
    symbol = st.selectbox(
        "Select Stock",
        ["OGDC", "PPL", "ENGRO", "HBL", "LUCK"]
    )
    
    # Technical Analysis
    st.subheader("Technical Analysis")
    
    # Sample data - replace with actual data from API
    technical_data = {
        "Indicator": ["RSI", "MACD", "Bollinger Bands", "Volume", "Moving Averages"],
        "Value": ["65.2", "Bullish", "Upper Band", "1.2M", "Golden Cross"],
        "Signal": ["Neutral", "Buy", "Sell", "Neutral", "Buy"]
    }
    
    technical_df = pd.DataFrame(technical_data)
    st.dataframe(technical_df, use_container_width=True)
    
    # Price Chart
    st.subheader("Price Chart")
    
    # Sample data - replace with actual data from API
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    prices = [100 + i * 0.5 for i in range(len(dates))]
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates,
        open=[p - 1 for p in prices],
        high=[p + 1 for p in prices],
        low=[p - 2 for p in prices],
        close=prices,
        name='Price'
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Date",
        yaxis_title="Price (PKR)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Settings Page
elif page == "Settings":
    st.title("⚙️ Settings")
    
    # User Profile
    st.subheader("User Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Full Name", value="John Doe")
        st.text_input("Email", value="john.doe@example.com")
    
    with col2:
        st.selectbox("Subscription Tier", ["Free", "Premium", "Enterprise"])
        st.button("Upgrade Subscription")
    
    # Notification Settings
    st.subheader("Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Email Notifications", value=True)
        st.checkbox("SMS Notifications", value=False)
    
    with col2:
        st.checkbox("Signal Alerts", value=True)
        st.checkbox("Portfolio Updates", value=True)
    
    # API Settings
    st.subheader("API Settings")
    
    st.text_input("DeepSeek API Key", type="password")
    st.text_input("Grok API Key", type="password")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>PSX Investment Application © 2024 | 
        <a href='#'>Documentation</a> | 
        <a href='#'>Support</a> | 
        <a href='#'>Privacy Policy</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
) 