"""
Mutual funds component for the PSX dashboard.
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from typing import Dict, Any

def load_mutual_funds_data(file_path: str):
    """
    Load mutual funds favorites data from CSV file.
    
    Args:
        file_path: Path to the mutual funds favorites CSV file
        
    Returns:
        DataFrame containing the mutual funds data
    """
    try:
        df = pd.read_csv(file_path)
        # Convert rupee_invested to millions for better readability
        df['investment_millions'] = df['rupee_invested_000'] / 1_000_000
        return df
    except Exception as e:
        st.error(f"Error loading mutual funds data: {str(e)}")
        return None

def display_mutual_funds(config: Dict[str, Any]):
    """Display mutual funds analysis."""
    st.subheader("Mutual Funds Analysis")
    
    # Get file path from config or use default
    file_path = config.get("mutual_funds_file", 
                          "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/reports/mutual_funds_favorites.csv")
    
    # Check if file exists
    if not os.path.exists(file_path):
        st.error(f"Mutual funds data file not found: {file_path}")
        return
    
    # Load data
    df = load_mutual_funds_data(file_path)
    if df is None:
        return
    
    # Show last update date
    latest_date = pd.to_datetime(df['update_date'].iloc[0]).strftime('%Y-%m-%d')
    st.info(f"Data last updated on: {latest_date}")
    
    # Show summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stocks", len(df))
    with col2:
        st.metric("Total Funds Invested", f"{df['no_of_funds_invested'].sum():,}")
    with col3:
        total_investment = df['rupee_invested_000'].sum() / 1_000_000_000
        st.metric("Total Investment", f"Rs. {total_investment:.2f} Billion")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Charts"])
    
    with tab1:
        # Filter options
        min_funds = st.slider("Minimum number of funds invested", 1, int(df['no_of_funds_invested'].max()), 5)
        filtered_df = df[df['no_of_funds_invested'] >= min_funds].sort_values(by='no_of_funds_invested', ascending=False)
        
        # Display data table
        st.dataframe(
            filtered_df[['symbol', 'no_of_funds_invested', 'investment_millions', 'date_added']].rename(
                columns={
                    'symbol': 'Symbol',
                    'no_of_funds_invested': 'Number of Funds',
                    'investment_millions': 'Investment (Million Rs.)',
                    'date_added': 'Date Added'
                }
            ).style.format({
                'Investment (Million Rs.)': '{:,.2f}'
            }),
            use_container_width=True
        )
    
    with tab2:
        # Create charts
        chart_type = st.radio("Select Chart Type", ["Top Stocks by Funds", "Top Stocks by Investment"])
        top_n = st.slider("Number of stocks to display", 5, 30, 10)
        
        if chart_type == "Top Stocks by Funds":
            chart_df = df.nlargest(top_n, 'no_of_funds_invested')
            fig = px.bar(
                chart_df,
                x='symbol',
                y='no_of_funds_invested',
                title=f'Top {top_n} Stocks by Number of Funds Invested',
                labels={'symbol': 'Stock Symbol', 'no_of_funds_invested': 'Number of Funds'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Top Stocks by Investment
            chart_df = df.nlargest(top_n, 'investment_millions')
            fig = px.bar(
                chart_df,
                x='symbol',
                y='investment_millions',
                title=f'Top {top_n} Stocks by Investment Amount (Million Rs.)',
                labels={'symbol': 'Stock Symbol', 'investment_millions': 'Investment (Million Rs.)'}
            )
            st.plotly_chart(fig, use_container_width=True) 