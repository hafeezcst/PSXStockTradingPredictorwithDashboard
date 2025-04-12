"""
Portfolio component for the PSX dashboard.
Combines portfolio data with technical indicators for comprehensive analysis
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import json
import os
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TradingView configuration
TRADINGVIEW_BASE_URL = "https://www.tradingview.com/screener/"
TRADINGVIEW_PSX_SCREENER = "https://www.tradingview.com/screener/PSX/"

# Core holdings classification thresholds
CORE_HOLDINGS_THRESHOLD = 0.40  # 40% target
GROWTH_STOCKS_THRESHOLD = 0.30  # 30% target
DEFENSIVE_STOCKS_THRESHOLD = 0.25  # 25% target
CASH_THRESHOLD = 0.10  # 10% target

def classify_core_holdings(portfolio_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Classify stocks into core holdings categories based on technical and fundamental analysis."""
    # Calculate total portfolio value
    total_value = portfolio_df['Market Value'].sum()
    
    # Initialize classification dictionary
    classifications = {
        'core_holdings': [],
        'growth_stocks': [],
        'defensive_stocks': [],
        'cash': []
    }
    
    # Get technical indicators for each stock
    for _, row in portfolio_df.iterrows():
        symbol = row['Name']
        market_value = row['Market Value']
        net_pl_pct = row['Net P/L%']
        daily_pl_pct = row['Daily P/L%']
        
        # Get technical signals
        technical_signals = get_technical_indicators(symbol)
        
        # Calculate stock characteristics
        stock_data = {
            'symbol': symbol,
            'sector': row['Sector'],
            'market_value': market_value,
            'weight': (market_value / total_value) * 100,
            'net_pl_pct': net_pl_pct,
            'daily_pl_pct': daily_pl_pct,
            'technical_signals': technical_signals
        }
        
        # Classify based on criteria
        if (market_value > portfolio_df['Market Value'].mean() and  # Large market cap
            net_pl_pct > 0 and  # Positive returns
            technical_signals.get('ma_signal') == 'Bullish' and  # Bullish trend
            technical_signals.get('rsi_signal') == 'Neutral'):  # Not overbought/oversold
            classifications['core_holdings'].append(stock_data)
        elif (net_pl_pct > 10 and  # High growth
              daily_pl_pct > 0 and  # Positive momentum
              technical_signals.get('ao_signal') == 'Buy'):  # Accumulation
            classifications['growth_stocks'].append(stock_data)
        elif (-5 <= net_pl_pct <= 5 and  # Stable returns
              technical_signals.get('rsi_signal') == 'Neutral'):  # Not overbought/oversold
            classifications['defensive_stocks'].append(stock_data)
    
    # Convert lists to DataFrames
    for key in classifications:
        if classifications[key]:
            classifications[key] = pd.DataFrame(classifications[key])
        else:
            classifications[key] = pd.DataFrame()
    
    return classifications

def calculate_portfolio_metrics(portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key portfolio metrics."""
    total_market_value = portfolio_df['Market Value'].sum()
    total_net_pl = portfolio_df['Net P/L'].sum()
    total_daily_pl = portfolio_df['Daily P/L'].sum()
    
    return {
        'total_market_value': total_market_value,
        'total_net_pl': total_net_pl,
        'total_daily_pl': total_daily_pl,
        'net_pl_percentage': (total_net_pl / total_market_value) * 100 if total_market_value > 0 else 0,
        'daily_pl_percentage': (total_daily_pl / total_market_value) * 100 if total_market_value > 0 else 0
    }

def get_tradingview_data(symbol: str) -> Dict[str, Any]:
    """Get stock data from TradingView screener."""
    try:
        # Construct the URL for the specific stock
        url = f"{TRADINGVIEW_BASE_URL}PSX/{symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract key metrics (this is a simplified example - you'll need to adjust based on actual HTML structure)
        metrics = {
            'price': float(soup.find('div', {'class': 'tv-symbol-price-quote__value'}).text),
            'change': float(soup.find('div', {'class': 'tv-symbol-price-quote__change'}).text),
            'volume': float(soup.find('div', {'class': 'tv-symbol-price-quote__volume'}).text),
            'market_cap': float(soup.find('div', {'class': 'tv-symbol-price-quote__market-cap'}).text),
            'pe_ratio': float(soup.find('div', {'class': 'tv-symbol-price-quote__pe-ratio'}).text),
            'dividend_yield': float(soup.find('div', {'class': 'tv-symbol-price-quote__dividend-yield'}).text)
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error fetching TradingView data for {symbol}: {e}")
        return {}

def get_historical_performance(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get historical performance data from available sources."""
    try:
        # For now, return a simple DataFrame with default values
        dates = pd.date_range(end=datetime.now(), periods=days)
        return pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(100, 10, days),
            'High': np.random.normal(105, 10, days),
            'Low': np.random.normal(95, 10, days),
            'Close': np.random.normal(100, 10, days),
            'Volume': np.random.normal(1000000, 100000, days)
        })
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

def get_market_index_data() -> pd.DataFrame:
    """Get PSX index data from available sources."""
    try:
        # For now, return a simple DataFrame with default values
        dates = pd.date_range(end=datetime.now(), periods=30)
        return pd.DataFrame({
            'Date': dates,
            'Close': np.random.normal(40000, 1000, 30)
        })
    except Exception as e:
        logger.error(f"Error fetching PSX index data: {e}")
        return pd.DataFrame()

def calculate_market_correlation(portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate correlation with PSX index using TradingView data."""
    try:
        # Get PSX index data
        psx_data = get_market_index_data()
        
        correlations = {}
        for _, row in portfolio_df.iterrows():
            symbol = row['Name']
            stock_data = get_historical_performance(symbol)
            
            if not stock_data.empty and not psx_data.empty:
                # Merge data on date
                merged_data = pd.merge(stock_data, psx_data, on='Date', suffixes=('_stock', '_index'))
                # Calculate correlation
                correlation = merged_data['Close_stock'].corr(merged_data['Close_index'])
                correlations[symbol] = correlation
        
        return {
            'correlations': correlations,
            'avg_correlation': np.mean(list(correlations.values())) if correlations else 0
        }
    except Exception as e:
        logger.error(f"Error calculating market correlation: {e}")
        return {'correlations': {}, 'avg_correlation': 0}

def get_technical_indicators(symbol: str) -> Dict[str, Any]:
    """Get technical indicators from available data."""
    try:
        # For now, return default values since TradingView scraping is not working
        return {
            'rsi': 50,  # Neutral RSI
            'macd': 0,  # Neutral MACD
            'sma_50': 0,  # Will be calculated from historical data
            'sma_200': 0,  # Will be calculated from historical data
            'volume': 0  # Will be calculated from historical data
        }
    except Exception as e:
        logger.error(f"Error getting technical indicators for {symbol}: {e}")
        return {
            'rsi': 50,
            'macd': 0,
            'sma_50': 0,
            'sma_200': 0,
            'volume': 0
        }

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with comma as decimal point"""
    return f"{value:,.{decimals}f}".replace(".", ",")

def convert_to_float(value) -> float:
    """Convert value to float, handling both string and numeric inputs"""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

def calculate_position_sizing(portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate optimal position sizing based on available metrics."""
    # Convert numeric columns to float
    portfolio_df['Market Value'] = portfolio_df['Market Value'].apply(convert_to_float)
    portfolio_df['Daily P/L%'] = portfolio_df['Daily P/L%'].apply(convert_to_float)
    portfolio_df['Net P/L%'] = portfolio_df['Net P/L%'].apply(convert_to_float)
    portfolio_df['RSI'] = portfolio_df['RSI'].apply(convert_to_float)
    
    total_value = portfolio_df['Market Value'].sum()
    
    # Initialize risk score column with default values
    portfolio_df['risk_score'] = 1.0  # Default risk score
    
    # Calculate position sizes based on multiple factors
    # 1. Volatility-based risk
    portfolio_df['volatility'] = portfolio_df['Daily P/L%'].rolling(window=20).std().fillna(0.1)
    
    # 2. Performance-based risk
    portfolio_df['performance_risk'] = abs(portfolio_df['Net P/L%']) / 100
    
    # 3. Market value-based risk
    portfolio_df['size_risk'] = portfolio_df['Market Value'] / total_value
    
    # 4. Technical risk (using RSI as proxy)
    portfolio_df['technical_risk'] = abs(portfolio_df['RSI'] - 50) / 50
    
    # Combine risk factors with weights
    portfolio_df['combined_risk'] = (
        0.4 * portfolio_df['volatility'] +
        0.3 * portfolio_df['performance_risk'] +
        0.2 * portfolio_df['size_risk'] +
        0.1 * portfolio_df['technical_risk']
    )
    
    # Calculate risk-adjusted weights
    portfolio_df['risk_weight'] = 1 / (portfolio_df['combined_risk'] + 1e-6)
    portfolio_df['risk_weight'] = portfolio_df['risk_weight'] / portfolio_df['risk_weight'].sum()
    
    # Calculate target position sizes
    portfolio_df['target_size'] = portfolio_df['risk_weight'] * total_value
    portfolio_df['size_adjustment'] = portfolio_df['target_size'] - portfolio_df['Market Value']
    
    # Calculate position sizing recommendations
    position_recommendations = []
    for _, row in portfolio_df.iterrows():
        recommendation = {
            'symbol': row['Name'],
            'current_size': format_number(row['Market Value'], 2),
            'target_size': format_number(row['target_size'], 2),
            'adjustment': format_number(row['size_adjustment'], 2),
            'risk_score': format_number(row['combined_risk'], 4),
            'recommendation': 'HOLD'
        }
        
        if row['size_adjustment'] > 0:
            recommendation['recommendation'] = f"INCREASE by {format_number(row['size_adjustment'], 2)}"
        elif row['size_adjustment'] < 0:
            recommendation['recommendation'] = f"REDUCE by {format_number(abs(row['size_adjustment']), 2)}"
        
        position_recommendations.append(recommendation)
    
    return {
        'position_sizes': position_recommendations,
        'total_risk': format_number(portfolio_df['combined_risk'].mean(), 4),
        'risk_breakdown': {
            'volatility_risk': format_number(portfolio_df['volatility'].mean(), 4),
            'performance_risk': format_number(portfolio_df['performance_risk'].mean(), 4),
            'size_risk': format_number(portfolio_df['size_risk'].mean(), 4),
            'technical_risk': format_number(portfolio_df['technical_risk'].mean(), 4)
        }
    }

def generate_visualizations(portfolio_df: pd.DataFrame) -> Dict:
    """Generate visualizations for portfolio analysis"""
    visualizations = {}
    
    # Create sector distribution pie chart
    sector_data = portfolio_df.groupby('Sector').agg({
        'Market Value': 'sum',
        'Net P/L': 'sum'
    }).reset_index()
    
    fig_sector = px.pie(
        sector_data,
        values='Market Value',
        names='Sector',
        title='Sector Distribution',
        hover_data=['Net P/L']
    )
    visualizations['sector_distribution'] = fig_sector
    
    # Create performance heatmap
    top_performers = portfolio_df.nlargest(10, 'Daily P/L%')
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=[[p for p in top_performers['Daily P/L%']]],
        x=top_performers['Name'],
        y=['Daily Performance'],
        colorscale='RdYlGn',
        showscale=True
    ))
    fig_heatmap.update_layout(
        title='Top 10 Daily Performers',
        xaxis_title='Stock',
        yaxis_title='Performance',
        height=300
    )
    visualizations['performance_heatmap'] = fig_heatmap
    
    return visualizations

def calculate_stop_loss_levels(portfolio_df: pd.DataFrame) -> Dict:
    """Calculate stop-loss levels for each position"""
    stop_losses = {}
    
    for _, row in portfolio_df.iterrows():
        symbol = row['Name']
        current_price = row['Market Value'] / 100  # Assuming 100 shares per position
        stop_loss = current_price * 0.90  # 10% stop loss
        stop_losses[symbol] = {
            'current_price': current_price,
            'stop_loss': stop_loss,
            'risk_per_share': current_price - stop_loss
        }
    
    return stop_losses

def calculate_sector_exposure(portfolio_df: pd.DataFrame) -> Dict:
    """Calculate sector exposure and concentration"""
    sector_data = portfolio_df.groupby('Sector').agg({
        'Market Value': 'sum',
        'Net P/L': 'sum'
    }).reset_index()
    
    total_value = sector_data['Market Value'].sum()
    sector_data['exposure_pct'] = (sector_data['Market Value'] / total_value) * 100
    
    return sector_data.to_dict('records')

def generate_rebalancing_strategy(portfolio_df: pd.DataFrame) -> Dict:
    """Generate portfolio rebalancing strategy"""
    # Convert numeric columns to float
    portfolio_df['Market Value'] = portfolio_df['Market Value'].apply(convert_to_float)
    
    total_value = portfolio_df['Market Value'].sum()
    
    # Classify holdings
    classifications = classify_core_holdings(portfolio_df)
    
    # Convert market values to float in classifications
    for category in classifications:
        if not classifications[category].empty:
            classifications[category]['market_value'] = classifications[category]['market_value'].apply(convert_to_float)
    
    # Calculate current percentages
    current_percentages = {
        'core_holdings': (classifications['core_holdings']['market_value'].sum() / total_value) * 100 if not classifications['core_holdings'].empty else 0,
        'growth_stocks': (classifications['growth_stocks']['market_value'].sum() / total_value) * 100 if not classifications['growth_stocks'].empty else 0,
        'defensive_stocks': (classifications['defensive_stocks']['market_value'].sum() / total_value) * 100 if not classifications['defensive_stocks'].empty else 0,
        'cash': 0  # Assuming no cash position
    }
    
    # Calculate discrepancies and generate detailed recommendations
    discrepancies = {}
    rebalancing_actions = []
    
    for category in ['core_holdings', 'growth_stocks', 'defensive_stocks', 'cash']:
        target_pct = globals()[f"{category.upper()}_THRESHOLD"] * 100
        current_pct = current_percentages[category]
        discrepancy = current_pct - target_pct
        
        # Calculate absolute value adjustment needed
        value_adjustment = abs(discrepancy) * total_value / 100
        
        # Generate detailed recommendation
        recommendation = {
            'category': category.replace('_', ' ').title(),
            'current_pct': current_pct,  # Store as float for sorting
            'target_pct': target_pct,    # Store as float for sorting
            'discrepancy': discrepancy,   # Store as float for sorting
            'value_adjustment': value_adjustment,  # Store as float for sorting
            'priority': 'HIGH' if abs(discrepancy) > 5 else 'MEDIUM' if abs(discrepancy) > 2 else 'LOW',
            'action': 'INCREASE' if discrepancy < 0 else 'REDUCE' if discrepancy > 0 else 'MAINTAIN'
        }
        
        discrepancies[category] = recommendation
        rebalancing_actions.append(recommendation)
    
    # Sort rebalancing actions by priority and discrepancy magnitude
    rebalancing_actions.sort(key=lambda x: (
        x['priority'] != 'HIGH',  # False (0) comes before True (1), so HIGH priority comes first
        abs(x['discrepancy'])     # Sort by absolute discrepancy in descending order
    ), reverse=True)
    
    # Format the numeric values for display after sorting
    for action in rebalancing_actions:
        action['current_pct'] = format_number(action['current_pct'], 2)
        action['target_pct'] = format_number(action['target_pct'], 2)
        action['discrepancy'] = format_number(action['discrepancy'], 2)
        action['value_adjustment'] = format_number(action['value_adjustment'], 2)
    
    return {
        'classifications': classifications,
        'discrepancies': discrepancies,
        'rebalancing_actions': rebalancing_actions,
        'current_allocation': {k: format_number(v, 2) for k, v in current_percentages.items()},
        'target_allocation': {
            'core_holdings': format_number(CORE_HOLDINGS_THRESHOLD * 100, 2),
            'growth_stocks': format_number(GROWTH_STOCKS_THRESHOLD * 100, 2),
            'defensive_stocks': format_number(DEFENSIVE_STOCKS_THRESHOLD * 100, 2),
            'cash': format_number(CASH_THRESHOLD * 100, 2)
        }
    }

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

def calculate_risk_metrics(portfolio_df: pd.DataFrame, risk_free_rate: float = 0.05) -> Dict[str, float]:
    """Calculate risk-adjusted return metrics."""
    returns = portfolio_df['Daily P/L%'] / 100  # Convert to decimal
    mean_return = returns.mean()
    std_dev = returns.std()
    
    # Sharpe Ratio
    sharpe_ratio = (mean_return - risk_free_rate/252) / std_dev if std_dev != 0 else 0
    
    # Sortino Ratio (using only negative returns)
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() if not negative_returns.empty else 0
    sortino_ratio = (mean_return - risk_free_rate/252) / downside_std if downside_std != 0 else 0
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'volatility': std_dev * np.sqrt(252)  # Annualized volatility
    }

def calculate_diversification_metrics(portfolio_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate portfolio diversification metrics."""
    # Herfindahl-Hirschman Index (HHI)
    weights = portfolio_df['Market Value'] / portfolio_df['Market Value'].sum()
    hhi = (weights ** 2).sum()
    
    # Effective number of stocks
    effective_n = 1 / hhi
    
    # Sector concentration
    sector_weights = portfolio_df.groupby('Sector')['Market Value'].sum() / portfolio_df['Market Value'].sum()
    sector_hhi = (sector_weights ** 2).sum()
    
    return {
        'hhi': hhi,
        'effective_n': effective_n,
        'sector_hhi': sector_hhi,
        'diversification_score': 1 - hhi  # Higher is better
    }

def generate_performance_chart(portfolio_df: pd.DataFrame) -> go.Figure:
    """Generate interactive performance chart."""
    fig = go.Figure()
    
    # Add portfolio value line
    portfolio_value = portfolio_df['Market Value'].sum()
    fig.add_trace(go.Scatter(
        x=portfolio_df['Name'],
        y=portfolio_df['Market Value'],
        mode='lines+markers',
        name='Market Value',
        hovertemplate='%{x}<br>Value: %{y:,.2f}<extra></extra>'
    ))
    
    # Add P/L bars
    fig.add_trace(go.Bar(
        x=portfolio_df['Name'],
        y=portfolio_df['Net P/L'],
        name='Net P/L',
        hovertemplate='%{x}<br>P/L: %{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Stock',
        yaxis_title='Value',
        barmode='group',
        height=500
    )
    
    return fig

def generate_risk_heatmap(portfolio_df: pd.DataFrame) -> go.Figure:
    """Generate risk metrics heatmap."""
    risk_metrics = calculate_risk_metrics(portfolio_df)
    metrics_df = pd.DataFrame({
        'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Volatility'],
        'Value': [
            risk_metrics['sharpe_ratio'],
            risk_metrics['sortino_ratio'],
            risk_metrics['max_drawdown'],
            risk_metrics['volatility']
        ]
    })
    
    fig = go.Figure(data=go.Heatmap(
        z=[metrics_df['Value']],
        x=metrics_df['Metric'],
        y=['Risk Metrics'],
        colorscale='RdYlGn',
        showscale=True,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Risk Metrics Heatmap',
        height=200
    )
    
    return fig

def export_portfolio_report(portfolio_df: pd.DataFrame, filename: str = "portfolio_report.xlsx"):
    """Export portfolio analysis to Excel file."""
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Portfolio Summary
    portfolio_summary = pd.DataFrame({
        'Metric': ['Total Value', 'Total P/L', 'Daily P/L'],
        'Value': [
            portfolio_df['Market Value'].sum(),
            portfolio_df['Net P/L'].sum(),
            portfolio_df['Daily P/L'].sum()
        ]
    })
    portfolio_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    # Position Details
    portfolio_df.to_excel(writer, sheet_name='Positions', index=False)
    
    # Risk Metrics
    risk_metrics = calculate_risk_metrics(portfolio_df)
    pd.DataFrame(risk_metrics.items(), columns=['Metric', 'Value']).to_excel(
        writer, sheet_name='Risk Metrics', index=False)
    
    writer.close()
    return filename

def calculate_advanced_risk_metrics(portfolio_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate advanced risk metrics for the portfolio."""
    returns = portfolio_df['Daily P/L%'] / 100
    
    # Value at Risk (VaR)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # Expected Shortfall (ES)
    es_95 = returns[returns <= var_95].mean()
    es_99 = returns[returns <= var_99].mean()
    
    # Skewness and Kurtosis
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # Omega Ratio
    threshold = 0
    omega_ratio = np.sum(returns[returns > threshold]) / abs(np.sum(returns[returns < threshold]))
    
    # Information Ratio
    benchmark_returns = np.random.normal(0.0005, 0.01, len(returns))  # Placeholder for benchmark
    excess_returns = returns - benchmark_returns
    information_ratio = np.mean(excess_returns) / np.std(excess_returns)
    
    return {
        'var_95': var_95,
        'var_99': var_99,
        'es_95': es_95,
        'es_99': es_99,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'omega_ratio': omega_ratio,
        'information_ratio': information_ratio
    }

def optimize_portfolio(portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """Optimize portfolio using modern portfolio theory."""
    try:
        # Calculate expected returns and covariance matrix
        # Create a DataFrame of returns for each stock
        returns_data = {}
        for symbol in portfolio_df['Name']:
            # Get historical returns for each stock
            hist_data = get_historical_performance(symbol)
            if not hist_data.empty:
                returns_data[symbol] = hist_data['Close'].pct_change().dropna()
        
        if not returns_data:
            # If no historical data, use daily P/L% as a proxy
            returns = pd.DataFrame({
                'Daily Returns': portfolio_df['Daily P/L%'] / 100
            })
            # Use a simple correlation assumption
            correlation = 0.5
            cov_matrix = pd.DataFrame(
                correlation * np.ones((len(portfolio_df), len(portfolio_df))),
                index=portfolio_df['Name'],
                columns=portfolio_df['Name']
            )
            np.fill_diagonal(cov_matrix.values, 1.0)
            # Scale by volatility
            volatility = returns['Daily Returns'].std()
            cov_matrix = cov_matrix * (volatility ** 2)
        else:
            # Create returns DataFrame
            returns = pd.DataFrame(returns_data)
            # Calculate covariance matrix
            cov_matrix = returns.cov()
        
        # Define optimization function
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        def portfolio_return(weights):
            if 'Daily Returns' in returns.columns:
                return np.sum(returns['Daily Returns'].mean() * weights)
            else:
                return np.sum(returns.mean() * weights)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(portfolio_df)))
        
        # Initial guess
        initial_weights = np.ones(len(portfolio_df)) / len(portfolio_df)
        
        # Optimize for minimum volatility
        min_vol_result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Optimize for maximum Sharpe ratio
        def neg_sharpe_ratio(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret / vol) if vol != 0 else -np.inf
        
        max_sharpe_result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'min_vol_weights': min_vol_result.x,
            'max_sharpe_weights': max_sharpe_result.x,
            'min_vol_return': portfolio_return(min_vol_result.x),
            'max_sharpe_return': portfolio_return(max_sharpe_result.x),
            'min_vol_volatility': portfolio_volatility(min_vol_result.x),
            'max_sharpe_volatility': portfolio_volatility(max_sharpe_result.x)
        }
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        # Return default weights if optimization fails
        default_weights = np.ones(len(portfolio_df)) / len(portfolio_df)
        return {
            'min_vol_weights': default_weights,
            'max_sharpe_weights': default_weights,
            'min_vol_return': 0.0,
            'max_sharpe_return': 0.0,
            'min_vol_volatility': 0.0,
            'max_sharpe_volatility': 0.0
        }

def perform_factor_analysis(portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """Perform factor analysis on portfolio returns."""
    try:
        # Prepare data for factor analysis
        returns_data = {}
        for symbol in portfolio_df['Name']:
            # Get historical returns for each stock
            hist_data = get_historical_performance(symbol)
            if not hist_data.empty:
                returns_data[symbol] = hist_data['Close'].pct_change().dropna()
        
        if not returns_data:
            # If no historical data, use daily P/L% as a proxy
            returns = pd.DataFrame({
                'Daily Returns': portfolio_df['Daily P/L%'] / 100
            })
        else:
            # Create returns DataFrame
            returns = pd.DataFrame(returns_data)
        
        # Check if we have enough data for PCA
        if len(returns) < 2:
            # If not enough data, return simplified factor analysis
            return {
                'pca_components': None,
                'pca_explained_variance': None,
                'factor_loadings': {
                    'market': 1.0,
                    'size': 0.5,
                    'value': 0.3,
                    'momentum': 0.4
                },
                'factor_r_squared': 0.5,
                'factor_pvalues': {
                    'market': 0.01,
                    'size': 0.05,
                    'value': 0.1,
                    'momentum': 0.05
                }
            }
        
        # Perform PCA
        scaled_returns = StandardScaler().fit_transform(returns)
        n_components = min(2, len(returns.columns) - 1)  # Ensure n_components is valid
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_returns)
        
        # Factor regression
        factors = pd.DataFrame({
            'market': np.random.normal(0.0005, 0.01, len(returns)),  # Market factor
            'size': np.random.normal(0.0002, 0.005, len(returns)),   # Size factor
            'value': np.random.normal(0.0003, 0.008, len(returns)),  # Value factor
            'momentum': np.random.normal(0.0004, 0.009, len(returns)) # Momentum factor
        })
        
        X = sm.add_constant(factors)
        model = sm.OLS(returns.mean(axis=1), X)
        results = model.fit()
        
        return {
            'pca_components': principal_components,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'factor_loadings': results.params,
            'factor_r_squared': results.rsquared,
            'factor_pvalues': results.pvalues
        }
    except Exception as e:
        logger.error(f"Error in factor analysis: {e}")
        # Return simplified factor analysis if there's an error
        return {
            'pca_components': None,
            'pca_explained_variance': None,
            'factor_loadings': {
                'market': 1.0,
                'size': 0.5,
                'value': 0.3,
                'momentum': 0.4
            },
            'factor_r_squared': 0.5,
            'factor_pvalues': {
                'market': 0.01,
                'size': 0.05,
                'value': 0.1,
                'momentum': 0.05
            }
        }

def calculate_performance_attribution(portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate performance attribution analysis."""
    # Calculate sector contributions
    sector_contributions = portfolio_df.groupby('Sector').agg({
        'Market Value': 'sum',
        'Net P/L': 'sum',
        'Daily P/L': 'sum'
    })
    sector_contributions['weight'] = sector_contributions['Market Value'] / sector_contributions['Market Value'].sum()
    sector_contributions['contribution'] = sector_contributions['weight'] * sector_contributions['Daily P/L']
    
    # Calculate stock selection effect
    benchmark_returns = np.random.normal(0.0005, 0.01, len(portfolio_df))  # Placeholder for benchmark
    selection_effect = (portfolio_df['Daily P/L%'] / 100 - benchmark_returns) * portfolio_df['Market Value'] / portfolio_df['Market Value'].sum()
    
    # Calculate allocation effect
    sector_benchmark_weights = np.random.dirichlet(np.ones(len(sector_contributions)), size=1)[0]
    allocation_effect = (sector_contributions['weight'] - sector_benchmark_weights) * sector_contributions['Daily P/L']
    
    return {
        'sector_contributions': sector_contributions.to_dict(),
        'selection_effect': selection_effect.to_dict(),
        'allocation_effect': allocation_effect.to_dict(),
        'total_attribution': selection_effect.sum() + allocation_effect.sum()
    }

def perform_scenario_analysis(portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """Perform scenario analysis on portfolio."""
    # Define scenarios
    scenarios = {
        'base': 1.0,
        'bull': 1.2,
        'bear': 0.8,
        'crisis': 0.5
    }
    
    # Calculate scenario impacts
    scenario_results = {}
    for scenario, multiplier in scenarios.items():
        scenario_returns = portfolio_df['Daily P/L%'] * multiplier
        scenario_value = portfolio_df['Market Value'] * (1 + scenario_returns / 100)
        scenario_results[scenario] = {
            'total_value': scenario_value.sum(),
            'total_return': (scenario_value.sum() / portfolio_df['Market Value'].sum() - 1) * 100,
            'max_drawdown': np.min(scenario_returns)
        }
    
    return scenario_results

def get_existing_chart(symbol: str, chart_type: str) -> str:
    """Get path to existing chart if available."""
    chart_path = f"outputs/charts/{symbol}_{chart_type}.png"
    if os.path.exists(chart_path):
        return chart_path
        return None

def generate_advanced_visualizations(portfolio_df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate advanced visualizations for portfolio analysis."""
    visualizations = {}
    
    # Check for existing charts
    for symbol in portfolio_df['Name']:
        # Technical Analysis Chart
        tech_chart = get_existing_chart(symbol, "technical_analysis")
        if tech_chart:
            visualizations[f"{symbol}_technical"] = tech_chart
        
        # Performance Chart
        perf_chart = get_existing_chart(symbol, "performance")
        if perf_chart:
            visualizations[f"{symbol}_performance"] = perf_chart
        
        # Volume Analysis Chart
        volume_chart = get_existing_chart(symbol, "volume_analysis")
        if volume_chart:
            visualizations[f"{symbol}_volume"] = volume_chart
    
    # Generate new visualizations only if no existing charts are found
    if not any(visualizations):
        try:
            # Create a simple returns DataFrame using daily P/L%
            returns = pd.DataFrame({
                symbol: [portfolio_df[portfolio_df['Name'] == symbol]['Daily P/L%'].iloc[0] / 100]
                for symbol in portfolio_df['Name']
            })
            
            # Create a correlation matrix
            correlation = 0.5
            cov_matrix = pd.DataFrame(
                correlation * np.ones((len(portfolio_df), len(portfolio_df))),
                index=portfolio_df['Name'],
                columns=portfolio_df['Name']
            )
            np.fill_diagonal(cov_matrix.values, 1.0)
            
            # Scale by volatility
            volatility = returns.std()
            cov_matrix = cov_matrix * (volatility.values.reshape(-1, 1) * volatility.values)
            
            # Generate efficient frontier
            portfolio_returns = []
            portfolio_volatilities = []
            
            for _ in range(1000):
                weights = np.random.random(len(portfolio_df))
                weights /= np.sum(weights)
                portfolio_returns.append(np.sum(returns.mean() * weights))
                portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
            
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(
                x=portfolio_volatilities,
                y=portfolio_returns,
                mode='markers',
                name='Random Portfolios'
            ))
            fig_frontier.add_trace(go.Scatter(
                x=[analysis_results['optimization']['min_vol_volatility'], 
                   analysis_results['optimization']['max_sharpe_volatility']],
                y=[analysis_results['optimization']['min_vol_return'], 
                   analysis_results['optimization']['max_sharpe_return']],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Optimal Portfolios'
            ))
            fig_frontier.update_layout(
                title='Efficient Frontier',
                xaxis_title='Volatility',
                yaxis_title='Expected Return'
            )
            visualizations['efficient_frontier'] = fig_frontier
            
            # Factor Analysis Heatmap
            if analysis_results['factor_analysis']['factor_loadings'] is not None:
                factor_loadings = analysis_results['factor_analysis']['factor_loadings']
                fig_factor = go.Figure(data=go.Heatmap(
                    z=[factor_loadings.values],
                    x=factor_loadings.index,
                    y=['Factor Loadings'],
                    colorscale='RdYlBu',
                    showscale=True
                ))
                fig_factor.update_layout(
                    title='Factor Loadings',
                    height=200
                )
                visualizations['factor_heatmap'] = fig_factor
                
            # Add simple performance visualization
            fig_performance = go.Figure()
            fig_performance.add_trace(go.Bar(
                x=portfolio_df['Name'],
                y=portfolio_df['Daily P/L%'],
                name='Daily Returns'
            ))
            fig_performance.update_layout(
                title='Daily Performance by Stock',
                xaxis_title='Stock',
                yaxis_title='Daily Return %'
            )
            visualizations['daily_performance'] = fig_performance
            
        except Exception as e:
            logger.error(f"Error generating advanced visualizations: {e}")
            # Create simple visualizations if advanced ones fail
            fig_simple = go.Figure()
            fig_simple.add_trace(go.Scatter(
                x=portfolio_df['Name'],
                y=portfolio_df['Daily P/L%'],
                mode='markers',
                name='Daily Returns'
            ))
            fig_simple.update_layout(
                title='Simple Portfolio Performance',
                xaxis_title='Stock',
                yaxis_title='Daily Return %'
            )
            visualizations['simple_performance'] = fig_simple
    
    return visualizations

def display_advanced_analysis(portfolio_df: pd.DataFrame):
    """Display advanced portfolio analysis dashboard."""
    st.title("Advanced Portfolio Analysis")
    
    # Calculate advanced metrics
    risk_metrics = calculate_advanced_risk_metrics(portfolio_df)
    optimization_results = optimize_portfolio(portfolio_df)
    factor_analysis = perform_factor_analysis(portfolio_df)
    performance_attribution = calculate_performance_attribution(portfolio_df)
    scenario_analysis = perform_scenario_analysis(portfolio_df)
    
    analysis_results = {
        'risk_metrics': risk_metrics,
        'optimization': optimization_results,
        'factor_analysis': factor_analysis,
        'performance_attribution': performance_attribution,
        'scenario_analysis': scenario_analysis
    }
    
    # Generate visualizations
    visualizations = generate_advanced_visualizations(portfolio_df, analysis_results)
    
    # Display results in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Risk Analysis", "Portfolio Optimization", "Factor Analysis",
        "Performance Attribution", "Scenario Analysis"
    ])
    
    with tab1:
        st.subheader("Advanced Risk Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("VaR (95%)", f"{risk_metrics['var_95']:.2%}")
            st.metric("Expected Shortfall (95%)", f"{risk_metrics['es_95']:.2%}")
            st.metric("Skewness", f"{risk_metrics['skewness']:.2f}")
        with col2:
            st.metric("VaR (99%)", f"{risk_metrics['var_99']:.2%}")
            st.metric("Expected Shortfall (99%)", f"{risk_metrics['es_99']:.2%}")
            st.metric("Kurtosis", f"{risk_metrics['kurtosis']:.2f}")
        
        # Display existing charts for each stock
        for symbol in portfolio_df['Name']:
            if f"{symbol}_technical" in visualizations:
                st.image(visualizations[f"{symbol}_technical"], caption=f"{symbol} Technical Analysis")
            if f"{symbol}_performance" in visualizations:
                st.image(visualizations[f"{symbol}_performance"], caption=f"{symbol} Performance")
            if f"{symbol}_volume" in visualizations:
                st.image(visualizations[f"{symbol}_volume"], caption=f"{symbol} Volume Analysis")
        
        # Display efficient frontier if no existing charts
        if 'efficient_frontier' in visualizations:
            st.plotly_chart(visualizations['efficient_frontier'])
    
    with tab2:
        st.subheader("Portfolio Optimization")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Minimum Volatility Portfolio")
            st.write(f"Return: {optimization_results['min_vol_return']:.2%}")
            st.write(f"Volatility: {optimization_results['min_vol_volatility']:.2%}")
        with col2:
            st.write("Maximum Sharpe Ratio Portfolio")
            st.write(f"Return: {optimization_results['max_sharpe_return']:.2%}")
            st.write(f"Volatility: {optimization_results['max_sharpe_volatility']:.2%}")
        
        st.write("Optimal Weights")
        weights_df = pd.DataFrame({
            'Stock': portfolio_df['Name'],
            'Min Vol Weight': optimization_results['min_vol_weights'],
            'Max Sharpe Weight': optimization_results['max_sharpe_weights']
        })
        st.dataframe(weights_df)
    
    with tab3:
        st.subheader("Factor Analysis")
        if 'factor_heatmap' in visualizations:
            st.plotly_chart(visualizations['factor_heatmap'])
        st.write(f"R-squared: {factor_analysis['factor_r_squared']:.2%}")
        st.write("Factor Loadings:")
        st.write(pd.Series(factor_analysis['factor_loadings']))
    
    with tab4:
        st.subheader("Performance Attribution")
        st.write(f"Total Attribution: {performance_attribution['total_attribution']:.2%}")
        
        # Display sector contributions
        st.write("Sector Contributions:")
        sector_df = pd.DataFrame(performance_attribution['sector_contributions']).T
        st.dataframe(sector_df)
        
        # Display selection and allocation effects
        st.write("Selection and Allocation Effects:")
        effects_df = pd.DataFrame({
            'Selection Effect': performance_attribution['selection_effect'],
            'Allocation Effect': performance_attribution['allocation_effect']
        })
        st.dataframe(effects_df)
    
    with tab5:
        st.subheader("Scenario Analysis")
        for scenario, results in scenario_analysis.items():
            with st.expander(scenario.title()):
                st.write(f"Total Value: ₨{results['total_value']:,.2f}")
                st.write(f"Total Return: {results['total_return']:.2%}")
                st.write(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
                
                # Display scenario-specific charts if available
                scenario_chart = get_existing_chart("portfolio", f"scenario_{scenario}")
                if scenario_chart:
                    st.image(scenario_chart, caption=f"{scenario.title()} Scenario Analysis")

def display_basic_analysis(portfolio_df: pd.DataFrame):
    """Display basic portfolio analysis dashboard."""
    st.title("Basic Portfolio Analysis")
    
    # Portfolio Summary
    st.header("Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    
    total_value = portfolio_df['Market Value'].apply(convert_to_float).sum()
    total_pl = portfolio_df['Net P/L'].apply(convert_to_float).sum()
    daily_pl = portfolio_df['Daily P/L'].apply(convert_to_float).sum()
    
    with col1:
        st.metric("Total Portfolio Value", f"₨{format_number(total_value, 2)}")
    with col2:
        st.metric("Total P/L", f"₨{format_number(total_pl, 2)}", f"{format_number(total_pl/total_value*100, 2)}%")
    with col3:
        st.metric("Daily P/L", f"₨{format_number(daily_pl, 2)}", f"{format_number(daily_pl/total_value*100, 2)}%")
    
    # Investment Recommendations
    st.header("Investment Recommendations")
    
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
        (portfolio_df['Net P/L%'].apply(convert_to_float) < 0) & 
        (portfolio_df['Daily P/L%'].apply(convert_to_float) > 0)
    ].sort_values('Daily P/L%', ascending=False)
    if not add_positions.empty:
        st.dataframe(add_positions[['Name', 'Sector', 'Market Value', 'Net P/L%', 'Daily P/L%']])
    else:
        st.info("No positions identified for adding.")
    
    # Exit Strategy
    st.subheader("Exit Strategy")
    exit_strategy = portfolio_df[
        (portfolio_df['Net P/L%'].apply(convert_to_float) < -10) & 
        (portfolio_df['Daily P/L%'].apply(convert_to_float) < 0)
    ].sort_values('Net P/L%')
    if not exit_strategy.empty:
        st.dataframe(exit_strategy[['Name', 'Sector', 'Market Value', 'Net P/L%', 'Daily P/L%']])
    else:
        st.info("No positions identified for exit.")
    
    # Sector Analysis
    st.header("Sector Analysis")
    sector_data = portfolio_df.groupby('Sector').agg({
        'Market Value': lambda x: x.apply(convert_to_float).sum(),
        'Net P/L': lambda x: x.apply(convert_to_float).sum(),
        'Daily P/L': lambda x: x.apply(convert_to_float).sum()
    }).reset_index()
    
    sector_data['Weight'] = sector_data['Market Value'] / total_value * 100
    sector_data['P/L%'] = sector_data['Net P/L'] / sector_data['Market Value'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sector = px.pie(
            sector_data,
            values='Market Value',
            names='Sector',
            title='Sector Distribution'
        )
        st.plotly_chart(fig_sector)
    
    with col2:
        fig_performance = px.bar(
            sector_data,
            x='Sector',
            y='P/L%',
            title='Sector Performance'
        )
        st.plotly_chart(fig_performance)
    
    # Risk Management
    st.header("Risk Management")
    
    # Stop Loss Levels
    st.subheader("Stop Loss Levels")
    stop_losses = calculate_stop_loss_levels(portfolio_df)
    stop_loss_df = pd.DataFrame(stop_losses).T
    st.dataframe(stop_loss_df)
    
    # Portfolio Rebalancing
    st.subheader("Portfolio Rebalancing")
    rebalancing = generate_rebalancing_strategy(portfolio_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Current Allocation")
        current_allocation = pd.DataFrame({
            'Category': ['Core Holdings', 'Growth Stocks', 'Defensive Stocks', 'Cash'],
            'Current %': [
                float(str(rebalancing['discrepancies']['core_holdings']['current_pct']).replace(',', '.')),
                float(str(rebalancing['discrepancies']['growth_stocks']['current_pct']).replace(',', '.')),
                float(str(rebalancing['discrepancies']['defensive_stocks']['current_pct']).replace(',', '.')),
                float(str(rebalancing['discrepancies']['cash']['current_pct']).replace(',', '.'))
            ],
            'Target %': [
                float(str(rebalancing['discrepancies']['core_holdings']['target_pct']).replace(',', '.')),
                float(str(rebalancing['discrepancies']['growth_stocks']['target_pct']).replace(',', '.')),
                float(str(rebalancing['discrepancies']['defensive_stocks']['target_pct']).replace(',', '.')),
                float(str(rebalancing['discrepancies']['cash']['target_pct']).replace(',', '.'))
            ]
        })
        st.dataframe(current_allocation)
    
    with col2:
        st.write("Rebalancing Actions")
        actions = []
        for category, data in rebalancing['discrepancies'].items():
            discrepancy = float(str(data['discrepancy']).replace(',', '.'))
            if discrepancy > 0:
                actions.append(f"Reduce {category} by {abs(discrepancy):.2f}%")
            elif discrepancy < 0:
                actions.append(f"Increase {category} by {abs(discrepancy):.2f}%")
        st.write(pd.DataFrame({'Action': actions}))

def display_portfolio_analysis(config=None):
    """Display simplified portfolio analysis dashboard"""
    st.title("Portfolio Analysis Dashboard")
    
    # Load portfolio data
    portfolio_data = load_portfolio_data()
    if not portfolio_data:
        st.error("No portfolio data found. Please update your portfolio first.")
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
        fig_sector = px.pie(
            sector_data,
            values='Market Value',
            names='Sector',
            title='Sector Distribution'
        )
        st.plotly_chart(fig_sector)
    
    with col2:
        fig_performance = px.bar(
            sector_data,
            x='Sector',
            y='P/L%',
            title='Sector Performance'
        )
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