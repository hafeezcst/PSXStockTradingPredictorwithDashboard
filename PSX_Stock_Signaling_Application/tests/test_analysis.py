import pytest
import pandas as pd
from src.core.stock_analyzer import StockAnalyzer
from config.path_resolver import path_resolver

@pytest.fixture
def sample_data():
    """Generate sample stock data for testing"""
    return pd.DataFrame({
        'date': pd.date_range(start='2025-01-01', periods=14),
        'close': [100, 102, 101, 105, 103, 107, 110, 108, 109, 112, 115, 114, 116, 118]
    })

def test_rsi_calculation(sample_data):
    """Test RSI calculation logic"""
    analyzer = StockAnalyzer()
    result = analyzer.calculate_rsi(sample_data, period=14)
    
    assert 'rsi' in result.columns
    assert len(result) == len(sample_data)
    assert 0 <= result['rsi'].iloc[-1] <= 100

def test_signal_generation(sample_data):
    """Test buy/sell signal generation"""
    analyzer = StockAnalyzer()
    result = analyzer.generate_signals(sample_data)
    
    assert 'signal' in result.columns
    assert result['signal'].isin(['BUY', 'SELL', 'HOLD']).all()

def test_trend_identification(sample_data):
    """Test trend detection logic"""
    analyzer = StockAnalyzer()
    result = analyzer.identify_trends(sample_data)
    
    assert 'trend' in result.columns
    assert result['trend'].isin(['UP', 'DOWN', 'SIDEWAYS']).all()

def test_technical_indicators(sample_data):
    """Test all technical indicators"""
    analyzer = StockAnalyzer()
    result = analyzer.calculate_indicators(sample_data)
    
    required_indicators = ['sma_20', 'ema_12', 'macd', 'ao']
    for indicator in required_indicators:
        assert indicator in result.columns