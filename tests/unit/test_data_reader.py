"""
Unit tests for the data reader module
"""

import pytest
from src.psx_predictor.data.data_reader import DataReader

def test_data_reader_initialization():
    """Test DataReader initialization"""
    reader = DataReader()
    assert reader is not None

def test_fetch_stock_data():
    """Test stock data fetching"""
    reader = DataReader()
    data = reader.fetch_stock_data("AAPL")
    assert data is not None
    assert len(data) > 0

def test_process_stock_data():
    """Test stock data processing"""
    reader = DataReader()
    raw_data = reader.fetch_stock_data("AAPL")
    processed_data = reader.process_stock_data(raw_data)
    assert processed_data is not None
    assert len(processed_data) > 0 