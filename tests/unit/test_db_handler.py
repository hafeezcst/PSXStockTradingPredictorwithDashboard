import pytest
import asyncio
import pytest_asyncio
from src.data_processing.stock_analysis import db_handler

@pytest.mark.asyncio
async def test_fetch_signals():
    """Test async fetch_signals returns a list."""
    results = await db_handler.fetch_signals(limit=5)
    assert isinstance(results, list)

@pytest.mark.asyncio
async def test_fetch_stock_signals():
    """Test async fetch_stock_signals returns a list."""
    results = await db_handler.fetch_stock_signals(limit=5)
    assert isinstance(results, list)

def test_signal_model_instantiation():
    """Test Signal ORM model instantiation."""
    signal = db_handler.Signal(buy_stocks='A', sell_stocks='B', neutral_stocks='C')
    assert signal.buy_stocks == 'A' 