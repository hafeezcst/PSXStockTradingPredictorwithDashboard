import pytest
from pathlib import Path
from src.core.database import DatabaseManager
from config.path_resolver import path_resolver

@pytest.fixture
def db_manager():
    """Fixture providing initialized DatabaseManager"""
    db_path = path_resolver.resolve('data', 'databases', 'PSX_KMI30.db')
    return DatabaseManager(db_path)

def test_database_connection(db_manager):
    """Test database connection establishment"""
    assert db_manager.is_connected() is True

def test_table_exists(db_manager):
    """Verify required tables exist"""
    required_tables = ['stocks', 'signals', 'historical_data']
    for table in required_tables:
        assert db_manager.table_exists(table) is True

def test_data_retrieval(db_manager):
    """Test basic data retrieval"""
    test_symbol = 'OGDC'
    data = db_manager.get_stock_data(test_symbol)
    assert len(data) > 0
    assert 'date' in data[0]
    assert 'close' in data[0]

def test_signal_storage(db_manager):
    """Test signal storage functionality"""
    test_signal = {
        'symbol': 'TEST',
        'signal': 'BUY', 
        'strength': 0.85,
        'date': '2025-06-14'
    }
    assert db_manager.store_signal(test_signal) is True