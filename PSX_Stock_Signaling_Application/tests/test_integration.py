import pytest
from unittest.mock import patch, MagicMock
from src.main import TradingApp
from src.core.stock_analyzer import StockAnalyzer
from src.data.database import StockDatabase
from src.notifications.telegram_bot import TelegramNotifier

@pytest.fixture
def mock_components():
    """Mock all external components"""
    with patch('src.data.database.StockDatabase') as mock_db, \
         patch('src.notifications.telegram_bot.TelegramNotifier') as mock_notifier:
        
        mock_db.return_value.get_latest_data.return_value = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=14),
            'close': [100, 102, 101, 105, 103, 107, 110, 108, 109, 112, 115, 114, 116, 118]
        })
        
        mock_notifier.return_value.send_notification.return_value = True
        
        yield mock_db, mock_notifier

def test_full_workflow(mock_components):
    """Test complete application workflow"""
    mock_db, mock_notifier = mock_components
    
    app = TradingApp()
    results = app.run_analysis(['AAPL'])
    
    # Verify database interaction
    mock_db.return_value.get_latest_data.assert_called_once_with('AAPL', days=14)
    
    # Verify analysis occurred
    assert len(results) == 1
    assert 'AAPL' in results[0].symbol
    
    # Verify notification sent
    mock_notifier.return_value.send_notification.assert_called_once()

def test_error_handling(mock_components):
    """Test error handling across components"""
    mock_db, mock_notifier = mock_components
    mock_db.return_value.get_latest_data.side_effect = Exception("DB Error")
    
    app = TradingApp()
    with pytest.raises(Exception):
        app.run_analysis(['AAPL'])
    
    # Verify no notifications sent on error
    mock_notifier.return_value.send_notification.assert_not_called()

def test_multiple_symbols(mock_components):
    """Test processing multiple symbols"""
    mock_db, mock_notifier = mock_components
    
    app = TradingApp()
    results = app.run_analysis(['AAPL', 'MSFT', 'GOOG'])
    
    assert len(results) == 3
    assert mock_notifier.return_value.send_notification.call_count == 3

def test_empty_data_handling(mock_components):
    """Test handling of empty data responses"""
    mock_db, mock_notifier = mock_components
    mock_db.return_value.get_latest_data.return_value = pd.DataFrame()
    
    app = TradingApp()
    results = app.run_analysis(['AAPL'])
    
    assert len(results) == 0
    mock_notifier.return_value.send_notification.assert_not_called()