import pytest
from unittest.mock import MagicMock, patch
from src.notifications.telegram_bot import TelegramNotifier
from src.core.stock_analyzer import StockSignal

@pytest.fixture
def mock_bot():
    """Mock Telegram bot instance"""
    return MagicMock()

@pytest.fixture
def sample_signal():
    """Generate sample stock signal"""
    return StockSignal(
        symbol='AAPL',
        price=150.25,
        signal='BUY',
        rsi=32.5,
        ao=1.8,
        timestamp='2025-06-14 16:15:00'
    )

def test_message_formatting(mock_bot, sample_signal):
    """Test Telegram message formatting"""
    notifier = TelegramNotifier(mock_bot)
    message = notifier.format_message(sample_signal)
    
    assert 'AAPL' in message
    assert 'BUY' in message
    assert 'RSI: 32.5' in message
    assert 'AO: 1.8' in message

@patch('src.notifications.telegram_bot.requests')
def test_send_notification(mock_requests, mock_bot, sample_signal):
    """Test notification sending"""
    mock_requests.post.return_value.status_code = 200
    notifier = TelegramNotifier(mock_bot)
    result = notifier.send_notification(sample_signal)
    
    assert result is True
    mock_bot.send_message.assert_called_once()

@patch('src.notifications.telegram_bot.requests')
def test_notification_failure(mock_requests, mock_bot, sample_signal):
    """Test notification failure handling"""
    mock_requests.post.return_value.status_code = 500
    notifier = TelegramNotifier(mock_bot)
    result = notifier.send_notification(sample_signal)
    
    assert result is False
    mock_bot.send_message.assert_not_called()

def test_batch_notifications(mock_bot):
    """Test batch notification processing"""
    notifier = TelegramNotifier(mock_bot)
    signals = [
        StockSignal('AAPL', 150.25, 'BUY', 32.5, 1.8, '2025-06-14 16:15:00'),
        StockSignal('MSFT', 420.10, 'SELL', 72.3, -0.5, '2025-06-14 16:15:00')
    ]
    
    results = notifier.process_batch(signals)
    assert len(results) == 2
    assert mock_bot.send_message.call_count == 2