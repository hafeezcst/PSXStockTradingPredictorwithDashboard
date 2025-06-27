import pytest
from src.data_processing.stock_analysis.analytics_engine import AnalyticsEngine

def test_engine_instantiation():
    """Test AnalyticsEngine instantiation."""
    engine = AnalyticsEngine()
    assert isinstance(engine, AnalyticsEngine)

def test_add_remove_notification_rule():
    """Test adding and removing notification rules."""
    engine = AnalyticsEngine()
    rule = {'id': 1, 'type': 'price', 'threshold': 100}
    engine.add_notification_rule(rule)
    assert len(engine.notification_rules) == 1
    engine.remove_notification_rule(1)
    assert len(engine.notification_rules) == 0

def test_export_to_dataframe():
    """Test export_to_dataframe returns a DataFrame."""
    engine = AnalyticsEngine()
    data = [{'symbol': 'ABC', 'signal': 'BUY'}, {'symbol': 'XYZ', 'signal': 'SELL'}]
    df = engine.export_to_dataframe(data)
    assert df.shape[0] == 2 