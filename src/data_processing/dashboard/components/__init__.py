"""
Dashboard components package initialization.
"""

# Import the DatabaseManager class first since it's a dependency
from .database_manager import DatabaseManager

# Expose the components that will be used in the dashboard
__all__ = [
    'DatabaseManager',
    'display_trading_signals',
    'display_signal_tracker',
    'display_charts',
    'display_indicator_analysis',
    'display_financial_reports',
    'display_portfolio_analysis',
    'display_market',
    'display_mutual_funds',
    'display_dividend_analysis',
    'apply_shared_styles',
    'create_custom_header',
    'create_custom_subheader',
    'create_custom_divider',
    'create_chart_container'
]

# Import other components after DatabaseManager to avoid circular imports
from .trading_signals import display_trading_signals
from .signal_tracker import display_signal_tracker
from .charts import display_charts
from .indicator_analysis import display_indicator_analysis
from .financial_reports import display_financial_reports
from .portfolio import display_portfolio_analysis
from .market import display_market
from .mutual_funds import display_mutual_funds
from .dividend_analysis import display_dividend_analysis
from .shared_styles import (
    apply_shared_styles,
    create_custom_header,
    create_custom_subheader,
    create_custom_divider,
    create_chart_container,
)
