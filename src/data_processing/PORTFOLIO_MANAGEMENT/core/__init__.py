"""
Core Portfolio Management Components

This module contains the main portfolio management functionality:
- SimplePortfolioManager: Main portfolio management class (working)
- PortfolioManager: Advanced portfolio manager (has issues, not imported)
- PortfolioConfig: Configuration management
- PortfolioAnalytics: Analytics and reporting
- RiskManager: Risk management utilities
"""

from .simple_portfolio_manager import SimplePortfolioManager

# Import working modules only
try:
    from .portfolio_config import ALL_CONFIGS
except ImportError:
    print("Warning: portfolio_config not available")
    ALL_CONFIGS = {}

try:
    from .portfolio_analytics import PortfolioAnalytics
except ImportError:
    print("Warning: portfolio_analytics not available")
    PortfolioAnalytics = None

try:
    from .risk_manager import RiskManager
except ImportError:
    print("Warning: risk_manager not available")
    RiskManager = None

# Don't import portfolio_manager due to indentation issues
# from .portfolio_manager import PortfolioManager

__all__ = [
    'SimplePortfolioManager',
    'PortfolioAnalytics', 
    'RiskManager',
    'ALL_CONFIGS'
]
