"""
Portfolio Management Package

A comprehensive portfolio management system for PSX (Pakistan Stock Exchange) 
that integrates with signal databases for automated trading and portfolio optimization.

Modules:
- core: Core portfolio management functionality
- utils: Utility functions and tools
- tests: Test suite for portfolio management
- docs: Documentation and analysis

Author: Portfolio Management System
Version: 1.0.0
Date: June 18, 2025
"""

__version__ = "1.0.0"
__author__ = "Portfolio Management System"

from .core.simple_portfolio_manager import SimplePortfolioManager

# Make the main class available at package level
__all__ = ['SimplePortfolioManager']
