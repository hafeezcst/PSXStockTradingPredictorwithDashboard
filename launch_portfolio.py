#!/usr/bin/env python3
"""
Portfolio Management System Launcher
Convenient launcher from project root
"""

import sys
import os

# Add the portfolio management module to path
portfolio_path = os.path.join(os.path.dirname(__file__), 'src', 'data_processing', 'portfolio_management')
sys.path.insert(0, portfolio_path)

if __name__ == "__main__":
    # Change working directory to portfolio management folder
    os.chdir(portfolio_path)
    
    # Import and run the main launcher
    from launch_portfolio_system import main
    main()
