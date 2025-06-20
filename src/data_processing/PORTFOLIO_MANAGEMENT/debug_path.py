#!/usr/bin/env python3
"""
Debug script to test database path calculation
"""

import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

from core.portfolio_config import ALL_CONFIGS

def test_path_calculation():
    """Test the database path calculation"""
    
    print("=== DATABASE PATH CALCULATION TEST ===")
    
    # Current approach in launch_portfolio_system.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    project_root = os.path.abspath(project_root)
    relative_db_path = ALL_CONFIGS['signals']['signal_db_path']
    db_path = os.path.join(project_root, relative_db_path)
    db_path = os.path.abspath(db_path)
    
    print(f"Current file: {__file__}")
    print(f"Current dir: {current_dir}")
    print(f"Project root (3 levels up): {project_root}")
    print(f"Relative DB path from config: {relative_db_path}")
    print(f"Final calculated DB path: {db_path}")
    print(f"Database exists: {os.path.exists(db_path)}")
    
    # Test what the simple_portfolio_manager approach does
    print("\n=== SIMPLE PORTFOLIO MANAGER APPROACH ===")
    from core.simple_portfolio_manager import SimplePortfolioManager
    pm = SimplePortfolioManager()
    print(f"SimplePortfolioManager DB path: {pm.signal_db_path}")
    print(f"SPM Database exists: {os.path.exists(pm.signal_db_path)}")
    
    # Manual verification of correct path
    print("\n=== MANUAL PATH VERIFICATION ===")
    # From portfolio_management folder, we need to go up 3 levels to get to project root
    manual_project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    manual_db_path = os.path.join(manual_project_root, "data", "databases", "production", "PSX_investing_Stocks_KMI100.db")
    manual_db_path = os.path.abspath(manual_db_path)
    print(f"Manual project root: {manual_project_root}")
    print(f"Manual DB path: {manual_db_path}")
    print(f"Manual DB exists: {os.path.exists(manual_db_path)}")
    
    # Test if database file is accessible
    if os.path.exists(manual_db_path):
        print("\n=== DATABASE CONNECTION TEST ===")
        import sqlite3
        try:
            conn = sqlite3.connect(manual_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM buy_stocks WHERE Status='Buy' AND Success='Yes'")
            buy_count = cursor.fetchone()[0]
            print(f"✅ Database connection successful! Buy signals: {buy_count}")
            conn.close()
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
    else:
        print("❌ Database file not found at manual path")

if __name__ == "__main__":
    test_path_calculation()
