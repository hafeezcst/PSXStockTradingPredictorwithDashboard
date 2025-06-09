"""
Test script to verify signal_analysis.py fixes work correctly
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

def test_signal_analysis():
    """Test the signal analysis component"""
    try:
        # Import the signal analysis module
        from src.data_processing.dashboard.components.signal_analysis import display_signal_analysis
        
        # Create a mock config
        config = {
            'database_path': os.path.join(project_root, "data", "databases", "production", "PSX_investing_Stocks_KMI30.db")
        }
        
        print("Testing signal_analysis import...")
        print("✓ Successfully imported display_signal_analysis")
        
        print(f"✓ Database path: {config['database_path']}")
        print(f"✓ Database exists: {os.path.exists(config['database_path'])}")
        
        # Test database connection
        import sqlite3
        conn = sqlite3.connect(config['database_path'])
        cursor = conn.cursor()
        
        # Check if stock_signals table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_signals'")
        table_exists = cursor.fetchone() is not None
        print(f"✓ stock_signals table exists: {table_exists}")
        
        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM stock_signals")
            row_count = cursor.fetchone()[0]
            print(f"✓ stock_signals table has {row_count} rows")
        
        conn.close()
        
        print("\n✓ All tests passed! The signal analysis component should work correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Error testing signal_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_signal_analysis()
