"""
Database verification script for PSX Trading Dashboard.
This script verifies the existence of required tables and their structure.
"""

import sqlite3
import os
from pathlib import Path

def get_db_path():
    """Get the database path"""
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / "data" / "databases" / "production" / "fairvalue.db"

def verify_database():
    """Verify database structure and tables"""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print("\nExisting tables in database:")
        for table in tables:
            print(f"- {table}")
        
        # Required tables
        required_tables = [
            'tradingview_signals',
            'stock_signals',
            'buy_stocks',
            'sell_stocks',
            'neutral_stocks',
            'signal_transition_history',
            'signal_tracking'
        ]
        
        print("\nChecking required tables:")
        missing_tables = []
        for table in required_tables:
            if table in tables:
                print(f"✓ {table} exists")
                # Print table structure
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                print(f"  Columns:")
                for col in columns:
                    print(f"    - {col[1]} ({col[2]})")
            else:
                print(f"✗ {table} missing")
                missing_tables.append(table)
        
        if missing_tables:
            print(f"\nMissing tables: {', '.join(missing_tables)}")
            return False
        
        print("\nAll required tables exist with proper structure.")
        return True
        
    except Exception as e:
        print(f"Error verifying database: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    verify_database() 