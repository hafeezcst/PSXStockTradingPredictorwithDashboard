"""
This script checks the database structure for the PSX Stock Trading Predictor dashboard.
"""
import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Get the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))

def check_database():
    """Check database structure and tables"""
    db_path = os.path.join(project_root, "data", "databases", "production", "PSX_investing_Stocks_KMI30.db")
    
    print(f"Checking database: {db_path}")
    print(f"Database exists: {os.path.exists(db_path)}")
    
    if not os.path.exists(db_path):
        print("Database file not found!")
        return
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\nFound {len(tables)} tables:")
        for table in tables:
            table_name = table[0]
            print(f"\n- Table: {table_name}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"  Columns ({len(columns)}):")
            for col in columns:
                print(f"    - {col[1]} ({col[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"  Row count: {row_count}")
        
        conn.close()
        print("\nDatabase check completed.")
    
    except Exception as e:
        print(f"Error checking database: {str(e)}")

if __name__ == "__main__":
    check_database()
