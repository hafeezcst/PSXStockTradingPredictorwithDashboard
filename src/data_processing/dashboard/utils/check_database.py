"""
This script checks the database structure and verifies that the necessary tables exist.
"""
import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def check_database_structure(db_path):
    """
    Check the structure of the database and print information about tables.
    """
    print(f"Checking database structure for: {db_path}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Found {len(tables)} tables: {', '.join(tables)}")
        
        # Get columns for each table
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor]
            print(f"\nTable: {table}")
            print(f"Columns ({len(columns)}): {', '.join(columns)}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            print(f"Row count: {row_count}")
            
            # Sample data (first row)
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                sample = cursor.fetchone()
                print(f"Sample data: {dict(zip(columns, sample))}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error checking database structure: {str(e)}")
        return False

if __name__ == "__main__":    # Get the database path from command line arguments or use default
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = os.path.join(project_root, "data", "databases", "production", "PSX_investing_Stocks_KMI30.db")
    
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        sys.exit(1)
    
    check_database_structure(db_path)
