"""
Script to check database tables and their contents.
"""

import sqlite3
import os
from pathlib import Path

def get_db_path():
    """Get the database path at the project root."""
    project_root = Path(__file__).resolve().parents[4]
    db_path = project_root / "data" / "databases" / "production" / "fairvalue.db"
    os.makedirs(db_path.parent, exist_ok=True)
    print(f"Using database path: {db_path}")
    return db_path

def check_tables():
    """Check database tables and their contents"""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print("\nAll tables in database:")
        for table in tables:
            print(f"\n=== {table} ===")
            
            # Get table structure
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            print("\nColumns:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"\nRow count: {count}")
            
            # Get sample data
            if count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                sample = cursor.fetchone()
                print("\nSample row:")
                for col, val in zip(columns, sample):
                    print(f"  - {col[1]}: {val}")
        
    except Exception as e:
        print(f"Error checking database: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_tables() 