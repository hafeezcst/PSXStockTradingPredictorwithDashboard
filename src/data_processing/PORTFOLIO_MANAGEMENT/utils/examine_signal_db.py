#!/usr/bin/env python3
"""
Script to examine the structure of the PSX_investing_Stocks_KMI100.db database
"""

import sqlite3
import pandas as pd
import os

def examine_database():
    db_path = r"data\databases\production\PSX_investing_Stocks_KMI100.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Available tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check if the specific tables exist
        required_tables = ['buy_stocks', 'sell_stocks', 'neutral_stocks']
        
        for table_name in required_tables:
            print(f"\n--- Table: {table_name} ---")
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            if cursor.fetchone() is None:
                print(f"Table {table_name} does not exist!")
                continue
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            print("Columns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
            
            # Get sample data
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"Row count: {count}")
            
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                sample_data = cursor.fetchall()
                
                print("Sample data:")
                for row in sample_data:
                    print(f"  {row}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error examining database: {e}")

if __name__ == "__main__":
    examine_database()
