#!/usr/bin/env python3
"""
Advanced duplicate index fixer for PSX stock data.
This script addresses the "cannot reindex on an axis with duplicate labels" error
by ensuring all DataFrames have unique indices when they're loaded from the database.
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_table_index_duplicates(db_path: str, table_name: str) -> bool:
    """
    Fix duplicate indices in a specific table by recreating it with unique dates.
    
    Args:
        db_path: Path to the database file
        table_name: Name of the table to fix
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Read the data
        df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
        logger.info(f"Loaded {len(df)} rows from {table_name}")
        
        if df.empty:
            logger.info(f"Table {table_name} is empty, skipping")
            conn.close()
            return True
            
        # Check if Date column exists
        if 'Date' not in df.columns:
            logger.warning(f"No Date column in {table_name}, skipping")
            conn.close()
            return True
            
        # Check for duplicate dates
        initial_count = len(df)
        df_unique = df.drop_duplicates(subset=['Date'], keep='last')
        duplicates_removed = initial_count - len(df_unique)
        
        if duplicates_removed > 0:
            logger.info(f"Found {duplicates_removed} duplicate dates in {table_name}")
            
            # Backup the original table
            backup_table = f"{table_name}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            conn.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}")
            
            # Drop the original table
            conn.execute(f"DROP TABLE {table_name}")
            
            # Recreate the table with unique data
            df_unique.to_sql(table_name, conn, if_exists='replace', index=False)
            
            conn.commit()
            logger.info(f"Fixed {table_name}: removed {duplicates_removed} duplicates")
        else:
            logger.info(f"No duplicates found in {table_name}")
            
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error fixing {table_name} in {db_path}: {e}")
        if 'conn' in locals():
            conn.close()
        return False

def find_and_fix_all_duplicates():
    """Find all PSX databases and fix duplicate indices in stock tables."""
    
    # Find all database files
    base_dir = Path("c:/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard")
    db_files = []
    
    # Look for PSX databases
    for pattern in ["*PSX*.db", "*.db"]:
        db_files.extend(base_dir.glob(pattern))
    
    # Also check common directories
    for subdir in ["data", "src", "databases"]:
        subdir_path = base_dir / subdir
        if subdir_path.exists():
            for pattern in ["*PSX*.db", "*.db"]:
                db_files.extend(subdir_path.rglob(pattern))
    
    logger.info(f"Found {len(db_files)} database files")
    
    total_fixed = 0
    for db_file in db_files:
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Filter for stock data tables
            stock_tables = [t for t in tables if 'stock_data' in t.lower() or '_psx_' in t.lower()]
            
            if stock_tables:
                logger.info(f"Processing {len(stock_tables)} stock tables in {db_file.name}")
                
                for table in stock_tables:
                    if fix_table_index_duplicates(str(db_file), table):
                        total_fixed += 1
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error processing {db_file}: {e}")
    
    logger.info(f"Fixed duplicates in {total_fixed} tables total")

if __name__ == "__main__":
    find_and_fix_all_duplicates()
    print("Duplicate index fixing completed!")
