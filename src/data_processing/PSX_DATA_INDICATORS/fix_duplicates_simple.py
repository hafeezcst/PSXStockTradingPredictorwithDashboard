#!/usr/bin/env python3
"""
Simple Database Finder and Duplicate Checker for PSX Stock Data
This script finds databases and checks for tables with duplicate records.
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_databases():
    """Find all database files in the project."""
    current_dir = Path.cwd()
    project_root = current_dir.parent.parent.parent
    
    print("🔍 Searching for database files...")
    
    # Search patterns
    search_paths = [
        project_root,
        current_dir,
        project_root / "src" / "data",
        project_root / "data",
    ]
    
    databases = []
    for search_path in search_paths:
        if search_path.exists():
            db_files = list(search_path.rglob("*.db"))
            databases.extend(db_files)
    
    # Remove duplicates
    unique_dbs = list(set(databases))
    
    print(f"Found {len(unique_dbs)} database files:")
    for i, db in enumerate(unique_dbs, 1):
        size_mb = db.stat().st_size / (1024 * 1024)
        print(f"  {i}. {db}")
        print(f"     Size: {size_mb:.1f} MB")
    
    return unique_dbs

def check_database_for_stock_tables(db_path: Path):
    """Check a database for stock-related tables and duplicates."""
    print(f"\\n📊 Checking database: {db_path.name}")
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            if not all_tables:
                print("   No tables found")
                return
            
            # Look for stock-related tables
            stock_patterns = ['PSX_', 'stock', 'data']
            stock_tables = []
            
            for table in all_tables:
                if any(pattern.lower() in table.lower() for pattern in stock_patterns):
                    stock_tables.append(table)
            
            print(f"   Total tables: {len(all_tables)}")
            print(f"   Stock-related tables: {len(stock_tables)}")
            
            if stock_tables:
                print("   Stock tables found:")
                duplicates_found = False
                
                for table in stock_tables[:10]:  # Check first 10
                    try:
                        # Get record count
                        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                        total_records = cursor.fetchone()[0]
                        
                        if total_records == 0:
                            print(f"     📄 {table}: 0 records")
                            continue
                        
                        # Check for Date column
                        cursor.execute(f"PRAGMA table_info(`{table}`)")
                        columns = [col[1] for col in cursor.fetchall()]
                        
                        if 'Date' in columns:
                            # Check for date duplicates
                            cursor.execute(f"SELECT COUNT(DISTINCT Date) FROM `{table}`")
                            unique_dates = cursor.fetchone()[0]
                            duplicate_count = total_records - unique_dates
                            
                            if duplicate_count > 0:
                                print(f"     ⚠️  {table}: {total_records:,} records, {duplicate_count} DUPLICATES!")
                                duplicates_found = True
                            else:
                                print(f"     ✅ {table}: {total_records:,} records, no duplicates")
                        else:
                            print(f"     📄 {table}: {total_records:,} records, no Date column")
                    
                    except Exception as e:
                        print(f"     ❌ {table}: Error - {e}")
                
                if duplicates_found:
                    print(f"\\n⚠️  DUPLICATES FOUND in {db_path.name}")
                    return db_path
    
    except Exception as e:
        print(f"   ❌ Error accessing database: {e}")
    
    return None

def simple_fix_duplicates(db_path: Path):
    """Simple duplicate fixing for a database."""
    print(f"\\n🔧 Fixing duplicates in: {db_path.name}")
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            
            # Get tables with duplicates
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            fixed_count = 0
            
            for table in tables:
                try:
                    # Check if table has Date column and duplicates
                    cursor.execute(f"PRAGMA table_info(`{table}`)")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    if 'Date' not in columns:
                        continue
                    
                    # Check for duplicates
                    cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                    total_records = cursor.fetchone()[0]
                    
                    if total_records == 0:
                        continue
                    
                    cursor.execute(f"SELECT COUNT(DISTINCT Date) FROM `{table}`")
                    unique_dates = cursor.fetchone()[0]
                    duplicate_count = total_records - unique_dates
                    
                    if duplicate_count > 0:
                        print(f"   🔧 Fixing {table} ({duplicate_count} duplicates)...")
                        
                        # Create backup
                        backup_table = f"{table}_backup_{int(time.time())}"
                        cursor.execute(f"CREATE TABLE `{backup_table}` AS SELECT * FROM `{table}`")
                        
                        # Read data into pandas for duplicate removal
                        df = pd.read_sql_query(f"SELECT * FROM `{table}`", conn, parse_dates=['Date'])
                        
                        # Remove duplicates (keep last)
                        df_cleaned = df.drop_duplicates(subset=['Date'], keep='last')
                        
                        # Replace table data
                        cursor.execute(f"DELETE FROM `{table}`")
                        df_cleaned.to_sql(table, conn, if_exists='append', index=False)
                        
                        removed = len(df) - len(df_cleaned)
                        print(f"     ✅ Removed {removed} duplicates from {table}")
                        fixed_count += 1
                
                except Exception as e:
                    print(f"     ❌ Error fixing {table}: {e}")
            
            print(f"\\n✅ Fixed duplicates in {fixed_count} tables")
            
    except Exception as e:
        print(f"❌ Error fixing database: {e}")

def main():
    """Main function."""
    print("=" * 80)
    print("🔧 PSX DATABASE DUPLICATE FINDER & FIXER")
    print("=" * 80)
    
    # Find all databases
    databases = find_databases()
    
    if not databases:
        print("❌ No database files found!")
        return
    
    # Check each database for stock tables and duplicates
    databases_with_duplicates = []
    
    for db in databases:
        if db.stat().st_size > 0:  # Skip empty databases
            result = check_database_for_stock_tables(db)
            if result:
                databases_with_duplicates.append(result)
    
    if not databases_with_duplicates:
        print("\\n✅ No duplicates found in any database!")
        return
    
    print(f"\\n⚠️  Found duplicates in {len(databases_with_duplicates)} database(s)")
    
    # Ask if user wants to fix
    print("\\n" + "─" * 60)
    response = input("Do you want to fix all duplicates? (y/N): ").strip().lower()
    
    if response == 'y':
        for db in databases_with_duplicates:
            simple_fix_duplicates(db)
        
        print("\\n🎉 All duplicates have been fixed!")
        print("You can now run the enhanced processor without duplicate errors!")
    else:
        print("❌ No changes made")

if __name__ == "__main__":
    main()
