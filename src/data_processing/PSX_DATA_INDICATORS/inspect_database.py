#!/usr/bin/env python3
"""
Database Table Inspector for PSX Stock Data
"""

import sqlite3
import pandas as pd
from pathlib import Path

def inspect_database():
    """Inspect the PSX database to see what tables exist."""
    
    # Find database
    db_locations = [
        Path.cwd().parent.parent.parent / "PSX_Stock_Data.db",
        Path.cwd() / "PSX_Stock_Data.db",
        Path.cwd() / "data" / "databases" / "production" / "PSX_consolidated_data_PSX.db",
        Path("../../../PSX_Stock_Data.db"),
        Path("../../../../PSX_Stock_Data.db"),
    ]
    
    db_path = None
    for location in db_locations:
        if location.exists():
            db_path = location
            break
    
    if not db_path:
        print("❌ Could not find PSX database file!")
        return
    
    print(f"🔍 Inspecting database: {db_path}")
    print(f"   Size: {db_path.stat().st_size / (1024*1024):.1f} MB")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name, sql FROM sqlite_master 
            WHERE type='table'
            ORDER BY name
        """)
        tables = cursor.fetchall()
        
        print(f"\n📊 Found {len(tables)} tables:")
        print("-" * 80)
        
        for table_name, create_sql in tables:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = cursor.fetchone()[0]
            
            # Get column info
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = cursor.fetchall()
            
            print(f"\n🗂️  Table: {table_name}")
            print(f"   📈 Row count: {row_count:,}")
            print(f"   📋 Columns: {len(columns)}")
            
            # Show column names
            column_names = [col[1] for col in columns]
            print(f"   🏷️  Column names: {', '.join(column_names)}")
            
            # Check for Date column and potential duplicates
            if 'Date' in column_names and row_count > 0:
                cursor.execute(f"SELECT COUNT(DISTINCT Date) FROM `{table_name}`")
                unique_dates = cursor.fetchone()[0]
                duplicates = row_count - unique_dates
                
                if duplicates > 0:
                    print(f"   ⚠️  DUPLICATES: {duplicates} duplicate records found!")
                else:
                    print(f"   ✅ No duplicates found")
            
            # Show sample data
            if row_count > 0:
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
                sample_data = cursor.fetchall()
                print(f"   📄 Sample data (first 3 rows):")
                for i, row in enumerate(sample_data, 1):
                    print(f"      {i}: {row}")
        
        print("\n" + "=" * 80)
        print("✅ Database inspection complete!")

if __name__ == "__main__":
    inspect_database()
