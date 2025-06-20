#!/usr/bin/env python3
"""
Check KMI30 database for stock data
"""

import sqlite3
from pathlib import Path

def check_kmi30_database():
    db_path = Path(r'c:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\src\data\databases\production\PSX_investing_Stocks_KMI30.db')
    
    if not db_path.exists():
        print("❌ KMI30 database not found")
        return
    
    print(f"✅ Found KMI30 database: {db_path}")
    print(f"   Size: {db_path.stat().st_size / (1024*1024):.1f} MB")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        all_tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📊 Total tables: {len(all_tables)}")
        
        # Look for stock-related tables
        stock_tables = [t for t in all_tables if 'stock' in t.lower()]
        psx_tables = [t for t in all_tables if t.startswith('PSX_')]
        
        print(f"📈 Stock tables: {len(stock_tables)}")
        print(f"🏢 PSX tables: {len(psx_tables)}")
        
        # Show first 10 tables with record counts
        print("\n📋 Sample tables:")
        for table in all_tables[:10]:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                count = cursor.fetchone()[0]
                
                # Check for Date column
                cursor.execute(f"PRAGMA table_info(`{table}`)")
                columns = [col[1] for col in cursor.fetchall()]
                has_date = 'Date' in columns
                
                status = "📅" if has_date else "❌"
                print(f"   {status} {table}: {count:,} records")
                
                # Check for duplicates if has Date and records > 0
                if has_date and count > 0:
                    cursor.execute(f"SELECT COUNT(DISTINCT Date) FROM `{table}`")
                    unique_dates = cursor.fetchone()[0]
                    duplicates = count - unique_dates
                    if duplicates > 0:
                        print(f"      ⚠️  {duplicates} duplicate records!")
                
            except Exception as e:
                print(f"   ❌ {table}: Error - {e}")

if __name__ == "__main__":
    check_kmi30_database()
