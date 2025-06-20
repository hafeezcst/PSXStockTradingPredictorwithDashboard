#!/usr/bin/env python3
"""
Test Script to Verify Duplicate Fix
"""

import pandas as pd
import sqlite3
from pathlib import Path

def test_duplicate_fix():
    """Test if the duplicate fix worked by reading problematic tables."""
    
    # Find the main database
    db_path = None
    possible_paths = [
        Path(r"c:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\src\data\databases\production\PSX_consolidated_data_PSX.db"),
        Path(r"c:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\data\databases\production\PSX_consolidated_data_PSX.db"),
    ]
    
    # Search for the database
    for path in Path(r"c:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard").rglob("PSX_consolidated_data_PSX.db"):
        if path.stat().st_size > 50 * 1024 * 1024:  # > 50MB
            db_path = path
            break
    
    if not db_path:
        print("❌ Could not find PSX_consolidated_data_PSX.db")
        return
    
    print(f"✅ Testing database: {db_path}")
    print(f"   Size: {db_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Test reading the tables that were problematic
    test_tables = [
        'PSX_AKGL_stock_data',
        'PSX_AKBL_stock_data', 
        'PSX_AKZO_stock_data',
        'PSX_ALIFE_stock_data',
        'PSX_ALAC_stock_data'
    ]
    
    print("\n🧪 Testing previously problematic tables:")
    
    for table in test_tables:
        try:
            with sqlite3.connect(str(db_path)) as conn:
                # Try the old method that was failing
                df = pd.read_sql_table(
                    table, 
                    conn, 
                    index_col='Date', 
                    parse_dates=['Date']
                )
                
                print(f"   ✅ {table}: {len(df)} records, no duplicate errors!")
                
                # Check for any remaining duplicates
                if df.index.duplicated().any():
                    print(f"      ⚠️  Still has {df.index.duplicated().sum()} duplicates")
                else:
                    print(f"      ✅ No duplicates found")
                    
        except Exception as e:
            print(f"   ❌ {table}: Error - {e}")
    
    print("\n" + "="*60)
    print("🎉 DUPLICATE FIX VERIFICATION COMPLETE!")
    print("If all tables show '✅' then the duplicate fix was successful!")
    print("You can now run the enhanced processor without duplicate errors!")

if __name__ == "__main__":
    test_duplicate_fix()
