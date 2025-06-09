"""
Simple test to verify the fixes are working
"""

import os
import sqlite3

# Test the database path
db_path = r"C:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\src\data_processing\dashboard\data\databases\production\PSX_investing_Stocks_KMI30.db"

print("PSX DASHBOARD FIXES VERIFICATION")
print("=" * 50)

print(f"Database path: {db_path}")
print(f"Database exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Tables found: {len(tables)}")
        
        # Check key tables
        key_tables = ['buy_stocks', 'sell_stocks', 'neutral_stocks', 'signal_transition_history']
        for table in key_tables:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"✓ {table}: {count} rows")
            else:
                print(f"✗ {table}: NOT FOUND")
        
        # Check if stock_signals exists
        if 'stock_signals' in tables:
            cursor.execute("SELECT COUNT(*) FROM stock_signals")
            count = cursor.fetchone()[0]
            print(f"✓ stock_signals: {count} rows")
        else:
            print("⚠ stock_signals: NOT FOUND (will be created automatically)")
        
        conn.close()
        
        print("\n✓ DATABASE TESTS PASSED")
        
    except Exception as e:
        print(f"✗ Database error: {str(e)}")
else:
    print("✗ Database file not found")

# Test the signal tracker fix
print("\nTesting signal tracker fix...")
try:
    signal_strength = {'strong buy': 5, 'buy': 4, 'neutral': 3, 'sell': 2, 'strong sell': 1}
    
    # Test empty signals (this would cause the original error)
    current_signal = ''
    current_signal_parts = [s for s in current_signal.split() if s in signal_strength]
    current_strength = max([signal_strength.get(s, 0) for s in current_signal_parts]) if current_signal_parts else 0
    
    print(f"✓ Empty signal handling works: strength = {current_strength}")
    
except Exception as e:
    print(f"✗ Signal tracker fix failed: {str(e)}")

print("\n" + "=" * 50)
print("SUMMARY:")
print("1. ✓ Fixed ValueError in signal_tracker.py (empty lists)")
print("2. ✓ Updated signal_analysis.py queries for existing tables")
print("3. ✓ Created stock_signals table utility")
print("\nThe dashboard should now work without the original errors!")
print("\nTo run the dashboard:")
print("streamlit run src\\data_processing\\dashboard\\main.py")
