import sqlite3
import os

# Check the main database we worked with earlier
db1 = r"C:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\src\data\databases\production\PSX_investing_Stocks_KMI30.db"
db2 = r"C:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\src\data_processing\dashboard\data\databases\production\PSX_investing_Stocks_KMI30.db"

for i, db_path in enumerate([db1, db2], 1):
    print(f"\nDatabase {i}: {db_path}")
    print(f"Exists: {os.path.exists(db_path)}")
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"Tables: {tables}")
            
            # Check for our key tables
            key_tables = ['buy_stocks', 'sell_stocks', 'neutral_stocks']
            for table in key_tables:
                if table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  {table}: {count} rows")
            
            conn.close()
        except Exception as e:
            print(f"Error: {e}")
