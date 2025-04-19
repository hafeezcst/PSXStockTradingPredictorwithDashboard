import sqlite3
import pandas as pd
from datetime import datetime
import os

# Get the absolute path to the database
db_path = "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_investing_Stocks_KMI30.db"

# Check if database exists
if not os.path.exists(db_path):
    print(f"Database not found at: {db_path}")
    exit(1)

# Connect to the database
conn = sqlite3.connect(db_path)

# Query for specific stocks
stocks = ['CRTM', 'ANL', 'INKL']
stocks_str = ','.join([f"'{stock}'" for stock in stocks])

query = f"""
SELECT 
    Stock,
    Date,
    Signal_Date,
    Signal_Close,
    Close,
    Holding_Days,
    "% P/L",
    Success,
    Status,
    Update_Date
FROM buy_stocks 
WHERE Stock IN ({stocks_str})
ORDER BY Signal_Date DESC
"""

try:
    # Execute query and convert to DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Convert date columns to datetime
    date_columns = ['Date', 'Signal_Date', 'Update_Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Calculate holding days if not provided
    if 'Holding_Days' not in df.columns:
        df['Holding_Days'] = (datetime.now() - df['Signal_Date']).dt.days
    
    # Calculate profit/loss if not provided
    if '% P/L' not in df.columns and 'Signal_Close' in df.columns and 'Close' in df.columns:
        df['% P/L'] = ((df['Close'] - df['Signal_Close']) / df['Signal_Close']) * 100
    
    if df.empty:
        print(f"\nNo buy signals found for stocks: {', '.join(stocks)}")
    else:
        # Format the output
        print("\nLatest Buy Signals:")
        print("=" * 80)
        for _, row in df.iterrows():
            print(f"\nStock: {row['Stock']}")
            print(f"Signal Date: {row['Signal_Date'].strftime('%Y-%m-%d')}")
            print(f"Signal Price: Rs. {row['Signal_Close']:.2f}")
            print(f"Current Price: Rs. {row['Close']:.2f}")
            print(f"Holding Days: {row['Holding_Days']}")
            print(f"Profit/Loss: {row['% P/L']:.2f}%")
            print(f"Status: {row['Status']}")
            print("-" * 40)

except Exception as e:
    print(f"Error querying database: {str(e)}")

finally:
    # Close the connection
    conn.close() 