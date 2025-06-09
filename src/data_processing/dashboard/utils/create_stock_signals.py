# Create stock_signals table for compatibility

import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def create_stock_signals_table(db_path):
    """
    Create a stock_signals table for compatibility with the signal_analysis component.
    This table combines data from buy_stocks, sell_stocks, and neutral_stocks tables.
    
    Args:
        db_path: Path to the SQLite database
    """
    print(f"Creating stock_signals table in database: {db_path}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('buy_stocks', 'sell_stocks', 'neutral_stocks')")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'buy_stocks' not in tables or 'sell_stocks' not in tables or 'neutral_stocks' not in tables:
            print("Required tables not found. Make sure buy_stocks, sell_stocks, and neutral_stocks tables exist.")
            return False
        
        # Check if the table already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_signals'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            print("stock_signals table already exists. Dropping and recreating...")
            cursor.execute("DROP TABLE stock_signals")
            conn.commit()
        
        # Create stock_signals table
        cursor.execute("""
        CREATE TABLE stock_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date TEXT,
            signal TEXT,
            close REAL,
            volume REAL,
            rsi_weekly REAL,
            rsi_3months REAL,
            ao_weekly REAL,
            ma_30 REAL,
            confidence_score REAL,
            technical_score REAL,
            multibagger TEXT,
            free_float_ratio TEXT,
            success TEXT,
            profit_loss REAL,
            signal_date TEXT,
            signal_close REAL,
            holding_days INTEGER,
            status TEXT,
            analysis_summary TEXT,
            created_at TEXT
        )
        """)
        conn.commit()
        
        # Populate the table with data from buy_stocks
        cursor.execute("""
        INSERT INTO stock_signals (
            symbol, date, signal, close, volume, rsi_weekly, rsi_3months, ao_weekly, ma_30,
            confidence_score, technical_score, multibagger, free_float_ratio, success,
            profit_loss, signal_date, signal_close, holding_days, status, created_at
        )
        SELECT 
            Stock, Date, 'BUY', Close, Volume, RSI_Weekly_Avg, RSI_3Months_Avg_Recent, 
            AO_Weekly, MA_30, RSI_Weekly_Avg, MA_30, Multibagger, FreeFloatRatio, Success,
            "% P/L", Signal_Date, Signal_Close, Holding_Days, Status, datetime('now')
        FROM buy_stocks
        """)
        conn.commit()
        print(f"Inserted data from buy_stocks table")
        
        # Populate the table with data from sell_stocks
        cursor.execute("""
        INSERT INTO stock_signals (
            symbol, date, signal, close, volume, rsi_weekly, rsi_3months, ao_weekly, ma_30,
            confidence_score, technical_score, multibagger, free_float_ratio, success,
            profit_loss, signal_date, signal_close, holding_days, status, created_at
        )
        SELECT 
            Stock, Date, 'SELL', Close, Volume, RSI_Weekly_Avg, RSI_3Months_Avg_Recent, 
            AO_Weekly, MA_30, RSI_Weekly_Avg, MA_30, Multibagger, FreeFloatRatio, Success,
            "% P/L", Signal_Date, Signal_Close, Holding_Days, Status, datetime('now')
        FROM sell_stocks
        """)
        conn.commit()
        print(f"Inserted data from sell_stocks table")
        
        # Populate the table with data from neutral_stocks
        cursor.execute("""
        INSERT INTO stock_signals (
            symbol, date, signal, close, volume, rsi_weekly, rsi_3months, ao_weekly, ma_30,
            confidence_score, technical_score, multibagger, free_float_ratio, status, created_at
        )
        SELECT 
            Stock, Date, 'NEUTRAL', Close, Volume, RSI_Weekly_Avg, RSI_3Months_Avg_Recent, 
            AO_Weekly, MA_30, RSI_Weekly_Avg, MA_30, Multibagger, FreeFloatRatio, Status, datetime('now')
        FROM neutral_stocks
        """)
        conn.commit()
        print(f"Inserted data from neutral_stocks table")
        
        # Update analysis summary based on signal
        cursor.execute("""
        UPDATE stock_signals
        SET analysis_summary = 
            CASE 
                WHEN signal = 'BUY' THEN 'Bullish signal based on RSI and MA indicators'
                WHEN signal = 'SELL' THEN 'Bearish signal based on RSI and MA indicators'
                WHEN signal = 'NEUTRAL' THEN 'Neutral trend detected based on RSI and MA indicators'
            END
        """)
        conn.commit()
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM stock_signals")
        row_count = cursor.fetchone()[0]
        print(f"Created stock_signals table with {row_count} rows")
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX idx_stock_signals_symbol ON stock_signals(symbol)")
        cursor.execute("CREATE INDEX idx_stock_signals_date ON stock_signals(date)")
        cursor.execute("CREATE INDEX idx_stock_signals_signal ON stock_signals(signal)")
        conn.commit()
        print("Created indexes on stock_signals table")
        
        conn.close()
        print("stock_signals table creation completed.")
        return True
    
    except Exception as e:
        print(f"Error creating stock_signals table: {str(e)}")
        return False

if __name__ == "__main__":
    # Get the database path from command line arguments or use default
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = os.path.join(project_root, "data", "databases", "production", "PSX_investing_Stocks_KMI30.db")
    
    create_stock_signals_table(db_path)
