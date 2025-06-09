"""
Database population script for PSX Trading Dashboard.
This script populates the database with initial data for testing.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

def get_db_path():
    """Get the database path at the project root."""
    project_root = Path(__file__).resolve().parents[4]
    db_path = project_root / "data" / "databases" / "production" / "fairvalue.db"
    os.makedirs(db_path.parent, exist_ok=True)
    print(f"Using database path: {db_path}")
    return db_path

def populate_database():
    """Populate database with initial data"""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Sample stock symbols
        symbols = ['OGDC', 'PPL', 'HBL', 'UBL', 'LUCK', 'ENGRO', 'MCB', 'FFC', 'PSO', 'POL']
        
        # Get current date
        current_date = datetime.now().date()
        
        # Populate tradingview_ta table
        print("Populating tradingview_ta table...")
        for symbol in symbols:
            # Generate some random technical analysis data
            cursor.execute("""
                INSERT INTO tradingview_ta (
                    symbol, date, close, recommendation,
                    buy_signals, sell_signals, neutral_signals,
                    change_percent, volume, rsi, macd, macd_signal,
                    macd_hist, sma_20, sma_50, sma_200,
                    ema_20, ema_50, ema_200,
                    bollinger_upper, bollinger_middle, bollinger_lower
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                current_date.strftime('%Y-%m-%d'),
                100.0 + (hash(symbol) % 100),  # Random price between 100-200
                'BUY' if hash(symbol) % 3 == 0 else 'SELL' if hash(symbol) % 3 == 1 else 'NEUTRAL',
                hash(symbol) % 5,  # Random number of buy signals
                hash(symbol) % 3,  # Random number of sell signals
                hash(symbol) % 4,  # Random number of neutral signals
                (hash(symbol) % 20) - 10,  # Random change percent between -10 and 10
                1000000 + (hash(symbol) % 1000000),  # Random volume
                30 + (hash(symbol) % 40),  # Random RSI between 30-70
                0.5 + (hash(symbol) % 10) / 10,  # Random MACD
                0.3 + (hash(symbol) % 10) / 10,  # Random MACD signal
                0.2 + (hash(symbol) % 10) / 10,  # Random MACD hist
                95 + (hash(symbol) % 10),  # Random SMA20
                90 + (hash(symbol) % 15),  # Random SMA50
                85 + (hash(symbol) % 20),  # Random SMA200
                96 + (hash(symbol) % 10),  # Random EMA20
                91 + (hash(symbol) % 15),  # Random EMA50
                86 + (hash(symbol) % 20),  # Random EMA200
                105 + (hash(symbol) % 10),  # Random Bollinger upper
                100 + (hash(symbol) % 5),   # Random Bollinger middle
                95 + (hash(symbol) % 10)    # Random Bollinger lower
            ))
        
        # Populate tradingview_signals table
        print("Populating tradingview_signals table...")
        for symbol in symbols:
            signal_type = 'BUY' if hash(symbol) % 3 == 0 else 'SELL' if hash(symbol) % 3 == 1 else 'NEUTRAL'
            indicators = {
                'rsi': 30 + (hash(symbol) % 40),
                'macd': 0.5 + (hash(symbol) % 10) / 10,
                'sma': 95 + (hash(symbol) % 10)
            }
            
            cursor.execute("""
                INSERT INTO tradingview_signals (
                    symbol, date, signal_type, signal_strength,
                    price, volume, indicators, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                current_date.strftime('%Y-%m-%d'),
                signal_type,
                0.5 + (hash(symbol) % 10) / 10,  # Random signal strength
                100.0 + (hash(symbol) % 100),    # Random price
                1000000 + (hash(symbol) % 1000000),  # Random volume
                json.dumps(indicators),
                f"Initial signal for {symbol}"
            ))
        
        # Populate signal_tracking table
        print("Populating signal_tracking table...")
        for symbol in symbols:
            cursor.execute("""
                INSERT INTO signal_tracking (
                    Stock, Current_Signal, Signal_Changes,
                    Total_Days, Notes, Last_Updated
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                'BUY' if hash(symbol) % 3 == 0 else 'SELL' if hash(symbol) % 3 == 1 else 'NEUTRAL',
                hash(symbol) % 10,  # Random number of signal changes
                hash(symbol) % 30,  # Random number of days
                f"Tracking {symbol}",
                current_date.strftime('%Y-%m-%d')
            ))
        
        conn.commit()
        print("Successfully populated database with initial data")
        return True
        
    except Exception as e:
        print(f"Error populating database: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    populate_database() 