"""
Database initialization script for PSX Trading Dashboard.
This script creates all necessary tables for signal tracking and analysis.
"""

import sqlite3
import os
from pathlib import Path

def get_db_path():
    """Get the database path at the project root."""
    project_root = Path(__file__).resolve().parents[4]
    db_path = project_root / "data" / "databases" / "production" / "fairvalue.db"
    os.makedirs(db_path.parent, exist_ok=True)
    print(f"Using database path: {db_path}")
    return db_path

def create_tables():
    """Create all required tables in the database."""
    db_path = get_db_path()
    
    # Remove existing database if it exists
    if db_path.exists():
        print(f"Removed existing database at {db_path}")
        db_path.unlink()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tradingview_ta table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tradingview_ta (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        close REAL,
        recommendation TEXT,
        buy_signals INTEGER,
        sell_signals INTEGER,
        neutral_signals INTEGER,
        change_percent REAL,
        volume INTEGER,
        rsi REAL,
        macd REAL,
        macd_signal REAL,
        macd_hist REAL,
        sma_20 REAL,
        sma_50 REAL,
        sma_200 REAL,
        ema_20 REAL,
        ema_50 REAL,
        ema_200 REAL,
        bollinger_upper REAL,
        bollinger_middle REAL,
        bollinger_lower REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create tradingview_signals table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tradingview_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        signal_type TEXT NOT NULL,
        signal_strength REAL,
        price REAL,
        volume INTEGER,
        indicators TEXT,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create stock_signals table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        signal_date DATE NOT NULL,
        price REAL,
        volume INTEGER,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create buy_stocks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS buy_stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        entry_date DATE NOT NULL,
        entry_price REAL NOT NULL,
        target_price REAL,
        stop_loss REAL,
        quantity INTEGER,
        status TEXT DEFAULT 'ACTIVE',
        exit_date DATE,
        exit_price REAL,
        profit_loss REAL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create sell_stocks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sell_stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        entry_date DATE NOT NULL,
        entry_price REAL NOT NULL,
        target_price REAL,
        stop_loss REAL,
        quantity INTEGER,
        status TEXT DEFAULT 'ACTIVE',
        exit_date DATE,
        exit_price REAL,
        profit_loss REAL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create neutral_stocks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS neutral_stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        entry_date DATE NOT NULL,
        entry_price REAL NOT NULL,
        target_price REAL,
        stop_loss REAL,
        quantity INTEGER,
        status TEXT DEFAULT 'ACTIVE',
        exit_date DATE,
        exit_price REAL,
        profit_loss REAL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create signal_transition_history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signal_transition_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Stock TEXT NOT NULL,
        Previous_Signal TEXT NOT NULL,
        Current_Signal TEXT NOT NULL,
        transition_date DATE NOT NULL,
        Previous_Close REAL,
        Current_Close REAL,
        Profit_Loss_Pct REAL,
        Days_In_Signal INTEGER,
        Notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create signal_tracking table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signal_tracking (
        Stock TEXT PRIMARY KEY,
        Signal_Changes INTEGER DEFAULT 0,
        Total_Days INTEGER DEFAULT 0,
        Notes TEXT,
        Last_Updated DATE
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Successfully created all required tables in {db_path}")

if __name__ == "__main__":
    create_tables() 