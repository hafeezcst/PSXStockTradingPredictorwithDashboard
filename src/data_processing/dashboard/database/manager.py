"""
Database management utilities for the PSX dashboard.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager
from src.data_processing.dashboard.config.settings import PSX_SIGNALS_DB_PATH

class DatabaseManager:
    """A context manager for database connections."""
    def __init__(self):
        self.db_path = PSX_SIGNALS_DB_PATH
        self.conn = None
        self.cursor = None
        self._create_signal_tables()

    def _create_signal_tables(self):
        """Create signal tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create buy_stocks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS buy_stocks (
                        stock TEXT,
                        date DATE,
                        signal_date DATE,
                        signal_close REAL,
                        confidence_score REAL,
                        PRIMARY KEY (stock, date)
                    )
                """)
                
                # Create sell_stocks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sell_stocks (
                        stock TEXT,
                        date DATE,
                        signal_date DATE,
                        signal_close REAL,
                        confidence_score REAL,
                        PRIMARY KEY (stock, date)
                    )
                """)
                
                # Create neutral_stocks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS neutral_stocks (
                        stock TEXT,
                        date DATE,
                        signal_date DATE,
                        signal_close REAL,
                        confidence_score REAL,
                        PRIMARY KEY (stock, date)
                    )
                """)
                
                conn.commit()
        except Exception as e:
            logging.error(f"Error creating signal tables: {str(e)}")

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

@contextmanager
def get_db_connection(db_path: str):
    """Get a database connection with automatic closing."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        yield conn
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def execute_query(db_path: str, query: str, params: Optional[tuple] = None) -> Any:
    """Execute a database query and return results."""
    try:
        with DatabaseManager() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    except Exception as e:
        logging.error(f"Query execution error: {str(e)}")
        logging.error(f"Query: {query}")
        logging.error(f"Parameters: {params}")
        raise

def table_exists(db_path: str, table_name: str) -> bool:
    """Check if a table exists in the database."""
    query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """
    try:
        with DatabaseManager() as cursor:
            cursor.execute(query, (table_name,))
            return bool(cursor.fetchone())
    except Exception as e:
        logging.error(f"Error checking table existence: {str(e)}")
        return False

def create_signal_tables(db_path: str) -> bool:
    """Create the signal tables if they don't exist."""
    try:
        with DatabaseManager() as cursor:
            # Create buy_stocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS buy_stocks (
                    Stock TEXT,
                    Date DATE,
                    Close REAL,
                    RSI_Weekly_Avg REAL,
                    AO_Weekly REAL,
                    Signal_Date DATE,
                    Signal_Close REAL,
                    update_date DATE,
                    PRIMARY KEY (Stock, Date)
                )
            """)
            
            # Create sell_stocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sell_stocks (
                    Stock TEXT,
                    Date DATE,
                    Close REAL,
                    RSI_Weekly_Avg REAL,
                    AO_Weekly REAL,
                    Signal_Date DATE,
                    Signal_Close REAL,
                    update_date DATE,
                    PRIMARY KEY (Stock, Date)
                )
            """)
            
            # Create neutral_stocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS neutral_stocks (
                    Stock TEXT,
                    Date DATE,
                    Close REAL,
                    RSI_Weekly_Avg REAL,
                    AO_Weekly REAL,
                    Signal_Date DATE,
                    Signal_Close REAL,
                    update_date DATE,
                    PRIMARY KEY (Stock, Date)
                )
            """)
            
            return True
    except Exception as e:
        logging.error(f"Error creating signal tables: {str(e)}")
        return False 