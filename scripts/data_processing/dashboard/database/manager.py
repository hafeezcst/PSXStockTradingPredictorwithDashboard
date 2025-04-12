"""
Database management utilities for the PSX dashboard.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager

class DatabaseManager:
    """A context manager for database connections."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

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
        with DatabaseManager(db_path) as cursor:
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
        with DatabaseManager(db_path) as cursor:
            cursor.execute(query, (table_name,))
            return bool(cursor.fetchone())
    except Exception as e:
        logging.error(f"Error checking table existence: {str(e)}")
        return False 