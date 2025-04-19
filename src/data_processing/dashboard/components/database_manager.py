"""
Database manager component for the PSX dashboard.
"""

import sqlite3
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from pathlib import Path

class DatabaseManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.cursor = None

    def connect(self, db_path: str) -> bool:
        """Connect to SQLite database."""
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            return False

    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def execute_query(self, query: str, params: tuple = None) -> Optional[List[tuple]]:
        """Execute a SQL query and return results."""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            return None

    def execute_many(self, query: str, params_list: List[tuple]) -> bool:
        """Execute multiple SQL queries."""
        try:
            self.cursor.executemany(query, params_list)
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error executing multiple queries: {str(e)}")
            self.conn.rollback()
            return False

    def create_table(self, table_name: str, columns: Dict[str, str]) -> bool:
        """Create a table if it doesn't exist."""
        try:
            columns_str = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
            self.cursor.execute(query)
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error creating table: {str(e)}")
            return False

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        try:
            self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            return bool(self.cursor.fetchone())
        except Exception as e:
            self.logger.error(f"Error checking table existence: {str(e)}")
            return False

    def get_table_columns(self, table_name: str) -> List[str]:
        """Get list of columns in a table."""
        try:
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            return [row[1] for row in self.cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting table columns: {str(e)}")
            return []

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> bool:
        """Insert a pandas DataFrame into a table."""
        try:
            df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
            return True
        except Exception as e:
            self.logger.error(f"Error inserting DataFrame: {str(e)}")
            return False

    def read_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        try:
            if params:
                return pd.read_sql_query(query, self.conn, params=params)
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            self.logger.error(f"Error reading query: {str(e)}")
            return pd.DataFrame()

    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            backup_conn = sqlite3.connect(backup_path)
            self.conn.backup(backup_conn)
            backup_conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Error creating database backup: {str(e)}")
            return False

    def vacuum_database(self) -> bool:
        """Optimize database by removing unused space."""
        try:
            self.conn.execute("VACUUM")
            return True
        except Exception as e:
            self.logger.error(f"Error vacuuming database: {str(e)}")
            return False

    def get_table_size(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return self.cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Error getting table size: {str(e)}")
            return 0

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        try:
            info = {
                'tables': [],
                'total_rows': 0,
                'size_bytes': 0
            }
            
            # Get list of tables
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in self.cursor.fetchall()]
            
            # Get info for each table
            for table in tables:
                row_count = self.get_table_size(table)
                columns = self.get_table_columns(table)
                info['tables'].append({
                    'name': table,
                    'rows': row_count,
                    'columns': columns
                })
                info['total_rows'] += row_count
            
            # Get database file size
            if self.conn:
                db_path = Path(self.conn.path)
                if db_path.exists():
                    info['size_bytes'] = db_path.stat().st_size
            
            return info
        except Exception as e:
            self.logger.error(f"Error getting database info: {str(e)}")
            return {} 