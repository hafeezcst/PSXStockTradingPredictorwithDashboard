from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from .config import PSXConfig

class DatabaseManager:
    """Manages database operations for PSX announcements"""
    
    def __init__(self):
        """Initialize database connection with connection pooling"""
        self.engine = create_engine(
            PSXConfig.get_db_url(),
            pool_size=PSXConfig.DB_POOL_CONFIG['pool_size'],
            max_overflow=PSXConfig.DB_POOL_CONFIG['max_overflow'],
            pool_timeout=PSXConfig.DB_POOL_CONFIG['pool_timeout'],
            pool_recycle=PSXConfig.DB_POOL_CONFIG['pool_recycle']
        )
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
    
    def save_announcements(self, announcements: List[Dict], table_name: str) -> bool:
        """
        Save announcements to database table
        
        Args:
            announcements: List of announcement dictionaries
            table_name: Name of the table to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not announcements:
            logging.info(f"No announcements to save for table {table_name}")
            return True
            
        try:
            df = pd.DataFrame(announcements)
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logging.info(f"Saved {len(announcements)} announcements to {table_name}")
            return True
        except SQLAlchemyError as e:
            logging.error(f"Error saving announcements to {table_name}: {e}")
            return False
    
    def verify_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """
        Verify data exists in database for given symbol and date range
        
        Args:
            symbol: Company symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            bool: True if data exists, False otherwise
        """
        table_name = f'PSX_{symbol}_announcements'
        inspector = inspect(self.engine)
        
        if table_name not in inspector.get_table_names():
            return False
            
        try:
            with self.engine.connect().execution_options(statement_timeout=60) as conn:
                query = text(f"SELECT COUNT(*) FROM {table_name} WHERE Date BETWEEN :start_date AND :end_date")
                result = conn.execute(query, {'start_date': start_date, 'end_date': end_date}).scalar()
                return result > 0
        except SQLAlchemyError as e:
            logging.error(f"Error verifying data for {symbol}: {e}")
            return False
    
    def check_database_integrity(self) -> bool:
        """
        Perform basic integrity checks on the database
        
        Returns:
            bool: True if integrity check passes, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA integrity_check;"))
            logging.info("Database passed the integrity check")
            return True
        except SQLAlchemyError as e:
            logging.error(f"Database integrity check failed: {e}")
            return False
    
    def get_table_schema(self, table_name: str) -> Optional[List[str]]:
        """
        Get column names for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Optional[List[str]]: List of column names if table exists, None otherwise
        """
        inspector = inspect(self.engine)
        if table_name not in inspector.get_table_names():
            return None
            
        return [col['name'] for col in inspector.get_columns(table_name)]
    
    def recreate_table(self, table_name: str, backup: bool = True) -> bool:
        """
        Recreate a table with updated schema
        
        Args:
            table_name: Name of the table to recreate
            backup: Whether to create a backup of the existing table
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if backup:
                backup_table_name = f"{table_name}_backup"
                with self.engine.connect() as conn:
                    conn.execute(text(f"CREATE TABLE IF NOT EXISTS {backup_table_name} AS SELECT * FROM {table_name}"))
                    conn.execute(text(f"DROP TABLE {table_name}"))
            else:
                with self.engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    
            logging.info(f"Successfully recreated table {table_name}")
            return True
        except SQLAlchemyError as e:
            logging.error(f"Error recreating table {table_name}: {e}")
            return False 