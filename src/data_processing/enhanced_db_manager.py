"""
Enhanced database management utilities for PSX data operations.
"""

import os
import shutil
import logging
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List
from sqlalchemy import create_engine, MetaData, Table, inspect, text, Engine
from sqlalchemy.exc import SQLAlchemyError
from .exceptions import DatabaseConnectionError
from .config_manager import DatabaseConfig

class EnhancedDatabaseManager:
    """Enhanced database manager with improved functionality"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metadata = MetaData()
        
        # Initialize engines
        self.engine = self._create_engine(config.main_db_path)
        self.alt_engine = self._create_engine(config.alt_db_path)
        
        # Ensure database directories exist
        self._ensure_db_directories()
    
    def _create_engine(self, db_path: str) -> Engine:
        """Create SQLAlchemy engine with connection pooling"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            engine = create_engine(
                f'sqlite:///{db_path}',
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info(f"Database engine created successfully: {db_path}")
            return engine
            
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to create database engine: {e}", db_path)
    
    def _ensure_db_directories(self):
        """Ensure all database directories exist"""
        for db_path in [self.config.main_db_path, self.config.alt_db_path]:
            dir_path = os.path.dirname(db_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                self.logger.info(f"Created database directory: {dir_path}")
    
    def check_connection(self, use_alt: bool = False) -> bool:
        """Check database connection health"""
        engine = self.alt_engine if use_alt else self.engine
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            db_path = self.config.alt_db_path if use_alt else self.config.main_db_path
            self.logger.error(f"Database connection check failed for {db_path}: {e}")
            return False
    
    def get_engine(self, use_alt: bool = False) -> Engine:
        """Get the appropriate database engine"""
        return self.alt_engine if use_alt else self.engine
    
    def table_exists(self, table_name: str, use_alt: bool = False) -> bool:
        """Check if a table exists in the database"""
        engine = self.get_engine(use_alt)
        inspector = inspect(engine)
        return table_name in inspector.get_table_names()
    
    def get_table_names(self, use_alt: bool = False) -> List[str]:
        """Get list of all table names"""
        engine = self.get_engine(use_alt)
        inspector = inspect(engine)
        return inspector.get_table_names()
    
    def get_last_date(self, symbol: str, use_alt: bool = False) -> Optional[date]:
        """Get the last date for a symbol's data"""
        table_name = f'PSX_{symbol}_stock_data'
        engine = self.get_engine(use_alt)
        
        if not self.table_exists(table_name, use_alt):
            return None
        
        try:
            table = Table(table_name, self.metadata, autoload_with=engine)
            query = table.select().order_by(table.c.Date.desc()).limit(1)
            
            with engine.connect() as conn:
                result = conn.execute(query).fetchone()
                if result:
                    import pandas as pd
                    return pd.to_datetime(result[0]).date()
        except Exception as e:
            self.logger.error(f"Error getting last date for {symbol}: {e}")
        
        return None
    
    def verify_data_exists(self, symbol: str, start_date: date, end_date: date, use_alt: bool = False) -> bool:
        """Verify that data exists for a symbol in the given date range"""
        table_name = f'PSX_{symbol}_stock_data'
        engine = self.get_engine(use_alt)
        
        if not self.table_exists(table_name, use_alt):
            return False
        
        try:
            with engine.connect().execution_options(statement_timeout=60) as conn:
                query = text(f"SELECT COUNT(*) FROM {table_name} WHERE Date BETWEEN :start_date AND :end_date")
                result = conn.execute(query, {"start_date": start_date, "end_date": end_date}).fetchone()
                return result[0] > 0
        except Exception as e:
            self.logger.error(f"Database verification timeout for {table_name}: {e}")
            return False
    
    def save_data(self, data, table_name: str, use_alt: bool = False):
        """Save DataFrame to database"""
        if data.empty:
            self.logger.info(f"No data to save for table {table_name}")
            return
        
        engine = self.get_engine(use_alt)
        
        try:
            data.to_sql(table_name, engine, if_exists='append', index=True, method='multi')
            self.logger.debug(f"Successfully saved {len(data)} records to {table_name}")
        except Exception as e:
            self.logger.error(f"Failed to save data to {table_name}: {e}")
            raise DatabaseConnectionError(f"Failed to save data: {e}")
    
    def delete_table(self, table_name: str, use_alt: bool = False):
        """Delete a table from the database"""
        engine = self.get_engine(use_alt)
        
        try:
            with engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()
            self.logger.info(f"Table {table_name} deleted successfully")
        except Exception as e:
            self.logger.error(f"Failed to delete table {table_name}: {e}")
            raise DatabaseConnectionError(f"Failed to delete table: {e}")
    
    def delete_unused_tables(self, valid_symbols: List[str], use_alt: bool = False):
        """Delete tables that don't correspond to valid symbols"""
        valid_table_names = {f"PSX_{symbol}_stock_data" for symbol in valid_symbols}
        existing_tables = self.get_table_names(use_alt)
        
        deleted_count = 0
        for table_name in existing_tables:
            if table_name.startswith("PSX_") and table_name.endswith("_stock_data"):
                if table_name not in valid_table_names:
                    self.delete_table(table_name, use_alt)
                    deleted_count += 1
        
        if deleted_count > 0:
            self.logger.info(f"Deleted {deleted_count} unused tables")
    
    def check_integrity(self, use_alt: bool = False) -> bool:
        """Perform database integrity check"""
        engine = self.get_engine(use_alt)
        db_path = self.config.alt_db_path if use_alt else self.config.main_db_path
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text("PRAGMA integrity_check")).fetchone()
                if result[0] == "ok":
                    self.logger.info(f"Database integrity check passed: {db_path}")
                    return True
                else:
                    self.logger.error(f"Database integrity check failed: {result[0]}")
                    return False
        except Exception as e:
            self.logger.error(f"Database integrity check failed for {db_path}: {e}")
            return False
    
    def optimize_database(self, use_alt: bool = False):
        """Optimize database performance"""
        engine = self.get_engine(use_alt)
        db_path = self.config.alt_db_path if use_alt else self.config.main_db_path
        
        try:
            with engine.connect() as conn:
                # Analyze tables for query optimization
                conn.execute(text("ANALYZE"))
                
                # Vacuum to reclaim space
                conn.execute(text("VACUUM"))
                
                # Update statistics
                conn.execute(text("PRAGMA optimize"))
                
            self.logger.info(f"Database optimization completed: {db_path}")
        except Exception as e:
            self.logger.error(f"Database optimization failed for {db_path}: {e}")
    
    def backup_database(self, backup_path: Optional[str] = None, use_alt: bool = False) -> str:
        """Create a backup of the database"""
        source_db = self.config.alt_db_path if use_alt else self.config.main_db_path
        
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_name = Path(source_db).stem
            backup_dir = Path(source_db).parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f"{db_name}_backup_{timestamp}.db"
        
        try:
            shutil.copy2(source_db, backup_path)
            self.logger.info(f"Database backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            raise DatabaseConnectionError(f"Backup failed: {e}")
    
    def get_database_stats(self, use_alt: bool = False) -> dict:
        """Get database statistics"""
        engine = self.get_engine(use_alt)
        db_path = self.config.alt_db_path if use_alt else self.config.main_db_path
        
        stats = {
            'database_path': db_path,
            'file_size_mb': 0,
            'table_count': 0,
            'total_records': 0
        }
        
        try:
            # File size
            if os.path.exists(db_path):
                stats['file_size_mb'] = os.path.getsize(db_path) / (1024 * 1024)
            
            # Table count and record count
            tables = self.get_table_names(use_alt)
            stats['table_count'] = len(tables)
            
            with engine.connect() as conn:
                total_records = 0
                for table_name in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
                        total_records += result[0]
                    except:
                        pass  # Skip tables with issues
                stats['total_records'] = total_records
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
        
        return stats
    
    def close_connections(self):
        """Close all database connections"""
        try:
            self.engine.dispose()
            self.alt_engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
