"""
Enhanced database management utilities for PSX data operations.
"""

import os
import shutil
import logging
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from sqlalchemy import create_engine, MetaData, Table, inspect, text, Engine
from sqlalchemy.exc import SQLAlchemyError
from exceptions import DatabaseConnectionError
from config_manager import DatabaseConfig

class StockStatus:
    """Enum for stock status types"""
    ACTIVE = "ACTIVE"
    DELISTED = "DELISTED"
    MERGED = "MERGED"
    RENAMED = "RENAMED"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"

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
        
        # Initialize stock status tracking
        self._initialize_stock_tracking_tables()
        
        # Stock status cache for performance
        self._stock_status_cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(hours=24)  # Cache for 24 hours

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
    
    def _initialize_stock_tracking_tables(self):
        """Initialize tables for tracking stock status and changes"""
        try:
            for engine in [self.engine, self.alt_engine]:
                with engine.connect() as conn:
                    # Stock status tracking table
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS stock_status_tracking (
                            symbol TEXT PRIMARY KEY,
                            status TEXT NOT NULL,
                            last_trading_date DATE,
                            reason TEXT,
                            merged_into TEXT,
                            renamed_to TEXT,
                            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_date DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    
                    # Historical symbol changes table
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS symbol_changes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            old_symbol TEXT NOT NULL,
                            new_symbol TEXT NOT NULL,
                            change_type TEXT NOT NULL,
                            change_date DATE,
                            notes TEXT,
                            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    
                    # Failed download tracking
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS failed_downloads (
                            symbol TEXT,
                            failure_date DATE,
                            failure_count INTEGER DEFAULT 1,
                            last_error TEXT,
                            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (symbol, failure_date)
                        )
                    """))
                    
                    conn.commit()
                    
            self.logger.info("Stock tracking tables initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize stock tracking tables: {e}")

    def update_stock_status(self, symbol: str, status: str, **kwargs):
        """Update stock status in tracking table"""
        try:
            with self.engine.connect() as conn:
                # Check if record exists
                result = conn.execute(
                    text("SELECT symbol FROM stock_status_tracking WHERE symbol = :symbol"),
                    {"symbol": symbol}
                ).fetchone()
                
                if result:
                    # Update existing record
                    update_query = """
                        UPDATE stock_status_tracking 
                        SET status = :status, updated_date = CURRENT_TIMESTAMP
                    """
                    params = {"symbol": symbol, "status": status}
                    
                    # Add optional fields
                    if kwargs.get('last_trading_date'):
                        update_query += ", last_trading_date = :last_trading_date"
                        params['last_trading_date'] = kwargs['last_trading_date']
                    if kwargs.get('reason'):
                        update_query += ", reason = :reason"
                        params['reason'] = kwargs['reason']
                    if kwargs.get('merged_into'):
                        update_query += ", merged_into = :merged_into"
                        params['merged_into'] = kwargs['merged_into']
                    if kwargs.get('renamed_to'):
                        update_query += ", renamed_to = :renamed_to"
                        params['renamed_to'] = kwargs['renamed_to']
                    
                    update_query += " WHERE symbol = :symbol"
                    conn.execute(text(update_query), params)
                else:
                    # Insert new record
                    conn.execute(text("""
                        INSERT INTO stock_status_tracking 
                        (symbol, status, last_trading_date, reason, merged_into, renamed_to)
                        VALUES (:symbol, :status, :last_trading_date, :reason, :merged_into, :renamed_to)
                    """), {
                        "symbol": symbol,
                        "status": status,
                        "last_trading_date": kwargs.get('last_trading_date'),
                        "reason": kwargs.get('reason'),
                        "merged_into": kwargs.get('merged_into'),
                        "renamed_to": kwargs.get('renamed_to')
                    })
                
                conn.commit()
                # Clear cache for this symbol
                if symbol in self._stock_status_cache:
                    del self._stock_status_cache[symbol]
                    del self._cache_expiry[symbol]
                
                self.logger.info(f"Updated stock status for {symbol}: {status}")
                
        except Exception as e:
            self.logger.error(f"Failed to update stock status for {symbol}: {e}")

    def get_stock_status(self, symbol: str) -> Dict:
        """Get stock status with caching"""
        # Check cache first
        if (symbol in self._stock_status_cache and 
            symbol in self._cache_expiry and 
            datetime.now() < self._cache_expiry[symbol]):
            return self._stock_status_cache[symbol]
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT * FROM stock_status_tracking WHERE symbol = :symbol"),
                    {"symbol": symbol}
                ).fetchone()
                
                if result:
                    status_info = {
                        'symbol': result[0],
                        'status': result[1],
                        'last_trading_date': result[2],
                        'reason': result[3],
                        'merged_into': result[4],
                        'renamed_to': result[5]
                    }
                else:
                    status_info = {
                        'symbol': symbol,
                        'status': StockStatus.UNKNOWN,
                        'last_trading_date': None,
                        'reason': None,
                        'merged_into': None,
                        'renamed_to': None
                    }
                
                # Cache the result
                self._stock_status_cache[symbol] = status_info
                self._cache_expiry[symbol] = datetime.now() + self._cache_duration
                
                return status_info
                
        except Exception as e:
            self.logger.error(f"Failed to get stock status for {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': StockStatus.UNKNOWN,
                'last_trading_date': None,
                'reason': f"Error: {e}",
                'merged_into': None,
                'renamed_to': None
            }

    def track_failed_download(self, symbol: str, error_message: str):
        """Track failed downloads with retry logic"""
        try:
            today = date.today()
            with self.engine.connect() as conn:
                # Check existing failures for today
                result = conn.execute(
                    text("SELECT failure_count FROM failed_downloads WHERE symbol = :symbol AND failure_date = :date"),
                    {"symbol": symbol, "date": today}
                ).fetchone()
                
                if result:
                    # Increment failure count
                    new_count = result[0] + 1
                    conn.execute(text("""
                        UPDATE failed_downloads 
                        SET failure_count = :count, last_error = :error
                        WHERE symbol = :symbol AND failure_date = :date
                    """), {
                        "count": new_count,
                        "error": error_message,
                        "symbol": symbol,
                        "date": today
                    })
                else:
                    # Insert new failure record
                    conn.execute(text("""
                        INSERT INTO failed_downloads (symbol, failure_date, failure_count, last_error)
                        VALUES (:symbol, :date, 1, :error)
                    """), {
                        "symbol": symbol,
                        "date": today,
                        "error": error_message
                    })
                
                conn.commit()
                
                # Check if we should mark as problematic
                failure_count = new_count if result else 1
                if failure_count >= 3:  # After 3 failures, investigate
                    self._investigate_stock_issues(symbol, failure_count, error_message)
                
        except Exception as e:
            self.logger.error(f"Failed to track download failure for {symbol}: {e}")

    def _investigate_stock_issues(self, symbol: str, failure_count: int, last_error: str):
        """Investigate stock issues and suggest status changes"""
        self.logger.warning(f"Stock {symbol} has failed {failure_count} times. Last error: {last_error}")
        
        # Pattern-based detection of common issues
        error_lower = last_error.lower()
        
        if any(keyword in error_lower for keyword in ['not found', '404', 'does not exist']):
            self.logger.warning(f"Stock {symbol} appears to be delisted or renamed")
            self.update_stock_status(
                symbol, 
                StockStatus.DELISTED, 
                reason=f"Repeated 404 errors after {failure_count} attempts",
                last_trading_date=date.today() - timedelta(days=1)
            )
        elif any(keyword in error_lower for keyword in ['suspended', 'trading halt']):
            self.logger.warning(f"Stock {symbol} appears to be suspended")
            self.update_stock_status(
                symbol, 
                StockStatus.SUSPENDED, 
                reason=f"Trading suspension detected after {failure_count} attempts"
            )
        elif any(keyword in error_lower for keyword in ['merged', 'acquisition']):
            self.logger.warning(f"Stock {symbol} appears to be merged")
            self.update_stock_status(
                symbol, 
                StockStatus.MERGED, 
                reason=f"Merger/acquisition detected after {failure_count} attempts"
            )

    def should_skip_download(self, symbol: str) -> Tuple[bool, str]:
        """Determine if a symbol should be skipped for download"""
        status_info = self.get_stock_status(symbol)
          # Skip if marked as delisted, merged, or suspended
        if status_info['status'] in [StockStatus.DELISTED, StockStatus.MERGED]:
            return True, f"Stock is {status_info['status'].lower()}: {status_info.get('reason', 'No reason provided')}"
        
        # Check recent failures
        try:
            with self.engine.connect() as conn:
                # Check failures in last 7 days
                week_ago = date.today() - timedelta(days=7)
                result = conn.execute(text("""
                    SELECT SUM(failure_count) as total_failures 
                    FROM failed_downloads 
                    WHERE symbol = :symbol AND failure_date >= :week_ago
                """), {"symbol": symbol, "week_ago": week_ago}).fetchone()
                
                if result and result[0] and result[0] >= 10:  # 10+ failures in a week
                    return True, f"Too many recent failures ({result[0]} in last 7 days)"
        
        except Exception as e:
            self.logger.error(f"Error checking failure history for {symbol}: {e}")
        
        return False, ""

    def get_alternative_symbol(self, symbol: str) -> Optional[str]:
        """Get alternative symbol if stock was renamed or merged"""
        status_info = self.get_stock_status(symbol)
        
        if status_info['status'] == StockStatus.RENAMED and status_info['renamed_to']:
            return status_info['renamed_to']
        elif status_info['status'] == StockStatus.MERGED and status_info['merged_into']:
            return status_info['merged_into']
        
        # Check symbol changes table
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT new_symbol FROM symbol_changes 
                    WHERE old_symbol = :symbol 
                    ORDER BY change_date DESC LIMIT 1
                """), {"symbol": symbol}).fetchone()
                
                if result:
                    return result[0]
        
        except Exception as e:
            self.logger.error(f"Error finding alternative symbol for {symbol}: {e}")
        
        return None

    def record_symbol_change(self, old_symbol: str, new_symbol: str, change_type: str, change_date: date = None, notes: str = None):
        """Record symbol changes (mergers, renames, etc.)"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO symbol_changes (old_symbol, new_symbol, change_type, change_date, notes)
                    VALUES (:old_symbol, :new_symbol, :change_type, :change_date, :notes)
                """), {
                    "old_symbol": old_symbol,
                    "new_symbol": new_symbol,
                    "change_type": change_type,
                    "change_date": change_date or date.today(),
                    "notes": notes
                })
                conn.commit()
                
                # Update stock status
                if change_type.upper() == "RENAME":
                    self.update_stock_status(old_symbol, StockStatus.RENAMED, renamed_to=new_symbol)
                    self.update_stock_status(new_symbol, StockStatus.ACTIVE)
                elif change_type.upper() == "MERGER":
                    self.update_stock_status(old_symbol, StockStatus.MERGED, merged_into=new_symbol)
                
                self.logger.info(f"Recorded symbol change: {old_symbol} -> {new_symbol} ({change_type})")
                
        except Exception as e:
            self.logger.error(f"Failed to record symbol change: {e}")

    def cleanup_old_failure_records(self, days_to_keep: int = 30):
        """Clean up old failure records"""
        try:
            cutoff_date = date.today() - timedelta(days=days_to_keep)
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    DELETE FROM failed_downloads WHERE failure_date < :cutoff_date
                """), {"cutoff_date": cutoff_date})
                
                deleted_count = result.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old failure records")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup failure records: {e}")

    def get_problematic_stocks_report(self) -> Dict:
        """Generate a report of problematic stocks"""
        report = {
            'delisted': [],
            'merged': [],
            'renamed': [],
            'suspended': [],
            'frequent_failures': []
        }
        
        try:
            with self.engine.connect() as conn:
                # Get stocks by status
                for status_key, status_value in [
                    ('delisted', StockStatus.DELISTED),
                    ('merged', StockStatus.MERGED), 
                    ('renamed', StockStatus.RENAMED),
                    ('suspended', StockStatus.SUSPENDED)
                ]:
                    result = conn.execute(text("""
                        SELECT symbol, reason, last_trading_date, merged_into, renamed_to
                        FROM stock_status_tracking 
                        WHERE status = :status
                    """), {"status": status_value}).fetchall()
                    
                    report[status_key] = [
                        {
                            'symbol': row[0],
                            'reason': row[1],
                            'last_trading_date': row[2],
                            'merged_into': row[3],
                            'renamed_to': row[4]
                        } for row in result
                    ]
                
                # Get frequently failing stocks
                week_ago = date.today() - timedelta(days=7)
                result = conn.execute(text("""
                    SELECT symbol, SUM(failure_count) as total_failures
                    FROM failed_downloads 
                    WHERE failure_date >= :week_ago
                    GROUP BY symbol
                    HAVING total_failures >= 5
                    ORDER BY total_failures DESC
                """), {"week_ago": week_ago}).fetchall()
                
                report['frequent_failures'] = [
                    {'symbol': row[0], 'failure_count': row[1]} for row in result
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to generate problematic stocks report: {e}")
        
        return report

    def verify_data_exists(self, symbol: str, start_date: date, end_date: date, use_alt: bool = False) -> bool:
        """Enhanced verification that handles stock status issues"""
        # First check if we should skip this symbol
        should_skip, skip_reason = self.should_skip_download(symbol)
        if should_skip:
            self.logger.info(f"Skipping verification for {symbol}: {skip_reason}")
            return True  # Return True to avoid retries for known problematic stocks
        
        # Try alternative symbol if available
        alternative_symbol = self.get_alternative_symbol(symbol)
        if alternative_symbol:
            self.logger.info(f"Using alternative symbol {alternative_symbol} for {symbol}")
            return self._verify_data_exists_internal(alternative_symbol, start_date, end_date, use_alt)
        
        # Standard verification
        return self._verify_data_exists_internal(symbol, start_date, end_date, use_alt)

    def _verify_data_exists_internal(self, symbol: str, start_date: date, end_date: date, use_alt: bool = False) -> bool:
        """Internal verification method"""
        table_name = f'PSX_{symbol}_stock_data'
        engine = self.get_engine(use_alt)
        
        if not self.table_exists(table_name, use_alt):
            self.logger.warning(f"Table {table_name} does not exist")
            return False
        
        try:
            with engine.connect().execution_options(statement_timeout=60) as conn:
                # First, let's check what columns exist in the table
                columns_result = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                self.logger.debug(f"Table {table_name} columns: {[col[1] for col in columns_result]}")
                
                # Check if we have a Date column or if date is in the index
                has_date_column = any(col[1].lower() == 'date' for col in columns_result)
                
                if has_date_column:
                    # Use Date column
                    query = text(f"SELECT COUNT(*) FROM {table_name} WHERE Date BETWEEN :start_date AND :end_date")
                    result = conn.execute(query, {"start_date": start_date, "end_date": end_date}).fetchone()
                else:
                    # Try using rowid or check for date in index - for now, just get total count
                    self.logger.warning(f"No Date column found in {table_name}, checking total records")
                    query = text(f"SELECT COUNT(*) FROM {table_name}")
                    result = conn.execute(query).fetchone()
                
                count = result[0]
                self.logger.debug(f"Verification for {table_name}: found {count} records {'in date range' if has_date_column else 'total'}")
                
                # If no data found and stock status is unknown, mark for investigation
                if count == 0:
                    status_info = self.get_stock_status(symbol)
                    if status_info['status'] == StockStatus.UNKNOWN:
                        self.track_failed_download(symbol, "No data found during verification")
                
                return count > 0
                
        except Exception as e:
            self.logger.error(f"Database verification error for {table_name}: {e}")
            self.track_failed_download(symbol, str(e))
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
