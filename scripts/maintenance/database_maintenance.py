"""
Database Maintenance Script for PSX Stock Trading Predictor

This script performs regular maintenance tasks to ensure database:
- Performance optimization
- Data integrity
- Space efficiency
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='database_maintenance.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DatabaseMaintenance:
    def __init__(self, db_paths):
        """
        Initialize with list of database paths to maintain
        
        Args:
            db_paths: List of paths to SQLite database files
        """
        self.db_paths = db_paths
        self.current_dir = os.getcwd()
        
    def create_backup(self, db_path):
        """Create timestamped backup of database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.bak_{timestamp}"
        try:
            with sqlite3.connect(db_path) as src:
                with sqlite3.connect(backup_path) as dst:
                    src.backup(dst)
            logging.info(f"Created backup of {db_path} at {backup_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to backup {db_path}: {str(e)}")
            return False
            
    def vacuum_database(self, db_path):
        """Perform VACUUM operation to optimize database"""
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("VACUUM")
            logging.info(f"Successfully vacuumed {db_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to vacuum {db_path}: {str(e)}")
            return False
            
    def integrity_check(self, db_path):
        """Perform integrity check on database"""
        try:
            with sqlite3.connect(db_path) as conn:
                result = conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] == "ok":
                logging.info(f"Integrity check passed for {db_path}")
                return True
            else:
                logging.error(f"Integrity check failed for {db_path}: {result[0]}")
                return False
        except Exception as e:
            logging.error(f"Failed integrity check for {db_path}: {str(e)}")
            return False
            
    def optimize_indexes(self, db_path):
        """Create or optimize indexes for better query performance"""
        try:
            with sqlite3.connect(db_path) as conn:
                # Get list of tables
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
                
                for table in tables['name']:
                    # Skip sqlite_ system tables
                    if table.startswith('sqlite_'):
                        continue
                        
                    # Check if Date index exists
                    index_check = pd.read_sql(
                        f"PRAGMA index_list({table})", 
                        conn
                    )
                    
                    # Create index if it doesn't exist
                    if not any(index_check['name'].str.contains('idx_date')):
                        conn.execute(f"CREATE INDEX idx_{table}_date ON {table}(Date)")
                        logging.info(f"Created Date index on {table}")
                        
                    # Create index on Symbol if table name contains symbol
                    if 'PSX_' in table:
                        symbol = table.split('_')[1]
                        if not any(index_check['name'].str.contains('idx_symbol')):
                            conn.execute(f"CREATE INDEX idx_{table}_symbol ON {table}(Symbol)")
                            logging.info(f"Created Symbol index on {table}")
                            
            return True
        except Exception as e:
            logging.error(f"Failed to optimize indexes for {db_path}: {str(e)}")
            return False
            
    def analyze_database(self, db_path):
        """Analyze database for query optimization"""
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("ANALYZE")
            logging.info(f"Successfully analyzed {db_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to analyze {db_path}: {str(e)}")
            return False
            
    def run_maintenance(self):
        """Run complete maintenance routine on all databases"""
        results = []
        for db_path in self.db_paths:
            db_results = {
                'database': db_path,
                'backup': False,
                'vacuum': False,
                'integrity': False,
                'indexes': False,
                'analyze': False
            }
            
            # Create full path if not absolute
            if not os.path.isabs(db_path):
                db_path = os.path.join(self.current_dir, db_path)
                
            if not os.path.exists(db_path):
                logging.warning(f"Database file not found: {db_path}")
                continue
                
            # Perform maintenance tasks
            db_results['backup'] = self.create_backup(db_path)
            db_results['vacuum'] = self.vacuum_database(db_path)
            db_results['integrity'] = self.integrity_check(db_path)
            db_results['indexes'] = self.optimize_indexes(db_path)
            db_results['analyze'] = self.analyze_database(db_path)
            
            results.append(db_results)
            
        return results

if __name__ == "__main__":
    # List of databases to maintain
    databases = [
        "databases/production/psx_main.db",
        "databases/production/psx_indicators.db",
        "databases/production/psx_kmi100.db",
        "databases/production/psx_kmiall.db",
        "databases/production/psx_kmi30.db"
    ]
    
    # Run maintenance
    maintenance = DatabaseMaintenance(databases)
    results = maintenance.run_maintenance()
    
    # Print summary
    print("\nDatabase Maintenance Summary:")
    print("=" * 50)
    for result in results:
        print(f"\nDatabase: {result['database']}")
        print(f"- Backup created: {'Yes' if result['backup'] else 'No'}")
        print(f"- Vacuum performed: {'Yes' if result['vacuum'] else 'No'}")
        print(f"- Integrity check passed: {'Yes' if result['integrity'] else 'No'}")
        print(f"- Indexes optimized: {'Yes' if result['indexes'] else 'No'}")
        print(f"- Query analysis performed: {'Yes' if result['analyze'] else 'No'}")