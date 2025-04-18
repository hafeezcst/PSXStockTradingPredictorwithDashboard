import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import sqlite3
import time
import logging
from typing import Tuple
from pathlib import Path
from config.paths import (
    DATA_LOGS_DIR,
    PRODUCTION_DB_DIR,
    PSX_DB_PATH,
    PSX_INVESTING_DB_PATH
)

# Configure logging with proper path
logging.basicConfig(
    filename=DATA_LOGS_DIR / 'sql_duplicate_remover.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def remove_duplicates(cursor, table_name: str) -> Tuple[int, int]:
    """Remove duplicate records from a table based on the Date column.
    
    Args:
        cursor: SQLite database cursor
        table_name: Name of the table to process
        
    Returns:
        Tuple containing (total duplicates found, total records deleted)
    """
    try:
        # Find duplicate records
        cursor.execute(f"""
            SELECT Date, COUNT(*) as count
            FROM {table_name}
            GROUP BY Date
            HAVING count > 1
        """)
        duplicates = cursor.fetchall()
        total_duplicates = sum(row[1] - 1 for row in duplicates)
        
        if total_duplicates > 0:
            # Create temporary table with unique records
            cursor.execute(f"""
                CREATE TEMPORARY TABLE temp_{table_name} AS
                SELECT * FROM (
                    SELECT DISTINCT * FROM {table_name}
                    ORDER BY Date DESC
                )
            """)
            
            # Drop original table and rename temp table
            cursor.execute(f"DROP TABLE {table_name}")
            cursor.execute(f"ALTER TABLE temp_{table_name} RENAME TO {table_name}")
            
            return total_duplicates, total_duplicates
        
        return 0, 0
        
    except Exception as e:
        logging.error(f"Error removing duplicates from {table_name}: {e}")
        return 0, 0

def process_database(conn, db_name: str) -> None:
    """Process all tables in a database to remove duplicates.
    
    Args:
        conn: SQLite database connection
        db_name: Name of the database for logging purposes
    """
    try:
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        total_duplicates = 0
        total_deleted = 0
        
        # Process each table
        for table in tables:
            table_name = table[0]
            print(f"\nProcessing table: {table_name}")
            
            dups, deleted = remove_duplicates(cursor, table_name)
            total_duplicates += dups
            total_deleted += deleted
            
            if dups > 0:
                print(f"Found {dups} duplicates in {table_name}")
                print(f"Deleted {deleted} records from {table_name}")
            
            # Commit after each table
            conn.commit()
        
        # Final reporting
        print(f"Total number of duplicate records found in {db_name}: {total_duplicates}")
        print(f"Total number of records deleted in {db_name}: {total_deleted}")
        
    except Exception as e:
        logging.error(f"Error processing database {db_name}: {e}")
    finally:
        # Close the connection
        conn.close()

def main():
    """Main function to execute the duplicate removal process."""
    start_time = time.time()
    
    # Database paths with proper error handling
    databases = [
        (PSX_DB_PATH, "PSX Main Database"),
        (PSX_INVESTING_DB_PATH, "PSX Alternative Database")
    ]
    
    for db_path, db_name in databases:
        try:
            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Processing database: {db_path}")
            conn = sqlite3.connect(db_path)
            process_database(conn, db_name)
            
        except Exception as e:
            logging.error(f"Failed to connect to {db_path}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

