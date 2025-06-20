#!/usr/bin/env python3
"""
Duplicate Records Fixer for PSX Stock Data

This script identifies and removes duplicate records from all stock tables in the PSX database.
It handles the "cannot reindex on an axis with duplicate labels" error by cleaning the data first.
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('duplicate_fix.log')
    ]
)
logger = logging.getLogger(__name__)

class DuplicateFixer:
    """Class to handle duplicate record detection and removal."""
      def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        logger.info(f"Initialized DuplicateFixer for database: {self.db_path}")
    
    def get_stock_tables(self) -> List[str]:
        """Get all stock data tables from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Try multiple patterns to find stock tables
            patterns = [
                "PSX_%_stock_data",
                "%_stock_data", 
                "PSX_%",
                "%stock%"
            ]
            
            all_tables = []
            for pattern in patterns:
                cursor.execute(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE '{pattern}'
                    ORDER BY name
                """)
                tables = [row[0] for row in cursor.fetchall()]
                all_tables.extend(tables)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_tables = []
            for table in all_tables:
                if table not in seen:
                    seen.add(table)
                    unique_tables.append(table)
        
        logger.info(f"Found {len(unique_tables)} stock-related tables")
        return unique_tables
    
    def check_table_duplicates(self, table_name: str) -> Dict:
        """Check for duplicate records in a specific table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Read table structure first
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'Date' not in columns:
                    logger.warning(f"No 'Date' column found in {table_name}")
                    return {'has_duplicates': False, 'error': 'No Date column'}
                
                # Count total records
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_records = cursor.fetchone()[0]
                
                if total_records == 0:
                    return {
                        'table_name': table_name,
                        'has_duplicates': False,
                        'total_records': 0,
                        'duplicate_count': 0
                    }
                
                # Count unique dates
                cursor.execute(f"SELECT COUNT(DISTINCT Date) FROM {table_name}")
                unique_dates = cursor.fetchone()[0]
                
                duplicate_count = total_records - unique_dates
                has_duplicates = duplicate_count > 0
                
                result = {
                    'table_name': table_name,
                    'has_duplicates': has_duplicates,
                    'total_records': total_records,
                    'unique_dates': unique_dates,
                    'duplicate_count': duplicate_count,
                    'duplicate_percentage': (duplicate_count / total_records * 100) if total_records > 0 else 0
                }
                
                if has_duplicates:
                    logger.warning(f"{table_name}: {duplicate_count} duplicates ({result['duplicate_percentage']:.1f}%)")
                
                return result
                
        except Exception as e:
            logger.error(f"Error checking duplicates in {table_name}: {e}")
            return {'table_name': table_name, 'has_duplicates': False, 'error': str(e)}
    
    def fix_table_duplicates(self, table_name: str, strategy: str = 'keep_latest') -> Dict:
        """Remove duplicate records from a specific table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, backup the original table
                backup_table = f"{table_name}_backup_{int(time.time())}"
                conn.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}")
                logger.info(f"Created backup table: {backup_table}")
                
                # Read data into pandas for easier duplicate handling
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, parse_dates=['Date'])
                
                if df.empty:
                    return {'table_name': table_name, 'removed_count': 0, 'error': 'Empty table'}
                
                original_count = len(df)
                
                # Remove duplicates based on strategy
                if strategy == 'keep_latest':
                    # Sort by date and keep the last occurrence
                    df_cleaned = df.drop_duplicates(subset=['Date'], keep='last')
                elif strategy == 'keep_earliest':
                    # Sort by date and keep the first occurrence
                    df_cleaned = df.drop_duplicates(subset=['Date'], keep='first')
                elif strategy == 'keep_highest_volume':
                    # Keep the record with highest volume for each date
                    if 'Volume' in df.columns:
                        df_cleaned = df.loc[df.groupby('Date')['Volume'].idxmax()]
                    else:
                        df_cleaned = df.drop_duplicates(subset=['Date'], keep='last')
                else:
                    # Default to keep_latest
                    df_cleaned = df.drop_duplicates(subset=['Date'], keep='last')
                
                final_count = len(df_cleaned)
                removed_count = original_count - final_count
                
                if removed_count > 0:
                    # Clear the original table and insert cleaned data
                    conn.execute(f"DELETE FROM {table_name}")
                    df_cleaned.to_sql(table_name, conn, if_exists='append', index=False)
                    
                    logger.info(f"{table_name}: Removed {removed_count} duplicates ({removed_count/original_count*100:.1f}%)")
                
                return {
                    'table_name': table_name,
                    'original_count': original_count,
                    'final_count': final_count,
                    'removed_count': removed_count,
                    'backup_table': backup_table,
                    'strategy': strategy
                }
                
        except Exception as e:
            logger.error(f"Error fixing duplicates in {table_name}: {e}")
            return {'table_name': table_name, 'removed_count': 0, 'error': str(e)}
    
    def fix_all_duplicates(self, strategy: str = 'keep_latest') -> Dict:
        """Fix duplicates in all stock tables."""
        logger.info("Starting duplicate fixing process for all tables...")
        
        tables = self.get_stock_tables()
        results = {
            'total_tables': len(tables),
            'tables_with_duplicates': 0,
            'total_duplicates_removed': 0,
            'table_results': [],
            'errors': []
        }
        
        for table_name in tables:
            logger.info(f"Processing {table_name}...")
            
            # Check for duplicates first
            check_result = self.check_table_duplicates(table_name)
            
            if check_result.get('has_duplicates', False):
                results['tables_with_duplicates'] += 1
                
                # Fix duplicates
                fix_result = self.fix_table_duplicates(table_name, strategy)
                results['table_results'].append(fix_result)
                
                if 'error' not in fix_result:
                    results['total_duplicates_removed'] += fix_result.get('removed_count', 0)
                else:
                    results['errors'].append(f"{table_name}: {fix_result['error']}")
            else:
                results['table_results'].append({
                    'table_name': table_name,
                    'removed_count': 0,
                    'status': 'no_duplicates'
                })
                if 'error' in check_result:
                    results['errors'].append(f"{table_name}: {check_result['error']}")
        
        logger.info(f"Completed! Fixed {results['tables_with_duplicates']} tables, removed {results['total_duplicates_removed']} duplicates")
        return results

def main():
    """Main function to run the duplicate fixer."""
    print("=" * 80)
    print("🔧 PSX DATABASE DUPLICATE FIXER")
    print("=" * 80)
      # Common database locations - expanded search
    db_locations = [
        # Main project root
        Path.cwd().parent.parent.parent / "PSX_Stock_Data.db",
        Path.cwd() / "PSX_Stock_Data.db",
        
        # Data directories
        Path.cwd() / "data" / "databases" / "production" / "PSX_consolidated_data_PSX.db",
        Path.cwd() / "data" / "databases" / "production" / "psx_consolidated_data_PSX.db",
        Path.cwd().parent.parent.parent / "data" / "databases" / "production" / "PSX_consolidated_data_PSX.db",
        Path.cwd().parent.parent.parent / "data" / "databases" / "production" / "psx_consolidated_data_PSX.db",
        
        # Other possible locations
        Path("../../../PSX_Stock_Data.db"),
        Path("../../../../PSX_Stock_Data.db"),
        Path("../../../data/PSX_Stock_Data.db"),
        Path("../../../../data/PSX_Stock_Data.db"),
        
        # Search in the entire project directory for any .db files
        *list(Path.cwd().parent.parent.parent.rglob("*.db")),
    ]
    
    # Find database
    db_path = None
    for location in db_locations:
        if location.exists():
            db_path = location
            break
    
    if not db_path:
        print("❌ Could not find PSX database file!")
        print("Please ensure one of these files exists:")
        for location in db_locations:
            print(f"   - {location}")
        return
    
    print(f"✅ Found database: {db_path}")
    print(f"   Size: {db_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        fixer = DuplicateFixer(str(db_path))
        
        # Show duplicate statistics first
        print("\n📊 CHECKING FOR DUPLICATES...\n")
        tables = fixer.get_stock_tables()
        
        tables_with_duplicates = []
        total_duplicates = 0
        
        for table in tables[:10]:  # Check first 10 tables for preview
            check_result = fixer.check_table_duplicates(table)
            if check_result.get('has_duplicates', False):
                tables_with_duplicates.append(table)
                total_duplicates += check_result.get('duplicate_count', 0)
                print(f"   ⚠️  {table}: {check_result.get('duplicate_count', 0)} duplicates")
        
        if not tables_with_duplicates:
            print("✅ No duplicates found in sample tables!")
            return
        
        print(f"\n Found duplicates in {len(tables_with_duplicates)} tables (from sample of 10)")
        print(f" Estimated total duplicates: ~{total_duplicates}")
        
        # Ask for confirmation
        print("\n" + "─" * 60)
        response = input("Do you want to fix all duplicates? (y/N): ").strip().lower()
        
        if response != 'y':
            print("❌ Operation cancelled")
            return
        
        # Choose strategy
        print("\nChoose duplicate removal strategy:")
        print("1. Keep Latest (recommended)")
        print("2. Keep Earliest") 
        print("3. Keep Highest Volume")
        
        strategy_choice = input("Enter choice (1-3, default=1): ").strip()
        strategy_map = {
            '1': 'keep_latest',
            '2': 'keep_earliest', 
            '3': 'keep_highest_volume'
        }
        strategy = strategy_map.get(strategy_choice, 'keep_latest')
        
        print(f"\n🚀 Starting duplicate removal with strategy: {strategy}")
        print("This may take several minutes...")
        
        # Fix all duplicates
        results = fixer.fix_all_duplicates(strategy)
        
        # Show results
        print("\n" + "=" * 60)
        print("✅ DUPLICATE FIXING COMPLETED!")
        print("=" * 60)
        print(f"📊 Total tables processed: {results['total_tables']}")
        print(f"🔧 Tables with duplicates fixed: {results['tables_with_duplicates']}")
        print(f"🗑️  Total duplicates removed: {results['total_duplicates_removed']}")
        
        if results['errors']:
            print(f"\n⚠️  Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
        
        print(f"\n💾 Backup tables created for modified tables")
        print(f"📝 Detailed log saved to: duplicate_fix.log")
        print("\nYou can now run the enhanced processor without duplicate errors!")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
