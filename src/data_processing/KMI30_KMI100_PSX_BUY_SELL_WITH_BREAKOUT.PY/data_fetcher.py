import sqlite3
import os
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from .db_handler import db_connection
from tenacity import retry, stop_after_attempt, wait_fixed
import logging

def debug_database_schema(db_path: str) -> None:
    """Debug function to print database schema"""
    print(f"\nAttempting to check database at: {db_path}")
    if not os.path.exists(db_path):
        print(f"Error: Database file does not exist at {db_path}")
        return
    
    try:
        print(f"Connecting to database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("Connected successfully")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in database")
            return
            
        print(f"\nFound {len(tables)} tables in {db_path}:")
        for table in tables:
            print(f"\nTable: {table[0]}")
            cursor.execute(f"PRAGMA table_info({table[0]});")
            columns = cursor.fetchall()
            if not columns:
                print("  No columns found")
                continue
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
                
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"\nError checking schema: {str(e)}")
        import traceback
        traceback.print_exc()

# Set up logger
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_freefloatratio(symbol: str) -> Optional[float]:
    """Get the free float ratio from psxsymbols.db"""
    try:
        with db_connection('data/databases/production/psxsymbols.db') as conn:
            cursor = conn.cursor()
            query = "SELECT freefloatratio FROM KMIALL WHERE symbol = ?;"
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
    except sqlite3.OperationalError as e:
        logger.error(f"Error fetching freefloatratio for {symbol}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_multibagger_symbols() -> List[str]:
    """Fetch multibagger symbols from the database"""
    try:
        with db_connection('data/databases/production/psxsymbols.db') as conn:
            cursor = conn.cursor()
            query = "SELECT symbol FROM ROIC_GT_25;"
            cursor.execute(query)
            results = cursor.fetchall()
            return [row[0].strip().upper() for row in results]
    except sqlite3.OperationalError as e:
        logger.error(f"Error fetching multibagger symbols: {e}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_table_names(cursor: sqlite3.Cursor) -> List[str]:
    """Fetch both KMI30 and KMI100 tables from the database"""
    try:
        symbols_file_path = os.path.join(os.getcwd(), 'data/databases/production/psxsymbols.xlsx')
        
        # Read KMI30 symbols
        kmi30_df = pd.read_excel(symbols_file_path, sheet_name='KMI30')
        KMI30_symbols = set(kmi30_df.iloc[:, 0].tolist()[:30])
        
        # Read KMI100 symbols
        kmi100_df = pd.read_excel(symbols_file_path, sheet_name='KMI100')
        KMI100_symbols = set(kmi100_df.iloc[:, 0].tolist()[:100])
        
        # Get all tables from database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Filter tables that match KMI symbols
        return [table[0] for table in tables 
                if table[0].startswith('PSX_') 
                and table[0].endswith('_stock_data')
                and table[0].replace('PSX_', '').replace('_stock_data', '').strip().upper() in KMI30_symbols.union(KMI100_symbols)]
    except Exception as e:
        logger.error(f"Error fetching KMI table names: {e}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_stock_data(cursor: sqlite3.Cursor, table_name: str, limit: int = 30) -> Optional[List[Tuple]]:
    """Fetch stock data for the last N dates"""
    try:
        # Try with different column name variations
        try:
            query = f"""
                SELECT Date, Close, Volume, RSI_weekly_Avg, RSI_Monthly, RSI_3Months_Avg, 
                       RSI_Monthly_Avg, AO_weekly, MA_30, pct_change 
                FROM {table_name} 
                ORDER BY Date DESC 
                LIMIT ?;
            """
            cursor.execute(query, (limit,))
            return cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"First column name attempt failed, trying alternatives: {e}")
            try:
                # Try original column names if first attempt failed
                query = f"""
                    SELECT Date, Close, Volume, RSI_Weekly_Avg, RSI_Monthly, RSI_3Months_Avg, 
                           RSI_Monthly_Avg, AO_Weekly, MA_30, pct_change 
                    FROM {table_name} 
                    ORDER BY Date DESC 
                    LIMIT ?;
                """
                cursor.execute(query, (limit,))
                return cursor.fetchall()
            except sqlite3.OperationalError as e:
                logger.error(f"Failed to fetch stock data with all column name variations: {e}")
                # Log actual columns for debugging
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                logger.error(f"Available columns in {table_name}: {[col[1] for col in columns]}")
                return None
        if results and len(results) >= 2:
            return results
        return None
    except Exception as e:
        logger.error(f"Error fetching stock data for {table_name}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_dividend_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Get dividend information for a specific symbol if available"""
    db_path = 'data/databases/production/PSX_Dividend_Schedule.db'
    if not os.path.exists(db_path):
        return None
    
    try:
        with db_connection(db_path) as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            query = """
                SELECT symbol, company_name, face_value, dividend_amount, right_amount, 
                       bc_to, last_close, payout_text
                FROM dividend_schedule 
                WHERE bc_to >= ? AND bc_to != '-' AND UPPER(symbol) = UPPER(?)
                ORDER BY date(bc_to) LIMIT 1;
            """
            cursor.execute(query, (today, symbol))
            row = cursor.fetchone()
            if row:
                return {
                    'symbol': row[0],
                    'company_name': row[1],
                    'face_value': row[2],
                    'dividend_amount': row[3],
                    'right_amount': row[4],
                    'bc_to': row[5],
                    'last_close': row[6],
                    'payout_text': row[7]
                }
    except Exception as e:
        logger.error(f"Error getting dividend info for {symbol}: {e}")
    return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_ao_change_date(cursor: sqlite3.Cursor, table_name: str) -> Tuple[Optional[str], Optional[float]]:
    """Get the date when AO changed from negative to positive"""
    try:
        query = f"SELECT Date, Close, AO_weekly FROM {table_name} ORDER BY Date DESC;"
        cursor.execute(query)
        results = cursor.fetchall()

        previous_ao = None
        for date, close, ao_weekly in results:
            if previous_ao is not None and ao_weekly is not None and ao_weekly < 0 <= previous_ao:
                return date.split(' ')[0], close
            previous_ao = ao_weekly
        return None, None
    except Exception as e:
        logger.error(f"Error getting AO change date for {table_name}: {e}")
        return None, None
