from typing import Dict, List, Optional
import logging
import sqlite3
import os
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
TECHNICAL_DB_PATH = "data\\databases\\production\\psx_consolidated_data_indicators_PSX.db"
FAIRVALUE_DB_PATH = "data\\databases\\production\\fairvalue.db"

def ensure_database_exists():
    """Ensure the databases exist and have the required tables"""
    try:
        # Ensure technical database exists
        if not os.path.exists(TECHNICAL_DB_PATH):
            logger.info(f"Creating new technical database at {TECHNICAL_DB_PATH}")
            conn = sqlite3.connect(TECHNICAL_DB_PATH)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            create_tables(cursor)
            
            conn.commit()
            conn.close()
            logger.info("Technical database created successfully with required tables")
        else:
            # Check if tables exist, create if they don't
            conn = sqlite3.connect(TECHNICAL_DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [t[0] for t in tables]
            
            if not all(table in table_names for table in ['stock_data', 'analysis_results', 'signals']):
                logger.info("Some tables missing in technical database, creating required tables")
                create_tables(cursor)
                conn.commit()
            
            conn.close()
            
        # Ensure fairvalue database exists
        if not os.path.exists(FAIRVALUE_DB_PATH):
            logger.info(f"Creating new fairvalue database at {FAIRVALUE_DB_PATH}")
            conn = sqlite3.connect(FAIRVALUE_DB_PATH)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            create_tables(cursor)
            
            conn.commit()
            conn.close()
            logger.info("Fairvalue database created successfully with required tables")
        else:
            # Check if tables exist, create if they don't
            conn = sqlite3.connect(FAIRVALUE_DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [t[0] for t in tables]
            
            if not all(table in table_names for table in ['stock_data', 'analysis_results', 'signals']):
                logger.info("Some tables missing in fairvalue database, creating required tables")
                create_tables(cursor)
                conn.commit()
            
            conn.close()
            
    except Exception as e:
        logger.error(f"Error ensuring databases exist: {e}")

def create_tables(cursor):
    """Create required database tables"""
    try:
        # Stock data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                ema_20 REAL,
                ema_50 REAL,
                ema_200 REAL,
                bollinger_upper REAL,
                bollinger_middle REAL,
                bollinger_lower REAL,
                stoch_k REAL,
                stoch_d REAL,
                ichimoku_tenkan REAL,
                ichimoku_kijun REAL,
                ichimoku_senkou_span_a REAL,
                ichimoku_senkou_span_b REAL,
                ichimoku_cloud_green INTEGER,
                ichimoku_cloud_red INTEGER,
                support_level REAL,
                resistance_level REAL,
                trend TEXT,
                momentum TEXT,
                volume_profile TEXT,
                pattern TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')

        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                technical_score REAL,
                financial_score REAL,
                ai_score REAL,
                overall_score REAL,
                recommendation TEXT,
                confidence REAL,
                risk_assessment TEXT,
                intrinsic_value REAL,
                margin_of_safety REAL,
                analysis_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')

        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_strength REAL,
                price_at_signal REAL,
                target_price REAL,
                stop_loss REAL,
                confidence REAL,
                status TEXT DEFAULT 'open',
                entry_date TEXT,
                exit_date TEXT,
                exit_price REAL,
                profit_loss REAL,
                signal_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

def save_stock_data(data: List[Dict]) -> int:
    """Save stock data to fairvalue database"""
    try:
        conn = sqlite3.connect(FAIRVALUE_DB_PATH)
        cursor = conn.cursor()
        
        count = 0
        for item in data:
            if not item or 'symbol' not in item or 'date' not in item:
                continue
                
            columns = [
                'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 
                'macd', 'macd_signal', 'macd_hist', 'sma_20', 'sma_50', 'sma_200', 
                'ema_20', 'ema_50', 'ema_200', 'bollinger_upper', 'bollinger_middle', 
                'bollinger_lower', 'stoch_k', 'stoch_d', 'ichimoku_tenkan', 
                'ichimoku_kijun', 'ichimoku_senkou_span_a', 'ichimoku_senkou_span_b', 
                'ichimoku_cloud_green', 'ichimoku_cloud_red', 'support_level', 
                'resistance_level', 'trend', 'momentum', 'volume_profile', 'pattern'
            ]
            
            values = [item.get(col, None) for col in columns]
            placeholders = ','.join(['?' for _ in columns])
            columns_str = ','.join(columns)
            
            query = f"""
                INSERT OR REPLACE INTO stock_data ({columns_str})
                VALUES ({placeholders})
            """
            
            cursor.execute(query, values)
            count += 1
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {count} stock data records to fairvalue database")
        return count
        
    except Exception as e:
        logger.error(f"Error saving stock data to fairvalue database: {e}")
        return 0

def save_analysis_results(results: List[Dict]) -> int:
    """Save analysis results to fairvalue database"""
    try:
        conn = sqlite3.connect(FAIRVALUE_DB_PATH)
        cursor = conn.cursor()
        
        count = 0
        for item in results:
            if not item or 'symbol' not in item or 'date' not in item:
                continue
                
            columns = [
                'symbol', 'date', 'technical_score', 'financial_score', 'ai_score', 
                'overall_score', 'recommendation', 'confidence', 'risk_assessment', 
                'intrinsic_value', 'margin_of_safety', 'analysis_details'
            ]
            
            values = [item.get(col, None) for col in columns]
            # Convert analysis_details to JSON string if it's a dict
            if isinstance(values[-1], dict):
                values[-1] = json.dumps(values[-1])
                
            placeholders = ','.join(['?' for _ in columns])
            columns_str = ','.join(columns)
            
            query = f"""
                INSERT OR REPLACE INTO analysis_results ({columns_str})
                VALUES ({placeholders})
            """
            
            cursor.execute(query, values)
            count += 1
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {count} analysis results to fairvalue database")
        return count
        
    except Exception as e:
        logger.error(f"Error saving analysis results to fairvalue database: {e}")
        return 0

def save_signals(signals: List[Dict]) -> int:
    """Save trading signals to fairvalue database"""
    try:
        conn = sqlite3.connect(FAIRVALUE_DB_PATH)
        cursor = conn.cursor()
        
        count = 0
        for item in signals:
            if not item or 'symbol' not in item or 'date' not in item or 'signal_type' not in item:
                continue
                
            columns = [
                'symbol', 'date', 'signal_type', 'signal_strength', 'price_at_signal', 
                'target_price', 'stop_loss', 'confidence', 'status', 'entry_date', 
                'exit_date', 'exit_price', 'profit_loss', 'signal_details'
            ]
            
            values = [item.get(col, None) for col in columns]
            # Convert signal_details to JSON string if it's a dict
            if isinstance(values[-1], dict):
                values[-1] = json.dumps(values[-1])
                
            placeholders = ','.join(['?' for _ in columns])
            columns_str = ','.join(columns)
            
            query = f"""
                INSERT INTO signals ({columns_str})
                VALUES ({placeholders})
            """
            
            cursor.execute(query, values)
            count += 1
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {count} signals to fairvalue database")
        return count
        
    except Exception as e:
        logger.error(f"Error saving signals to fairvalue database: {e}")
        return 0

def get_latest_stock_data(symbol: str, limit: int = 1) -> List[Dict]:
    """Get latest stock data for a symbol from technical database"""
    try:
        conn = sqlite3.connect(TECHNICAL_DB_PATH)
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM stock_data 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT ?
        """
        
        cursor.execute(query, (symbol, limit))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        result = []
        for row in rows:
            result.append(dict(zip(columns, row)))
            
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error getting latest stock data for {symbol}: {e}")
        return []

def get_latest_analysis(symbol: str, limit: int = 1) -> List[Dict]:
    """Get latest analysis results for a symbol from fairvalue database"""
    try:
        conn = sqlite3.connect(FAIRVALUE_DB_PATH)
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM analysis_results 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT ?
        """
        
        cursor.execute(query, (symbol, limit))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        result = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            # Parse analysis_details if it's a JSON string
            if row_dict.get('analysis_details'):
                try:
                    row_dict['analysis_details'] = json.loads(row_dict['analysis_details'])
                except:
                    pass
            result.append(row_dict)
            
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error getting latest analysis for {symbol}: {e}")
        return []

def get_active_signals(symbol: Optional[str] = None) -> List[Dict]:
    """Get active signals, optionally filtered by symbol from fairvalue database"""
    try:
        conn = sqlite3.connect(FAIRVALUE_DB_PATH)
        cursor = conn.cursor()
        
        if symbol:
            query = """
                SELECT * FROM signals 
                WHERE status = 'open' AND symbol = ?
                ORDER BY date DESC
            """
            cursor.execute(query, (symbol,))
        else:
            query = """
                SELECT * FROM signals 
                WHERE status = 'open'
                ORDER BY date DESC
            """
            cursor.execute(query)
            
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        result = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            # Parse signal_details if it's a JSON string
            if row_dict.get('signal_details'):
                try:
                    row_dict['signal_details'] = json.loads(row_dict['signal_details'])
                except:
                    pass
            result.append(row_dict)
            
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        return []

def update_signal_status(signal_id: int, status: str, exit_price: Optional[float] = None, 
                        exit_date: Optional[str] = None, profit_loss: Optional[float] = None) -> bool:
    """Update signal status and related fields in fairvalue database"""
    try:
        conn = sqlite3.connect(FAIRVALUE_DB_PATH)
        cursor = conn.cursor()
        
        query = """
            UPDATE signals 
            SET status = ?,
                exit_price = ?,
                exit_date = ?,
                profit_loss = ?
            WHERE id = ?
        """
        
        cursor.execute(query, (status, exit_price, exit_date, profit_loss, signal_id))
        conn.commit()
        conn.close()
        
        logger.info(f"Updated signal {signal_id} status to {status}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating signal status: {e}")
        return False
