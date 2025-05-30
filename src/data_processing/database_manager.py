from __future__ import annotations
import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for stock analysis data.
    
    This class handles all database-related operations including initialization,
    connection management, and data storage for the Fair Value Calculator.
    
    Attributes:
        db_path (str): Path to the main analysis database
        dividend_db_path (str): Path to the dividend database
    """
    
    def __init__(self, db_path: str, dividend_db_path: str) -> None:
        """Initialize the DatabaseManager with database paths.
        
        Args:
            db_path (str): Path to the main analysis database
            dividend_db_path (str): Path to the dividend database
        """
        self.db_path = db_path
        self.dividend_db_path = dividend_db_path
        self._init_database()

    def _handle_error(self, error: Exception, context: str, default_return=None):
        """Utility method to handle exceptions with consistent logging.
        
        Args:
            error: The exception object caught.
            context: A string describing the context of the error.
            default_return: The value to return in case of error, if applicable.
        
        Returns:
            The default_return value if provided, otherwise None.
        """
        logger.error(f"Error in {context}: {str(error)}")
        return default_return

    def _init_database(self):
        """Initialize the SQLite database and create necessary tables"""
        try:
            self._create_database_directory()
            self._connect_to_database()
            self._create_tables()
            self._verify_tables()
            self._close_database_connection()
        except Exception as e:
            self._handle_error(e, "initializing database")
            if hasattr(self, 'conn'):
                self.conn.close()
            raise

    def _create_database_directory(self):
        """Create the directory for the database if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _connect_to_database(self):
        """Connect to the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _create_tables(self):
        """Create necessary tables in the database"""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tradingview_ta (
            symbol TEXT,
            date TEXT,
            recommendation TEXT,
            buy_signals INTEGER,
            sell_signals INTEGER,
            neutral_signals INTEGER,
            rsi REAL,
            stoch_k REAL,
            stoch_d REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            sma_20 REAL,
            sma_50 REAL,
            sma_200 REAL,
            ema_20 REAL,
            ema_50 REAL,
            ema_200 REAL,
            close REAL,
            open REAL,
            high REAL,
            low REAL,
            volume REAL,
            change REAL,
            change_percent REAL,
            bb_upper REAL,
            bb_lower REAL,
            ao REAL,
            psar REAL,
            vwma REAL,
            hull_ma9 REAL,
            source TEXT,
            last_updated TEXT,
            PRIMARY KEY (symbol, date)
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tradingview_signals (
            symbol TEXT,
            date TEXT,
            signal_type TEXT,
            signal_strength REAL,
            confidence_score REAL,
            technical_score REAL,
            trend_score REAL,
            momentum_score REAL,
            volume_score REAL,
            volatility_score REAL,
            support_level REAL,
            resistance_level REAL,
            stop_loss REAL,
            take_profit REAL,
            risk_reward_ratio REAL,
            analysis_summary TEXT,
            indicators_used TEXT,
            last_updated TEXT,
            ai_score REAL,
            ai_confidence REAL,
            ai_pattern_recognition TEXT,
            ai_signal_strength TEXT,
            ai_risk_assessment TEXT,
            ai_recommendation TEXT,
            ai_price_targets TEXT,
            ai_entry_points TEXT,
            ai_exit_points TEXT,
            ai_analysis_date TEXT,
            PRIMARY KEY (symbol, date)
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_reports (
            symbol TEXT,
            report_date TEXT,
            eps_growth REAL,
            revenue_growth REAL,
            profit_margin REAL,
            debt_to_equity REAL,
            current_ratio REAL,
            roe REAL,
            last_updated TEXT,
            PRIMARY KEY (symbol, report_date)
        )
        ''')

        self.conn.commit()

    def _verify_tables(self):
        """Verify tables exist and have correct structure"""
        self.cursor.execute("SELECT COUNT(*) FROM tradingview_ta")
        logger.info(f"tradingview_ta table initialized with {self.cursor.fetchone()[0]} records")

        self.cursor.execute("SELECT COUNT(*) FROM tradingview_signals")
        logger.info(f"tradingview_signals table initialized with {self.cursor.fetchone()[0]} records")

        self.cursor.execute("SELECT COUNT(*) FROM financial_reports")
        logger.info(f"financial_reports table initialized with {self.cursor.fetchone()[0]} records")

    def _close_database_connection(self):
        """Close the database connection"""
        self.conn.close()
        logger.info(f"Database initialized successfully at {self.db_path}")

    def save_tradingview_ta_data_to_db(self, symbol: str, data: Dict):
        """Save TradingView TA data to the database with duplicate validation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            # Check if record exists for this symbol and date
            cursor.execute("""
                SELECT * FROM tradingview_ta 
                WHERE symbol = ? AND date = ?
            """, (symbol, current_date))
            
            existing_record = cursor.fetchone()
            
            if existing_record:
                # Get column names
                columns = [description[0] for description in cursor.description]
                existing_data = dict(zip(columns, existing_record))
                
                # Compare values and build update query only for changed fields
                update_fields = []
                update_values = []
                
                for key, new_value in data.items():
                    if key in columns and key not in ['symbol', 'date']:  # Skip primary key fields
                        old_value = existing_data.get(key)
                        if new_value != old_value and new_value is not None:
                            update_fields.append(f"{key} = ?")
                            update_values.append(new_value)
                
                if update_fields:  # Only update if there are changes
                    update_query = f"""
                    UPDATE tradingview_ta 
                    SET {', '.join(update_fields)}, last_updated = ?
                    WHERE symbol = ? AND date = ?
                    """
                    update_values.extend([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), symbol, current_date])
                    
                    cursor.execute(update_query, update_values)
                    logger.info(f"Updated {len(update_fields)} fields for {symbol} on {current_date}")
                else:
                    logger.info(f"No changes detected for {symbol} on {current_date}")
            else:
                # Insert new record
                # Get all column names from the table
                cursor.execute("PRAGMA table_info(tradingview_ta)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Prepare values list with None for missing columns
                values = []
                for column in columns:
                    if column == 'symbol':
                        values.append(symbol)
                    elif column == 'date':
                        values.append(current_date)
                    elif column == 'last_updated':
                        values.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    elif column == 'source':
                        values.append('tradingview_ta')
                    else:
                        values.append(data.get(column))
                
                # Create placeholders for the SQL query
                placeholders = ','.join(['?' for _ in columns])
                
                # Insert new record
                cursor.execute(f'''
                INSERT INTO tradingview_ta 
                ({', '.join(columns)})
                VALUES ({placeholders})
                ''', values)
                
                logger.info(f"Inserted new record for {symbol} on {current_date}")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving TradingView TA data for {symbol} to database: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return False

    def get_latest_data(self, symbol: str) -> Dict:
        """Get the latest data for a symbol from the database with enhanced validation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the latest data with validation
            cursor.execute("""
                SELECT * FROM tradingview_ta 
                WHERE symbol = ? 
                AND close IS NOT NULL 
                AND volume IS NOT NULL 
                AND date IS NOT NULL 
                ORDER BY date DESC 
                LIMIT 1
            """, (symbol,))
            
            columns = [description[0] for description in cursor.description]
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                data = dict(zip(columns, row))
                # Validate the data
                required_fields = ['close', 'open', 'high', 'low', 'volume', 'date']
                if all(data.get(field) is not None for field in required_fields):
                    logger.info(f"Found valid cached data for {symbol} from {data['date']}")
                    return data
                else:
                    logger.warning(f"Found incomplete cached data for {symbol}")
                    return {}
            
            logger.info(f"No valid cached data found for {symbol}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {str(e)}")
            return {}
