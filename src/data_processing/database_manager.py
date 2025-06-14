import sqlite3
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import closing
from queue import Queue
from threading import Lock

logger = logging.getLogger(__name__)

class ConnectionPool:
    """SQLite connection pool for efficient connection reuse"""
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = Queue(max_connections)
        self._lock = Lock()
        for _ in range(max_connections):
            self._pool.put(sqlite3.connect(db_path, check_same_thread=False))

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool"""
        return self._pool.get()

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        self._pool.put(conn)

    def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database and create necessary tables"""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cursor = conn.cursor()
                    
                    # Create tradingview_ta table
                    cursor.execute('''
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
                    
                    # Create tradingview_signals table
                    cursor.execute('''
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
                    
                    # Create financial_reports table
                    cursor.execute('''
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
                    
                    logger.info(f"Database initialized successfully at {self.db_path}")
                    
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def batch_save_tradingview_ta_data(self, data_list: List[Tuple[str, Dict]]) -> bool:
        """Save multiple TradingView TA records in a single transaction"""
        conn = self.connection_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                
                # Prepare bulk insert statement
                cursor.execute("PRAGMA table_info(tradingview_ta)")
                columns = [column[1] for column in cursor.fetchall()]
                placeholders = ','.join(['?' for _ in columns])
                
                for symbol, data in data_list:
                    current_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
                    
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
                    
                    # Use INSERT OR REPLACE for atomic upsert
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO tradingview_ta
                        ({', '.join(columns)})
                        VALUES ({placeholders})
                    ''', values)
                
                return True
        except Exception as e:
            logger.error(f"Error saving TradingView TA data: {str(e)}")
            return False
        finally:
            self.connection_pool.return_connection(conn)

    def save_analysis(self, symbol: str, analysis: Dict) -> bool:
        """Save analysis results to the database using connection pool"""
        conn = self.connection_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                cursor.execute(
                    "INSERT OR REPLACE INTO tradingview_signals (symbol, date, signal_type, signal_strength, confidence_score, technical_score, trend_score, momentum_score, volume_score, volatility_score, support_level, resistance_level, stop_loss, take_profit, risk_reward_ratio, analysis_summary, indicators_used, last_updated, ai_score, ai_confidence, ai_pattern_recognition, ai_signal_strength, ai_risk_assessment, ai_recommendation, ai_price_targets, ai_entry_points, ai_exit_points, ai_analysis_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                    symbol,
                    datetime.now().strftime('%Y-%m-%d'),
                    analysis.get('signal_type'),
                    analysis.get('signal_strength'),
                    analysis.get('confidence_score'),
                    analysis.get('technical_score'),
                    analysis.get('trend_score'),
                    analysis.get('momentum_score'),
                    analysis.get('volume_score'),
                    analysis.get('volatility_score'),
                    analysis.get('support_level'),
                    analysis.get('resistance_level'),
                    analysis.get('stop_loss'),
                    analysis.get('take_profit'),
                    analysis.get('risk_reward_ratio'),
                    '|'.join(analysis.get('analysis_summary', [])),
                    '|'.join(analysis.get('indicators_used', [])),
                    current_time,
                    analysis.get('ai_score', 0.0),
                    analysis.get('ai_confidence', 0.0),
                    analysis.get('ai_pattern_recognition', ''),
                    analysis.get('ai_signal_strength', ''),
                    analysis.get('ai_risk_assessment', ''),
                    analysis.get('ai_recommendation', ''),
                    analysis.get('ai_price_targets', ''),
                    analysis.get('ai_entry_points', ''),
                    analysis.get('ai_exit_points', ''),
                    current_time
                ))
                
                return True
                
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            return False
        finally:
            self.connection_pool.return_connection(conn)

    _DATA_CACHE = {}
    _CACHE_EXPIRY = 3600  # 1 hour in seconds

    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get the latest data for a symbol with caching"""
        cache_key = f"latest_{symbol}"
        
        # Check cache first
        if cache_key in self._DATA_CACHE:
            cached_data, timestamp = self._DATA_CACHE[cache_key]
            if time.time() - timestamp < self._CACHE_EXPIRY:
                return cached_data.copy()
        
        conn = self.connection_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                
                # Optimized query - only select needed columns
                cursor.execute("""
                    SELECT
                        symbol, date, close, open, high, low, volume,
                        rsi, macd, sma_20, ema_20, ao, recommendation
                    FROM tradingview_ta
                    WHERE symbol = ?
                    AND close IS NOT NULL
                    AND volume IS NOT NULL
                    AND date IS NOT NULL
                    ORDER BY date DESC
                    LIMIT 1
                """, (symbol,))
                
                columns = [description[0] for description in cursor.description]
                row = cursor.fetchone()
                
                if row:
                    data = dict(zip(columns, row))
                    required_fields = ['close', 'open', 'high', 'low', 'volume', 'date']
                    if all(data.get(field) is not None for field in required_fields):
                        # Update cache
                        self._DATA_CACHE[cache_key] = (data.copy(), time.time())
                        return data
                
                return None
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            return None
        finally:
            self.connection_pool.return_connection(conn)

    def get_previous_analysis(self, symbol: str) -> Optional[Dict]:
        """Get the previous analysis for a symbol with caching"""
        cache_key = f"analysis_{symbol}"
        
        # Check cache first
        if cache_key in self._DATA_CACHE:
            cached_data, timestamp = self._DATA_CACHE[cache_key]
            if time.time() - timestamp < self._CACHE_EXPIRY:
                return cached_data.copy()
        
        conn = self.connection_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                
                # Optimized query - only select needed columns
                cursor.execute("""
                    SELECT
                        symbol, date, signal_type, signal_strength,
                        confidence_score, technical_score, trend_score,
                        momentum_score, volume_score, volatility_score,
                        support_level, resistance_level, stop_loss,
                        take_profit, risk_reward_ratio, ai_recommendation
                    FROM tradingview_signals
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 1
                """, (symbol,))
                
                columns = [description[0] for description in cursor.description]
                row = cursor.fetchone()
                
                if row:
                    data = dict(zip(columns, row))
                    # Update cache
                    self._DATA_CACHE[cache_key] = (data.copy(), time.time())
                    return data
                
                return None
        except Exception as e:
            logger.error(f"Error getting previous analysis: {str(e)}")
            return None
        finally:
            self.connection_pool.return_connection(conn)

    def verify_database_data(self) -> Dict:
        """Verify the quality and completeness of data in the database with caching"""
        cache_key = "database_stats"
        
        # Check cache first
        if cache_key in self._DATA_CACHE:
            cached_data, timestamp = self._DATA_CACHE[cache_key]
            if time.time() - timestamp < self._CACHE_EXPIRY:
                return cached_data.copy()
        
        conn = self.connection_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                
                # Execute all queries in a single transaction
                with conn:
                    # Get all stats in one optimized query
                    cursor.execute("""
                        SELECT
                            COUNT(*) as total_records,
                            COUNT(DISTINCT symbol) as unique_symbols,
                            MAX(date) as latest_date,
                            SUM(CASE WHEN rsi IS NOT NULL
                                 AND macd IS NOT NULL
                                 AND sma_20 IS NOT NULL
                                 AND ema_20 IS NOT NULL
                                 AND close IS NOT NULL
                                 THEN 1 ELSE 0 END) as complete_records
                        FROM tradingview_ta
                    """)
                    
                    row = cursor.fetchone()
                    if row:
                        total_records = row[0]
                        unique_symbols = row[1]
                        latest_date = row[2]
                        complete_records = row[3]
                        
                        # Calculate completeness percentage
                        completeness = (complete_records / total_records * 100) if total_records > 0 else 0
                        
                        result = {
                            'total_records': total_records,
                            'complete_records': complete_records,
                            'completeness': completeness,
                            'latest_date': latest_date,
                            'unique_symbols': unique_symbols
                        }
                        
                        # Update cache
                        self._DATA_CACHE[cache_key] = (result.copy(), time.time())
                        return result
                
                return {}
        except Exception as e:
            logger.error(f"Error verifying database data: {str(e)}")
            return {}
        finally:
            self.connection_pool.return_connection(conn)
