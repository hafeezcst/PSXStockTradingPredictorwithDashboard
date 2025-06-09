from typing import Dict, List, Optional, Callable, Any, Union
import requests
from bs4 import BeautifulSoup
import re
import logging
import sqlite3
from datetime import datetime
import os
from tradingview_ta import TA_Handler, Interval
import time
import pandas as pd
import json
from dotenv import load_dotenv
from contextlib import closing, contextmanager
import sys
import numpy as np
from scipy import stats
import concurrent.futures
from functools import wraps
import traceback

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def retry(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator for retrying operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {str(e)}")
            
            logger.error(f"Operation failed after {max_retries} attempts: {last_exception}")
            raise last_exception
        return wrapper
    return decorator

@contextmanager
def db_connection(db_path: str):
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def safe_convert(value: Any, target_type: type, default: Any = None) -> Any:
    """Safely convert a value to the target type"""
    try:
        if value is None:
            return default
        return target_type(value)
    except (ValueError, TypeError):
        return default

def validate_required_fields(data: Dict, required_fields: List[str]) -> Dict:
    """Validate required fields in a dictionary"""
    validation = {
        'is_valid': True,
        'missing_fields': [],
        'issues': []
    }
    
    for field in required_fields:
        if field not in data or data[field] is None:
            validation['is_valid'] = False
            validation['missing_fields'].append(field)
            validation['issues'].append(f"Missing required field: {field}")
    
    return validation

def detect_outliers(data: List[float], threshold: float = 3.0) -> List[int]:
    """Detect outliers in a list of numbers using z-score"""
    if not data:
        return []
    
    z_scores = np.abs(stats.zscore(data))
    return [i for i, z in enumerate(z_scores) if z > threshold]

def calculate_moving_average(data: List[float], window: int) -> List[float]:
    """Calculate moving average for a list of numbers"""
    if not data or window <= 0:
        return []
    
    return pd.Series(data).rolling(window=window).mean().fillna(0).tolist()

def format_number(value: float, places: int = 2) -> float:
    """Format a number to specified decimal places"""
    try:
        return round(float(value), places)
    except (ValueError, TypeError):
        return 0.0

def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two numbers"""
    try:
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0

def safe_json_dumps(data: Any) -> str:
    """Safely convert data to JSON string"""
    try:
        return json.dumps(data, default=str)
    except Exception:
        return "{}"

def get_indicator_safely(indicators: Dict, key: str, default: Any = None) -> Any:
    """Safely get an indicator value from a dictionary"""
    try:
        return indicators.get(key, default)
    except (KeyError, AttributeError):
        return default

class FairValueCalculator:
    def __init__(self, db_path: str):
        """Initialize the FairValueCalculator with database path"""
        self.db_path = db_path
        self.dividend_db_path = os.path.join(os.path.dirname(db_path), 'PSX_Dividend_Schedule.db')
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.max_workers = 4  # Maximum number of parallel workers
        self.batch_size = 100  # Batch size for database operations
        self._init_database()
        
        # Initialize analysis cache
        self.analysis_cache = {}
        self._last_analysis_time = {}
        self._analysis_cooldown = 3600  # 1 hour cooldown between analyses
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize database with optimized settings"""
        try:
            with db_connection(self.db_path) as conn:
                conn.executescript("""
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA cache_size=10000;
                    PRAGMA temp_store=MEMORY;
                    PRAGMA foreign_keys=ON;
                """)
                self._create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    @retry(max_retries=3)
    def _fetch_data_from_tradingview(self, symbol: str) -> Dict:
        """Fetch technical analysis data from TradingView"""
        try:
            handler = TA_Handler(
                symbol=symbol,
                exchange="PSX",
                screener="pakistan",
                interval=Interval.INTERVAL_1_DAY
            )
            
            analysis = handler.get_analysis()
            
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'open': get_indicator_safely(analysis.indicators, 'open'),
                'high': get_indicator_safely(analysis.indicators, 'high'),
                'low': get_indicator_safely(analysis.indicators, 'low'),
                'close': get_indicator_safely(analysis.indicators, 'close'),
                'volume': get_indicator_safely(analysis.indicators, 'volume'),
                'rsi': get_indicator_safely(analysis.indicators, 'RSI'),
                'macd': get_indicator_safely(analysis.indicators, 'MACD.macd'),
                'macd_signal': get_indicator_safely(analysis.indicators, 'MACD.signal'),
                'macd_hist': get_indicator_safely(analysis.indicators, 'MACD.hist'),
                'sma_20': get_indicator_safely(analysis.indicators, 'SMA20'),
                'sma_50': get_indicator_safely(analysis.indicators, 'SMA50'),
                'sma_200': get_indicator_safely(analysis.indicators, 'SMA200'),
                'ema_20': get_indicator_safely(analysis.indicators, 'EMA20'),
                'ema_50': get_indicator_safely(analysis.indicators, 'EMA50'),
                'ema_200': get_indicator_safely(analysis.indicators, 'EMA200'),
                'bollinger_upper': get_indicator_safely(analysis.indicators, 'BB.upperband'),
                'bollinger_middle': get_indicator_safely(analysis.indicators, 'BB.middleband'),
                'bollinger_lower': get_indicator_safely(analysis.indicators, 'BB.lowerband'),
                'stoch_k': get_indicator_safely(analysis.indicators, 'Stoch.K'),
                'stoch_d': get_indicator_safely(analysis.indicators, 'Stoch.D'),
                'ichimoku_tenkan': get_indicator_safely(analysis.indicators, 'Ichimoku.Tenkan-sen'),
                'ichimoku_kijun': get_indicator_safely(analysis.indicators, 'Ichimoku.Kijun-sen'),
                'ichimoku_senkou_span_a': get_indicator_safely(analysis.indicators, 'Ichimoku.Senkou Span A'),
                'ichimoku_senkou_span_b': get_indicator_safely(analysis.indicators, 'Ichimoku.Senkou Span B'),
                'ichimoku_cloud_green': 1 if get_indicator_safely(analysis.indicators, 'Ichimoku.Senkou Span A', 0) > get_indicator_safely(analysis.indicators, 'Ichimoku.Senkou Span B', 0) else 0,
                'ichimoku_cloud_red': 1 if get_indicator_safely(analysis.indicators, 'Ichimoku.Senkou Span A', 0) < get_indicator_safely(analysis.indicators, 'Ichimoku.Senkou Span B', 0) else 0,
                'support_level': get_indicator_safely(analysis.indicators, 'Pivot.M.Classic.S3'),
                'resistance_level': get_indicator_safely(analysis.indicators, 'Pivot.M.Classic.R3'),
                'trend': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                'momentum': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                'volume_profile': 'HIGH' if get_indicator_safely(analysis.indicators, 'volume', 0) > get_indicator_safely(analysis.indicators, 'SMA20', 0) else 'LOW',
                'pattern': None,
                'signal': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                'target_price': None,
                'stop_loss': None,
                'position_size': None,
                'risk_score': None,
                'confidence_score': None
            }
            
            # Validate required fields
            validation = validate_required_fields(data, ['open', 'high', 'low', 'close', 'volume'])
            if not validation['is_valid']:
                logger.warning(f"Missing required fields for {symbol}: {validation['missing_fields']}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from TradingView for {symbol}: {e}")
            return None
    
    def save_tradingview_ta_data_to_db(self, symbol: str, data: Dict, conn: sqlite3.Connection):
        """Save TradingView TA data to database"""
        try:
            cursor = conn.cursor()
            
            # Check if data already exists
            cursor.execute("""
                SELECT * FROM tradingview_signals
                WHERE symbol = ? AND date = ?
            """, (symbol, data['date']))
            
            if cursor.fetchone():
                # Update existing data
                cursor.execute("""
                    UPDATE tradingview_signals
                    SET open = ?, high = ?, low = ?, close = ?, volume = ?,
                        rsi = ?, macd = ?, macd_signal = ?, macd_hist = ?,
                        sma_20 = ?, sma_50 = ?, sma_200 = ?,
                        ema_20 = ?, ema_50 = ?, ema_200 = ?,
                        bollinger_upper = ?, bollinger_middle = ?, bollinger_lower = ?,
                        stoch_k = ?, stoch_d = ?,
                        ichimoku_tenkan = ?, ichimoku_kijun = ?,
                        ichimoku_senkou_span_a = ?, ichimoku_senkou_span_b = ?,
                        ichimoku_cloud_green = ?, ichimoku_cloud_red = ?,
                        support_level = ?, resistance_level = ?,
                        trend = ?, momentum = ?, volume_profile = ?, pattern = ?,
                        signal = ?, target_price = ?, stop_loss = ?, position_size = ?,
                        risk_score = ?, confidence_score = ?
                    WHERE symbol = ? AND date = ?
                """, (
                    data['open'], data['high'], data['low'], data['close'], data['volume'],
                    data['rsi'], data['macd'], data['macd_signal'], data['macd_hist'],
                    data['sma_20'], data['sma_50'], data['sma_200'],
                    data['ema_20'], data['ema_50'], data['ema_200'],
                    data['bollinger_upper'], data['bollinger_middle'], data['bollinger_lower'],
                    data['stoch_k'], data['stoch_d'],
                    data['ichimoku_tenkan'], data['ichimoku_kijun'],
                    data['ichimoku_senkou_span_a'], data['ichimoku_senkou_span_b'],
                    data['ichimoku_cloud_green'], data['ichimoku_cloud_red'],
                    data['support_level'], data['resistance_level'],
                    data['trend'], data['momentum'], data['volume_profile'], data['pattern'],
                    data['signal'], data['target_price'], data['stop_loss'], data['position_size'],
                    data['risk_score'], data['confidence_score'],
                    symbol, data['date']
                ))
            else:
                # Insert new data
                values = (
                    symbol, data['date'], data['open'], data['high'], data['low'], data['close'], data['volume'],
                    data['rsi'], data['macd'], data['macd_signal'], data['macd_hist'],
                    data['sma_20'], data['sma_50'], data['sma_200'],
                    data['ema_20'], data['ema_50'], data['ema_200'],
                    data['bollinger_upper'], data['bollinger_middle'], data['bollinger_lower'],
                    data['stoch_k'], data['stoch_d'],
                    data['ichimoku_tenkan'], data['ichimoku_kijun'],
                    data['ichimoku_senkou_span_a'], data['ichimoku_senkou_span_b'],
                    data['ichimoku_cloud_green'], data['ichimoku_cloud_red'],
                    data['support_level'], data['resistance_level'],
                    data['trend'], data['momentum'], data['volume_profile'], data['pattern'],
                    data['signal'], data['target_price'], data['stop_loss'], data['position_size'],
                    data['risk_score'], data['confidence_score']
                )
                
                # Debug: Print values and SQL
                logger.info(f"Number of values: {len(values)}")
                logger.info(f"Values: {values}")
                
                sql = """
                    INSERT INTO tradingview_signals (
                        symbol, date, open, high, low, close, volume,
                        rsi, macd, macd_signal, macd_hist,
                        sma_20, sma_50, sma_200,
                        ema_20, ema_50, ema_200,
                        bollinger_upper, bollinger_middle, bollinger_lower,
                        stoch_k, stoch_d,
                        ichimoku_tenkan, ichimoku_kijun,
                        ichimoku_senkou_span_a, ichimoku_senkou_span_b,
                        ichimoku_cloud_green, ichimoku_cloud_red,
                        support_level, resistance_level,
                        trend, momentum, volume_profile, pattern,
                        signal, target_price, stop_loss, position_size,
                        risk_score, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                logger.info(f"SQL placeholders: {sql.count('?')}")
                cursor.execute(sql, values)
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error saving TradingView TA data for {symbol} to database: {e}")
            return False

    def fetch_psx_symbols(self) -> List[str]:
        """Fetch list of PSX symbols from Excel file"""
        try:
            # Read symbols from Excel file
            excel_path = 'src/data_processing/psxsymbols.xlsx'
            if not os.path.exists(excel_path):
                logger.error(f"Excel file not found at {excel_path}")
                return []
            
            # Read the Excel file
            df = pd.read_excel(excel_path)
            
            # Assuming the symbols are in a column named 'Symbol' or the first column
            symbol_column = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
            symbols = df[symbol_column].astype(str).tolist()
            
            # Clean and filter symbols
            symbols = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
            symbols = list(set(symbols))  # Remove duplicates
            
            logger.info(f"Successfully loaded {len(symbols)} symbols from Excel file")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching PSX symbols from Excel: {e}")
            return []

    def should_update_data(self, symbol: str) -> bool:
        """Check if data needs to be updated for a symbol with weekly timeframe logic"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the latest data for the symbol
            cursor.execute("""
                SELECT date, last_updated 
                FROM tradingview_ta 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                logger.info(f"No existing data found for {symbol}, will fetch new data")
                return True
                
            last_date = datetime.strptime(result[0], '%Y-%m-%d')
            last_updated = datetime.strptime(result[1], '%Y-%m-%d %H:%M:%S')
            current_time = datetime.now()
            
            # For weekly data, we only need to update once per week
            if last_date.isocalendar()[1] == current_time.isocalendar()[1]:
                logger.info(f"Current week's data exists for {symbol}, skipping update")
                return False
            
            # Check if the last update was within the last 24 hours
            if (current_time - last_updated).total_seconds() < 86400:  # 24 hours
                logger.info(f"Data for {symbol} was updated within the last 24 hours, skipping update")
                return False
            
            logger.info(f"Data for {symbol} needs update")
            return True
            
        except Exception as e:
            logger.error(f"Error checking update status for {symbol}: {str(e)}")
            return True

    def get_market_status(self) -> bool:
        """Check if the market is currently open with enhanced timezone handling"""
        try:
            current_time = datetime.now()
            
            # Check if it's a trading day (Monday to Friday)
            if current_time.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                logger.info("Market is closed: Weekend")
                return False
            
            # Convert current time to PKT (Pakistan Time)
            # Pakistan is UTC+5
            pkt_hour = current_time.hour
            pkt_minute = current_time.minute
            
            # Market hours in PKT:
            # Pre-market: 9:00 AM - 9:30 AM
            # Regular market: 9:30 AM - 3:30 PM
            # Post-market: 3:30 PM - 4:00 PM
            
            # Check if it's within market hours
            if (pkt_hour < 9 or 
                (pkt_hour == 9 and pkt_minute < 30) or 
                pkt_hour >= 15 or 
                (pkt_hour == 15 and pkt_minute > 30)):
                logger.info(f"Market is closed: Outside trading hours (Current PKT: {pkt_hour:02d}:{pkt_minute:02d})")
                return False
            
            # Check for market holidays (you can expand this list)
            holidays = [
                "2024-01-01",  # New Year's Day
                "2024-03-23",  # Pakistan Day
                "2024-05-01",  # Labour Day
                "2024-08-14",  # Independence Day
                "2024-09-06",  # Defence Day
                "2024-12-25",  # Christmas Day
            ]
            
            current_date = current_time.strftime('%Y-%m-%d')
            if current_date in holidays:
                logger.info(f"Market is closed: Holiday ({current_date})")
                return False
            
            logger.info(f"Market is open (Current PKT: {pkt_hour:02d}:{pkt_minute:02d})")
            return True
            
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False

    def read_psx_announcements(self) -> Dict:
        """Read and process PSX announcements from Excel file"""
        try:
            excel_path = 'data/announcements/PSX_Announcements.xlsx'
            if not os.path.exists(excel_path):
                logger.error(f"Announcements file not found at {excel_path}")
                return {}
            
            # Read the Excel file
            df = pd.read_excel(excel_path)
            
            # Process announcements
            announcements = {}
            for _, row in df.iterrows():
                symbol = row.get('Symbol', '').strip().upper()
                if not symbol:
                    continue
                
                if symbol not in announcements:
                    announcements[symbol] = []
                
                announcement = {
                    'date': row.get('Date', ''),
                    'title': row.get('Title', ''),
                    'link': row.get('Link', ''),
                    'type': row.get('Type', ''),
                    'impact': self._analyze_announcement_impact(row.get('Title', ''), row.get('Type', ''))
                }
                announcements[symbol].append(announcement)
            
            logger.info(f"Successfully loaded announcements for {len(announcements)} symbols")
            return announcements
            
        except Exception as e:
            logger.error(f"Error reading PSX announcements: {e}")
            return {}

    def _analyze_announcement_impact(self, title: str, announcement_type: str) -> float:
        """Analyze the potential impact of an announcement"""
        try:
            impact_score = 0.0
            
            # Convert to lowercase for case-insensitive matching
            title_lower = title.lower()
            type_lower = announcement_type.lower()
            
            # Financial Results Impact
            if 'financial result' in title_lower or 'quarterly report' in title_lower:
                if 'profit' in title_lower or 'increase' in title_lower:
                    impact_score += 0.3
                elif 'loss' in title_lower or 'decrease' in title_lower:
                    impact_score -= 0.3
            
            # Dividend Impact
            if 'dividend' in title_lower:
                if 'declare' in title_lower or 'announce' in title_lower:
                    impact_score += 0.2
                elif 'cancel' in title_lower or 'suspend' in title_lower:
                    impact_score -= 0.2
            
            # Corporate Actions Impact
            if 'right issue' in title_lower or 'bonus share' in title_lower:
                impact_score += 0.15
            elif 'merger' in title_lower or 'acquisition' in title_lower:
                impact_score += 0.2
            elif 'delisting' in title_lower:
                impact_score -= 0.3
            
            # Regulatory Impact
            if 'notice' in title_lower or 'compliance' in title_lower:
                impact_score -= 0.1
            
            # Type-based adjustments
            if 'positive' in type_lower:
                impact_score += 0.1
            elif 'negative' in type_lower:
                impact_score -= 0.1
            
            return impact_score
            
        except Exception as e:
            logger.error(f"Error analyzing announcement impact: {e}")
            return 0.0

    def analyze_financial_data(self, symbol: str) -> Dict:
        """Analyze financial data for a given symbol"""
        try:
            # Get financial data from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get latest financial data
            cursor.execute("""
                SELECT * FROM financial_reports 
                WHERE symbol = ? 
                ORDER BY report_date DESC 
                LIMIT 2
            """, (symbol,))
            
            financial_reports = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            latest_reports = []
            for report in financial_reports:
                latest_reports.append(dict(zip(columns, report)))
            
            conn.close()
            
            # Calculate financial metrics
            financial_analysis = {
                'symbol': symbol,
                'reports': latest_reports,
                'metrics': {}
            }
            
            if len(latest_reports) >= 2:
                current = latest_reports[0]
                previous = latest_reports[1]
                
                # Calculate growth rates and ratios
                financial_analysis['metrics'] = {
                    'eps_growth': self._calculate_percentage_change(
                        current.get('eps', 0),
                        previous.get('eps', 0)
                    ),
                    'revenue_growth': self._calculate_percentage_change(
                        current.get('revenue', 0),
                        previous.get('revenue', 0)
                    ),
                    'profit_margin': (current.get('net_income', 0) / current.get('revenue', 1)) * 100 if current.get('revenue', 0) != 0 else 0,
                    'debt_to_equity': current.get('total_debt', 0) / current.get('total_equity', 1) if current.get('total_equity', 0) != 0 else 0,
                    'current_ratio': current.get('current_assets', 0) / current.get('current_liabilities', 1) if current.get('current_liabilities', 0) != 0 else 0,
                    'roe': (current.get('net_income', 0) / current.get('total_equity', 1)) * 100 if current.get('total_equity', 0) != 0 else 0
                }
            
            return financial_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing financial data for {symbol}: {e}")
            return {'symbol': symbol, 'reports': [], 'metrics': {}}

    def analyze_with_ai(self, symbol: str, technical_data: Dict, financial_data: Dict) -> Dict:
        """Analyze stock data using AI to enhance signal generation with focus on investment perspective"""
        try:
            current_time = time.time()
            
            # Check cache first
            if symbol in self.ai_analysis_cache:
                cache_time = self._last_ai_call_time.get(symbol, 0)
                if current_time - cache_time < self._ai_call_cooldown:
                    logger.info(f"Using cached AI analysis for {symbol}")
                    return self.ai_analysis_cache[symbol]
            
            logger.info(f"Starting AI analysis for symbol: {symbol}")
            
            # Get financial announcements and reports
            announcements = self.read_psx_announcements()
            symbol_announcements = announcements.get(symbol, [])
            
            # Get dividend analysis
            dividend_analysis = self.analyze_dividend_data(symbol)
            
            # Get latest financial reports
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get latest financial data
            cursor.execute("""
                SELECT * FROM financial_reports 
                WHERE symbol = ? 
                ORDER BY report_date DESC 
                LIMIT 2
            """, (symbol,))
            
            financial_reports = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            latest_reports = []
            for report in financial_reports:
                latest_reports.append(dict(zip(columns, report)))
            
            conn.close()
            
            # Prepare data for AI analysis
            analysis_data = {
                'symbol': symbol,
                'technical': {
                    'price': {
                        'close': technical_data.get('close'),
                        'open': technical_data.get('open'),
                        'high': technical_data.get('high'),
                        'low': technical_data.get('low'),
                        'volume': technical_data.get('volume'),
                        'change_percent': technical_data.get('change_percent')
                    },
                    'indicators': {
                        'rsi': technical_data.get('rsi'),
                        'macd': technical_data.get('macd'),
                        'macd_signal': technical_data.get('macd_signal'),
                        'sma_20': technical_data.get('sma_20'),
                        'sma_50': technical_data.get('sma_50'),
                        'sma_200': technical_data.get('sma_200'),
                        'bb_upper': technical_data.get('bb_upper'),
                        'bb_lower': technical_data.get('bb_lower')
                    },
                    'signals': {
                        'trend_score': technical_data.get('trend_score'),
                        'momentum_score': technical_data.get('momentum_score'),
                        'volume_score': technical_data.get('volume_score'),
                        'volatility_score': technical_data.get('volatility_score')
                    }
                },
                'financial': {
                    'current': {
                        'eps_growth': financial_data.get('eps_growth'),
                        'revenue_growth': financial_data.get('revenue_growth'),
                        'profit_margin': financial_data.get('profit_margin'),
                        'debt_to_equity': financial_data.get('debt_to_equity'),
                        'current_ratio': financial_data.get('current_ratio'),
                        'roe': financial_data.get('roe')
                    },
                    'reports': latest_reports,
                    'announcements': symbol_announcements
                },
                'dividend': dividend_analysis
            }
            
            # Construct the prompt for AI analysis
            prompt = f"""Analyze the following stock data for {symbol} and provide a comprehensive investment analysis:

Technical Analysis:
- Current Price: {analysis_data['technical']['price']['close']}
- Price Change: {analysis_data['technical']['price']['change_percent']}%
- Volume: {analysis_data['technical']['price']['volume']}
- RSI: {analysis_data['technical']['indicators']['rsi']}
- MACD: {analysis_data['technical']['indicators']['macd']}
- MACD Signal: {analysis_data['technical']['indicators']['macd_signal']}
- Moving Averages:
  * SMA20: {analysis_data['technical']['indicators']['sma_20']}
  * SMA50: {analysis_data['technical']['indicators']['sma_50']}
  * SMA200: {analysis_data['technical']['indicators']['sma_200']}
- Bollinger Bands:
  * Upper: {analysis_data['technical']['indicators']['bb_upper']}
  * Lower: {analysis_data['technical']['indicators']['bb_lower']}
- Technical Scores:
  * Trend: {analysis_data['technical']['signals']['trend_score']}
  * Momentum: {analysis_data['technical']['signals']['momentum_score']}
  * Volume: {analysis_data['technical']['signals']['volume_score']}
  * Volatility: {analysis_data['technical']['signals']['volatility_score']}

Financial Analysis:
- EPS Growth: {analysis_data['financial']['current']['eps_growth']}%
- Revenue Growth: {analysis_data['financial']['current']['revenue_growth']}%
- Profit Margin: {analysis_data['financial']['current']['profit_margin']}%
- Debt-to-Equity: {analysis_data['financial']['current']['debt_to_equity']}
- Current Ratio: {analysis_data['financial']['current']['current_ratio']}
- ROE: {analysis_data['financial']['current']['roe']}%

Recent Announcements:
{json.dumps(analysis_data['financial']['announcements'], indent=2)}

Dividend Analysis:
{json.dumps(analysis_data['dividend'], indent=2)}

Please provide a detailed analysis including:
1. Company Overview
2. Financial Health
3. Investment Thesis
4. Valuation Analysis
5. Investment Recommendation
6. Monitoring Points
7. Confidence Score (0-1)
8. Fair Value Estimate
9. Target Price
10. Entry Range
11. Investment Horizon
12. Position Size Recommendation
13. DCF Value
14. Peer Comparison
15. Risk Assessment
16. Growth Catalysts
17. Management Quality
18. Corporate Governance
19. Dividend Analysis
20. Technical Analysis
21. Market Sentiment
22. Industry Analysis
23. Regulatory Analysis
24. Liquidity Analysis
25. Volatility Analysis

Format the response in clear sections with specific metrics and recommendations."""

            # Call AI model
            ai_analysis = self.call_ai_model(prompt)
            
            if ai_analysis:
                # Cache the analysis
                self.ai_analysis_cache[symbol] = ai_analysis
                self._last_ai_call_time[symbol] = current_time
                logger.info(f"Successfully integrated AI analysis for {symbol}")
                return ai_analysis
            else:
                logger.warning(f"No AI analysis returned for {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            return None

    def adjust_signal_with_ai(self, technical_analysis: Dict, financial_analysis: Dict) -> Dict:
        """Adjust trading signals using AI-based analysis"""
        try:
            # Get AI analysis
            symbol = technical_analysis.get('symbol', '')
            ai_analysis = self.analyze_with_ai(symbol, technical_analysis, financial_analysis)
            
            if ai_analysis:
                # Initialize adjusted analysis
                adjusted_analysis = technical_analysis.copy()
                
                # Calculate weighted scores
                technical_weight = 0.4
                financial_weight = 0.3
                ai_weight = 0.3
                
                # Adjust technical score based on AI insights
                if ai_analysis.get('confidence_score', 0) > 0.7:
                    # Strong AI confidence
                    ai_impact = 0.3
                elif ai_analysis.get('confidence_score', 0) > 0.5:
                    # Moderate AI confidence
                    ai_impact = 0.1
                elif ai_analysis.get('confidence_score', 0) < 0.3:
                    # Low AI confidence
                    ai_impact = -0.3
                elif ai_analysis.get('confidence_score', 0) < 0.5:
                    # Very low AI confidence
                    ai_impact = -0.1
                else:
                    ai_impact = 0
                
                # Calculate final score
                adjusted_score = (
                    technical_analysis['technical_score'] * technical_weight +
                    financial_analysis['financial_score'] * financial_weight +
                    (ai_impact * 100) * ai_weight  # Scale AI impact
                )
                
                # Determine if signal should be adjusted
                if (technical_analysis['signal_type'] in ['STRONG_SELL', 'SELL'] and 
                    (financial_analysis['financial_signal'] in ['STRONG_BUY', 'BUY'] or 
                     ai_impact > 0.2)):
                    # If technical is bearish but other indicators are bullish, move to neutral
                    if abs(technical_analysis['technical_score']) > 40:
                        adjusted_analysis['signal_type'] = 'NEUTRAL'
                        adjusted_analysis['signal_strength'] = 0.5
                        adjusted_analysis['analysis_summary'].append(
                            "Signal adjusted to NEUTRAL due to strong financial performance and AI analysis"
                        )
                
                elif (technical_analysis['signal_type'] in ['STRONG_BUY', 'BUY'] and 
                      (financial_analysis['financial_signal'] in ['STRONG_SELL', 'SELL'] or 
                       ai_impact < -0.2)):
                    # If technical is bullish but other indicators are bearish, reduce signal strength
                    if technical_analysis['signal_strength'] > 0.7:
                        adjusted_analysis['signal_strength'] *= 0.7
                        adjusted_analysis['analysis_summary'].append(
                            "Signal strength reduced due to poor financial performance and AI analysis"
                        )
                
                # Add AI analysis to summary
                adjusted_analysis['analysis_summary'].append("\nAI Analysis:")
                
                # Safely add AI analysis sections if they exist
                if ai_analysis.get('market_overview'):
                    adjusted_analysis['analysis_summary'].append(ai_analysis['market_overview'])
                if ai_analysis.get('technical_analysis'):
                    adjusted_analysis['analysis_summary'].append(ai_analysis['technical_analysis'])
                if ai_analysis.get('risk_assessment'):
                    adjusted_analysis['analysis_summary'].append(ai_analysis['risk_assessment'])
                if ai_analysis.get('trading_recommendation'):
                    adjusted_analysis['analysis_summary'].append(ai_analysis['trading_recommendation'])
                
                # Update confidence score
                adjusted_analysis['confidence_score'] = (
                    technical_analysis['confidence_score'] * technical_weight +
                    financial_analysis['confidence'] * financial_weight +
                    ai_analysis.get('confidence_score', 0) * ai_weight
                )
                
                # Add price targets and entry/exit points if they exist
                if ai_analysis.get('price_targets'):
                    adjusted_analysis['price_targets'] = ai_analysis['price_targets']
                if ai_analysis.get('entry_points'):
                    adjusted_analysis['entry_points'] = ai_analysis['entry_points']
                if ai_analysis.get('exit_points'):
                    adjusted_analysis['exit_points'] = ai_analysis['exit_points']
                
                return adjusted_analysis
            
            return technical_analysis
            
        except Exception as e:
            logger.error(f"Error adjusting signal with AI: {e}")
            return technical_analysis

    def analyze_stock_indicators(self, stock_data: Dict) -> Dict:
        """Analyze stock data using multiple technical indicators with enhanced AI integration"""
        try:
            symbol = stock_data['symbol']
            logger.info(f"Starting stock analysis for {symbol}")
            
            # Get previous analysis from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM tradingview_signals 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            """, (symbol,))
            
            previous_analysis = cursor.fetchone()
            conn.close()
            
            # Convert previous analysis to dict if exists
            if previous_analysis:
                columns = [description[0] for description in cursor.description]
                previous_analysis = dict(zip(columns, previous_analysis))
                logger.info(f"Found previous analysis for {symbol}")
            
            # Get last signal information from KMI30 database
            last_signal_info = self._get_last_signal_info(symbol)
            if last_signal_info:
                logger.info(f"Found last signal for {symbol}: {last_signal_info['last_signal_type']} "
                          f"at {last_signal_info['last_signal_price']} on {last_signal_info['last_signal_date']}")
            
            # Perform technical analysis
            analysis = self._perform_technical_analysis(stock_data, previous_analysis)
            logger.info(f"Completed technical analysis for {symbol}")
            
            # Add last signal information to analysis
            if last_signal_info:
                analysis['last_signal_date'] = last_signal_info.get('last_signal_date')
                analysis['last_signal_price'] = last_signal_info.get('last_signal_price')
                analysis['last_signal_type'] = last_signal_info.get('last_signal_type')
                
                # Calculate price change since last signal
                if analysis.get('close') and last_signal_info.get('last_signal_price'):
                    price_change = round(((analysis['close'] - last_signal_info['last_signal_price']) / 
                                  last_signal_info['last_signal_price'] * 100), 2)
                    analysis['price_change_since_last_signal'] = price_change
                    logger.info(f"Price change since last signal: {price_change:.2f}%")
            
            # Get financial analysis
            financial_analysis = self.analyze_financial_data(symbol)
            logger.info(f"Completed financial analysis for {symbol}")
            
            # Get AI analysis
            logger.info(f"Starting AI analysis integration for {symbol}")
            ai_analysis = self.analyze_with_ai(symbol, analysis, financial_analysis)
            
            if ai_analysis:
                # Integrate AI analysis
                analysis['ai_analysis'] = ai_analysis
                
                # Adjust signal based on AI insights
                analysis = self.adjust_signal_with_ai(analysis, financial_analysis)
                
                # Add AI metrics to analysis
                analysis['ai_score'] = ai_analysis.get('confidence_score', 0.0)
                analysis['ai_confidence'] = ai_analysis.get('confidence_score', 0.0)
                analysis['ai_pattern_recognition'] = ai_analysis.get('pattern_recognition', '')
                analysis['ai_signal_strength'] = ai_analysis.get('signal_strength', '')
                analysis['ai_risk_assessment'] = ai_analysis.get('risk_assessment', '')
                analysis['ai_recommendation'] = ai_analysis.get('recommendation', '')
                analysis['ai_price_targets'] = ai_analysis.get('price_targets', {})
                analysis['ai_entry_points'] = ai_analysis.get('entry_points', [])
                analysis['ai_exit_points'] = ai_analysis.get('exit_points', [])
                
                logger.info(f"Successfully integrated AI analysis for {symbol}")
                logger.info(f"Final AI Score: {analysis['ai_score']}")
                logger.info(f"Final AI Confidence: {analysis['ai_confidence']}")
            else:
                logger.warning(f"No AI analysis available for {symbol}")
            
            # Check for signal transitions and send notifications
            self.check_signal_transitions(symbol, analysis, previous_analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock indicators for {symbol}: {str(e)}")
            return None

    def _perform_technical_analysis(self, stock_data: Dict, previous_analysis: Dict = None) -> Dict:
        """Perform comprehensive technical analysis on stock data"""
        try:
            # Convert stock data to pandas Series for easier calculations
            close = pd.Series(stock_data['close'])
            high = pd.Series(stock_data['high'])
            low = pd.Series(stock_data['low'])
            volume = pd.Series(stock_data['volume'])
            
            # Calculate 52-week high
            high_52 = high.tail(252).max()
            
            # Initialize analysis dictionary with all required keys
            analysis = {
                'indicators': {},
                'patterns': {},
                'signals': {},
                'analyses': {},
                'indicators_used': [],  # Initialize this key
                'trend_score': 0,       # Initialize this key
                'momentum_score': 0,    # Initialize this key
                'volume_score': 0,      # Initialize this key
                'volatility_score': 0   # Initialize this key
            }
            
            # Calculate moving averages
            analysis['indicators']['sma_20'] = close.rolling(window=20).mean().iloc[-1]
            analysis['indicators']['sma_50'] = close.rolling(window=50).mean().iloc[-1]
            analysis['indicators']['sma_200'] = close.rolling(window=200).mean().iloc[-1]
            analysis['indicators']['ema_20'] = close.ewm(span=20, adjust=False).mean().iloc[-1]
            analysis['indicators']['ema_50'] = close.ewm(span=50, adjust=False).mean().iloc[-1]
            analysis['indicators']['ema_200'] = close.ewm(span=200, adjust=False).mean().iloc[-1]
            
            # Calculate Bollinger Bands
            analysis['indicators']['bollinger_middle'] = close.rolling(window=20).mean().iloc[-1]
            std = close.rolling(window=20).std().iloc[-1]
            analysis['indicators']['bollinger_upper'] = analysis['indicators']['bollinger_middle'] + (std * 2)
            analysis['indicators']['bollinger_lower'] = analysis['indicators']['bollinger_middle'] - (std * 2)
            
            # Calculate RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            analysis['indicators']['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Calculate MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            analysis['indicators']['macd'] = exp1.iloc[-1] - exp2.iloc[-1]
            analysis['indicators']['macd_signal'] = pd.Series(analysis['indicators']['macd']).ewm(span=9, adjust=False).mean().iloc[-1]
            analysis['indicators']['macd_hist'] = analysis['indicators']['macd'] - analysis['indicators']['macd_signal']
            
            # Calculate Stochastic Oscillator
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            analysis['indicators']['stoch_k'] = 100 * ((close.iloc[-1] - low_14.iloc[-1]) / (high_14.iloc[-1] - low_14.iloc[-1]))
            analysis['indicators']['stoch_d'] = pd.Series(analysis['indicators']['stoch_k']).rolling(window=3).mean().iloc[-1]
            
            # Calculate Ichimoku Cloud
            analysis['indicators']['ichimoku_tenkan'] = (high.rolling(window=9).max().iloc[-1] + low.rolling(window=9).min().iloc[-1]) / 2
            analysis['indicators']['ichimoku_kijun'] = (high.rolling(window=26).max().iloc[-1] + low.rolling(window=26).min().iloc[-1]) / 2
            analysis['indicators']['ichimoku_senkou_span_a'] = ((analysis['indicators']['ichimoku_tenkan'] + analysis['indicators']['ichimoku_kijun']) / 2)
            analysis['indicators']['ichimoku_senkou_span_b'] = ((high.rolling(window=52).max().iloc[-1] + low.rolling(window=52).min().iloc[-1]) / 2)
            analysis['indicators']['ichimoku_cloud_green'] = 1 if analysis['indicators']['ichimoku_senkou_span_a'] > analysis['indicators']['ichimoku_senkou_span_b'] else 0
            analysis['indicators']['ichimoku_cloud_red'] = 1 if analysis['indicators']['ichimoku_senkou_span_a'] < analysis['indicators']['ichimoku_senkou_span_b'] else 0
            
            # Calculate volume profile
            analysis['indicators']['volume_profile'] = volume.rolling(window=20).mean().iloc[-1]
            
            # Calculate support and resistance levels
            analysis['patterns']['support_resistance'] = self._calculate_support_resistance(close, high, low)
            
            # Identify patterns
            analysis['patterns']['chart_patterns'] = self._identify_patterns(close, high, low, volume)
            
            # Analyze trend
            self._analyze_trend(stock_data, analysis, previous_analysis)
            
            # Analyze momentum
            self._analyze_momentum(stock_data, analysis, previous_analysis)
            
            # Analyze volume
            self._analyze_volume(stock_data, analysis, previous_analysis)
            
            # Analyze volatility
            self._analyze_volatility(stock_data, analysis, previous_analysis)
            
            # Calculate final scores
            self._calculate_final_scores(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            raise

    def _calculate_support_resistance(self, close_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series) -> Dict:
        """Calculate support and resistance levels using multiple methods"""
        try:
            levels = {
                'support': [],
                'resistance': [],
                'pivot_points': {}
            }
            
            # Pivot Points
            high = high_prices.iloc[-1]
            low = low_prices.iloc[-1]
            close = close_prices.iloc[-1]
            
            # Classic Pivot Points
            pp = (high + low + close) / 3
            r1 = 2 * pp - low
            s1 = 2 * pp - high
            r2 = pp + (high - low)
            s2 = pp - (high - low)
            
            levels['pivot_points'] = {
                'pp': pp,
                'r1': r1,
                'r2': r2,
                's1': s1,
                's2': s2
            }
            
            # Fibonacci Retracement Levels
            price_range = high - low
            levels['fibonacci'] = {
                '0.236': high - price_range * 0.236,
                '0.382': high - price_range * 0.382,
                '0.5': high - price_range * 0.5,
                '0.618': high - price_range * 0.618,
                '0.786': high - price_range * 0.786
            }
            
            # Recent Highs and Lows
            recent_highs = high_prices.rolling(window=20).max().dropna()
            recent_lows = low_prices.rolling(window=20).min().dropna()
            
            # Cluster similar levels
            def cluster_levels(levels, threshold=0.02):
                if not levels:
                    return []
                levels = sorted(levels)
                clusters = []
                current_cluster = [levels[0]]
                
                for level in levels[1:]:
                    if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                        current_cluster.append(level)
                    else:
                        clusters.append(sum(current_cluster) / len(current_cluster))
                        current_cluster = [level]
                
                clusters.append(sum(current_cluster) / len(current_cluster))
                return clusters
            
            levels['support'] = cluster_levels(recent_lows.tolist())
            levels['resistance'] = cluster_levels(recent_highs.tolist())
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating support and resistance: {e}")
            return {}

    def _identify_patterns(self, close_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series, volume: pd.Series) -> Dict:
        """Identify common chart patterns"""
        try:
            patterns = {
                'candlestick': [],
                'chart': [],
                'volume': []
            }
            
            # Candlestick Patterns
            def is_doji(open_price, close_price, high, low):
                body = abs(close_price - open_price)
                total_range = high - low
                return body <= total_range * 0.1
            
            def is_hammer(open_price, close_price, high, low):
                body = abs(close_price - open_price)
                lower_shadow = min(open_price, close_price) - low
                upper_shadow = high - max(open_price, close_price)
                return lower_shadow > body * 2 and upper_shadow < body * 0.1
            
            def is_engulfing(open1, close1, open2, close2):
                return (close1 > open1 and close2 < open2 and close1 < open2 and open1 > close2) or \
                       (close1 < open1 and close2 > open2 and close1 > open2 and open1 < close2)
            
            # Check last few candles
            for i in range(len(close_prices) - 1):
                if is_doji(close_prices.iloc[i], close_prices.iloc[i+1], high_prices.iloc[i+1], low_prices.iloc[i+1]):
                    patterns['candlestick'].append('doji')
                if is_hammer(close_prices.iloc[i], close_prices.iloc[i+1], high_prices.iloc[i+1], low_prices.iloc[i+1]):
                    patterns['candlestick'].append('hammer')
                if is_engulfing(close_prices.iloc[i], close_prices.iloc[i+1], close_prices.iloc[i+2], close_prices.iloc[i+3]):
                    patterns['candlestick'].append('engulfing')
            
            # Chart Patterns
            def is_double_top(prices, threshold=0.02):
                peaks = []
                for i in range(1, len(prices) - 1):
                    if prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1]:
                        peaks.append(prices.iloc[i])
                if len(peaks) >= 2:
                    return abs(peaks[-1] - peaks[-2]) / peaks[-2] < threshold
                return False
            
            def is_double_bottom(prices, threshold=0.02):
                troughs = []
                for i in range(1, len(prices) - 1):
                    if prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                        troughs.append(prices.iloc[i])
                if len(troughs) >= 2:
                    return abs(troughs[-1] - troughs[-2]) / troughs[-2] < threshold
                return False
            
            if is_double_top(close_prices):
                patterns['chart'].append('double_top')
            if is_double_bottom(close_prices):
                patterns['chart'].append('double_bottom')
            
            # Volume Patterns
            def is_volume_spike(volumes, threshold=2):
                avg_volume = volumes.rolling(window=20).mean()
                return volumes.iloc[-1] > avg_volume.iloc[-1] * threshold
            
            if is_volume_spike(volume):
                patterns['volume'].append('volume_spike')
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return {}

    def _analyze_trend(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze trend indicators"""
        try:
            if all(x is not None for x in [stock_data['close'], stock_data['sma_20'], stock_data['sma_50'], stock_data['sma_200']]):
                analysis['indicators_used'].append('SMA')
                close = self._round_decimal(stock_data['close'])
                sma20 = self._round_decimal(stock_data['sma_20'])
                sma50 = self._round_decimal(stock_data['sma_50'])
                sma200 = self._round_decimal(stock_data['sma_200'])
                
                # Calculate price position relative to SMAs
                price_above_sma20 = self._calculate_percentage_change(close, sma20)
                price_above_sma50 = self._calculate_percentage_change(close, sma50)
                price_above_sma200 = self._calculate_percentage_change(close, sma200)
                
                trend_strength = 0
                
                # Golden Cross (SMA20 crosses above SMA50)
                if sma20 > sma50 and previous_analysis and previous_analysis.get('sma_20', 0) <= previous_analysis.get('sma_50', 0):
                    trend_strength += 15
                    analysis['analysis_summary'].append("Golden Cross detected: SMA20 crossed above SMA50")
                
                # Death Cross (SMA20 crosses below SMA50)
                elif sma20 < sma50 and previous_analysis and previous_analysis.get('sma_20', 0) >= previous_analysis.get('sma_50', 0):
                    trend_strength -= 15
                    analysis['analysis_summary'].append("Death Cross detected: SMA20 crossed below SMA50")
                
                # Strong uptrend conditions
                if close > sma20 > sma50 > sma200:
                    if price_above_sma20 > 5:
                        trend_strength += 25
                        analysis['analysis_summary'].append(f"Strong uptrend: Price {price_above_sma20:.2f}% above SMA20")
                    else:
                        trend_strength += 15
                        analysis['analysis_summary'].append(f"Moderate uptrend: Price {price_above_sma20:.2f}% above SMA20")
                # Strong downtrend conditions
                elif close < sma20 < sma50 < sma200:
                    if price_above_sma20 < -5:
                        trend_strength -= 25
                        analysis['analysis_summary'].append(f"Strong downtrend: Price {abs(price_above_sma20):.2f}% below SMA20")
                    else:
                        trend_strength -= 15
                        analysis['analysis_summary'].append(f"Moderate downtrend: Price {abs(price_above_sma20):.2f}% below SMA20")
                
                analysis['trend_score'] = self._round_decimal(trend_strength)
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")

    def _analyze_momentum(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze momentum indicators"""
        try:
            if all(x is not None for x in [stock_data['rsi'], stock_data['macd'], stock_data['macd_signal'], stock_data['ao']]):
                analysis['indicators_used'].extend(['RSI', 'MACD', 'AO'])
                rsi = self._round_decimal(stock_data['rsi'])
                macd = self._round_decimal(stock_data['macd'])
                macd_signal = self._round_decimal(stock_data['macd_signal'])
                ao = self._round_decimal(stock_data['ao'])
                
                momentum_strength = 0
                
                # RSI Analysis
                if rsi < 30:
                    momentum_strength += 15
                    analysis['analysis_summary'].append(f"Strong oversold: RSI at {rsi:.2f}")
                elif rsi < 40:
                    momentum_strength += 10
                    analysis['analysis_summary'].append(f"Moderately oversold: RSI at {rsi:.2f}")
                elif rsi > 70:
                    momentum_strength -= 15
                    analysis['analysis_summary'].append(f"Strong overbought: RSI at {rsi:.2f}")
                elif rsi > 60:
                    momentum_strength -= 10
                    analysis['analysis_summary'].append(f"Moderately overbought: RSI at {rsi:.2f}")
                
                # MACD Analysis
                macd_diff = self._round_decimal(macd - macd_signal)
                macd_diff_percent = self._calculate_percentage_change(macd, macd_signal) if macd_signal != 0 else 0
                
                if previous_analysis:
                    prev_macd = self._round_decimal(previous_analysis.get('macd', 0))
                    prev_macd_signal = self._round_decimal(previous_analysis.get('macd_signal', 0))
                    
                    if macd > macd_signal and prev_macd <= prev_macd_signal:
                        momentum_strength += 10
                        analysis['analysis_summary'].append("Bullish MACD crossover detected")
                    elif macd < macd_signal and prev_macd >= prev_macd_signal:
                        momentum_strength -= 10
                        analysis['analysis_summary'].append("Bearish MACD crossover detected")
                
                # Awesome Oscillator Analysis
                ao_abs = self._round_decimal(abs(ao))
                ao_threshold = 50
                
                if previous_analysis:
                    prev_ao = self._round_decimal(previous_analysis.get('ao', 0))
                    if ao > 0 and prev_ao <= 0:
                        momentum_strength += 5
                        analysis['analysis_summary'].append("Bullish AO crossover detected")
                    elif ao < 0 and prev_ao >= 0:
                        momentum_strength -= 5
                        analysis['analysis_summary'].append("Bearish AO crossover detected")
                
                if ao > ao_threshold:
                    momentum_strength += 5
                    analysis['analysis_summary'].append(f"Strong bullish AO: {ao:.2f}")
                elif ao > 0:
                    momentum_strength += 2
                    analysis['analysis_summary'].append(f"Moderate bullish AO: {ao:.2f}")
                elif ao < -ao_threshold:
                    momentum_strength -= 5
                    analysis['analysis_summary'].append(f"Strong bearish AO: {ao:.2f}")
                elif ao < 0:
                    momentum_strength -= 2
                    analysis['analysis_summary'].append(f"Moderate bearish AO: {ao:.2f}")
                
                analysis['momentum_score'] = self._round_decimal(momentum_strength)
                
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")

    def _analyze_volume(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze volume indicators"""
        try:
            if all(x is not None for x in [stock_data['volume'], stock_data['change']]):
                analysis['indicators_used'].append('Volume')
                volume = self._round_decimal(stock_data['volume'])
                change = self._round_decimal(stock_data['change'])
                change_percent = self._round_decimal(stock_data.get('change_percent', 0))
                
                volume_strength = 0
                
                # Volume analysis
                if volume > 2000000:  # High volume threshold
                    if change_percent > 5:
                        volume_strength = 20
                        analysis['analysis_summary'].append(f"Very high volume with strong price increase: {change_percent:.2f}%")
                    elif change_percent > 2:
                        volume_strength = 15
                        analysis['analysis_summary'].append(f"High volume with moderate price increase: {change_percent:.2f}%")
                    elif change_percent < -5:
                        volume_strength = -20
                        analysis['analysis_summary'].append(f"Very high volume with strong price decrease: {abs(change_percent):.2f}%")
                    elif change_percent < -2:
                        volume_strength = -15
                        analysis['analysis_summary'].append(f"High volume with moderate price decrease: {abs(change_percent):.2f}%")
                elif volume > 1000000:  # Moderate volume threshold
                    if change_percent > 2:
                        volume_strength = 10
                        analysis['analysis_summary'].append(f"Moderate volume with price increase: {change_percent:.2f}%")
                    elif change_percent < -2:
                        volume_strength = -10
                        analysis['analysis_summary'].append(f"Moderate volume with price decrease: {abs(change_percent):.2f}%")
                
                # Volume trend analysis
                if previous_analysis:
                    prev_volume = self._round_decimal(previous_analysis.get('volume', 0))
                    volume_change = self._calculate_percentage_change(volume, prev_volume)
                    
                    if volume_change > 50:  # 50% volume increase
                        volume_strength += 5
                        analysis['analysis_summary'].append(f"Significant volume increase detected: {volume_change:.2f}%")
                    elif volume_change < -50:  # 50% volume decrease
                        volume_strength -= 5
                        analysis['analysis_summary'].append(f"Significant volume decrease detected: {abs(volume_change):.2f}%")
                
                analysis['volume_score'] = self._round_decimal(volume_strength)
                
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")

    def _analyze_volatility(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze volatility indicators"""
        try:
            if all(x is not None for x in [stock_data['bb_upper'], stock_data['bb_lower'], stock_data['close']]):
                analysis['indicators_used'].append('Bollinger Bands')
                bb_upper = self._round_decimal(stock_data['bb_upper'])
                bb_lower = self._round_decimal(stock_data['bb_lower'])
                close = self._round_decimal(stock_data['close'])
                
                bb_range = self._round_decimal(bb_upper - bb_lower)
                volatility = self._round_decimal((bb_range / close * 100) if close != 0 else 0)
                price_position = self._round_decimal(((close - bb_lower) / bb_range * 100) if bb_range != 0 else 0)
                
                volatility_strength = 0
                
                # Volatility analysis
                if volatility > 15:
                    volatility_strength = -20
                    analysis['analysis_summary'].append(f"Very high volatility: BB range {volatility:.2f}%")
                elif volatility > 10:
                    volatility_strength = -15
                    analysis['analysis_summary'].append(f"High volatility: BB range {volatility:.2f}%")
                elif volatility < 5:
                    volatility_strength = 15
                    analysis['analysis_summary'].append(f"Low volatility: BB range {volatility:.2f}%")
                elif volatility < 8:
                    volatility_strength = 10
                    analysis['analysis_summary'].append(f"Moderate volatility: BB range {volatility:.2f}%")
                
                # Calculate support and resistance levels
                analysis['support_level'] = self._round_decimal(bb_lower)
                analysis['resistance_level'] = self._round_decimal(bb_upper)
                
                # Price position relative to BB
                if price_position > 80:
                    analysis['analysis_summary'].append(f"Price near upper BB: {price_position:.2f}% of range")
                    volatility_strength -= 5
                elif price_position < 20:
                    analysis['analysis_summary'].append(f"Price near lower BB: {price_position:.2f}% of range")
                    volatility_strength += 5
                
                analysis['volatility_score'] = self._round_decimal(volatility_strength)
                
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")

    def _calculate_final_scores(self, analysis: Dict):
        """Calculate final scores and determine signal"""
        try:
            # Calculate technical score
            analysis['technical_score'] = self._round_float(
                analysis['trend_score'] +
                analysis['momentum_score'] +
                analysis['volume_score'] +
                analysis['volatility_score']
            )
            
            # Determine signal type and strength
            if analysis['technical_score'] >= 70:
                analysis['signal_type'] = 'STRONG_BUY'
                analysis['signal_strength'] = self._round_float(min(analysis['technical_score'] / 70, 1.0))
            elif analysis['technical_score'] >= 40:
                analysis['signal_type'] = 'BUY'
                analysis['signal_strength'] = self._round_float(min(analysis['technical_score'] / 50, 0.8))
            elif analysis['technical_score'] <= -70:
                analysis['signal_type'] = 'STRONG_SELL'
                analysis['signal_strength'] = self._round_float(min(abs(analysis['technical_score']) / 70, 1.0))
            elif analysis['technical_score'] <= -40:
                analysis['signal_type'] = 'SELL'
                analysis['signal_strength'] = self._round_float(min(abs(analysis['technical_score']) / 50, 0.8))
            else:
                analysis['signal_type'] = 'NEUTRAL'
                analysis['signal_strength'] = 0.50
            
            # Calculate stop loss and take profit levels
            if analysis.get('close') is not None:
                current_price = self._round_float(analysis['close'])
                
                # Calculate stop loss based on volatility and support levels
                if analysis.get('bb_lower') is not None and analysis.get('bb_upper') is not None:
                    stop_loss_long = self._round_float(analysis['bb_lower'])
                    stop_loss_short = self._round_float(analysis['bb_upper'])
                elif analysis.get('sma_20') is not None:
                    stop_loss_long = self._round_float(analysis['sma_20'] * 0.95)
                    stop_loss_short = self._round_float(analysis['sma_20'] * 1.05)
                else:
                    stop_loss_long = self._round_float(current_price * 0.95)
                    stop_loss_short = self._round_float(current_price * 1.05)
                
                # Calculate take profit based on risk-reward ratio and volatility
                if analysis.get('volatility_score') is not None:
                    volatility_factor = self._round_float(1 + (abs(analysis['volatility_score']) / 100))
                else:
                    volatility_factor = 1.00
                
                # Set take profit levels based on signal type
                if analysis['signal_type'] in ['STRONG_BUY', 'BUY']:
                    analysis['stop_loss'] = stop_loss_long
                    # Take profit at 2:1 risk-reward ratio minimum
                    analysis['take_profit'] = current_price + (2 * (current_price - stop_loss_long)) * volatility_factor
                    logger.debug("Calculated levels for BUY signal")
                elif analysis['signal_type'] in ['STRONG_SELL', 'SELL']:
                    analysis['stop_loss'] = stop_loss_short
                    # Take profit at 2:1 risk-reward ratio minimum
                    analysis['take_profit'] = current_price - (2 * (stop_loss_short - current_price)) * volatility_factor
                    logger.debug("Calculated levels for SELL signal")
                else:
                    # For neutral signals, set both levels but with wider ranges
                    analysis['stop_loss'] = current_price * 0.90  # 10% below
                    analysis['take_profit'] = current_price * 1.10  # 10% above
                    logger.debug("Calculated levels for NEUTRAL signal")
                
                # Calculate risk-reward ratio
                if analysis['signal_type'] in ['STRONG_BUY', 'BUY']:
                    risk = current_price - analysis['stop_loss']
                    reward = analysis['take_profit'] - current_price
                elif analysis['signal_type'] in ['STRONG_SELL', 'SELL']:
                    risk = analysis['stop_loss'] - current_price
                    reward = current_price - analysis['take_profit']
                else:
                    risk = current_price - analysis['stop_loss']
                    reward = analysis['take_profit'] - current_price
                
                if risk != 0:
                    analysis['risk_reward_ratio'] = round(reward / risk, 2)
                    logger.debug(f"Calculated risk-reward ratio: {analysis['risk_reward_ratio']:.2f}")
                else:
                    analysis['risk_reward_ratio'] = 0.00
                    logger.debug("Risk is zero, setting risk-reward ratio to 0.00")
                
                logger.info(f"Calculated trading levels for {analysis.get('symbol', 'unknown')}:")
                logger.info(f"Stop Loss: {analysis['stop_loss']:.2f}")
                logger.info(f"Take Profit: {analysis['take_profit']:.2f}")
                logger.info(f"Risk-Reward Ratio: {analysis['risk_reward_ratio']:.2f}")
                
                # Round calculated values to 2 decimal places
                analysis['stop_loss'] = round(analysis['stop_loss'], 2)
                analysis['take_profit'] = round(analysis['take_profit'], 2)
                analysis['technical_score'] = round(analysis['technical_score'], 2)
                analysis['trend_score'] = round(analysis['trend_score'], 2)
                analysis['momentum_score'] = round(analysis['momentum_score'], 2)
                analysis['volume_score'] = round(analysis['volume_score'], 2)
                analysis['volatility_score'] = round(analysis['volatility_score'], 2)
                analysis['signal_strength'] = round(analysis['signal_strength'], 2)
                analysis['confidence_score'] = round(analysis['confidence_score'], 2)
            else:
                logger.warning(f"Missing close price for {analysis.get('symbol', 'unknown')}, cannot calculate trading levels")
                analysis['stop_loss'] = None
                analysis['take_profit'] = None
                analysis['risk_reward_ratio'] = None
            
        except Exception as e:
            logger.error(f"Error calculating final scores: {e}")
            logger.error(f"Error details: {str(e)}")
            # Set default values in case of error
            analysis['stop_loss'] = None
            analysis['take_profit'] = None
            analysis['risk_reward_ratio'] = None

    def send_telegram_notification(self, message: str):
        """Send notification to Telegram channel"""
        try:
            # Get Telegram bot token and channel ID from environment
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            channel_id = os.getenv('TELEGRAM_CHANNEL_ID')
            
            if not bot_token or not channel_id:
                logger.warning("Telegram credentials not found in environment variables")
                return False
            
            # Prepare the API URL
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            # Send the message
            payload = {
                "chat_id": channel_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("Telegram notification sent successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram notification: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False

    def check_signal_transitions(self, symbol: str, current_analysis: Dict, previous_analysis: Dict = None):
        """Check for signal transitions and send notifications"""
        try:
            if not previous_analysis:
                return
            
            # Get current and previous signals
            current_signal = current_analysis['signal_type']
            previous_signal = previous_analysis['signal_type']
            
            # Check for signal transitions
            if current_signal != previous_signal:
                # Prepare message
                message = f"🔄 <b>Signal Change Alert</b>\n\n"
                message += f"Symbol: <b>{symbol}</b>\n"
                message += f"Previous Signal: {previous_signal}\n"
                message += f"New Signal: {current_signal}\n"
                message += f"Current Price: {current_analysis.get('close', 'N/A')}\n"
                
                # Add confidence score
                message += f"Confidence: {current_analysis['confidence_score']:.2f}\n\n"
                
                # Add analysis summary
                if current_analysis['analysis_summary']:
                    message += "<b>Analysis Summary:</b>\n"
                    for summary in current_analysis['analysis_summary'][:3]:  # Show top 3 points
                        message += f"• {summary}\n"
                
                # Add price target if available
                if current_analysis.get('price_target'):
                    message += f"\nPrice Target: {current_analysis['price_target']}"
                
                # Send notification
                self.send_telegram_notification(message)
            
            # Enhanced notification for buy signals
            if current_signal in ['BUY', 'STRONG_BUY']:
                message = f"🎯 <b>Buy Signal Alert</b>\n\n"
                message += f"Symbol: <b>{symbol}</b>\n"
                message += f"Signal Type: {current_signal}\n"
                message += f"Current Price: {current_analysis.get('close', 'N/A')}\n"
                message += f"Signal Strength: {current_analysis.get('signal_strength', 'N/A')}\n"
                message += f"Confidence Score: {current_analysis.get('confidence_score', 'N/A')}\n"
                
                # Add last signal information if available
                if current_analysis.get('last_signal_date'):
                    message += f"\nLast Signal: {current_analysis['last_signal_type']}\n"
                    message += f"Last Signal Date: {current_analysis['last_signal_date']}\n"
                    message += f"Last Signal Price: {current_analysis['last_signal_price']}\n"
                    if current_analysis.get('price_change_since_last_signal'):
                        message += f"Price Change: {current_analysis['price_change_since_last_signal']:.2f}%\n"
                message += "\n"
                
                # Add technical indicators
                message += "<b>Technical Indicators:</b>\n"
                if current_analysis.get('rsi'):
                    message += f"• RSI: {current_analysis['rsi']:.2f}\n"
                if current_analysis.get('macd'):
                    message += f"• MACD: {current_analysis['macd']:.2f}\n"
                if current_analysis.get('macd_signal'):
                    message += f"• MACD Signal: {current_analysis['macd_signal']:.2f}\n"
                
                # Add moving averages
                message += "\n<b>Moving Averages:</b>\n"
                if current_analysis.get('sma_20'):
                    message += f"• SMA20: {current_analysis['sma_20']:.2f}\n"
                if current_analysis.get('sma_50'):
                    message += f"• SMA50: {current_analysis['sma_50']:.2f}\n"
                if current_analysis.get('sma_200'):
                    message += f"• SMA200: {current_analysis['sma_200']:.2f}\n"
                
                # Add trading levels
                message += "\n<b>Trading Levels:</b>\n"
                if current_analysis.get('support_level'):
                    message += f"• Support: {current_analysis['support_level']:.2f}\n"
                if current_analysis.get('resistance_level'):
                    message += f"• Resistance: {current_analysis['resistance_level']:.2f}\n"
                if current_analysis.get('stop_loss'):
                    message += f"• Stop Loss: {current_analysis['stop_loss']:.2f}\n"
                if current_analysis.get('take_profit'):
                    message += f"• Take Profit: {current_analysis['take_profit']:.2f}\n"
                if current_analysis.get('risk_reward_ratio'):
                    message += f"• Risk/Reward: {current_analysis['risk_reward_ratio']:.2f}\n"
                
                # Add AI analysis if available
                if current_analysis.get('ai_analysis'):
                    ai_data = current_analysis['ai_analysis']
                    message += "\n<b>AI Analysis:</b>\n"
                    if ai_data.get('confidence_score'):
                        message += f"• AI Confidence: {ai_data['confidence_score']:.2f}\n"
                    if ai_data.get('recommendation'):
                        message += f"• AI Recommendation: {ai_data['recommendation']}\n"
                    if ai_data.get('price_targets'):
                        message += f"• AI Price Targets: {json.dumps(ai_data['price_targets'])}\n"
                
                # Add analysis summary
                if current_analysis.get('analysis_summary'):
                    message += "\n<b>Key Points:</b>\n"
                    for summary in current_analysis['analysis_summary'][:5]:  # Show top 5 points
                        message += f"• {summary}\n"
                
                # Send notification
                self.send_telegram_notification(message)
            
            # Check for profit-taking opportunities
            if current_signal in ['BUY', 'STRONG_BUY'] and previous_signal in ['BUY', 'STRONG_BUY']:
                current_price = current_analysis.get('close')
                previous_price = previous_analysis.get('close')
                
                if current_price and previous_price:
                    price_change = ((current_price - previous_price) / previous_price) * 100
                    
                    # If price increased by more than 5%, suggest profit taking
                    if price_change > 5:
                        message = f"💰 <b>Profit Taking Alert</b>\n\n"
                        message += f"Symbol: <b>{symbol}</b>\n"
                        message += f"Current Signal: {current_signal}\n"
                        message += f"Price Change: +{price_change:.2f}%\n"
                        message += f"Current Price: {current_price}\n"
                        message += f"Previous Price: {previous_price}\n\n"
                        
                        # Add technical indicators
                        if current_analysis.get('rsi'):
                            message += f"RSI: {current_analysis['rsi']:.2f}\n"
                        if current_analysis.get('macd'):
                            message += f"MACD: {current_analysis['macd']:.2f}\n"
                        
                        # Add support and resistance
                        if current_analysis.get('support_level') and current_analysis.get('resistance_level'):
                            message += f"\nSupport: {current_analysis['support_level']:.2f}\n"
                            message += f"Resistance: {current_analysis['resistance_level']:.2f}"
                        
                        # Send notification
                        self.send_telegram_notification(message)
            
        except Exception as e:
            logger.error(f"Error checking signal transitions for {symbol}: {e}")

    def save_analysis_to_db(self, symbol: str, analysis: Dict):
        """Save stock analysis to the database with enhanced AI data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Prepare AI-related fields
            ai_data = analysis.get('ai_analysis', {})
            
            # Convert dictionary values to JSON strings with proper error handling
            def safe_json_dumps(value, default=None):
                try:
                    if isinstance(value, (dict, list)):
                        return json.dumps(value)
                    return str(value) if value is not None else default
                except Exception as e:
                    logger.warning(f"Error converting value to JSON: {e}")
                    return default
            
            # Convert all complex data types to JSON strings
            ai_price_targets = safe_json_dumps(ai_data.get('price_targets', {}), '{}')
            ai_entry_points = safe_json_dumps(ai_data.get('entry_points', []), '[]')
            ai_exit_points = safe_json_dumps(ai_data.get('exit_points', []), '[]')
            ai_pattern_recognition = safe_json_dumps(ai_data.get('pattern_recognition', ''), '')
            ai_signal_strength = safe_json_dumps(ai_data.get('signal_strength', ''), '')
            ai_risk_assessment = safe_json_dumps(ai_data.get('risk_assessment', {}), '{}')
            ai_recommendation = safe_json_dumps(ai_data.get('recommendation', ''), '')
            
            # Convert analysis summary to string if it's a list
            analysis_summary = analysis.get('analysis_summary', [])
            if isinstance(analysis_summary, list):
                analysis_summary = '|'.join(str(item) for item in analysis_summary)
            elif isinstance(analysis_summary, dict):
                analysis_summary = safe_json_dumps(analysis_summary, '')
            
            # Convert indicators used to string if it's a list
            indicators_used = analysis.get('indicators_used', [])
            if isinstance(indicators_used, list):
                indicators_used = '|'.join(str(item) for item in indicators_used)
            elif isinstance(indicators_used, dict):
                indicators_used = safe_json_dumps(indicators_used, '')
            
            # Ensure all values are strings or numbers
            def safe_convert(value):
                if isinstance(value, (dict, list)):
                    return safe_json_dumps(value, '')
                return value
            
            cursor.execute('''
            INSERT OR REPLACE INTO tradingview_signals
            (symbol, date, signal_type, signal_strength, confidence_score,
             technical_score, trend_score, momentum_score, volume_score,
             volatility_score, support_level, resistance_level, stop_loss,
             take_profit, risk_reward_ratio, analysis_summary, indicators_used,
             last_updated, ai_score, ai_confidence, ai_pattern_recognition,
             ai_signal_strength, ai_risk_assessment, ai_recommendation,
             ai_price_targets, ai_entry_points, ai_exit_points, ai_analysis_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now().strftime('%Y-%m-%d'),
                safe_convert(analysis.get('signal_type')),
                safe_convert(analysis.get('signal_strength')),
                safe_convert(analysis.get('confidence_score')),
                safe_convert(analysis.get('technical_score')),
                safe_convert(analysis.get('trend_score')),
                safe_convert(analysis.get('momentum_score')),
                safe_convert(analysis.get('volume_score')),
                safe_convert(analysis.get('volatility_score')),
                safe_convert(analysis.get('support_level')),
                safe_convert(analysis.get('resistance_level')),
                safe_convert(analysis.get('stop_loss')),
                safe_convert(analysis.get('take_profit')),
                safe_convert(analysis.get('risk_reward_ratio')),
                analysis_summary,
                indicators_used,
                current_time,
                safe_convert(ai_data.get('confidence_score', 0.0)),
                safe_convert(ai_data.get('confidence_score', 0.0)),
                ai_pattern_recognition,
                ai_signal_strength,
                ai_risk_assessment,
                ai_recommendation,
                ai_price_targets,
                ai_entry_points,
                ai_exit_points,
                current_time
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Successfully saved analysis for {symbol} to database with AI data")
            
        except Exception as e:
            logger.error(f"Error saving analysis to database: {e}")
            if 'conn' in locals():
                conn.close()

    def fetch_tradingview_ta_data(self, symbol: str) -> Dict:
        """Fetch data using tradingview_ta library with weekly timeframe and optimized database operations"""
        try:
            # Fetch data from tradingview_ta
            data = self._fetch_data_from_tradingview(symbol)
            if not data:
                return None
            
            # Save data to database
            conn = sqlite3.connect(self.db_path)
            save_result = self.save_tradingview_ta_data_to_db(symbol, data, conn)
            conn.close()
            
            if not save_result:
                logger.error(f"Failed to save data for {symbol} to database")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching TradingView TA data for {symbol}: {e}")
            return None

    def analyze_database(self):
        """Analyze the database for null values and data quality"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total number of records
            cursor.execute("SELECT COUNT(*) FROM tradingview_signals")
            total_records = cursor.fetchone()[0]
            
            logger.info(f"Total records in database: {total_records}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error analyzing database: {e}")
            if 'conn' in locals():
                conn.close()

    def analyze_stock_signals(self):
        """Analyze stock data and generate AI-based signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the latest data for each symbol
            cursor.execute("""
                WITH latest_dates AS (
                    SELECT symbol, MAX(date) as max_date
                    FROM tradingview_signals
                    GROUP BY symbol
                )
                SELECT t.*
                FROM tradingview_signals t
                JOIN latest_dates ld ON t.symbol = ld.symbol AND t.date = ld.max_date
                ORDER BY t.symbol
            """)
            
            stocks_data = cursor.fetchall()
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries for easier processing
            stocks = []
            for row in stocks_data:
                stock = dict(zip(columns, row))
                stocks.append(stock)
            
            # Analyze each stock
            signals = []
            for stock in stocks:
                signal = self._generate_stock_signal(stock)
                signals.append(signal)
            
            # Save signals to database
            self._save_signals_to_db(signals)
            
            conn.close()
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing stock signals: {e}")
            if 'conn' in locals():
                conn.close()
            return []

    def _generate_stock_signal(self, stock):
        """Generate trading signal for a stock based on technical and financial analysis"""
        try:
            symbol = stock['symbol']
            
            # Get technical analysis
            technical_analysis = self.analyze_stock_indicators(stock)
            
            # Get financial analysis
            financial_analysis = self.analyze_financial_data(symbol)
            
            # Calculate final signal
            signal = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'technical_score': technical_analysis.get('final_score', 0),
                'financial_score': self._calculate_financial_score(financial_analysis),
                'recommendation': self._generate_recommendation(
                    technical_analysis.get('final_score', 0),
                    self._calculate_financial_score(financial_analysis)
                ),
                'confidence': self._calculate_confidence(
                    technical_analysis,
                    financial_analysis
                ),
                'analysis': {
                    'technical': technical_analysis,
                    'financial': financial_analysis
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {stock.get('symbol', 'unknown')}: {e}")
            return None

    def _calculate_financial_score(self, financial_analysis: Dict) -> float:
        """Calculate financial score based on financial metrics"""
        try:
            metrics = financial_analysis.get('metrics', {})
            score = 0.0
            weights = {
                'eps_growth': 0.2,
                'revenue_growth': 0.2,
                'profit_margin': 0.2,
                'debt_to_equity': 0.15,
                'current_ratio': 0.15,
                'roe': 0.1
            }
            
            # EPS Growth
            if metrics.get('eps_growth') is not None:
                eps_growth = metrics['eps_growth']
                if eps_growth > 20:
                    score += weights['eps_growth']
                elif eps_growth > 10:
                    score += weights['eps_growth'] * 0.8
                elif eps_growth > 0:
                    score += weights['eps_growth'] * 0.5
                else:
                    score += weights['eps_growth'] * 0.2
            
            # Revenue Growth
            if metrics.get('revenue_growth') is not None:
                revenue_growth = metrics['revenue_growth']
                if revenue_growth > 15:
                    score += weights['revenue_growth']
                elif revenue_growth > 10:
                    score += weights['revenue_growth'] * 0.8
                elif revenue_growth > 5:
                    score += weights['revenue_growth'] * 0.5
                else:
                    score += weights['revenue_growth'] * 0.2
            
            # Profit Margin
            if metrics.get('profit_margin') is not None:
                profit_margin = metrics['profit_margin']
                if profit_margin > 20:
                    score += weights['profit_margin']
                elif profit_margin > 15:
                    score += weights['profit_margin'] * 0.8
                elif profit_margin > 10:
                    score += weights['profit_margin'] * 0.5
                else:
                    score += weights['profit_margin'] * 0.2
            
            # Debt to Equity
            if metrics.get('debt_to_equity') is not None:
                debt_to_equity = metrics['debt_to_equity']
                if debt_to_equity < 0.5:
                    score += weights['debt_to_equity']
                elif debt_to_equity < 1:
                    score += weights['debt_to_equity'] * 0.8
                elif debt_to_equity < 1.5:
                    score += weights['debt_to_equity'] * 0.5
                else:
                    score += weights['debt_to_equity'] * 0.2
            
            # Current Ratio
            if metrics.get('current_ratio') is not None:
                current_ratio = metrics['current_ratio']
                if current_ratio > 2:
                    score += weights['current_ratio']
                elif current_ratio > 1.5:
                    score += weights['current_ratio'] * 0.8
                elif current_ratio > 1:
                    score += weights['current_ratio'] * 0.5
                else:
                    score += weights['current_ratio'] * 0.2
            
            # ROE
            if metrics.get('roe') is not None:
                roe = metrics['roe']
                if roe > 20:
                    score += weights['roe']
                elif roe > 15:
                    score += weights['roe'] * 0.8
                elif roe > 10:
                    score += weights['roe'] * 0.5
                else:
                    score += weights['roe'] * 0.2
            
            return min(score, 1.0)  # Normalize to 0-1 range
            
        except Exception as e:
            logger.error(f"Error calculating financial score: {e}")
            return 0.5  # Return neutral score on error

    def _generate_recommendation(self, technical_score: float, financial_score: float) -> str:
        """Generate trading recommendation based on technical and financial scores"""
        try:
            # Calculate weighted average score
            weighted_score = (technical_score * 0.6) + (financial_score * 0.4)
            
            # Generate recommendation based on weighted score
            if weighted_score >= 0.8:
                return "STRONG_BUY"
            elif weighted_score >= 0.6:
                return "BUY"
            elif weighted_score >= 0.4:
                return "HOLD"
            elif weighted_score >= 0.2:
                return "SELL"
            else:
                return "STRONG_SELL"
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "HOLD"  # Return neutral recommendation on error

    def _calculate_confidence(self, technical_analysis: Dict, financial_analysis: Dict) -> float:
        """Calculate confidence score for the analysis"""
        try:
            # Calculate confidence based on data quality and completeness
            technical_confidence = 0.0
            financial_confidence = 0.0
            
            # Technical confidence
            if technical_analysis:
                required_indicators = ['rsi', 'macd', 'sma_20', 'sma_50', 'sma_200']
                available_indicators = sum(1 for ind in required_indicators if technical_analysis.get(ind) is not None)
                technical_confidence = available_indicators / len(required_indicators)
            
            # Financial confidence
            if financial_analysis:
                required_metrics = ['eps_growth', 'revenue_growth', 'profit_margin', 'debt_to_equity', 'current_ratio', 'roe']
                available_metrics = sum(1 for metric in required_metrics if financial_analysis.get('metrics', {}).get(metric) is not None)
                financial_confidence = available_metrics / len(required_metrics)
            
            # Calculate weighted average confidence
            confidence = (technical_confidence * 0.6) + (financial_confidence * 0.4)
            
            return min(confidence, 1.0)  # Normalize to 0-1 range
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Return neutral confidence on error

    def _save_signals_to_db(self, signals):
        """Save generated signals to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create signals table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_signals (
                symbol TEXT,
                date TEXT,
                signal TEXT,
                confidence REAL,
                score REAL,
                reasons TEXT,
                PRIMARY KEY (symbol, date)
            )
            ''')
            
            # Insert signals
            for signal in signals:
                if signal:
                    cursor.execute('''
                    INSERT OR REPLACE INTO stock_signals
                    (symbol, date, signal, confidence, score, reasons)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        signal['symbol'],
                        signal['date'],
                        signal['signal'],
                        signal['confidence'],
                        signal['score'],
                        '|'.join(signal['reasons'])
                    ))
            
            conn.commit()
            conn.close()
            logger.info(f"Successfully saved {len(signals)} signals to database")
            
        except Exception as e:
            logger.error(f"Error saving signals to database: {e}")
            if 'conn' in locals():
                conn.close()

    def verify_database_data(self):
        """Verify the quality and completeness of data in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total number of records
            cursor.execute("SELECT COUNT(*) FROM tradingview_signals")
            total_records = cursor.fetchone()[0]
            
            # Get count of records with complete data
            cursor.execute("""
                SELECT COUNT(*) FROM tradingview_signals 
                WHERE rsi IS NOT NULL 
                AND macd IS NOT NULL 
                AND sma_20 IS NOT NULL 
                AND ema_20 IS NOT NULL 
                AND close IS NOT NULL
            """)
            complete_records = cursor.fetchone()[0]
            
            # Get latest date in database
            cursor.execute("SELECT MAX(date) FROM tradingview_ta")
            latest_date = cursor.fetchone()[0]
            
            # Get unique symbols count
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM tradingview_ta")
            unique_symbols = cursor.fetchone()[0]
            
            # Get data completeness percentage
            completeness = (complete_records / total_records * 100) if total_records > 0 else 0
            
            logger.info(f"Database Verification Report:")
            logger.info(f"Total records: {total_records}")
            logger.info(f"Complete records: {complete_records} ({completeness:.2f}%)")
            logger.info(f"Latest data date: {latest_date}")
            logger.info(f"Unique symbols: {unique_symbols}")
            
            # Check for any symbols with missing data
            cursor.execute("""
                SELECT symbol, COUNT(*) as total_records,
                       SUM(CASE WHEN rsi IS NULL THEN 1 ELSE 0 END) as missing_rsi,
                       SUM(CASE WHEN macd IS NULL THEN 1 ELSE 0 END) as missing_macd,
                       SUM(CASE WHEN sma_20 IS NULL THEN 1 ELSE 0 END) as missing_sma20
                FROM tradingview_ta
                GROUP BY symbol
                HAVING missing_rsi > 0 OR missing_macd > 0 OR missing_sma20 > 0
            """)
            
            incomplete_symbols = cursor.fetchall()
            if incomplete_symbols:
                logger.warning(f"Found {len(incomplete_symbols)} symbols with incomplete data")
                for symbol_data in incomplete_symbols:
                    symbol, total, missing_rsi, missing_macd, missing_sma20 = symbol_data
                    logger.warning(f"Symbol {symbol}: Missing RSI: {missing_rsi}, MACD: {missing_macd}, SMA20: {missing_sma20}")
            
            conn.close()
            return {
                'total_records': total_records,
                'complete_records': complete_records,
                'completeness': completeness,
                'latest_date': latest_date,
                'unique_symbols': unique_symbols,
                'incomplete_symbols': len(incomplete_symbols) if incomplete_symbols else 0
            }
            
        except Exception as e:
            logger.error(f"Error verifying database data: {e}")
            if 'conn' in locals():
                conn.close()
            return None

    def analyze_dividend_data(self, symbol: str) -> Dict:
        """Analyze dividend data for a symbol"""
        try:
            conn = sqlite3.connect(self.dividend_db_path)
            cursor = conn.cursor()
            
            # Get current dividend schedule
            cursor.execute("""
                SELECT * FROM dividend_schedule 
                WHERE symbol = ? 
                ORDER BY bc_from DESC 
                LIMIT 1
            """, (symbol,))
            
            current_dividend = cursor.fetchone()
            
            # Get dividend statistics
            cursor.execute("""
                SELECT * FROM dividend_statistics 
                WHERE symbol = ?
            """, (symbol,))
            
            dividend_stats = cursor.fetchone()
            
            # Get historical dividends
            cursor.execute("""
                SELECT * FROM dividend_history 
                WHERE symbol = ? 
                ORDER BY bc_from DESC 
                LIMIT 5
            """, (symbol,))
            
            historical_dividends = cursor.fetchall()
            
            # Get column names
            schedule_columns = [description[0] for description in cursor.description]
            
            # Process the data
            dividend_analysis = {
                'current_dividend': None,
                'dividend_stats': None,
                'historical_dividends': [],
                'dividend_yield': None,
                'dividend_growth': None,
                'payout_ratio': None,
                'dividend_sustainability': None
            }
            
            if current_dividend:
                current_dividend_dict = dict(zip(schedule_columns, current_dividend))
                dividend_analysis['current_dividend'] = current_dividend_dict
                
                # Calculate dividend yield if we have last close price
                if current_dividend_dict.get('last_close') and current_dividend_dict.get('dividend_amount'):
                    dividend_analysis['dividend_yield'] = (
                        current_dividend_dict['dividend_amount'] / current_dividend_dict['last_close'] * 100
                    )
            
            if dividend_stats:
                stats_columns = [description[0] for description in cursor.description]
                dividend_analysis['dividend_stats'] = dict(zip(stats_columns, dividend_stats))
            
            if historical_dividends:
                for dividend in historical_dividends:
                    dividend_analysis['historical_dividends'].append(
                        dict(zip(schedule_columns, dividend))
                    )
                
                # Calculate dividend growth if we have enough historical data
                if len(historical_dividends) >= 2:
                    current_div = historical_dividends[0][schedule_columns.index('dividend_amount')]
                    previous_div = historical_dividends[1][schedule_columns.index('dividend_amount')]
                    if previous_div and previous_div != 0:
                        dividend_analysis['dividend_growth'] = (
                            (current_div - previous_div) / previous_div * 100
                        )
            
            # Calculate dividend sustainability
            if dividend_analysis['dividend_yield'] is not None:
                if dividend_analysis['dividend_yield'] < 2:
                    dividend_analysis['dividend_sustainability'] = 'High'
                elif dividend_analysis['dividend_yield'] < 4:
                    dividend_analysis['dividend_sustainability'] = 'Moderate'
                else:
                    dividend_analysis['dividend_sustainability'] = 'Low'
            
            conn.close()
            return dividend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing dividend data for {symbol}: {e}")
            if 'conn' in locals():
                conn.close()
            return None

    def _retry_with_backoff(self, func, max_retries=3, initial_delay=1, max_delay=32):
        """Helper method to retry operations with exponential backoff"""
        delay = initial_delay
        last_exception = None
        
        for retry in range(max_retries):
            try:
                return func()
            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Request timed out (attempt {retry + 1}/{max_retries}). Retrying in {delay} seconds...")
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Connection error (attempt {retry + 1}/{max_retries}). Retrying in {delay} seconds...")
            except Exception as e:
                last_exception = e
                logger.warning(f"Error during API call (attempt {retry + 1}/{max_retries}): {str(e)}. Retrying in {delay} seconds...")
            
            time.sleep(delay)
            delay = min(delay * 2, max_delay)  # Exponential backoff with max delay
        
        logger.error(f"Failed after {max_retries} retries. Last error: {str(last_exception)}")
        return None

    def _validate_api_key(self) -> bool:
        """Validate the DeepSeek API key with a simple test call"""
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                logger.warning("DeepSeek API key not found in environment variables")
                return False
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Test API key validation"},
                    {"role": "user", "content": "Test"}
                ],
                "max_tokens": 10,
                "temperature": 0.7
            }
            
            def test_api_call():
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return True
                elif response.status_code == 401:
                    logger.error("Invalid API key or authentication failed")
                    return False
                elif response.status_code == 429:
                    raise requests.exceptions.RequestException("Rate limit exceeded")
                else:
                    raise requests.exceptions.RequestException(f"API request failed with status code {response.status_code}")
            
            # Try the test call with retries
            result = self._retry_with_backoff(test_api_call, max_retries=2, initial_delay=1)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return False
    def call_ai_model(self, prompt: str) -> Dict:
        """Call AI model for analysis with improved error handling and retries"""
        try:
            # Validate API key first
            if not self._validate_api_key():
                logger.error("Failed to validate DeepSeek API key")
                return None
            
            # Get API key from environment
            api_key = os.getenv('DEEPSEEK_API_KEY')
            logger.info("DeepSeek API key validated, proceeding with API call")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Enhanced prompt with structured sections and specific metrics
            enhanced_prompt = f"""Analyze the following stock data and provide a comprehensive investment analysis in a structured format:

{prompt}

Please provide your analysis in the following structured format with specific metrics and values:

1. COMPANY OVERVIEW
- Company Description: [text]
- Market Position: [text]
- Key Business Segments: [list]
- Competitive Advantages: [list]

2. FINANCIAL HEALTH
- Revenue Growth: [percentage]
- Profit Margin: [percentage]
- Debt-to-Equity: [ratio]
- Current Ratio: [ratio]
- ROE: [percentage]
- Cash Flow Analysis: [text]

3. INVESTMENT THESIS
- Key Investment Drivers: [list]
- Growth Opportunities: [list]
- Competitive Advantages: [list]
- Market Positioning: [text]

4. VALUATION ANALYSIS
- Fair Value Estimate: [number]
- Target Price: [number]
- Entry Range: [min] - [max]
- DCF Value: [number]
- Peer Comparison: {{
    "P/E Ratio": [number],
    "P/B Ratio": [number],
    "EV/EBITDA": [number],
    "Dividend Yield": [percentage]
}}

5. INVESTMENT RECOMMENDATION
- Recommendation: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
- Confidence Score: [0-1]
- Investment Horizon: [time period]
- Position Size: [percentage]

6. MONITORING POINTS
- Key Metrics: [list]
- Risk Factors: [list]
- Growth Catalysts: [list]
- Red Flags: [list]

7. RISK ASSESSMENT
{{
    "Market Risk": {{
        "Level": [LOW/MEDIUM/HIGH],
        "Description": [text]
    }},
    "Business Risk": {{
        "Level": [LOW/MEDIUM/HIGH],
        "Description": [text]
    }},
    "Financial Risk": {{
        "Level": [LOW/MEDIUM/HIGH],
        "Description": [text]
    }},
    "Regulatory Risk": {{
        "Level": [LOW/MEDIUM/HIGH],
        "Description": [text]
    }}
}}

8. TECHNICAL ANALYSIS
{{
    "Trend": {{
        "Direction": [BULLISH/BEARISH/NEUTRAL],
        "Strength": [0-1],
        "Description": [text]
    }},
    "Support Levels": [list of numbers],
    "Resistance Levels": [list of numbers],
    "Momentum": {{
        "RSI": [number],
        "MACD": [number],
        "Description": [text]
    }},
    "Volume Analysis": {{
        "Volume Trend": [INCREASING/DECREASING/STABLE],
        "Description": [text]
    }}
}}

9. MARKET SENTIMENT
{{
    "Overall Sentiment": [BULLISH/BEARISH/NEUTRAL],
    "Institutional Interest": [HIGH/MEDIUM/LOW],
    "Retail Sentiment": [HIGH/MEDIUM/LOW],
    "Analyst Ratings": {{
        "Buy": [number],
        "Hold": [number],
        "Sell": [number]
    }}
}}

10. INDUSTRY ANALYSIS
{{
    "Industry Trends": [text],
    "Competitive Position": [text],
    "Market Share": [percentage],
    "Growth Prospects": [text]
}}

11. REGULATORY ANALYSIS
{{
    "Current Regulations": [text],
    "Potential Changes": [text],
    "Compliance Status": [COMPLIANT/NON-COMPLIANT]
}}

12. LIQUIDITY ANALYSIS
{{
    "Trading Volume": [number],
    "Bid-Ask Spread": [percentage],
    "Market Depth": [HIGH/MEDIUM/LOW]
}}

13. VOLATILITY ANALYSIS
{{
    "Historical Volatility": [percentage],
    "Implied Volatility": [percentage],
    "Volatility Trend": [INCREASING/DECREASING/STABLE]
}}

14. DIVIDEND ANALYSIS
{{
    "Dividend Yield": [percentage],
    "Payout Ratio": [percentage],
    "Dividend Growth": [percentage],
    "Sustainability": [HIGH/MEDIUM/LOW]
}}

15. MANAGEMENT QUALITY
- Leadership Assessment: [text]
- Track Record: [text]
- Strategic Vision: [text]
- Corporate Governance: [text]

16. CORPORATE GOVERNANCE
- Board Structure: [text]
- Shareholder Rights: [text]
- Transparency: [HIGH/MEDIUM/LOW]
- Ethical Practices: [text]

Please ensure all sections are filled with specific data points and metrics where applicable. Use numerical values for quantitative metrics and descriptive text for qualitative analysis."""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a professional financial analyst providing detailed stock analysis."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                "max_tokens": 4000,
                "temperature": 0.7
            }
            
            def make_api_call():
                logger.info("Making API call to DeepSeek...")
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180  # Increased timeout for longer analysis
                )
                
                logger.info(f"API Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                elif response.status_code == 429:
                    raise requests.exceptions.RequestException("Rate limit exceeded")
                elif response.status_code == 401:
                    raise requests.exceptions.RequestException("Authentication failed")
                else:
                    raise requests.exceptions.RequestException(f"API request failed with status code {response.status_code}")
            
            # Make API call with retries
            analysis = self._retry_with_backoff(make_api_call)
            
            if analysis:
                logger.info("Successfully received AI analysis")
                
                # Initialize analysis dictionary with all required sections
                ai_analysis = {
                    'company_overview': {},
                    'financial_health': {},
                    'investment_thesis': {},
                    'valuation_analysis': {},
                    'investment_recommendation': {},
                    'monitoring_points': {},
                    'risk_assessment': {},
                    'technical_analysis': {},
                    'market_sentiment': {},
                    'industry_analysis': {},
                    'regulatory_analysis': {},
                    'liquidity_analysis': {},
                    'volatility_analysis': {},
                    'dividend_analysis': {},
                    'management_quality': {},
                    'corporate_governance': {}
                }
                
                try:
                    # Extract sections using enhanced regex patterns
                    section_patterns = {
                        'company_overview': r'1\.\s*COMPANY OVERVIEW\s*([\s\S]*?)(?=2\.\s*FINANCIAL HEALTH)',
                        'financial_health': r'2\.\s*FINANCIAL HEALTH\s*([\s\S]*?)(?=3\.\s*INVESTMENT THESIS)',
                        'investment_thesis': r'3\.\s*INVESTMENT THESIS\s*([\s\S]*?)(?=4\.\s*VALUATION ANALYSIS)',
                        'valuation_analysis': r'4\.\s*VALUATION ANALYSIS\s*([\s\S]*?)(?=5\.\s*INVESTMENT RECOMMENDATION)',
                        'investment_recommendation': r'5\.\s*INVESTMENT RECOMMENDATION\s*([\s\S]*?)(?=6\.\s*MONITORING POINTS)',
                        'monitoring_points': r'6\.\s*MONITORING POINTS\s*([\s\S]*?)(?=7\.\s*RISK ASSESSMENT)',
                        'risk_assessment': r'7\.\s*RISK ASSESSMENT\s*({[\s\S]*?})(?=8\.\s*TECHNICAL ANALYSIS)',
                        'technical_analysis': r'8\.\s*TECHNICAL ANALYSIS\s*({[\s\S]*?})(?=9\.\s*MARKET SENTIMENT)',
                        'market_sentiment': r'9\.\s*MARKET SENTIMENT\s*({[\s\S]*?})(?=10\.\s*INDUSTRY ANALYSIS)',
                        'industry_analysis': r'10\.\s*INDUSTRY ANALYSIS\s*({[\s\S]*?})(?=11\.\s*REGULATORY ANALYSIS)',
                        'regulatory_analysis': r'11\.\s*REGULATORY ANALYSIS\s*({[\s\S]*?})(?=12\.\s*LIQUIDITY ANALYSIS)',
                        'liquidity_analysis': r'12\.\s*LIQUIDITY ANALYSIS\s*({[\s\S]*?})(?=13\.\s*VOLATILITY ANALYSIS)',
                        'volatility_analysis': r'13\.\s*VOLATILITY ANALYSIS\s*({[\s\S]*?})(?=14\.\s*DIVIDEND ANALYSIS)',
                        'dividend_analysis': r'14\.\s*DIVIDEND ANALYSIS\s*({[\s\S]*?})(?=15\.\s*MANAGEMENT QUALITY)',
                        'management_quality': r'15\.\s*MANAGEMENT QUALITY\s*([\s\S]*?)(?=16\.\s*CORPORATE GOVERNANCE)',
                        'corporate_governance': r'16\.\s*CORPORATE GOVERNANCE\s*([\s\S]*?)(?=\Z)'
                    }
                    
                    for section, pattern in section_patterns.items():
                        match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
                        if match:
                            content = match.group(1).strip()
                            
                            # Handle JSON sections
                            if section in ['risk_assessment', 'technical_analysis', 'market_sentiment', 
                                         'industry_analysis', 'regulatory_analysis', 'liquidity_analysis', 
                                         'volatility_analysis', 'dividend_analysis']:
                                try:
                                    # Clean and parse JSON
                                    json_str = content.strip()
                                    # Remove any leading/trailing whitespace and newlines
                                    json_str = re.sub(r'^\s+|\s+$', '', json_str, flags=re.MULTILINE)
                                    # Replace single quotes with double quotes
                                    json_str = json_str.replace("'", '"')
                                    # Fix property names
                                    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                                    # Fix string values
                                    json_str = re.sub(r':\s*([^",\{\}\[\]\d][^",\{\}\[\]]*?)([,}])', r':"\1"\2', json_str)
                                    # Fix control characters
                                    json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
                                    # Fix missing commas
                                    json_str = re.sub(r'"\s*}\s*"', '", "', json_str)
                                    # Fix trailing commas
                                    json_str = re.sub(r',\s*}', '}', json_str)
                                    json_str = re.sub(r',\s*]', ']', json_str)
                                    
                                    # Try to parse the cleaned JSON
                                    try:
                                        ai_analysis[section] = json.loads(json_str)
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse JSON for {section}: {e}")
                                        # Try to extract key-value pairs manually
                                        try:
                                            # Extract key-value pairs with more flexible pattern
                                            pairs = re.findall(r'"([^"]+)":\s*"([^"]+)"', json_str)
                                            if pairs:
                                                ai_analysis[section] = dict(pairs)
                                            else:
                                                # Try alternative pattern for nested structures
                                                pairs = re.findall(r'"?([^"]+)"?\s*:\s*({[^}]+})', json_str)
                                                if pairs:
                                                    nested_dict = {}
                                                    for key, value in pairs:
                                                        nested_dict[key.strip()] = value.strip()
                                                    ai_analysis[section] = nested_dict
                                                else:
                                                    ai_analysis[section] = {}
                                        except Exception as e:
                                            logger.error(f"Failed to extract key-value pairs for {section}: {e}")
                                            ai_analysis[section] = {}
                                except Exception as e:
                                    logger.error(f"Error processing JSON for {section}: {e}")
                                    ai_analysis[section] = {}
                            else:
                                # Handle non-JSON sections
                                try:
                                    # Extract key-value pairs
                                    pairs = re.findall(r'-?\s*([^:]+):\s*([^\n]+)', content)
                                    if pairs:
                                        section_dict = {}
                                        for key, value in pairs:
                                            key = key.strip().lower().replace(' ', '_')
                                            value = value.strip()
                                            # Try to convert numeric values
                                            try:
                                                if '%' in value:
                                                    value = float(value.replace('%', ''))
                                                elif value.replace('.', '').isdigit():
                                                    value = float(value)
                                            except ValueError:
                                                pass
                                            section_dict[key] = value
                                        ai_analysis[section] = section_dict
                                    else:
                                        # Store as text if no key-value pairs found
                                        ai_analysis[section] = {'text': content}
                                except Exception as e:
                                    logger.error(f"Failed to parse {section}: {e}")
                                    ai_analysis[section] = {'text': content}
                    
                    # Extract specific metrics using enhanced patterns
                    metric_patterns = {
                        'confidence_score': r'confidence score.*?(\d+\.?\d*)',
                        'fair_value': r'fair value.*?(\d+\.?\d*)',
                        'target_price': r'target price.*?(\d+\.?\d*)',
                        'entry_range': r'entry range.*?(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',
                        'investment_horizon': r'investment horizon.*?(\d+\s*(?:months|years))',
                        'position_size': r'position size.*?(\d+\.?\d*%)',
                        'dcf_value': r'dcf value.*?(\d+\.?\d*)'
                    }
                    
                    for metric, pattern in metric_patterns.items():
                        match = re.search(pattern, analysis.lower())
                        if match:
                            if metric == 'entry_range':
                                ai_analysis['valuation_analysis'][metric] = [float(match.group(1)), float(match.group(2))]
                            elif metric in ['confidence_score', 'fair_value', 'target_price', 'dcf_value']:
                                ai_analysis['valuation_analysis'][metric] = float(match.group(1))
                            else:
                                ai_analysis['valuation_analysis'][metric] = match.group(1)
                    
                    # Validate and clean the analysis
                    for key, value in ai_analysis.items():
                        if isinstance(value, dict):
                            # Remove empty values
                            value = {k: v for k, v in value.items() if v is not None and v != ''}
                            if not value:
                                ai_analysis[key] = {}
                        elif isinstance(value, str) and not value.strip():
                            ai_analysis[key] = {}
                        elif isinstance(value, list) and not value:
                            ai_analysis[key] = []
                    
                    logger.info("Successfully parsed AI analysis")
                    return ai_analysis
                    
                except Exception as e:
                    logger.error(f"Error parsing AI response: {e}")
                    return None
            else:
                logger.warning("No analysis received from AI model")
                return None
                
        except Exception as e:
            logger.error(f"Error in AI model call: {e}")
            return None

    def get_symbols_with_missing_data(self) -> List[str]:
        """Get list of symbols that have missing data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for symbols with missing critical data
            cursor.execute("""
                SELECT symbol, COUNT(*) as total_records,
                       SUM(CASE WHEN rsi IS NULL THEN 1 ELSE 0 END) as missing_rsi,
                       SUM(CASE WHEN macd IS NULL THEN 1 ELSE 0 END) as missing_macd,
                       SUM(CASE WHEN sma_20 IS NULL THEN 1 ELSE 0 END) as missing_sma20
                FROM tradingview_ta
                GROUP BY symbol
                HAVING missing_rsi > 0 OR missing_macd > 0 OR missing_sma20 > 0
            """)
            
            incomplete_symbols = cursor.fetchall()
            conn.close()
            
            if incomplete_symbols:
                symbols_with_missing_data = [symbol_data[0] for symbol_data in incomplete_symbols]
                logger.info(f"Found {len(symbols_with_missing_data)} symbols with missing data")
                return symbols_with_missing_data
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting symbols with missing data: {e}")
            if 'conn' in locals():
                conn.close()
            return []

    def requery_missing_data(self, symbols: List[str], max_retries: int = 3) -> Dict[str, bool]:
        """Requery data for symbols with missing data"""
        try:
            results = {}
            for symbol in symbols:
                logger.info(f"Requerying data for {symbol}")
                success = False
                
                for attempt in range(max_retries):
                    try:
                        # Force update by temporarily modifying should_update_data
                        original_should_update = self.should_update_data
                        self.should_update_data = lambda x: True
                        
                        # Fetch new data
                        data = self.fetch_tradingview_ta_data(symbol)
                        
                        # Restore original should_update_data
                        self.should_update_data = original_should_update
                        
                        if data and all(data.get(field) is not None for field in ['rsi', 'macd', 'sma_20']):
                            success = True
                            logger.info(f"Successfully requeried data for {symbol} on attempt {attempt + 1}")
                            break
                        else:
                            logger.warning(f"Attempt {attempt + 1} failed for {symbol} - data still incomplete")
                            time.sleep(2)  # Add delay between retries
                            
                    except Exception as e:
                        logger.error(f"Error requerying data for {symbol} on attempt {attempt + 1}: {e}")
                        time.sleep(2)  # Add delay between retries
                
                results[symbol] = success
                
            return results
            
        except Exception as e:
            logger.error(f"Error in requery_missing_data: {e}")
            return {symbol: False for symbol in symbols}

    def _get_last_signal_info(self, symbol: str) -> Dict:
        """Get the last signal date and price for a symbol from KMI30 database"""
        try:
            kmi30_db_path = 'data/databases/production/PSX_investing_Stocks_KMI30.db'
            
            if not os.path.exists(kmi30_db_path):
                logger.warning(f"KMI30 database not found at {kmi30_db_path}")
                return {}
            
            conn = sqlite3.connect(kmi30_db_path)
            cursor = conn.cursor()
            
            # Get the latest signal
            cursor.execute("""
                SELECT date, price, signal_type
                FROM buy_stocks 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'last_signal_date': result[0],
                    'last_signal_price': result[1],
                    'last_signal_type': result[2]
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting last signal info for {symbol}: {e}")
            if 'conn' in locals():
                conn.close()
            return {}

    def _round_float(self, value: float) -> float:
        """Helper method to round float values to 2 decimal places"""
        if value is None:
            return None
        return round(float(value), 2)

    def _round_decimal(self, value: float, places: int = 2) -> float:
        """Helper method to round decimal values to specified number of places"""
        if value is None:
            return None
        return round(float(value), places)

    def _calculate_percentage_change(self, current: float, previous: float) -> float:
        """Calculate percentage change with 2 decimal places"""
        if previous is None or previous == 0:
            return None
        return self._round_decimal(((current - previous) / previous) * 100)

    def _calculate_fair_value(self, symbol: str) -> Dict:
        """Calculate fair value using multiple valuation methods with risk adjustment"""
        try:
            # Get required data
            technical_data = self.get_latest_data(symbol)
            financial_data = self.analyze_financial_data(symbol)
            dividend_data = self.analyze_dividend_data(symbol)
            
            # Initialize valuation results
            valuations = {
                'dcf_value': None,
                'pe_based': None,
                'pb_based': None,
                'dividend_discount': None,
                'industry_comparison': None,
                'risk_adjusted': None
            }
            
            # DCF Valuation
            try:
                growth_rate = financial_data.get('eps_growth', 0) / 100
                discount_rate = 0.15  # Base discount rate
                terminal_growth = 0.03  # Terminal growth rate
                
                # Calculate DCF value
                current_eps = financial_data.get('eps', 0)
                if current_eps > 0:
                    # Project cash flows
                    projected_cash_flows = []
                    for year in range(1, 6):
                        projected_cash_flows.append(current_eps * (1 + growth_rate) ** year)
                    
                    # Calculate terminal value
                    terminal_value = projected_cash_flows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
                    
                    # Calculate present value
                    present_value = sum(cf / (1 + discount_rate) ** year for year, cf in enumerate(projected_cash_flows, 1))
                    present_value += terminal_value / (1 + discount_rate) ** 5
                    
                    valuations['dcf_value'] = present_value
            except Exception as e:
                logger.error(f"Error in DCF calculation for {symbol}: {e}")
            
            # PE-based Valuation
            try:
                industry_pe = self._get_industry_pe_ratio(symbol)
                if industry_pe and financial_data.get('eps'):
                    valuations['pe_based'] = industry_pe * financial_data['eps']
            except Exception as e:
                logger.error(f"Error in PE-based valuation for {symbol}: {e}")
            
            # PB-based Valuation
            try:
                industry_pb = self._get_industry_pb_ratio(symbol)
                if industry_pb and financial_data.get('book_value'):
                    valuations['pb_based'] = industry_pb * financial_data['book_value']
            except Exception as e:
                logger.error(f"Error in PB-based valuation for {symbol}: {e}")
            
            # Dividend Discount Model
            try:
                if dividend_data.get('dividend_yield') and dividend_data.get('dividend_growth_rate'):
                    required_return = 0.12  # Required rate of return
                    current_price = technical_data.get('close', 0)
                    current_dividend = current_price * dividend_data['dividend_yield'] / 100
                    growth_rate = dividend_data['dividend_growth_rate'] / 100
                    
                    if required_return > growth_rate:
                        valuations['dividend_discount'] = current_dividend * (1 + growth_rate) / (required_return - growth_rate)
            except Exception as e:
                logger.error(f"Error in dividend discount model for {symbol}: {e}")
            
            # Industry Comparison
            try:
                peer_valuations = self._get_peer_valuations(symbol)
                if peer_valuations:
                    valuations['industry_comparison'] = sum(peer_valuations) / len(peer_valuations)
            except Exception as e:
                logger.error(f"Error in industry comparison for {symbol}: {e}")
            
            # Risk Adjustment
            try:
                # Calculate risk factors
                market_risk = self._calculate_market_risk(symbol)
                financial_risk = self._calculate_financial_risk(financial_data)
                liquidity_risk = self._calculate_liquidity_risk(technical_data)
                
                # Calculate weighted risk score
                risk_score = (
                    market_risk * 0.4 +
                    financial_risk * 0.4 +
                    liquidity_risk * 0.2
                )
                
                # Adjust valuations based on risk
                risk_adjusted_valuations = {}
                for method, value in valuations.items():
                    if value is not None:
                        # Higher risk score leads to lower valuation
                        risk_adjusted_valuations[method] = value * (1 - risk_score)
                
                valuations['risk_adjusted'] = risk_adjusted_valuations
            except Exception as e:
                logger.error(f"Error in risk adjustment for {symbol}: {e}")
            
            # Calculate final fair value
            valid_valuations = [v for v in valuations.values() if v is not None]
            if valid_valuations:
                if isinstance(valid_valuations[0], dict):  # Risk-adjusted valuations
                    final_fair_value = sum(valid_valuations[0].values()) / len(valid_valuations[0]) if valid_valuations else None
                else:
                    final_fair_value = sum(valid_valuations) / len(valid_valuations)
            else:
                final_fair_value = None
            
            return {
                'symbol': symbol,
                'valuations': valuations,
                'fair_value': final_fair_value,
                'risk_score': risk_score if 'risk_score' in locals() else None,
                'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error calculating fair value for {symbol}: {e}")
            return None

    def _calculate_market_risk(self, symbol: str) -> float:
        """Calculate market risk based on beta and market volatility"""
        try:
            # Get historical data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate beta
            cursor.execute("""
                SELECT close, date 
                FROM tradingview_ta 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 252
            """, (symbol,))
            
            stock_returns = []
            market_returns = []
            
            for row in cursor.fetchall():
                stock_returns.append(row[0])
            
            # Calculate volatility
            if len(stock_returns) > 1:
                returns = pd.Series(stock_returns).pct_change().dropna()
                volatility = returns.std()
                
                # Normalize risk score between 0 and 1
                risk_score = min(volatility * 2, 1)  # Assuming 50% volatility is maximum risk
                return risk_score
            
            return 0.5  # Default risk score if not enough data
            
        except Exception as e:
            logger.error(f"Error calculating market risk for {symbol}: {e}")
            return 0.5

    def _calculate_financial_risk(self, financial_data: Dict) -> float:
        """Calculate financial risk based on financial metrics"""
        try:
            risk_factors = []
            
            # Debt to Equity
            if financial_data.get('debt_to_equity'):
                de_ratio = financial_data['debt_to_equity']
                if de_ratio > 2:
                    risk_factors.append(1.0)
                elif de_ratio > 1:
                    risk_factors.append(0.7)
                else:
                    risk_factors.append(0.3)
            
            # Current Ratio
            if financial_data.get('current_ratio'):
                cr = financial_data['current_ratio']
                if cr < 1:
                    risk_factors.append(1.0)
                elif cr < 1.5:
                    risk_factors.append(0.7)
                else:
                    risk_factors.append(0.3)
            
            # Profit Margin
            if financial_data.get('profit_margin'):
                pm = financial_data['profit_margin']
                if pm < 0:
                    risk_factors.append(1.0)
                elif pm < 5:
                    risk_factors.append(0.7)
                else:
                    risk_factors.append(0.3)
            
            return sum(risk_factors) / len(risk_factors) if risk_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating financial risk: {e}")
            return 0.5

    def _calculate_liquidity_risk(self, technical_data: Dict) -> float:
        """Calculate liquidity risk based on trading volume and bid-ask spread"""
        try:
            risk_factors = []
            
            # Volume Analysis
            if technical_data.get('volume'):
                avg_volume = technical_data['volume']
                if avg_volume < 10000:
                    risk_factors.append(1.0)
                elif avg_volume < 50000:
                    risk_factors.append(0.7)
                else:
                    risk_factors.append(0.3)
            
            # Price Volatility
            if technical_data.get('volatility_score'):
                vol_score = technical_data['volatility_score']
                risk_factors.append(vol_score)
            
            return sum(risk_factors) / len(risk_factors) if risk_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 0.5

    def _get_industry_pe_ratio(self, symbol: str) -> float:
        """Get industry average PE ratio"""
        try:
            # This would typically come from a market data provider
            # For now, return a default value
            return 15.0
        except Exception as e:
            logger.error(f"Error getting industry PE ratio for {symbol}: {e}")
            return None

    def _get_industry_pb_ratio(self, symbol: str) -> float:
        """Get industry average PB ratio"""
        try:
            # This would typically come from a market data provider
            # For now, return a default value
            return 2.0
        except Exception as e:
            logger.error(f"Error getting industry PB ratio for {symbol}: {e}")
            return None

    def _get_peer_valuations(self, symbol: str) -> List[float]:
        """Get valuations of peer companies"""
        try:
            # This would typically come from a market data provider
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error getting peer valuations for {symbol}: {e}")
            return []

    def _calculate_position_size(self, stock_data: Dict, portfolio_value: float, risk_per_trade: float = 0.02) -> Dict:
        """Calculate optimal position size based on risk management"""
        try:
            position = {
                'symbol': stock_data['symbol'],
                'current_price': stock_data['close'][-1],
                'position_size': 0,
                'shares': 0,
                'risk_amount': 0,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward_ratio': None
            }
            
            # Calculate volatility-based stop loss
            volatility = self._calculate_volatility(stock_data)
            atr = self._calculate_atr(stock_data)
            
            # Set stop loss at 2 ATR below current price
            stop_loss = position['current_price'] - (2 * atr)
            position['stop_loss'] = stop_loss
            
            # Calculate risk per share
            risk_per_share = position['current_price'] - stop_loss
            
            # Calculate maximum position size based on risk
            max_risk_amount = portfolio_value * risk_per_trade
            max_shares = int(max_risk_amount / risk_per_share)
            
            # Calculate position size
            position['shares'] = max_shares
            position['position_size'] = max_shares * position['current_price']
            position['risk_amount'] = max_shares * risk_per_share
            
            # Set take profit at 2x risk (1:2 risk-reward ratio)
            position['take_profit'] = position['current_price'] + (2 * (position['current_price'] - stop_loss))
            position['risk_reward_ratio'] = 2.0
            
            return position
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None
    
    def _calculate_portfolio_risk(self, positions: List[Dict], portfolio_value: float) -> Dict:
        """Calculate portfolio-level risk metrics"""
        try:
            risk_metrics = {
                'total_risk': 0.0,
                'diversification_score': 0.0,
                'correlation_matrix': {},
                'var_95': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'beta': 0.0,
                'sector_exposure': {},
                'risk_decomposition': {}
            }
            
            # Calculate portfolio weights
            weights = [pos['position_size'] / portfolio_value for pos in positions]
            
            # Calculate returns for each position
            returns = []
            for pos in positions:
                if 'returns' in pos:
                    returns.append(pos['returns'])
            
            if returns:
                returns_df = pd.DataFrame(returns)
                
                # Calculate Value at Risk (VaR)
                portfolio_returns = returns_df.dot(weights)
                risk_metrics['var_95'] = np.percentile(portfolio_returns, 5)
                
                # Calculate Maximum Drawdown
                cumulative_returns = (1 + portfolio_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = cumulative_returns / rolling_max - 1
                risk_metrics['max_drawdown'] = drawdowns.min()
                
                # Calculate Sharpe Ratio
                risk_free_rate = 0.02  # Assuming 2% risk-free rate
                excess_returns = portfolio_returns - risk_free_rate/252
                risk_metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
                
                # Calculate Beta
                market_returns = self._get_market_returns()  # Implement this method
                if market_returns is not None:
                    covariance = np.cov(portfolio_returns, market_returns)[0,1]
                    market_variance = np.var(market_returns)
                    risk_metrics['beta'] = covariance / market_variance
                
                # Calculate correlation matrix
                risk_metrics['correlation_matrix'] = returns_df.corr().to_dict()
                
                # Calculate diversification score
                avg_correlation = np.mean(np.abs(returns_df.corr().values - np.eye(len(returns_df))))
                risk_metrics['diversification_score'] = 1 - avg_correlation
            
            # Calculate sector exposure
            sector_exposure = {}
            for pos in positions:
                if 'sector' in pos:
                    sector = pos['sector']
                    if sector not in sector_exposure:
                        sector_exposure[sector] = 0
                    sector_exposure[sector] += pos['position_size'] / portfolio_value
            risk_metrics['sector_exposure'] = sector_exposure
            
            # Calculate risk decomposition
            for pos in positions:
                if 'volatility' in pos:
                    risk_metrics['risk_decomposition'][pos['symbol']] = {
                        'volatility': pos['volatility'],
                        'weight': pos['position_size'] / portfolio_value,
                        'marginal_risk': pos['volatility'] * (pos['position_size'] / portfolio_value)
                    }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return None
    
    def _calculate_volatility(self, stock_data: Dict) -> float:
        """Calculate historical volatility"""
        try:
            if 'close' in stock_data and len(stock_data['close']) > 0:
                returns = pd.Series(stock_data['close']).pct_change().dropna()
                return returns.std() * np.sqrt(252)  # Annualized volatility
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_atr(self, stock_data: Dict, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if all(field in stock_data for field in ['high', 'low', 'close']):
                high = pd.Series(stock_data['high'])
                low = pd.Series(stock_data['low'])
                close = pd.Series(stock_data['close'])
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=period).mean().iloc[-1]
                
                return atr
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    def _get_market_returns(self) -> Optional[pd.Series]:
        """Get market returns for beta calculation"""
        try:
            # Implement market returns calculation
            # This could be from an index like KSE-100
            return None
        except Exception as e:
            logger.error(f"Error getting market returns: {e}")
            return None
    
    def _calculate_risk_adjusted_return(self, stock_data: Dict) -> Dict:
        """Calculate risk-adjusted return metrics"""
        try:
            metrics = {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'information_ratio': 0.0,
                'treynor_ratio': 0.0
            }
            
            if 'close' in stock_data and len(stock_data['close']) > 0:
                returns = pd.Series(stock_data['close']).pct_change().dropna()
                
                # Calculate Sharpe Ratio
                risk_free_rate = 0.02  # Assuming 2% risk-free rate
                excess_returns = returns - risk_free_rate/252
                metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
                
                # Calculate Sortino Ratio
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    metrics['sortino_ratio'] = np.sqrt(252) * excess_returns.mean() / downside_std
                
                # Calculate Calmar Ratio
                cumulative_returns = (1 + returns).cumprod()
                max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
                if max_drawdown != 0:
                    metrics['calmar_ratio'] = returns.mean() * 252 / abs(max_drawdown)
                
                # Calculate Information Ratio
                market_returns = self._get_market_returns()
                if market_returns is not None:
                    tracking_error = (returns - market_returns).std()
                    if tracking_error != 0:
                        metrics['information_ratio'] = (returns.mean() - market_returns.mean()) / tracking_error
                
                # Calculate Treynor Ratio
                beta = self._calculate_beta(returns, market_returns)
                if beta != 0:
                    metrics['treynor_ratio'] = excess_returns.mean() / beta
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted returns: {e}")
            return {}
    
    def _calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        try:
            if market_returns is not None and len(returns) == len(market_returns):
                covariance = np.cov(returns, market_returns)[0,1]
                market_variance = np.var(market_returns)
                return covariance / market_variance
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 0.0

    def _analyze_market_conditions(self, stock_data: Dict) -> Dict:
        """Analyze overall market conditions and sector rotation"""
        try:
            market_analysis = {
                'market_trend': None,
                'sector_rotation': {},
                'market_breadth': {},
                'correlation_analysis': {},
                'macro_impact': {},
                'sentiment_analysis': {}
            }
            
            # Analyze market trend
            if 'close' in stock_data and len(stock_data['close']) > 0:
                prices = pd.Series(stock_data['close'])
                sma_20 = prices.rolling(window=20).mean()
                sma_50 = prices.rolling(window=50).mean()
                sma_200 = prices.rolling(window=200).mean()
                
                current_price = prices.iloc[-1]
                market_analysis['market_trend'] = {
                    'trend': 'bullish' if current_price > sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1] else
                            'bearish' if current_price < sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1] else
                            'neutral',
                    'strength': abs((current_price - sma_200.iloc[-1]) / sma_200.iloc[-1]) * 100
                }
            
            # Analyze sector rotation
            sectors = self._get_sector_data()  # Implement this method
            if sectors:
                for sector, data in sectors.items():
                    returns = pd.Series(data['close']).pct_change()
                    momentum = returns.rolling(window=20).mean().iloc[-1]
                    volatility = returns.std() * np.sqrt(252)
                    market_analysis['sector_rotation'][sector] = {
                        'momentum': momentum,
                        'volatility': volatility,
                        'relative_strength': momentum / volatility if volatility != 0 else 0
                    }
            
            # Calculate market breadth
            if 'advances' in stock_data and 'declines' in stock_data:
                advances = stock_data['advances']
                declines = stock_data['declines']
                market_analysis['market_breadth'] = {
                    'adv_dec_ratio': advances / declines if declines != 0 else float('inf'),
                    'new_highs_lows_ratio': stock_data.get('new_highs', 0) / stock_data.get('new_lows', 1),
                    'put_call_ratio': stock_data.get('put_volume', 0) / stock_data.get('call_volume', 1)
                }
            
            # Analyze correlations
            market_analysis['correlation_analysis'] = self._analyze_correlations(stock_data)
            
            # Analyze macroeconomic impact
            market_analysis['macro_impact'] = self._analyze_macro_impact(stock_data)
            
            # Analyze market sentiment
            market_analysis['sentiment_analysis'] = self._analyze_market_sentiment(stock_data)
            
            return market_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {}
    
    def _analyze_correlations(self, stock_data: Dict) -> Dict:
        """Analyze correlations with market indices and sectors"""
        try:
            correlations = {
                'market_correlation': 0.0,
                'sector_correlations': {},
                'peer_correlations': {},
                'factor_correlations': {}
            }
            
            # Calculate market correlation
            market_returns = self._get_market_returns()
            if market_returns is not None and 'close' in stock_data:
                stock_returns = pd.Series(stock_data['close']).pct_change()
                correlations['market_correlation'] = stock_returns.corr(market_returns)
            
            # Calculate sector correlations
            sectors = self._get_sector_data()
            if sectors:
                for sector, data in sectors.items():
                    sector_returns = pd.Series(data['close']).pct_change()
                    if 'close' in stock_data:
                        stock_returns = pd.Series(stock_data['close']).pct_change()
                        correlations['sector_correlations'][sector] = stock_returns.corr(sector_returns)
            
            # Calculate peer correlations
            peers = self._get_peer_data()  # Implement this method
            if peers:
                for peer, data in peers.items():
                    peer_returns = pd.Series(data['close']).pct_change()
                    if 'close' in stock_data:
                        stock_returns = pd.Series(stock_data['close']).pct_change()
                        correlations['peer_correlations'][peer] = stock_returns.corr(peer_returns)
            
            # Calculate factor correlations
            factors = self._get_factor_data()  # Implement this method
            if factors:
                for factor, data in factors.items():
                    factor_returns = pd.Series(data['returns'])
                    if 'close' in stock_data:
                        stock_returns = pd.Series(stock_data['close']).pct_change()
                        correlations['factor_correlations'][factor] = stock_returns.corr(factor_returns)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    def _analyze_macro_impact(self, stock_data: Dict) -> Dict:
        """Analyze impact of macroeconomic factors"""
        try:
            macro_impact = {
                'interest_rate_impact': 0.0,
                'inflation_impact': 0.0,
                'gdp_impact': 0.0,
                'currency_impact': 0.0,
                'commodity_impact': 0.0
            }
            
            # Get macroeconomic data
            macro_data = self._get_macro_data()  # Implement this method
            if macro_data:
                # Calculate interest rate impact
                if 'interest_rates' in macro_data and 'close' in stock_data:
                    rate_changes = pd.Series(macro_data['interest_rates']).pct_change()
                    stock_returns = pd.Series(stock_data['close']).pct_change()
                    macro_impact['interest_rate_impact'] = stock_returns.corr(rate_changes)
                
                # Calculate inflation impact
                if 'inflation' in macro_data and 'close' in stock_data:
                    inflation_changes = pd.Series(macro_data['inflation']).pct_change()
                    stock_returns = pd.Series(stock_data['close']).pct_change()
                    macro_impact['inflation_impact'] = stock_returns.corr(inflation_changes)
                
                # Calculate GDP impact
                if 'gdp' in macro_data and 'close' in stock_data:
                    gdp_changes = pd.Series(macro_data['gdp']).pct_change()
                    stock_returns = pd.Series(stock_data['close']).pct_change()
                    macro_impact['gdp_impact'] = stock_returns.corr(gdp_changes)
                
                # Calculate currency impact
                if 'exchange_rates' in macro_data and 'close' in stock_data:
                    currency_changes = pd.Series(macro_data['exchange_rates']).pct_change()
                    stock_returns = pd.Series(stock_data['close']).pct_change()
                    macro_impact['currency_impact'] = stock_returns.corr(currency_changes)
                
                # Calculate commodity impact
                if 'commodity_prices' in macro_data and 'close' in stock_data:
                    commodity_changes = pd.Series(macro_data['commodity_prices']).pct_change()
                    stock_returns = pd.Series(stock_data['close']).pct_change()
                    macro_impact['commodity_impact'] = stock_returns.corr(commodity_changes)
            
            return macro_impact
            
        except Exception as e:
            logger.error(f"Error analyzing macro impact: {e}")
            return {}
    
    def _analyze_market_sentiment(self, stock_data: Dict) -> Dict:
        """Analyze market sentiment indicators"""
        try:
            sentiment = {
                'technical_sentiment': 0.0,
                'fundamental_sentiment': 0.0,
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'overall_sentiment': 0.0
            }
            
            # Calculate technical sentiment
            if 'close' in stock_data and len(stock_data['close']) > 0:
                prices = pd.Series(stock_data['close'])
                rsi = self._calculate_rsi(prices)
                macd = self._calculate_macd(prices)
                sentiment['technical_sentiment'] = (
                    (rsi - 50) / 50 +  # RSI contribution
                    (macd['histogram'] / prices.iloc[-1])  # MACD contribution
                ) / 2
            
            # Calculate fundamental sentiment
            if 'pe_ratio' in stock_data and 'pb_ratio' in stock_data:
                pe_sentiment = (stock_data['pe_ratio'] - 15) / 15  # Assuming 15 as neutral PE
                pb_sentiment = (stock_data['pb_ratio'] - 1.5) / 1.5  # Assuming 1.5 as neutral PB
                sentiment['fundamental_sentiment'] = (pe_sentiment + pb_sentiment) / 2
            
            # Calculate news sentiment
            news_data = self._get_news_data()  # Implement this method
            if news_data:
                sentiment['news_sentiment'] = np.mean([article['sentiment'] for article in news_data])
            
            # Calculate social sentiment
            social_data = self._get_social_data()  # Implement this method
            if social_data:
                sentiment['social_sentiment'] = np.mean([post['sentiment'] for post in social_data])
            
            # Calculate overall sentiment
            sentiment['overall_sentiment'] = np.mean([
                sentiment['technical_sentiment'],
                sentiment['fundamental_sentiment'],
                sentiment['news_sentiment'],
                sentiment['social_sentiment']
            ])
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {}
    
    def _generate_analysis_report(self, stock_data: Dict, analysis_results: Dict) -> str:
        """Generate a comprehensive analysis report"""
        try:
            report = []
            
            # Add header
            report.append(f"Analysis Report for {stock_data['symbol']}")
            report.append("=" * 50)
            
            # Add technical analysis
            report.append("\nTechnical Analysis:")
            report.append("-" * 20)
            if 'technical_analysis' in analysis_results:
                tech = analysis_results['technical_analysis']
                report.append(f"Trend: {tech.get('trend', 'N/A')}")
                report.append(f"Support Level: {tech.get('support_level', 'N/A')}")
                report.append(f"Resistance Level: {tech.get('resistance_level', 'N/A')}")
                report.append(f"RSI: {tech.get('rsi', 'N/A')}")
                report.append(f"MACD: {tech.get('macd', 'N/A')}")
            
            # Add fundamental analysis
            report.append("\nFundamental Analysis:")
            report.append("-" * 20)
            if 'fundamental_analysis' in analysis_results:
                fund = analysis_results['fundamental_analysis']
                report.append(f"PE Ratio: {fund.get('pe_ratio', 'N/A')}")
                report.append(f"PB Ratio: {fund.get('pb_ratio', 'N/A')}")
                report.append(f"Dividend Yield: {fund.get('dividend_yield', 'N/A')}")
                report.append(f"ROE: {fund.get('roe', 'N/A')}")
            
            # Add risk analysis
            report.append("\nRisk Analysis:")
            report.append("-" * 20)
            if 'risk_analysis' in analysis_results:
                risk = analysis_results['risk_analysis']
                report.append(f"Volatility: {risk.get('volatility', 'N/A')}")
                report.append(f"Beta: {risk.get('beta', 'N/A')}")
                report.append(f"Value at Risk (95%): {risk.get('var_95', 'N/A')}")
                report.append(f"Maximum Drawdown: {risk.get('max_drawdown', 'N/A')}")
            
            # Add market analysis
            report.append("\nMarket Analysis:")
            report.append("-" * 20)
            if 'market_analysis' in analysis_results:
                market = analysis_results['market_analysis']
                report.append(f"Market Trend: {market.get('market_trend', 'N/A')}")
                report.append(f"Market Breadth: {market.get('market_breadth', 'N/A')}")
                report.append(f"Sector Rotation: {market.get('sector_rotation', 'N/A')}")
            
            # Add recommendations
            report.append("\nRecommendations:")
            report.append("-" * 20)
            if 'recommendations' in analysis_results:
                recs = analysis_results['recommendations']
                report.append(f"Signal: {recs.get('signal', 'N/A')}")
                report.append(f"Target Price: {recs.get('target_price', 'N/A')}")
                report.append(f"Stop Loss: {recs.get('stop_loss', 'N/A')}")
                report.append(f"Position Size: {recs.get('position_size', 'N/A')}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            return "Error generating report"

    def _create_tables(self):
        """Create necessary database tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tradingview_signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tradingview_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
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
                    signal TEXT,
                    target_price REAL,
                    stop_loss REAL,
                    position_size INTEGER,
                    risk_score REAL,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tradingview_signals_symbol ON tradingview_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tradingview_signals_date ON tradingview_signals(date)')
            
            # Create dividend_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dividend_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    announcement_date TEXT NOT NULL,
                    ex_date TEXT NOT NULL,
                    payment_date TEXT NOT NULL,
                    dividend_amount REAL NOT NULL,
                    dividend_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, ex_date)
                )
            """)
            
            # Create market_analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    market_trend TEXT,
                    market_breadth TEXT,
                    sector_rotation TEXT,
                    correlation_matrix TEXT,
                    macro_impact TEXT,
                    market_sentiment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            # Create signal_transitions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    previous_signal TEXT,
                    current_signal TEXT,
                    transition_type TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dividend_data_symbol_ex_date ON dividend_data(symbol, ex_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_analysis_date ON market_analysis(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_transitions_symbol_date ON signal_transitions(symbol, date)")
            
            conn.commit()
            conn.close()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    def get_latest_data(self, symbol: str) -> Dict:
        """Get the latest data for a symbol from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM tradingview_signals
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
            """, (symbol,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return None

    def get_data_for_date_range(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Get data for a symbol within a date range"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM tradingview_signals
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date ASC
            """, (symbol, start_date, end_date))
            
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
            return []
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol} between {start_date} and {end_date}: {e}")
            return []

    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT symbol FROM tradingview_signals
                ORDER BY symbol
            """)
            
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting all symbols: {e}")
            return []

    def get_latest_date(self) -> str:
        """Get the latest date in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT MAX(date) FROM tradingview_signals")
            latest_date = cursor.fetchone()[0]
            conn.close()
            
            return latest_date
            
        except Exception as e:
            logger.error(f"Error getting latest date: {e}")
            return None

    def get_symbol_count(self) -> int:
        """Get the number of unique symbols in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM tradingview_signals")
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting symbol count: {e}")
            return 0

    def _fetch_data_from_tradingview(self, symbol: str) -> Dict:
        """Fetch technical analysis data from TradingView"""
        try:
            from tradingview_ta import TA_Handler, Interval
            
            # Initialize TA Handler for PSX symbol
            handler = TA_Handler(
                symbol=symbol,
                exchange="PSX",
                screener="pakistan",
                interval=Interval.INTERVAL_1_DAY
            )
            
            # Get analysis
            analysis = handler.get_analysis()
            
            # Helper function to safely get indicator value
            def get_indicator(key, default=None):
                try:
                    return analysis.indicators.get(key, default)
                except (KeyError, AttributeError):
                    return default
            
            # Extract required data with safe fallbacks
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'open': get_indicator('open'),
                'high': get_indicator('high'),
                'low': get_indicator('low'),
                'close': get_indicator('close'),
                'volume': get_indicator('volume'),
                'rsi': get_indicator('RSI'),
                'macd': get_indicator('MACD.macd'),
                'macd_signal': get_indicator('MACD.signal'),
                'macd_hist': get_indicator('MACD.hist'),
                'sma_20': get_indicator('SMA20'),
                'sma_50': get_indicator('SMA50'),
                'sma_200': get_indicator('SMA200'),
                'ema_20': get_indicator('EMA20'),
                'ema_50': get_indicator('EMA50'),
                'ema_200': get_indicator('EMA200'),
                'bollinger_upper': get_indicator('BB.upperband'),
                'bollinger_middle': get_indicator('BB.middleband'),
                'bollinger_lower': get_indicator('BB.lowerband'),
                'stoch_k': get_indicator('Stoch.K'),
                'stoch_d': get_indicator('Stoch.D'),
                'ichimoku_tenkan': get_indicator('Ichimoku.Tenkan-sen'),
                'ichimoku_kijun': get_indicator('Ichimoku.Kijun-sen'),
                'ichimoku_senkou_span_a': get_indicator('Ichimoku.Senkou Span A'),
                'ichimoku_senkou_span_b': get_indicator('Ichimoku.Senkou Span B'),
                'ichimoku_cloud_green': 1 if get_indicator('Ichimoku.Senkou Span A', 0) > get_indicator('Ichimoku.Senkou Span B', 0) else 0,
                'ichimoku_cloud_red': 1 if get_indicator('Ichimoku.Senkou Span A', 0) < get_indicator('Ichimoku.Senkou Span B', 0) else 0,
                'support_level': get_indicator('Pivot.M.Classic.S3'),
                'resistance_level': get_indicator('Pivot.M.Classic.R3'),
                'trend': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                'momentum': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                'volume_profile': 'HIGH' if get_indicator('volume', 0) > get_indicator('SMA20', 0) else 'LOW',
                'pattern': None,  # Will be calculated separately
                'signal': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                'target_price': None,  # Will be calculated separately
                'stop_loss': None,  # Will be calculated separately
                'position_size': None,  # Will be calculated separately
                'risk_score': None,  # Will be calculated separately
                'confidence_score': None  # Will be calculated separately
            }
            print(f"DEBUG: Data fetched from TradingView: {data}")
            
            # Validate required fields
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            missing_fields = [field for field in required_fields if data[field] is None]
            
            if missing_fields:
                logger.warning(f"Missing required fields for {symbol}: {missing_fields}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from TradingView for {symbol}: {e}")
            return None

def test_tradingview_integration():
    """Test the TradingView integration"""
    try:
        # Initialize calculator
        print("\n1. Initializing FairValueCalculator...")
        calculator = FairValueCalculator("PSX_Stock_Data.db")
        
        # Test symbol
        test_symbol = "LUCK"
        print(f"\n2. Testing with symbol: {test_symbol}")
        
        # Fetch data
        print("   Fetching data from TradingView...")
        data = calculator._fetch_data_from_tradingview(test_symbol)
        
        if not data:
            print("✗ Failed to fetch data from TradingView")
            return
        
        print("✓ Successfully fetched data from TradingView")
        print(f"  - Date: {data['date']}")
        print(f"  - Close: {data['close']}")
        print(f"  - RSI: {data['rsi']}")
        print(f"  - MACD: {data['macd']}")
        
        # Save to database
        print("\n   Saving data to database...")
        conn = sqlite3.connect("PSX_Stock_Data.db")
        save_result = calculator.save_tradingview_ta_data_to_db(test_symbol, data, conn)
        
        if save_result:
            print("✓ Successfully saved data to database")
        else:
            print("✗ Failed to save data to database")
            return
        
        # Verify data in database
        print("\n3. Verifying data in database...")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM tradingview_signals 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 1
        """, (test_symbol,))
        
        row = cursor.fetchone()
        if row:
            print("✓ Successfully retrieved data from database")
            columns = [description[0] for description in cursor.description]
            db_data = dict(zip(columns, row))
            print(f"  - Date: {db_data['date']}")
            print(f"  - Close: {db_data['close']}")
            print(f"  - RSI: {db_data['rsi']}")
            print(f"  - MACD: {db_data['macd']}")
        else:
            print("✗ Failed to retrieve data from database")
        
        conn.close()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        logging.error(f"Test failed: {e}")

if __name__ == "__main__":
    # Run the test
    test_tradingview_integration()

