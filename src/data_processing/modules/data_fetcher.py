from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
import re
import logging
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from tradingview_ta import TA_Handler, Interval
import time

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

@retry(max_retries=3)
def fetch_data_from_tradingview(symbol: str) -> Dict:
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
        logger.error(f"Error fetching data from TradingView for {symbol}: {str(e)}")
        logger.error(f"Check if symbol {symbol} exists on TradingView with the correct exchange (PSX) and screener (pakistan).")
        return None

def get_indicator_safely(indicators: Dict, key: str, default: Optional[float] = None) -> Optional[float]:
    """Safely get an indicator value from a dictionary"""
    try:
        return indicators.get(key, default)
    except (KeyError, AttributeError):
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

def fetch_psx_symbols() -> List[str]:
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

def read_psx_announcements() -> Dict:
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
                'impact': analyze_announcement_impact(row.get('Title', ''), row.get('Type', ''))
            }
            announcements[symbol].append(announcement)
        
        logger.info(f"Successfully loaded announcements for {len(announcements)} symbols")
        return announcements
        
    except Exception as e:
        logger.error(f"Error reading PSX announcements: {e}")
        return {}

def analyze_announcement_impact(title: str, announcement_type: str) -> float:
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
