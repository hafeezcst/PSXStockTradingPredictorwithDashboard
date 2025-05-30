from __future__ import annotations
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tradingview_ta import TA_Handler, Interval

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class FairValueCalculator:
    """A comprehensive stock analysis tool that calculates fair value using multiple indicators.
    
    This class integrates technical analysis, fundamental analysis, and AI-based insights
    to generate trading signals and fair value estimates for stocks. It maintains a database
    of historical analysis results and provides notification capabilities.
    
    Key Features:
    - Technical analysis using indicators like RSI, MACD, Bollinger Bands
    - Fundamental analysis of financial reports and announcements
    - AI-enhanced signal generation and analysis
    - Database storage of analysis results
    - Telegram notification system
    - Discounted Cash Flow (DCF) valuation
    
    Attributes:
        db_path (str): Path to the main analysis database
        dividend_db_path (str): Path to the dividend database
        headers (dict): HTTP headers for web requests
    """
    
    def __init__(self) -> None:
        """Initialize the FairValueCalculator with database paths and headers.
        
        Sets up:
        - Paths to SQLite databases
        - HTTP headers for web requests
        - Initializes the database structure
        """
        self.db_path = 'data/databases/production/fairvalue.db'
        self.dividend_db_path = 'data/databases/production/PSX_Dividend_Schedule.db'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
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

    def calculate_dcf_value(self, symbol: str, years: int = 5, discount_rate: float = 0.1, terminal_growth_rate: float = 0.03) -> float:
        """Calculate the Discounted Cash Flow (DCF) value for a given stock symbol.
        
        This method uses the DCF model to estimate the intrinsic value of a stock based on projected
        free cash flows, a discount rate, and a terminal growth rate. It includes estimates for capital
        expenditures and changes in working capital based on historical data or industry averages if
        specific data is unavailable.
        
        Args:
            symbol (str): The stock symbol to calculate the DCF value for.
            years (int): Number of years for cash flow projections. Default is 5.
            discount_rate (float): The discount rate used in the DCF calculation. Default is 0.1 (10%).
            terminal_growth_rate (float): The terminal growth rate for perpetuity. Default is 0.03 (3%).
        
        Returns:
            float: The calculated DCF value of the stock.
        
        Raises:
            ValueError: If the required financial data is not available or invalid.
        """
        try:
            # Get the latest financial data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM financial_reports 
                WHERE symbol = ? 
                ORDER BY report_date DESC 
                LIMIT 1
            """, (symbol,))
            
            financial_data = cursor.fetchone()
            conn.close()
            
            if not financial_data:
                raise ValueError(f"No financial data available for symbol: {symbol}")
            
            # Extract necessary financial metrics
            revenue = financial_data['revenue_growth']
            ebitda = financial_data['profit_margin'] * revenue / 100  # Assuming profit margin is EBITDA margin
            
            # Estimate capital expenditures (capex) and change in working capital (change_in_wc)
            # Using industry average assumptions if specific data is not available
            # Assuming capex is 10% of revenue as a rough estimate
            capex = revenue * 0.1
            # Assuming change in working capital is 5% of revenue growth
            change_in_wc = revenue * 0.05
            
            # Calculate free cash flow
            fcf = ebitda - capex - change_in_wc
            
            # Project future cash flows with a more conservative growth adjustment
            projected_cash_flows = []
            for year in range(1, years + 1):
                growth_factor = min(revenue / 100, 0.2)  # Cap growth rate at 20% to avoid unrealistic projections
                projected_cash_flow = fcf * (1 + growth_factor) ** year
                projected_cash_flows.append(projected_cash_flow / (1 + discount_rate) ** year)
            
            # Calculate terminal value with a sanity check
            if discount_rate <= terminal_growth_rate:
                raise ValueError(f"Discount rate ({discount_rate}) must be greater than terminal growth rate ({terminal_growth_rate})")
            terminal_value = projected_cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
            terminal_value_discounted = terminal_value / (1 + discount_rate) ** years
            
            # Sum up the present value of projected cash flows and terminal value
            dcf_value = sum(projected_cash_flows) + terminal_value_discounted
            
            # Ensure DCF value is not negative
            return max(dcf_value, 0.0)
            
        except Exception as e:
            self._handle_error(e, f"calculating DCF value for {symbol}")
            raise

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

    def fetch_psx_symbols(self) -> List[str]:
        """Fetch list of PSX symbols from Excel file.
        
        Reads symbols from an Excel file and returns them as a cleaned list.
        Handles duplicate symbols and invalid entries.
        
        Returns:
            List[str]: List of cleaned and validated stock symbols in uppercase
            
        Example:
            >>> calculator.fetch_psx_symbols()
            ['OGDC', 'PPL', 'LUCK', 'ENGRO']
            
        Raises:
            Exception: If the Excel file cannot be read or processed
        """
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
        """Determine if new data should be fetched for a given symbol based on update frequency rules.
        
        This method implements a weekly update cadence for stock data, with additional checks to:
        - Avoid redundant updates within 24 hours
        - Handle cases where no existing data exists
        
        Args:
            symbol (str): The stock symbol to check update status for
            
        Returns:
            bool: True if data should be updated, False otherwise
            
        Example:
            >>> calculator = FairValueCalculator()
            >>> calculator.should_update_data('OGDC')
            True  # If no existing data or weekly update needed
            
        Notes:
            - Weekly updates are determined by comparing ISO week numbers
            - A 24-hour cooldown period is enforced between updates
            - Returns True if no existing data is found for the symbol
        """
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
        """Determine if the Pakistan Stock Exchange (PSX) is currently open for trading.
        
        Checks multiple factors to determine market status:
        - Current day of week (closed weekends)
        - Trading hours in Pakistan Time (PKT)
        - Official market holidays
        
        Returns:
            bool: True if market is open, False otherwise
            
        Example:
            >>> calculator = FairValueCalculator()
            >>> calculator.get_market_status()
            False  # If called outside trading hours
            
        Notes:
            - Uses Pakistan Standard Time (UTC+5) for time calculations
            - Market hours: 9:30 AM - 3:30 PM PKT (regular session)
            - Pre-market: 9:00 AM - 9:30 AM PKT
            - Post-market: 3:30 PM - 4:00 PM PKT
            - Automatically checks for holidays (list can be expanded)
        """
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
        """Analyze financial reports and news for the symbol"""
        try:
            # Get financial data from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the latest financial data
            cursor.execute("""
                SELECT * FROM financial_reports 
                WHERE symbol = ? 
                ORDER BY report_date DESC 
                LIMIT 1
            """, (symbol,))
            
            financial_data = cursor.fetchone()
            
            # Get recent announcements
            announcements = self.read_psx_announcements()
            symbol_announcements = announcements.get(symbol, [])
            
            # Initialize analysis
            analysis = {
                'financial_score': 0,
                'financial_signal': 'NEUTRAL',
                'analysis': [],
                'confidence': 0.0,
                'recent_announcements': []
            }
            
            # Process financial data if available
            if financial_data:
                # Extract financial metrics
                metrics = {
                    'eps_growth': financial_data['eps_growth'],
                    'revenue_growth': financial_data['revenue_growth'],
                    'profit_margin': financial_data['profit_margin'],
                    'debt_to_equity': financial_data['debt_to_equity'],
                    'current_ratio': financial_data['current_ratio'],
                    'roe': financial_data['roe']
                }
                
                # Score each metric
                if metrics['eps_growth'] is not None:
                    if metrics['eps_growth'] > 20:
                        analysis['financial_score'] += 20
                        analysis['analysis'].append(f"Strong EPS growth: {metrics['eps_growth']:.2f}%")
                    elif metrics['eps_growth'] > 10:
                        analysis['financial_score'] += 10
                        analysis['analysis'].append(f"Moderate EPS growth: {metrics['eps_growth']:.2f}%")
                    elif metrics['eps_growth'] < -20:
                        analysis['financial_score'] -= 20
                        analysis['analysis'].append(f"Poor EPS growth: {metrics['eps_growth']:.2f}%")
                    elif metrics['eps_growth'] < -10:
                        analysis['financial_score'] -= 10
                        analysis['analysis'].append(f"Negative EPS growth: {metrics['eps_growth']:.2f}%")
                
                if metrics['revenue_growth'] is not None:
                    if metrics['revenue_growth'] > 15:
                        analysis['financial_score'] += 15
                        analysis['analysis'].append(f"Strong revenue growth: {metrics['revenue_growth']:.2f}%")
                    elif metrics['revenue_growth'] > 5:
                        analysis['financial_score'] += 7
                        analysis['analysis'].append(f"Moderate revenue growth: {metrics['revenue_growth']:.2f}%")
                    elif metrics['revenue_growth'] < -15:
                        analysis['financial_score'] -= 15
                        analysis['analysis'].append(f"Poor revenue growth: {metrics['revenue_growth']:.2f}%")
                    elif metrics['revenue_growth'] < -5:
                        analysis['financial_score'] -= 7
                        analysis['analysis'].append(f"Negative revenue growth: {metrics['revenue_growth']:.2f}%")
                
                if metrics['profit_margin'] is not None:
                    if metrics['profit_margin'] > 20:
                        analysis['financial_score'] += 15
                        analysis['analysis'].append(f"Strong profit margin: {metrics['profit_margin']:.2f}%")
                    elif metrics['profit_margin'] > 10:
                        analysis['financial_score'] += 7
                        analysis['analysis'].append(f"Good profit margin: {metrics['profit_margin']:.2f}%")
                    elif metrics['profit_margin'] < 5:
                        analysis['financial_score'] -= 15
                        analysis['analysis'].append(f"Low profit margin: {metrics['profit_margin']:.2f}%")
                
                if metrics['debt_to_equity'] is not None:
                    if metrics['debt_to_equity'] < 0.5:
                        analysis['financial_score'] += 10
                        analysis['analysis'].append(f"Low debt-to-equity: {metrics['debt_to_equity']:.2f}")
                    elif metrics['debt_to_equity'] > 2:
                        analysis['financial_score'] -= 10
                        analysis['analysis'].append(f"High debt-to-equity: {metrics['debt_to_equity']:.2f}")
                
                if metrics['current_ratio'] is not None:
                    if metrics['current_ratio'] > 2:
                        analysis['financial_score'] += 10
                        analysis['analysis'].append(f"Strong current ratio: {metrics['current_ratio']:.2f}")
                    elif metrics['current_ratio'] < 1:
                        analysis['financial_score'] -= 10
                        analysis['analysis'].append(f"Poor current ratio: {metrics['current_ratio']:.2f}")
                
                if metrics['roe'] is not None:
                    if metrics['roe'] > 20:
                        analysis['financial_score'] += 15
                        analysis['analysis'].append(f"Strong ROE: {metrics['roe']:.2f}%")
                    elif metrics['roe'] > 10:
                        analysis['financial_score'] += 7
                        analysis['analysis'].append(f"Good ROE: {metrics['roe']:.2f}%")
                    elif metrics['roe'] < 5:
                        analysis['financial_score'] -= 15
                        analysis['analysis'].append(f"Poor ROE: {metrics['roe']:.2f}%")
                
                # Add recent announcements impact
                if symbol_announcements:
                    announcement_impact = 0.0
                    for announcement in symbol_announcements:
                        announcement_impact += announcement['impact']
                        analysis['recent_announcements'].append({
                            'date': announcement['date'],
                            'title': announcement['title'],
                            'impact': announcement['impact']
                        })
                    
                    # Adjust financial score based on announcements
                    analysis['financial_score'] += (announcement_impact * 10)  # Scale impact to match financial metrics
                    
                    # Add announcement summary to analysis
                    if announcement_impact > 0:
                        analysis['analysis'].append(f"Positive recent announcements: {announcement_impact:.2f} impact")
                    elif announcement_impact < 0:
                        analysis['analysis'].append(f"Negative recent announcements: {abs(announcement_impact):.2f} impact")
            
            # Determine financial signal with announcement consideration
            if analysis['financial_score'] >= 50:
                analysis['financial_signal'] = 'STRONG_BUY'
            elif analysis['financial_score'] >= 25:
                analysis['financial_signal'] = 'BUY'
            elif analysis['financial_score'] <= -50:
                analysis['financial_signal'] = 'STRONG_SELL'
            elif analysis['financial_score'] <= -25:
                analysis['financial_signal'] = 'SELL'
            
            # Calculate confidence based on available data
            available_metrics = sum(1 for v in metrics.values() if v is not None) if 'metrics' in locals() else 0
            announcement_factor = 1.0 if symbol_announcements else 0.5
            analysis['confidence'] = (available_metrics / len(metrics) if 'metrics' in locals() else 0) * announcement_factor
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing financial data for {symbol}: {e}")
            return {
                'financial_score': 0,
                'financial_signal': 'NEUTRAL',
                'analysis': [],
                'confidence': 0.0,
                'recent_announcements': []
            }

    def analyze_with_ai(self, symbol: str, technical_data: Dict, financial_data: Dict) -> Dict:
        """Analyze stock data using AI to enhance signal generation with focus on investment perspective"""
        try:
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
                        'eps_growth': financial_data.get('eps_growth', 'N/A'),
                        'revenue_growth': financial_data.get('revenue_growth', 'N/A'),
                        'profit_margin': financial_data.get('profit_margin', 'N/A'),
                        'debt_to_equity': financial_data.get('debt_to_equity', 'N/A'),
                        'current_ratio': financial_data.get('current_ratio', 'N/A'),
                        'roe': financial_data.get('roe', 'N/A')
                    },
                    'reports': latest_reports,
                    'announcements': symbol_announcements
                },
                'dividend': {
                    'current_dividend': dividend_analysis.get('current_dividend') if dividend_analysis else None,
                    'dividend_yield': dividend_analysis.get('dividend_yield') if dividend_analysis else None,
                    'dividend_growth': dividend_analysis.get('dividend_growth') if dividend_analysis else None,
                    'dividend_sustainability': dividend_analysis.get('dividend_sustainability') if dividend_analysis else None
                }
            }
            
            # Prepare AI analysis prompt with enhanced investment focus
            prompt = f"""Analyze this stock data and provide a professional investment analysis:

Symbol: {symbol}

TECHNICAL ANALYSIS
-----------------
Current Price: {analysis_data['technical']['price']['close']}
Price Change: {analysis_data['technical']['price']['change_percent']}%
Volume: {analysis_data['technical']['price']['volume']}

Key Indicators:
- RSI: {analysis_data['technical']['indicators']['rsi']}
- MACD: {analysis_data['technical']['indicators']['macd']}
- Signal Line: {analysis_data['technical']['indicators']['macd_signal']}
- SMA20: {analysis_data['technical']['indicators']['sma_20']}
- SMA50: {analysis_data['technical']['indicators']['sma_50']}
- SMA200: {analysis_data['technical']['indicators']['sma_200']}

Technical Scores:
- Trend Score: {analysis_data['technical']['signals']['trend_score']}
- Momentum Score: {analysis_data['technical']['signals']['momentum_score']}
- Volume Score: {analysis_data['technical']['signals']['volume_score']}
- Volatility Score: {analysis_data['technical']['signals']['volatility_score']}

FINANCIAL ANALYSIS
-----------------
Current Metrics:
- EPS Growth: {analysis_data['financial']['current']['eps_growth']}%
- Revenue Growth: {analysis_data['financial']['current']['revenue_growth']}%
- Profit Margin: {analysis_data['financial']['current']['profit_margin']}%
- Debt-to-Equity: {analysis_data['financial']['current']['debt_to_equity']}
- Current Ratio: {analysis_data['financial']['current']['current_ratio']}
- ROE: {analysis_data['financial']['current']['roe']}%

DIVIDEND ANALYSIS
----------------
Current Dividend:
- Amount: {analysis_data['dividend']['current_dividend']['dividend_amount'] if analysis_data['dividend']['current_dividend'] else 'N/A'}
- Yield: {f"{analysis_data['dividend']['dividend_yield']:.2f}%" if analysis_data['dividend']['dividend_yield'] else 'N/A'}
- Growth: {f"{analysis_data['dividend']['dividend_growth']:.2f}%" if analysis_data['dividend']['dividend_growth'] else 'N/A'}
- Sustainability: {analysis_data['dividend']['dividend_sustainability']}

RECENT ANNOUNCEMENTS
-------------------
{chr(10).join([f"- {announcement['date']}: {announcement['title']}" for announcement in analysis_data['financial']['announcements']]) if analysis_data['financial']['announcements'] else 'No recent announcements'}

Provide a detailed analysis in the following format:

1. COMPANY OVERVIEW
- Market position and competitive advantages
- Growth potential
- Recent developments

2. FINANCIAL HEALTH
- Overall financial health assessment
- Key strengths and weaknesses
- Growth potential and risks

3. INVESTMENT THESIS
- Growth catalysts
- Risk factors
- Competitive position
- Management quality

4. VALUATION ANALYSIS
- Peer comparison
- Historical valuation ranges
- Fair value estimate
- Margin of safety

5. INVESTMENT RECOMMENDATION
- Clear recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
- Detailed rationale
- Risk factors
- Price targets

6. MONITORING POINTS
- Important announcements to watch
- Risk factors to monitor
- Exit criteria

7. TECHNICAL ANALYSIS
- Key technical indicators and their implications
- Trend analysis
- Support and resistance levels

8. RISK MANAGEMENT
- Position sizing
- Stop loss and take profit strategies
- Risk-reward ratio analysis

Include specific metrics where possible:
- Confidence Score (0-1)
- Fair Value
- DCF Value
- Target Price
- Entry Range
- Investment Horizon
- Position Size
- Risk-Reward Ratio
"""
            
            logger.info(f"Prepared AI analysis prompt for {symbol}")
            
            # Call AI model
            ai_analysis = self.call_ai_model(prompt)
            
            if ai_analysis:
                # Process AI analysis with enhanced investment focus
                processed_analysis = {
                    'company_overview': ai_analysis.get('company_overview', ''),
                    'financial_health': ai_analysis.get('financial_health', ''),
                    'investment_thesis': ai_analysis.get('investment_thesis', ''),
                    'valuation_analysis': ai_analysis.get('valuation_analysis', ''),
                    'investment_recommendation': ai_analysis.get('investment_recommendation', ''),
                    'monitoring_points': ai_analysis.get('monitoring_points', ''),
                    'confidence_score': ai_analysis.get('confidence_score', 0.0),
                    'fair_value': ai_analysis.get('fair_value', None),
                    'target_price': ai_analysis.get('target_price', None),
                    'entry_range': ai_analysis.get('entry_range', []),
                    'investment_horizon': ai_analysis.get('investment_horizon', ''),
                    'position_size': ai_analysis.get('position_size', ''),
                    'dcf_value': ai_analysis.get('dcf_value', None),
                    'peer_comparison': ai_analysis.get('peer_comparison', {}),
                    'risk_assessment': ai_analysis.get('risk_assessment', {}),
                    'growth_catalysts': ai_analysis.get('growth_catalysts', []),
                    'management_quality': ai_analysis.get('management_quality', ''),
                    'corporate_governance': ai_analysis.get('corporate_governance', ''),
                    'dividend_analysis': dividend_analysis
                }
                
                # Format the analysis for logging
                formatted_analysis = f"""
AI Investment Analysis for {symbol}
=================================

Company Overview
---------------
{processed_analysis['company_overview']}

Financial Health
---------------
{processed_analysis['financial_health']}

Investment Thesis
----------------
{processed_analysis['investment_thesis']}

Valuation Analysis
-----------------
{processed_analysis['valuation_analysis']}

Investment Recommendation
------------------------
{processed_analysis['investment_recommendation']}

Monitoring Points
----------------
{processed_analysis['monitoring_points']}

Key Metrics
-----------
Confidence Score: {processed_analysis['confidence_score']}
Fair Value: {processed_analysis['fair_value']}
DCF Value: {processed_analysis['dcf_value']}
Target Price: {processed_analysis['target_price']}
Entry Range: {processed_analysis['entry_range']}
Investment Horizon: {processed_analysis['investment_horizon']}
Position Size: {processed_analysis['position_size']}

Dividend Analysis
----------------
Current Dividend: {json.dumps(processed_analysis['dividend_analysis']['current_dividend'], indent=2) if processed_analysis['dividend_analysis']['current_dividend'] else 'N/A'}
Dividend Yield: {f"{processed_analysis['dividend_analysis']['dividend_yield']:.2f}%" if processed_analysis['dividend_analysis']['dividend_yield'] else 'N/A'}
Dividend Growth: {f"{processed_analysis['dividend_analysis']['dividend_growth']:.2f}%" if processed_analysis['dividend_analysis']['dividend_growth'] else 'N/A'}
Dividend Sustainability: {processed_analysis['dividend_analysis']['dividend_sustainability']}

Additional Analysis
------------------
Peer Comparison: {json.dumps(processed_analysis['peer_comparison'], indent=2)}
Risk Assessment: {json.dumps(processed_analysis['risk_assessment'], indent=2)}
Growth Catalysts: {json.dumps(processed_analysis['growth_catalysts'], indent=2)}
Management Quality: {processed_analysis['management_quality']}
Corporate Governance: {processed_analysis['corporate_governance']}
"""
                
                logger.info(f"Successfully processed AI investment analysis for {symbol}")
                logger.info(formatted_analysis)
                
                return processed_analysis
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
        """Perform technical analysis on stock data"""
        try:
            analysis = {
                'signal_type': 'NEUTRAL',
                'signal_strength': 0.0,
                'confidence_score': 0.0,
                'technical_score': 0.0,
                'trend_score': 0.0,
                'momentum_score': 0.0,
                'volume_score': 0.0,
                'volatility_score': 0.0,
                'support_level': None,
                'resistance_level': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward_ratio': None,
                'analysis_summary': [],
                'indicators_used': []
            }
            
            # Add symbol to analysis
            analysis['symbol'] = stock_data['symbol']
            
            # Copy price and indicator data from stock_data to analysis
            price_fields = ['close', 'open', 'high', 'low', 'volume', 'change', 'change_percent']
            indicator_fields = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_200', 'bb_upper', 'bb_lower']
            
            for field in price_fields + indicator_fields:
                if field in stock_data:
                    analysis[field] = stock_data[field]
            
            # Perform trend analysis
            self._analyze_trend(stock_data, analysis, previous_analysis)
            
            # Perform momentum analysis
            self._analyze_momentum(stock_data, analysis, previous_analysis)
            
            # Perform volume analysis
            self._analyze_volume(stock_data, analysis, previous_analysis)
            
            # Perform volatility analysis
            self._analyze_volatility(stock_data, analysis, previous_analysis)
            
            # Calculate final scores and determine signal
            self._calculate_final_scores(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing technical analysis: {e}")
            return None

    def _analyze_trend(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze trend indicators"""
        try:
            if all(x is not None for x in [stock_data['close'], stock_data['sma_20'], stock_data['sma_50'], stock_data['sma_200']]):
                analysis['indicators_used'].append('SMA')
                close = stock_data['close']
                sma20 = stock_data['sma_20']
                sma50 = stock_data['sma_50']
                sma200 = stock_data['sma_200']
                
                # Calculate price position relative to SMAs
                price_above_sma20 = (close - sma20) / sma20 * 100
                price_above_sma50 = (close - sma50) / sma50 * 100
                price_above_sma200 = (close - sma200) / sma200 * 100
                
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
                
                analysis['trend_score'] = trend_strength
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")

    def _analyze_momentum(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze momentum indicators"""
        try:
            if all(x is not None for x in [stock_data['rsi'], stock_data['macd'], stock_data['macd_signal'], stock_data['ao']]):
                analysis['indicators_used'].extend(['RSI', 'MACD', 'AO'])
                rsi = stock_data['rsi']
                macd = stock_data['macd']
                macd_signal = stock_data['macd_signal']
                ao = stock_data['ao']
                
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
                macd_diff = macd - macd_signal
                macd_diff_percent = (macd_diff / abs(macd_signal)) * 100 if macd_signal != 0 else 0
                
                if previous_analysis:
                    prev_macd = previous_analysis.get('macd', 0)
                    prev_macd_signal = previous_analysis.get('macd_signal', 0)
                    
                    if macd > macd_signal and prev_macd <= prev_macd_signal:
                        momentum_strength += 10
                        analysis['analysis_summary'].append("Bullish MACD crossover detected")
                    elif macd < macd_signal and prev_macd >= prev_macd_signal:
                        momentum_strength -= 10
                        analysis['analysis_summary'].append("Bearish MACD crossover detected")
                
                # Awesome Oscillator Analysis
                ao_abs = abs(ao)
                ao_threshold = 50
                
                if previous_analysis:
                    prev_ao = previous_analysis.get('ao', 0)
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
                
                analysis['momentum_score'] = momentum_strength
                
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")

    def _analyze_volume(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze volume indicators"""
        try:
            if all(x is not None for x in [stock_data['volume'], stock_data['change']]):
                analysis['indicators_used'].append('Volume')
                volume = stock_data['volume']
                change = stock_data['change']
                change_percent = stock_data.get('change_percent', 0)
                
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
                    prev_volume = previous_analysis.get('volume', 0)
                    if volume > prev_volume * 1.5:  # 50% volume increase
                        volume_strength += 5
                        analysis['analysis_summary'].append("Significant volume increase detected")
                    elif volume < prev_volume * 0.5:  # 50% volume decrease
                        volume_strength -= 5
                        analysis['analysis_summary'].append("Significant volume decrease detected")
                
                analysis['volume_score'] = volume_strength
                
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")

    def _analyze_volatility(self, stock_data: Dict, analysis: Dict, previous_analysis: Dict = None):
        """Analyze volatility indicators"""
        try:
            if all(x is not None for x in [stock_data['bb_upper'], stock_data['bb_lower'], stock_data['close']]):
                analysis['indicators_used'].append('Bollinger Bands')
                bb_upper = stock_data['bb_upper']
                bb_lower = stock_data['bb_lower']
                close = stock_data['close']
                
                bb_range = round(bb_upper - bb_lower, 2)
                volatility = round(bb_range / close * 100, 2)
                price_position = round((close - bb_lower) / bb_range * 100, 2)
                
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
                analysis['support_level'] = bb_lower
                analysis['resistance_level'] = bb_upper
                
                # Price position relative to BB
                if price_position > 80:
                    analysis['analysis_summary'].append(f"Price near upper BB: {price_position:.2f}% of range")
                    volatility_strength -= 5
                elif price_position < 20:
                    analysis['analysis_summary'].append(f"Price near lower BB: {price_position:.2f}% of range")
                    volatility_strength += 5
                
                analysis['volatility_score'] = volatility_strength
                
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")

    def _calculate_final_scores(self, analysis: Dict):
        """Calculate final scores and determine signal"""
        try:
            # Calculate technical score
            analysis['technical_score'] = (
                analysis['trend_score'] +
                analysis['momentum_score'] +
                analysis['volume_score'] +
                analysis['volatility_score']
            )
            
            # Determine signal type and strength
            if analysis['technical_score'] >= 70:
                analysis['signal_type'] = 'STRONG_BUY'
                analysis['signal_strength'] = min(analysis['technical_score'] / 70, 1.0)
            elif analysis['technical_score'] >= 40:
                analysis['signal_type'] = 'BUY'
                analysis['signal_strength'] = min(analysis['technical_score'] / 50, 0.8)
            elif analysis['technical_score'] <= -70:
                analysis['signal_type'] = 'STRONG_SELL'
                analysis['signal_strength'] = min(abs(analysis['technical_score']) / 70, 1.0)
            elif analysis['technical_score'] <= -40:
                analysis['signal_type'] = 'SELL'
                analysis['signal_strength'] = min(abs(analysis['technical_score']) / 50, 0.8)
            else:
                analysis['signal_type'] = 'NEUTRAL'
                analysis['signal_strength'] = 0.5
            
            # Debug logging for required values
            logger.debug(f"Required values for {analysis.get('symbol', 'unknown')}:")
            logger.debug(f"Close price: {analysis.get('close')}")
            logger.debug(f"BB Upper: {analysis.get('bb_upper')}")
            logger.debug(f"BB Lower: {analysis.get('bb_lower')}")
            logger.debug(f"SMA20: {analysis.get('sma_20')}")
            logger.debug(f"Signal Type: {analysis.get('signal_type')}")
            logger.debug(f"Volatility Score: {analysis.get('volatility_score')}")
            
            # Calculate stop loss and take profit levels
            if analysis.get('close') is not None:
                current_price = analysis['close']
                logger.debug(f"Using current price: {current_price}")
                
                # Calculate stop loss based on volatility and support levels
                if analysis.get('bb_lower') is not None and analysis.get('bb_upper') is not None:
                    # Use Bollinger Bands for stop loss
                    stop_loss_long = analysis['bb_lower']
                    stop_loss_short = analysis['bb_upper']
                    logger.debug("Using Bollinger Bands for stop loss calculation")
                elif analysis.get('sma_20') is not None:
                    # Use SMA20 as fallback
                    stop_loss_long = analysis['sma_20'] * 0.95  # 5% below SMA20
                    stop_loss_short = analysis['sma_20'] * 1.05  # 5% above SMA20
                    logger.debug("Using SMA20 as fallback for stop loss calculation")
                else:
                    # Default to percentage-based stop loss
                    stop_loss_long = current_price * 0.95  # 5% below current price
                    stop_loss_short = current_price * 1.05  # 5% above current price
                    logger.debug("Using percentage-based stop loss calculation")
                
                # Calculate take profit based on risk-reward ratio and volatility
                if analysis.get('volatility_score') is not None:
                    # Adjust take profit based on volatility
                    volatility_factor = 1 + (abs(analysis['volatility_score']) / 100)
                    logger.debug(f"Using volatility factor: {volatility_factor}")
                else:
                    volatility_factor = 1.0
                    logger.debug("No volatility score available, using default factor")
                
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
            
            # Convert dictionary values to JSON strings
            ai_price_targets = json.dumps(ai_data.get('price_targets', {}))
            ai_entry_points = json.dumps(ai_data.get('entry_points', []))
            ai_exit_points = json.dumps(ai_data.get('exit_points', []))
            ai_pattern_recognition = json.dumps(ai_data.get('pattern_recognition', ''))
            ai_signal_strength = json.dumps(ai_data.get('signal_strength', ''))
            ai_risk_assessment = json.dumps(ai_data.get('risk_assessment', ''))
            ai_recommendation = json.dumps(ai_data.get('recommendation', ''))
            
            cursor.execute('''
            INSERT OR REPLACE INTO tradingview_signals
            (symbol, date, signal_type, signal_strength, confidence_score,
             technical_score, trend_score, momentum_score, volume_score,
             volatility_score, support_level, resistance_level, stop_loss,
             take_profit, risk_reward_ratio, analysis_summary, indicators_used,
             last_updated, ai_score, ai_confidence, ai_pattern_recognition,
             ai_signal_strength, ai_risk_assessment, ai_recommendation,
             ai_price_targets, ai_entry_points, ai_exit_points, ai_analysis_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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
                ai_data.get('confidence_score', 0.0),
                ai_data.get('confidence_score', 0.0),
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
        """Fetch data using tradingview_ta library with weekly timeframe and improved caching.
        Optimized to prioritize the most common symbol format and implement better rate limiting.
        """
        try:
            # First check if we have valid cached data
            cached_data = self.get_latest_data(symbol)
            if cached_data:
                cache_date = datetime.strptime(cached_data.get('date', ''), '%Y-%m-%d')
                current_date = datetime.now()
                
                # If cached data is from current week, use it
                if cache_date.isocalendar()[1] == current_date.isocalendar()[1]:
                    logger.info(f"Using current week's cached data for {symbol}")
                    return cached_data
                
                # If cached data is less than 24 hours old, use it
                if (current_date - cache_date).total_seconds() < 86400:  # 24 hours
                    logger.info(f"Using recent cached data for {symbol} (less than 24 hours old)")
                    return cached_data
            
            # Check if we need to update the data
            if not self.should_update_data(symbol):
                logger.info(f"Using existing data for {symbol}")
                return cached_data
            
            # Prioritize the most common symbol format to reduce API calls
            symbol_formats = [
                symbol,           # Just the symbol (most common)
                f"{symbol}.PSX",  # PSX suffix (second most common)
                f"PSX:{symbol}",  # PSX prefix with colon (third most common)
                f"PSX-{symbol}"   # PSX prefix with hyphen (fourth most common)
            ]
            
            data = {}
            success = False
            max_retries = 2
            initial_retry_delay = 2
            max_delay = 10
            
            for symbol_format in symbol_formats:
                retry_delay = initial_retry_delay
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Attempting to fetch data for {symbol} using format: {symbol_format} (Attempt {attempt + 1}/{max_retries})")
                        
                        handler = TA_Handler(
                            symbol=symbol_format,
                            screener="pakistan",
                            exchange="PSX",
                            interval=Interval.INTERVAL_1_WEEK
                        )
                        
                        analysis = handler.get_analysis()
                        
                        # Validate the analysis data
                        if not analysis or not analysis.summary or not analysis.indicators:
                            logger.warning(f"Invalid analysis data received for {symbol} using format {symbol_format}")
                            continue
                        
                        # Extract summary data
                        if analysis.summary:
                            data.update({
                                'recommendation': analysis.summary.get('RECOMMENDATION'),
                                'buy_signals': analysis.summary.get('BUY'),
                                'sell_signals': analysis.summary.get('SELL'),
                                'neutral_signals': analysis.summary.get('NEUTRAL')
                            })
                        
                        # Extract all indicators from the indicators dictionary
                        if analysis.indicators:
                            # Calculate price changes and metrics
                            close = analysis.indicators.get('close')
                            open_price = analysis.indicators.get('open')
                            high = analysis.indicators.get('high')
                            low = analysis.indicators.get('low')
                            
                            # Initialize price change metrics
                            price_metrics = {
                                'change': None,
                                'change_percent': None,
                                'high_low_range': None,
                                'high_low_range_percent': None,
                                'volatility': None
                            }
                            
                            # Calculate daily change if we have both close and open
                            if close is not None and open_price is not None:
                                price_metrics['change'] = close - open_price
                                if open_price != 0:
                                    price_metrics['change_percent'] = (price_metrics['change'] / open_price) * 100
                            
                            # Calculate high-low range if we have both high and low
                            if high is not None and low is not None:
                                price_metrics['high_low_range'] = high - low
                                if low != 0:
                                    price_metrics['high_low_range_percent'] = (price_metrics['high_low_range'] / low) * 100
                            
                            # Calculate volatility (standard deviation of price changes)
                            if all(x is not None for x in [close, open_price, high, low]):
                                price_metrics['volatility'] = price_metrics['high_low_range_percent'] / 2
                            
                            indicator_data = {
                                # Oscillators
                                'rsi': analysis.indicators.get('RSI[1]'),
                                'stoch_k': analysis.indicators.get('Stoch.K[1]'),
                                'stoch_d': analysis.indicators.get('Stoch.D[1]'),
                                'macd': analysis.indicators.get('MACD.macd'),
                                'macd_signal': analysis.indicators.get('MACD.signal'),
                                'macd_hist': analysis.indicators.get('MACD.macd') - analysis.indicators.get('MACD.signal') if analysis.indicators.get('MACD.macd') is not None and analysis.indicators.get('MACD.signal') is not None else None,
                                
                                # Moving Averages
                                'sma_20': analysis.indicators.get('SMA20'),
                                'sma_50': analysis.indicators.get('SMA50'),
                                'sma_200': analysis.indicators.get('SMA200'),
                                'ema_20': analysis.indicators.get('EMA20'),
                                'ema_50': analysis.indicators.get('EMA50'),
                                'ema_200': analysis.indicators.get('EMA200'),
                                
                                # Price and Volume
                                'close': close,
                                'open': open_price,
                                'high': high,
                                'low': low,
                                'volume': analysis.indicators.get('volume'),
                                'change': price_metrics['change'],
                                'change_percent': price_metrics['change_percent'],
                                'high_low_range': price_metrics['high_low_range'],
                                'high_low_range_percent': price_metrics['high_low_range_percent'],
                                'volatility': price_metrics['volatility'],
                                
                                # Additional Indicators
                                'bb_upper': analysis.indicators.get('BB.upper'),
                                'bb_lower': analysis.indicators.get('BB.lower'),
                                'ao': analysis.indicators.get('AO[2]'),
                                'psar': analysis.indicators.get('P.SAR'),
                                'vwma': analysis.indicators.get('VWMA'),
                                'hull_ma9': analysis.indicators.get('HullMA9')
                            }
                            
                            # Validate required fields
                            required_fields = ['close', 'open', 'high', 'low', 'volume']
                            if all(indicator_data.get(field) is not None for field in required_fields):
                                data.update(indicator_data)
                                success = True
                                break
                            else:
                                missing_fields = [field for field in required_fields if indicator_data.get(field) is None]
                                logger.warning(f"Missing required fields for {symbol} using format {symbol_format}: {missing_fields}")
                                continue
                        
                        if success:
                            break
                            
                    except Exception as e:
                        error_msg = str(e)
                        if "Exchange or symbol not found" in error_msg:
                            logger.warning(f"Symbol format {symbol_format} not found for {symbol}")
                        else:
                            logger.error(f"Attempt {attempt + 1} failed for {symbol} using format {symbol_format}: {error_msg}")
                        
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, max_delay)  # Exponential backoff for retries
                        continue
                
                if success:
                    break
            
            if success and data:
                # Add timestamp for caching
                data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # Ensure symbol is present in data
                data['symbol'] = symbol
                # Add current date
                data['date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Save data to database
                save_result = self.save_tradingview_ta_data_to_db(symbol, data)
                if save_result:
                    logger.info(f"Successfully saved data for {symbol} to database")
                else:
                    logger.error(f"Failed to save data for {symbol} to database")
                
                # Analyze and save signals
                analysis_result = self.analyze_stock_indicators(data)
                if analysis_result:
                    self.save_analysis_to_db(symbol, analysis_result)
                    logger.info(f"Successfully saved analysis for {symbol} to database")
                else:
                    logger.error(f"Failed to save analysis for {symbol} to database")
                
                return data
            else:
                logger.warning(f"Could not fetch valid data for {symbol} using any symbol format")
                # Return cached data if available, even if it's old
                if cached_data:
                    logger.info(f"Returning cached data for {symbol} as fallback")
                    return cached_data
                return {}
            
        except Exception as e:
            logger.error(f"Error fetching TradingView TA data for {symbol}: {str(e)}")
            # Return cached data if available, even if it's old
            cached_data = self.get_latest_data(symbol)
            if cached_data:
                logger.info(f"Returning cached data for {symbol} after error")
                return cached_data
            return {}

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

    def analyze_database(self):
        """Analyze the database for null values and data quality"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total number of records
            cursor.execute("SELECT COUNT(*) FROM tradingview_ta")
            total_records = cursor.fetchone()[0]
            
            logger.info(f"Total records in database: {total_records}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error analyzing database: {e}")
            if 'conn' in locals():
                conn.close()

    def analyze_stock_signals(self):
        """Analyze stock data and generate AI-based signals for all symbols.
        
        This method fetches the latest data for each symbol from the database,
        performs a comprehensive analysis using technical indicators, and integrates
        AI-based insights to generate trading signals. It leverages the 
        `analyze_stock_indicators` method for individual stock analysis to avoid
        code duplication and ensure consistency.
        
        Returns:
            List[Dict]: A list of signal dictionaries for each analyzed stock.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the latest data for each symbol
            cursor.execute("""
                WITH latest_dates AS (
                    SELECT symbol, MAX(date) as max_date
                    FROM tradingview_ta
                    GROUP BY symbol
                )
                SELECT t.*
                FROM tradingview_ta t
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
                symbol = stock['symbol']
                logger.info(f"Analyzing signals for {symbol}")
                signal = self.analyze_stock_indicators(stock)
                if signal:
                    signals.append(signal)
                    self.save_analysis_to_db(symbol, signal)
            
            conn.close()
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing stock signals: {e}")
            if 'conn' in locals():
                conn.close()
            return []

    def verify_database_data(self):
        """Verify the quality and completeness of data in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total number of records
            cursor.execute("SELECT COUNT(*) FROM tradingview_ta")
            total_records = cursor.fetchone()[0]
            
            # Get count of records with complete data
            cursor.execute("""
                SELECT COUNT(*) FROM tradingview_ta 
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
            
            logger.info("DeepSeek API key found, proceeding with API call")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a professional financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            # Define the API call function
            def make_api_call():
                logger.info("Making API call to DeepSeek...")
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120  # Increased timeout
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
                    'company_overview': '',
                    'financial_health': '',
                    'investment_thesis': '',
                    'valuation_analysis': '',
                    'investment_recommendation': '',
                    'monitoring_points': '',
                    'confidence_score': 0.0,
                    'fair_value': None,
                    'target_price': None,
                    'entry_range': [],
                    'investment_horizon': '',
                    'position_size': '',
                    'dcf_value': None,
                    'peer_comparison': {},
                    'risk_assessment': {},
                    'growth_catalysts': [],
                    'management_quality': '',
                    'corporate_governance': '',
                    'dividend_analysis': {},
                    'technical_analysis': {},
                    'market_sentiment': {},
                    'industry_analysis': {},
                    'regulatory_analysis': {},
                    'liquidity_analysis': {},
                    'volatility_analysis': {}
                }
                
                # Process the analysis sections and extract data
                try:
                    # Split the analysis into sections
                    sections = analysis.split('\n\n')
                    current_section = None
                    section_content = []
                    
                    for section in sections:
                        section = section.strip()
                        if not section:
                            continue
                        
                        # Check for section headers and process content
                        for section_name in ai_analysis.keys():
                            header = section_name.upper().replace('_', ' ')
                            if header in section:
                                if current_section:
                                    ai_analysis[current_section] = '\n'.join(section_content)
                                current_section = section_name
                                section_content = []
                                break
                        else:
                            if current_section:
                                section_content.append(section)
                    
                    # Add the last section
                    if current_section and section_content:
                        ai_analysis[current_section] = '\n'.join(section_content)
                    
                    # Extract metrics using regex patterns
                    patterns = {
                        'confidence_score': r'confidence score.*?(\d+\.?\d*)',
                        'fair_value': r'fair value.*?(\d+\.?\d*)',
                        'target_price': r'target price.*?(\d+\.?\d*)',
                        'entry_range': r'entry range.*?(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',
                        'investment_horizon': r'investment horizon.*?(\d+\s*(?:months|years))',
                        'position_size': r'position size.*?(\d+\.?\d*%)',
                        'dcf_value': r'dcf value.*?(\d+\.?\d*)'
                    }
                    
                    for metric, pattern in patterns.items():
                        match = re.search(pattern, analysis.lower())
                        if match:
                            if metric == 'entry_range':
                                ai_analysis[metric] = [float(match.group(1)), float(match.group(2))]
                            elif metric in ['confidence_score', 'fair_value', 'target_price', 'dcf_value']:
                                ai_analysis[metric] = float(match.group(1))
                            else:
                                ai_analysis[metric] = match.group(1)
                    
                    # Extract JSON-formatted sections
                    json_sections = ['peer_comparison', 'risk_assessment', 'growth_catalysts', 
                                   'technical_analysis', 'market_sentiment', 'industry_analysis',
                                   'regulatory_analysis', 'liquidity_analysis', 'volatility_analysis']
                    
                    for section in json_sections:
                        pattern = f"{section.replace('_', ' ')}.*?({{\n.*?\n}})"
                        match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
                        if match:
                            try:
                                ai_analysis[section] = json.loads(match.group(1))
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON for {section}")
                    
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

def main():
    """Main function to create and initialize the database"""
    try:
        # Configure logging with debug level
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fair_value_calculator.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create calculator instance which will initialize the database
        calculator = FairValueCalculator()
        print("Database initialized successfully!")
        
        # Verify existing data
        print("\nVerifying existing data...")
        verification = calculator.verify_database_data()
        if verification:
            print(f"\nData Verification Results:")
            print(f"Total records: {verification['total_records']}")
            print(f"Complete records: {verification['complete_records']} ({verification['completeness']:.2f}%)")
            print(f"Latest data date: {verification['latest_date']}")
            print(f"Unique symbols: {verification['unique_symbols']}")
            if verification['incomplete_symbols'] > 0:
                print(f"Warning: Found {verification['incomplete_symbols']} symbols with incomplete data")
        
        # Check market status
        if not calculator.get_market_status():
            print("\nMarket is currently closed. Using cached data.")
        
        # Fetch all PSX symbols
        symbols = calculator.fetch_psx_symbols()
        print(f"Found {len(symbols)} PSX symbols")
        
        # Track success and failure
        successful = 0
        failed = 0
        skipped = 0
        
        # Fetch data for each symbol with rate limiting
        for i, symbol in enumerate(symbols, 1):
            print(f"Processing {symbol} ({i}/{len(symbols)})...")
            
            # Check if we need to update
            if not calculator.should_update_data(symbol):
                print(f"Skipping data update for {symbol} - current data available")
                skipped += 1
            else:
                data = calculator.fetch_tradingview_ta_data(symbol)
                if data:
                    print(f"Successfully fetched data for {symbol}")
                    successful += 1
                else:
                    print(f"Failed to fetch data for {symbol}")
                    failed += 1
            
            # Always perform analysis regardless of data update
            print(f"Performing analysis for {symbol}...")
            try:
                # Get the latest data for analysis
                latest_data = calculator.get_latest_data(symbol)
                if latest_data:
                    # Get financial analysis
                    financial_analysis = calculator.analyze_financial_data(symbol)
                    
                    # Perform technical analysis
                    technical_analysis = calculator._perform_technical_analysis(latest_data)
                    
                    # Get AI analysis
                    ai_analysis = calculator.analyze_with_ai(symbol, technical_analysis, financial_analysis)
                    
                    if ai_analysis:
                        # Integrate AI analysis
                        technical_analysis['ai_analysis'] = ai_analysis
                        
                        # Adjust signal based on AI insights
                        final_analysis = calculator.adjust_signal_with_ai(technical_analysis, financial_analysis)
                        
                        # Save the analysis
                        calculator.save_analysis_to_db(symbol, final_analysis)
                        print(f"Successfully completed analysis for {symbol}")
                    else:
                        print(f"Warning: No AI analysis available for {symbol}")
                else:
                    print(f"Warning: No data available for analysis of {symbol}")
            except Exception as e:
                print(f"Error performing analysis for {symbol}: {e}")
                logger.error(f"Error performing analysis for {symbol}: {e}")
            
            # Add delay to avoid rate limiting
            time.sleep(2)
        
        print(f"\nData collection completed:")
        print(f"Successfully fetched: {successful} symbols")
        print(f"Failed to fetch: {failed} symbols")
        print(f"Skipped (current data): {skipped} symbols")
        
        # Get symbols with missing data and requery them
        print("\nChecking for symbols with missing data...")
        symbols_with_missing_data = calculator.get_symbols_with_missing_data()
        
        if symbols_with_missing_data:
            print(f"\nFound {len(symbols_with_missing_data)} symbols with missing data. Attempting to requery...")
            requery_results = calculator.requery_missing_data(symbols_with_missing_data)
            
            # Print requery results
            successful_requeries = sum(1 for success in requery_results.values() if success)
            print(f"\nRequery Results:")
            print(f"Successfully requeried: {successful_requeries} symbols")
            print(f"Failed to requery: {len(requery_results) - successful_requeries} symbols")
            
            # Print details for failed requeries
            failed_symbols = [symbol for symbol, success in requery_results.items() if not success]
            if failed_symbols:
                print("\nFailed to requery data for the following symbols:")
                for symbol in failed_symbols:
                    print(f"- {symbol}")
        
        # Verify data after fetching and requerying
        print("\nVerifying updated data...")
        verification = calculator.verify_database_data()
        if verification:
            print(f"\nUpdated Data Verification Results:")
            print(f"Total records: {verification['total_records']}")
            print(f"Complete records: {verification['complete_records']} ({verification['completeness']:.2f}%)")
            print(f"Latest data date: {verification['latest_date']}")
            print(f"Unique symbols: {verification['unique_symbols']}")
            if verification['incomplete_symbols'] > 0:
                print(f"Warning: Found {verification['incomplete_symbols']} symbols with incomplete data")
        
        # Check for signal transitions
        print("\nChecking for signal transitions...")
        conn = sqlite3.connect(calculator.db_path)
        cursor = conn.cursor()
        
        # Get all symbols
        cursor.execute("SELECT DISTINCT symbol FROM tradingview_signals")
        symbols = [row[0] for row in cursor.fetchall()]
        
        for symbol in symbols:
            # Get current and previous signals
            cursor.execute("""
                SELECT * FROM tradingview_signals 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 2
            """, (symbol,))
            
            results = cursor.fetchall()
            if len(results) >= 2:
                columns = [description[0] for description in cursor.description]
                current_analysis = dict(zip(columns, results[0]))
                previous_analysis = dict(zip(columns, results[1]))
                
                # Check for transitions
                calculator.check_signal_transitions(symbol, current_analysis, previous_analysis)
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
