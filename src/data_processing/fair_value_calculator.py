from typing import Dict, List
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

logger = logging.getLogger(__name__)

class FairValueCalculator:
    def __init__(self):
        """Initialize the FairValueCalculator with database path and headers"""
        self.db_path = 'data/databases/production/fairvalue.db'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database and create necessary tables"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tradingview_ta table with updated schema
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
            
            # Verify table creation
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tradingview_ta'")
            if not cursor.fetchone():
                raise Exception("Failed to create tradingview_ta table")
            
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
            
            conn.commit()
            
            # Verify tables exist and have correct structure
            cursor.execute("SELECT COUNT(*) FROM tradingview_ta")
            logger.info(f"tradingview_ta table initialized with {cursor.fetchone()[0]} records")
            
            cursor.execute("SELECT COUNT(*) FROM tradingview_signals")
            logger.info(f"tradingview_signals table initialized with {cursor.fetchone()[0]} records")
            
            cursor.execute("SELECT COUNT(*) FROM financial_reports")
            logger.info(f"financial_reports table initialized with {cursor.fetchone()[0]} records")
            
            conn.close()
            logger.info(f"Database initialized successfully at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            if 'conn' in locals():
                conn.close()
            raise  # Re-raise the exception to handle it in the calling code

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

    def analyze_with_deepseek(self, symbol: str, financial_data: Dict, announcements: List[Dict]) -> Dict:
        """Analyze stock data using DeepSeek AI"""
        try:
            # Get API key from environment
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                logger.warning("DeepSeek API key not found in environment variables")
                return None
            
            # Prepare financial data summary
            financial_summary = f"""
Financial Metrics:
- EPS Growth: {financial_data.get('eps_growth', 'N/A')}%
- Revenue Growth: {financial_data.get('revenue_growth', 'N/A')}%
- Profit Margin: {financial_data.get('profit_margin', 'N/A')}%
- Debt-to-Equity: {financial_data.get('debt_to_equity', 'N/A')}
- Current Ratio: {financial_data.get('current_ratio', 'N/A')}
- ROE: {financial_data.get('roe', 'N/A')}%
"""
            
            # Prepare announcements summary
            announcements_summary = "Recent Announcements:\n"
            for announcement in announcements:
                announcements_summary += f"- {announcement['date']}: {announcement['title']}\n"
            
            # Prepare the prompt
            prompt = f"""You are a professional financial analyst. Analyze this stock:

Symbol: {symbol}

{financial_summary}

{announcements_summary}

Provide a detailed analysis in the following format:
1. FINANCIAL HEALTH
- Overall financial health assessment
- Key strengths and weaknesses
- Growth potential and risks

2. RECENT DEVELOPMENTS
- Impact of recent announcements
- Market sentiment analysis
- Competitive position

3. TECHNICAL ANALYSIS
- Current market position
- Support and resistance levels
- Trend analysis

4. INVESTMENT RECOMMENDATION
- Clear recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
- Detailed rationale
- Risk factors
- Price targets (if applicable)
"""
            
            # Call DeepSeek API
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
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                try:
                    analysis = response.json()['choices'][0]['message']['content']
                    
                    # Parse the analysis to extract key components
                    ai_analysis = {
                        'financial_health': '',
                        'recent_developments': '',
                        'technical_analysis': '',
                        'recommendation': '',
                        'confidence_score': 0.0,
                        'price_target': None
                    }
                    
                    # Extract sections from the analysis
                    sections = analysis.split('\n\n')
                    for section in sections:
                        if 'FINANCIAL HEALTH' in section:
                            ai_analysis['financial_health'] = section
                        elif 'RECENT DEVELOPMENTS' in section:
                            ai_analysis['recent_developments'] = section
                        elif 'TECHNICAL ANALYSIS' in section:
                            ai_analysis['technical_analysis'] = section
                        elif 'INVESTMENT RECOMMENDATION' in section:
                            ai_analysis['recommendation'] = section
                            
                            # Extract confidence score and price target
                            if 'Strong Buy' in section:
                                ai_analysis['confidence_score'] = 0.9
                            elif 'Buy' in section:
                                ai_analysis['confidence_score'] = 0.7
                            elif 'Hold' in section:
                                ai_analysis['confidence_score'] = 0.5
                            elif 'Sell' in section:
                                ai_analysis['confidence_score'] = 0.3
                            elif 'Strong Sell' in section:
                                ai_analysis['confidence_score'] = 0.1
                            
                            # Try to extract price target
                            price_target_match = re.search(r'price target.*?(\d+\.?\d*)', section.lower())
                            if price_target_match:
                                ai_analysis['price_target'] = float(price_target_match.group(1))
                    
                    return ai_analysis
                    
                except Exception as e:
                    logger.error(f"Error parsing DeepSeek response: {e}")
                    return None
            else:
                logger.error(f"DeepSeek API request failed with status code {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error in DeepSeek analysis: {e}")
            return None

    def adjust_signal_with_ai(self, technical_analysis: Dict, financial_analysis: Dict) -> Dict:
        """Adjust trading signals using AI-based analysis"""
        try:
            # Initialize adjusted analysis
            adjusted_analysis = technical_analysis.copy()
            
            # Get DeepSeek AI analysis
            symbol = financial_analysis.get('symbol', '')
            announcements = financial_analysis.get('recent_announcements', [])
            ai_analysis = self.analyze_with_deepseek(symbol, financial_analysis, announcements)
            
            if ai_analysis:
                # Calculate weighted scores
                technical_weight = 0.4
                financial_weight = 0.3
                ai_weight = 0.3
                
                # Adjust technical score based on financial analysis and AI insights
                if financial_analysis['confidence'] > 0.5:  # Only adjust if we have sufficient data
                    # Calculate announcement impact
                    announcement_impact = 0.0
                    if financial_analysis['recent_announcements']:
                        announcement_impact = sum(a['impact'] for a in financial_analysis['recent_announcements'])
                    
                    # Calculate AI impact
                    ai_impact = 0.0
                    if ai_analysis['confidence_score'] > 0.7:
                        ai_impact = 0.3
                    elif ai_analysis['confidence_score'] > 0.5:
                        ai_impact = 0.1
                    elif ai_analysis['confidence_score'] < 0.3:
                        ai_impact = -0.3
                    elif ai_analysis['confidence_score'] < 0.5:
                        ai_impact = -0.1
                    
                    adjusted_score = (
                        technical_analysis['technical_score'] * technical_weight +
                        financial_analysis['financial_score'] * financial_weight +
                        (ai_impact * 100) * ai_weight  # Scale AI impact
                    )
                    
                    # Determine if signal should be adjusted
                    if (technical_analysis['signal_type'] in ['STRONG_SELL', 'SELL'] and 
                        (financial_analysis['financial_signal'] in ['STRONG_BUY', 'BUY'] or 
                         announcement_impact > 0.3 or ai_impact > 0.2)):
                        # If technical is bearish but other indicators are bullish, move to neutral
                        if abs(technical_analysis['technical_score']) > 40:
                            adjusted_analysis['signal_type'] = 'NEUTRAL'
                            adjusted_analysis['signal_strength'] = 0.5
                            adjusted_analysis['analysis_summary'].append(
                                "Signal adjusted to NEUTRAL due to strong financial performance, positive announcements, and AI analysis"
                            )
                    
                    elif (technical_analysis['signal_type'] in ['STRONG_BUY', 'BUY'] and 
                          (financial_analysis['financial_signal'] in ['STRONG_SELL', 'SELL'] or 
                           announcement_impact < -0.3 or ai_impact < -0.2)):
                        # If technical is bullish but other indicators are bearish, reduce signal strength
                        if technical_analysis['signal_strength'] > 0.7:
                            adjusted_analysis['signal_strength'] *= 0.7
                            adjusted_analysis['analysis_summary'].append(
                                "Signal strength reduced due to poor financial performance, negative announcements, and AI analysis"
                            )
                    
                    # Add AI analysis to summary
                    adjusted_analysis['analysis_summary'].append("\nAI Analysis:")
                    adjusted_analysis['analysis_summary'].append(ai_analysis['financial_health'])
                    adjusted_analysis['analysis_summary'].append(ai_analysis['recent_developments'])
                    adjusted_analysis['analysis_summary'].append(ai_analysis['technical_analysis'])
                    adjusted_analysis['analysis_summary'].append(ai_analysis['recommendation'])
                    
                    # Update confidence score
                    adjusted_analysis['confidence_score'] = (
                        technical_analysis['confidence_score'] * technical_weight +
                        financial_analysis['confidence'] * financial_weight +
                        ai_analysis['confidence_score'] * ai_weight
                    )
                    
                    # Add price target if available
                    if ai_analysis['price_target']:
                        adjusted_analysis['price_target'] = ai_analysis['price_target']
            
            return adjusted_analysis
            
        except Exception as e:
            logger.error(f"Error adjusting signal with AI: {e}")
            return technical_analysis

    def analyze_stock_indicators(self, stock_data: Dict) -> Dict:
        """Analyze stock data using multiple technical indicators with enhanced logic"""
        try:
            # Get previous analysis from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM tradingview_signals 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            """, (stock_data['symbol'],))
            
            previous_analysis = cursor.fetchone()
            conn.close()
            
            # Convert previous analysis to dict if exists
            if previous_analysis:
                columns = [description[0] for description in cursor.description]
                previous_analysis = dict(zip(columns, previous_analysis))
            
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
            
            # 1. Trend Analysis (40 points)
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
                
                # Enhanced trend analysis
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
                    if price_above_sma20 > 5:  # Price significantly above SMA20
                        trend_strength += 25
                        analysis['analysis_summary'].append(f"Strong uptrend: Price {price_above_sma20:.2f}% above SMA20")
                    else:
                        trend_strength += 15
                        analysis['analysis_summary'].append(f"Moderate uptrend: Price {price_above_sma20:.2f}% above SMA20")
                # Strong downtrend conditions
                elif close < sma20 < sma50 < sma200:
                    if price_above_sma20 < -5:  # Price significantly below SMA20
                        trend_strength -= 25
                        analysis['analysis_summary'].append(f"Strong downtrend: Price {abs(price_above_sma20):.2f}% below SMA20")
                    else:
                        trend_strength -= 15
                        analysis['analysis_summary'].append(f"Moderate downtrend: Price {abs(price_above_sma20):.2f}% below SMA20")
                
                analysis['trend_score'] = trend_strength
            
            # 2. Momentum Analysis (35 points)
            if all(x is not None for x in [stock_data['rsi'], stock_data['macd'], stock_data['macd_signal'], stock_data['ao']]):
                analysis['indicators_used'].extend(['RSI', 'MACD', 'AO'])
                rsi = stock_data['rsi']
                macd = stock_data['macd']
                macd_signal = stock_data['macd_signal']
                ao = stock_data['ao']
                
                momentum_strength = 0
                
                # Enhanced RSI Analysis (15 points)
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
                
                # Enhanced MACD Analysis (10 points)
                macd_diff = macd - macd_signal
                macd_diff_percent = (macd_diff / abs(macd_signal)) * 100 if macd_signal != 0 else 0
                
                # Check for MACD crossover
                if previous_analysis:
                    prev_macd = previous_analysis.get('macd', 0)
                    prev_macd_signal = previous_analysis.get('macd_signal', 0)
                    
                    if macd > macd_signal and prev_macd <= prev_macd_signal:
                        momentum_strength += 10
                        analysis['analysis_summary'].append("Bullish MACD crossover detected")
                    elif macd < macd_signal and prev_macd >= prev_macd_signal:
                        momentum_strength -= 10
                        analysis['analysis_summary'].append("Bearish MACD crossover detected")
                
                if macd_diff_percent > 5:
                    momentum_strength += 5
                    analysis['analysis_summary'].append(f"Strong bullish MACD: {macd_diff_percent:.2f}% above signal")
                elif macd_diff_percent > 2:
                    momentum_strength += 2
                    analysis['analysis_summary'].append(f"Moderate bullish MACD: {macd_diff_percent:.2f}% above signal")
                elif macd_diff_percent < -5:
                    momentum_strength -= 5
                    analysis['analysis_summary'].append(f"Strong bearish MACD: {abs(macd_diff_percent):.2f}% below signal")
                elif macd_diff_percent < -2:
                    momentum_strength -= 2
                    analysis['analysis_summary'].append(f"Moderate bearish MACD: {abs(macd_diff_percent):.2f}% below signal")
                
                # Enhanced Awesome Oscillator (AO) Analysis (10 points)
                ao_abs = abs(ao)
                ao_threshold = 50  # Threshold for strong signals
                
                # Check for AO crossover
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
            
            # 3. Volume Analysis (20 points)
            if all(x is not None for x in [stock_data['volume'], stock_data['change']]):
                analysis['indicators_used'].append('Volume')
                volume = stock_data['volume']
                change = stock_data['change']
                change_percent = stock_data.get('change_percent', 0)
                
                volume_strength = 0
                
                # Enhanced volume analysis
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
                
                # Check for volume trend
                if previous_analysis:
                    prev_volume = previous_analysis.get('volume', 0)
                    if volume > prev_volume * 1.5:  # 50% volume increase
                        volume_strength += 5
                        analysis['analysis_summary'].append("Significant volume increase detected")
                    elif volume < prev_volume * 0.5:  # 50% volume decrease
                        volume_strength -= 5
                        analysis['analysis_summary'].append("Significant volume decrease detected")
                
                analysis['volume_score'] = volume_strength
            
            # 4. Volatility Analysis (20 points)
            if all(x is not None for x in [stock_data['bb_upper'], stock_data['bb_lower'], stock_data['close']]):
                analysis['indicators_used'].append('Bollinger Bands')
                bb_upper = stock_data['bb_upper']
                bb_lower = stock_data['bb_lower']
                close = stock_data['close']
                
                bb_range = bb_upper - bb_lower
                volatility = bb_range / close * 100
                price_position = (close - bb_lower) / bb_range * 100
                
                volatility_strength = 0
                
                # Enhanced volatility analysis
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
                
                # Add price position relative to BB
                if price_position > 80:
                    analysis['analysis_summary'].append(f"Price near upper BB: {price_position:.2f}% of range")
                    volatility_strength -= 5  # Additional bearish signal
                elif price_position < 20:
                    analysis['analysis_summary'].append(f"Price near lower BB: {price_position:.2f}% of range")
                    volatility_strength += 5  # Additional bullish signal
                
                analysis['volatility_score'] = volatility_strength
            
            # Calculate final scores
            analysis['technical_score'] = (
                analysis['trend_score'] +
                analysis['momentum_score'] +
                analysis['volume_score'] +
                analysis['volatility_score']
            )
            
            # Enhanced signal determination with more granular thresholds
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
            
            # Calculate confidence score based on available indicators and their strength
            total_possible_indicators = 4  # Trend, Momentum, Volume, Volatility
            available_indicators = len(analysis['indicators_used'])
            indicator_completeness = available_indicators / total_possible_indicators
            
            # Adjust confidence based on signal strength and indicator agreement
            signal_strength_factor = abs(analysis['technical_score']) / 70  # Normalize to max score
            
            # Calculate indicator agreement
            indicator_scores = [
                analysis['trend_score'],
                analysis['momentum_score'],
                analysis['volume_score'],
                analysis['volatility_score']
            ]
            positive_scores = sum(1 for score in indicator_scores if score > 0)
            negative_scores = sum(1 for score in indicator_scores if score < 0)
            indicator_agreement = max(positive_scores, negative_scores) / len(indicator_scores)
            
            analysis['confidence_score'] = (
                indicator_completeness * 0.4 +  # Weight for data completeness
                signal_strength_factor * 0.3 +  # Weight for signal strength
                indicator_agreement * 0.3        # Weight for indicator agreement
            )
            
            # Calculate stop loss and take profit levels with enhanced risk management
            if analysis['support_level'] and analysis['resistance_level'] and stock_data['close']:
                current_price = stock_data['close']
                support = analysis['support_level']
                resistance = analysis['resistance_level']
                
                # Calculate price ranges
                price_to_support = abs(current_price - support)
                price_to_resistance = abs(resistance - current_price)
                
                # Calculate ATR-based volatility if available
                atr = None
                if 'atr' in stock_data and stock_data['atr'] is not None:
                    atr = stock_data['atr']
                    # Use ATR to adjust stop loss and take profit levels
                    atr_multiplier = 2.0  # Standard multiplier for stop loss
                    volatility_adjusted_support = support - (atr * atr_multiplier)
                    volatility_adjusted_resistance = resistance + (atr * atr_multiplier)
                else:
                    # Fallback to percentage-based adjustment
                    volatility_adjusted_support = support * 0.98  # 2% below support
                    volatility_adjusted_resistance = resistance * 1.02  # 2% above resistance
                
                # Set stop loss and take profit based on signal type
                if analysis['signal_type'] in ['BUY', 'STRONG_BUY']:
                    analysis['stop_loss'] = volatility_adjusted_support
                    analysis['take_profit'] = volatility_adjusted_resistance
                    risk = abs(current_price - volatility_adjusted_support)
                    reward = abs(volatility_adjusted_resistance - current_price)
                else:
                    analysis['stop_loss'] = volatility_adjusted_resistance
                    analysis['take_profit'] = volatility_adjusted_support
                    risk = abs(volatility_adjusted_resistance - current_price)
                    reward = abs(current_price - volatility_adjusted_support)
                
                # Calculate risk-reward ratio
                if risk > 0:
                    analysis['risk_reward_ratio'] = reward / risk
                    
                    # Add risk management metrics
                    analysis['risk_metrics'] = {
                        'price_to_support': price_to_support,
                        'price_to_resistance': price_to_resistance,
                        'risk_percent': (risk / current_price) * 100,
                        'reward_percent': (reward / current_price) * 100,
                        'volatility_adjusted': atr is not None,
                        'atr_value': atr
                    }
                    
                    # Add risk assessment to analysis summary
                    risk_assessment = f"Risk-Reward Analysis:\n"
                    risk_assessment += f"Current Price: {current_price:.2f}\n"
                    risk_assessment += f"Stop Loss: {analysis['stop_loss']:.2f} ({analysis['risk_metrics']['risk_percent']:.2f}% risk)\n"
                    risk_assessment += f"Take Profit: {analysis['take_profit']:.2f} ({analysis['risk_metrics']['reward_percent']:.2f}% reward)\n"
                    risk_assessment += f"Risk-Reward Ratio: {analysis['risk_reward_ratio']:.2f}\n"
                    
                    if atr:
                        risk_assessment += f"ATR-based volatility adjustment applied (ATR: {atr:.2f})"
                    else:
                        risk_assessment += "Percentage-based volatility adjustment applied (2%)"
                    
                    analysis['analysis_summary'].append(risk_assessment)
                    
                    # Add risk warning if ratio is unfavorable
                    if analysis['risk_reward_ratio'] < 1.5:
                        analysis['analysis_summary'].append("Warning: Risk-Reward ratio is below recommended minimum of 1.5")
                    elif analysis['risk_reward_ratio'] > 3:
                        analysis['analysis_summary'].append("Strong Risk-Reward ratio above 3.0")
                else:
                    analysis['risk_reward_ratio'] = None
                    analysis['analysis_summary'].append("Warning: Could not calculate risk-reward ratio (risk is zero)")
            
            # After calculating technical analysis, get financial analysis
            financial_analysis = self.analyze_financial_data(stock_data['symbol'])
            
            # Adjust signals using AI
            final_analysis = self.adjust_signal_with_ai(analysis, financial_analysis)
            
            # Check for signal transitions and send notifications
            self.check_signal_transitions(stock_data['symbol'], final_analysis, previous_analysis)
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock indicators: {str(e)}")
            return None

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
        """Save stock analysis to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('''
            INSERT OR REPLACE INTO tradingview_signals
            (symbol, date, signal_type, signal_strength, confidence_score,
             technical_score, trend_score, momentum_score, volume_score,
             volatility_score, support_level, resistance_level, stop_loss,
             take_profit, risk_reward_ratio, analysis_summary, indicators_used,
             last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                current_time
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Successfully saved analysis for {symbol} to database")
            
        except Exception as e:
            logger.error(f"Error saving analysis to database: {e}")
            if 'conn' in locals():
                conn.close()

    def fetch_tradingview_ta_data(self, symbol: str) -> Dict:
        """Fetch data using tradingview_ta library with weekly timeframe"""
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
            
            # Try different symbol formats with retry mechanism
            # Start with the most common format first
            symbol_formats = [
                symbol,           # Just the symbol (most common)
                f"{symbol}.PSX",  # PSX suffix (second most common)
                f"PSX:{symbol}",  # PSX prefix with colon (third most common)
                f"PSX-{symbol}"   # PSX prefix with hyphen (fourth most common)
            ]
            
            data = {}
            success = False
            max_retries = 2
            retry_delay = 1
            
            for symbol_format in symbol_formats:
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
                        
                        # Debug logging for raw data
                        logger.info(f"Raw analysis for {symbol} ({symbol_format}):")
                        logger.info(f"Summary: {analysis.summary}")
                        logger.info(f"Oscillators: {analysis.oscillators}")
                        logger.info(f"Moving Averages: {analysis.moving_averages}")
                        logger.info(f"Indicators: {analysis.indicators}")
                        
                        # Extract summary data
                        if analysis.summary:
                            data.update({
                                'recommendation': analysis.summary.get('RECOMMENDATION'),
                                'buy_signals': analysis.summary.get('BUY'),
                                'sell_signals': analysis.summary.get('SELL'),
                                'neutral_signals': analysis.summary.get('NEUTRAL')
                            })
                            logger.info(f"Extracted summary data for {symbol}: {data}")
                        
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
                                # Calculate absolute change
                                price_metrics['change'] = close - open_price
                                
                                # Calculate percentage change
                                if open_price != 0:
                                    price_metrics['change_percent'] = (price_metrics['change'] / open_price) * 100
                                    logger.info(f"Calculated change: {price_metrics['change']:.2f} ({price_metrics['change_percent']:.2f}%)")
                                else:
                                    logger.warning(f"Open price is zero for {symbol}, cannot calculate percentage change")
                            
                            # Calculate high-low range if we have both high and low
                            if high is not None and low is not None:
                                price_metrics['high_low_range'] = high - low
                                if low != 0:
                                    price_metrics['high_low_range_percent'] = (price_metrics['high_low_range'] / low) * 100
                                    logger.info(f"High-Low range: {price_metrics['high_low_range']:.2f} ({price_metrics['high_low_range_percent']:.2f}%)")
                            
                            # Calculate volatility (standard deviation of price changes)
                            if all(x is not None for x in [close, open_price, high, low]):
                                # Simple volatility calculation based on high-low range
                                price_metrics['volatility'] = price_metrics['high_low_range_percent'] / 2
                                logger.info(f"Calculated volatility: {price_metrics['volatility']:.2f}%")
                            
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
                                logger.info(f"Extracted all indicators for {symbol}: {indicator_data}")
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
        """Analyze stock data and generate AI-based signals"""
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
        """Generate AI-based signal for a single stock"""
        try:
            # Initialize signal dictionary
            signal = {
                'symbol': stock['symbol'],
                'date': stock['date'],
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reasons': []
            }
            
            # Technical Analysis Score (0-100)
            ta_score = 0
            reasons = []
            
            # 1. Trend Analysis (30 points)
            trend_score = 0
            if stock['sma_20'] is not None and stock['sma_50'] is not None:
                if stock['close'] > stock['sma_20'] > stock['sma_50']:
                    trend_score += 15
                    reasons.append("Strong uptrend: Price above both SMAs")
                elif stock['close'] > stock['sma_20']:
                    trend_score += 10
                    reasons.append("Moderate uptrend: Price above SMA20")
                elif stock['close'] < stock['sma_20'] < stock['sma_50']:
                    trend_score -= 15
                    reasons.append("Strong downtrend: Price below both SMAs")
                elif stock['close'] < stock['sma_20']:
                    trend_score -= 10
                    reasons.append("Moderate downtrend: Price below SMA20")
            
            # 2. Momentum Analysis (30 points)
            momentum_score = 0
            if stock['rsi'] is not None:
                if stock['rsi'] > 70:
                    momentum_score -= 10
                    reasons.append("Overbought: RSI above 70")
                elif stock['rsi'] < 30:
                    momentum_score += 10
                    reasons.append("Oversold: RSI below 30")
            
            if stock['macd'] is not None and stock['macd_signal'] is not None:
                if stock['macd'] > stock['macd_signal']:
                    momentum_score += 10
                    reasons.append("Positive MACD crossover")
                else:
                    momentum_score -= 10
                    reasons.append("Negative MACD crossover")
            
            # 3. Volume Analysis (20 points)
            volume_score = 0
            if stock['volume'] is not None and stock['change'] is not None:
                if stock['change'] > 0 and stock['volume'] > 1000000:  # High volume with price increase
                    volume_score += 10
                    reasons.append("High volume with price increase")
                elif stock['change'] < 0 and stock['volume'] > 1000000:  # High volume with price decrease
                    volume_score -= 10
                    reasons.append("High volume with price decrease")
            
            # 4. Volatility Analysis (20 points)
            volatility_score = 0
            if stock['bb_upper'] is not None and stock['bb_lower'] is not None:
                bb_range = stock['bb_upper'] - stock['bb_lower']
                volatility = bb_range / stock['close'] * 100
                price_position = (stock['close'] - stock['bb_lower']) / bb_range * 100
                
                # Volatility score based on BB range
                if volatility > 15:
                    volatility_score = -20
                    reasons.append("Very high volatility: BB range > 15%")
                elif volatility > 10:
                    volatility_score = -15
                    reasons.append("High volatility: BB range > 10%")
                elif volatility < 5:
                    volatility_score = 15
                    reasons.append("Low volatility: BB range < 5%")
                elif volatility < 8:
                    volatility_score = 10
                    reasons.append("Moderate volatility: BB range < 8%")
                
                # Calculate support and resistance levels
                signal['support_level'] = stock['bb_lower']
                signal['resistance_level'] = stock['bb_upper']
                
                # Add price position relative to BB
                if price_position > 80:
                    reasons.append("Price near upper BB: 80% of range")
                    volatility_score -= 5
                elif price_position < 20:
                    reasons.append("Price near lower BB: 20% of range")
                    volatility_score += 5
            
            # Calculate final score
            final_score = trend_score + momentum_score + volume_score + volatility_score
            
            # Determine signal and confidence
            if final_score >= 30:
                signal['signal'] = 'STRONG_BUY'
                signal['confidence'] = min(final_score / 50, 1.0)
            elif final_score >= 15:
                signal['signal'] = 'BUY'
                signal['confidence'] = min(final_score / 40, 0.8)
            elif final_score <= -30:
                signal['signal'] = 'STRONG_SELL'
                signal['confidence'] = min(abs(final_score) / 50, 1.0)
            elif final_score <= -15:
                signal['signal'] = 'SELL'
                signal['confidence'] = min(abs(final_score) / 40, 0.8)
            else:
                signal['signal'] = 'NEUTRAL'
                signal['confidence'] = 0.5
            
            signal['reasons'] = reasons
            signal['score'] = final_score
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {stock['symbol']}: {e}")
            return None

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
                print(f"Skipping {symbol} - current data available")
                skipped += 1
                continue
            
            data = calculator.fetch_tradingview_ta_data(symbol)
            if data:
                print(f"Successfully fetched data for {symbol}")
                successful += 1
            else:
                print(f"Failed to fetch data for {symbol}")
                failed += 1
            
            # Add delay to avoid rate limiting
            time.sleep(2)
        
        print(f"\nData collection completed:")
        print(f"Successfully fetched: {successful} symbols")
        print(f"Failed to fetch: {failed} symbols")
        print(f"Skipped (current data): {skipped} symbols")
        
        # Verify data after fetching
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
        
        # Generate and save signals
        print("\nGenerating stock signals...")
        signals = calculator.analyze_stock_signals()
        print(f"Generated signals for {len(signals)} stocks")
        
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
