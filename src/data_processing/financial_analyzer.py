from __future__ import annotations
import logging
import sqlite3
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """Handles financial analysis of stock data including reports and announcements.
    
    This class performs analysis on financial reports, announcements, and dividend data
    to generate financial scores and signals for stocks.
    
    Attributes:
        db_path (str): Path to the main analysis database
        dividend_db_path (str): Path to the dividend database
    """
    
    def __init__(self, db_path: str, dividend_db_path: str) -> None:
        """Initialize the FinancialAnalyzer with database paths.
        
        Args:
            db_path (str): Path to the main analysis database
            dividend_db_path (str): Path to the dividend database
        """
        self.db_path = db_path
        self.dividend_db_path = dividend_db_path

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

    def analyze_financial_data(self, symbol: str) -> Dict:
        """Analyze financial reports and news for the symbol.
        
        Args:
            symbol (str): Stock symbol to analyze.
            
        Returns:
            Dict: Financial analysis results including score and signal.
        """
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

    def read_psx_announcements(self) -> Dict:
        """Read and process PSX announcements from Excel file.
        
        Returns:
            Dict: Dictionary of announcements by symbol.
        """
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
        """Analyze the potential impact of an announcement.
        
        Args:
            title (str): Title of the announcement.
            announcement_type (str): Type of announcement.
            
        Returns:
            float: Impact score of the announcement.
        """
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

    def analyze_dividend_data(self, symbol: str) -> Dict:
        """Analyze dividend data for a symbol.
        
        Args:
            symbol (str): Stock symbol to analyze.
            
        Returns:
            Dict: Dividend analysis results.
        """
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
