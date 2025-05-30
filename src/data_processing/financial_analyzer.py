from __future__ import annotations
import logging
import sqlite3
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

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
        self._round_float = lambda x: round(float(x), 2) if x is not None else None

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

    def analyze_financials(self, financial_data: Dict) -> Dict:
        """Analyze financial data and calculate key metrics"""
        try:
            analysis = {
                'financial_score': 0.0,
                'profitability_score': 0.0,
                'growth_score': 0.0,
                'efficiency_score': 0.0,
                'liquidity_score': 0.0,
                'debt_score': 0.0,
                'valuation_score': 0.0,
                'analysis_summary': [],
                'metrics_used': []
            }
            
            # Add symbol to analysis
            analysis['symbol'] = financial_data['symbol']
            
            # Copy financial data
            for key, value in financial_data.items():
                if key != 'symbol':
                    analysis[key] = value
            
            # Perform analysis
            self._analyze_profitability(financial_data, analysis)
            self._analyze_growth(financial_data, analysis)
            self._analyze_efficiency(financial_data, analysis)
            self._analyze_liquidity(financial_data, analysis)
            self._analyze_debt(financial_data, analysis)
            self._analyze_valuation(financial_data, analysis)
            
            # Calculate final financial score
            self._calculate_final_scores(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing financials: {str(e)}")
            return None

    def _analyze_profitability(self, data: Dict, analysis: Dict):
        """Analyze profitability metrics"""
        try:
            if all(x is not None for x in [data.get('net_income'), data.get('revenue'), data.get('total_assets')]):
                analysis['metrics_used'].extend(['Net Income', 'Revenue', 'ROA'])
                
                # Calculate ROA
                roa = (data['net_income'] / data['total_assets']) * 100
                analysis['roa'] = self._round_float(roa)
                
                # Calculate profit margin
                profit_margin = (data['net_income'] / data['revenue']) * 100
                analysis['profit_margin'] = self._round_float(profit_margin)
                
                # Score profitability
                profitability_score = 0
                
                if roa > 15:
                    profitability_score += 25
                    analysis['analysis_summary'].append(f"Excellent ROA: {roa:.2f}%")
                elif roa > 10:
                    profitability_score += 20
                    analysis['analysis_summary'].append(f"Good ROA: {roa:.2f}%")
                elif roa > 5:
                    profitability_score += 15
                    analysis['analysis_summary'].append(f"Average ROA: {roa:.2f}%")
                elif roa > 0:
                    profitability_score += 10
                    analysis['analysis_summary'].append(f"Low ROA: {roa:.2f}%")
                else:
                    profitability_score -= 10
                    analysis['analysis_summary'].append(f"Negative ROA: {roa:.2f}%")
                
                if profit_margin > 20:
                    profitability_score += 25
                    analysis['analysis_summary'].append(f"Excellent profit margin: {profit_margin:.2f}%")
                elif profit_margin > 15:
                    profitability_score += 20
                    analysis['analysis_summary'].append(f"Good profit margin: {profit_margin:.2f}%")
                elif profit_margin > 10:
                    profitability_score += 15
                    analysis['analysis_summary'].append(f"Average profit margin: {profit_margin:.2f}%")
                elif profit_margin > 5:
                    profitability_score += 10
                    analysis['analysis_summary'].append(f"Low profit margin: {profit_margin:.2f}%")
                else:
                    profitability_score -= 10
                    analysis['analysis_summary'].append(f"Very low profit margin: {profit_margin:.2f}%")
                
                analysis['profitability_score'] = self._round_float(profitability_score)
                
        except Exception as e:
            logger.error(f"Error analyzing profitability: {e}")

    def _analyze_growth(self, data: Dict, analysis: Dict):
        """Analyze growth metrics"""
        try:
            if all(x is not None for x in [data.get('revenue_growth'), data.get('eps_growth')]):
                analysis['metrics_used'].extend(['Revenue Growth', 'EPS Growth'])
                
                revenue_growth = data['revenue_growth']
                eps_growth = data['eps_growth']
                
                growth_score = 0
                
                # Score revenue growth
                if revenue_growth > 20:
                    growth_score += 25
                    analysis['analysis_summary'].append(f"Excellent revenue growth: {revenue_growth:.2f}%")
                elif revenue_growth > 15:
                    growth_score += 20
                    analysis['analysis_summary'].append(f"Strong revenue growth: {revenue_growth:.2f}%")
                elif revenue_growth > 10:
                    growth_score += 15
                    analysis['analysis_summary'].append(f"Good revenue growth: {revenue_growth:.2f}%")
                elif revenue_growth > 5:
                    growth_score += 10
                    analysis['analysis_summary'].append(f"Moderate revenue growth: {revenue_growth:.2f}%")
                elif revenue_growth > 0:
                    growth_score += 5
                    analysis['analysis_summary'].append(f"Low revenue growth: {revenue_growth:.2f}%")
                else:
                    growth_score -= 10
                    analysis['analysis_summary'].append(f"Negative revenue growth: {revenue_growth:.2f}%")
                
                # Score EPS growth
                if eps_growth > 20:
                    growth_score += 25
                    analysis['analysis_summary'].append(f"Excellent EPS growth: {eps_growth:.2f}%")
                elif eps_growth > 15:
                    growth_score += 20
                    analysis['analysis_summary'].append(f"Strong EPS growth: {eps_growth:.2f}%")
                elif eps_growth > 10:
                    growth_score += 15
                    analysis['analysis_summary'].append(f"Good EPS growth: {eps_growth:.2f}%")
                elif eps_growth > 5:
                    growth_score += 10
                    analysis['analysis_summary'].append(f"Moderate EPS growth: {eps_growth:.2f}%")
                elif eps_growth > 0:
                    growth_score += 5
                    analysis['analysis_summary'].append(f"Low EPS growth: {eps_growth:.2f}%")
                else:
                    growth_score -= 10
                    analysis['analysis_summary'].append(f"Negative EPS growth: {eps_growth:.2f}%")
                
                analysis['growth_score'] = self._round_float(growth_score)
                
        except Exception as e:
            logger.error(f"Error analyzing growth: {e}")

    def _analyze_efficiency(self, data: Dict, analysis: Dict):
        """Analyze efficiency metrics"""
        try:
            if all(x is not None for x in [data.get('total_assets'), data.get('revenue'), data.get('inventory')]):
                analysis['metrics_used'].extend(['Asset Turnover', 'Inventory Turnover'])
                
                # Calculate asset turnover
                asset_turnover = data['revenue'] / data['total_assets']
                analysis['asset_turnover'] = self._round_float(asset_turnover)
                
                # Calculate inventory turnover
                inventory_turnover = data['revenue'] / data['inventory'] if data['inventory'] != 0 else 0
                analysis['inventory_turnover'] = self._round_float(inventory_turnover)
                
                efficiency_score = 0
                
                # Score asset turnover
                if asset_turnover > 2:
                    efficiency_score += 25
                    analysis['analysis_summary'].append(f"Excellent asset turnover: {asset_turnover:.2f}")
                elif asset_turnover > 1.5:
                    efficiency_score += 20
                    analysis['analysis_summary'].append(f"Good asset turnover: {asset_turnover:.2f}")
                elif asset_turnover > 1:
                    efficiency_score += 15
                    analysis['analysis_summary'].append(f"Average asset turnover: {asset_turnover:.2f}")
                elif asset_turnover > 0.5:
                    efficiency_score += 10
                    analysis['analysis_summary'].append(f"Low asset turnover: {asset_turnover:.2f}")
                else:
                    efficiency_score -= 10
                    analysis['analysis_summary'].append(f"Very low asset turnover: {asset_turnover:.2f}")
                
                # Score inventory turnover
                if inventory_turnover > 10:
                    efficiency_score += 25
                    analysis['analysis_summary'].append(f"Excellent inventory turnover: {inventory_turnover:.2f}")
                elif inventory_turnover > 7:
                    efficiency_score += 20
                    analysis['analysis_summary'].append(f"Good inventory turnover: {inventory_turnover:.2f}")
                elif inventory_turnover > 5:
                    efficiency_score += 15
                    analysis['analysis_summary'].append(f"Average inventory turnover: {inventory_turnover:.2f}")
                elif inventory_turnover > 3:
                    efficiency_score += 10
                    analysis['analysis_summary'].append(f"Low inventory turnover: {inventory_turnover:.2f}")
                else:
                    efficiency_score -= 10
                    analysis['analysis_summary'].append(f"Very low inventory turnover: {inventory_turnover:.2f}")
                
                analysis['efficiency_score'] = self._round_float(efficiency_score)
                
        except Exception as e:
            logger.error(f"Error analyzing efficiency: {e}")

    def _analyze_liquidity(self, data: Dict, analysis: Dict):
        """Analyze liquidity metrics"""
        try:
            if all(x is not None for x in [data.get('current_assets'), data.get('current_liabilities')]):
                analysis['metrics_used'].extend(['Current Ratio', 'Quick Ratio'])
                
                # Calculate current ratio
                current_ratio = data['current_assets'] / data['current_liabilities']
                analysis['current_ratio'] = self._round_float(current_ratio)
                
                # Calculate quick ratio
                quick_ratio = (data['current_assets'] - data.get('inventory', 0)) / data['current_liabilities']
                analysis['quick_ratio'] = self._round_float(quick_ratio)
                
                liquidity_score = 0
                
                # Score current ratio
                if current_ratio > 2:
                    liquidity_score += 25
                    analysis['analysis_summary'].append(f"Excellent current ratio: {current_ratio:.2f}")
                elif current_ratio > 1.5:
                    liquidity_score += 20
                    analysis['analysis_summary'].append(f"Good current ratio: {current_ratio:.2f}")
                elif current_ratio > 1:
                    liquidity_score += 15
                    analysis['analysis_summary'].append(f"Average current ratio: {current_ratio:.2f}")
                elif current_ratio > 0.8:
                    liquidity_score += 10
                    analysis['analysis_summary'].append(f"Low current ratio: {current_ratio:.2f}")
                else:
                    liquidity_score -= 10
                    analysis['analysis_summary'].append(f"Very low current ratio: {current_ratio:.2f}")
                
                # Score quick ratio
                if quick_ratio > 1.5:
                    liquidity_score += 25
                    analysis['analysis_summary'].append(f"Excellent quick ratio: {quick_ratio:.2f}")
                elif quick_ratio > 1:
                    liquidity_score += 20
                    analysis['analysis_summary'].append(f"Good quick ratio: {quick_ratio:.2f}")
                elif quick_ratio > 0.8:
                    liquidity_score += 15
                    analysis['analysis_summary'].append(f"Average quick ratio: {quick_ratio:.2f}")
                elif quick_ratio > 0.5:
                    liquidity_score += 10
                    analysis['analysis_summary'].append(f"Low quick ratio: {quick_ratio:.2f}")
                else:
                    liquidity_score -= 10
                    analysis['analysis_summary'].append(f"Very low quick ratio: {quick_ratio:.2f}")
                
                analysis['liquidity_score'] = self._round_float(liquidity_score)
                
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {e}")

    def _analyze_debt(self, data: Dict, analysis: Dict):
        """Analyze debt metrics"""
        try:
            if all(x is not None for x in [data.get('total_debt'), data.get('total_assets'), data.get('ebitda')]):
                analysis['metrics_used'].extend(['Debt-to-Equity', 'Interest Coverage'])
                
                # Calculate debt-to-equity ratio
                debt_to_equity = data['total_debt'] / (data['total_assets'] - data['total_debt'])
                analysis['debt_to_equity'] = self._round_float(debt_to_equity)
                
                # Calculate interest coverage ratio
                interest_coverage = data['ebitda'] / data.get('interest_expense', 1)
                analysis['interest_coverage'] = self._round_float(interest_coverage)
                
                debt_score = 0
                
                # Score debt-to-equity ratio
                if debt_to_equity < 0.5:
                    debt_score += 25
                    analysis['analysis_summary'].append(f"Excellent debt-to-equity ratio: {debt_to_equity:.2f}")
                elif debt_to_equity < 1:
                    debt_score += 20
                    analysis['analysis_summary'].append(f"Good debt-to-equity ratio: {debt_to_equity:.2f}")
                elif debt_to_equity < 1.5:
                    debt_score += 15
                    analysis['analysis_summary'].append(f"Average debt-to-equity ratio: {debt_to_equity:.2f}")
                elif debt_to_equity < 2:
                    debt_score += 10
                    analysis['analysis_summary'].append(f"High debt-to-equity ratio: {debt_to_equity:.2f}")
                else:
                    debt_score -= 10
                    analysis['analysis_summary'].append(f"Very high debt-to-equity ratio: {debt_to_equity:.2f}")
                
                # Score interest coverage ratio
                if interest_coverage > 5:
                    debt_score += 25
                    analysis['analysis_summary'].append(f"Excellent interest coverage: {interest_coverage:.2f}")
                elif interest_coverage > 3:
                    debt_score += 20
                    analysis['analysis_summary'].append(f"Good interest coverage: {interest_coverage:.2f}")
                elif interest_coverage > 2:
                    debt_score += 15
                    analysis['analysis_summary'].append(f"Average interest coverage: {interest_coverage:.2f}")
                elif interest_coverage > 1:
                    debt_score += 10
                    analysis['analysis_summary'].append(f"Low interest coverage: {interest_coverage:.2f}")
                else:
                    debt_score -= 10
                    analysis['analysis_summary'].append(f"Very low interest coverage: {interest_coverage:.2f}")
                
                analysis['debt_score'] = self._round_float(debt_score)
                
        except Exception as e:
            logger.error(f"Error analyzing debt: {e}")

    def _analyze_valuation(self, data: Dict, analysis: Dict):
        """Analyze valuation metrics"""
        try:
            if all(x is not None for x in [data.get('market_cap'), data.get('eps'), data.get('book_value')]):
                analysis['metrics_used'].extend(['P/E Ratio', 'P/B Ratio'])
                
                # Calculate P/E ratio
                pe_ratio = data['market_cap'] / (data['eps'] * data.get('shares_outstanding', 1))
                analysis['pe_ratio'] = self._round_float(pe_ratio)
                
                # Calculate P/B ratio
                pb_ratio = data['market_cap'] / (data['book_value'] * data.get('shares_outstanding', 1))
                analysis['pb_ratio'] = self._round_float(pb_ratio)
                
                valuation_score = 0
                
                # Score P/E ratio
                if pe_ratio < 10:
                    valuation_score += 25
                    analysis['analysis_summary'].append(f"Excellent P/E ratio: {pe_ratio:.2f}")
                elif pe_ratio < 15:
                    valuation_score += 20
                    analysis['analysis_summary'].append(f"Good P/E ratio: {pe_ratio:.2f}")
                elif pe_ratio < 20:
                    valuation_score += 15
                    analysis['analysis_summary'].append(f"Average P/E ratio: {pe_ratio:.2f}")
                elif pe_ratio < 25:
                    valuation_score += 10
                    analysis['analysis_summary'].append(f"High P/E ratio: {pe_ratio:.2f}")
                else:
                    valuation_score -= 10
                    analysis['analysis_summary'].append(f"Very high P/E ratio: {pe_ratio:.2f}")
                
                # Score P/B ratio
                if pb_ratio < 1:
                    valuation_score += 25
                    analysis['analysis_summary'].append(f"Excellent P/B ratio: {pb_ratio:.2f}")
                elif pb_ratio < 1.5:
                    valuation_score += 20
                    analysis['analysis_summary'].append(f"Good P/B ratio: {pb_ratio:.2f}")
                elif pb_ratio < 2:
                    valuation_score += 15
                    analysis['analysis_summary'].append(f"Average P/B ratio: {pb_ratio:.2f}")
                elif pb_ratio < 3:
                    valuation_score += 10
                    analysis['analysis_summary'].append(f"High P/B ratio: {pb_ratio:.2f}")
                else:
                    valuation_score -= 10
                    analysis['analysis_summary'].append(f"Very high P/B ratio: {pb_ratio:.2f}")
                
                analysis['valuation_score'] = self._round_float(valuation_score)
                
        except Exception as e:
            logger.error(f"Error analyzing valuation: {e}")

    def _calculate_final_scores(self, analysis: Dict):
        """Calculate final financial score"""
        try:
            # Calculate weighted financial score
            weights = {
                'profitability_score': 0.25,
                'growth_score': 0.20,
                'efficiency_score': 0.15,
                'liquidity_score': 0.15,
                'debt_score': 0.15,
                'valuation_score': 0.10
            }
            
            financial_score = sum(
                analysis[score] * weight
                for score, weight in weights.items()
                if score in analysis
            )
            
            analysis['financial_score'] = self._round_float(financial_score)
            
            # Add overall financial health assessment
            if financial_score >= 80:
                analysis['financial_health'] = 'EXCELLENT'
            elif financial_score >= 60:
                analysis['financial_health'] = 'GOOD'
            elif financial_score >= 40:
                analysis['financial_health'] = 'AVERAGE'
            elif financial_score >= 20:
                analysis['financial_health'] = 'POOR'
            else:
                analysis['financial_health'] = 'CRITICAL'
            
            analysis['analysis_summary'].append(
                f"Overall financial health: {analysis['financial_health']} "
                f"(Score: {financial_score:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Error calculating final scores: {e}")
            analysis['financial_score'] = 0.00
            analysis['financial_health'] = 'UNKNOWN'
