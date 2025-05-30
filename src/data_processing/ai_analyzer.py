from __future__ import annotations
import logging
import os
import re
import json
import requests
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Handles AI-based analysis and signal adjustment for stock data.
    
    This class integrates AI models to enhance stock analysis by providing deeper insights
    and adjusting trading signals based on AI recommendations.
    
    Attributes:
        None
    """
    
    def __init__(self) -> None:
        """Initialize the AIAnalyzer."""
        pass

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

    def _retry_with_backoff(self, func, max_retries=3, initial_delay=1, max_delay=32):
        """Helper method to retry operations with exponential backoff.
        
        Args:
            func: Function to retry.
            max_retries (int): Maximum number of retries.
            initial_delay (int): Initial delay between retries in seconds.
            max_delay (int): Maximum delay between retries in seconds.
            
        Returns:
            The result of the function if successful, None otherwise.
        """
        import time
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
        """Validate the DeepSeek API key with a simple test call.
        
        Returns:
            bool: True if API key is valid, False otherwise.
        """
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
        """Call AI model for analysis with improved error handling and retries.
        
        Args:
            prompt (str): Prompt to send to the AI model.
            
        Returns:
            Dict: Parsed AI analysis results.
        """
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

    def analyze_with_ai(self, symbol: str, technical_data: Dict, financial_data: Dict) -> Dict:
        """Analyze stock data using AI to enhance signal generation with focus on investment perspective.
        
        Args:
            symbol (str): Stock symbol to analyze.
            technical_data (Dict): Technical analysis data.
            financial_data (Dict): Financial analysis data.
            
        Returns:
            Dict: AI-enhanced analysis results.
        """
        try:
            logger.info(f"Starting AI analysis for symbol: {symbol}")
            
            # Placeholder for dividend analysis (assuming it's part of financial_data or fetched separately)
            dividend_analysis = financial_data.get('dividend_analysis', {})
            
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
                    'reports': financial_data.get('reports', []),
                    'announcements': financial_data.get('recent_announcements', [])
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
                
                logger.info(f"Successfully processed AI investment analysis for {symbol}")
                return processed_analysis
            else:
                logger.warning(f"No AI analysis returned for {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            return None

    def adjust_signal_with_ai(self, technical_analysis: Dict, financial_analysis: Dict) -> Dict:
        """Adjust trading signals using AI-based analysis.
        
        Args:
            technical_analysis (Dict): Technical analysis results.
            financial_analysis (Dict): Financial analysis results.
            
        Returns:
            Dict: Adjusted analysis with AI insights.
        """
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
