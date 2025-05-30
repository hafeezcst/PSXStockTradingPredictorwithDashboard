from __future__ import annotations
import logging
import os
import re
import json
import requests
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dotenv import load_dotenv
import time

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Handles AI-based analysis and signal adjustment for stock data.
    
    This class integrates AI models to enhance stock analysis by providing deeper insights
    and adjusting trading signals based on AI recommendations.
    
    Attributes:
        None
    """
    
    def validate_env_file(self) -> Dict[str, bool]:
        """Validate the .env file and API keys.
        
        Returns:
            Dict[str, bool]: Dictionary containing validation results for each key
        """
        validation_results = {
            'env_file_exists': False,
            'deepseek_key_valid': False,
            'grok_key_valid': False,
            'deepseek_key_format': False,
            'grok_key_format': False
        }
        
        try:
            # Check if .env file exists
            env_path = "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/.env"
            if os.path.exists(env_path):
                validation_results['env_file_exists'] = True
                logger.info(f"Found .env file at {env_path}")
                
                # Load environment variables
                load_dotenv(env_path, override=True)
                
                # Validate DeepSeek API key
                deepseek_key = os.getenv('DEEPSEEK_API_KEY')
                if deepseek_key:
                    validation_results['deepseek_key_valid'] = True
                    if deepseek_key.startswith('ds-') or deepseek_key.startswith('sk-'):
                        validation_results['deepseek_key_format'] = True
                        logger.info("DeepSeek API key format is valid")
                    else:
                        logger.warning("DeepSeek API key has invalid format")
                else:
                    logger.warning("DeepSeek API key not found in .env file")
                
                # Validate Grok API key (XAI)
                grok_key = os.getenv('XAI_API_KEY')
                if grok_key:
                    validation_results['grok_key_valid'] = True
                    if grok_key.startswith('xai-'):
                        validation_results['grok_key_format'] = True
                        logger.info("Grok API key format is valid")
                    else:
                        logger.warning("Grok API key has invalid format")
                else:
                    logger.warning("Grok API key not found in .env file")
                
                # Print validation summary
                print("\nAPI Key Validation Results:")
                print("=" * 30)
                print(f"✓ .env file exists: {validation_results['env_file_exists']}")
                print("\nDeepSeek API Key:")
                print(f"  • Key present: {validation_results['deepseek_key_valid']}")
                print(f"  • Valid format: {validation_results['deepseek_key_format']}")
                if validation_results['deepseek_key_valid'] and not validation_results['deepseek_key_format']:
                    print("  ⚠ Format should start with 'ds-' or 'sk-'")
                
                print("\nGrok API Key (XAI):")
                print(f"  • Key present: {validation_results['grok_key_valid']}")
                print(f"  • Valid format: {validation_results['grok_key_format']}")
                if validation_results['grok_key_valid'] and not validation_results['grok_key_format']:
                    print("  ⚠ Format should start with 'xai-'")
                
                print("\nRecommendations:")
                if not validation_results['env_file_exists']:
                    print("• Create a .env file in the project root")
                if not validation_results['deepseek_key_valid']:
                    print("• Add DeepSeek API key to .env file")
                if not validation_results['grok_key_valid']:
                    print("• Add Grok API key to .env file")
                if validation_results['deepseek_key_valid'] and not validation_results['deepseek_key_format']:
                    print("• Update DeepSeek API key format")
                if validation_results['grok_key_valid'] and not validation_results['grok_key_format']:
                    print("• Update Grok API key format")
                
                print("\nExample .env format:")
                print("DEEPSEEK_API_KEY=ds-your-key-here")
                print("XAI_API_KEY=xai-your-key-here")
                print("=" * 30)
                
            else:
                logger.error(f".env file not found at {env_path}")
                print("\n⚠️  Error: .env file not found!")
                print(f"Please create a .env file at: {env_path}")
                print("\nRequired format:")
                print("DEEPSEEK_API_KEY=ds-your-key-here")
                print("XAI_API_KEY=xai-your-key-here")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating .env file: {str(e)}")
            return validation_results

    def __init__(self) -> None:
        """Initialize the AIAnalyzer."""
        # Validate .env file and API keys first
        self.validate_env_file()
        
        # Load environment variables from the correct path
        env_path = "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/.env"
        if os.path.exists(env_path):
            logger.debug(f"Loading environment variables from {env_path}")
            load_dotenv(env_path, override=True)
        else:
            logger.warning(f".env file not found at {env_path}")
        
        # Load API keys
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.grok_api_key = os.getenv('XAI_API_KEY')  # Using XAI API key for Grok
        
        # Log API key status (safely)
        if self.deepseek_api_key:
            if len(self.deepseek_api_key) < 20:
                logger.warning("DeepSeek API key appears to be incomplete")
            else:
                logger.info("DeepSeek API key loaded successfully")
                logger.debug(f"DeepSeek API key starts with: {self.deepseek_api_key[:8]}...")
        else:
            logger.warning("DeepSeek API key not found in environment variables")
            
        if self.grok_api_key:
            if self.grok_api_key == 'xai-':
                logger.warning("Grok API key appears to be incomplete (only prefix provided)")
            elif len(self.grok_api_key) < 20:
                logger.warning("Grok API key appears to be incomplete")
            else:
                logger.info("Grok API key loaded successfully")
                logger.debug(f"Grok API key starts with: {self.grok_api_key[:8]}...")
        else:
            logger.warning("Grok API key not found in environment variables")
        
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        self.grok_api_url = "https://api.grok.ai/v1/chat/completions"  # Replace with actual Grok API endpoint
        self._round_float = lambda x: round(float(x), 2) if x is not None else None
        self._cache = {}
        self._last_call_time = {}
        self._cooldown = int(os.getenv('ANALYSIS_COOLDOWN', '300'))  # 5 minutes cooldown between calls for same symbol

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

    def _validate_api_keys(self) -> bool:
        """Validate both DeepSeek and Grok API keys."""
        if not self.deepseek_api_key and not self.grok_api_key:
            logger.warning("No API keys found. AI analysis will be disabled.")
            return False
        
        # Validate DeepSeek API key if present
        if self.deepseek_api_key:
            # Remove any whitespace or newlines
            self.deepseek_api_key = self.deepseek_api_key.strip()
            
            # Check for minimum length
            if len(self.deepseek_api_key) < 20:
                logger.error("DeepSeek API key is too short. Please check your API key format.")
                return False
                
            # Check for valid prefix
            if not (self.deepseek_api_key.startswith('ds-') or self.deepseek_api_key.startswith('sk-')):
                logger.error("Invalid DeepSeek API key format. API key should start with 'ds-' or 'sk-'")
                return False
            
            # Log key format (safely)
            logger.debug(f"DeepSeek API key format: {self.deepseek_api_key[:8]}...{self.deepseek_api_key[-4:]}")
            
        # Validate Grok API key if present (using XAI API key)
        if self.grok_api_key:
            # Remove any whitespace or newlines
            self.grok_api_key = self.grok_api_key.strip()
            
            # Check for minimum length
            if len(self.grok_api_key) < 20:
                logger.error("Grok API key is too short. Please check your API key format.")
                return False
                
            # Check for valid prefix (using XAI prefix)
            if not self.grok_api_key.startswith('xai-'):
                logger.error("Invalid Grok API key format. API key should start with 'xai-'")
                return False
            
            # Check if key is complete
            if self.grok_api_key == 'xai-':
                logger.error("Grok API key is incomplete. Please provide the complete API key.")
                return False
            
            # Log key format (safely)
            logger.debug(f"Grok API key format: {self.grok_api_key[:8]}...{self.grok_api_key[-4:]}")
            
        return True

    def call_ai_model(self, prompt: str) -> Dict:
        """Call AI model for analysis with improved error handling and retries.
        
        Args:
            prompt (str): Prompt to send to the AI model.
            
        Returns:
            Dict: Parsed AI analysis results.
        """
        try:
            # Validate API key first
            if not self._validate_api_keys():
                logger.error("Failed to validate API keys")
                return None
            
            # Get API key from environment
            api_key = self.deepseek_api_key or self.grok_api_key
            logger.info("API key validated, proceeding with API call")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a professional financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            # Define the API call function
            def make_api_call():
                logger.info("Making API call to OpenAI...")
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
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

    def analyze_with_ai(self, symbol: str, technical_data: Dict, financial_data: Dict) -> Optional[Dict]:
        """Analyze stock data using AI"""
        try:
            # Check if API key is available
            if not self._validate_api_keys():
                logger.warning("Skipping AI analysis due to missing API keys")
                return None

            # Check cache first
            current_time = datetime.now().timestamp()
            if symbol in self._cache:
                last_call = self._last_call_time.get(symbol, 0)
                if current_time - last_call < self._cooldown:
                    logger.info(f"Using cached AI analysis for {symbol}")
                    return self._cache[symbol]

            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data(symbol, technical_data, financial_data)
            
            # Make API call
            response = self._make_api_call(analysis_data)
            
            if response:
                # Update cache
                self._cache[symbol] = response
                self._last_call_time[symbol] = current_time
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {str(e)}")
            return None

    def _prepare_analysis_data(self, symbol: str, technical_data: Dict, financial_data: Dict) -> Dict:
        """Prepare data for AI analysis"""
        try:
            # Technical Analysis Data
            technical_metrics = {
                'current_price': technical_data.get('close'),
                'price_change': technical_data.get('change'),
                'volume': technical_data.get('volume'),
                'rsi': technical_data.get('rsi'),
                'macd': technical_data.get('macd'),
                'sma_20': technical_data.get('sma_20'),
                'sma_50': technical_data.get('sma_50'),
                'sma_200': technical_data.get('sma_200'),
                'bb_upper': technical_data.get('bb_upper'),
                'bb_lower': technical_data.get('bb_lower')
            }
            
            # Financial Analysis Data
            financial_metrics = {
                'eps_growth': financial_data.get('eps_growth'),
                'revenue_growth': financial_data.get('revenue_growth'),
                'profit_margin': financial_data.get('profit_margin'),
                'debt_to_equity': financial_data.get('debt_to_equity'),
                'current_ratio': financial_data.get('current_ratio'),
                'roe': financial_data.get('roe')
            }
            
            # Recent Announcements
            announcements = technical_data.get('recent_announcements', [])
            
            # Dividend Analysis
            dividend_data = {
                'dividend_yield': technical_data.get('dividend_yield'),
                'dividend_growth': technical_data.get('dividend_growth'),
                'payout_ratio': technical_data.get('payout_ratio')
            }
            
            return {
                'symbol': symbol,
                'technical_metrics': technical_metrics,
                'financial_metrics': financial_metrics,
                'announcements': announcements,
                'dividend_data': dividend_data
            }
            
        except Exception as e:
            logger.error(f"Error preparing analysis data: {str(e)}")
            return {}

    def _make_api_call(self, analysis_data: Dict) -> Optional[Dict]:
        """Make API call to DeepSeek or Grok based on available keys."""
        try:
            if not self._validate_api_keys():
                return None

            # Try DeepSeek first if available
            if self.deepseek_api_key and (self.deepseek_api_key.startswith('ds-') or self.deepseek_api_key.startswith('sk-')):
                logger.info("Using DeepSeek API with provided key")
                return self._make_deepseek_call(analysis_data)
            # Fall back to Grok if DeepSeek is not available
            elif self.grok_api_key and self.grok_api_key.startswith('xai-'):
                logger.info("Using Grok API with provided key")
                return self._make_grok_call(analysis_data)
            else:
                logger.warning("No valid API keys available for AI analysis")
                return None
                
        except Exception as e:
            logger.error(f"Error making API call: {str(e)}")
            return None

    def _make_deepseek_call(self, analysis_data: Dict) -> Optional[Dict]:
        """Make API call to DeepSeek."""
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            # Log request details for debugging (mask API key)
            masked_key = f"{self.deepseek_api_key[:8]}...{self.deepseek_api_key[-4:]}"
            logger.debug(f"Making DeepSeek API call to {self.deepseek_api_url}")
            logger.debug(f"Request headers: {{'Authorization': 'Bearer {masked_key}', 'Content-Type': 'application/json'}}")
            
            prompt = self._construct_prompt(analysis_data)
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional stock market analyst with expertise in technical and fundamental analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # Increase timeout and add retry logic
            max_retries = 3
            timeout = (30, 120)  # (connect timeout, read timeout)
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.deepseek_api_url,
                        headers=headers,
                        json=data,
                        timeout=timeout
                    )
                    
                    # Log response details for debugging
                    logger.debug(f"Response status code: {response.status_code}")
                    logger.debug(f"Response headers: {response.headers}")
                    
                    if response.status_code == 401:
                        error_msg = "Authentication failed. Please check your DeepSeek API key."
                        try:
                            error_data = response.json()
                            if 'error' in error_data and 'message' in error_data['error']:
                                error_msg = f"Authentication failed: {error_data['error']['message']}"
                        except:
                            pass
                        logger.error(error_msg)
                        logger.debug(f"Response body: {response.text}")
                        return None
                        
                    response.raise_for_status()
                    
                    result = response.json()
                    analysis_text = result['choices'][0]['message']['content']
                    
                    # Parse the analysis into structured format
                    return self._parse_analysis(analysis_text, analysis_data)
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Exponential backoff
                        logger.warning(f"Request timed out (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error("All retry attempts timed out. Falling back to Grok API.")
                        return self._make_grok_call(analysis_data)
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Error making DeepSeek API call: {str(e)}")
            return None

    def _make_grok_call(self, analysis_data: Dict) -> Optional[Dict]:
        """Make API call to Grok."""
        try:
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            # Log request details for debugging (mask API key)
            masked_key = f"{self.grok_api_key[:8]}...{self.grok_api_key[-4:]}"
            logger.debug(f"Making Grok API call to {self.grok_api_url}")
            logger.debug(f"Request headers: {{'Authorization': 'Bearer {masked_key}', 'Content-Type': 'application/json'}}")
            
            prompt = self._construct_prompt(analysis_data)
            
            data = {
                "model": "grok-1",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional stock market analyst with expertise in technical and fundamental analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            response = requests.post(self.grok_api_url, headers=headers, json=data, timeout=30)
            
            # Log response details for debugging
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            if response.status_code == 401:
                error_msg = "Authentication failed. Please check your Grok API key."
                try:
                    error_data = response.json()
                    if 'error' in error_data and 'message' in error_data['error']:
                        error_msg = f"Authentication failed: {error_data['error']['message']}"
                except:
                    pass
                logger.error(error_msg)
                logger.debug(f"Response body: {response.text}")
                return None
                
            response.raise_for_status()
            
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            
            # Parse the analysis into structured format
            return self._parse_analysis(analysis_text, analysis_data)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                error_msg = "Invalid Grok API key. Please check your API key in the .env file."
                try:
                    error_data = e.response.json()
                    if 'error' in error_data and 'message' in error_data['error']:
                        error_msg = f"Authentication failed: {error_data['error']['message']}"
                except:
                    pass
                logger.error(error_msg)
                logger.debug(f"Response body: {e.response.text}")
            else:
                logger.error(f"HTTP error making API call: {str(e)}")
            return None
        except requests.exceptions.Timeout:
            logger.error("Grok API call timed out. Please try again.")
            return None
        except Exception as e:
            logger.error(f"Error making Grok API call: {str(e)}")
            return None

    def _construct_prompt(self, analysis_data: Dict) -> str:
        """Construct prompt for AI analysis"""
        try:
            symbol = analysis_data['symbol']
            tech = analysis_data['technical_metrics']
            fin = analysis_data['financial_metrics']
            div = analysis_data['dividend_data']
            
            prompt = f"""Please provide a comprehensive investment analysis for {symbol} based on the following data:

Technical Analysis:
- Current Price: ${tech['current_price']}
- Price Change: {tech['price_change']}%
- Volume: {tech['volume']}
- RSI: {tech['rsi']}
- MACD: {tech['macd']}
- Moving Averages: SMA20=${tech['sma_20']}, SMA50=${tech['sma_50']}, SMA200=${tech['sma_200']}
- Bollinger Bands: Upper=${tech['bb_upper']}, Lower=${tech['bb_lower']}

Financial Analysis:
- EPS Growth: {fin['eps_growth']}%
- Revenue Growth: {fin['revenue_growth']}%
- Profit Margin: {fin['profit_margin']}%
- Debt-to-Equity: {fin['debt_to_equity']}
- Current Ratio: {fin['current_ratio']}
- ROE: {fin['roe']}%

Dividend Analysis:
- Dividend Yield: {div['dividend_yield']}%
- Dividend Growth: {div['dividend_growth']}%
- Payout Ratio: {div['payout_ratio']}%

Recent Announcements:
{chr(10).join(analysis_data['announcements'])}

Please provide a detailed analysis covering:
1. Company Overview
2. Technical Analysis
3. Financial Health
4. Growth Prospects
5. Risk Assessment
6. Dividend Analysis
7. Investment Thesis
8. Price Targets
9. Risk Management
10. Trading Strategy
11. Market Sentiment
12. Industry Position
13. Competitive Advantages
14. Management Quality
15. Corporate Governance
16. Environmental Factors
17. Social Impact
18. Regulatory Environment
19. Market Trends
20. Economic Outlook
21. Sector Analysis
22. Peer Comparison
23. Valuation Metrics
24. Investment Timeline
25. Exit Strategy

Format the response as a JSON object with the following structure:
{{
    "analysis": {{
        "overview": "",
        "technical_analysis": "",
        "financial_analysis": "",
        "growth_analysis": "",
        "risk_analysis": "",
        "dividend_analysis": "",
        "investment_thesis": "",
        "price_targets": {{
            "short_term": "",
            "medium_term": "",
            "long_term": ""
        }},
        "risk_management": "",
        "trading_strategy": "",
        "market_sentiment": "",
        "industry_position": "",
        "competitive_advantages": [],
        "management_quality": "",
        "corporate_governance": "",
        "environmental_factors": "",
        "social_impact": "",
        "regulatory_environment": "",
        "market_trends": "",
        "economic_outlook": "",
        "sector_analysis": "",
        "peer_comparison": "",
        "valuation_metrics": {{}},
        "investment_timeline": "",
        "exit_strategy": ""
    }},
    "recommendation": {{
        "action": "",
        "confidence": 0.0,
        "timeframe": "",
        "risk_level": "",
        "position_size": ""
    }},
    "key_metrics": {{
        "technical_score": 0.0,
        "financial_score": 0.0,
        "growth_score": 0.0,
        "risk_score": 0.0,
        "overall_score": 0.0
    }}
}}"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error constructing prompt: {str(e)}")
            return ""

    def _parse_analysis(self, analysis_text: str, original_data: Dict) -> Dict:
        """Parse AI analysis into structured format"""
        try:
            # Log the raw response for debugging
            logger.debug(f"Raw AI response: {analysis_text[:500]}...")  # Log first 500 chars
            
            # Clean the response text
            cleaned_text = self._clean_response_text(analysis_text)
            
            # Try to parse JSON response
            try:
                analysis = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {str(e)}")
                logger.debug("Attempting to extract JSON from response...")
                
                # Try to find JSON-like structure in the response
                json_match = re.search(r'({[\s\S]*})', cleaned_text)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(1))
                        logger.info("Successfully extracted JSON from response")
                    except json.JSONDecodeError:
                        logger.error("Failed to parse extracted JSON")
                        return self._create_fallback_analysis(original_data)
                else:
                    logger.error("No JSON structure found in response")
                    return self._create_fallback_analysis(original_data)
            
            # Validate required fields
            required_fields = ['analysis', 'recommendation', 'key_metrics']
            missing_fields = [field for field in required_fields if field not in analysis]
            
            if missing_fields:
                logger.error(f"Missing required fields in AI response: {missing_fields}")
                return self._create_fallback_analysis(original_data)
            
            # Add original data
            analysis['original_data'] = original_data
            
            # Add timestamp
            analysis['timestamp'] = datetime.now().isoformat()
            
            # Round all float values
            self._round_dict_values(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing analysis: {str(e)}")
            return self._create_fallback_analysis(original_data)

    def _clean_response_text(self, text: str) -> str:
        """Clean the response text to handle various formats."""
        try:
            # Remove markdown code block markers
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            # Remove any leading/trailing whitespace
            text = text.strip()
            
            # Handle potential markdown formatting
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markers
            text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic markers
            text = re.sub(r'`(.*?)`', r'\1', text)        # Remove inline code markers
            
            # Handle potential HTML entities
            text = text.replace('&quot;', '"')
            text = text.replace('&amp;', '&')
            text = text.replace('&lt;', '<')
            text = text.replace('&gt;', '>')
            
            # Handle potential escaped characters
            text = text.replace('\\n', ' ')
            text = text.replace('\\r', ' ')
            text = text.replace('\\t', ' ')
            
            # Remove multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning response text: {str(e)}")
            return text  # Return original text if cleaning fails

    def _create_fallback_analysis(self, original_data: Dict) -> Dict:
        """Create a fallback analysis when AI response parsing fails"""
        try:
            logger.info("Creating fallback analysis structure")
            
            # Extract basic metrics from original data
            tech_data = original_data.get('technical_metrics', {})
            fin_data = original_data.get('financial_metrics', {})
            
            # Calculate basic scores
            tech_score = self._calculate_technical_score(tech_data)
            fin_score = self._calculate_financial_score(fin_data)
            
            return {
                'analysis': {
                    'overview': 'AI analysis was not available. Using technical and financial metrics only.',
                    'technical_analysis': f"Technical Score: {tech_score:.2f}",
                    'financial_analysis': f"Financial Score: {fin_score:.2f}",
                    'investment_thesis': 'Based on technical and financial metrics only.'
                },
                'recommendation': {
                    'action': 'NEUTRAL',
                    'confidence': 0.5,
                    'timeframe': 'MEDIUM_TERM',
                    'risk_level': 'MODERATE',
                    'position_size': 'STANDARD'
                },
                'key_metrics': {
                    'technical_score': tech_score,
                    'financial_score': fin_score,
                    'growth_score': 0.0,
                    'risk_score': 0.0,
                    'overall_score': (tech_score * 0.6 + fin_score * 0.4)
                },
                'original_data': original_data,
                'timestamp': datetime.now().isoformat(),
                'parse_error': True
            }
        except Exception as e:
            logger.error(f"Error creating fallback analysis: {str(e)}")
            return {}

    def _calculate_technical_score(self, tech_data: Dict) -> float:
        """Calculate technical score from technical metrics"""
        try:
            scores = []
            
            # RSI scoring
            rsi = tech_data.get('rsi')
            if rsi is not None:
                if rsi > 70:
                    scores.append(30)  # Overbought
                elif rsi < 30:
                    scores.append(30)  # Oversold
                else:
                    scores.append(70)  # Neutral
            
            # MACD scoring
            macd = tech_data.get('macd')
            macd_signal = tech_data.get('macd_signal')
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    scores.append(70)  # Bullish
                else:
                    scores.append(30)  # Bearish
            
            # Moving averages scoring
            price = tech_data.get('current_price')
            sma20 = tech_data.get('sma_20')
            sma50 = tech_data.get('sma_50')
            
            if all(x is not None for x in [price, sma20, sma50]):
                if price > sma20 and sma20 > sma50:
                    scores.append(80)  # Strong uptrend
                elif price < sma20 and sma20 < sma50:
                    scores.append(20)  # Strong downtrend
                else:
                    scores.append(50)  # Mixed signals
            
            return sum(scores) / len(scores) if scores else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 50.0

    def _calculate_financial_score(self, fin_data: Dict) -> float:
        """Calculate financial score from financial metrics"""
        try:
            scores = []
            
            # EPS Growth scoring
            eps_growth = fin_data.get('eps_growth')
            if eps_growth is not None:
                if eps_growth > 15:
                    scores.append(80)
                elif eps_growth > 10:
                    scores.append(70)
                elif eps_growth > 5:
                    scores.append(60)
                elif eps_growth > 0:
                    scores.append(50)
                else:
                    scores.append(30)
            
            # Revenue Growth scoring
            revenue_growth = fin_data.get('revenue_growth')
            if revenue_growth is not None:
                if revenue_growth > 20:
                    scores.append(80)
                elif revenue_growth > 15:
                    scores.append(70)
                elif revenue_growth > 10:
                    scores.append(60)
                elif revenue_growth > 5:
                    scores.append(50)
                else:
                    scores.append(30)
            
            # Profit Margin scoring
            profit_margin = fin_data.get('profit_margin')
            if profit_margin is not None:
                if profit_margin > 20:
                    scores.append(80)
                elif profit_margin > 15:
                    scores.append(70)
                elif profit_margin > 10:
                    scores.append(60)
                elif profit_margin > 5:
                    scores.append(50)
                else:
                    scores.append(30)
            
            return sum(scores) / len(scores) if scores else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating financial score: {str(e)}")
            return 50.0

    def _round_dict_values(self, data: Dict):
        """Recursively round all float values in dictionary"""
        for key, value in data.items():
            if isinstance(value, dict):
                self._round_dict_values(value)
            elif isinstance(value, float):
                data[key] = self._round_float(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._round_dict_values(item)
                    elif isinstance(item, float):
                        value[i] = self._round_float(item)

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
