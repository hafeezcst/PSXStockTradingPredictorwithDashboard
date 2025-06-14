from typing import Dict, List, Optional
import logging
import os
import requests
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
XAI_API_KEY = os.getenv('xai_api_key')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Endpoint for DeepSeek
XAI_API_URL = "https://api.x.ai/v1/chat/completions"  # Alternative endpoint for XAI

def analyze_with_ai(symbol: str, technical_data: Dict, financial_data: Dict, announcements: List[Dict]) -> Dict:
    """Analyze stock data using AI model"""
    try:
        if not DEEPSEEK_API_KEY and not XAI_API_KEY:
            logger.error("Neither DEEPSEEK_API_KEY nor XAI_API_KEY found in environment variables")
            return {
                'ai_analysis': 'AI analysis unavailable: API key not configured',
                'ai_score': 0.0,
                'ai_recommendation': 'N/A',
                'ai_confidence': 0.0,
                'ai_risk_assessment': 'N/A'
            }
            
        # Prepare data for AI analysis
        prompt = prepare_ai_prompt(symbol, technical_data, financial_data, announcements)
        
        # Get AI response
        response = get_ai_response(prompt)
        
        if not response:
            return {
                'ai_analysis': 'Failed to get AI analysis',
                'ai_score': 0.0,
                'ai_recommendation': 'N/A',
                'ai_confidence': 0.0,
                'ai_risk_assessment': 'N/A'
            }
            
        # Parse AI response
        parsed_response = parse_ai_response(response)
        
        return parsed_response
        
    except Exception as e:
        logger.error(f"Error in AI analysis for {symbol}: {e}")
        return {
            'ai_analysis': f'Error in AI analysis: {str(e)}',
            'ai_score': 0.0,
            'ai_recommendation': 'N/A',
            'ai_confidence': 0.0,
            'ai_risk_assessment': 'N/A'
        }

def prepare_ai_prompt(symbol: str, technical_data: Dict, financial_data: Dict, announcements: List[Dict]) -> str:
    """Prepare detailed prompt for AI analysis"""
    try:
        prompt = f"""
You are a highly skilled financial analyst with expertise in stock market analysis. 
Your task is to provide a comprehensive analysis of {symbol} listed on the Pakistan Stock Exchange (PSX).
Use the provided technical, financial, and announcement data to evaluate the stock's current position, 
future potential, and associated risks. Provide a detailed analysis, a numerical score between -1 (strong sell) 
and 1 (strong buy), a clear recommendation (Buy, Hold, Sell), confidence level in your analysis (0-1), 
and a risk assessment (Low, Medium, High).

**Technical Data:**
{format_technical_data(technical_data)}

**Financial Data:**
{format_financial_data(financial_data)}

**Recent Announcements:**
{format_announcements(announcements)}

**Analysis Requirements:**
1. Evaluate the technical indicators and identify key trends, support/resistance levels, and momentum.
2. Assess the financial health based on profitability, liquidity, solvency, efficiency, growth, and valuation metrics.
3. Consider the impact of recent announcements on future stock performance.
4. Identify potential catalysts or risks that could significantly affect the stock price.
5. Provide an overall score between -1 (strong sell) and 1 (strong buy) based on your analysis.
6. Give a clear recommendation: Buy, Hold, or Sell.
7. Indicate your confidence level in this analysis (0 to 1).
8. Assess the risk level associated with this stock (Low, Medium, High).

**Response Format:**
Please structure your response in JSON format as follows:
{{
    "ai_analysis": "Your detailed analysis text here (at least 200 words covering all aspects)",
    "ai_score": 0.7,
    "ai_recommendation": "Buy",
    "ai_confidence": 0.85,
    "ai_risk_assessment": "Medium"
}}
"""
        return prompt
        
    except Exception as e:
        logger.error(f"Error preparing AI prompt: {e}")
        return ""

def format_technical_data(data: Dict) -> str:
    """Format technical data for AI prompt"""
    try:
        if not data:
            return "No technical data available"
            
        formatted = []
        for key, value in data.items():
            if isinstance(value, dict):
                formatted.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                formatted.append(f"{key}: {value}")
                
        return "\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error formatting technical data: {e}")
        return "Error formatting technical data"

def format_financial_data(data: Dict) -> str:
    """Format financial data for AI prompt"""
    try:
        if not data:
            return "No financial data available"
            
        formatted = []
        for key, value in data.items():
            if isinstance(value, dict):
                formatted.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                formatted.append(f"{key}: {value}")
                
        return "\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error formatting financial data: {e}")
        return "Error formatting financial data"

def format_announcements(announcements: List[Dict]) -> str:
    """Format announcements for AI prompt"""
    try:
        if not announcements:
            return "No recent announcements"
            
        formatted = []
        for idx, ann in enumerate(announcements[:3], 1):  # Limit to 3 most recent
            formatted.append(f"Announcement {idx}:")
            for key, value in ann.items():
                formatted.append(f"  {key}: {value}")
                
        return "\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error formatting announcements: {e}")
        return "Error formatting announcements"

def get_ai_response(prompt: str, max_retries: int = 3) -> Optional[str]:
    """Get response from AI API with retries"""
    try:
        if DEEPSEEK_API_KEY:
            api_key = DEEPSEEK_API_KEY
            key_type = "DEEPSEEK_API_KEY"
            api_url = DEEPSEEK_API_URL
            model = "deepseek-chat"
        else:
            api_key = XAI_API_KEY
            key_type = "XAI_API_KEY"
            api_url = XAI_API_URL
            model = "xai-model"  # Use different model for XAI if needed
            
        logger.info(f"Using {key_type} for AI API request with endpoint {api_url}")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_p": 0.9,
            "stream": False,
            "response_format": {"type": "json_object"}
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
                
        logger.error(f"Failed to get AI response after {max_retries} attempts")
        return None
        
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return None

def parse_ai_response(response: str) -> Dict:
    """Parse AI response into structured format"""
    try:
        # Clean response if needed
        response = response.strip()
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
            
        # Parse JSON response
        parsed = json.loads(response)
        
        # Validate required fields with defaults
        required_fields = {
            'ai_analysis': 'No detailed analysis provided',
            'ai_score': 0.0,
            'ai_recommendation': 'N/A',
            'ai_confidence': 0.0,
            'ai_risk_assessment': 'N/A'
        }
        
        result = {}
        for field, default in required_fields.items():
            result[field] = parsed.get(field, default)
            
        # Validate score range
        result['ai_score'] = max(min(result['ai_score'], 1.0), -1.0)
        result['ai_confidence'] = max(min(result['ai_confidence'], 1.0), 0.0)
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        logger.error(f"Raw response: {response}")
        return {
            'ai_analysis': 'Failed to parse AI response',
            'ai_score': 0.0,
            'ai_recommendation': 'N/A',
            'ai_confidence': 0.0,
            'ai_risk_assessment': 'N/A'
        }
    except Exception as e:
        logger.error(f"Error parsing AI response: {e}")
        return {
            'ai_analysis': f'Error parsing AI response: {str(e)}',
            'ai_score': 0.0,
            'ai_recommendation': 'N/A',
            'ai_confidence': 0.0,
            'ai_risk_assessment': 'N/A'
        }
