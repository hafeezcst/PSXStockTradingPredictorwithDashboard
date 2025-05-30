import os
import requests
from dotenv import load_dotenv
import urllib3
import ssl
import json
import pandas as pd
from datetime import datetime
import sqlite3
from typing import Dict, Any, Optional, List, Tuple

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

class GrokChat:
    def __init__(self):
        self.api_key = os.getenv('xai_api_key')
        if not self.api_key:
            raise ValueError("Grok API key not found in environment variables. Please check your .env file.")
        
        if self.api_key == "your_actual_grok_api_key_here":
            raise ValueError("Please replace the placeholder API key in your .env file with your actual xAI API key")
        
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize database connection
        self.db_path = os.path.join(os.path.dirname(__file__), "data", "databases", "production", "PSX_investing_Stocks_KMI30.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a message to Grok API and get the response
        """
        try:
            # Prepare system message based on context
            system_message = "You are a helpful assistant."
            if context and context.get('analysis_type') == 'financial':
                system_message = """You are a professional financial analyst assistant. 
                You provide detailed analysis of financial reports and market data.
                Focus on key metrics, trends, and actionable insights."""
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "model": "grok-3-latest",
                "stream": False,
                "temperature": 0.7
            }
            
            # Add context if provided
            if context:
                payload["messages"].insert(1, {
                    "role": "system",
                    "content": f"Context: {json.dumps(context)}"
                })
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                verify=False
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if 'choices' not in response_data or not response_data['choices']:
                return "Sorry, I couldn't generate a response. Please try again."
            
            return response_data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            return f"Error: Unable to connect to the API. Please check your internet connection."
        except json.JSONDecodeError:
            return "Error: Invalid response from the API. Please try again."
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"

    def analyze_financial_report(self, company_symbol: str, report_date: Optional[str] = None) -> str:
        """
        Analyze financial reports for a specific company
        """
        try:
            # Get company data
            company_data = self._get_company_data(company_symbol, report_date)
            if not company_data:
                return f"No data found for company {company_symbol}"
            
            # Get latest signals
            signals = self._get_latest_signals(company_symbol)
            
            # Prepare analysis context
            context = {
                "analysis_type": "financial",
                "company_data": company_data,
                "signals": signals,
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(company_data, signals)
            
            # Get AI analysis
            analysis = self.chat(prompt, context)
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing financial report: {str(e)}"

    def _get_company_data(self, company_symbol: str, report_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get company data from database
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT * FROM company_data 
                WHERE Symbol = ? 
                """
                params = [company_symbol]
                
                if report_date:
                    query += " AND Report_Date = ?"
                    params.append(report_date)
                
                query += " ORDER BY Report_Date DESC LIMIT 1"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return {}
                
                return df.iloc[0].to_dict()
                
        except Exception as e:
            print(f"Error getting company data: {str(e)}")
            return {}

    def _get_latest_signals(self, company_symbol: str) -> Dict[str, Any]:
        """
        Get latest trading signals for a company
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT * FROM buy_stocks 
                WHERE Symbol = ? 
                ORDER BY Signal_Date DESC 
                LIMIT 1
                """
                
                df = pd.read_sql_query(query, conn, params=[company_symbol])
                
                if df.empty:
                    return {}
                
                return df.iloc[0].to_dict()
                
        except Exception as e:
            print(f"Error getting latest signals: {str(e)}")
            return {}

    def _generate_analysis_prompt(self, company_data: Dict[str, Any], signals: Dict[str, Any]) -> str:
        """
        Generate a prompt for financial analysis
        """
        prompt = f"""Please analyze the following financial data for {company_data.get('Symbol', 'Unknown Company')}:

Company Information:
- Symbol: {company_data.get('Symbol', 'N/A')}
- Report Date: {company_data.get('Report_Date', 'N/A')}
- Revenue: {company_data.get('Revenue', 'N/A')}
- Net Income: {company_data.get('Net_Income', 'N/A')}
- EPS: {company_data.get('EPS', 'N/A')}

Latest Trading Signals:
- Signal Date: {signals.get('Signal_Date', 'N/A')}
- Signal Type: {signals.get('Signal_Type', 'N/A')}
- Signal Price: {signals.get('Signal_Price', 'N/A')}
- Confidence: {signals.get('Confidence', 'N/A')}

Please provide a comprehensive analysis including:
1. Financial Performance Analysis
2. Key Metrics and Trends
3. Market Position and Competitive Analysis
4. Risk Assessment
5. Investment Recommendation

Focus on actionable insights and specific recommendations."""
        
        return prompt

def main():
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        chat = GrokChat()
        
        print("🤖 Grok Financial Analysis Chat")
        print("Commands:")
        print("- analyze <company_symbol> [report_date] - Analyze financial reports")
        print("- chat <message> - Regular chat")
        print("- quit - Exit")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
                
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower().startswith('analyze '):
                parts = user_input.split()
                if len(parts) >= 2:
                    company_symbol = parts[1]
                    report_date = parts[2] if len(parts) > 2 else None
                    response = chat.analyze_financial_report(company_symbol, report_date)
                else:
                    response = "Please provide a company symbol. Usage: analyze <company_symbol> [report_date]"
            else:
                response = chat.chat(user_input)
            
            print(f"\nGrok: {response}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 