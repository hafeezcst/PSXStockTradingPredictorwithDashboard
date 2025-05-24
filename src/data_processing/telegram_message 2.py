"""
Telegram messaging module for PSX application.

This module provides functions to send messages and images to a Telegram bot.
"""

import os
import requests
import logging
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import json
from datetime import datetime

# Set up logging
logging.basicConfig(filename='telegram_message.log', level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_DELAY = 1  # Minimum delay between messages in seconds
MAX_RETRIES = 3  # Maximum number of retries for rate-limited requests

def send_telegram_message(message: str) -> bool:
    """
    Send a text message to a Telegram bot with rate limiting.
    
    Args:
        message (str): Message text to send
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get bot token and chat ID from environment variables
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.warning("Telegram bot token or chat ID not set in environment variables")
            return False
        
        # Construct the API URL
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Prepare the payload
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        # Send the request with retries for rate limiting
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(url, data=payload, timeout=10)
                
                # Check if the request was successful
                if response.status_code == 200:
                    logger.info("Message sent successfully")
                    return True
                elif response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.json().get('parameters', {}).get('retry_after', RATE_LIMIT_DELAY))
                    logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds before retry.")
                    time.sleep(retry_after)
                    continue
                else:
                    logger.error(f"Failed to send message: {response.text}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_DELAY)
                    continue
                return False
            
            # Add delay between messages to prevent rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
    except Exception as e:
        logger.error(f"Error sending Telegram message: {str(e)}")
        return False

def send_telegram_message_with_image(image_path: str, caption: str = "") -> bool:
    """
    Send an image with optional caption to a Telegram bot.
    
    Args:
        image_path (str): Path to the image file
        caption (str): Optional caption text
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get bot token and chat ID from environment variables
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.warning("Telegram bot token or chat ID not set in environment variables")
            return False
        
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            send_telegram_message(f"Error: Image file not found: {image_path}")
            return False
        
        # Construct the API URL
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        
        # Prepare the payload
        payload = {
            'chat_id': chat_id,
            'caption': caption,
            'parse_mode': 'HTML'
        }
        
        # Prepare the files
        files = {
            'photo': open(image_path, 'rb')
        }
        
        # Send the request
        response = requests.post(url, data=payload, files=files, timeout=30)
        
        # Close the file
        files['photo'].close()
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info(f"Image sent successfully: {image_path}")
            return True
        else:
            logger.error(f"Failed to send image: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending Telegram image: {str(e)}")
        return False

def send_telegram_message_with_image_and_message(image_path: str, message_text: str) -> bool:
    """
    Send both an image and a separate text message to Telegram.
    
    Args:
        image_path (str): Path to the image file
        message_text (str): Text message to send
        
    Returns:
        bool: True if both operations successful, False otherwise
    """
    image_sent = send_telegram_message_with_image(image_path, "")
    message_sent = send_telegram_message(message_text)
    
    return image_sent and message_sent

class TelegramMessageFormatter:
    @staticmethod
    def format_buy_signal(symbol: str, analysis: dict) -> str:
        """Format a buy signal message for Telegram with enhanced details"""
        try:
            # Get AI analysis data
            ai_analysis = analysis.get('ai_analysis', {})
            
            # Format the message with emojis and sections
            message = f"🚨 <b>BUY SIGNAL ALERT</b> 🚨\n\n"
            
            # Symbol and Date
            message += f"📊 <b>Symbol:</b> {symbol}\n"
            message += f"📅 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Signal Details
            message += f"🎯 <b>Signal Type:</b> {analysis.get('signal_type', 'N/A')}\n"
            message += f"💪 <b>Signal Strength:</b> {analysis.get('signal_strength', 0):.2f}\n"
            message += f"🎯 <b>Confidence Score:</b> {analysis.get('confidence_score', 0):.2f}\n\n"
            
            # Price Information
            message += f"💰 <b>Current Price:</b> {analysis.get('close', 'N/A')}\n"
            message += f"📈 <b>Price Change:</b> {analysis.get('change_percent', 0):.2f}%\n\n"
            
            # Technical Indicators
            message += "📊 <b>Technical Indicators:</b>\n"
            message += f"• RSI: {analysis.get('rsi', 'N/A')}\n"
            message += f"• MACD: {analysis.get('macd', 'N/A')}\n"
            message += f"• SMA20: {analysis.get('sma_20', 'N/A')}\n"
            message += f"• SMA50: {analysis.get('sma_50', 'N/A')}\n"
            message += f"• SMA200: {analysis.get('sma_200', 'N/A')}\n\n"
            
            # Trading Levels
            message += "🎯 <b>Trading Levels:</b>\n"
            message += f"• Entry: {analysis.get('close', 'N/A')}\n"
            message += f"• Stop Loss: {analysis.get('stop_loss', 'N/A')}\n"
            message += f"• Take Profit: {analysis.get('take_profit', 'N/A')}\n"
            message += f"• Risk/Reward: {analysis.get('risk_reward_ratio', 'N/A'):.2f}\n\n"
            
            # Support and Resistance
            message += "📈 <b>Support & Resistance:</b>\n"
            message += f"• Support: {analysis.get('support_level', 'N/A')}\n"
            message += f"• Resistance: {analysis.get('resistance_level', 'N/A')}\n\n"
            
            # AI Analysis
            if ai_analysis:
                message += "🤖 <b>AI Analysis:</b>\n"
                
                # Market Overview
                if ai_analysis.get('market_overview'):
                    message += f"📊 <b>Market Overview:</b>\n{ai_analysis['market_overview']}\n\n"
                
                # Technical Analysis
                if ai_analysis.get('technical_analysis'):
                    message += f"📈 <b>Technical Analysis:</b>\n{ai_analysis['technical_analysis']}\n\n"
                
                # Risk Assessment
                if ai_analysis.get('risk_assessment'):
                    message += f"⚠️ <b>Risk Assessment:</b>\n{ai_analysis['risk_assessment']}\n\n"
                
                # Trading Recommendation
                if ai_analysis.get('trading_recommendation'):
                    message += f"🎯 <b>Trading Recommendation:</b>\n{ai_analysis['trading_recommendation']}\n\n"
                
                # Price Targets
                if ai_analysis.get('price_targets'):
                    message += "💰 <b>Price Targets:</b>\n"
                    for timeframe, target in ai_analysis['price_targets'].items():
                        message += f"• {timeframe}: {target}\n"
                    message += "\n"
                
                # Entry/Exit Points
                if ai_analysis.get('entry_points'):
                    message += "📥 <b>Entry Points:</b>\n"
                    for point in ai_analysis['entry_points']:
                        message += f"• {point}\n"
                    message += "\n"
                
                if ai_analysis.get('exit_points'):
                    message += "📤 <b>Exit Points:</b>\n"
                    for point in ai_analysis['exit_points']:
                        message += f"• {point}\n"
                    message += "\n"
            
            # Analysis Summary
            if analysis.get('analysis_summary'):
                message += "📝 <b>Analysis Summary:</b>\n"
                for summary in analysis['analysis_summary'][:5]:  # Show top 5 points
                    message += f"• {summary}\n"
                message += "\n"
            
            # Risk Warning
            message += "⚠️ <b>Risk Warning:</b>\n"
            message += "This is not financial advice. Always do your own research and trade responsibly.\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting buy signal message: {e}")
            return f"Error formatting message for {symbol}: {str(e)}"
    
    @staticmethod
    def format_signal_transition(symbol: str, current_analysis: dict, previous_analysis: dict) -> str:
        """Format a signal transition message for Telegram"""
        try:
            message = f"🔄 <b>Signal Change Alert</b>\n\n"
            
            # Symbol and Date
            message += f"📊 <b>Symbol:</b> {symbol}\n"
            message += f"📅 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Signal Change
            message += f"📈 <b>Previous Signal:</b> {previous_analysis.get('signal_type', 'N/A')}\n"
            message += f"📉 <b>New Signal:</b> {current_analysis.get('signal_type', 'N/A')}\n\n"
            
            # Price Information
            message += f"💰 <b>Current Price:</b> {current_analysis.get('close', 'N/A')}\n"
            message += f"📈 <b>Price Change:</b> {current_analysis.get('change_percent', 0):.2f}%\n\n"
            
            # Confidence Score
            message += f"🎯 <b>Confidence Score:</b> {current_analysis.get('confidence_score', 0):.2f}\n\n"
            
            # Analysis Summary
            if current_analysis.get('analysis_summary'):
                message += "📝 <b>Analysis Summary:</b>\n"
                for summary in current_analysis['analysis_summary'][:3]:  # Show top 3 points
                    message += f"• {summary}\n"
                message += "\n"
            
            # Risk Warning
            message += "⚠️ <b>Risk Warning:</b>\n"
            message += "This is not financial advice. Always do your own research and trade responsibly.\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting signal transition message: {e}")
            return f"Error formatting transition message for {symbol}: {str(e)}"
    
    @staticmethod
    def format_profit_taking_alert(symbol: str, current_analysis: dict, previous_analysis: dict) -> str:
        """Format a profit taking alert message for Telegram"""
        try:
            message = f"💰 <b>Profit Taking Alert</b>\n\n"
            
            # Symbol and Date
            message += f"📊 <b>Symbol:</b> {symbol}\n"
            message += f"📅 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Current Signal
            message += f"📈 <b>Current Signal:</b> {current_analysis.get('signal_type', 'N/A')}\n"
            
            # Price Information
            current_price = current_analysis.get('close')
            previous_price = previous_analysis.get('close')
            if current_price and previous_price:
                price_change = ((current_price - previous_price) / previous_price) * 100
                message += f"💰 <b>Price Change:</b> +{price_change:.2f}%\n"
                message += f"📊 <b>Current Price:</b> {current_price}\n"
                message += f"📈 <b>Previous Price:</b> {previous_price}\n\n"
            
            # Technical Indicators
            message += "📊 <b>Technical Indicators:</b>\n"
            if current_analysis.get('rsi'):
                message += f"• RSI: {current_analysis['rsi']:.2f}\n"
            if current_analysis.get('macd'):
                message += f"• MACD: {current_analysis['macd']:.2f}\n"
            
            # Support and Resistance
            if current_analysis.get('support_level') and current_analysis.get('resistance_level'):
                message += f"\n📈 <b>Support:</b> {current_analysis['support_level']:.2f}\n"
                message += f"📉 <b>Resistance:</b> {current_analysis['resistance_level']:.2f}\n"
            
            # Risk Warning
            message += "\n⚠️ <b>Risk Warning:</b>\n"
            message += "This is not financial advice. Always do your own research and trade responsibly.\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting profit taking alert message: {e}")
            return f"Error formatting profit taking alert for {symbol}: {str(e)}"
    
    @staticmethod
    def format_portfolio_analysis(portfolio_data: dict) -> str:
        """Format a portfolio analysis message for Telegram"""
        try:
            message = f"📊 <b>Portfolio Analysis Report</b>\n\n"
            
            # Portfolio Summary
            message += f"💰 <b>Portfolio Summary:</b>\n"
            message += f"• Total Value: {portfolio_data.get('total_value', 'N/A')}\n"
            message += f"• Daily Change: {portfolio_data.get('daily_change', 'N/A')}%\n"
            message += f"• Total Return: {portfolio_data.get('total_return', 'N/A')}%\n\n"
            
            # Top Performers
            if portfolio_data.get('top_performers'):
                message += "📈 <b>Top Performers:</b>\n"
                for stock in portfolio_data['top_performers'][:3]:
                    message += f"• {stock['symbol']}: +{stock['return']}%\n"
                message += "\n"
            
            # Underperformers
            if portfolio_data.get('underperformers'):
                message += "📉 <b>Underperformers:</b>\n"
                for stock in portfolio_data['underperformers'][:3]:
                    message += f"• {stock['symbol']}: {stock['return']}%\n"
                message += "\n"
            
            # Risk Analysis
            if portfolio_data.get('risk_metrics'):
                message += "⚠️ <b>Risk Analysis:</b>\n"
                message += f"• Portfolio Beta: {portfolio_data['risk_metrics'].get('beta', 'N/A')}\n"
                message += f"• Volatility: {portfolio_data['risk_metrics'].get('volatility', 'N/A')}%\n"
                message += f"• Sharpe Ratio: {portfolio_data['risk_metrics'].get('sharpe_ratio', 'N/A')}\n\n"
            
            # Recommendations
            if portfolio_data.get('recommendations'):
                message += "🎯 <b>Recommendations:</b>\n"
                for rec in portfolio_data['recommendations']:
                    message += f"• {rec}\n"
                message += "\n"
            
            # Risk Warning
            message += "⚠️ <b>Risk Warning:</b>\n"
            message += "This is not financial advice. Always do your own research and trade responsibly.\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting portfolio analysis message: {e}")
            return f"Error formatting portfolio analysis: {str(e)}"
    
    @staticmethod
    def format_dividend_announcement(dividend_data: dict) -> str:
        """Format a dividend announcement message for Telegram"""
        try:
            message = f"💰 <b>Dividend Announcement</b>\n\n"
            
            # Company Info
            message += f"🏢 <b>Company:</b> {dividend_data.get('company', 'N/A')}\n"
            message += f"📊 <b>Symbol:</b> {dividend_data.get('symbol', 'N/A')}\n\n"
            
            # Dividend Details
            message += "📅 <b>Dividend Details:</b>\n"
            message += f"• Type: {dividend_data.get('type', 'N/A')}\n"
            message += f"• Amount: {dividend_data.get('amount', 'N/A')}\n"
            message += f"• Record Date: {dividend_data.get('record_date', 'N/A')}\n"
            message += f"• Payment Date: {dividend_data.get('payment_date', 'N/A')}\n\n"
            
            # Additional Info
            if dividend_data.get('additional_info'):
                message += "ℹ️ <b>Additional Information:</b>\n"
                for info in dividend_data['additional_info']:
                    message += f"• {info}\n"
                message += "\n"
            
            # Risk Warning
            message += "⚠️ <b>Risk Warning:</b>\n"
            message += "This is not financial advice. Always do your own research and trade responsibly.\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting dividend announcement message: {e}")
            return f"Error formatting dividend announcement: {str(e)}"
    
    @staticmethod
    def format_market_status(status_data: dict) -> str:
        """Format a market status update message for Telegram"""
        try:
            message = f"🏛️ <b>Market Status Update</b>\n\n"
            
            # Market Status
            message += f"📊 <b>Status:</b> {status_data.get('status', 'N/A')}\n"
            message += f"⏰ <b>Time:</b> {status_data.get('time', 'N/A')}\n\n"
            
            # Session Info
            if status_data.get('session_info'):
                message += "📅 <b>Session Information:</b>\n"
                message += f"• Current Session: {status_data['session_info'].get('current_session', 'N/A')}\n"
                message += f"• Next Session: {status_data['session_info'].get('next_session', 'N/A')}\n"
                message += f"• Remaining Time: {status_data['session_info'].get('remaining_time', 'N/A')}\n\n"
            
            # Market Summary
            if status_data.get('market_summary'):
                message += "📈 <b>Market Summary:</b>\n"
                message += f"• KSE-100: {status_data['market_summary'].get('kse100', 'N/A')}\n"
                message += f"• Change: {status_data['market_summary'].get('change', 'N/A')}%\n"
                message += f"• Volume: {status_data['market_summary'].get('volume', 'N/A')}\n\n"
            
            # Risk Warning
            message += "⚠️ <b>Risk Warning:</b>\n"
            message += "This is not financial advice. Always do your own research and trade responsibly.\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting market status message: {e}")
            return f"Error formatting market status: {str(e)}"
    
    @staticmethod
    def format_error_notification(error_data: dict) -> str:
        """Format an error notification message for Telegram"""
        try:
            message = f"⚠️ <b>Error Notification</b>\n\n"
            
            # Error Details
            message += f"🔍 <b>Error Type:</b> {error_data.get('type', 'Unknown Error')}\n"
            message += f"⏰ <b>Time:</b> {error_data.get('time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n\n"
            
            # Error Description
            if error_data.get('description'):
                message += "📝 <b>Description:</b>\n"
                message += f"{error_data['description']}\n\n"
            
            # Affected Components
            if error_data.get('affected_components'):
                message += "🔧 <b>Affected Components:</b>\n"
                for component in error_data['affected_components']:
                    message += f"• {component}\n"
                message += "\n"
            
            # Resolution Steps
            if error_data.get('resolution_steps'):
                message += "🛠️ <b>Resolution Steps:</b>\n"
                for step in error_data['resolution_steps']:
                    message += f"• {step}\n"
                message += "\n"
            
            # Contact Information
            if error_data.get('contact_info'):
                message += "📞 <b>Contact Information:</b>\n"
                message += f"{error_data['contact_info']}\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting error notification message: {e}")
            return f"Error formatting error notification: {str(e)}"
