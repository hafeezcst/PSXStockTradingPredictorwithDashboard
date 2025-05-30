from __future__ import annotations
import logging
import os
import requests
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class NotificationManager:
    """Handles notifications for stock analysis signals and alerts.
    
    This class manages sending notifications through various channels, such as Telegram,
    to alert users about signal changes or important stock events.
    
    Attributes:
        None
    """
    
    def __init__(self) -> None:
        """Initialize the NotificationManager."""
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

    def send_telegram_notification(self, message: str) -> bool:
        """Send notification to Telegram channel.
        
        Args:
            message (str): Message to send via Telegram.
            
        Returns:
            bool: True if notification was sent successfully, False otherwise.
        """
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
        """Check for signal transitions and send notifications.
        
        Args:
            symbol (str): Stock symbol.
            current_analysis (Dict): Current analysis data with signals.
            previous_analysis (Dict, optional): Previous analysis data for comparison.
        """
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
