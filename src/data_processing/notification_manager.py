from __future__ import annotations
import logging
import os
import requests
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class NotificationManager:
    """Handles notifications for stock analysis signals and alerts.
    
    This class manages sending notifications through various channels, such as Telegram,
    to alert users about signal changes or important stock events.
    
    Attributes:
        None
    """
    
    def __init__(self):
        load_dotenv()
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self._round_float = lambda x: round(float(x), 2) if x is not None else None
        self._notification_cache = {}
        self._cooldown = 300  # 5 minutes cooldown between notifications for same symbol

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

    def send_notification(self, symbol: str, analysis: Dict, notification_type: str = 'analysis') -> bool:
        """Send notification about stock analysis"""
        try:
            # Check cache first
            current_time = datetime.now().timestamp()
            cache_key = f"{symbol}_{notification_type}"
            
            if cache_key in self._notification_cache:
                last_notification = self._notification_cache[cache_key]
                if current_time - last_notification < self._cooldown:
                    logger.info(f"Skipping notification for {symbol} due to cooldown")
                    return False

            # Prepare notification message
            message = self._prepare_notification_message(symbol, analysis, notification_type)
            
            # Send to Telegram
            if self.telegram_token and self.telegram_chat_id:
                success = self._send_telegram_message(message)
                if success:
                    self._notification_cache[cache_key] = current_time
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending notification for {symbol}: {str(e)}")
            return False

    def _prepare_notification_message(self, symbol: str, analysis: Dict, notification_type: str) -> str:
        """Prepare notification message based on analysis type"""
        try:
            if notification_type == 'analysis':
                return self._prepare_analysis_notification(symbol, analysis)
            elif notification_type == 'signal':
                return self._prepare_signal_notification(symbol, analysis)
            elif notification_type == 'alert':
                return self._prepare_alert_notification(symbol, analysis)
            else:
                return self._prepare_generic_notification(symbol, analysis)
                
        except Exception as e:
            logger.error(f"Error preparing notification message: {str(e)}")
            return f"Error preparing notification for {symbol}"

    def _prepare_analysis_notification(self, symbol: str, analysis: Dict) -> str:
        """Prepare detailed analysis notification"""
        try:
            recommendation = analysis.get('recommendation', {})
            key_metrics = analysis.get('key_metrics', {})
            
            message = f"🔍 *Stock Analysis: {symbol}*\n\n"
            
            # Add recommendation
            if recommendation:
                action = recommendation.get('action', 'NEUTRAL')
                confidence = recommendation.get('confidence', 0.0)
                timeframe = recommendation.get('timeframe', 'N/A')
                risk_level = recommendation.get('risk_level', 'N/A')
                
                message += f"*Recommendation:* {action}\n"
                message += f"Confidence: {self._round_float(confidence * 100)}%\n"
                message += f"Timeframe: {timeframe}\n"
                message += f"Risk Level: {risk_level}\n\n"
            
            # Add key metrics
            if key_metrics:
                message += "*Key Metrics:*\n"
                message += f"Technical Score: {self._round_float(key_metrics.get('technical_score', 0.0))}\n"
                message += f"Financial Score: {self._round_float(key_metrics.get('financial_score', 0.0))}\n"
                message += f"Growth Score: {self._round_float(key_metrics.get('growth_score', 0.0))}\n"
                message += f"Risk Score: {self._round_float(key_metrics.get('risk_score', 0.0))}\n"
                message += f"Overall Score: {self._round_float(key_metrics.get('overall_score', 0.0))}\n\n"
            
            # Add price targets
            price_targets = analysis.get('analysis', {}).get('price_targets', {})
            if price_targets:
                message += "*Price Targets:*\n"
                message += f"Short Term: {price_targets.get('short_term', 'N/A')}\n"
                message += f"Medium Term: {price_targets.get('medium_term', 'N/A')}\n"
                message += f"Long Term: {price_targets.get('long_term', 'N/A')}\n\n"
            
            # Add summary
            message += "*Summary:*\n"
            message += analysis.get('analysis', {}).get('overview', 'No summary available')
            
            return message
            
        except Exception as e:
            logger.error(f"Error preparing analysis notification: {str(e)}")
            return f"Error preparing analysis notification for {symbol}"

    def _prepare_signal_notification(self, symbol: str, analysis: Dict) -> str:
        """Prepare trading signal notification"""
        try:
            message = f"🚨 *Trading Signal: {symbol}*\n\n"
            
            # Add signal details
            signal_type = analysis.get('signal_type', 'NEUTRAL')
            signal_strength = analysis.get('signal_strength', 0.0)
            current_price = analysis.get('close', 0.0)
            
            message += f"*Signal Type:* {signal_type}\n"
            message += f"Signal Strength: {self._round_float(signal_strength * 100)}%\n"
            message += f"Current Price: ${self._round_float(current_price)}\n\n"
            
            # Add trading levels
            stop_loss = analysis.get('stop_loss')
            take_profit = analysis.get('take_profit')
            risk_reward = analysis.get('risk_reward_ratio')
            
            if all(x is not None for x in [stop_loss, take_profit, risk_reward]):
                message += "*Trading Levels:*\n"
                message += f"Stop Loss: ${self._round_float(stop_loss)}\n"
                message += f"Take Profit: ${self._round_float(take_profit)}\n"
                message += f"Risk/Reward: {self._round_float(risk_reward)}\n\n"
            
            # Add technical indicators
            message += "*Technical Indicators:*\n"
            message += f"RSI: {self._round_float(analysis.get('rsi', 0.0))}\n"
            message += f"MACD: {self._round_float(analysis.get('macd', 0.0))}\n"
            message += f"SMA20: ${self._round_float(analysis.get('sma_20', 0.0))}\n"
            message += f"SMA50: ${self._round_float(analysis.get('sma_50', 0.0))}\n"
            message += f"SMA200: ${self._round_float(analysis.get('sma_200', 0.0))}\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error preparing signal notification: {str(e)}")
            return f"Error preparing signal notification for {symbol}"

    def _prepare_alert_notification(self, symbol: str, analysis: Dict) -> str:
        """Prepare alert notification"""
        try:
            message = f"⚠️ *Alert: {symbol}*\n\n"
            
            # Add alert details
            alert_type = analysis.get('alert_type', 'GENERAL')
            alert_message = analysis.get('alert_message', 'No message available')
            current_price = analysis.get('close', 0.0)
            
            message += f"*Alert Type:* {alert_type}\n"
            message += f"Current Price: ${self._round_float(current_price)}\n\n"
            message += f"*Message:*\n{alert_message}\n\n"
            
            # Add relevant metrics
            if 'technical_score' in analysis:
                message += f"Technical Score: {self._round_float(analysis['technical_score'])}\n"
            if 'financial_score' in analysis:
                message += f"Financial Score: {self._round_float(analysis['financial_score'])}\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error preparing alert notification: {str(e)}")
            return f"Error preparing alert notification for {symbol}"

    def _prepare_generic_notification(self, symbol: str, analysis: Dict) -> str:
        """Prepare generic notification"""
        try:
            message = f"📊 *Update: {symbol}*\n\n"
            
            # Add basic information
            current_price = analysis.get('close', 0.0)
            change = analysis.get('change', 0.0)
            volume = analysis.get('volume', 0)
            
            message += f"Current Price: ${self._round_float(current_price)}\n"
            message += f"Change: {self._round_float(change)}%\n"
            message += f"Volume: {volume:,}\n\n"
            
            # Add any available analysis
            if 'analysis_summary' in analysis:
                message += "*Summary:*\n"
                for point in analysis['analysis_summary']:
                    message += f"• {point}\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error preparing generic notification: {str(e)}")
            return f"Error preparing notification for {symbol}"

    def _send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram"""
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                logger.warning("Telegram credentials not configured")
                return False
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
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
                self.send_notification(symbol, {'analysis': current_analysis}, 'signal')
            
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
                self.send_notification(symbol, {'analysis': current_analysis}, 'signal')
            
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
                        self.send_notification(symbol, {'analysis': current_analysis}, 'alert')
            
        except Exception as e:
            logger.error(f"Error checking signal transitions for {symbol}: {e}")
