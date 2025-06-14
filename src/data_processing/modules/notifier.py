from typing import Dict, List, Optional
import logging
import os
import requests
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

def send_telegram_notification(message: str, chat_id: Optional[str] = None) -> bool:
    """Send notification via Telegram"""
    try:
        if not TELEGRAM_BOT_TOKEN or not (chat_id or TELEGRAM_CHAT_ID):
            logger.error("Telegram notification credentials not fully configured")
            return False
            
        target_chat_id = chat_id if chat_id else TELEGRAM_CHAT_ID
        
        payload = {
            "chat_id": target_chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(TELEGRAM_API_URL, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Telegram notification sent successfully to {target_chat_id}")
            return True
        else:
            logger.error(f"Failed to send Telegram notification: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending Telegram notification: {e}")
        return False

def format_signal_notification(signal: Dict) -> str:
    """Format a trading signal into a notification message"""
    try:
        symbol = signal.get('symbol', 'N/A')
        signal_type = signal.get('signal_type', 'N/A')
        price = signal.get('price_at_signal', 'N/A')
        target = signal.get('target_price', 'N/A')
        stop_loss = signal.get('stop_loss', 'N/A')
        confidence = signal.get('confidence', 0.0)
        date = signal.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        message = f"""
<b>🚨 New Trading Signal for {symbol} 🚨</b>
📅 Date: {date}
📊 Signal: <b>{signal_type.upper()}</b>
💰 Entry Price: {price}
🎯 Target Price: {target}
🛑 Stop Loss: {stop_loss}
🔍 Confidence: {confidence:.2%}

#PSX #TradingSignal #StockMarket
"""
        return message
        
    except Exception as e:
        logger.error(f"Error formatting signal notification: {e}")
        return "Error formatting signal notification"

def format_analysis_report(symbol: str, analysis: Dict) -> str:
    """Format analysis results into a notification report"""
    try:
        date = analysis.get('date', datetime.now().strftime('%Y-%m-%d'))
        overall_score = analysis.get('overall_score', 0.0)
        recommendation = analysis.get('recommendation', 'N/A')
        confidence = analysis.get('confidence', 0.0)
        risk = analysis.get('risk_assessment', 'N/A')
        intrinsic_value = analysis.get('intrinsic_value', 0.0)
        margin_safety = analysis.get('margin_of_safety', 0.0)
        current_price = analysis.get('current_price', 0.0)
        
        # Determine emoji based on recommendation
        rec_emoji = "🟢" if recommendation.lower() == "buy" else "🔴" if recommendation.lower() == "sell" else "🟡"
        
        # Format numeric values only if they are numbers, otherwise use as-is
        current_price_str = f"{float(current_price):.2f}" if isinstance(current_price, (int, float)) else str(current_price)
        intrinsic_value_str = f"{float(intrinsic_value):.2f}" if isinstance(intrinsic_value, (int, float)) else str(intrinsic_value)
        margin_safety_str = f"{float(margin_safety):.2%}" if isinstance(margin_safety, (int, float)) else str(margin_safety)
        overall_score_str = f"{overall_score:.2f}" if isinstance(overall_score, (int, float)) else str(overall_score)
        confidence_str = f"{confidence:.2%}" if isinstance(confidence, (int, float)) else str(confidence)
        
        message = f"""
<b>📈 {symbol} Analysis Report 📈</b>
📅 Date: {date}
📊 Overall Score: {overall_score_str}/1.0
{rec_emoji} Recommendation: <b>{recommendation.upper()}</b>
🔍 Confidence: {confidence_str}
⚠️ Risk Level: {risk}
💰 Current Price: {current_price_str}
🎯 Intrinsic Value: {intrinsic_value_str}
🛡️ Margin of Safety: {margin_safety_str}

#PSX #StockAnalysis #Investment
"""
        return message
        
    except Exception as e:
        logger.error(f"Error formatting analysis report: {e}")
        return "Error formatting analysis report"

def notify_new_signals(signals: List[Dict], chat_id: Optional[str] = None) -> int:
    """Send notifications for new trading signals"""
    try:
        if not signals:
            logger.info("No new signals to notify")
            return 0
            
        count = 0
        for signal in signals:
            message = format_signal_notification(signal)
            if send_telegram_notification(message, chat_id):
                count += 1
                
        logger.info(f"Sent notifications for {count} new signals")
        return count
        
    except Exception as e:
        logger.error(f"Error notifying new signals: {e}")
        return 0

def notify_analysis_results(symbols: List[str], analyses: List[Dict], chat_id: Optional[str] = None) -> int:
    """Send notifications for analysis results"""
    try:
        if not symbols or not analyses or len(symbols) != len(analyses):
            logger.error("Invalid input for analysis notifications")
            return 0
            
        count = 0
        for symbol, analysis in zip(symbols, analyses):
            message = format_analysis_report(symbol, analysis)
            if send_telegram_notification(message, chat_id):
                count += 1
                
        logger.info(f"Sent analysis notifications for {count} symbols")
        return count
        
    except Exception as e:
        logger.error(f"Error notifying analysis results: {e}")
        return 0

def notify_batch_analysis_results(analyses: List[Dict], symbols: List[str], chat_id: Optional[str] = None) -> bool:
    """Send a batch notification for multiple analysis results"""
    try:
        if not analyses or not symbols or len(analyses) != len(symbols):
            logger.error("Invalid input for batch analysis notification")
            return False
            
        # Create a summary report
        message_lines = ["<b>📊 PSX Batch Analysis Report 📊</b>"]
        message_lines.append(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
        message_lines.append(f"📈 Total Stocks Analyzed: {len(symbols)}")
        message_lines.append("\n<b>Summary:</b>")
        
        buy_count = sum(1 for a in analyses if a.get('recommendation', '').lower() == 'buy')
        hold_count = sum(1 for a in analyses if a.get('recommendation', '').lower() == 'hold')
        sell_count = sum(1 for a in analyses if a.get('recommendation', '').lower() == 'sell')
        
        message_lines.append(f"🟢 Buy Recommendations: {buy_count}")
        message_lines.append(f"🟡 Hold Recommendations: {hold_count}")
        message_lines.append(f"🔴 Sell Recommendations: {sell_count}")
        
        # Add top buy recommendations
        if buy_count > 0:
            message_lines.append("\n<b>Top Buy Recommendations:</b>")
            buy_analyses = [(s, a) for s, a in zip(symbols, analyses) if a.get('recommendation', '').lower() == 'buy']
            buy_analyses.sort(key=lambda x: x[1].get('overall_score', 0.0), reverse=True)
            
            for symbol, analysis in buy_analyses[:3]:  # Top 3 buy recommendations
                score = analysis.get('overall_score', 0.0)
                confidence = analysis.get('confidence', 0.0)
                message_lines.append(f"🟢 {symbol}: Score={score:.2f}, Confidence={confidence:.2%}")
        
        message_lines.append("\n#PSX #StockAnalysis #Investment")
        message = "\n".join(message_lines)
        
        if send_telegram_notification(message, chat_id):
            logger.info("Sent batch analysis notification")
            return True
        else:
            logger.error("Failed to send batch analysis notification")
            return False
            
    except Exception as e:
        logger.error(f"Error sending batch analysis notification: {e}")
        return False
