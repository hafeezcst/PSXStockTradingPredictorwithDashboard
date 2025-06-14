from typing import Dict, List, Optional
import logging
import os
import pandas as pd
from datetime import datetime
import json
import concurrent.futures
import time

# Import modules
from modules.data_fetcher import fetch_data_from_tradingview, fetch_psx_symbols, read_psx_announcements
from modules.technical_analyzer import analyze_technical_indicators
from modules.financial_analyzer import load_financial_data, analyze_financials, calculate_intrinsic_value
from modules.ai_analyzer import analyze_with_ai
from modules.db_manager import ensure_database_exists, save_stock_data, save_analysis_results, save_signals, get_latest_stock_data, get_latest_analysis, get_active_signals, update_signal_status
from modules.notifier import notify_new_signals, notify_analysis_results, notify_batch_analysis_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_stock(symbol: str, use_ai: bool = True, notify: bool = True) -> Dict:
    """Analyze a single stock comprehensively"""
    try:
        logger.info(f"Starting analysis for {symbol}")
        
        # Fetch technical data with retry mechanism for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            technical_data = fetch_data_from_tradingview(symbol)
            if technical_data:
                break
            else:
                logger.warning(f"Failed to fetch technical data for {symbol}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 30s, 60s, 120s, etc.
                    wait_time = 30 * (2 ** attempt)
                    logger.info(f"Retrying after {wait_time} seconds due to potential rate limit")
                    time.sleep(wait_time)
        if not technical_data:
            logger.error(f"Failed to fetch technical data for {symbol} after {max_retries} attempts")
            return {}
            
        # Analyze technical indicators
        technical_analysis = analyze_technical_indicators(technical_data)
        
        # Temporarily disable financial data scraping, focus on technical indicators
        financial_data = {}
        financial_analysis = {'financial_score': 0.0, 'confidence': 0.0}
        intrinsic_valuation = {'intrinsic_value': 0.0, 'margin_of_safety': 0.0}
        logger.info(f"Financial data scraping disabled for {symbol}, focusing on technical indicators")
        
        # Read announcements
        announcements_data = read_psx_announcements()
        symbol_announcements = announcements_data.get(symbol, [])
        
        # AI analysis if enabled
        ai_analysis = {}
        if use_ai:
            ai_analysis = analyze_with_ai(symbol, technical_data, financial_analysis, symbol_announcements)
            if not ai_analysis:  # In case AI analysis fails or returns empty
                ai_analysis = {
                    'ai_analysis': 'AI analysis failed or unavailable',
                    'ai_score': 0.0,
                    'ai_recommendation': 'N/A',
                    'ai_confidence': 0.0,
                    'ai_risk_assessment': 'N/A'
                }
        else:
            ai_analysis = {
                'ai_analysis': 'AI analysis disabled',
                'ai_score': 0.0,
                'ai_recommendation': 'N/A',
                'ai_confidence': 0.0,
                'ai_risk_assessment': 'N/A'
            }
        
        # Calculate overall score, focusing on technical indicators
        technical_score = technical_analysis.get('technical_score', 0.0)
        financial_score = 0.0  # Financial score disabled
        ai_score = ai_analysis.get('ai_score', 0.0)
        
        # Weighted overall score, adjusted for disabled financial analysis
        if use_ai:
            overall_score = (technical_score * 0.6 + ai_score * 0.4)
        else:
            overall_score = technical_score
        logger.info(f"Overall score for {symbol} calculated based on technical indicators only")
        
        # Generate recommendation based on overall score with detailed justification
        justification = ""
        if overall_score > 0.3:
            recommendation = "BUY"
            justification = "Recommendation to BUY due to strong positive overall score (above 0.3), indicating bullish technical indicators"
            if use_ai and ai_score > 0.3:
                justification += " supported by positive AI analysis"
            justification += "."
        elif overall_score < -0.3:
            recommendation = "SELL"
            justification = "Recommendation to SELL due to strong negative overall score (below -0.3), indicating bearish technical indicators"
            if use_ai and ai_score < -0.3:
                justification += " supported by negative AI analysis"
            justification += "."
        else:
            recommendation = "HOLD"
            justification = "Recommendation to HOLD as the overall score is neutral (between -0.3 and 0.3), suggesting no clear trend in technical indicators"
            if use_ai:
                justification += " and AI analysis does not provide a strong directional signal"
            justification += "."
            
        # Calculate confidence (average of available confidences from all analysis components)
        # Provide default confidence values if not set by modules
        technical_confidence = technical_analysis.get('confidence', 0.5 if technical_score != 0.0 else 0.0)
        financial_confidence = financial_analysis.get('confidence', 0.5 if financial_score != 0.0 else 0.0)
        ai_confidence = ai_analysis.get('ai_confidence', 0.0)
        confidences = [technical_confidence, financial_confidence, ai_confidence]
        confidence = sum(confidences) / len([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0.0
        
        # Determine risk assessment with fallback if AI analysis is unavailable
        risk_assessment = ai_analysis.get('ai_risk_assessment', 'Medium (Default - AI Unavailable)')
        
        # Compile analysis results with fallback messages for missing data
        intrinsic_value = intrinsic_valuation.get('intrinsic_value', 0.0)
        margin_of_safety = intrinsic_valuation.get('margin_of_safety', 0.0)
        analysis_result = {
            'symbol': symbol,
            'date': technical_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'technical_score': technical_score,
            'financial_score': financial_score,
            'ai_score': ai_score,
            'overall_score': overall_score,
            'recommendation': recommendation,
            'justification': justification,
            'confidence': confidence,
            'risk_assessment': risk_assessment,
            'intrinsic_value': intrinsic_value if intrinsic_value != 0.0 else "Unavailable (No Financial Data)",
            'margin_of_safety': margin_of_safety if margin_of_safety != 0.0 else "Unavailable (No Financial Data)",
            'current_price': technical_data.get('close', 0.0),
            'analysis_details': {
                'technical_analysis': technical_analysis,
                'financial_analysis': financial_analysis,
                'ai_analysis': ai_analysis,
                'intrinsic_valuation': intrinsic_valuation,
                'announcements': symbol_announcements[:3] if symbol_announcements else []
            }
        }
        
        # Save data to database
        save_stock_data([technical_data])
        save_analysis_results([analysis_result])
        
        # Generate and save signals if applicable
        signals = generate_signals(symbol, analysis_result, technical_data)
        if signals:
            save_signals(signals)
            if notify:
                notify_new_signals(signals)
        
        # Send notification if enabled
        if notify:
            formatted_result = analysis_result.copy()
            # Pre-format all values as strings to avoid formatting errors in notifier
            formatted_result['intrinsic_value'] = str(intrinsic_value if isinstance(intrinsic_value, str) else (f"{intrinsic_value:.2f}" if intrinsic_value != 0.0 else "Unavailable (No Financial Data)"))
            formatted_result['margin_of_safety'] = str(margin_of_safety if isinstance(margin_of_safety, str) else (f"{margin_of_safety:.2%}" if margin_of_safety != 0.0 else "Unavailable (No Financial Data)"))
            formatted_result['risk_assessment'] = str(risk_assessment if risk_assessment != 'N/A' else 'Medium (Default - AI Unavailable)')
            formatted_result['current_price'] = str(f"{technical_data.get('close', 0.0):.2f}")
            formatted_result['justification'] = justification
            notify_analysis_results([symbol], [formatted_result])
            
        logger.info(f"Completed analysis for {symbol}: Score={overall_score:.2f}, Recommendation={recommendation}")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing stock {symbol}: {e}")
        return {}

def analyze_batch_stocks(symbols: Optional[List[str]] = None, use_ai: bool = True, notify: bool = True, delay_seconds: int = 5) -> List[Dict]:
    """Analyze multiple stocks sequentially with a delay to avoid rate limits"""
    try:
        if symbols is None:
            symbols = fetch_psx_symbols()
            
        if not symbols:
            logger.error("No symbols provided for batch analysis")
            return []
            
        logger.info(f"Starting batch analysis for {len(symbols)} stocks")
        
        # Ensure database exists
        ensure_database_exists()
        
        results = []
        # Process stocks sequentially with a delay to avoid rate limits
        for i, symbol in enumerate(symbols):
            try:
                result = analyze_stock(symbol, use_ai, notify)
                if result:
                    results.append(result)
                    logger.info(f"Completed analysis for {symbol}")
                else:
                    logger.warning(f"Analysis failed for {symbol}")
                # Add delay between requests to avoid rate limiting
                if i < len(symbols) - 1:  # No delay after the last symbol
                    time.sleep(delay_seconds)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Send batch notification if enabled
        if notify and results:
            notify_batch_analysis_results(results, [r['symbol'] for r in results])
            
        logger.info(f"Completed batch analysis for {len(results)}/{len(symbols)} stocks")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch stock analysis: {e}")
        return []

def generate_signals(symbol: str, analysis: Dict, technical_data: Dict) -> List[Dict]:
    """Generate trading signals based on analysis results"""
    try:
        signals = []
        overall_score = analysis.get('overall_score', 0.0)
        recommendation = analysis.get('recommendation', 'HOLD')
        confidence = analysis.get('confidence', 0.0)
        date = analysis.get('date', datetime.now().strftime('%Y-%m-%d'))
        current_price = technical_data.get('close', 0.0)
        
        # Check for BUY signal
        if recommendation == "BUY" and overall_score > 0.3 and confidence > 0.6:
            signal_strength = overall_score * confidence
            target_price = current_price * 1.1  # 10% target
            stop_loss = current_price * 0.95   # 5% stop loss
            
            signals.append({
                'symbol': symbol,
                'date': date,
                'signal_type': 'BUY',
                'signal_strength': signal_strength,
                'price_at_signal': current_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'signal_details': {
                    'reason': 'Strong buy recommendation based on comprehensive analysis',
                    'overall_score': overall_score,
                    'technical_indicators': analysis.get('analysis_details', {}).get('technical_analysis', {}).get('key_indicators', [])
                }
            })
        
        # Check for SELL signal
        elif recommendation == "SELL" and overall_score < -0.3 and confidence > 0.6:
            signal_strength = abs(overall_score) * confidence
            target_price = current_price * 0.9  # 10% target down
            stop_loss = current_price * 1.05   # 5% stop loss up
            
            signals.append({
                'symbol': symbol,
                'date': date,
                'signal_type': 'SELL',
                'signal_strength': signal_strength,
                'price_at_signal': current_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'signal_details': {
                    'reason': 'Strong sell recommendation based on comprehensive analysis',
                    'overall_score': overall_score,
                    'technical_indicators': analysis.get('analysis_details', {}).get('technical_analysis', {}).get('key_indicators', [])
                }
            })
            
        return signals
        
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")
        return []

def monitor_signals():
    """Monitor active signals and update their status"""
    try:
        active_signals = get_active_signals()
        if not active_signals:
            logger.info("No active signals to monitor")
            return
            
        logger.info(f"Monitoring {len(active_signals)} active signals")
        
        for signal in active_signals:
            symbol = signal['symbol']
            signal_id = signal['id']
            signal_type = signal['signal_type']
            target_price = signal['target_price']
            stop_loss = signal['stop_loss']
            price_at_signal = signal['price_at_signal']
            
            # Get latest price data
            latest_data = get_latest_stock_data(symbol)
            if not latest_data:
                logger.warning(f"No price data available for {symbol}")
                continue
                
            current_price = latest_data[0]['close']
            current_date = latest_data[0]['date']
            
            # Check if target reached or stop loss triggered
            status_updated = False
            profit_loss = 0.0
            
            if signal_type == "BUY":
                if current_price >= target_price:
                    logger.info(f"BUY target reached for {symbol}: {current_price} >= {target_price}")
                    status_updated = update_signal_status(signal_id, "target_reached", current_price, current_date)
                    profit_loss = current_price - price_at_signal
                elif current_price <= stop_loss:
                    logger.info(f"BUY stop loss triggered for {symbol}: {current_price} <= {stop_loss}")
                    status_updated = update_signal_status(signal_id, "stop_loss", current_price, current_date)
                    profit_loss = current_price - price_at_signal
                    
            elif signal_type == "SELL":
                if current_price <= target_price:
                    logger.info(f"SELL target reached for {symbol}: {current_price} <= {target_price}")
                    status_updated = update_signal_status(signal_id, "target_reached", current_price, current_date)
                    profit_loss = price_at_signal - current_price
                elif current_price >= stop_loss:
                    logger.info(f"SELL stop loss triggered for {symbol}: {current_price} >= {stop_loss}")
                    status_updated = update_signal_status(signal_id, "stop_loss", current_price, current_date)
                    profit_loss = price_at_signal - current_price
                    
            if status_updated and profit_loss != 0.0:
                update_signal_status(signal_id, None, None, None, profit_loss)
                
        logger.info("Completed monitoring active signals")
        
    except Exception as e:
        logger.error(f"Error monitoring signals: {e}")

def main():
    """Main function to run stock analysis"""
    try:
        logger.info("Starting PSX Stock Analysis Tool")
        
        # Ensure database exists
        ensure_database_exists()
        
        # Get symbols to analyze
        symbols = fetch_psx_symbols()
        if not symbols:
            logger.error("No symbols available for analysis")
            return
            
        logger.info(f"Analyzing {len(symbols)} stocks...")
        
        # Analyze stocks in batch
        results = analyze_batch_stocks(symbols, use_ai=True, notify=True, delay_seconds=30)
        
        # Monitor signals
        monitor_signals()
        
        logger.info("PSX Stock Analysis completed")
        
    except Exception as e:
        logger.error(f"Error in main analysis process: {e}")

if __name__ == "__main__":
    main()
