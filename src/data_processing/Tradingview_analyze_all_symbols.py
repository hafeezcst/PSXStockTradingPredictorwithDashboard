import pandas as pd
from fair_value_calculator import FairValueCalculator
import logging
import time
from datetime import datetime
import os
from tradingview_ta import TA_Handler, Interval
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('symbol_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def round_decimal(value, places=2):
    """Helper function to round decimal values to specified places"""
    if value is None:
        return None
    try:
        return round(float(value), places)
    except (ValueError, TypeError):
        return value

def fetch_and_store_data_for_all_symbols():
    """Fetch and store technical data for all symbols using tradingview_ta if not already in the database."""
    db_path = 'data/database/stock_analysis.db'
    db_manager = DatabaseManager(db_path)
    excel_path = "src/data_processing/psxsymbols.xlsx"
    logger.info(f"Reading symbols from {excel_path} for data fetch")
    df = pd.read_excel(excel_path, engine='openpyxl')
    symbol_column = [col for col in df.columns if 'symbol' in col.lower()][0]
    symbols = df[symbol_column].tolist()
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"Checking data for symbol {i}/{len(symbols)}: {symbol}")
            data = db_manager.get_latest_data(symbol)
            if data:
                logger.info(f"Data already exists for {symbol}, skipping fetch.")
                continue
            logger.info(f"Fetching data for {symbol} from TradingView...")
            handler = TA_Handler(
                symbol=symbol,
                exchange="PSX",
                screener="pakistan",
                interval=Interval.INTERVAL_1_DAY
            )
            analysis = handler.get_analysis()
            indicators = analysis.indicators
            data_dict = {
                'symbol': symbol,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'close': round_decimal(indicators.get('close')),
                'open': round_decimal(indicators.get('open')),
                'high': round_decimal(indicators.get('high')),
                'low': round_decimal(indicators.get('low')),
                'volume': indicators.get('volume'),
                'rsi': round_decimal(indicators.get('RSI')),
                'macd': round_decimal(indicators.get('MACD.macd')),
                'macd_signal': round_decimal(indicators.get('MACD.signal')),
                'sma_20': round_decimal(indicators.get('SMA20')),
                'sma_50': round_decimal(indicators.get('SMA50')),
                'sma_200': round_decimal(indicators.get('SMA200')),
                'ema_20': round_decimal(indicators.get('EMA20')),
                'ema_50': round_decimal(indicators.get('EMA50')),
                'ema_200': round_decimal(indicators.get('EMA200')),
                'bb_upper': round_decimal(indicators.get('BB.upper')),
                'bb_lower': round_decimal(indicators.get('BB.lower')),
                'change': None,
                'change_percent': None,
            }
            if data_dict['close'] is not None and data_dict['open'] is not None:
                data_dict['change'] = round_decimal(data_dict['close'] - data_dict['open'])
                data_dict['change_percent'] = round_decimal(((data_dict['close'] - data_dict['open']) / data_dict['open']) * 100 if data_dict['open'] else 0)
            db_manager.save_tradingview_ta_data(symbol, data_dict)
            logger.info(f"Saved TradingView data for {symbol}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error fetching/saving TradingView data for {symbol}: {str(e)}")
            continue

def analyze_all_symbols():
    try:
        # Initialize the calculator
        calculator = FairValueCalculator()
        
        # Read the Excel file
        excel_path = "src/data_processing/psxsymbols.xlsx"
        logger.info(f"Reading symbols from {excel_path}")
        
        df = pd.read_excel(excel_path, engine='openpyxl')
        
        # Get the symbol column (assuming it's named 'Symbol' or similar)
        symbol_column = [col for col in df.columns if 'symbol' in col.lower()][0]
        symbols = df[symbol_column].tolist()
        
        logger.info(f"Found {len(symbols)} symbols to analyze")
        
        # Create results directory if it doesn't exist
        results_dir = "../data/analysis_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results list
        results = []
        
        # Analyze each symbol
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"Analyzing symbol {i}/{len(symbols)}: {symbol}")
                
                # Get recommendation
                recommendation = calculator.get_stock_recommendation(symbol)
                
                if recommendation:
                    results.append({
                        'symbol': symbol,
                        'timestamp': recommendation['timestamp'],
                        'action': recommendation['recommendation']['action'],
                        'confidence': round_decimal(recommendation['recommendation']['confidence']),
                        'overall_score': round_decimal(recommendation['key_metrics']['overall_score']),
                        'technical_score': round_decimal(recommendation['key_metrics']['technical_score']),
                        'financial_score': round_decimal(recommendation['key_metrics']['financial_score'])
                    })
                    
                    # Save individual analysis to file
                    analysis_file = f"{results_dir}/{symbol}_analysis.txt"
                    with open(analysis_file, 'w') as f:
                        f.write(f"Analysis for {symbol}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Timestamp: {recommendation['timestamp']}\n")
                        f.write(f"Action: {recommendation['recommendation']['action']}\n")
                        f.write(f"Confidence: {round_decimal(recommendation['recommendation']['confidence']):.2f}\n")
                        f.write(f"Overall Score: {round_decimal(recommendation['key_metrics']['overall_score']):.2f}\n")
                        f.write("\nAnalysis Summary:\n")
                        for point in recommendation['summary']:
                            if point:
                                f.write(f"• {point}\n")
                else:
                    logger.warning(f"No analysis available for {symbol}")
                
                # Add a small delay to avoid overwhelming the system
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Save summary results to Excel
        if results:
            summary_df = pd.DataFrame(results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_file = f"{results_dir}/analysis_summary_{timestamp}.xlsx"
            summary_df.to_excel(summary_file, index=False)
            logger.info(f"Analysis summary saved to {summary_file}")
        
        logger.info("Analysis completed!")
        
    except Exception as e:
        logger.error(f"Error in analyze_all_symbols: {str(e)}")

if __name__ == "__main__":
    fetch_and_store_data_for_all_symbols()
    analyze_all_symbols() 