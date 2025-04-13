import logging
from typing import Tuple, List, Optional
import pandas as pd
from datetime import datetime

def analyze_stock_signals(stock_file: str, buy_file: str, sell_file: str) -> List[Tuple[str, str]]:
    """
    Analyze stock signals by comparing stock data with buy and sell signals.
    
    Args:
        stock_file (str): Path to the stock data file
        buy_file (str): Path to the buy signals file
        sell_file (str): Path to the sell signals file
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (stock_line, signal_type)
    """
    try:
        # Read the files
        stock_df = pd.read_csv(stock_file)
        buy_df = pd.read_csv(buy_file)
        sell_df = pd.read_csv(sell_file)
        
        results = []
        found = False
        
        # Iterate through each stock line
        for _, stock_row in stock_df.iterrows():
            stock_line = stock_row.to_string()
            signal_type = "neutral"
            
            # Check against buy signals
            for _, buy_row in buy_df.iterrows():
                if stock_line == buy_row.to_string():
                    signal_type = "buy signal ---- Found"
                    found = True
                    break
            
            # If not found in buy signals, check sell signals
            if not found:
                for _, sell_row in sell_df.iterrows():
                    if stock_line == sell_row.to_string():
                        signal_type = "sell signal ---- Found"
                        found = True
                        break
            
            results.append((stock_line, signal_type))
            
            # Reset found flag for next iteration
            found = False
        
        return results
        
    except Exception as e:
        logging.error(f"Error analyzing stock signals: {str(e)}")
        return []

def main():
    # Example usage
    stock_file = "data/stock_data.csv"
    buy_file = "data/buy_signals.csv"
    sell_file = "data/sell_signals.csv"
    
    results = analyze_stock_signals(stock_file, buy_file, sell_file)
    
    # Print results
    for stock_line, signal_type in results:
        print(f"{stock_line} - {signal_type}")

if __name__ == "__main__":
    main() 