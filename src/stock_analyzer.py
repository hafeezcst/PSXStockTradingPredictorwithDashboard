def analyze_stocks(filebuy_path: str, filesell_path: str, filestock_path: str) -> str:
    """
    Analyzes stock data by comparing stock file with buy and sell signal files.
    
    Args:
        filebuy_path (str): Path to the buy signals file
        filesell_path (str): Path to the sell signals file
        filestock_path (str): Path to the stock data file
        
    Returns:
        str: Analysis results showing buy/sell signals or neutral status
    """
    required_line = ""
    
    try:
        # Open all files
        with open(filestock_path, 'r') as stock_file, \
             open(filebuy_path, 'r') as buy_file, \
             open(filesell_path, 'r') as sell_file:
            
            # Read through stock file line by line
            for stock_line in stock_file:
                stock_line = stock_line.strip()
                found = False
                
                # Reset buy and sell files to start for each stock line
                buy_file.seek(0)
                sell_file.seek(0)
                
                # Read through buy and sell files simultaneously
                for buy_line, sell_line in zip(buy_file, sell_file):
                    buy_line = buy_line.strip()
                    sell_line = sell_line.strip()
                    
                    if stock_line == buy_line:
                        required_line += f"{stock_line} ---- buy signal ---- Found\n"
                        found = True
                        break
                    elif stock_line == sell_line:
                        required_line += f"{stock_line} ---- sell signal ---- Found\n"
                        found = True
                        break
                
                if not found:
                    required_line += f"{stock_line} ---- neutral\n"
                    
    except FileNotFoundError as e:
        return f"Error: File not found - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
    
    return required_line

if __name__ == "__main__":
    # Example usage
    buy_file = "data/buy_signals.txt"
    sell_file = "data/sell_signals.txt"
    stock_file = "data/stock_data.txt"
    
    result = analyze_stocks(buy_file, sell_file, stock_file)
    print(result) 