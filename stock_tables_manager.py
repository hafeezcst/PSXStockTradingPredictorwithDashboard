import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from signal_tracker import SignalState, SignalTracker

class StockTablesManager:
    def __init__(self, db_path: str = "data/databases/production/PSX_investing_Stocks_KMI30.db"):
        """
        Initialize the StockTablesManager with database connection.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.signal_tracker = SignalTracker()
    
    def update_stock_signal(self, symbol: str, new_signal: SignalState, price: float, 
                          volume: float = None,
                          rsi_weekly: float = None,
                          rsi_3months: float = None,
                          ao_weekly: float = None,
                          ma_30: float = None,
                          multibagger: str = None,
                          free_float_ratio: str = None) -> Dict:
        """
        Update stock signal and manage tables accordingly.
        
        Args:
            symbol (str): Stock symbol
            new_signal (SignalState): New signal state
            price (float): Current price
            volume (float, optional): Trading volume
            rsi_weekly (float, optional): Weekly RSI average
            rsi_3months (float, optional): 3-month RSI average
            ao_weekly (float, optional): Weekly Awesome Oscillator
            ma_30 (float, optional): 30-day Moving Average
            multibagger (str, optional): Multibagger status
            free_float_ratio (str, optional): Free float ratio
        
        Returns:
            Dict: Information about the signal transition
        """
        # Update signal tracker
        transition_info = self.signal_tracker.update_signal(symbol, new_signal, price)
        
        if transition_info:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_date = datetime.now().strftime('%Y-%m-%d')
                
                # Remove from all tables first
                cursor.execute("DELETE FROM buy_stocks WHERE Stock = ?", (symbol,))
                cursor.execute("DELETE FROM neutral_stocks WHERE Stock = ?", (symbol,))
                cursor.execute("DELETE FROM sell_stocks WHERE Stock = ?", (symbol,))
                
                # Common fields for all tables
                common_fields = {
                    'Stock': symbol,
                    'Data Source': 'Signal Tracker',
                    'Date': current_date,
                    'Close': price,
                    'Volume': volume,
                    'RSI_Weekly_Avg': rsi_weekly,
                    'RSI_3Months_Avg_Recent': rsi_3months,
                    'AO_Weekly': ao_weekly,
                    'MA_30': ma_30,
                    'Multibagger': multibagger,
                    'FreeFloatRatio': free_float_ratio,
                    'Signal_Date': current_date,
                    'Signal_Close': price,
                    'Status': new_signal.name
                }
                
                # Insert into appropriate table based on new signal
                if new_signal == SignalState.BUY:
                    fields = list(common_fields.keys())
                    placeholders = ', '.join(['?' for _ in fields])
                    values = list(common_fields.values())
                    
                    cursor.execute(f'''
                        INSERT INTO buy_stocks ({', '.join(fields)})
                        VALUES ({placeholders})
                    ''', values)
                
                elif new_signal == SignalState.NEUTRAL:
                    fields = list(common_fields.keys())
                    placeholders = ', '.join(['?' for _ in fields])
                    values = list(common_fields.values())
                    
                    cursor.execute(f'''
                        INSERT INTO neutral_stocks ({', '.join(fields)})
                        VALUES ({placeholders})
                    ''', values)
                
                elif new_signal == SignalState.SELL:
                    # Calculate profit/loss if we have previous entry
                    profit_loss = None
                    holding_days = None
                    
                    # Get previous entry from buy_stocks
                    cursor.execute("SELECT Signal_Date, Signal_Close FROM buy_stocks WHERE Stock = ?", (symbol,))
                    prev_entry = cursor.fetchone()
                    
                    if prev_entry:
                        entry_date = datetime.strptime(prev_entry[0], '%Y-%m-%d')
                        entry_price = prev_entry[1]
                        current_date_obj = datetime.strptime(current_date, '%Y-%m-%d')
                        holding_days = (current_date_obj - entry_date).days
                        profit_loss = ((price - entry_price) / entry_price) * 100
                    
                    common_fields['% P/L'] = profit_loss
                    common_fields['Holding_Days'] = holding_days
                    
                    fields = list(common_fields.keys())
                    placeholders = ', '.join(['?' for _ in fields])
                    values = list(common_fields.values())
                    
                    cursor.execute(f'''
                        INSERT INTO sell_stocks ({', '.join(fields)})
                        VALUES ({placeholders})
                    ''', values)
                
                conn.commit()
        
        return transition_info
    
    def get_buy_stocks(self) -> List[Dict]:
        """Get all stocks in buy state."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM buy_stocks")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_neutral_stocks(self) -> List[Dict]:
        """Get all stocks in neutral state."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM neutral_stocks")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_sell_stocks(self) -> List[Dict]:
        """Get all stocks in sell state."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sell_stocks")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_stock_status(self, symbol: str) -> Dict:
        """
        Get current status of a stock from all tables.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Current status of the stock
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check buy_stocks
            cursor.execute("SELECT * FROM buy_stocks WHERE Stock = ?", (symbol,))
            buy_data = cursor.fetchone()
            if buy_data:
                return {
                    'status': 'BUY',
                    'data': dict(zip([col[0] for col in cursor.description], buy_data))
                }
            
            # Check neutral_stocks
            cursor.execute("SELECT * FROM neutral_stocks WHERE Stock = ?", (symbol,))
            neutral_data = cursor.fetchone()
            if neutral_data:
                return {
                    'status': 'NEUTRAL',
                    'data': dict(zip([col[0] for col in cursor.description], neutral_data))
                }
            
            # Check sell_stocks
            cursor.execute("SELECT * FROM sell_stocks WHERE Stock = ?", (symbol,))
            sell_data = cursor.fetchone()
            if sell_data:
                return {
                    'status': 'SELL',
                    'data': dict(zip([col[0] for col in cursor.description], sell_data))
                }
            
            return {'status': 'NOT_FOUND'}

# Example usage
if __name__ == "__main__":
    # Create tables manager instance
    manager = StockTablesManager()
    
    # Example of updating stock signals
    symbol = "PSX"
    
    # Update to buy signal
    manager.update_stock_signal(
        symbol=symbol,
        new_signal=SignalState.BUY,
        price=100.0,
        volume=1000000,
        rsi_weekly=65.0,
        rsi_3months=60.0,
        ao_weekly=2.5,
        ma_30=98.0,
        multibagger="Yes",
        free_float_ratio="0.75"
    )
    
    # Get current buy stocks
    buy_stocks = manager.get_buy_stocks()
    print("\nBuy Stocks:")
    print(buy_stocks)
    
    # Update to neutral signal
    manager.update_stock_signal(
        symbol=symbol,
        new_signal=SignalState.NEUTRAL,
        price=105.0,
        volume=1200000,
        rsi_weekly=70.0,
        rsi_3months=65.0,
        ao_weekly=3.0,
        ma_30=100.0
    )
    
    # Get current neutral stocks
    neutral_stocks = manager.get_neutral_stocks()
    print("\nNeutral Stocks:")
    print(neutral_stocks)
    
    # Update to sell signal
    manager.update_stock_signal(
        symbol=symbol,
        new_signal=SignalState.SELL,
        price=110.0,
        volume=1500000,
        rsi_weekly=75.0,
        rsi_3months=70.0,
        ao_weekly=3.5,
        ma_30=102.0
    )
    
    # Get current sell stocks
    sell_stocks = manager.get_sell_stocks()
    print("\nSell Stocks:")
    print(sell_stocks) 