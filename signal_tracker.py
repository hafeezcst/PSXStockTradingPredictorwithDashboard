from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class SignalState(Enum):
    NEUTRAL = "NEUTRAL"
    BUY = "BUY"
    SELL = "SELL"

class SignalTracker:
    def __init__(self):
        self.signal_history: Dict[str, List[Dict]] = {}
        self.current_signals: Dict[str, SignalState] = {}
    
    def update_signal(self, symbol: str, new_signal: SignalState, price: float, timestamp: Optional[datetime] = None) -> Dict:
        """
        Update the signal for a given stock symbol and track the transition.
        
        Args:
            symbol (str): Stock symbol
            new_signal (SignalState): New signal state
            price (float): Current price of the stock
            timestamp (datetime, optional): Timestamp of the signal. Defaults to current time.
        
        Returns:
            Dict: Information about the signal transition
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
            self.current_signals[symbol] = SignalState.NEUTRAL
            
        previous_signal = self.current_signals[symbol]
        
        # Only record if there's a change in signal
        if previous_signal != new_signal:
            transition_info = {
                'timestamp': timestamp,
                'previous_signal': previous_signal.value,
                'new_signal': new_signal.value,
                'price': price,
                'transition': f"{previous_signal.value} -> {new_signal.value}"
            }
            
            self.signal_history[symbol].append(transition_info)
            self.current_signals[symbol] = new_signal
            
            return transition_info
        return None
    
    def get_signal_history(self, symbol: str) -> pd.DataFrame:
        """
        Get the signal history for a specific symbol as a pandas DataFrame.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Signal history for the symbol
        """
        if symbol not in self.signal_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.signal_history[symbol])
    
    def get_current_signal(self, symbol: str) -> SignalState:
        """
        Get the current signal state for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            SignalState: Current signal state
        """
        return self.current_signals.get(symbol, SignalState.NEUTRAL)
    
    def get_all_current_signals(self) -> Dict[str, SignalState]:
        """
        Get all current signals for all tracked symbols.
        
        Returns:
            Dict[str, SignalState]: Dictionary of all current signals
        """
        return self.current_signals.copy()
    
    def get_signal_transitions(self, symbol: str) -> List[Dict]:
        """
        Get all signal transitions for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[Dict]: List of all signal transitions
        """
        return self.signal_history.get(symbol, [])

# Example usage
if __name__ == "__main__":
    # Create a signal tracker instance
    tracker = SignalTracker()
    
    # Example of tracking signals for a stock
    symbol = "AAPL"
    
    # Update signals with some example data
    tracker.update_signal(symbol, SignalState.NEUTRAL, 150.0)
    tracker.update_signal(symbol, SignalState.BUY, 155.0)
    tracker.update_signal(symbol, SignalState.SELL, 160.0)
    
    # Get the signal history
    history_df = tracker.get_signal_history(symbol)
    print("\nSignal History:")
    print(history_df)
    
    # Get current signal
    current_signal = tracker.get_current_signal(symbol)
    print(f"\nCurrent Signal for {symbol}: {current_signal.value}") 