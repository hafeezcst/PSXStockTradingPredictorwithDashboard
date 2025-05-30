import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from signal_tracker import SignalState

class SignalDatabase:
    def __init__(self, db_path: str = "data/databases/signals/signal_tracking.db"):
        """
        Initialize the SignalDatabase with database connection.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create signal_tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_tracking (
                    symbol TEXT PRIMARY KEY,
                    current_signal TEXT,
                    last_signal TEXT,
                    current_price REAL,
                    last_price REAL,
                    last_update DATETIME,
                    signal_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create signal_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    signal TEXT,
                    price REAL,
                    timestamp DATETIME,
                    FOREIGN KEY (symbol) REFERENCES signal_tracking(symbol)
                )
            ''')
            
            # Create signal_transition_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_transition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    from_signal TEXT,
                    to_signal TEXT,
                    from_price REAL,
                    to_price REAL,
                    transition_date DATETIME,
                    price_change REAL,
                    price_change_percent REAL,
                    FOREIGN KEY (symbol) REFERENCES signal_tracking(symbol)
                )
            ''')
            
            conn.commit()
    
    def update_signal(self, symbol: str, new_signal: SignalState, price: float) -> Dict:
        """
        Update the signal for a stock and record the transition.
        
        Args:
            symbol (str): Stock symbol
            new_signal (SignalState): New signal state
            price (float): Current price
            
        Returns:
            Dict: Information about the signal transition
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            current_time = datetime.now()
            
            # Get current signal state
            cursor.execute("""
                SELECT current_signal, current_price, signal_count 
                FROM signal_tracking 
                WHERE symbol = ?
            """, (symbol,))
            result = cursor.fetchone()
            
            if result:
                current_signal, current_price, signal_count = result
                if current_signal == new_signal.name:
                    return None  # No change in signal
                
                # Update signal tracking
                cursor.execute("""
                    UPDATE signal_tracking 
                    SET current_signal = ?,
                        last_signal = ?,
                        current_price = ?,
                        last_price = ?,
                        last_update = ?,
                        signal_count = signal_count + 1
                    WHERE symbol = ?
                """, (new_signal.name, current_signal, price, current_price, current_time, symbol))
                
                # Calculate price changes
                price_change = price - current_price
                price_change_percent = (price_change / current_price) * 100 if current_price else 0
                
                # Record transition
                cursor.execute("""
                    INSERT INTO signal_transition_history 
                    (symbol, from_signal, to_signal, from_price, to_price, 
                     transition_date, price_change, price_change_percent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, current_signal, new_signal.name, current_price, price,
                      current_time, price_change, price_change_percent))
                
            else:
                # First time tracking this symbol
                cursor.execute("""
                    INSERT INTO signal_tracking 
                    (symbol, current_signal, last_signal, current_price, last_price, 
                     last_update, signal_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, new_signal.name, None, price, None, current_time, 1))
            
            # Record in signal history
            cursor.execute("""
                INSERT INTO signal_history 
                (symbol, signal, price, timestamp)
                VALUES (?, ?, ?, ?)
            """, (symbol, new_signal.name, price, current_time))
            
            conn.commit()
            
            return {
                'symbol': symbol,
                'previous_signal': current_signal if result else None,
                'new_signal': new_signal.name,
                'previous_price': current_price if result else None,
                'new_price': price,
                'timestamp': current_time,
                'price_change': price_change if result else 0,
                'price_change_percent': price_change_percent if result else 0
            }
    
    def get_current_signal(self, symbol: str) -> Dict:
        """
        Get the current signal state for a stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Current signal information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT current_signal, current_price, last_update, signal_count
                FROM signal_tracking
                WHERE symbol = ?
            """, (symbol,))
            result = cursor.fetchone()
            
            if result:
                return {
                    'symbol': symbol,
                    'current_signal': result[0],
                    'current_price': result[1],
                    'last_update': result[2],
                    'signal_count': result[3]
                }
            return None
    
    def get_signal_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get the signal history for a stock.
        
        Args:
            symbol (str): Stock symbol
            limit (int): Maximum number of history records to return
            
        Returns:
            List[Dict]: List of signal history records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT signal, price, timestamp
                FROM signal_history
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))
            
            return [{
                'signal': row[0],
                'price': row[1],
                'timestamp': row[2]
            } for row in cursor.fetchall()]
    
    def get_transition_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get the signal transition history for a stock.
        
        Args:
            symbol (str): Stock symbol
            limit (int): Maximum number of transition records to return
            
        Returns:
            List[Dict]: List of transition history records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT from_signal, to_signal, from_price, to_price,
                       transition_date, price_change, price_change_percent
                FROM signal_transition_history
                WHERE symbol = ?
                ORDER BY transition_date DESC
                LIMIT ?
            """, (symbol, limit))
            
            return [{
                'from_signal': row[0],
                'to_signal': row[1],
                'from_price': row[2],
                'to_price': row[3],
                'transition_date': row[4],
                'price_change': row[5],
                'price_change_percent': row[6]
            } for row in cursor.fetchall()]

# Example usage
if __name__ == "__main__":
    # Create database instance
    db = SignalDatabase()
    
    # Example of updating signals
    symbol = "PSX"
    
    # Update to buy signal
    transition = db.update_signal(symbol, SignalState.BUY, 100.0)
    print("\nBuy Signal Transition:")
    print(transition)
    
    # Get current signal
    current = db.get_current_signal(symbol)
    print("\nCurrent Signal:")
    print(current)
    
    # Update to neutral signal
    transition = db.update_signal(symbol, SignalState.NEUTRAL, 105.0)
    print("\nNeutral Signal Transition:")
    print(transition)
    
    # Get signal history
    history = db.get_signal_history(symbol)
    print("\nSignal History:")
    print(history)
    
    # Get transition history
    transitions = db.get_transition_history(symbol)
    print("\nTransition History:")
    print(transitions) 