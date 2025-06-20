#!/usr/bin/env python3
"""
Simple Portfolio Management System - Working Version
Integrates with existing signal database for automated trading
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class SimplePortfolioManager:
    def __init__(self, 
                 portfolio_value: float = 35_000_000,  # 35 Million PKR
                 signal_db_path: str = None,  # Will be set dynamically
                 portfolio_file: str = None,  # Will be set dynamically
                 max_positions: int = 50,
                 min_position_size: float = 100_000,  # 100K PKR minimum
                 max_position_size: float = 2_000_000,  # 2M PKR maximum
                 transaction_cost: float = 0.002):  # 0.2% transaction cost
          # Set default paths relative to current file location
        if signal_db_path is None:
            # Go up to project root and find the database
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..', '..')
            project_root = os.path.abspath(project_root)
            signal_db_path = os.path.join(project_root, "data", "databases", "production", "PSX_investing_Stocks_KMI100.db")
            signal_db_path = os.path.abspath(signal_db_path)
        
        if portfolio_file is None:
            # Use data folder in current module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            portfolio_file = os.path.join(current_dir, '..', 'data', 'simple_portfolio.json')
            portfolio_file = os.path.abspath(portfolio_file)
        
        self.portfolio_value = portfolio_value
        self.signal_db_path = signal_db_path
        self.portfolio_file = portfolio_file
        self.max_positions = max_positions
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('SimplePortfolioManager')
        
        # Initialize portfolio data
        self.load_or_create_portfolio()
        
        self.logger.info(f"Simple Portfolio Manager initialized with {self.portfolio_value:,.0f} PKR")
    
    def load_or_create_portfolio(self):
        """Load existing portfolio or create new one"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    portfolio = json.load(f)
                self.cash_balance = portfolio.get('cash_balance', self.portfolio_value)
                self.positions = portfolio.get('positions', {})
                self.trade_history = portfolio.get('trade_history', [])
                self.logger.info("Loaded existing portfolio data")
            else:
                self.cash_balance = self.portfolio_value
                self.positions = {}
                self.trade_history = []
                self.save_portfolio()
                self.logger.info("Created new portfolio")
        except Exception as e:
            self.logger.error(f"Error loading portfolio: {e}")
            self.cash_balance = self.portfolio_value
            self.positions = {}
            self.trade_history = []
    
    def save_portfolio(self):
        """Save portfolio data"""
        try:
            os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)
            portfolio_data = {
                'cash_balance': self.cash_balance,                'positions': self.positions,
                'trade_history': self.trade_history,
                'last_update': datetime.now().isoformat()
            }
            with open(self.portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=2, default=str)
            self.logger.info("Portfolio data saved")
        except Exception as e:
            self.logger.error(f"Error saving portfolio: {e}")
    
    def get_current_signals(self) -> pd.DataFrame:
        """
        Get latest 50 buy signals from database (ordered by signal date)
        
        UPDATED: Now orders by Signal_Date DESC to get the most recent signals,
        rather than ordering by P&L performance. This ensures the portfolio
        targets the latest market opportunities instead of historical performers.
        
        Fallback: If Signal_Date ordering fails, uses rowid DESC (insertion order)
        which approximates recency based on when signals were generated.
        
        Returns:
            pd.DataFrame: Latest 50 buy signals with stock data
        """
        try:
            conn = sqlite3.connect(self.signal_db_path)
            
            # First try ordering by Signal_Date if column exists
            query_with_date = """
            SELECT Stock, Close, Volume, RSI_Weekly_Avg, [% P/L] as PnL_Percent, 
                   Signal_Date, Signal_Close, Status, Success
            FROM buy_stocks 
            WHERE Status = 'Buy'
            ORDER BY Signal_Date DESC
            LIMIT 50
            """
            
            # Fallback query if Signal_Date doesn't exist or fails
            query_fallback = """
            SELECT Stock, Close, Volume, RSI_Weekly_Avg, [% P/L] as PnL_Percent, 
                   Signal_Date, Signal_Close, Status, Success
            FROM buy_stocks 
            WHERE Status = 'Buy' AND Success = 'Yes'
            ORDER BY rowid DESC
            LIMIT 50
            """
            
            try:
                # Try date-based ordering first
                signals = pd.read_sql_query(query_with_date, conn)
                self.logger.info(f"Retrieved {len(signals)} latest buy signals (ordered by Signal_Date)")
            except Exception as date_error:
                # Fallback to rowid ordering (insertion order = latest)
                self.logger.warning(f"Signal_Date ordering failed ({date_error}), using rowid ordering")
                signals = pd.read_sql_query(query_fallback, conn)
                self.logger.info(f"Retrieved {len(signals)} latest buy signals (ordered by insertion order)")
            
            conn.close()
            return signals
            
        except Exception as e:
            self.logger.error(f"Error fetching signals: {e}")
            return pd.DataFrame()
    
    def get_sell_signals(self) -> List[str]:
        """Get list of stocks with sell signals"""
        try:
            conn = sqlite3.connect(self.signal_db_path)
            
            query = "SELECT DISTINCT Stock FROM sell_stocks WHERE Status = 'Sell'"
            result = pd.read_sql_query(query, conn)
            conn.close()
            
            return result['Stock'].tolist() if not result.empty else []
            
        except Exception as e:
            self.logger.error(f"Error fetching sell signals: {e}")
            return []
    
    def calculate_position_size(self, stock_price: float, signal_strength: float = 1.0) -> int:
        """Calculate position size based on equal weighting"""
        try:
            # Simple equal weighting approach
            available_cash = self.cash_balance
            target_positions = min(self.max_positions, 50)  # Up to 50 positions
            
            if target_positions == 0:
                return 0
            
            # Base position size
            base_position_value = available_cash / target_positions
            
            # Adjust for signal strength
            position_value = base_position_value * signal_strength
            
            # Apply constraints
            position_value = max(self.min_position_size, 
                               min(self.max_position_size, position_value))
            
            # Calculate shares
            shares = int(position_value / (stock_price * (1 + self.transaction_cost)))
            
            # Check available cash
            total_cost = shares * stock_price * (1 + self.transaction_cost)
            if total_cost > available_cash:
                shares = int(available_cash / (stock_price * (1 + self.transaction_cost)))
            
            return max(0, shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def execute_buy_order(self, stock: str, shares: int, price: float) -> bool:
        """Execute buy order"""
        try:
            total_cost = shares * price * (1 + self.transaction_cost)
            
            if total_cost > self.cash_balance:
                self.logger.warning(f"Insufficient funds for {stock}")
                return False
            
            # Update or create position
            if stock in self.positions:
                current_shares = self.positions[stock]['shares']
                current_cost = self.positions[stock]['total_cost']
                new_shares = current_shares + shares
                new_cost = current_cost + total_cost
                
                self.positions[stock] = {
                    'shares': new_shares,
                    'avg_price': new_cost / new_shares,
                    'total_cost': new_cost,
                    'last_update': datetime.now().isoformat()
                }
            else:
                self.positions[stock] = {
                    'shares': shares,
                    'avg_price': price * (1 + self.transaction_cost),
                    'total_cost': total_cost,
                    'entry_date': datetime.now().isoformat(),
                    'last_update': datetime.now().isoformat()
                }
            
            # Update cash balance
            self.cash_balance -= total_cost
            
            # Record trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'stock': stock,
                'action': 'BUY',
                'shares': shares,
                'price': price,
                'total_cost': total_cost,
                'cash_after': self.cash_balance
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"BUY: {shares} shares of {stock} at {price:.2f} PKR")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            return False
    
    def execute_sell_order(self, stock: str, price: float) -> bool:
        """Execute sell order for entire position"""
        try:
            if stock not in self.positions:
                return False
            
            position = self.positions[stock]
            shares = position['shares']
            avg_cost = position['avg_price']
            
            gross_proceeds = shares * price
            transaction_cost = gross_proceeds * self.transaction_cost
            net_proceeds = gross_proceeds - transaction_cost
            
            # Calculate P&L
            pnl = (price - avg_cost) * shares - transaction_cost
            pnl_percent = (pnl / (avg_cost * shares)) * 100
            
            # Remove position
            del self.positions[stock]
            
            # Update cash
            self.cash_balance += net_proceeds
            
            # Record trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'stock': stock,
                'action': 'SELL',
                'shares': shares,
                'price': price,
                'net_proceeds': net_proceeds,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'cash_after': self.cash_balance
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"SELL: {shares} shares of {stock} at {price:.2f} PKR (P&L: {pnl:,.0f} PKR)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return False
    
    def rebalance_portfolio(self):
        """Main rebalancing logic"""
        try:
            self.logger.info("Starting portfolio rebalancing...")
            
            # Get current signals
            buy_signals = self.get_current_signals()
            sell_signals = self.get_sell_signals()
            
            if buy_signals.empty:
                self.logger.warning("No buy signals available")
                return
            
            # Get target stocks (top signals)
            target_stocks = set(buy_signals['Stock'].tolist())
            current_stocks = set(self.positions.keys())
            
            # Create price lookup
            price_lookup = dict(zip(buy_signals['Stock'], buy_signals['Close']))
            
            # Step 1: Sell positions not in target or with sell signals
            stocks_to_sell = (current_stocks - target_stocks) | (set(sell_signals) & current_stocks)
            
            for stock in stocks_to_sell:
                if stock in self.positions:
                    price = price_lookup.get(stock, self.positions[stock]['avg_price'])
                    self.execute_sell_order(stock, price)
            
            # Step 2: Buy new positions
            stocks_to_buy = target_stocks - current_stocks
            
            for stock in stocks_to_buy:
                if stock in price_lookup:
                    price = price_lookup[stock]
                    
                    # Calculate signal strength based on P&L
                    signal_row = buy_signals[buy_signals['Stock'] == stock].iloc[0]
                    pnl_percent = signal_row['PnL_Percent']
                    signal_strength = min(2.0, max(0.5, pnl_percent / 50))  # Scale 0.5-2.0
                    
                    shares = self.calculate_position_size(price, signal_strength)
                    
                    if shares > 0:
                        self.execute_buy_order(stock, shares, price)
            
            # Save portfolio state
            self.save_portfolio()
            
            self.logger.info("Portfolio rebalancing completed")
            
        except Exception as e:
            self.logger.error(f"Error in rebalancing: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            # Get current prices for valuation
            buy_signals = self.get_current_signals()
            price_lookup = dict(zip(buy_signals['Stock'], buy_signals['Close']))
            
            total_value = self.cash_balance
            position_values = {}
            
            for stock, position in self.positions.items():
                current_price = price_lookup.get(stock, position['avg_price'])
                position_value = position['shares'] * current_price
                total_value += position_value
                
                unrealized_pnl = position_value - position['total_cost']
                unrealized_pnl_pct = (unrealized_pnl / position['total_cost']) * 100
                
                position_values[stock] = {
                    'shares': position['shares'],
                    'avg_price': position['avg_price'],
                    'current_price': current_price,
                    'position_value': position_value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct
                }
            
            # Calculate realized P&L
            realized_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history if trade['action'] == 'SELL')
            
            total_return = total_value - self.portfolio_value
            total_return_pct = (total_return / self.portfolio_value) * 100
            
            return {
                'portfolio_value': total_value,
                'cash_balance': self.cash_balance,
                'invested_amount': total_value - self.cash_balance,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': sum(pos['unrealized_pnl'] for pos in position_values.values()),
                'num_positions': len(self.positions),
                'positions': position_values
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def print_portfolio_report(self):
        """Print comprehensive portfolio report"""
        try:
            summary = self.get_portfolio_summary()
            
            print("\n" + "="*80)
            print("PORTFOLIO MANAGEMENT SYSTEM - REPORT")
            print("="*80)
            
            print(f"\nPORTFOLIO SUMMARY:")
            print(f"Portfolio Value: {summary.get('portfolio_value', 0):,.0f} PKR")
            print(f"Cash Balance: {summary.get('cash_balance', 0):,.0f} PKR")
            print(f"Invested Amount: {summary.get('invested_amount', 0):,.0f} PKR")
            print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"Realized P&L: {summary.get('realized_pnl', 0):,.0f} PKR")
            print(f"Unrealized P&L: {summary.get('unrealized_pnl', 0):,.0f} PKR")
            print(f"Number of Positions: {summary.get('num_positions', 0)}")
            
            # Show top positions
            positions = summary.get('positions', {})
            if positions:
                print(f"\nTOP POSITIONS:")
                sorted_positions = sorted(positions.items(), 
                                        key=lambda x: x[1]['position_value'], 
                                        reverse=True)
                
                for i, (stock, pos) in enumerate(sorted_positions[:10]):
                    print(f"{i+1:2d}. {stock:8s}: {pos['shares']:6,} shares @ {pos['current_price']:8.2f} PKR "
                          f"(Value: {pos['position_value']:10,.0f} PKR, "
                          f"P&L: {pos['unrealized_pnl_pct']:6.2f}%)")
            
            # Show recent trades
            if self.trade_history:
                print(f"\nRECENT TRADES:")
                for trade in self.trade_history[-5:]:
                    action = trade['action']
                    stock = trade['stock']
                    shares = trade['shares']
                    price = trade['price']
                    timestamp = trade['timestamp'][:16]
                    
                    if action == 'SELL':
                        pnl = trade.get('pnl', 0)
                        print(f"{timestamp} - {action:4s} {shares:6,} {stock:8s} @ {price:8.2f} PKR "
                              f"(P&L: {pnl:8,.0f} PKR)")
                    else:
                        total_cost = trade.get('total_cost', 0)
                        print(f"{timestamp} - {action:4s} {shares:6,} {stock:8s} @ {price:8.2f} PKR "
                              f"(Cost: {total_cost:8,.0f} PKR)")
            
            print("\n" + "="*80)
            
        except Exception as e:
            self.logger.error(f"Error printing report: {e}")

def main():
    """Main execution"""
    try:
        # Initialize portfolio manager
        pm = SimplePortfolioManager()
        
        # Run initial rebalancing
        pm.rebalance_portfolio()
        
        # Print report
        pm.print_portfolio_report()
        
        # Ask user for continuous monitoring
        print("\nOptions:")
        print("1. Exit")
        print("2. Run another rebalancing cycle")
        print("3. Show portfolio summary only")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "2":
            pm.rebalance_portfolio()
            pm.print_portfolio_report()
        elif choice == "3":
            pm.print_portfolio_report()
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
