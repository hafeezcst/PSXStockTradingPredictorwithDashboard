#!/usr/bin/env python3
"""
Robust Portfolio Management System
Integrates with existing signal database for automated trading decisions
Portfolio Value: 35 Million PKR
Strategy: Top 50 Buy Signals with Dynamic Rebalancing
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from risk_manager import RiskManager
from portfolio_config import ALL_CONFIGS

class PortfolioManager:
    def __init__(self, 
                 portfolio_value: float = None,
                 signal_db_path: str = None,
                 portfolio_file: str = "data/portfolio_data.json",
                 config: Dict = None):
        
        # Load configuration
        self.config = config or ALL_CONFIGS
        
        # Initialize from config
        self.portfolio_value = portfolio_value or self.config['portfolio']['initial_capital']
        self.signal_db_path = signal_db_path or self.config['signals']['signal_db_path']
        self.portfolio_file = portfolio_file
        self.max_positions = self.config['portfolio']['max_positions']
        self.min_position_size = self.config['portfolio']['min_position_size']
        self.max_position_size = self.config['portfolio']['max_position_size']
        self.transaction_cost = self.config['portfolio']['transaction_cost']
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize portfolio data
        self.portfolio = self.load_portfolio()
        self.cash_balance = self.portfolio.get('cash_balance', self.portfolio_value)
        self.positions = self.portfolio.get('positions', {})
        self.trade_history = self.portfolio.get('trade_history', [])
        self.performance_metrics = self.portfolio.get('performance_metrics', {})
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            portfolio_value=self.portfolio_value,
            positions=self.positions,
            signal_db_path=self.signal_db_path
        )
        
        # Signal tracking
        self.current_signals = {}
        self.last_signal_update = None
        
        self.logger.info(f"Portfolio Manager initialized with {self.portfolio_value:,.0f} PKR")
        self.logger.info(f"Risk management enabled with {len(self.config)} configuration modules")
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/portfolio_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PortfolioManager')
        
    def load_portfolio(self) -> Dict:
        """Load existing portfolio data or create new one"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    portfolio = json.load(f)
                self.logger.info("Loaded existing portfolio data")
                return portfolio
            else:
                portfolio = {
                    'cash_balance': self.portfolio_value,
                    'positions': {},
                    'trade_history': [],
                    'performance_metrics': {},
                    'created_date': datetime.now().isoformat(),
                    'last_update': datetime.now().isoformat()
                }
                self.save_portfolio(portfolio)
                self.logger.info("Created new portfolio")
                return portfolio
        except Exception as e:
            self.logger.error(f"Error loading portfolio: {e}")
            return {}
    
    def save_portfolio(self, portfolio_data: Dict = None):
        """Save portfolio data to file"""
        try:
            if portfolio_data is None:
                portfolio_data = {
                    'cash_balance': self.cash_balance,
                    'positions': self.positions,
                    'trade_history': self.trade_history,
                    'performance_metrics': self.performance_metrics,
                    'last_update': datetime.now().isoformat()
                }
            
            # Create backup
            if os.path.exists(self.portfolio_file):
                backup_file = f"{self.portfolio_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.portfolio_file, backup_file)
            
            # Save current data
            os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)
            with open(self.portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=2, default=str)
            
            self.logger.info("Portfolio data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio: {e}")
    
    def get_current_signals(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch current buy, sell, and neutral signals from database"""
        try:
            conn = sqlite3.connect(self.signal_db_path)
            
            # Get buy signals - top 50 based on performance and signal strength
            buy_query = """
            SELECT Stock, Close, Volume, RSI_Weekly_Avg, RSI_3Months_Avg_Recent, 
                   AO_Weekly, MA_30, Multibagger, FreeFloatRatio, Success, 
                   [% P/L] as PnL_Percent, Signal_Date, Signal_Close, Holding_Days,
                   Status, Update_Date, Date
            FROM buy_stocks 
            WHERE Status = 'Buy' 
            ORDER BY [% P/L] DESC, RSI_Weekly_Avg DESC
            LIMIT 50
            """
            
            buy_signals = pd.read_sql_query(buy_query, conn)
            
            # Get sell signals
            sell_query = """
            SELECT Stock, Close, Volume, RSI_Weekly_Avg, RSI_3Months_Avg_Recent,
                   AO_Weekly, MA_30, Multibagger, FreeFloatRatio, Success,
                   [% P/L] as PnL_Percent, Signal_Date, Signal_Close, Holding_Days,
                   Status, Update_Date, Date
            FROM sell_stocks 
            WHERE Status = 'Sell'
            ORDER BY Update_Date DESC
            """
            
            sell_signals = pd.read_sql_query(sell_query, conn)
            
            # Get neutral signals
            neutral_query = """
            SELECT Stock, Close, Volume, RSI_Weekly_Avg, RSI_3Months_Avg_Recent,
                   AO_Weekly, MA_30, Multibagger, FreeFloatRatio, Trend_Direction,
                   Status, Update_Date, Date
            FROM neutral_stocks 
            WHERE Status = 'Neutral'
            ORDER BY Update_Date DESC
            """
            
            neutral_signals = pd.read_sql_query(neutral_query, conn)
            
            conn.close()
            
            self.last_signal_update = datetime.now()
            self.logger.info(f"Retrieved {len(buy_signals)} buy signals, {len(sell_signals)} sell signals, {len(neutral_signals)} neutral signals")
            
            return buy_signals, sell_signals, neutral_signals
            
        except Exception as e:
            self.logger.error(f"Error fetching signals: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def calculate_position_size(self, stock_price: float, signal_strength: float = 1.0) -> int:
        """Calculate optimal position size based on price and signal strength"""
        try:
            # Base position size (equal weight initially)
            available_cash = self.cash_balance
            target_positions = min(self.max_positions, len(self.get_target_stocks()))
            
            if target_positions == 0:
                return 0
            
            base_position_value = available_cash / target_positions
            
            # Adjust for signal strength (0.5 to 1.5 multiplier)
            adjusted_position_value = base_position_value * (0.5 + signal_strength)
            
            # Apply constraints
            position_value = max(self.min_position_size, 
                               min(self.max_position_size, adjusted_position_value))
            
            # Calculate shares (rounded down to avoid overallocation)            shares = int(position_value / stock_price)
            
            # Ensure we don't exceed available cash
            total_cost = shares * stock_price * (1 + self.transaction_cost)
            if total_cost > available_cash:
                shares = int(available_cash / (stock_price * (1 + self.transaction_cost)))
            
            return max(0, shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_target_stocks(self) -> List[str]:
        """Get list of stocks that should be in portfolio based on current signals"""
        try:
            buy_signals, _, _ = self.get_current_signals()
            return buy_signals['Stock'].tolist()
        except Exception as e:
            self.logger.error(f"Error getting target stocks: {e}")
            return []
    
    def execute_buy_order(self, stock: str, shares: int, price: float, signal_data: Dict = None) -> bool:
        """Execute a buy order with integrated risk management"""
        try:
            # Validate with risk manager
            signal_data = signal_data or {}
            validation_result = self.risk_manager.validate_new_position(
                stock=stock, 
                shares=shares, 
                price=price, 
                signal_data=signal_data
            )
            
            is_valid, reason, adjusted_shares = validation_result
            
            if not is_valid and adjusted_shares == 0:
                self.logger.warning(f"Buy order rejected for {stock}: {reason}")
                return False
            
            # Use adjusted shares if provided
            if adjusted_shares != shares:
                self.logger.info(f"Position size adjusted for {stock}: {shares} -> {adjusted_shares} shares ({reason})")
                shares = adjusted_shares
            
            total_cost = shares * price * (1 + self.transaction_cost)
            
            if total_cost > self.cash_balance:
                self.logger.warning(f"Insufficient funds for {stock}: need {total_cost:,.0f}, have {self.cash_balance:,.0f}")
                return False
            
            # Update portfolio
            if stock in self.positions:
                # Add to existing position
                current_shares = self.positions[stock]['shares']
                current_cost = self.positions[stock]['total_cost']
                new_total_shares = current_shares + shares
                new_total_cost = current_cost + total_cost
                
                self.positions[stock] = {
                    'shares': new_total_shares,
                    'avg_price': new_total_cost / new_total_shares,
                    'total_cost': new_total_cost,
                    'last_update': datetime.now().isoformat(),
                    'entry_date': self.positions[stock]['entry_date']
                }
            else:
                # New position
                self.positions[stock] = {
                    'shares': shares,
                    'avg_price': price * (1 + self.transaction_cost),
                    'total_cost': total_cost,
                    'entry_date': datetime.now().isoformat(),
                    'last_update': datetime.now().isoformat()
                }
            
            # Update cash balance
            self.cash_balance -= total_cost
            
            # Update risk manager with new positions
            self.risk_manager.positions = self.positions
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'stock': stock,
                'action': 'BUY',
                'shares': shares,
                'price': price,
                'total_cost': total_cost,
                'transaction_cost': total_cost - (shares * price),
                'cash_balance_after': self.cash_balance,
                'risk_validation': reason,
                'signal_data': signal_data
            }
            self.trade_history.append(trade_record)
            
            self.logger.info(f"BUY: {shares} shares of {stock} at {price:.2f} PKR (Total: {total_cost:,.0f} PKR)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing buy order for {stock}: {e}")
            return False
      def execute_sell_order(self, stock: str, shares: int, price: float, reason: str = "signal") -> bool:
        """Execute a sell order with integrated risk management"""
        try:
            # Validate with risk manager
            validation_result = self.risk_manager.validate_exit_decision(stock, shares, reason)
            is_valid, validation_reason = validation_result
            
            if not is_valid:
                self.logger.warning(f"Sell order rejected for {stock}: {validation_reason}")
                return False
            
            if stock not in self.positions:
                self.logger.warning(f"Cannot sell {stock}: not in portfolio")
                return False
            
            available_shares = self.positions[stock]['shares']
            if shares > available_shares:
                self.logger.warning(f"Cannot sell {shares} shares of {stock}: only {available_shares} available")
                shares = available_shares
            
            gross_proceeds = shares * price
            transaction_cost = gross_proceeds * self.transaction_cost
            net_proceeds = gross_proceeds - transaction_cost
            
            # Calculate P&L
            avg_cost = self.positions[stock]['avg_price']
            pnl = (price - avg_cost) * shares - transaction_cost
            pnl_percent = (pnl / (avg_cost * shares)) * 100
            
            # Update position
            if shares == available_shares:
                # Sell entire position
                del self.positions[stock]
            else:
                # Partial sale
                remaining_shares = available_shares - shares
                total_cost_remaining = remaining_shares * avg_cost
                self.positions[stock] = {
                    'shares': remaining_shares,
                    'avg_price': avg_cost,
                    'total_cost': total_cost_remaining,
                    'entry_date': self.positions[stock]['entry_date'],
                    'last_update': datetime.now().isoformat()
                }
            
            # Update cash balance
            self.cash_balance += net_proceeds
            
            # Update risk manager with new positions
            self.risk_manager.positions = self.positions
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'stock': stock,
                'action': 'SELL',
                'shares': shares,
                'price': price,
                'gross_proceeds': gross_proceeds,
                'net_proceeds': net_proceeds,
                'transaction_cost': transaction_cost,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'cash_balance_after': self.cash_balance,
                'exit_reason': reason,
                'risk_validation': validation_reason
            }
            self.trade_history.append(trade_record)
            
            self.logger.info(f"SELL: {shares} shares of {stock} at {price:.2f} PKR (P&L: {pnl:,.0f} PKR, {pnl_percent:.2f}%)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing sell order for {stock}: {e}")
            return False
    
    def rebalance_portfolio(self):
        """Main portfolio rebalancing logic"""
        try:
            self.logger.info("Starting portfolio rebalancing...")
            
            # Get current signals
            buy_signals, sell_signals, neutral_signals = self.get_current_signals()
            
            if buy_signals.empty:
                self.logger.warning("No buy signals available")
                return
            
            # Get current target stocks (top 50 buy signals)
            target_stocks = set(buy_signals['Stock'].tolist())
            current_stocks = set(self.positions.keys())
            
            # Create price lookup from signals
            price_lookup = {}
            for _, row in buy_signals.iterrows():
                price_lookup[row['Stock']] = row['Close']
            
            # Add sell signal prices
            for _, row in sell_signals.iterrows():
                price_lookup[row['Stock']] = row['Close']
            
            # Step 1: Sell positions that are no longer in top 50 or have sell signals
            stocks_to_sell = current_stocks - target_stocks
            sell_signal_stocks = set(sell_signals['Stock'].tolist())
            stocks_to_sell.update(sell_signal_stocks & current_stocks)
              for stock in stocks_to_sell:
                if stock in self.positions:
                    shares = self.positions[stock]['shares']
                    price = price_lookup.get(stock, self.positions[stock]['avg_price'])
                    reason = "top_50_removal" if stock not in target_stocks else "sell_signal"
                    self.execute_sell_order(stock, shares, price, reason)
            
            # Step 2: Buy new positions or add to existing ones
            stocks_to_buy = target_stocks - current_stocks
            
            # Calculate signal strength for position sizing
            signal_strength_lookup = {}
            for _, row in buy_signals.iterrows():
                stock = row['Stock']
                # Signal strength based on P&L percentage and RSI
                strength = min(2.0, max(0.1, (row['PnL_Percent'] / 100 + row['RSI_Weekly_Avg'] / 100) / 2))
                signal_strength_lookup[stock] = strength
            
            # Execute buy orders
            for stock in stocks_to_buy:
                if stock in price_lookup:
                    price = price_lookup[stock]
                    signal_strength = signal_strength_lookup.get(stock, 1.0)
                    shares = self.calculate_position_size(price, signal_strength)
                          if shares > 0:
                    # Get signal data for this stock
                    stock_signal = buy_signals[buy_signals['Stock'] == stock].iloc[0].to_dict()
                    self.execute_buy_order(stock, shares, price, stock_signal)
            
            # Step 3: Rebalance existing positions if needed
            self.rebalance_existing_positions(buy_signals, signal_strength_lookup, price_lookup)
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Save portfolio
            self.save_portfolio()
            
            self.logger.info("Portfolio rebalancing completed")
            
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {e}")
    
    def rebalance_existing_positions(self, buy_signals: pd.DataFrame, 
                                   signal_strength_lookup: Dict, price_lookup: Dict):
        """Rebalance existing positions based on updated signal strengths"""
        try:
            target_stocks = set(buy_signals['Stock'].tolist())
            
            for stock in list(self.positions.keys()):
                if stock in target_stocks and stock in price_lookup:
                    current_value = self.positions[stock]['shares'] * price_lookup[stock]
                    signal_strength = signal_strength_lookup.get(stock, 1.0)
                    
                    # Calculate target position size
                    target_shares = self.calculate_position_size(price_lookup[stock], signal_strength)
                    target_value = target_shares * price_lookup[stock]
                    current_shares = self.positions[stock]['shares']
                    
                    # Rebalance if difference is significant (>20%)
                    value_diff_percent = abs(target_value - current_value) / current_value
                    
                    if value_diff_percent > 0.2:  # 20% threshold                        if target_shares > current_shares:
                            # Buy more shares
                            additional_shares = target_shares - current_shares
                            stock_signal = buy_signals[buy_signals['Stock'] == stock].iloc[0].to_dict()
                            self.execute_buy_order(stock, additional_shares, price_lookup[stock], stock_signal)
                        elif target_shares < current_shares:
                            # Sell some shares
                            shares_to_sell = current_shares - target_shares
                            self.execute_sell_order(stock, shares_to_sell, price_lookup[stock], "rebalance")
            
        except Exception as e:
            self.logger.error(f"Error rebalancing existing positions: {e}")
    
    def update_performance_metrics(self):
        """Calculate and update portfolio performance metrics"""
        try:
            # Get current prices for portfolio valuation
            buy_signals, sell_signals, neutral_signals = self.get_current_signals()
            all_signals = pd.concat([buy_signals, sell_signals, neutral_signals], ignore_index=True)
            price_lookup = dict(zip(all_signals['Stock'], all_signals['Close']))
            
            # Calculate current portfolio value
            portfolio_value = self.cash_balance
            position_values = {}
            
            for stock, position in self.positions.items():
                if stock in price_lookup:
                    current_price = price_lookup[stock]
                    position_value = position['shares'] * current_price
                    portfolio_value += position_value
                    
                    # Calculate unrealized P&L
                    cost_basis = position['shares'] * position['avg_price']
                    unrealized_pnl = position_value - cost_basis
                    unrealized_pnl_percent = (unrealized_pnl / cost_basis) * 100
                    
                    position_values[stock] = {
                        'shares': position['shares'],
                        'current_price': current_price,
                        'position_value': position_value,
                        'cost_basis': cost_basis,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_percent': unrealized_pnl_percent
                    }
            
            # Calculate realized P&L from trade history
            realized_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history if trade['action'] == 'SELL')
            
            # Calculate total return
            total_return = portfolio_value - self.portfolio_value
            total_return_percent = (total_return / self.portfolio_value) * 100
            
            # Update metrics
            self.performance_metrics = {
                'current_portfolio_value': portfolio_value,
                'cash_balance': self.cash_balance,
                'invested_amount': portfolio_value - self.cash_balance,
                'initial_capital': self.portfolio_value,
                'total_return': total_return,
                'total_return_percent': total_return_percent,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': sum(pos.get('unrealized_pnl', 0) for pos in position_values.values()),
                'number_of_positions': len(self.positions),
                'position_details': position_values,
                'last_updated': datetime.now().isoformat()
            }
            
            self.logger.info(f"Portfolio Value: {portfolio_value:,.0f} PKR, Return: {total_return_percent:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive portfolio report"""
        try:
            self.update_performance_metrics()
            
            report = {
                'report_date': datetime.now().isoformat(),
                'portfolio_summary': self.performance_metrics,
                'positions': self.positions,
                'recent_trades': self.trade_history[-10:],  # Last 10 trades
                'risk_metrics': self.calculate_risk_metrics(),
                'signal_analysis': self.analyze_current_signals()
            }
            
            # Save report
            report_file = f"data/reports/portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Portfolio report generated: {report_file}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return {}
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate portfolio risk metrics"""
        try:
            if not self.positions:
                return {}
            
            # Position concentration
            total_invested = sum(pos['shares'] * pos['avg_price'] for pos in self.positions.values())
            position_weights = {}
            
            for stock, position in self.positions.items():
                position_value = position['shares'] * position['avg_price']
                weight = (position_value / total_invested) * 100
                position_weights[stock] = weight
            
            # Calculate concentration metrics
            max_position_weight = max(position_weights.values()) if position_weights else 0
            top_5_concentration = sum(sorted(position_weights.values(), reverse=True)[:5])
            
            return {
                'max_position_weight': max_position_weight,
                'top_5_concentration': top_5_concentration,
                'number_of_positions': len(self.positions),
                'cash_percentage': (self.cash_balance / (self.cash_balance + total_invested)) * 100,
                'position_weights': position_weights
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def analyze_current_signals(self) -> Dict:
        """Analyze current signal distribution and quality"""
        try:
            buy_signals, sell_signals, neutral_signals = self.get_current_signals()
            
            return {
                'buy_signals_count': len(buy_signals),
                'sell_signals_count': len(sell_signals),
                'neutral_signals_count': len(neutral_signals),
                'avg_buy_signal_performance': buy_signals['PnL_Percent'].mean() if not buy_signals.empty else 0,
                'top_buy_signals': buy_signals.head(10)[['Stock', 'Close', 'PnL_Percent', 'RSI_Weekly_Avg']].to_dict('records') if not buy_signals.empty else [],
                'last_signal_update': self.last_signal_update.isoformat() if self.last_signal_update else None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing signals: {e}")
            return {}
    
    def run_continuous_monitoring(self, check_interval: int = 300):  # 5 minutes
        """Run continuous portfolio monitoring and rebalancing"""
        self.logger.info("Starting continuous portfolio monitoring...")
        
        try:
            while True:
                self.logger.info("Checking for portfolio updates...")
                
                # Check if signals have been updated
                self.rebalance_portfolio()
                
                # Generate report every hour
                current_time = datetime.now()
                if current_time.minute == 0:  # Top of the hour
                    self.generate_report()
                
                # Wait for next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Continuous monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error in continuous monitoring: {e}")

def main():
    """Main execution function"""
    try:
        # Initialize portfolio manager
        pm = PortfolioManager()
        
        # Run initial rebalancing
        pm.rebalance_portfolio()
        
        # Generate initial report
        report = pm.generate_report()
        
        print("\n" + "="*80)
        print("PORTFOLIO MANAGEMENT SYSTEM - INITIAL REPORT")
        print("="*80)
        
        if 'portfolio_summary' in report:
            summary = report['portfolio_summary']
            print(f"Portfolio Value: {summary.get('current_portfolio_value', 0):,.0f} PKR")
            print(f"Cash Balance: {summary.get('cash_balance', 0):,.0f} PKR")
            print(f"Invested Amount: {summary.get('invested_amount', 0):,.0f} PKR")
            print(f"Total Return: {summary.get('total_return_percent', 0):.2f}%")
            print(f"Number of Positions: {summary.get('number_of_positions', 0)}")
        
        print("\nCurrent Positions:")
        for stock, details in report.get('portfolio_summary', {}).get('position_details', {}).items():
            print(f"  {stock}: {details['shares']} shares @ {details['current_price']:.2f} PKR "
                  f"(Value: {details['position_value']:,.0f} PKR, P&L: {details['unrealized_pnl_percent']:.2f}%)")
        
        print("\nRecent Trades:")
        for trade in report.get('recent_trades', [])[-5:]:
            print(f"  {trade['timestamp'][:19]} - {trade['action']} {trade['shares']} {trade['stock']} @ {trade['price']:.2f}")
        
        print("\n" + "="*80)
        
        # Ask user if they want to start continuous monitoring
        response = input("\nStart continuous monitoring? (y/n): ").lower().strip()
        if response == 'y':
            pm.run_continuous_monitoring()
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
