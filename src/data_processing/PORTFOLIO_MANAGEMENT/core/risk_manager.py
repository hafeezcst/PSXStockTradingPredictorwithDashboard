#!/usr/bin/env python3
"""
Advanced Risk Management Module
Implements comprehensive risk controls, position sizing, and portfolio protection
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from portfolio_config import RISK_CONFIG, PORTFOLIO_CONFIG, TRADING_RULES

class RiskManager:
    def __init__(self, portfolio_value: float, positions: Dict, 
                 signal_db_path: str = r"data\databases\production\PSX_investing_Stocks_KMI100.db"):
        self.portfolio_value = portfolio_value
        self.positions = positions
        self.signal_db_path = signal_db_path
        self.risk_config = RISK_CONFIG
        self.portfolio_config = PORTFOLIO_CONFIG
        self.trading_rules = TRADING_RULES
        
        # Setup logging
        self.logger = logging.getLogger('RiskManager')
        
        # Risk tracking
        self.daily_pnl = 0
        self.risk_violations = []
        self.position_limits = {}
        
    def validate_new_position(self, stock: str, shares: int, price: float, 
                            signal_data: Dict) -> Tuple[bool, str, int]:
        """
        Validate if a new position meets all risk criteria
        Returns: (is_valid, reason, adjusted_shares)
        """
        try:
            position_value = shares * price * (1 + self.portfolio_config['transaction_cost'])
            
            # 1. Check maximum single position limit
            portfolio_value = self.calculate_current_portfolio_value()
            position_weight = position_value / portfolio_value
            
            if position_weight > self.risk_config['max_single_position']:
                max_allowed_value = portfolio_value * self.risk_config['max_single_position']
                adjusted_shares = int(max_allowed_value / (price * (1 + self.portfolio_config['transaction_cost'])))
                return False, f"Position exceeds {self.risk_config['max_single_position']*100:.1f}% limit", adjusted_shares
            
            # 2. Check minimum position size
            if position_value < self.portfolio_config['min_position_size']:
                return False, f"Position below minimum size of {self.portfolio_config['min_position_size']:,.0f} PKR", 0
            
            # 3. Check maximum position size
            if position_value > self.portfolio_config['max_position_size']:
                adjusted_shares = int(self.portfolio_config['max_position_size'] / (price * (1 + self.portfolio_config['transaction_cost'])))
                return False, f"Position exceeds maximum size, adjusted to {adjusted_shares} shares", adjusted_shares
            
            # 4. Check cash reserve requirement
            if not self.check_cash_reserve(position_value):
                return False, "Insufficient cash reserve", 0
            
            # 5. Check sector concentration (if sector data available)
            sector_risk = self.check_sector_concentration(stock, position_value)
            if not sector_risk[0]:
                return False, sector_risk[1], 0
            
            # 6. Check signal quality
            signal_quality = self.validate_signal_quality(signal_data)
            if not signal_quality[0]:
                return False, signal_quality[1], 0
            
            # 7. Check volume and liquidity
            liquidity_check = self.check_liquidity(stock, shares, signal_data)
            if not liquidity_check[0]:
                return False, liquidity_check[1], 0
            
            # 8. Check correlation risk (if multiple positions in similar stocks)
            correlation_risk = self.check_correlation_risk(stock)
            if not correlation_risk[0]:
                self.logger.warning(f"Correlation risk warning for {stock}: {correlation_risk[1]}")
            
            return True, "Position approved", shares
            
        except Exception as e:
            self.logger.error(f"Error validating position for {stock}: {e}")
            return False, f"Validation error: {e}", 0
    
    def validate_exit_decision(self, stock: str, shares: int, reason: str) -> Tuple[bool, str]:
        """Validate if an exit decision should be executed"""
        try:
            # 1. Check if position exists
            if stock not in self.positions:
                return False, f"No position in {stock} to exit"
            
            # 2. Check if shares to sell don't exceed position
            available_shares = self.positions[stock]['shares']
            if shares > available_shares:
                return False, f"Cannot sell {shares} shares, only {available_shares} available"
            
            # 3. Check stop-loss conditions
            if reason == "stop_loss":
                current_price = self.get_current_price(stock)
                if current_price:
                    position_pnl_pct = ((current_price - self.positions[stock]['avg_price']) / self.positions[stock]['avg_price']) * 100
                    
                    # Only allow stop-loss if actually losing money
                    if position_pnl_pct > 0:
                        return False, "Cannot stop-loss a profitable position"
            
            # 4. Check market hours (if configured)
            if self.trading_rules['market_hours_only']:
                if not self.is_market_open():
                    return False, "Market is closed"
            
            # 5. Check daily trade limit
            if not self.check_daily_trade_limit():
                return False, "Daily trade limit exceeded"
            
            return True, "Exit approved"
            
        except Exception as e:
            self.logger.error(f"Error validating exit for {stock}: {e}")
            return False, f"Validation error: {e}"
    
    def calculate_position_size_kelly(self, stock: str, signal_data: Dict) -> int:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            # Get historical performance data for this signal type
            win_rate = signal_data.get('historical_win_rate', 0.6)  # Default 60%
            avg_win = signal_data.get('avg_win_percent', 0.15)     # Default 15%
            avg_loss = signal_data.get('avg_loss_percent', 0.08)   # Default 8%
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss <= 0:
                return 0
            
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly fraction to prevent over-leverage
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25% of portfolio
            
            # Calculate position size
            portfolio_value = self.calculate_current_portfolio_value()
            position_value = portfolio_value * kelly_fraction
            
            # Apply minimum and maximum constraints
            position_value = max(self.portfolio_config['min_position_size'], 
                               min(self.portfolio_config['max_position_size'], position_value))
            
            current_price = signal_data.get('Close', signal_data.get('price', 0))
            if current_price > 0:
                shares = int(position_value / (current_price * (1 + self.portfolio_config['transaction_cost'])))
                return shares
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {e}")
            return 0
    
    def check_cash_reserve(self, required_amount: float) -> bool:
        """Check if sufficient cash reserve is maintained"""
        try:
            current_cash = self.get_current_cash_balance()
            portfolio_value = self.calculate_current_portfolio_value()
            min_cash_required = portfolio_value * self.risk_config['cash_reserve_minimum']
            
            available_cash = current_cash - required_amount
            
            return available_cash >= min_cash_required
            
        except Exception as e:
            self.logger.error(f"Error checking cash reserve: {e}")
            return False
    
    def check_sector_concentration(self, stock: str, position_value: float) -> Tuple[bool, str]:
        """Check if adding position would violate sector concentration limits"""
        try:
            # This would require sector classification data
            # For now, implement basic check - can be enhanced with sector data
            
            portfolio_value = self.calculate_current_portfolio_value()
            position_weight = position_value / portfolio_value
            
            # If single position exceeds sector limit, flag it
            if position_weight > self.risk_config['max_sector_exposure']:
                return False, f"Position would exceed sector exposure limit of {self.risk_config['max_sector_exposure']*100:.1f}%"
            
            return True, "Sector concentration acceptable"
            
        except Exception as e:
            self.logger.error(f"Error checking sector concentration: {e}")
            return True, "Could not verify sector concentration"
    
    def validate_signal_quality(self, signal_data: Dict) -> Tuple[bool, str]:
        """Validate the quality of the trading signal"""
        try:
            # 1. Check RSI levels
            rsi_weekly = signal_data.get('RSI_Weekly_Avg', 0)
            if rsi_weekly < 40:
                return False, f"RSI too low: {rsi_weekly:.1f}"
            
            # 2. Check signal age
            signal_date = signal_data.get('Signal_Date', '')
            if signal_date:
                signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
                days_old = (datetime.now() - signal_dt).days
                
                if days_old > 30:  # Signal older than 30 days
                    return False, f"Signal too old: {days_old} days"
            
            # 3. Check success rate
            success = signal_data.get('Success', '')
            if success == 'No':
                return False, "Signal marked as unsuccessful"
            
            # 4. Check P&L history
            pnl_percent = signal_data.get('PnL_Percent', 0)
            if pnl_percent < 0:
                return False, f"Negative historical P&L: {pnl_percent:.2f}%"
            
            return True, "Signal quality acceptable"
            
        except Exception as e:
            self.logger.error(f"Error validating signal quality: {e}")
            return True, "Could not validate signal quality"
    
    def check_liquidity(self, stock: str, shares: int, signal_data: Dict) -> Tuple[bool, str]:
        """Check if the stock has sufficient liquidity for the trade"""
        try:
            volume = signal_data.get('Volume', 0)
            trade_volume = shares
            
            # Check if trade volume is reasonable compared to average volume
            if volume > 0 and trade_volume > volume * 0.1:  # Don't trade more than 10% of daily volume
                return False, f"Trade volume too large compared to daily volume"
            
            # Check minimum volume requirement
            if volume < 10000:  # Minimum 10K shares daily volume
                return False, f"Insufficient liquidity: {volume:,.0f} daily volume"
            
            return True, "Liquidity sufficient"
            
        except Exception as e:
            self.logger.error(f"Error checking liquidity: {e}")
            return True, "Could not verify liquidity"
    
    def check_correlation_risk(self, stock: str) -> Tuple[bool, str]:
        """Check for correlation risk with existing positions"""
        try:
            # This would require correlation analysis between stocks
            # For now, implement basic sector-based check
            
            # Count positions in similar companies (basic name matching)
            similar_stocks = []
            for existing_stock in self.positions.keys():
                # Basic similarity check (could be enhanced with proper sector/industry data)
                if len(set(stock.lower()) & set(existing_stock.lower())) > 2:
                    similar_stocks.append(existing_stock)
            
            if len(similar_stocks) > 2:
                return False, f"High correlation risk with: {', '.join(similar_stocks)}"
            
            return True, "Correlation risk acceptable"
            
        except Exception as e:
            self.logger.error(f"Error checking correlation risk: {e}")
            return True, "Could not assess correlation risk"
    
    def monitor_daily_risk_limits(self) -> Dict[str, bool]:
        """Monitor if daily risk limits are being adhered to"""
        try:
            risk_status = {}
            
            # 1. Daily loss limit
            daily_pnl_percent = (self.daily_pnl / self.portfolio_value) * 100
            risk_status['daily_loss_limit'] = daily_pnl_percent >= self.risk_config['daily_loss_limit']
            
            # 2. Position concentration
            portfolio_value = self.calculate_current_portfolio_value()
            max_position_weight = 0
            
            for stock, position in self.positions.items():
                position_value = position['shares'] * self.get_current_price(stock, position['avg_price'])
                weight = (position_value / portfolio_value) * 100
                max_position_weight = max(max_position_weight, weight)
            
            risk_status['position_concentration'] = max_position_weight <= self.risk_config['max_single_position'] * 100
            
            # 3. Cash reserve
            current_cash = self.get_current_cash_balance()
            cash_percentage = (current_cash / portfolio_value) * 100
            risk_status['cash_reserve'] = cash_percentage >= self.risk_config['cash_reserve_minimum'] * 100
            
            # 4. Number of positions
            risk_status['position_count'] = len(self.positions) <= self.portfolio_config['max_positions']
            
            return risk_status
            
        except Exception as e:
            self.logger.error(f"Error monitoring daily risk limits: {e}")
            return {}
    
    def calculate_var_and_expected_shortfall(self, confidence_level: float = 0.95) -> Dict:
        """Calculate Value at Risk and Expected Shortfall"""
        try:
            # This would require historical returns data
            # For now, implement basic VaR calculation based on position values
            
            portfolio_value = self.calculate_current_portfolio_value()
            position_values = []
            
            for stock, position in self.positions.items():
                current_price = self.get_current_price(stock, position['avg_price'])
                position_value = position['shares'] * current_price
                position_values.append(position_value)
            
            if not position_values:
                return {}
            
            # Simple VaR calculation (would be enhanced with proper historical data)
            position_weights = np.array(position_values) / portfolio_value
            assumed_volatilities = np.random.normal(0.02, 0.01, len(position_values))  # Placeholder
            
            portfolio_volatility = np.sqrt(np.dot(position_weights, assumed_volatilities**2))
            
            # VaR calculation
            var_multiplier = 1.645 if confidence_level == 0.95 else 2.326  # For 95% or 99%
            var_amount = portfolio_value * portfolio_volatility * var_multiplier
            
            return {
                'var_amount': var_amount,
                'var_percentage': (var_amount / portfolio_value) * 100,
                'portfolio_volatility': portfolio_volatility * 100,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return {}
    
    def get_current_price(self, stock: str, fallback_price: float = 0) -> float:
        """Get current price for a stock from the signal database"""
        try:
            conn = sqlite3.connect(self.signal_db_path)
            
            # Try to get from buy_stocks first
            query = "SELECT Close FROM buy_stocks WHERE Stock = ? ORDER BY Date DESC LIMIT 1"
            result = conn.execute(query, (stock,)).fetchone()
            
            if result:
                conn.close()
                return result[0]
            
            # Try sell_stocks
            query = "SELECT Close FROM sell_stocks WHERE Stock = ? ORDER BY Date DESC LIMIT 1"
            result = conn.execute(query, (stock,)).fetchone()
            
            if result:
                conn.close()
                return result[0]
            
            # Try neutral_stocks
            query = "SELECT Close FROM neutral_stocks WHERE Stock = ? ORDER BY Date DESC LIMIT 1"
            result = conn.execute(query, (stock,)).fetchone()
            
            conn.close()
            
            if result:
                return result[0]
            else:
                return fallback_price
                
        except Exception as e:
            self.logger.error(f"Error getting current price for {stock}: {e}")
            return fallback_price
    
    def calculate_current_portfolio_value(self) -> float:
        """Calculate current total portfolio value"""
        try:
            total_value = self.get_current_cash_balance()
            
            for stock, position in self.positions.items():
                current_price = self.get_current_price(stock, position['avg_price'])
                position_value = position['shares'] * current_price
                total_value += position_value
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.portfolio_value
    
    def get_current_cash_balance(self) -> float:
        """Get current cash balance - placeholder implementation"""
        # This would be retrieved from the portfolio manager
        return self.portfolio_value * 0.1  # Placeholder: assume 10% cash
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        try:
            now = datetime.now()
            current_time = now.time()
            current_day = now.strftime('%A').lower()
            
            # Check if it's a trading day
            if current_day in [day.lower() for day in self.trading_rules['no_trade_days']]:
                return False
            
            # Check trading hours
            market_open = datetime.strptime(self.trading_rules['market_open_time'], '%H:%M').time()
            market_close = datetime.strptime(self.trading_rules['market_close_time'], '%H:%M').time()
            
            return market_open <= current_time <= market_close
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            return True  # Default to allow trading if check fails
    
    def check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit has been exceeded"""
        try:
            # This would require tracking daily trades
            # Placeholder implementation
            return True  # Allow trading by default
            
        except Exception as e:
            self.logger.error(f"Error checking daily trade limit: {e}")
            return True
    
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        try:
            risk_report = {
                'timestamp': datetime.now().isoformat(),
                'daily_risk_status': self.monitor_daily_risk_limits(),
                'var_analysis': self.calculate_var_and_expected_shortfall(),
                'portfolio_metrics': {
                    'total_value': self.calculate_current_portfolio_value(),
                    'cash_balance': self.get_current_cash_balance(),
                    'number_of_positions': len(self.positions),
                    'largest_position_value': self.get_largest_position_value(),
                    'portfolio_concentration': self.calculate_portfolio_concentration()
                },
                'risk_violations': self.risk_violations,
                'recommendations': self.generate_risk_recommendations()
            }
            
            return risk_report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {}
    
    def get_largest_position_value(self) -> float:
        """Get the value of the largest position"""
        try:
            max_value = 0
            for stock, position in self.positions.items():
                current_price = self.get_current_price(stock, position['avg_price'])
                position_value = position['shares'] * current_price
                max_value = max(max_value, position_value)
            return max_value
        except:
            return 0
    
    def calculate_portfolio_concentration(self) -> float:
        """Calculate Herfindahl-Hirschman Index for portfolio concentration"""
        try:
            portfolio_value = self.calculate_current_portfolio_value()
            if portfolio_value == 0:
                return 0
            
            hhi = 0
            for stock, position in self.positions.items():
                current_price = self.get_current_price(stock, position['avg_price'])
                position_value = position['shares'] * current_price
                weight = position_value / portfolio_value
                hhi += weight ** 2
            
            return hhi * 10000  # Scale to 0-10000 range
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio concentration: {e}")
            return 0
    
    def generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        try:
            recommendations = []
            
            # Check risk status
            risk_status = self.monitor_daily_risk_limits()
            
            if not risk_status.get('daily_loss_limit', True):
                recommendations.append("Daily loss limit exceeded - consider reducing positions")
            
            if not risk_status.get('position_concentration', True):
                recommendations.append("Position concentration too high - diversify portfolio")
            
            if not risk_status.get('cash_reserve', True):
                recommendations.append("Cash reserve below minimum - consider selling some positions")
            
            if len(self.positions) > self.portfolio_config['max_positions'] * 0.9:
                recommendations.append("Approaching maximum position limit - be selective with new positions")
            
            # Portfolio concentration check
            concentration = self.calculate_portfolio_concentration()
            if concentration > 2500:
                recommendations.append("High portfolio concentration detected - consider diversification")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []

def main():
    """Test the risk manager"""
    try:
        # Example usage
        test_positions = {
            'AGIL': {'shares': 1000, 'avg_price': 137.0, 'entry_date': '2024-11-21'},
            'AGP': {'shares': 500, 'avg_price': 187.01, 'entry_date': '2023-10-25'}
        }
        
        risk_manager = RiskManager(35_000_000, test_positions)
        
        # Generate risk report
        risk_report = risk_manager.generate_risk_report()
        
        print("Risk Management Report:")
        print("=" * 50)
        
        for key, value in risk_report.items():
            if key != 'recommendations':
                print(f"{key}: {value}")
        
        print("\nRecommendations:")
        for rec in risk_report.get('recommendations', []):
            print(f"- {rec}")
            
    except Exception as e:
        print(f"Error in risk manager test: {e}")

if __name__ == "__main__":
    main()
