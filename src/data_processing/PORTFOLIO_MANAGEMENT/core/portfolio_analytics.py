#!/usr/bin/env python3
"""
Portfolio Analytics Module
Advanced analytics, performance metrics, and reporting for the portfolio management system
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalytics:
    def __init__(self, portfolio_file: str = "data/portfolio_data.json",
                 signal_db_path: str = r"data\databases\production\PSX_investing_Stocks_KMI100.db"):
        self.portfolio_file = portfolio_file
        self.signal_db_path = signal_db_path
        self.portfolio_data = self.load_portfolio_data()
        
    def load_portfolio_data(self) -> Dict:
        """Load portfolio data from file"""
        try:
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading portfolio data: {e}")
            return {}
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Convert trade history to DataFrame for analysis"""
        try:
            trades = self.portfolio_data.get('trade_history', [])
            if not trades:
                return pd.DataFrame()
            
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            return df
        except Exception as e:
            print(f"Error creating trade history DataFrame: {e}")
            return pd.DataFrame()
    
    def calculate_daily_returns(self) -> pd.DataFrame:
        """Calculate daily portfolio returns"""
        try:
            trades_df = self.get_trade_history_df()
            if trades_df.empty:
                return pd.DataFrame()
            
            # Group trades by date and calculate daily P&L
            daily_pnl = trades_df[trades_df['action'] == 'SELL'].groupby('date')['pnl'].sum()
            
            # Calculate cumulative returns
            cumulative_pnl = daily_pnl.cumsum()
            initial_capital = self.portfolio_data.get('initial_capital', 35_000_000)
            
            daily_returns = pd.DataFrame({
                'date': daily_pnl.index,
                'daily_pnl': daily_pnl.values,
                'cumulative_pnl': cumulative_pnl.values,
                'portfolio_value': initial_capital + cumulative_pnl.values,
                'daily_return_pct': (daily_pnl.values / initial_capital) * 100,
                'cumulative_return_pct': (cumulative_pnl.values / initial_capital) * 100
            })
            
            return daily_returns
        except Exception as e:
            print(f"Error calculating daily returns: {e}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            daily_returns = self.calculate_daily_returns()
            trades_df = self.get_trade_history_df()
            
            if daily_returns.empty or trades_df.empty:
                return {}
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[(trades_df['action'] == 'SELL') & (trades_df['pnl'] > 0)])
            losing_trades = len(trades_df[(trades_df['action'] == 'SELL') & (trades_df['pnl'] <= 0)])
            
            win_rate = (winning_trades / (winning_trades + losing_trades)) * 100 if (winning_trades + losing_trades) > 0 else 0
            
            # P&L metrics
            total_pnl = trades_df[trades_df['action'] == 'SELL']['pnl'].sum()
            avg_win = trades_df[(trades_df['action'] == 'SELL') & (trades_df['pnl'] > 0)]['pnl'].mean()
            avg_loss = trades_df[(trades_df['action'] == 'SELL') & (trades_df['pnl'] <= 0)]['pnl'].mean()
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Return metrics
            daily_returns_pct = daily_returns['daily_return_pct']
            total_return_pct = daily_returns['cumulative_return_pct'].iloc[-1] if not daily_returns.empty else 0
            
            # Risk metrics
            volatility = daily_returns_pct.std() * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = (daily_returns_pct.mean() * 252) / volatility if volatility > 0 else 0
            
            # Drawdown calculation
            portfolio_values = daily_returns['portfolio_value']
            rolling_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
            
            # Trading frequency
            trading_days = (daily_returns['date'].max() - daily_returns['date'].min()).days
            trades_per_month = (total_trades / max(trading_days, 1)) * 30
            
            return {
                'total_return_percent': total_return_pct,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'trades_per_month': trades_per_month,
                'trading_period_days': trading_days
            }
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {}
    
    def analyze_position_performance(self) -> pd.DataFrame:
        """Analyze performance of individual positions"""
        try:
            trades_df = self.get_trade_history_df()
            if trades_df.empty:
                return pd.DataFrame()
            
            # Group trades by stock to analyze position performance
            sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
            
            position_stats = sell_trades.groupby('stock').agg({
                'pnl': ['sum', 'mean', 'count'],
                'pnl_percent': ['mean', 'std'],
                'shares': 'sum',
                'price': 'mean'
            }).round(2)
            
            # Flatten column names
            position_stats.columns = ['total_pnl', 'avg_pnl', 'trade_count', 
                                    'avg_pnl_percent', 'pnl_volatility', 
                                    'total_shares', 'avg_price']
            
            # Calculate win rate per stock
            win_rates = sell_trades.groupby('stock')['pnl'].apply(
                lambda x: (x > 0).sum() / len(x) * 100
            ).round(2)
            position_stats['win_rate'] = win_rates
            
            # Sort by total P&L
            position_stats = position_stats.sort_values('total_pnl', ascending=False)
            
            return position_stats
            
        except Exception as e:
            print(f"Error analyzing position performance: {e}")
            return pd.DataFrame()
    
    def analyze_signal_effectiveness(self) -> Dict:
        """Analyze the effectiveness of trading signals"""
        try:
            # Get current signals
            conn = sqlite3.connect(self.signal_db_path)
            
            # Analyze buy signals that were acted upon
            trades_df = self.get_trade_history_df()
            if trades_df.empty:
                return {}
            
            buy_trades = trades_df[trades_df['action'] == 'BUY']['stock'].unique()
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            # Calculate signal-to-outcome mapping
            signal_effectiveness = {}
            
            for stock in buy_trades:
                stock_sells = sell_trades[sell_trades['stock'] == stock]
                if not stock_sells.empty:
                    avg_return = stock_sells['pnl_percent'].mean()
                    win_rate = (stock_sells['pnl'] > 0).sum() / len(stock_sells) * 100
                    total_pnl = stock_sells['pnl'].sum()
                    
                    signal_effectiveness[stock] = {
                        'average_return_percent': avg_return,
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'trades_count': len(stock_sells)
                    }
            
            # Overall signal effectiveness
            if signal_effectiveness:
                overall_avg_return = np.mean([s['average_return_percent'] for s in signal_effectiveness.values()])
                overall_win_rate = np.mean([s['win_rate'] for s in signal_effectiveness.values()])
                total_signal_pnl = sum([s['total_pnl'] for s in signal_effectiveness.values()])
                
                return {
                    'individual_stocks': signal_effectiveness,
                    'overall_average_return': overall_avg_return,
                    'overall_win_rate': overall_win_rate,
                    'total_signal_pnl': total_signal_pnl,
                    'stocks_traded': len(signal_effectiveness)
                }
            
            conn.close()
            return {}
            
        except Exception as e:
            print(f"Error analyzing signal effectiveness: {e}")
            return {}
    
    def generate_risk_analysis(self) -> Dict:
        """Generate comprehensive risk analysis"""
        try:
            daily_returns = self.calculate_daily_returns()
            current_positions = self.portfolio_data.get('positions', {})
            
            if daily_returns.empty:
                return {}
            
            # Value at Risk (VaR) calculation
            daily_returns_pct = daily_returns['daily_return_pct']
            var_95 = np.percentile(daily_returns_pct, 5)  # 95% VaR
            var_99 = np.percentile(daily_returns_pct, 1)  # 99% VaR
            
            # Expected Shortfall (Conditional VaR)
            es_95 = daily_returns_pct[daily_returns_pct <= var_95].mean()
            es_99 = daily_returns_pct[daily_returns_pct <= var_99].mean()
            
            # Position concentration risk
            if current_positions:
                total_position_value = sum(pos['shares'] * pos['avg_price'] for pos in current_positions.values())
                position_weights = {
                    stock: (pos['shares'] * pos['avg_price'] / total_position_value) * 100
                    for stock, pos in current_positions.items()
                }
                
                # Calculate Herfindahl-Hirschman Index for concentration
                hhi = sum(weight**2 for weight in position_weights.values())
                
                concentration_risk = {
                    'position_weights': position_weights,
                    'max_position_weight': max(position_weights.values()) if position_weights else 0,
                    'top_5_concentration': sum(sorted(position_weights.values(), reverse=True)[:5]),
                    'herfindahl_index': hhi,
                    'concentration_level': 'High' if hhi > 2500 else 'Medium' if hhi > 1500 else 'Low'
                }
            else:
                concentration_risk = {}
            
            # Volatility analysis
            volatility_metrics = {
                'daily_volatility': daily_returns_pct.std(),
                'annualized_volatility': daily_returns_pct.std() * np.sqrt(252),
                'volatility_trend': 'Increasing' if daily_returns_pct.rolling(10).std().iloc[-1] > daily_returns_pct.rolling(30).std().iloc[-1] else 'Decreasing'
            }
            
            return {
                'value_at_risk': {
                    'var_95_percent': var_95,
                    'var_99_percent': var_99,
                    'expected_shortfall_95': es_95,
                    'expected_shortfall_99': es_99
                },
                'concentration_risk': concentration_risk,
                'volatility_metrics': volatility_metrics,
                'risk_assessment': self.assess_overall_risk(var_95, hhi if current_positions else 0, volatility_metrics['annualized_volatility'])
            }
            
        except Exception as e:
            print(f"Error generating risk analysis: {e}")
            return {}
    
    def assess_overall_risk(self, var_95: float, hhi: float, volatility: float) -> str:
        """Assess overall portfolio risk level"""
        risk_score = 0
        
        # VaR component (higher negative VaR = higher risk)
        if var_95 < -3:
            risk_score += 3
        elif var_95 < -2:
            risk_score += 2
        elif var_95 < -1:
            risk_score += 1
        
        # Concentration component
        if hhi > 2500:
            risk_score += 3
        elif hhi > 1500:
            risk_score += 2
        elif hhi > 1000:
            risk_score += 1
        
        # Volatility component
        if volatility > 30:
            risk_score += 3
        elif volatility > 20:
            risk_score += 2
        elif volatility > 15:
            risk_score += 1
        
        if risk_score >= 7:
            return "High Risk"
        elif risk_score >= 4:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def create_performance_dashboard(self) -> Dict:
        """Create a comprehensive performance dashboard"""
        try:
            dashboard_data = {
                'generated_at': datetime.now().isoformat(),
                'performance_metrics': self.calculate_performance_metrics(),
                'position_analysis': self.analyze_position_performance().to_dict('index'),
                'signal_effectiveness': self.analyze_signal_effectiveness(),
                'risk_analysis': self.generate_risk_analysis(),
                'current_portfolio_status': self.get_current_portfolio_status()
            }
            
            return dashboard_data
            
        except Exception as e:
            print(f"Error creating performance dashboard: {e}")
            return {}
    
    def get_current_portfolio_status(self) -> Dict:
        """Get current portfolio status and positions"""
        try:
            performance_metrics = self.portfolio_data.get('performance_metrics', {})
            positions = self.portfolio_data.get('positions', {})
            
            return {
                'portfolio_value': performance_metrics.get('current_portfolio_value', 0),
                'cash_balance': performance_metrics.get('cash_balance', 0),
                'number_of_positions': len(positions),
                'invested_amount': performance_metrics.get('invested_amount', 0),
                'total_return_percent': performance_metrics.get('total_return_percent', 0),
                'last_updated': performance_metrics.get('last_updated', ''),
                'position_summary': {
                    stock: {
                        'shares': pos['shares'],
                        'avg_price': pos['avg_price'],
                        'entry_date': pos['entry_date']
                    }
                    for stock, pos in positions.items()
                }
            }
            
        except Exception as e:
            print(f"Error getting current portfolio status: {e}")
            return {}
    
    def export_analytics_report(self, output_format: str = 'json') -> str:
        """Export comprehensive analytics report"""
        try:
            dashboard_data = self.create_performance_dashboard()
            
            if not dashboard_data:
                return ""
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if output_format.lower() == 'json':
                filename = f"data/reports/portfolio_analytics_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(dashboard_data, f, indent=2, default=str)
            
            elif output_format.lower() == 'excel':
                filename = f"data/reports/portfolio_analytics_{timestamp}.xlsx"
                
                # Create multiple sheets
                with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                    # Performance metrics
                    if dashboard_data.get('performance_metrics'):
                        pd.DataFrame([dashboard_data['performance_metrics']]).to_excel(
                            writer, sheet_name='Performance_Metrics', index=False
                        )
                    
                    # Position analysis
                    if dashboard_data.get('position_analysis'):
                        pd.DataFrame(dashboard_data['position_analysis']).T.to_excel(
                            writer, sheet_name='Position_Analysis'
                        )
                    
                    # Risk analysis
                    if dashboard_data.get('risk_analysis'):
                        risk_df = pd.DataFrame([dashboard_data['risk_analysis']])
                        risk_df.to_excel(writer, sheet_name='Risk_Analysis', index=False)
            
            print(f"Analytics report exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error exporting analytics report: {e}")
            return ""

def main():
    """Main function to run analytics"""
    try:
        analytics = PortfolioAnalytics()
        
        print("Generating Portfolio Analytics Dashboard...")
        dashboard = analytics.create_performance_dashboard()
        
        if dashboard:
            print("\n" + "="*80)
            print("PORTFOLIO ANALYTICS DASHBOARD")
            print("="*80)
            
            # Performance Summary
            perf = dashboard.get('performance_metrics', {})
            print(f"\nPERFORMANCE SUMMARY:")
            print(f"Total Return: {perf.get('total_return_percent', 0):.2f}%")
            print(f"Win Rate: {perf.get('win_rate', 0):.2f}%")
            print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}%")
            print(f"Total Trades: {perf.get('total_trades', 0)}")
            
            # Risk Analysis
            risk = dashboard.get('risk_analysis', {})
            if risk:
                print(f"\nRISK ANALYSIS:")
                print(f"Overall Risk Level: {risk.get('risk_assessment', 'N/A')}")
                print(f"95% VaR: {risk.get('value_at_risk', {}).get('var_95_percent', 0):.2f}%")
                print(f"Portfolio Volatility: {risk.get('volatility_metrics', {}).get('annualized_volatility', 0):.2f}%")
            
            # Top Performing Positions
            positions = dashboard.get('position_analysis', {})
            if positions:
                print(f"\nTOP PERFORMING POSITIONS:")
                sorted_positions = sorted(positions.items(), key=lambda x: x[1].get('total_pnl', 0), reverse=True)
                for i, (stock, data) in enumerate(sorted_positions[:5]):
                    print(f"{i+1}. {stock}: {data.get('total_pnl', 0):,.0f} PKR ({data.get('avg_pnl_percent', 0):.2f}%)")
            
            # Export report
            filename = analytics.export_analytics_report('json')
            if filename:
                print(f"\nDetailed report saved to: {filename}")
            
        else:
            print("No analytics data available. Please ensure portfolio data exists.")
            
    except Exception as e:
        print(f"Error in analytics main: {e}")

if __name__ == "__main__":
    main()
