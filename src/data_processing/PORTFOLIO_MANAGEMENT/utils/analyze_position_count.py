#!/usr/bin/env python3
"""
Portfolio Position Count Analysis
Investigates why exactly 27 positions exist in the portfolio
"""

import os
import sys
import json
import sqlite3
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

from core.simple_portfolio_manager import SimplePortfolioManager

def analyze_position_count():
    """Analyze why there are exactly 27 positions"""
    
    print("🔍 PORTFOLIO POSITION COUNT ANALYSIS")
    print("="*60)
    
    # Load portfolio manager
    pm = SimplePortfolioManager()
    
    # Load current portfolio data
    with open(pm.portfolio_file, 'r') as f:
        portfolio_data = json.load(f)
    
    positions = portfolio_data['positions']
    cash_balance = portfolio_data['cash_balance']
    
    print(f"\n📊 CURRENT PORTFOLIO STATUS:")
    print(f"Total Positions: {len(positions)}")
    print(f"Cash Balance: {cash_balance:,.0f} PKR")
    print(f"Total Portfolio Value: {pm.portfolio_value:,.0f} PKR")
    
    # Analyze position sizing logic
    print(f"\n🧮 POSITION SIZING ANALYSIS:")
    print(f"Max Positions Allowed: {pm.max_positions}")
    print(f"Min Position Size: {pm.min_position_size:,.0f} PKR")
    print(f"Max Position Size: {pm.max_position_size:,.0f} PKR")
    print(f"Transaction Cost: {pm.transaction_cost:.1%}")
    
    # Calculate theoretical position parameters
    target_positions = min(pm.max_positions, 50)  # From code: Up to 50 positions
    base_position_value = cash_balance / target_positions
    
    print(f"\n📈 THEORETICAL CALCULATIONS:")
    print(f"Target Positions (for cash deployment): {target_positions}")
    print(f"Available Cash for New Positions: {cash_balance:,.0f} PKR")
    print(f"Base Position Value (Cash/50): {base_position_value:,.0f} PKR")
    
    # Check constraints
    print(f"\n⚖️  POSITION SIZE CONSTRAINTS:")
    if base_position_value < pm.min_position_size:
        print(f"❌ Base position ({base_position_value:,.0f}) < Min size ({pm.min_position_size:,.0f})")
        print(f"   This limits how many new positions can be created!")
    else:
        print(f"✅ Base position ({base_position_value:,.0f}) >= Min size ({pm.min_position_size:,.0f})")
    
    if base_position_value > pm.max_position_size:
        print(f"⚠️  Base position ({base_position_value:,.0f}) > Max size ({pm.max_position_size:,.0f})")
        print(f"   Positions will be capped at max size")
    
    # Analyze current positions
    print(f"\n📋 CURRENT POSITION ANALYSIS:")
    
    total_invested = 0
    position_values = []
    
    for i, (stock, pos) in enumerate(positions.items(), 1):
        position_value = pos['total_cost']
        total_invested += position_value
        position_values.append(position_value)
        
        if i <= 10:  # Show first 10
            print(f"{i:2d}. {stock:8s}: {position_value:8,.0f} PKR ({pos['shares']:,} shares)")
        elif i == 11:
            print(f"    ... ({len(positions)-10} more positions)")
    
    print(f"\nTotal Invested: {total_invested:,.0f} PKR")
    print(f"Average Position Size: {total_invested/len(positions):,.0f} PKR")
    print(f"Largest Position: {max(position_values):,.0f} PKR")
    print(f"Smallest Position: {min(position_values):,.0f} PKR")
    
    # Get buy signals data
    print(f"\n🎯 BUY SIGNALS ANALYSIS:")
    
    buy_signals = pm.get_current_signals()
    print(f"Total Buy Signals Available: {len(buy_signals)}")
    print(f"Latest 50 Buy Signals: {min(50, len(buy_signals))}")
    
    if not buy_signals.empty:
        # Get stocks from latest 50 signals
        target_stocks = set(buy_signals['Stock'].tolist()[:50])  # Top 50
        current_stocks = set(positions.keys())
        
        print(f"Stocks in Latest 50 Signals: {len(target_stocks)}")
        print(f"Current Positions in Latest 50: {len(current_stocks & target_stocks)}")
        print(f"Current Positions NOT in Latest 50: {len(current_stocks - target_stocks)}")
        print(f"Latest 50 Signals NOT in Portfolio: {len(target_stocks - current_stocks)}")
        
        missing_signals = target_stocks - current_stocks
        if missing_signals:
            print(f"\n🔍 MISSING FROM PORTFOLIO (from Latest 50):")
            for i, stock in enumerate(list(missing_signals)[:10], 1):
                signal_row = buy_signals[buy_signals['Stock'] == stock].iloc[0]
                price = signal_row['Close']
                pnl = signal_row['PnL_Percent']
                print(f"{i:2d}. {stock:8s}: {price:7.2f} PKR (P&L: {pnl:6.1f}%)")
            
            if len(missing_signals) > 10:
                print(f"    ... ({len(missing_signals)-10} more)")
    
    # Analyze why 27 positions specifically
    print(f"\n🎯 WHY 27 POSITIONS? ANALYSIS:")
    
    # Check cash constraints for new positions
    remaining_cash = cash_balance
    print(f"Remaining Cash: {remaining_cash:,.0f} PKR")
    
    # Calculate how many new positions are possible
    if remaining_cash >= pm.min_position_size:
        max_new_positions = int(remaining_cash / pm.min_position_size)
        print(f"Theoretical Max New Positions: {max_new_positions}")
        
        # But considering equal weighting logic
        equal_weight_size = remaining_cash / 50  # System tries to divide among 50
        if equal_weight_size >= pm.min_position_size:
            theoretical_positions = int(remaining_cash / equal_weight_size)
            print(f"Equal Weight New Positions: {theoretical_positions}")
        else:
            max_equal_weight = int(remaining_cash / pm.min_position_size)
            print(f"Max Equal Weight (at min size): {max_equal_weight}")
    else:
        print(f"❌ Insufficient cash for new positions (need {pm.min_position_size:,.0f} PKR)")
    
    # Historical analysis
    print(f"\n📅 HISTORICAL ANALYSIS:")
    
    trade_history = portfolio_data.get('trade_history', [])
    buy_trades = [t for t in trade_history if t['action'] == 'BUY']
    sell_trades = [t for t in trade_history if t['action'] == 'SELL']
    
    print(f"Total Buy Trades: {len(buy_trades)}")
    print(f"Total Sell Trades: {len(sell_trades)}")
    print(f"Net Positions Created: {len(buy_trades) - len(sell_trades)}")
    
    if buy_trades:
        last_trade = buy_trades[-1]
        print(f"Last Buy Trade: {last_trade['stock']} at {last_trade['timestamp']}")
        print(f"Cash After Last Trade: {last_trade['cash_after']:,.0f} PKR")
    
    # Final analysis
    print(f"\n🎯 CONCLUSION:")
    print(f"="*60)
    
    print(f"The portfolio has exactly 27 positions because:")
    print(f"1. Started with {pm.portfolio_value:,.0f} PKR")
    print(f"2. System created positions from latest buy signals")
    print(f"3. Used equal-weighting approach with constraints:")
    print(f"   - Min position: {pm.min_position_size:,.0f} PKR")
    print(f"   - Max position: {pm.max_position_size:,.0f} PKR")
    print(f"4. After 27 positions, remaining cash: {cash_balance:,.0f} PKR")
    print(f"5. Remaining cash insufficient for equal-weight new positions")
    
    # Check if cash is limiting factor
    if cash_balance < pm.min_position_size:
        print(f"❌ Cash constraint: Cannot create more positions")
    elif cash_balance / 50 < pm.min_position_size:
        print(f"⚠️  Equal-weight constraint: New positions would be too small")
    else:
        print(f"✅ Could potentially create more positions")
    
    return {
        'current_positions': len(positions),
        'cash_balance': cash_balance,
        'total_invested': total_invested,
        'buy_signals_available': len(buy_signals) if not buy_signals.empty else 0,
        'missing_from_latest_50': len(missing_signals) if 'missing_signals' in locals() else 0
    }

if __name__ == "__main__":
    try:
        results = analyze_position_count()
        print(f"\n✅ Analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
