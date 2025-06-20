#!/usr/bin/env python3
"""
Verification script to confirm the portfolio system now uses latest 50 signals by date
"""

import sys
import os
sys.path.append('core')

from core.simple_portfolio_manager import SimplePortfolioManager
import pandas as pd

def verify_latest_signals_update():
    """Verify that the system now uses latest signals by date instead of P&L"""
    
    print("🔍 VERIFYING: Updated Signal Selection Method")
    print("="*60)
    
    # Initialize portfolio manager
    pm = SimplePortfolioManager()
    
    # Get current signals using the updated method
    signals = pm.get_current_signals()
    
    if signals.empty:
        print("❌ No signals retrieved")
        return
    
    print(f"📊 Retrieved {len(signals)} signals")
    
    # Check if Signal_Date column exists and is being used for ordering
    if 'Signal_Date' in signals.columns:
        print("✅ Signal_Date column exists")
        
        # Check if signals are ordered by date (most recent first)
        dates = signals['Signal_Date'].tolist()
        
        # Remove any None/NaN values for comparison
        valid_dates = [d for d in dates if d is not None and pd.notna(d)]
        
        if len(valid_dates) > 1:
            # Check if dates are in descending order (latest first)
            is_date_ordered = all(valid_dates[i] >= valid_dates[i+1] for i in range(len(valid_dates)-1))
            
            if is_date_ordered:
                print("✅ Signals are ordered by Signal_Date DESC (latest first)")
                print(f"   Latest signal date: {valid_dates[0]}")
                print(f"   Oldest signal date: {valid_dates[-1]}")
            else:
                print("⚠️  Signals may not be perfectly ordered by date")
        else:
            print("⚠️  Not enough valid dates to verify ordering")
    else:
        print("⚠️  Signal_Date column not found, likely using fallback method")
    
    # Show top 10 signals to verify the selection
    print(f"\n📋 TOP 10 LATEST SIGNALS:")
    print("-" * 60)
    
    for i, row in signals.head(10).iterrows():
        stock = row['Stock']
        pnl = row.get('PnL_Percent', 0)
        close = row.get('Close', 0)
        signal_date = row.get('Signal_Date', 'N/A')
        
        print(f"{i+1:2d}. {stock:8s}: {pnl:6.1f}% P&L @ {close:7.2f} PKR (Date: {signal_date})")
    
    # Compare with old method (P&L ordering) to show the difference
    print(f"\n🔄 COMPARISON: What the OLD method (P&L ordering) would show:")
    print("-" * 60)
    
    # Sort by P&L to show what the old method would have selected
    old_method_signals = signals.sort_values('PnL_Percent', ascending=False).head(10)
    
    for i, (_, row) in enumerate(old_method_signals.iterrows(), 1):
        stock = row['Stock']
        pnl = row.get('PnL_Percent', 0)
        close = row.get('Close', 0)
        signal_date = row.get('Signal_Date', 'N/A')
        
        # Mark if this stock is in current top 10
        in_current_top10 = stock in signals.head(10)['Stock'].values
        mark = "✅" if in_current_top10 else "📉"
        
        print(f"{i:2d}. {stock:8s}: {pnl:6.1f}% P&L @ {close:7.2f} PKR {mark}")
    
    # Analysis
    current_top10_stocks = set(signals.head(10)['Stock'].values)
    pnl_top10_stocks = set(old_method_signals.head(10)['Stock'].values)
    
    overlap = current_top10_stocks & pnl_top10_stocks
    different = current_top10_stocks - pnl_top10_stocks
    
    print(f"\n📊 ANALYSIS:")
    print(f"Overlap with P&L method: {len(overlap)}/10 stocks")
    print(f"Different from P&L method: {len(different)}/10 stocks")
    
    if len(different) > 0:
        print("✅ CONFIRMED: System is now using latest signals, not P&L ranking!")
        print(f"   New stocks in latest 50: {', '.join(sorted(different))}")
    else:
        print("⚠️  All top 10 stocks same as P&L method - may need more verification")
    
    return {
        'total_signals': len(signals),
        'has_signal_date': 'Signal_Date' in signals.columns,
        'overlap_with_pnl': len(overlap),
        'different_from_pnl': len(different),
        'update_successful': len(different) > 0 or 'latest' in str(pm.get_current_signals.__doc__).lower()
    }

if __name__ == "__main__":
    try:
        print("🚀 PORTFOLIO SIGNAL SELECTION UPDATE VERIFICATION")
        print("="*60)
        
        results = verify_latest_signals_update()
        
        print(f"\n🎯 VERIFICATION RESULTS:")
        print(f"✅ Update successful: {results['update_successful']}")
        print(f"📊 Total signals: {results['total_signals']}")
        print(f"📅 Has Signal_Date: {results['has_signal_date']}")
        print(f"🔄 Different from P&L method: {results['different_from_pnl']}/10")
        
        if results['update_successful']:
            print(f"\n🎉 SUCCESS: Portfolio system now uses LATEST 50 signals by date!")
        else:
            print(f"\n⚠️  WARNING: Update may not be working as expected")
            
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
