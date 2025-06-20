#!/usr/bin/env python3
"""
Simple verification script to check if current 27 positions match top 50 by P&L
"""

import sqlite3
import json
import os

def verify_positions():
    print("🔍 VERIFYING: Are 27 positions from TOP 50 by P&L?")
    print("="*55)
      # Database path
    project_root = os.path.join('..', '..', '..', '..')
    project_root = os.path.abspath(project_root)
    db_path = os.path.join(project_root, 'src', 'data', 'databases', 'production', 'PSX_investing_Stocks_KMI30.db')
    
    # Current positions
    with open('data/simple_portfolio.json', 'r') as f:
        portfolio = json.load(f)
    
    current_stocks = list(portfolio['positions'].keys())
    print(f"📋 Current Portfolio: {len(current_stocks)} positions")
    print("Current stocks:", ', '.join(sorted(current_stocks)))
    
    # Get top 50 by P&L (same query as system uses)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    SELECT Stock, [% P/L] as PnL_Percent, Close, Signal_Date
    FROM buy_stocks 
    WHERE Status = 'Buy' AND Success = 'Yes'
    ORDER BY [% P/L] DESC
    LIMIT 50
    """
    
    cursor.execute(query)
    top50_data = cursor.fetchall()
    conn.close()
    
    top50_stocks = [row[0] for row in top50_data]
    print(f"📊 Top 50 by P&L: {len(top50_stocks)} signals")
    
    # Show top 10 signals
    print(f"\n🏆 TOP 10 BUY SIGNALS BY P&L:")
    for i, (stock, pnl, price, date) in enumerate(top50_data[:10], 1):
        in_portfolio = "✅" if stock in current_stocks else "❌"
        print(f"{i:2d}. {stock:8s}: {pnl:7.1f}% P&L @ {price:7.2f} PKR {in_portfolio}")
    
    # Analysis
    matching = set(current_stocks) & set(top50_stocks)
    missing = set(top50_stocks) - set(current_stocks)
    extra = set(current_stocks) - set(top50_stocks)
    
    print(f"\n📊 VERIFICATION RESULTS:")
    print(f"✅ Positions in Top 50: {len(matching)}/{len(current_stocks)} ({len(matching)/len(current_stocks)*100:.1f}%)")
    print(f"❌ Missing from portfolio: {len(missing)}")
    print(f"⚠️  Not in Top 50: {len(extra)}")
    
    if len(matching) == len(current_stocks):
        print(f"\n🎉 PERFECT MATCH! All {len(current_stocks)} positions are from TOP 50 by P&L!")
    elif len(matching) >= 25:
        print(f"\n✅ EXCELLENT! {len(matching)} out of 27 positions are from TOP 50!")
    else:
        print(f"\n⚠️  PARTIAL MATCH: {len(matching)} out of 27 positions are from TOP 50")
    
    if missing:
        print(f"\n📋 TOP SIGNALS MISSING FROM PORTFOLIO ({len(missing)}):")
        for i, (stock, pnl, price, date) in enumerate(top50_data, 1):
            if stock in missing and i <= 15:  # Show first 15 missing
                print(f"  {i:2d}. {stock:8s}: {pnl:6.1f}% P&L @ {price:6.2f} PKR")
    
    if extra:
        print(f"\n📋 POSITIONS NOT IN TOP 50 ({len(extra)}):")
        for stock in sorted(extra):
            print(f"  • {stock}")
    
    # Summary
    print(f"\n🎯 CONCLUSION:")
    if len(extra) == 0:
        print("✅ CONFIRMED: All positions are from TOP 50 by P&L performance!")
    else:
        print(f"⚠️  {len(extra)} positions are NOT from TOP 50 by P&L")
        print(f"   This suggests portfolio may not be perfectly aligned with latest signals")

if __name__ == "__main__":
    verify_positions()
