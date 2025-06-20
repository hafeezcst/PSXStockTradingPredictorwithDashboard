#!/usr/bin/env python3
"""
Verify if Current Positions are from Latest 50 Buy Signals
Compares current portfolio positions with latest 50 buy signals from database
"""

import os
import sys
import json
import sqlite3
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

def verify_positions_vs_latest50():
    """Verify if current positions match latest 50 buy signals"""
    
    print("🔍 VERIFYING: Are Current Positions from Latest 50 Buy Signals?")
    print("="*70)
    
    # Get database path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    project_root = os.path.abspath(project_root)
    db_path = os.path.join(project_root, "data", "databases", "production", "PSX_investing_Stocks_KMI100.db")
    
    print(f"📊 Database: {os.path.basename(db_path)}")
    
    # Load current portfolio positions
    portfolio_file = os.path.join(current_dir, 'data', 'simple_portfolio.json')
    with open(portfolio_file, 'r') as f:
        portfolio_data = json.load(f)
    
    current_positions = set(portfolio_data['positions'].keys())
    print(f"📋 Current Portfolio Positions: {len(current_positions)}")
    
    # Connect to database and get latest 50 buy signals
    conn = sqlite3.connect(db_path)
    
    # First, let's check what ordering method is used
    print(f"\n🔍 CHECKING SIGNAL ORDERING METHODS:")
    
    # Method 1: Order by % P/L (Performance-based)
    query_pnl = """
    SELECT Stock, [% P/L] as PnL_Percent, Signal_Date, Close, RSI_Weekly_Avg
    FROM buy_stocks 
    WHERE Status = 'Buy' AND Success = 'Yes'
    ORDER BY [% P/L] DESC
    LIMIT 50
    """
    
    pnl_signals = pd.read_sql_query(query_pnl, conn)
    pnl_stocks = set(pnl_signals['Stock'].tolist())
    
    print(f"📈 Top 50 by P&L Performance: {len(pnl_stocks)} stocks")
    
    # Method 2: Order by Signal Date (Latest/Most Recent)
    query_date = """
    SELECT Stock, [% P/L] as PnL_Percent, Signal_Date, Close, RSI_Weekly_Avg
    FROM buy_stocks 
    WHERE Status = 'Buy' AND Success = 'Yes'
    ORDER BY Signal_Date DESC
    LIMIT 50
    """
    
    try:
        date_signals = pd.read_sql_query(query_date, conn)
        date_stocks = set(date_signals['Stock'].tolist())
        print(f"📅 Latest 50 by Signal Date: {len(date_stocks)} stocks")
    except Exception as e:
        print(f"⚠️  Cannot order by Signal_Date: {e}")
        date_stocks = set()
        date_signals = pd.DataFrame()
    
    conn.close()
    
    # Compare current positions with both methods
    print(f"\n📊 COMPARISON ANALYSIS:")
    print(f"="*50)
    
    # Check match with P&L-based top 50
    pnl_match = current_positions & pnl_stocks
    pnl_missing_from_portfolio = pnl_stocks - current_positions
    pnl_extra_in_portfolio = current_positions - pnl_stocks
    
    print(f"🎯 MATCH WITH TOP 50 BY P&L:")
    print(f"✅ Positions matching: {len(pnl_match)}/{len(current_positions)} ({len(pnl_match)/len(current_positions)*100:.1f}%)")
    print(f"❌ Missing from portfolio: {len(pnl_missing_from_portfolio)}")
    print(f"⚠️  Extra in portfolio: {len(pnl_extra_in_portfolio)}")
    
    if len(pnl_missing_from_portfolio) > 0:
        print(f"\n📋 TOP P&L SIGNALS MISSING FROM PORTFOLIO:")
        missing_pnl_df = pnl_signals[pnl_signals['Stock'].isin(pnl_missing_from_portfolio)].head(10)
        for _, row in missing_pnl_df.iterrows():
            print(f"  • {row['Stock']:8s}: {row['PnL_Percent']:7.1f}% P&L @ {row['Close']:7.2f} PKR")
    
    # Check match with date-based latest 50 (if available)
    if not date_signals.empty:
        date_match = current_positions & date_stocks
        date_missing_from_portfolio = date_stocks - current_positions
        date_extra_in_portfolio = current_positions - date_stocks
        
        print(f"\n🎯 MATCH WITH LATEST 50 BY DATE:")
        print(f"✅ Positions matching: {len(date_match)}/{len(current_positions)} ({len(date_match)/len(current_positions)*100:.1f}%)")
        print(f"❌ Missing from portfolio: {len(date_missing_from_portfolio)}")
        print(f"⚠️  Extra in portfolio: {len(date_extra_in_portfolio)}")
        
        if len(date_missing_from_portfolio) > 0:
            print(f"\n📋 LATEST DATE SIGNALS MISSING FROM PORTFOLIO:")
            missing_date_df = date_signals[date_signals['Stock'].isin(date_missing_from_portfolio)].head(10)
            for _, row in missing_date_df.iterrows():
                print(f"  • {row['Stock']:8s}: {row['PnL_Percent']:7.1f}% P&L @ {row['Close']:7.2f} PKR (Date: {row['Signal_Date']})")
    
    # Show current positions that are in the analysis
    print(f"\n📋 CURRENT PORTFOLIO POSITIONS ANALYSIS:")
    print(f"="*50)
    
    portfolio_in_pnl = []
    portfolio_not_in_pnl = []
    
    for stock in current_positions:
        if stock in pnl_stocks:
            # Find the stock in pnl_signals
            stock_data = pnl_signals[pnl_signals['Stock'] == stock]
            if not stock_data.empty:
                pnl_val = stock_data.iloc[0]['PnL_Percent']
                price = stock_data.iloc[0]['Close']
                portfolio_in_pnl.append((stock, pnl_val, price))
        else:
            portfolio_not_in_pnl.append(stock)
    
    print(f"✅ POSITIONS IN TOP 50 BY P&L ({len(portfolio_in_pnl)}):")
    for stock, pnl, price in sorted(portfolio_in_pnl, key=lambda x: x[1], reverse=True)[:15]:
        print(f"  • {stock:8s}: {pnl:7.1f}% P&L @ {price:7.2f} PKR")
    
    if len(portfolio_in_pnl) > 15:
        print(f"    ... and {len(portfolio_in_pnl)-15} more")
    
    if portfolio_not_in_pnl:
        print(f"\n❌ POSITIONS NOT IN TOP 50 BY P&L ({len(portfolio_not_in_pnl)}):")
        for stock in sorted(portfolio_not_in_pnl)[:10]:
            print(f"  • {stock}")
        if len(portfolio_not_in_pnl) > 10:
            print(f"    ... and {len(portfolio_not_in_pnl)-10} more")
    
    # Final analysis
    print(f"\n🎯 FINAL ANALYSIS:")
    print(f"="*50)
    
    if len(pnl_match) == len(current_positions):
        print(f"✅ PERFECT MATCH: All {len(current_positions)} positions are from top 50 by P&L!")
    elif len(pnl_match) >= len(current_positions) * 0.9:  # 90%+ match
        print(f"✅ EXCELLENT MATCH: {len(pnl_match)}/{len(current_positions)} positions are from top 50 by P&L!")
    elif len(pnl_match) >= len(current_positions) * 0.8:  # 80%+ match
        print(f"⚠️  GOOD MATCH: {len(pnl_match)}/{len(current_positions)} positions are from top 50 by P&L")
    else:
        print(f"❌ POOR MATCH: Only {len(pnl_match)}/{len(current_positions)} positions are from top 50 by P&L")
    
    # Check if system is using latest 50 by date instead
    if not date_signals.empty and len(date_match) > len(pnl_match):
        print(f"💡 NOTE: Better match with latest 50 by date ({len(date_match)} vs {len(pnl_match)})")
        print(f"   System might be using date-based ordering instead of P&L-based")
    
    return {
        'current_positions_count': len(current_positions),
        'pnl_top50_match': len(pnl_match),
        'pnl_match_percentage': len(pnl_match)/len(current_positions)*100,
        'date_top50_match': len(date_match) if not date_signals.empty else 0,
        'date_match_percentage': len(date_match)/len(current_positions)*100 if not date_signals.empty else 0,
        'missing_from_pnl_top50': len(pnl_missing_from_portfolio),
        'extra_not_in_pnl_top50': len(pnl_extra_in_portfolio)
    }

if __name__ == "__main__":
    try:
        results = verify_positions_vs_latest50()
        
        print(f"\n📊 SUMMARY RESULTS:")
        print(f"Current Positions: {results['current_positions_count']}")
        print(f"Match with P&L Top 50: {results['pnl_top50_match']} ({results['pnl_match_percentage']:.1f}%)")
        if results['date_top50_match'] > 0:
            print(f"Match with Date Top 50: {results['date_top50_match']} ({results['date_match_percentage']:.1f}%)")
        
        print(f"\n✅ Verification completed!")
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
