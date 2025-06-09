"""
Test script for signal_analysis.py SQL queries
"""

import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Get the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))

def test_queries():
    """Test the SQL queries used in signal_analysis.py"""
    db_path = os.path.join(project_root, "data", "databases", "production", "PSX_investing_Stocks_KMI30.db")
    
    print(f"Testing queries on database: {db_path}")
    print(f"Database exists: {os.path.exists(db_path)}")
    
    if not os.path.exists(db_path):
        print("Database file not found!")
        return
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        print("\n1. Testing Signal Overview query...")
        query1 = """
        WITH all_signals AS (
            SELECT
                Stock as symbol, 
                'BUY' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                COUNT(*) as count
            FROM buy_stocks
            GROUP BY Stock
            
            UNION ALL
            
            SELECT
                Stock as symbol,
                'SELL' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                COUNT(*) as count
            FROM sell_stocks
            GROUP BY Stock
            
            UNION ALL
            
            SELECT
                Stock as symbol,
                'NEUTRAL' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                COUNT(*) as count
            FROM neutral_stocks
            GROUP BY Stock
        )
        SELECT
            signal,
            SUM(count) as count,
            AVG(COALESCE(confidence, 0)) as avg_confidence,
            AVG(COALESCE(score, 0)) as avg_score,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM all_signals
        GROUP BY signal
        ORDER BY count DESC
        """
        
        df1 = pd.read_sql_query(query1, conn)
        print("Result:")
        print(df1)
        
        print("\n2. Testing Recent Signal Changes query...")
        query2 = """
        SELECT 
            Stock as symbol,
            transition_date as date,
            Current_Signal as new_signal,
            Previous_Signal as previous_signal,
            Current_Close as current_close,
            Previous_Close as previous_close,
            Profit_Loss_Pct as profit_loss,
            Days_In_Signal as days_in_signal,
            Notes as reasons,
            CASE 
                WHEN Profit_Loss_Pct > 0 THEN 'Increasing'
                WHEN Profit_Loss_Pct < 0 THEN 'Decreasing'
                ELSE 'Stable'
            END as confidence_trend
        FROM signal_transition_history
        ORDER BY transition_date DESC
        LIMIT 10
        """
        
        df2 = pd.read_sql_query(query2, conn)
        print("Result:")
        print(df2)
        
        print("\n3. Testing Signal History query...")
        selected_symbol = "OGDC"  # Example symbol
        query3 = f"""
        WITH all_signals AS (
            -- Buy signals
            SELECT
                Stock as symbol,
                Date as date,
                'BUY' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                'Identified as BUY based on RSI and MA indicators' as reasons
            FROM buy_stocks
            WHERE Stock = '{selected_symbol}'
            
            UNION ALL
            
            -- Sell signals
            SELECT
                Stock as symbol,
                Date as date,
                'SELL' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                'Identified as SELL based on RSI and MA indicators' as reasons
            FROM sell_stocks
            WHERE Stock = '{selected_symbol}'
            
            UNION ALL
            
            -- Neutral signals
            SELECT
                Stock as symbol,
                Date as date,
                'NEUTRAL' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                'Identified as NEUTRAL based on RSI and MA indicators' as reasons
            FROM neutral_stocks
            WHERE Stock = '{selected_symbol}'
        ),
        signal_history AS (
            SELECT 
                date,
                signal,
                COALESCE(confidence, 0) as confidence,
                COALESCE(score, 0) as score,
                reasons,
                LAG(signal) OVER (ORDER BY date) as prev_signal,
                LAG(COALESCE(confidence, 0)) OVER (ORDER BY date) as prev_confidence,
                LAG(COALESCE(score, 0)) OVER (ORDER BY date) as prev_score
            FROM all_signals
        )
        SELECT 
            *,
            CASE 
                WHEN confidence > prev_confidence THEN 'Increasing'
                WHEN confidence < prev_confidence THEN 'Decreasing'
                ELSE 'Stable'
            END as confidence_trend,
            CASE 
                WHEN score > prev_score THEN 'Increasing'
                WHEN score < prev_score THEN 'Decreasing'
                ELSE 'Stable'
            END as score_trend
        FROM signal_history
        ORDER BY date DESC
        LIMIT 10
        """
        
        df3 = pd.read_sql_query(query3, conn)
        print("Result:")
        print(df3)
        
        print("\n4. Testing Signal Analysis query...")
        query4 = """
        WITH all_signals AS (
            -- Buy signals
            SELECT
                Stock as symbol,
                Date as date,
                'BUY' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                'Identified as BUY based on RSI and MA indicators' as reasons
            FROM buy_stocks
            
            UNION ALL
            
            -- Sell signals
            SELECT
                Stock as symbol,
                Date as date,
                'SELL' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                'Identified as SELL based on RSI and MA indicators' as reasons
            FROM sell_stocks
            
            UNION ALL
            
            -- Neutral signals
            SELECT
                Stock as symbol,
                Date as date,
                'NEUTRAL' as signal,
                RSI_Weekly_Avg as confidence,
                MA_30 as score,
                'Identified as NEUTRAL based on RSI and MA indicators' as reasons
            FROM neutral_stocks
        )
        
        SELECT 
            symbol,
            date,
            signal,
            COALESCE(confidence, 0) as confidence,
            COALESCE(score, 0) as score,
            reasons
        FROM all_signals
        ORDER BY date DESC
        LIMIT 10
        """
        
        df4 = pd.read_sql_query(query4, conn)
        print("Result:")
        print(df4)
        
        print("\n5. Testing Performance Metrics query...")
        query5 = """
        SELECT 
            Stock as symbol,
            transition_date as date,
            Current_Signal as signal,
            Current_Close as confidence,
            COALESCE(Profit_Loss_Pct, 0) as score,
            Previous_Signal as prev_signal,
            Previous_Close as prev_confidence,
            0 as prev_score
        FROM signal_transition_history
        ORDER BY transition_date DESC
        LIMIT 10
        """
        
        df5 = pd.read_sql_query(query5, conn)
        print("Result:")
        print(df5)
        
        conn.close()
        print("\nAll queries tested successfully!")
    
    except Exception as e:
        print(f"Error testing queries: {str(e)}")

if __name__ == "__main__":
    test_queries()
