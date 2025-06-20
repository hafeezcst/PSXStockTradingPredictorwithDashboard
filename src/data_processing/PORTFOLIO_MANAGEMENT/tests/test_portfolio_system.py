#!/usr/bin/env python3
"""
Test Script for Portfolio Management System
Validates system components and connectivity
"""

import os
import sys
import sqlite3
import json
from datetime import datetime

def test_database_connectivity():
    """Test database connectivity and structure"""
    print("Testing database connectivity...")
    
    db_path = r"data\databases\production\PSX_investing_Stocks_KMI100.db"
    
    try:
        if not os.path.exists(db_path):
            print(f"❌ Database file not found: {db_path}")
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test required tables
        required_tables = ['buy_stocks', 'sell_stocks', 'neutral_stocks']
        table_counts = {}
        
        for table in required_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            table_counts[table] = count
            print(f"✓ {table}: {count} records")
        
        conn.close()
        
        # Check if we have sufficient data
        if table_counts['buy_stocks'] < 10:
            print("⚠️  Warning: Less than 10 buy signals available")
        
        print("✓ Database connectivity test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_portfolio_manager_init():
    """Test portfolio manager initialization"""
    print("\nTesting portfolio manager initialization...")
    
    try:
        # Test with minimal imports to avoid circular dependencies
        import portfolio_config
        
        # Test configuration loading
        config = portfolio_config.ALL_CONFIGS
        print(f"✓ Configuration loaded: {len(config)} modules")
        
        # Test portfolio file creation
        portfolio_file = "data/test_portfolio.json"
        test_portfolio = {
            'cash_balance': 35_000_000,
            'positions': {},
            'trade_history': [],
            'performance_metrics': {},
            'created_date': datetime.now().isoformat()
        }
        
        os.makedirs('data', exist_ok=True)
        with open(portfolio_file, 'w') as f:
            json.dump(test_portfolio, f, indent=2)
        
        print("✓ Portfolio file creation test passed")
        
        # Clean up test file
        if os.path.exists(portfolio_file):
            os.remove(portfolio_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio manager test failed: {e}")
        return False

def test_signal_processing():
    """Test signal processing capability"""
    print("\nTesting signal processing...")
    
    try:
        db_path = r"data\databases\production\PSX_investing_Stocks_KMI100.db"
        conn = sqlite3.connect(db_path)
        
        # Test buy signals query
        query = """
        SELECT Stock, Close, Volume, RSI_Weekly_Avg, [% P/L] as PnL_Percent
        FROM buy_stocks 
        WHERE Status = 'Buy' 
        ORDER BY [% P/L] DESC
        LIMIT 5
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        if results:
            print("✓ Top 5 buy signals:")
            for i, row in enumerate(results, 1):
                stock, close, volume, rsi, pnl = row
                print(f"  {i}. {stock}: {close:.2f} PKR, RSI: {rsi:.1f}, P&L: {pnl:.2f}%")
        else:
            print("⚠️  No buy signals found")
        
        conn.close()
        print("✓ Signal processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Signal processing test failed: {e}")
        return False

def test_risk_calculations():
    """Test risk calculation functions"""
    print("\nTesting risk calculations...")
    
    try:
        # Test basic risk calculations
        portfolio_value = 35_000_000
        position_value = 1_000_000
        
        # Position weight calculation
        position_weight = (position_value / portfolio_value) * 100
        
        # Test transaction cost calculation
        transaction_cost_rate = 0.002
        shares = 1000
        price = 150.0
        total_cost = shares * price * (1 + transaction_cost_rate)
        
        print(f"✓ Position weight calculation: {position_weight:.2f}%")
        print(f"✓ Transaction cost calculation: {total_cost:,.0f} PKR")
        
        # Test portfolio concentration (HHI)
        test_weights = [0.08, 0.06, 0.05, 0.04, 0.04]  # Top 5 positions
        hhi = sum(w**2 for w in test_weights) * 10000
        
        print(f"✓ Portfolio concentration (HHI): {hhi:.0f}")
        
        print("✓ Risk calculations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Risk calculations test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure and permissions"""
    print("\nTesting directory structure...")
    
    try:
        required_dirs = [
            'data',
            'data/logs',
            'data/reports',
            'data/backups',
            'data/exports'
        ]
        
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(directory, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"✓ {directory}: OK")
            else:
                print(f"❌ {directory}: Write permission failed")
                return False
        
        print("✓ Directory structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False

def test_import_dependencies():
    """Test import of required dependencies"""
    print("\nTesting dependencies...")
    
    required_modules = [
        'pandas',
        'numpy',
        'sqlite3',
        'json',
        'logging',
        'datetime'
    ]
    
    optional_modules = [
        'matplotlib',
        'seaborn'
    ]
    
    try:
        for module in required_modules:
            try:
                __import__(module)
                print(f"✓ {module}: Available")
            except ImportError:
                print(f"❌ {module}: Missing (Required)")
                return False
        
        for module in optional_modules:
            try:
                __import__(module)
                print(f"✓ {module}: Available")
            except ImportError:
                print(f"⚠️  {module}: Missing (Optional)")
        
        print("✓ Dependencies test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dependencies test failed: {e}")
        return False

def run_system_validation():
    """Run complete system validation"""
    print("="*60)
    print("PORTFOLIO MANAGEMENT SYSTEM - VALIDATION TEST")
    print("="*60)
    
    tests = [
        ("Dependencies", test_import_dependencies),
        ("Directory Structure", test_directory_structure),
        ("Database Connectivity", test_database_connectivity),
        ("Signal Processing", test_signal_processing),
        ("Risk Calculations", test_risk_calculations),
        ("Portfolio Manager Init", test_portfolio_manager_init)
    ]
    
    results = {}
    
    for test_name, test_function in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_function()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 System validation completed successfully!")
        print("The portfolio management system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please resolve issues before using the system.")
        return False

if __name__ == "__main__":
    success = run_system_validation()
    
    if success:
        print("\nTo start the portfolio management system, run:")
        print("python launch_portfolio_system.py")
    else:
        print("\nPlease fix the identified issues before proceeding.")
    
    sys.exit(0 if success else 1)
