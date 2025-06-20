#!/usr/bin/env python3
"""
Simple Portfolio System Launcher - Fixed Version
Launches the portfolio management system with proper database path handling
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

from core.simple_portfolio_manager import SimplePortfolioManager

class SimplePortfolioLauncher:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Setup basic logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SimplePortfolioLauncher')
    
    def validate_database(self):
        """Quick database validation"""
        try:
            # Create a portfolio manager instance to get the correct database path
            pm = SimplePortfolioManager()
            db_path = pm.signal_db_path
            
            if not os.path.exists(db_path):
                self.logger.error(f"Database not found: {db_path}")
                return False
            
            # Test connection
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check for required tables
            required_tables = ['buy_stocks', 'sell_stocks', 'neutral_stocks']
            
            for table in required_tables:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    self.logger.error(f"Required table '{table}' not found")
                    conn.close()
                    return False
            
            conn.close()
            self.logger.info("✅ Database validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database validation failed: {e}")
            return False
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*80)
        print("🚀 PORTFOLIO MANAGEMENT SYSTEM")
        print("="*80)
        print("\nSelect your operation:")
        print("\n📊 PORTFOLIO OPERATIONS:")
        print("  1. Quick Portfolio Status Check")
        print("  2. Single Rebalancing Cycle")
        print("  3. Database Connection Test")
        print("\n📈 UTILITIES:")
        print("  4. Show Database Information")
        print("  5. Portfolio Summary")
        print("\n❌ EXIT:")
        print("  0. Exit System")
        print("\n" + "="*80)
        
        while True:
            try:
                choice = input("\n👉 Enter your choice (0-5): ").strip()
                if choice in ['0', '1', '2', '3', '4', '5']:
                    return choice
                else:
                    print("❌ Invalid choice. Please enter 0-5.")
            except KeyboardInterrupt:
                return '0'
    
    def run_portfolio_status(self):
        """Run portfolio status check"""
        try:
            print("\n📊 Running Portfolio Status Check...")
            pm = SimplePortfolioManager()
            pm.print_portfolio_report()
            return True
        except Exception as e:
            self.logger.error(f"Portfolio status failed: {e}")
            return False
    
    def run_rebalancing(self):
        """Run portfolio rebalancing"""
        try:
            print("\n📈 Portfolio Rebalancing")
            confirm = input("⚠️  This will update your portfolio. Continue? (y/N): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                print("\n🔄 Starting rebalancing...")
                pm = SimplePortfolioManager()
                pm.rebalance_portfolio()
                pm.print_portfolio_report()
                print("\n✅ Rebalancing completed!")
            else:
                print("❌ Rebalancing cancelled.")
            return True
        except Exception as e:
            self.logger.error(f"Rebalancing failed: {e}")
            return False
    
    def run_database_test(self):
        """Test database connection"""
        print("\n🔌 Testing Database Connection...")
        result = self.validate_database()
        if result:
            print("✅ Database connection successful!")
        else:
            print("❌ Database connection failed!")
        return result
    
    def show_database_info(self):
        """Show database information"""
        try:
            print("\n📊 Database Information")
            print("-" * 40)
            
            pm = SimplePortfolioManager()
            print(f"Database Path: {pm.signal_db_path}")
            print(f"Database Exists: {os.path.exists(pm.signal_db_path)}")
            
            if os.path.exists(pm.signal_db_path):
                conn = sqlite3.connect(pm.signal_db_path)
                cursor = conn.cursor()
                
                # Count signals
                cursor.execute("SELECT COUNT(*) FROM buy_stocks WHERE Status='Buy'")
                buy_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM sell_stocks WHERE Status='Sell'")
                sell_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM neutral_stocks WHERE Status='Neutral'")
                neutral_count = cursor.fetchone()[0]
                
                print(f"Buy Signals: {buy_count}")
                print(f"Sell Signals: {sell_count}")
                print(f"Neutral Signals: {neutral_count}")
                
                conn.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Database info failed: {e}")
            return False
    
    def show_portfolio_summary(self):
        """Show portfolio summary"""
        try:
            print("\n💼 Portfolio Summary")
            print("-" * 40)
            
            pm = SimplePortfolioManager()
            summary = pm.get_portfolio_summary()
            
            print(f"Portfolio Value: {summary.get('portfolio_value', 0):,.0f} PKR")
            print(f"Cash Balance: {summary.get('cash_balance', 0):,.0f} PKR")
            print(f"Invested Amount: {summary.get('invested_amount', 0):,.0f} PKR")
            print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"Number of Positions: {summary.get('num_positions', 0)}")
            
            return True
        except Exception as e:
            self.logger.error(f"Portfolio summary failed: {e}")
            return False
    
    def run(self):
        """Main run loop"""
        print("\n🌟 Welcome to Portfolio Management System!")
        print("💼 Managing your PSX investment portfolio")
        
        # Initial database validation
        if not self.validate_database():
            print("\n❌ Database validation failed. Please check your setup.")
            return
        
        while True:
            choice = self.display_menu()
            
            if choice == '0':
                print("\n👋 Thank you for using Portfolio Management System!")
                break
            elif choice == '1':
                self.run_portfolio_status()
            elif choice == '2':
                self.run_rebalancing()
            elif choice == '3':
                self.run_database_test()
            elif choice == '4':
                self.show_database_info()
            elif choice == '5':
                self.show_portfolio_summary()
            
            input("\n⏸️  Press Enter to continue...")

def main():
    """Main entry point"""
    try:
        launcher = SimplePortfolioLauncher()
        launcher.run()
    except KeyboardInterrupt:
        print("\n\n👋 System stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")

if __name__ == "__main__":
    main()
