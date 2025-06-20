#!/usr/bin/env python3
"""
Portfolio Management System Launcher - Working Version
Main entry point for the robust portfolio management system
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

# Add utils directory to path for Telegram imports
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils')
sys.path.append(utils_path)

from core.simple_portfolio_manager import SimplePortfolioManager
from core.portfolio_config import ALL_CONFIGS

# Import Telegram functionality
try:
    from telegram_message import send_telegram_message
    TELEGRAM_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
        from telegram_message import send_telegram_message
        TELEGRAM_AVAILABLE = True
    except ImportError:
        TELEGRAM_AVAILABLE = False
        print("⚠️  Telegram functionality not available. Install required dependencies or check configuration.")

class PortfolioSystemLauncher:
    def __init__(self):
        self.setup_directories()
        self.setup_logging()
        self.config = ALL_CONFIGS
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data',
            'data/logs',
            'data/reports',
            'data/backups',
            'data/exports'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_logging(self):
        """Setup system-wide logging"""
        log_config = ALL_CONFIGS['logging']
        
        logging.basicConfig(
            level=getattr(logging, log_config['log_level']),
            format=log_config['log_format'],
            handlers=[
                logging.FileHandler(f"data/logs/portfolio_system_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler() if log_config['console_output'] else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger('PortfolioSystemLauncher')
        self.logger.info("Portfolio Management System starting up...")
    
    def run_initial_setup(self):
        """Run initial system setup and validation"""
        try:
            self.logger.info("Running initial system setup...")
            
            # Check database connectivity
            if not self.validate_database_connection():
                self.logger.error("Database validation failed!")
                return False
            
            # Initialize portfolio manager
            portfolio_manager = SimplePortfolioManager()
            
            # Generate initial summary
            self.logger.info("Running initial portfolio assessment...")
            summary = portfolio_manager.get_portfolio_summary()
            
            if summary:
                self.logger.info("Portfolio summary generated successfully")
                self.print_portfolio_summary(summary)
            if summary:
                self.logger.info("Portfolio summary generated successfully")
                self.print_portfolio_summary(summary)

            return True
        except Exception as e:
            self.logger.error(f"Initial setup failed: {e}")
            return False
    
    def validate_database_connection(self) -> bool:
        """Validate that the signal database is accessible"""
        try:
            import sqlite3
            
            # Use SimplePortfolioManager to get the correct database path
            pm = SimplePortfolioManager()
            db_path = pm.signal_db_path
            
            if not os.path.exists(db_path):
                self.logger.error(f"Signal database not found: {db_path}")
                self.logger.info(f"Looked in: {db_path}")
                return False
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check for required tables
            required_tables = ['buy_stocks', 'sell_stocks', 'neutral_stocks']
            
            for table in required_tables:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    self.logger.error(f"Required table '{table}' not found in database")
                    conn.close()
                    return False
            
            conn.close()
            self.logger.info("Database validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Database validation error: {e}")
            return False
    
    def print_portfolio_summary(self, summary: Dict):
        """Print portfolio summary"""
        try:
            print("\n" + "="*80)
            print("PORTFOLIO MANAGEMENT SYSTEM - STATUS SUMMARY")
            print("="*80)
            
            print(f"\nPORTFOLIO STATUS:")
            print(f"Portfolio Value: {summary.get('portfolio_value', 0):,.0f} PKR")
            print(f"Cash Balance: {summary.get('cash_balance', 0):,.0f} PKR")
            print(f"Invested Amount: {summary.get('invested_amount', 0):,.0f} PKR")
            print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"Number of Positions: {summary.get('num_positions', 0)}")
            
            # Current positions
            positions = summary.get('positions', {})
            if positions:
                print(f"\nCURRENT POSITIONS (Top 5):")
                sorted_positions = sorted(positions.items(), 
                                        key=lambda x: x[1]['position_value'], 
                                        reverse=True)
                for i, (stock, pos) in enumerate(sorted_positions[:5]):
                    print(f"{i+1}. {stock}: {pos['shares']} shares @ {pos['current_price']:.2f} PKR "
                          f"(Value: {pos['position_value']:,.0f} PKR)")
            
            print("\n" + "="*80)
            
        except Exception as e:
            self.logger.error(f"Error printing summary: {e}")
    
    def run_portfolio_manager(self, mode: str = 'single'):
        """Run the portfolio manager in specified mode"""
        try:
            self.logger.info(f"Starting portfolio manager in {mode} mode...")
            
            portfolio_manager = SimplePortfolioManager()
            
            if mode == 'single':
                # Run single rebalancing cycle
                portfolio_manager.rebalance_portfolio()
                portfolio_manager.print_portfolio_report()
                
                print("\nRebalancing completed. Report generated.")
                return True
            
            elif mode == 'continuous':
                # For continuous mode, we'll just explain how to do it
                print(f"\nContinuous monitoring instructions:")
                print("The SimplePortfolioManager is designed for manual execution.")
                print("For automated continuous monitoring, run this command periodically:")
                print("python simple_portfolio_manager.py")
                print("\nYou can set up a scheduler (cron on Linux/Mac, Task Scheduler on Windows)")
                print("to run the portfolio manager at regular intervals (e.g., every hour).")
                return True
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Portfolio manager stopped by user")
            return True
        except Exception as e:
            self.logger.error(f"Error running portfolio manager: {e}")
            return False
    
    def run_analytics_only(self):
        """Run analytics and generate reports without trading"""
        try:
            self.logger.info("Running analytics module...")
            
            # Use SimplePortfolioManager for analytics
            portfolio_manager = SimplePortfolioManager()
            
            # Generate and display report
            portfolio_manager.print_portfolio_report()
            
            print("\nAnalytics completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running analytics: {e}")
            return False

    def display_main_menu(self):
        """Display comprehensive main menu with all available options"""
        print("\n" + "="*80)
        print("🚀 PORTFOLIO MANAGEMENT SYSTEM - MAIN MENU")
        print("="*80)
        print("\nSelect your desired operation:")
        print("\n📊 PORTFOLIO OPERATIONS:")
        print("  1. Quick Portfolio Status Check (Analytics Only)")
        print("  2. Single Rebalancing Cycle (Update Positions)")
        print("  3. Full System Setup & Validation")
        
        print("\n📈 ADVANCED OPTIONS:")
        print("  4. Continuous Monitoring Instructions")
        print("  5. Database Connection Test")
        print("  6. Export Portfolio Report (+ Telegram Options)")
        
        print("\n🔧 SYSTEM UTILITIES:")
        print("  7. View Current Configuration")
        print("  8. Signal Database Inspection")
        print("  9. System Performance Test")
        
        print("\n❌ EXIT:")
        print("  0. Exit Portfolio System")
        
        print("\n" + "="*80)
        return self.get_user_choice()
    
    def get_user_choice(self):
        """Get and validate user menu choice"""
        while True:
            try:
                choice = input("\n👉 Enter your choice (0-9): ").strip()
                
                if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    return choice
                else:
                    print("❌ Invalid choice. Please enter a number between 0-9.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return '0'
            except Exception as e:
                print(f"❌ Input error: {e}. Please try again.")

    def execute_menu_choice(self, choice: str):
        """Execute the selected menu option"""
        try:
            if choice == '0':
                print("\n👋 Thank you for using Portfolio Management System!")
                return False
                
            elif choice == '1':
                print("\n📊 Running Quick Portfolio Status Check...")
                self.run_analytics_only()
                
            elif choice == '2':
                print("\n📈 Starting Single Rebalancing Cycle...")
                print("⚠️  This will update your portfolio positions based on current signals.")
                confirm = input("Continue? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    self.run_portfolio_manager('single')
                else:
                    print("❌ Rebalancing cancelled.")
                    
            elif choice == '3':
                print("\n🔧 Running Full System Setup & Validation...")
                self.run_initial_setup()
                
            elif choice == '4':
                print("\n📈 Continuous Monitoring Setup Instructions...")
                self.show_continuous_monitoring_guide()
                
            elif choice == '5':
                print("\n🔌 Testing Database Connection...")
                if self.validate_database_connection():
                    print("✅ Database connection successful!")
                else:
                    print("❌ Database connection failed!")
                    
            elif choice == '6':
                print("\n📄 Exporting Portfolio Report (with Telegram options)...")
                self.export_portfolio_report()
                
            elif choice == '7':
                print("\n⚙️  Current System Configuration...")
                self.show_configuration()
                
            elif choice == '8':
                print("\n🔍 Signal Database Inspection...")
                self.inspect_signal_database()
                
            elif choice == '9':
                print("\n⚡ Running System Performance Test...")
                self.run_system_performance_test()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing menu choice {choice}: {e}")
            print(f"❌ Error: {e}")
            return True

    def show_continuous_monitoring_guide(self):
        """Display detailed continuous monitoring setup guide"""
        print("\n" + "="*80)
        print("📈 CONTINUOUS MONITORING SETUP GUIDE")
        print("="*80)
        
        print("\n🎯 RECOMMENDED APPROACH:")
        print("For automated portfolio monitoring, set up scheduled execution:")
        
        print("\n💻 Windows Task Scheduler:")
        print("1. Open Task Scheduler")
        print("2. Create Basic Task")
        print("3. Set trigger: Daily at market close (e.g., 4:00 PM)")
        print("4. Action: Start Program")
        print(f"5. Program: python")
        print(f"6. Arguments: {os.path.abspath('simple_portfolio_manager.py')}")
        
        print("\n🐧 Linux/Mac Cron Job:")
        print("1. Open terminal: crontab -e")
        print("2. Add line: 0 16 * * 1-5 cd /path/to/portfolio && python simple_portfolio_manager.py")
        print("   (Runs Mon-Fri at 4:00 PM)")
        
        print("\n⏰ RECOMMENDED FREQUENCY:")
        print("• Daily: After market close for position updates")
        print("• Hourly: During market hours for monitoring only")
        print("• Weekly: For comprehensive rebalancing")
        
        print("\n🔔 MONITORING OPTIONS:")
        print("• Manual: Run this system when needed")
        print("• Scheduled: Set up automated execution")
        print("• Hybrid: Automatic monitoring + manual rebalancing")
        
        input("\n📋 Press Enter to return to main menu...")

    def export_portfolio_report(self):
        """Export comprehensive portfolio report and optionally send to Telegram"""
        try:
            portfolio_manager = SimplePortfolioManager()
            summary = portfolio_manager.get_portfolio_summary()
            
            # Create exports directory
            os.makedirs('data/exports', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/exports/portfolio_report_{timestamp}.json"
            
            # Export as JSON
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"✅ Portfolio report exported to: {filename}")
            
            # Also create a readable text version
            text_filename = f"data/exports/portfolio_report_{timestamp}.txt"
            with open(text_filename, 'w') as f:
                f.write("PORTFOLIO MANAGEMENT SYSTEM - DETAILED REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Report Generated: {datetime.now()}\n\n")
                
                f.write(f"Portfolio Value: {summary.get('portfolio_value', 0):,.0f} PKR\n")
                f.write(f"Cash Balance: {summary.get('cash_balance', 0):,.0f} PKR\n")
                f.write(f"Invested Amount: {summary.get('invested_amount', 0):,.0f} PKR\n")
                f.write(f"Total Return: {summary.get('total_return_pct', 0):.2f}%\n")
                f.write(f"Number of Positions: {summary.get('num_positions', 0)}\n\n")
                
                positions = summary.get('positions', {})
                if positions:
                    f.write("CURRENT POSITIONS:\n")
                    f.write("-" * 60 + "\n")
                    for stock, pos in positions.items():
                        f.write(f"{stock}: {pos['shares']} shares @ {pos['current_price']:.2f} PKR\n")
                        f.write(f"  Value: {pos['position_value']:,.0f} PKR\n")
                        f.write(f"  P&L: {pos['unrealized_pnl_pct']:.2f}%\n\n")
            
            print(f"✅ Text report exported to: {text_filename}")
              # Ask user if they want to send to Telegram
            if TELEGRAM_AVAILABLE:
                send_to_telegram = input("\n📱 Would you like to send this report to Telegram? (y/N): ").strip().lower()
                if send_to_telegram in ['y', 'yes']:
                    # Ask user what type of message to send
                    print("\n📊 Choose message type:")
                    print("1. Full Portfolio Report (overview + holdings + active trades)")
                    print("2. Active Trades Only (up to 26 signals)")
                    choice = input("Enter choice (1-2): ").strip()
                    
                    if choice == '2':
                        self.send_active_trades_to_telegram()
                    else:
                        self.send_portfolio_report_to_telegram(summary)
            else:
                print("📱 Telegram functionality not available. Configure Telegram settings to enable this feature.")
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            print(f"❌ Export failed: {e}")

    def send_portfolio_report_to_telegram(self, summary: Dict):
        """Send formatted portfolio report to Telegram"""
        try:
            # Create formatted message for Telegram
            message = self.format_telegram_portfolio_message(summary)
            
            # Send to Telegram
            success = send_telegram_message(message)
            
            if success:
                print("✅ Portfolio report sent to Telegram successfully!")
                self.logger.info("Portfolio report sent to Telegram")
            else:
                print("❌ Failed to send portfolio report to Telegram")
                print("   Check your Telegram bot configuration and internet connection")
                self.logger.error("Failed to send portfolio report to Telegram")
                
        except Exception as e:
            self.logger.error(f"Error sending report to Telegram: {e}")
            print(f"❌ Telegram send failed: {e}")

    def format_telegram_portfolio_message(self, summary: Dict) -> str:
        """Format portfolio summary for Telegram message"""
        try:
            # Portfolio emoji indicators
            portfolio_value = summary.get('portfolio_value', 0)
            total_return = summary.get('total_return_pct', 0)
            
            # Choose emoji based on performance
            if total_return > 5:
                performance_emoji = "🚀"
            elif total_return > 0:
                performance_emoji = "📈"
            elif total_return > -5:
                performance_emoji = "📊"
            else:
                performance_emoji = "📉"
                
            message = f"{performance_emoji} *PSX Portfolio Report*\n"
            message += f"📅 {datetime.now().strftime('%d %b %Y, %H:%M')}\n\n"
            
            # Portfolio overview
            message += f"💰 *Portfolio Overview:*\n"
            message += f"• Total Value: `{portfolio_value:,.0f} PKR`\n"
            message += f"• Cash Balance: `{summary.get('cash_balance', 0):,.0f} PKR`\n"
            message += f"• Invested: `{summary.get('invested_amount', 0):,.0f} PKR`\n"
            message += f"• Total Return: `{total_return:.2f}%`\n"
            message += f"• Positions: `{summary.get('num_positions', 0)}`\n\n"
            
            # Top positions
            positions = summary.get('positions', {})
            if positions:
                message += f"📊 *Top 5 Holdings:*\n"
                sorted_positions = sorted(positions.items(), 
                                        key=lambda x: x[1]['position_value'], 
                                        reverse=True)
                
                for i, (stock, pos) in enumerate(sorted_positions[:]):
                    pnl = pos.get('unrealized_pnl_pct', 0)
                    pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
                    
                    message += f"{i+1}. *{stock}*\n"
                    message += f"   `{pos['shares']} shares @ {pos['current_price']:.2f} PKR`\n"
                    message += f"   `Value: {pos['position_value']:,.0f} PKR`\n"
                    message += f"   {pnl_emoji} `P&L: {pnl:.2f}%`\n\n"
            
            # Add active trades section
            message += self.get_active_trades_section()
            
            # Performance indicators
            if total_return > 0:
                message += f"✅ Portfolio performing well! Keep monitoring.\n"
            elif total_return > -5:
                message += f"⚠️ Portfolio slightly down. Consider rebalancing.\n"
            else:
                message += f"🔴 Portfolio needs attention. Review positions.\n"
                
            message += f"\n📱 _Generated by PSX Portfolio Management System_"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting Telegram message: {e}")
            return f"❌ Error generating portfolio report: {e}"

    def get_active_trades_section(self) -> str:
        """Get all active trades for Telegram message"""
        try:
            # Get all active signals from database
            portfolio_manager = SimplePortfolioManager()
            
            # Get buy signals
            buy_signals = self.get_all_buy_signals(portfolio_manager)
            sell_signals = self.get_all_sell_signals(portfolio_manager)
            
            message = f"🎯 *Active Trading Signals:*\n"
            
            # Buy signals
            if not buy_signals.empty:
                message += f"📈 *Buy Signals ({len(buy_signals)}):*\n"
                for _, signal in buy_signals.head(8).iterrows():  # Show top 8
                    stock = signal.get('Stock', 'N/A')
                    price = signal.get('Close', 0)
                    pnl = signal.get('PnL_Percent', 0)
                    signal_date = signal.get('Signal_Date', 'N/A')
                    
                    pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
                    
                    # Format signal date
                    try:
                        if pd.notna(signal_date) and signal_date != 'N/A':
                            date_obj = pd.to_datetime(signal_date)
                            date_str = date_obj.strftime('%m-%d')
                        else:
                            date_str = "N/A"
                    except:
                        date_str = "N/A"
                    
                    message += f"  • {stock}: `{price:.1f} PKR` {pnl_emoji}`{pnl:.1f}%` ({date_str})\n"
                
                if len(buy_signals) > 8:
                    message += f"  _...and {len(buy_signals) - 8} more buy signals_\n"
                message += "\n"
            
            # Sell signals
            if not sell_signals.empty:
                message += f"📉 *Sell Signals ({len(sell_signals)}):*\n"
                for _, signal in sell_signals.head(8).iterrows():  # Show top 8
                    stock = signal.get('Stock', 'N/A')
                    price = signal.get('Close', 0)
                    pnl = signal.get('PnL_Percent', 0)
                    signal_date = signal.get('Signal_Date', 'N/A')
                    
                    pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
                    
                    # Format signal date
                    try:
                        if pd.notna(signal_date) and signal_date != 'N/A':
                            date_obj = pd.to_datetime(signal_date)
                            date_str = date_obj.strftime('%m-%d')
                        else:
                            date_str = "N/A"
                    except:
                        date_str = "N/A"
                    
                    message += f"  • {stock}: `{price:.1f} PKR` {pnl_emoji}`{pnl:.1f}%` ({date_str})\n"
                
                if len(sell_signals) > 8:
                    message += f"  _...and {len(sell_signals) - 8} more sell signals_\n"
                message += "\n"
            
            if buy_signals.empty and sell_signals.empty:
                message += f"  _No active signals found_\n\n"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error getting active trades: {e}")
            return f"❌ *Active Trades:* Error loading signals\n\n"

    def send_active_trades_to_telegram(self):
        """Send only active trading signals to Telegram"""
        try:
            # Create formatted message for active trades only
            message = self.format_active_trades_message()
            
            # Send to Telegram
            success = send_telegram_message(message)
            
            if success:
                print("✅ Active trades sent to Telegram successfully!")
                self.logger.info("Active trades sent to Telegram")
            else:
                print("❌ Failed to send active trades to Telegram")
                print("   Check your Telegram bot configuration and internet connection")
                self.logger.error("Failed to send active trades to Telegram")
                
        except Exception as e:
            self.logger.error(f"Error sending active trades to Telegram: {e}")
            print(f"❌ Telegram send failed: {e}")

    def format_active_trades_message(self) -> str:
        """Format only active trading signals for Telegram message"""
        try:
            # Get all active signals from database
            portfolio_manager = SimplePortfolioManager()
            
            # Get buy and sell signals
            buy_signals = self.get_all_buy_signals(portfolio_manager)
            sell_signals = self.get_all_sell_signals(portfolio_manager)
            
            # Header with date
            message = f"🎯 *PSX Active Trading Signals*\n"
            message += f"📅 {datetime.now().strftime('%d %b %Y, %H:%M')}\n\n"
            
            total_signals = len(buy_signals) + len(sell_signals)
            if total_signals == 0:
                message += f"📊 No active trading signals found.\n\n"
                message += f"📱 _Generated by PSX Portfolio Management System_"
                return message
            
            message += f"📊 *Total Active Signals: {total_signals}*\n\n"
            
            # Buy signals (up to 13 to stay within 26 limit)
            if not buy_signals.empty:
                buy_count = min(13, len(buy_signals))
                message += f"📈 *Buy Signals ({len(buy_signals)}):*\n"
                
                for _, signal in buy_signals.head(buy_count).iterrows():
                    stock = signal.get('Stock', 'N/A')
                    price = signal.get('Close', 0)
                    pnl = signal.get('PnL_Percent', 0)
                    signal_date = signal.get('Signal_Date', 'N/A')
                    
                    pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
                    
                    # Format signal date
                    try:
                        if pd.notna(signal_date) and signal_date != 'N/A':
                            date_obj = pd.to_datetime(signal_date)
                            date_str = date_obj.strftime('%m-%d')
                        else:
                            date_str = "N/A"
                    except:
                        date_str = "N/A"
                    
                    message += f"• *{stock}*: `{price:.1f} PKR` {pnl_emoji}`{pnl:.1f}%` ({date_str})\n"
                
                if len(buy_signals) > buy_count:
                    message += f"_...and {len(buy_signals) - buy_count} more buy signals_\n"
                message += "\n"
            
            # Sell signals (up to 13 to stay within 26 limit)
            if not sell_signals.empty:
                sell_count = min(13, len(sell_signals))
                message += f"📉 *Sell Signals ({len(sell_signals)}):*\n"
                
                for _, signal in sell_signals.head(sell_count).iterrows():
                    stock = signal.get('Stock', 'N/A')
                    price = signal.get('Close', 0)
                    pnl = signal.get('PnL_Percent', 0)
                    signal_date = signal.get('Signal_Date', 'N/A')
                    
                    pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
                    
                    # Format signal date
                    try:
                        if pd.notna(signal_date) and signal_date != 'N/A':
                            date_obj = pd.to_datetime(signal_date)
                            date_str = date_obj.strftime('%m-%d')
                        else:
                            date_str = "N/A"
                    except:
                        date_str = "N/A"
                    
                    message += f"• *{stock}*: `{price:.1f} PKR` {pnl_emoji}`{pnl:.1f}%` ({date_str})\n"
                
                if len(sell_signals) > sell_count:
                    message += f"_...and {len(sell_signals) - sell_count} more sell signals_\n"
                message += "\n"
            
            # Trading recommendations based on signal counts
            buy_positive = len(buy_signals[buy_signals['PnL_Percent'] > 0]) if not buy_signals.empty else 0
            sell_positive = len(sell_signals[sell_signals['PnL_Percent'] > 0]) if not sell_signals.empty else 0
            
            if buy_positive > sell_positive:
                message += f"💡 *Market Outlook*: Bullish trend with {buy_positive} profitable buy signals\n"
            elif sell_positive > buy_positive:
                message += f"💡 *Market Outlook*: Bearish trend with {sell_positive} profitable sell signals\n"
            else:
                message += f"💡 *Market Outlook*: Mixed signals, exercise caution\n"
                
            message += f"\n📱 _Generated by PSX Portfolio Management System_"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting active trades message: {e}")
            return f"❌ Error generating active trades report: {e}"

    def get_all_buy_signals(self, portfolio_manager) -> pd.DataFrame:
        """Get all buy signals from database"""
        try:
            import sqlite3
            conn = sqlite3.connect(portfolio_manager.signal_db_path)
            
            query = """
            SELECT Stock, Close, [% P/L] as PnL_Percent, Signal_Date, Signal_Close, Status
            FROM buy_stocks 
            WHERE Status = 'Buy'
            ORDER BY Signal_Date DESC
            LIMIT 50
            """
            
            signals = pd.read_sql_query(query, conn)
            conn.close()
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting buy signals: {e}")
            return pd.DataFrame()

    def get_all_sell_signals(self, portfolio_manager) -> pd.DataFrame:
        """Get all sell signals from database"""
        try:
            import sqlite3
            conn = sqlite3.connect(portfolio_manager.signal_db_path)
            
            query = """
            SELECT Stock, Close, [% P/L] as PnL_Percent, Signal_Date, Signal_Close, Status
            FROM sell_stocks 
            WHERE Status = 'Sell'
            ORDER BY Signal_Date DESC
            LIMIT 50
            """
            
            signals = pd.read_sql_query(query, conn)
            conn.close()
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting sell signals: {e}")
            return pd.DataFrame()

    def show_configuration(self):
        """Display current system configuration"""
        print("\n" + "="*60)
        print("⚙️  PORTFOLIO SYSTEM CONFIGURATION")
        print("="*60)
        
        try:
            portfolio_config = self.config.get('portfolio', {})
            signals_config = self.config.get('signals', {})
            
            print(f"\n💰 Portfolio Settings:")
            print(f"  Initial Capital: {portfolio_config.get('initial_capital', 0):,.0f} PKR")
            print(f"  Max Positions: {portfolio_config.get('max_positions', 0)}")
            print(f"  Min Position Size: {portfolio_config.get('min_position_size', 0):,.0f} PKR")
            print(f"  Max Position Size: {portfolio_config.get('max_position_size', 0):,.0f} PKR")
            
            print(f"\n📊 Signal Settings:")
            print(f"  Database Path: {signals_config.get('signal_db_path', 'Not set')}")
            print(f"  Top Signals: {signals_config.get('top_signals_count', 0)}")
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
        
        input("\n📋 Press Enter to return to main menu...")

    def inspect_signal_database(self):
        """Quick inspection of signal database"""
        try:
            import sqlite3
            
            # Use SimplePortfolioManager to get the correct database path
            pm = SimplePortfolioManager()
            db_path = pm.signal_db_path
            
            conn = sqlite3.connect(db_path)
            
            print("\n📊 SIGNAL DATABASE SUMMARY:")
            print("-" * 40)
            
            # Check buy signals
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM buy_stocks WHERE Status='Buy'")
            buy_count = cursor.fetchone()[0]
            print(f"Active Buy Signals: {buy_count}")
            
            # Check sell signals  
            cursor.execute("SELECT COUNT(*) FROM sell_stocks WHERE Status='Sell'")
            sell_count = cursor.fetchone()[0]
            print(f"Active Sell Signals: {sell_count}")
            
            # Check neutral signals
            cursor.execute("SELECT COUNT(*) FROM neutral_stocks WHERE Status='Neutral'")
            neutral_count = cursor.fetchone()[0]
            print(f"Neutral Signals: {neutral_count}")
            
            # Top 5 buy signals
            cursor.execute("""
                SELECT Stock, [% P/L] as PnL, Close 
                FROM buy_stocks 
                WHERE Status='Buy' AND Success='Yes' 
                ORDER BY [% P/L] DESC 
                LIMIT 5
            """)
            top_signals = cursor.fetchall()
            
            print(f"\n🏆 Top 5 Buy Signals:")
            for i, (stock, pnl, price) in enumerate(top_signals, 1):
                print(f"  {i}. {stock}: {pnl:.1f}% P&L @ {price:.2f} PKR")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Database inspection failed: {e}")
        
        input("\n📋 Press Enter to return to main menu...")

    def run_system_performance_test(self):
        """Run comprehensive system performance test"""
        print("\n⚡ SYSTEM PERFORMANCE TEST")
        print("-" * 40)
        
        try:
            import time
            
            # Test 1: Database connection speed
            start_time = time.time()
            db_ok = self.validate_database_connection()
            db_time = time.time() - start_time
            print(f"Database Connection: {'✅' if db_ok else '❌'} ({db_time:.3f}s)")
            
            # Test 2: Portfolio manager initialization
            start_time = time.time()
            portfolio_manager = SimplePortfolioManager()
            init_time = time.time() - start_time
            print(f"Portfolio Manager Init: ✅ ({init_time:.3f}s)")
            
            # Test 3: Signal retrieval speed
            start_time = time.time()
            signals = portfolio_manager.get_current_signals()
            signal_time = time.time() - start_time
            print(f"Signal Retrieval: ✅ ({signal_time:.3f}s, {len(signals)} signals)")
            
            # Test 4: Portfolio summary generation
            start_time = time.time()
            summary = portfolio_manager.get_portfolio_summary()
            summary_time = time.time() - start_time
            print(f"Portfolio Summary: ✅ ({summary_time:.3f}s)")
            
            total_time = db_time + init_time + signal_time + summary_time
            print(f"\n🏁 Total Test Time: {total_time:.3f}s")
            
            if total_time < 1.0:
                print("🚀 Performance: Excellent!")
            elif total_time < 3.0:
                print("✅ Performance: Good")
            else:
                print("⚠️  Performance: Consider optimization")
                
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
        
        input("\n📋 Press Enter to return to main menu...")

def main():
    """Main entry point with interactive menu system"""
    parser = argparse.ArgumentParser(description='Portfolio Management System')
    parser.add_argument('--mode', choices=['setup', 'single', 'continuous', 'analytics', 'interactive'], 
                       default='interactive', help='Operation mode (default: interactive)')
    parser.add_argument('--config', help='Custom configuration file (JSON)')
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = PortfolioSystemLauncher()
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            launcher.config.update(custom_config)
            launcher.logger.info(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            launcher.logger.error(f"Failed to load custom config: {e}")
    
    try:
        if args.mode == 'interactive' or len(sys.argv) == 1:
            # Default interactive mode - show menu
            print("\n🌟 Welcome to Portfolio Management System!")
            print("💼 Managing your PSX investment portfolio with intelligent signal integration")
            
            # Run initial validation silently
            launcher.validate_database_connection()
            
            # Show interactive menu
            while True:
                choice = launcher.display_main_menu()
                
                if not launcher.execute_menu_choice(choice):
                    break  # User chose to exit
                
                # Pause before showing menu again (except for exit)
                if choice != '0':
                    input("\n⏸️  Press Enter to return to main menu...")
        
        elif args.mode == 'setup':
            # Run initial setup and show summary
            if launcher.run_initial_setup():
                response = input("\nSelect operation mode:\n1. Single rebalancing\n2. Continuous monitoring info\n3. Analytics only\nEnter choice (1-3): ").strip()
                
                mode_map = {'1': 'single', '2': 'continuous', '3': 'analytics'}
                selected_mode = mode_map.get(response, 'single')
                
                if selected_mode == 'analytics':
                    launcher.run_analytics_only()
                else:
                    launcher.run_portfolio_manager(selected_mode)
        
        elif args.mode == 'single':
            launcher.run_portfolio_manager('single')
        
        elif args.mode == 'continuous':
            launcher.run_portfolio_manager('continuous')
        
        elif args.mode == 'analytics':
            launcher.run_analytics_only()
        
    except KeyboardInterrupt:
        print("\n\n👋 System stopped by user. Goodbye!")
        launcher.logger.info("System stopped by user")
    except Exception as e:
        launcher.logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
