#!/usr/bin/env python3
"""
Portfolio Management System Launcher - Working Version
Main entry point for the robust portfolio management system
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_portfolio_manager import SimplePortfolioManager
from portfolio_config import ALL_CONFIGS

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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initial setup failed: {e}")
            return False
    
    def validate_database_connection(self) -> bool:
        """Validate that the signal database is accessible"""
        try:
            import sqlite3
            db_path = self.config['signals']['signal_db_path']
            
            if not os.path.exists(db_path):
                self.logger.error(f"Signal database not found: {db_path}")
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

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Portfolio Management System')
    parser.add_argument('--mode', choices=['setup', 'single', 'continuous', 'analytics'], 
                       default='setup', help='Operation mode')
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
        if args.mode == 'setup':
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
        launcher.logger.info("System stopped by user")
    except Exception as e:
        launcher.logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
