#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import glob
import sqlite3
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from tabulate import tabulate
import concurrent.futures
# Import the existing Telegram message module
from scripts.data_processing.telegram_message import send_telegram_message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_signal_tracker_runner.log'),
        logging.StreamHandler()
    ]
)

# Constants
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'alert_config.json')

def load_config():
    """Load configuration from config file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            config = {
                "telegram": {
                    "enabled": True,  # Enable Telegram by default since we're using the existing module
                },
                "alerts": {
                    "signal_transitions": True,
                    "profit_threshold": 5.0,
                    "loss_threshold": -5.0,
                    "days_in_signal_threshold": 14
                },
                "analysis": {
                    "trend_detection": True,
                    "volume_analysis": True,
                    "performance_metrics": True
                }
            }
            
            # Ensure config directory exists
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            
            # Save default config
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            
            return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return {}

def format_signals_for_telegram(transitions, performance_metrics):
    """Format signal data for Telegram message"""
    if not transitions:
        return None
    
    try:
        # Create header
        message = f"🔔 Stock Signal Transitions - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Add transitions
        message += "🔄 New Signal Transitions:\n"
        for t in transitions:
            profit_str = f"{t['Profit_Loss_Pct']:.2f}%"
            if t['Profit_Loss_Pct'] > 0:
                profit_str = f"✅ +{profit_str}"
            elif t['Profit_Loss_Pct'] < 0:
                profit_str = f"❌ {profit_str}"
            
            message += f"• {t['Stock']}: {t['Previous_Signal'] or 'New'} ➡️ {t['Current_Signal']} ({profit_str}, {t['Days_In_Signal']} days)\n"
        
        # Add summary
        if performance_metrics:
            message += f"\n📊 Performance Summary:\n"
            message += f"• Buy Signals: {performance_metrics.get('buy_count', 0)} "
            message += f"(Avg: {performance_metrics.get('buy_avg_profit', 0):.2f}%, "
            message += f"Profitable: {performance_metrics.get('buy_profitable_pct', 0):.1f}%)\n"
            
            message += f"• Sell Signals: {performance_metrics.get('sell_count', 0)} "
            message += f"(Avg: {performance_metrics.get('sell_avg_profit', 0):.2f}%, "
            message += f"Profitable: {performance_metrics.get('sell_profitable_pct', 0):.1f}%)\n"
            
            message += f"• Neutral Signals: {performance_metrics.get('neutral_count', 0)}\n"
        
        return message
    except Exception as e:
        logging.error(f"Error formatting Telegram message: {str(e)}")
        return None

def format_high_profit_for_telegram(signals, signal_type="Profit/Loss"):
    """Format high profit/loss signals for Telegram"""
    if signals.empty:
        return None
    
    try:
        message = f"💰 High {signal_type} Signals - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        for _, row in signals.iterrows():
            profit_str = f"{row['Profit_Loss_Pct']:.2f}%"
            if row['Profit_Loss_Pct'] > 0:
                profit_str = f"✅ +{profit_str}"
            elif row['Profit_Loss_Pct'] < 0:
                profit_str = f"❌ {profit_str}"
            
            message += f"• {row['Stock']} ({row['Current_Signal']}): {profit_str} in {row['Days_In_Signal']} days\n"
        
        return message
    except Exception as e:
        logging.error(f"Error formatting high profit message: {str(e)}")
        return None

def detect_signal_transitions(db_path):
    """
    Detect signal transitions by comparing current and previous signal states
    Returns:
        - New transitions: list of dictionaries with transition details
        - Performance metrics: dictionary with performance analysis
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        # Get current signal states
        current_signals = pd.read_sql('''
            SELECT Stock, Current_Signal, Days_In_Signal, Profit_Loss_Pct, Last_Updated
            FROM signal_tracking
        ''', conn)
        
        # Get previous signal states from transition history (if exists)
        try:
            previous_signals = pd.read_sql('''
                SELECT Stock, Current_Signal, transition_date 
                FROM signal_transition_history
                WHERE id IN (
                    SELECT MAX(id) FROM signal_transition_history GROUP BY Stock
                )
            ''', conn)
        except:
            # Table might not exist
            previous_signals = pd.DataFrame(columns=['Stock', 'Current_Signal', 'transition_date'])
        
        # Create transition history table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS signal_transition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Stock TEXT,
                Previous_Signal TEXT,
                Current_Signal TEXT, 
                transition_date TEXT,
                Profit_Loss_Pct REAL,
                Days_In_Signal INTEGER
            )
        ''')
        
        # Detect transitions
        new_transitions = []
        for _, row in current_signals.iterrows():
            stock = row['Stock']
            current_signal = row['Current_Signal']
            
            # Find previous signal for this stock
            prev_signal_row = previous_signals[previous_signals['Stock'] == stock]
            
            if not prev_signal_row.empty:
                prev_signal = prev_signal_row.iloc[0]['Current_Signal']
                
                # Check if signal changed
                if prev_signal != current_signal:
                    # Record the transition
                    transition = {
                        'Stock': stock,
                        'Previous_Signal': prev_signal,
                        'Current_Signal': current_signal,
                        'transition_date': row['Last_Updated'],
                        'Profit_Loss_Pct': row['Profit_Loss_Pct'],
                        'Days_In_Signal': row['Days_In_Signal']
                    }
                    new_transitions.append(transition)
                    
                    # Insert into history table
                    conn.execute('''
                        INSERT INTO signal_transition_history 
                        (Stock, Previous_Signal, Current_Signal, transition_date, Profit_Loss_Pct, Days_In_Signal)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        stock, prev_signal, current_signal, 
                        row['Last_Updated'], row['Profit_Loss_Pct'], row['Days_In_Signal']
                    ))
            else:
                # First record for this stock
                conn.execute('''
                    INSERT INTO signal_transition_history 
                    (Stock, Previous_Signal, Current_Signal, transition_date, Profit_Loss_Pct, Days_In_Signal)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    stock, None, current_signal, 
                    row['Last_Updated'], row['Profit_Loss_Pct'], row['Days_In_Signal']
                ))
        
        # Commit changes and close connection
        conn.commit()
        
        # Performance metrics for signals
        performance_metrics = {}
        if not current_signals.empty:
            buy_signals = current_signals[current_signals['Current_Signal'] == 'Buy']
            sell_signals = current_signals[current_signals['Current_Signal'] == 'Sell']
            neutral_signals = current_signals[current_signals['Current_Signal'] == 'Neutral']
            
            # Calculate average performance
            performance_metrics['buy_avg_profit'] = buy_signals['Profit_Loss_Pct'].mean() if not buy_signals.empty else 0
            performance_metrics['sell_avg_profit'] = sell_signals['Profit_Loss_Pct'].mean() if not sell_signals.empty else 0
            
            # Calculate profitable signal percentages
            if not buy_signals.empty:
                performance_metrics['buy_profitable_pct'] = len(buy_signals[buy_signals['Profit_Loss_Pct'] > 0]) / len(buy_signals) * 100
            else:
                performance_metrics['buy_profitable_pct'] = 0
                
            if not sell_signals.empty:
                performance_metrics['sell_profitable_pct'] = len(sell_signals[sell_signals['Profit_Loss_Pct'] > 0]) / len(sell_signals) * 100
            else:
                performance_metrics['sell_profitable_pct'] = 0
            
            # Signal counts
            performance_metrics['buy_count'] = len(buy_signals)
            performance_metrics['sell_count'] = len(sell_signals)
            performance_metrics['neutral_count'] = len(neutral_signals)
            
            # Average days in signal
            performance_metrics['buy_avg_days'] = buy_signals['Days_In_Signal'].mean() if not buy_signals.empty else 0
            performance_metrics['sell_avg_days'] = sell_signals['Days_In_Signal'].mean() if not sell_signals.empty else 0
        
        return new_transitions, performance_metrics
    
    except Exception as e:
        logging.error(f"Error detecting signal transitions: {str(e)}")
        return [], {}

def detect_high_profit_signals(db_path, profit_threshold=5.0, loss_threshold=-5.0):
    """Detect signals with high profit or significant loss"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Get high profit/loss signals
        high_profit_signals = pd.read_sql(f'''
            SELECT Stock, Current_Signal, Profit_Loss_Pct, Days_In_Signal
            FROM signal_tracking
            WHERE Profit_Loss_Pct > {profit_threshold} OR Profit_Loss_Pct < {loss_threshold}
            ORDER BY Profit_Loss_Pct DESC
        ''', conn)
        
        conn.close()
        return high_profit_signals
    
    except Exception as e:
        logging.error(f"Error detecting high profit signals: {str(e)}")
        return pd.DataFrame()

def detect_long_duration_signals(db_path, days_threshold=14):
    """Detect signals that have been active for a long duration"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Get long duration signals
        long_duration = pd.read_sql(f'''
            SELECT Stock, Current_Signal, Profit_Loss_Pct, Days_In_Signal
            FROM signal_tracking
            WHERE Days_In_Signal > {days_threshold}
            ORDER BY Days_In_Signal DESC
        ''', conn)
        
        conn.close()
        return long_duration
    
    except Exception as e:
        logging.error(f"Error detecting long duration signals: {str(e)}")
        return pd.DataFrame()

def generate_advanced_analysis(db_path, output_dir, config):
    """Generate advanced analysis beyond the basic reports"""
    try:
        conn = sqlite3.connect(db_path)
        
        analysis_results = {}
        
        # Additional analysis based on config
        if config.get('analysis', {}).get('trend_detection', True):
            # Detect trend patterns
            trend_analysis = pd.read_sql('''
                SELECT 
                    Current_Signal,
                    COUNT(*) as signal_count,
                    AVG(Profit_Loss_Pct) as avg_profit,
                    AVG(Days_In_Signal) as avg_days
                FROM signal_tracking
                GROUP BY Current_Signal
            ''', conn)
            
            analysis_results['trend_analysis'] = trend_analysis
        
        if config.get('analysis', {}).get('performance_metrics', True):
            # Performance by days in signal buckets
            performance_by_days = pd.read_sql('''
                SELECT 
                    Current_Signal,
                    CASE 
                        WHEN Days_In_Signal < 7 THEN '0-6 days'
                        WHEN Days_In_Signal < 14 THEN '7-13 days'
                        WHEN Days_In_Signal < 30 THEN '14-29 days'
                        ELSE '30+ days'
                    END as days_bucket,
                    COUNT(*) as count,
                    AVG(Profit_Loss_Pct) as avg_profit
                FROM signal_tracking
                GROUP BY Current_Signal, days_bucket
                ORDER BY Current_Signal, days_bucket
            ''', conn)
            
            analysis_results['performance_by_days'] = performance_by_days
        
        # Save analysis results
        for name, df in analysis_results.items():
            if not df.empty:
                output_file = os.path.join(output_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.csv')
                df.to_csv(output_file, index=False)
                logging.info(f"Advanced analysis saved to {output_file}")
        
        conn.close()
        return analysis_results
    
    except Exception as e:
        logging.error(f"Error generating advanced analysis: {str(e)}")
        return {}

def cleanup_backup(db_path):
    """Clean up backup files after successful execution"""
    try:
        # Get the directory and base filename
        db_dir = os.path.dirname(db_path)
        db_name = os.path.basename(db_path)
        
        # Find backup files created today
        backup_pattern = os.path.join(db_dir, f"backup_{db_name}_*")
        today_str = datetime.now().strftime("%Y%m%d")
        
        for backup_file in glob.glob(backup_pattern):
            # Check if this backup was created today
            if today_str in backup_file:
                logging.info(f"Removing backup file: {backup_file}")
                os.remove(backup_file)
                logging.info("Backup file cleaned up successfully")
    except Exception as e:
        logging.warning(f"Error cleaning up backup file: {str(e)}")

def run_tracker(db_path=None, create_backup=True, generate_report=True, visualize=True, 
                output_dir=None, cleanup_after_success=True, send_alerts=True):
    """Run the stock signal tracker with advanced analysis and alerts"""
    start_time = time.time()
    try:
        # Load configuration
        config = load_config()
        
        # Set default values if not provided
        if not db_path:
            db_path = '/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_investing_Stocks_KMI30_tracking.db'
        
        if not output_dir:
            # Create reports directory in project root
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'reports', 
                f'signal_tracking_{datetime.now().strftime("%Y%m%d")}'
            )
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare command-line arguments
        cmd_args = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_processing/stock_signal_tracker.py'),
            f'--db={db_path}'
        ]
        
        if create_backup:
            cmd_args.append('--backup')
        
        if generate_report:
            report_path = os.path.join(output_dir, f'stock_signal_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            cmd_args.append(f'--report={report_path}')
        
        if visualize:
            cmd_args.append('--visualize')
            cmd_args.append(f'--output-dir={output_dir}')
        
        # Run the tracker
        cmd = ' '.join(cmd_args)
        logging.info(f"Running command: {cmd}")
        
        import subprocess
        process = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        
        # Check if process was successful
        success = process.returncode == 0
        
        if success:
            logging.info("Stock signal tracker completed successfully")
            
            # Print the output
            if process.stdout:
                print(process.stdout)
            
            if process.stderr and process.stderr.strip():
                logging.warning(f"Stderr output: {process.stderr}")
            
            # Detect signal transitions and generate alerts
            if send_alerts and config.get('alerts', {}).get('signal_transitions', True):
                transitions, performance_metrics = detect_signal_transitions(db_path)
                
                if transitions:
                    # Generate and send Telegram alert using existing module
                    telegram_message = format_signals_for_telegram(transitions, performance_metrics)
                    if telegram_message:
                        send_telegram_message(telegram_message)
                    
                    # Log transitions
                    logging.info(f"Detected {len(transitions)} signal transitions:")
                    for t in transitions:
                        logging.info(f"  {t['Stock']}: {t['Previous_Signal']} -> {t['Current_Signal']}")
            
            # Detect high profit/loss signals
            profit_threshold = config.get('alerts', {}).get('profit_threshold', 5.0)
            loss_threshold = config.get('alerts', {}).get('loss_threshold', -5.0)
            high_profit_signals = detect_high_profit_signals(db_path, profit_threshold, loss_threshold)
            
            if not high_profit_signals.empty:
                logging.info(f"Detected {len(high_profit_signals)} high profit/loss signals")
                # Save to CSV
                high_profit_file = os.path.join(output_dir, f'high_profit_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                high_profit_signals.to_csv(high_profit_file, index=False)
                
                # Send Telegram alert for high profit signals
                if send_alerts:
                    high_profit_message = format_high_profit_for_telegram(high_profit_signals)
                    if high_profit_message:
                        send_telegram_message(high_profit_message)
            
            # Detect long duration signals
            days_threshold = config.get('alerts', {}).get('days_in_signal_threshold', 14)
            long_duration_signals = detect_long_duration_signals(db_path, days_threshold)
            
            if not long_duration_signals.empty:
                logging.info(f"Detected {len(long_duration_signals)} long duration signals")
                # Save to CSV
                duration_file = os.path.join(output_dir, f'long_duration_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                long_duration_signals.to_csv(duration_file, index=False)
                
                # Send Telegram alert for long duration signals
                if send_alerts:
                    duration_message = format_high_profit_for_telegram(long_duration_signals, "Long Duration")
                    if duration_message:
                        send_telegram_message(duration_message)
            
            # Generate advanced analysis
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(generate_advanced_analysis, db_path, output_dir, config)
                analysis_results = future.result()
            
            # Clean up backup if requested
            if cleanup_after_success and create_backup:
                logging.info("Cleaning up backup...")
                cleanup_backup(db_path)
        else:
            logging.error(f"Stock signal tracker failed with return code {process.returncode}")
            if process.stderr:
                logging.error(f"Error output: {process.stderr}")
        
        # Log execution time
        execution_time = time.time() - start_time
        logging.info(f"Total execution time: {execution_time:.2f} seconds")
        
        return success
    
    except Exception as e:
        execution_time = time.time() - start_time
        logging.error(f"Error running stock signal tracker: {str(e)}")
        logging.info(f"Execution time before error: {execution_time:.2f} seconds")
        return False

def main():
    """Main function to run the stock signal tracker"""
    parser = argparse.ArgumentParser(description='Run the stock signal tracker with advanced analytics')
    parser.add_argument('--db', type=str, 
                        help='Path to the SQLite database')
    parser.add_argument('--no-backup', action='store_true', 
                        help='Skip creating a backup of the database')
    parser.add_argument('--no-report', action='store_true', 
                        help='Skip generating a report')
    parser.add_argument('--no-visualize', action='store_true', 
                        help='Skip generating visualizations')
    parser.add_argument('--output-dir', type=str, 
                        help='Directory to save reports and visualizations')
    parser.add_argument('--keep-backup', action='store_true',
                        help='Keep backup file even after successful execution')
    parser.add_argument('--no-alerts', action='store_true',
                        help='Disable alerts for signal transitions')
    parser.add_argument('--config', type=str,
                        help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # If custom config file is provided, set the global CONFIG_FILE
    if args.config:
        global CONFIG_FILE
        CONFIG_FILE = args.config
    
    success = run_tracker(
        db_path=args.db,
        create_backup=not args.no_backup,
        generate_report=not args.no_report,
        visualize=not args.no_visualize,
        output_dir=args.output_dir,
        cleanup_after_success=not args.keep_backup,
        send_alerts=not args.no_alerts
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 