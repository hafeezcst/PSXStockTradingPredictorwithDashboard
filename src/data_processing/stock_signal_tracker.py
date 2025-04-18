import sqlite3
import pandas as pd
import logging
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_signal_tracker.log'),
        logging.StreamHandler()
    ]
)

class StockSignalTracker:
    def __init__(self, db_path, create_backup=True):
        """
        Initialize the Stock Signal Tracker
        
        Args:
            db_path (str): Path to the SQLite database
            create_backup (bool): Whether to create a backup of the database
        """
        self.db_path = db_path
        
        # Create a backup of the database if requested
        if create_backup:
            self._create_backup()
        
        # Connect to the database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Initialize tracking table if it doesn't exist
        self._initialize_tracking_table()
    
    def _create_backup(self):
        """Create a backup of the database"""
        backup_dir = os.path.dirname(self.db_path)
        backup_filename = f"backup_{os.path.basename(self.db_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        try:
            shutil.copy2(self.db_path, backup_path)
            logging.info(f"Created backup at {backup_path}")
        except Exception as e:
            logging.error(f"Failed to create backup: {str(e)}")
    
    def _initialize_tracking_table(self):
        """Initialize the signal tracking table if it doesn't exist"""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS "signal_tracking" (
            "id" INTEGER PRIMARY KEY AUTOINCREMENT,
            "Stock" TEXT,
            "Initial_Signal" TEXT,
            "Initial_Signal_Date" TEXT,
            "Initial_Close" REAL,
            "Current_Signal" TEXT,
            "Current_Signal_Date" TEXT,
            "Current_Close" REAL,
            "Days_In_Signal" INTEGER,
            "Total_Days" INTEGER,
            "Profit_Loss_Pct" REAL,
            "Signal_Changes" INTEGER,
            "Last_Updated" TEXT,
            "Notes" TEXT
        )
        ''')
        self.conn.commit()
        
        # Check if Total_Days column exists, add it if it doesn't
        try:
            self.cursor.execute("SELECT Total_Days FROM signal_tracking LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            self.cursor.execute("ALTER TABLE signal_tracking ADD COLUMN Total_Days INTEGER DEFAULT 0")
            self.conn.commit()
            logging.info("Added Total_Days column to signal_tracking table")
    
    def get_latest_signals(self):
        """Get the latest signals for all stocks from all three tables"""
        # First, let's check the data in source tables
        logging.info("Checking source table data...")
        
        # Check buy_stocks table
        buy_check = pd.read_sql('''
        SELECT 
            Stock,
            Date,
            Close,
            COUNT(*) as count
        FROM buy_stocks
        GROUP BY Stock
        ORDER BY Stock
        ''', self.conn)
        logging.info(f"Buy stocks data check:\n{buy_check.to_string()}")
        
        # Get buy signals with proper historical tracking
        buy_df = pd.read_sql('''
        WITH FirstSignals AS (
            SELECT 
                Stock,
                MIN(Date) as First_Signal_Date,
                Close as First_Close
            FROM buy_stocks
            GROUP BY Stock
        ),
        LatestSignals AS (
            SELECT 
                Stock,
                MAX(Date) as Latest_Date
            FROM buy_stocks
            GROUP BY Stock
        )
        SELECT 
            b.Stock,
            'Buy' as Signal,
            b.Date as Signal_Date,
            b.Close as Current_Close,
            fs.First_Signal_Date as Original_Signal_Date,
            fs.First_Close as Original_Close
        FROM buy_stocks b
        JOIN FirstSignals fs ON b.Stock = fs.Stock
        JOIN LatestSignals ls ON b.Stock = ls.Stock AND b.Date = ls.Latest_Date
        ''', self.conn)
        
        logging.info(f"Buy signals after processing:\n{buy_df[['Stock', 'Original_Signal_Date', 'Original_Close', 'Current_Close']].to_string()}")
        
        # Get sell signals with proper historical tracking
        sell_df = pd.read_sql('''
        WITH FirstSignals AS (
            SELECT 
                Stock,
                MIN(Date) as First_Signal_Date,
                Close as First_Close
            FROM sell_stocks
            GROUP BY Stock
        ),
        LatestSignals AS (
            SELECT 
                Stock,
                MAX(Date) as Latest_Date
            FROM sell_stocks
            GROUP BY Stock
        )
        SELECT 
            s.Stock,
            'Sell' as Signal,
            s.Date as Signal_Date,
            s.Close as Current_Close,
            fs.First_Signal_Date as Original_Signal_Date,
            fs.First_Close as Original_Close
        FROM sell_stocks s
        JOIN FirstSignals fs ON s.Stock = fs.Stock
        JOIN LatestSignals ls ON s.Stock = ls.Stock AND s.Date = ls.Latest_Date
        ''', self.conn)
        
        # Neutral signals (no historical tracking)
        neutral_df = pd.read_sql('''
        SELECT 
            Stock,
            'Neutral' as Signal,
            Date as Signal_Date,
            Close as Current_Close,
            NULL as Original_Signal_Date,
            NULL as Original_Close
        FROM neutral_stocks
        WHERE Date = (SELECT MAX(Date) FROM neutral_stocks)
        ''', self.conn)

        # Combine all signals
        all_signals = pd.concat([buy_df, sell_df, neutral_df], ignore_index=True)
        
        # Convert numeric columns
        for col in ['Current_Close', 'Original_Close']:
            all_signals[col] = pd.to_numeric(all_signals[col], errors='coerce')
        
        # Get latest unique signals per stock
        latest_signals = all_signals.sort_values('Signal_Date', ascending=False).drop_duplicates('Stock')
        
        # Log the signals for debugging
        logging.info("Latest signals retrieved:")
        logging.info(latest_signals[['Stock', 'Signal', 'Original_Signal_Date', 
                                   'Original_Close', 'Current_Close']].to_string())
        
        return latest_signals
    
    def update_tracking_table(self):
        """Update the tracking table with the latest signals"""
        try:
            latest_signals = self.get_latest_signals()
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get existing tracked stocks
            tracked_stocks = pd.read_sql("SELECT Stock, Initial_Signal, Initial_Signal_Date, Initial_Close, Signal, Signal_Date, Current_Close FROM signal_tracking", self.conn)
            
            # Log the data we're working with
            logging.info(f"Processing {len(latest_signals)} signals")
            logging.info("Sample of latest signals:")
            logging.info(latest_signals[['Stock', 'Signal', 'Original_Signal_Date', 'Original_Close', 'Current_Close']].head().to_string())
            
            for _, row in latest_signals.iterrows():
                try:
                    stock = row['Stock']
                    current_signal = row['Signal']
                    current_date_str = row['Signal_Date']
                    current_close = row['Current_Close']
                    signal_date = row['Original_Signal_Date']
                    signal_close = row['Original_Close']
                    
                    # Log the data for this stock
                    logging.info(f"\nProcessing stock: {stock}")
                    logging.info(f"Current Signal: {current_signal}")
                    logging.info(f"Current Date: {current_date_str}")
                    logging.info(f"Current Close: {current_close}")
                    logging.info(f"Original Signal Date: {signal_date}")
                    logging.info(f"Original Close: {signal_close}")
                    
                    # Calculate days in signal
                    if signal_date and current_date_str:
                        try:
                            days_in_signal = (datetime.strptime(current_date_str, '%Y-%m-%d') - datetime.strptime(signal_date, '%Y-%m-%d')).days
                            logging.info(f"Days in signal for {stock}: {days_in_signal} (From {signal_date} to {current_date_str})")
                        except Exception as e:
                            logging.error(f"Error calculating days in signal for {stock}: {str(e)}")
                            days_in_signal = 0
                    else:
                        days_in_signal = 0
                        logging.warning(f"No signal date data for {stock}")
                    
                    # Calculate profit/loss percentage based on signal type
                    if current_signal == 'Buy':
                        if signal_close and current_close and signal_close != 0:
                            profit_loss = ((current_close - signal_close) / signal_close) * 100
                            logging.info(f"Buy signal profit/loss calculation for {stock}: {profit_loss}% (Current: {current_close}, Initial: {signal_close})")
                        else:
                            profit_loss = 0
                            logging.warning(f"Missing or invalid price data for Buy signal {stock}")
                    elif current_signal == 'Sell':
                        if signal_close and current_close and signal_close != 0:
                            profit_loss = ((signal_close - current_close) / signal_close) * 100
                            logging.info(f"Sell signal profit/loss calculation for {stock}: {profit_loss}% (Current: {current_close}, Initial: {signal_close})")
                        else:
                            profit_loss = 0
                            logging.warning(f"Missing or invalid price data for Sell signal {stock}")
                    else:  # Neutral
                        profit_loss = 0
                        logging.info(f"Neutral signal for {stock}: profit/loss set to 0")
                    
                    # Check if stock is already being tracked
                    if stock in tracked_stocks['Stock'].values:
                        # Get existing record
                        tracked_stock = tracked_stocks[tracked_stocks['Stock'] == stock].iloc[0]
                        prev_signal = tracked_stock['Signal']
                        
                        # Update signal changes count if signal changed
                        signal_changes = 0
                        if prev_signal != current_signal:
                            signal_changes = 1
                            logging.info(f"Signal changed for {stock}: {prev_signal} -> {current_signal}")
                        
                        # Update the tracking record
                        self.cursor.execute('''
                        UPDATE signal_tracking 
                        SET Signal = ?, 
                            Signal_Date = ?, 
                            Current_Close = ?,
                            Days_In_Signal = ?,
                            Profit_Loss_Pct = ?,
                            Signal_Changes = Signal_Changes + ?,
                            Last_Updated = ?
                        WHERE Stock = ?
                        ''', (
                            current_signal, current_date_str, current_close,
                            days_in_signal, profit_loss, signal_changes,
                            current_date, stock
                        ))
                    else:
                        # For new entries, use current values as initial values
                        self.cursor.execute('''
                        INSERT INTO signal_tracking (
                            Stock, Initial_Signal, Initial_Signal_Date, Initial_Close,
                            Signal, Signal_Date, Current_Close,
                            Days_In_Signal, Profit_Loss_Pct, Signal_Changes, Last_Updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            stock, current_signal, signal_date, signal_close,
                            current_signal, current_date_str, current_close,
                            days_in_signal, profit_loss, 0, current_date
                        ))
                except Exception as e:
                    logging.error(f"Error processing stock {stock}: {str(e)}")
                    continue
            
            self.conn.commit()
            logging.info(f"Updated tracking table with {len(latest_signals)} stocks")
            
        except Exception as e:
            logging.error(f"Error updating tracking table: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def get_active_buy_signals(self):
        """Get stocks with active buy signals"""
        return pd.read_sql('''
        SELECT 
            Stock, Initial_Signal_Date, Initial_Close, 
            Current_Close, Days_In_Signal, Total_Days,
            Profit_Loss_Pct, Signal_Changes
        FROM signal_tracking
        WHERE Current_Signal = 'Buy'
        ORDER BY Days_In_Signal ASC, Profit_Loss_Pct DESC
        ''', self.conn)
    
    def get_active_sell_signals(self):
        """Get stocks with active sell signals"""
        return pd.read_sql('''
        SELECT 
            Stock, Initial_Signal_Date, Initial_Close, 
            Current_Close, Days_In_Signal, Total_Days,
            Profit_Loss_Pct, Signal_Changes
        FROM signal_tracking
        WHERE Current_Signal = 'Sell'
        ORDER BY Days_In_Signal ASC, Profit_Loss_Pct DESC
        ''', self.conn)
    
    def get_all_active_signals(self):
        """Get all active buy, sell and neutral signals directly from source tables"""
        # Get active buy signals directly from buy_stocks table
        buy_signals = pd.read_sql('''
        SELECT 
            Stock, 
            Signal_Date, 
            Signal_Close,
            Date as Current_Date,
            Close as Current_Close,
            (julianday(Date) - julianday(Signal_Date)) as Days_In_Signal,
            ((Close - Signal_Close) / Signal_Close * 100) as Profit_Loss_Pct,
            'Buy' as Signal_Type
        FROM buy_stocks
        ORDER BY Stock
        ''', self.conn)
        
        # Get active sell signals directly from sell_stocks table
        sell_signals = pd.read_sql('''
        SELECT 
            Stock, 
            Signal_Date, 
            Signal_Close,
            Date as Current_Date,
            Close as Current_Close,
            (julianday(Date) - julianday(Signal_Date)) as Days_In_Signal,
            ((Signal_Close - Close) / Signal_Close * 100) as Profit_Loss_Pct,
            'Sell' as Signal_Type
        FROM sell_stocks
        ORDER BY Stock
        ''', self.conn)
        
        # Get neutral signals directly from neutral_stocks table
        # Note: Neutral signals don't have Signal_Date and Signal_Close, so we use Date as both
        neutral_signals = pd.read_sql('''
        SELECT 
            Stock, 
            Date as Signal_Date, 
            Close as Signal_Close,
            Date as Current_Date,
            Close as Current_Close,
            0 as Days_In_Signal,
            0 as Profit_Loss_Pct,
            'Neutral' as Signal_Type,
            Trend_Direction
        FROM neutral_stocks
        ORDER BY Stock
        ''', self.conn)
        
        return buy_signals, sell_signals, neutral_signals
    
    def get_signal_transitions(self):
        """Get stocks that have transitioned between signals"""
        return pd.read_sql('''
        SELECT 
            Stock, Initial_Signal, Initial_Signal_Date, Initial_Close,
            Current_Signal, Current_Signal_Date, Current_Close,
            Days_In_Signal, Total_Days, Profit_Loss_Pct, Signal_Changes, Notes
        FROM signal_tracking
        WHERE Initial_Signal != Current_Signal OR Signal_Changes > 0
        ORDER BY Signal_Changes DESC, Profit_Loss_Pct DESC
        ''', self.conn)
    
    def analyze_signal_performance(self):
        """Analyze the performance of signals"""
        # Get all tracked signals
        all_tracked = pd.read_sql('''
        SELECT 
            Stock, Initial_Signal, Initial_Signal_Date, Initial_Close,
            Current_Signal, Current_Signal_Date, Current_Close,
            Days_In_Signal, Total_Days, Profit_Loss_Pct, Signal_Changes
        FROM signal_tracking
        ''', self.conn)
        
        if all_tracked.empty:
            logging.warning("No data available for performance analysis")
            return None
        
        # Calculate average profit/loss by current signal type
        signal_performance = all_tracked.groupby('Current_Signal')['Profit_Loss_Pct'].agg(
            ['mean', 'median', 'std', 'count']
        ).reset_index()
        
        # Calculate average holding days by current signal type
        holding_days = all_tracked.groupby('Current_Signal')['Days_In_Signal'].agg(
            ['mean', 'median', 'max', 'count']
        ).reset_index()
        
        # Calculate signal change frequency
        signal_changes = all_tracked.groupby('Current_Signal')['Signal_Changes'].agg(
            ['mean', 'max', 'sum']
        ).reset_index()
        
        # Combine the analyses
        performance_analysis = {
            'signal_performance': signal_performance,
            'holding_days': holding_days,
            'signal_changes': signal_changes
        }
        
        return performance_analysis
    
    def visualize_performance(self, output_dir='./reports'):
        """Generate visualizations of signal performance"""
        performance = self.analyze_signal_performance()
        
        if not performance:
            logging.warning("No data available for visualization")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all tracked signals
        all_tracked = pd.read_sql('''
        SELECT 
            Stock, Initial_Signal, Initial_Signal_Date, Initial_Close,
            Current_Signal, Current_Signal_Date, Current_Close,
            Days_In_Signal, Total_Days, Profit_Loss_Pct, Signal_Changes
        FROM signal_tracking
        ''', self.conn)
        
        # Set the style
        sns.set(style="whitegrid")
        
        # Plot 1: Profit/Loss Distribution by Current Signal
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Current_Signal', y='Profit_Loss_Pct', data=all_tracked)
        plt.title('Profit/Loss Distribution by Current Signal Type')
        plt.ylabel('Profit/Loss Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'profit_loss_distribution.png'))
        plt.close()
        
        # Plot 2: Signal Holding Period
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Current_Signal', y='Days_In_Signal', data=all_tracked)
        plt.title('Current Signal Holding Period')
        plt.ylabel('Days in Current Signal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'holding_period.png'))
        plt.close()
        
        # Plot 3: Signal Transitions
        signal_transitions = all_tracked.copy()
        signal_transitions['Transition'] = signal_transitions['Initial_Signal'] + ' → ' + signal_transitions['Current_Signal']
        
        plt.figure(figsize=(14, 8))
        transition_counts = signal_transitions['Transition'].value_counts()
        transition_counts.plot(kind='bar')
        plt.title('Signal Transition Counts')
        plt.ylabel('Number of Stocks')
        plt.xlabel('Transition Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'signal_transitions.png'))
        plt.close()
        
        # Plot 4: Days in Signal vs Profit/Loss %
        plt.figure(figsize=(14, 8))
        sns.scatterplot(x='Days_In_Signal', y='Profit_Loss_Pct', hue='Current_Signal', data=all_tracked)
        plt.title('Days in Current Signal vs Profit/Loss %')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'days_vs_profit.png'))
        plt.close()
        
        # Plot 5: Cumulative profit/loss over time for Buy signals
        if 'Buy' in all_tracked['Current_Signal'].values:
            buy_signals = all_tracked[all_tracked['Current_Signal'] == 'Buy'].copy()
            if not buy_signals.empty and buy_signals['Days_In_Signal'].max() > 0:
                buy_signals = buy_signals.sort_values('Days_In_Signal')
                plt.figure(figsize=(14, 8))
                plt.plot(buy_signals['Days_In_Signal'], buy_signals['Profit_Loss_Pct'].cumsum())
                plt.title('Cumulative Profit/Loss % Over Time (Buy Signals)')
                plt.xlabel('Days in Signal')
                plt.ylabel('Cumulative Profit/Loss %')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'cumulative_profit_buy.png'))
                plt.close()
                
        # Plot 6: Signal Holding Days Distribution
        plt.figure(figsize=(14, 8))
        sns.histplot(data=all_tracked, x='Days_In_Signal', hue='Current_Signal', multiple='stack', bins=20)
        plt.title('Distribution of Days in Current Signal')
        plt.xlabel('Days in Signal')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'days_distribution.png'))
        plt.close()
        
        logging.info(f"Visualizations saved to {output_dir}")
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive report of stock signals"""
        # Get data for report
        active_buy_signals = self.get_active_buy_signals()
        active_sell_signals = self.get_active_sell_signals()
        signal_transitions = self.get_signal_transitions()
        performance = self.analyze_signal_performance()
        
        # Get signals directly from source tables
        direct_buy_signals, direct_sell_signals, direct_neutral_signals = self.get_all_active_signals()
        
        # Format the report
        report = []
        report.append("=" * 80)
        report.append(f"STOCK SIGNAL TRACKING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Direct Buy Signals from Source Table
        report.append("-" * 80)
        report.append("BUY SIGNALS DIRECTLY FROM SOURCE TABLE")
        report.append("-" * 80)
        if not direct_buy_signals.empty:
            # Sort by days in signal (ascending) - latest signals first
            direct_buy_signals_sorted = direct_buy_signals.sort_values('Days_In_Signal', ascending=True)
            report.append(tabulate(direct_buy_signals_sorted, headers='keys', tablefmt='psql', showindex=False))
            
            # Add summary statistics
            report.append("")
            report.append("DIRECT BUY SIGNALS SUMMARY:")
            report.append(f"Total Buy Signals: {len(direct_buy_signals)}")
            report.append(f"Average Profit/Loss: {direct_buy_signals['Profit_Loss_Pct'].mean():.2f}%")
            report.append(f"Average Days in Signal: {direct_buy_signals['Days_In_Signal'].mean():.1f} days")
            
            # Profitable vs Unprofitable signals
            profitable = direct_buy_signals[direct_buy_signals['Profit_Loss_Pct'] > 0]
            if len(direct_buy_signals) > 0:
                report.append(f"Profitable Signals: {len(profitable)} ({len(profitable)/len(direct_buy_signals)*100:.1f}%)")
            
                if len(profitable) > 0:
                    report.append(f"Average Profit (profitable signals): {profitable['Profit_Loss_Pct'].mean():.2f}%")
        else:
            report.append("No buy signals found in source table.")
        report.append("")
        
        # Direct Sell Signals from Source Table
        report.append("-" * 80)
        report.append("SELL SIGNALS DIRECTLY FROM SOURCE TABLE")
        report.append("-" * 80)
        if not direct_sell_signals.empty:
            # Sort by days in signal (ascending) - latest signals first
            direct_sell_signals_sorted = direct_sell_signals.sort_values('Days_In_Signal', ascending=True)
            report.append(tabulate(direct_sell_signals_sorted, headers='keys', tablefmt='psql', showindex=False))
            
            # Add summary statistics
            report.append("")
            report.append("DIRECT SELL SIGNALS SUMMARY:")
            report.append(f"Total Sell Signals: {len(direct_sell_signals)}")
            report.append(f"Average Profit/Loss: {direct_sell_signals['Profit_Loss_Pct'].mean():.2f}%")
            report.append(f"Average Days in Signal: {direct_sell_signals['Days_In_Signal'].mean():.1f} days")
            
            # Profitable vs Unprofitable signals
            profitable = direct_sell_signals[direct_sell_signals['Profit_Loss_Pct'] > 0]
            if len(direct_sell_signals) > 0:
                report.append(f"Profitable Signals: {len(profitable)} ({len(profitable)/len(direct_sell_signals)*100:.1f}%)")
            
                if len(profitable) > 0:
                    report.append(f"Average Profit (profitable signals): {profitable['Profit_Loss_Pct'].mean():.2f}%")
        else:
            report.append("No sell signals found in source table.")
        report.append("")
        
        # Direct Neutral Signals from Source Table
        report.append("-" * 80)
        report.append("NEUTRAL SIGNALS DIRECTLY FROM SOURCE TABLE")
        report.append("-" * 80)
        if not direct_neutral_signals.empty:
            # Group by trend direction
            bullish_count = len(direct_neutral_signals[direct_neutral_signals['Trend_Direction'] == 'Bullish'])
            bearish_count = len(direct_neutral_signals[direct_neutral_signals['Trend_Direction'] == 'Bearish'])
            
            # Display signals
            report.append(tabulate(direct_neutral_signals, headers='keys', tablefmt='psql', showindex=False))
            
            # Add summary statistics
            report.append("")
            report.append("DIRECT NEUTRAL SIGNALS SUMMARY:")
            report.append(f"Total Neutral Signals: {len(direct_neutral_signals)}")
            report.append(f"Bullish Trend: {bullish_count} stocks ({bullish_count/len(direct_neutral_signals)*100:.1f}%)")
            report.append(f"Bearish Trend: {bearish_count} stocks ({bearish_count/len(direct_neutral_signals)*100:.1f}%)")
        else:
            report.append("No neutral signals found in source table.")
        report.append("")
        
        # Active Buy Signals from Tracking Table
        report.append("-" * 80)
        report.append("ACTIVE BUY SIGNALS FROM TRACKING")
        report.append("-" * 80)
        if not active_buy_signals.empty:
            # Sort by days in signal (ascending) - latest signals first
            report.append(tabulate(active_buy_signals, headers='keys', tablefmt='psql', showindex=False))
            
            # Add summary statistics for active buy signals
            report.append("")
            report.append("BUY SIGNALS SUMMARY:")
            report.append(f"Total Buy Signals: {len(active_buy_signals)}")
            report.append(f"Average Profit/Loss: {active_buy_signals['Profit_Loss_Pct'].mean():.2f}%")
            report.append(f"Average Days in Current Signal: {active_buy_signals['Days_In_Signal'].mean():.1f} days")
            report.append(f"Average Total Days: {active_buy_signals['Total_Days'].mean():.1f} days")
            
            # Profitable vs Unprofitable signals
            profitable = active_buy_signals[active_buy_signals['Profit_Loss_Pct'] > 0]
            if len(active_buy_signals) > 0:
                report.append(f"Profitable Signals: {len(profitable)} ({len(profitable)/len(active_buy_signals)*100:.1f}%)")
            
                if len(profitable) > 0:
                    report.append(f"Average Profit (profitable signals): {profitable['Profit_Loss_Pct'].mean():.2f}%")
        else:
            report.append("No active buy signals found.")
        report.append("")
        
        # Active Sell Signals from Tracking Table
        report.append("-" * 80)
        report.append("ACTIVE SELL SIGNALS FROM TRACKING")
        report.append("-" * 80)
        if not active_sell_signals.empty:
            # Sort by days in signal (ascending) - latest signals first
            report.append(tabulate(active_sell_signals, headers='keys', tablefmt='psql', showindex=False))
            
            # Add summary statistics for active sell signals
            report.append("")
            report.append("SELL SIGNALS SUMMARY:")
            report.append(f"Total Sell Signals: {len(active_sell_signals)}")
            report.append(f"Average Profit/Loss: {active_sell_signals['Profit_Loss_Pct'].mean():.2f}%")
            report.append(f"Average Days in Current Signal: {active_sell_signals['Days_In_Signal'].mean():.1f} days")
            report.append(f"Average Total Days: {active_sell_signals['Total_Days'].mean():.1f} days")
            
            # Profitable vs Unprofitable signals
            profitable = active_sell_signals[active_sell_signals['Profit_Loss_Pct'] > 0]
            if len(active_sell_signals) > 0:
                report.append(f"Profitable Signals: {len(profitable)} ({len(profitable)/len(active_sell_signals)*100:.1f}%)")
            
                if len(profitable) > 0:
                    report.append(f"Average Profit (profitable signals): {profitable['Profit_Loss_Pct'].mean():.2f}%")
        else:
            report.append("No active sell signals found.")
        report.append("")
        
        # Signal Transitions
        report.append("-" * 80)
        report.append("SIGNAL TRANSITIONS")
        report.append("-" * 80)
        if not signal_transitions.empty:
            report.append(tabulate(signal_transitions, headers='keys', tablefmt='psql', showindex=False))
            
            # Add transition statistics
            report.append("")
            report.append("TRANSITION SUMMARY:")
            for from_signal in ['Buy', 'Sell', 'Neutral']:
                for to_signal in ['Buy', 'Sell', 'Neutral']:
                    if from_signal != to_signal:
                        count = len(signal_transitions[
                            (signal_transitions['Initial_Signal'] == from_signal) & 
                            (signal_transitions['Current_Signal'] == to_signal)
                        ])
                        if count > 0:
                            report.append(f"{from_signal} → {to_signal}: {count} transitions")
        else:
            report.append("No signal transitions found.")
        report.append("")
        
        # Performance Analysis
        if performance:
            report.append("-" * 80)
            report.append("PERFORMANCE ANALYSIS")
            report.append("-" * 80)
            report.append("Signal Performance by Current Signal Type (Profit/Loss %):")
            report.append(tabulate(performance['signal_performance'], headers='keys', tablefmt='psql', showindex=False))
            report.append("")
            report.append("Holding Days by Current Signal Type:")
            report.append(tabulate(performance['holding_days'], headers='keys', tablefmt='psql', showindex=False))
            report.append("")
            report.append("Signal Changes:")
            report.append(tabulate(performance['signal_changes'], headers='keys', tablefmt='psql', showindex=False))
        
        # Join report lines
        report_text = "\n".join(report)
        
        # Print report to console
        print(report_text)
        
        # Save report to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logging.info(f"Report saved to {output_file}")
        
        return report_text
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed")


def main():
    """Main function to run the stock signal tracker"""
    parser = argparse.ArgumentParser(description='Track stock signals from database')
    parser.add_argument('--db', type=str, default='/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_investing_Stocks_KMI30_tracking.db',
                        help='Path to the SQLite database')
    parser.add_argument('--backup', action='store_true', help='Create a backup of the database')
    parser.add_argument('--report', type=str, help='Path to save the report')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--output-dir', type=str, default='./reports', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    try:
        # Initialize the tracker
        tracker = StockSignalTracker(args.db, create_backup=args.backup)
        
        # Update the tracking table
        tracker.update_tracking_table()
        
        # Generate a report
        tracker.generate_report(args.report)
        
        # Generate visualizations if requested
        if args.visualize:
            tracker.visualize_performance(args.output_dir)
        
        # Close the connection
        tracker.close()
        
        logging.info("Stock signal tracking completed successfully")
    except Exception as e:
        logging.error(f"Error in stock signal tracking: {str(e)}")


if __name__ == "__main__":
    main() 