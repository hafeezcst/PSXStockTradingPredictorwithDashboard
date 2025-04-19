"""
Stock Signal Tracker Runner

This module provides functions to run the StockSignalTracker functionality,
which tracks stock trading signals.
"""

import os
import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the stock signal tracker
from src.data_processing.stock_signal_tracker import StockSignalTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_signal_tracker_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_database_tables(db_path):
    """
    Check if required tables exist in the database
    
    Args:
        db_path (str): Path to the SQLite database
        
    Returns:
        tuple: (bool, list) - Whether tables exist and list of missing tables
    """
    required_tables = ['buy_stocks', 'sell_stocks', 'neutral_stocks', 'signal_tracking']
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Check if required tables exist
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            logger.warning(f"Missing tables in database: {', '.join(missing_tables)}")
            return False, missing_tables
        
        return True, []
    except Exception as e:
        logger.error(f"Error checking database tables: {str(e)}")
        return False, required_tables
    finally:
        if 'conn' in locals():
            conn.close()

def create_missing_tables(db_path, missing_tables):
    """
    Create missing tables in the database
    
    Args:
        db_path (str): Path to the SQLite database
        missing_tables (list): List of table names to create
        
    Returns:
        bool: Whether tables were created successfully
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Define schema for each table
        schema = {
            'buy_stocks': '''
                CREATE TABLE buy_stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Stock TEXT NOT NULL,
                    Date TEXT NOT NULL,
                    Close REAL,
                    Signal_Date TEXT,
                    Signal_Close REAL,
                    Holding_Days INTEGER,
                    "% P/L" REAL,
                    Success TEXT,
                    Status TEXT,
                    Update_Date TEXT
                )
            ''',
            'sell_stocks': '''
                CREATE TABLE sell_stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Stock TEXT NOT NULL,
                    Date TEXT NOT NULL,
                    Close REAL,
                    Signal_Date TEXT,
                    Signal_Close REAL,
                    Holding_Days INTEGER,
                    "% P/L" REAL,
                    Success TEXT,
                    Status TEXT,
                    Update_Date TEXT
                )
            ''',
            'neutral_stocks': '''
                CREATE TABLE neutral_stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Stock TEXT NOT NULL,
                    Date TEXT NOT NULL,
                    Close REAL,
                    Trend_Direction TEXT,
                    RSI_Weekly_Avg REAL,
                    RSI_Weekly INTEGER,
                    RSI_Daily INTEGER,
                    Stochastic_K INTEGER,
                    Stochastic_D INTEGER,
                    AO_Value REAL,
                    AO_Signal TEXT,
                    Update_Date TEXT
                )
            ''',
            'signal_tracking': '''
                CREATE TABLE signal_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Stock TEXT,
                    Initial_Signal TEXT,
                    Initial_Signal_Date TEXT,
                    Initial_Close REAL,
                    Current_Signal TEXT,
                    Current_Signal_Date TEXT,
                    Current_Close REAL,
                    Days_In_Signal INTEGER,
                    Total_Days INTEGER,
                    Profit_Loss_Pct REAL,
                    Signal_Changes INTEGER,
                    Last_Updated TEXT,
                    Notes TEXT
                )
            '''
        }
        
        # Create each missing table
        for table in missing_tables:
            if table in schema:
                logger.info(f"Creating table: {table}")
                cursor.execute(schema[table])
            else:
                logger.warning(f"No schema defined for table: {table}")
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def run_tracker(db_path=None, create_backup=True, generate_report=True, 
               visualize=False, output_dir=None, cleanup_after_success=True, 
               send_alerts=False):
    """
    Run the stock signal tracker
    
    Args:
        db_path (str): Path to the SQLite database
        create_backup (bool): Whether to create a database backup
        generate_report (bool): Whether to generate a report
        visualize (bool): Whether to generate visualizations
        output_dir (str): Directory to save reports and visualizations
        cleanup_after_success (bool): Whether to perform cleanup after successful run
        send_alerts (bool): Whether to send alerts
        
    Returns:
        bool: Whether the tracker ran successfully
    """
    # Default database path if none provided
    if not db_path:
        db_path = os.path.join(project_root, 'data', 'databases', 'production', 
                             'PSX_investing_Stocks_KMI30_tracking.db')
    
    logger.info(f"Using database path: {db_path}")
    
    # Ensure the database exists
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Created database directory: {db_dir}")
    
    # Create an empty database if it doesn't exist
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        conn.close()
        logger.info(f"Created empty database: {db_path}")
    
    # Check and create required tables
    tables_exist, missing_tables = check_database_tables(db_path)
    if not tables_exist:
        logger.info("Creating missing tables...")
        if not create_missing_tables(db_path, missing_tables):
            logger.error("Failed to create missing tables, aborting")
            return False
    
    # Default output directory if none provided
    if output_dir is None:
        output_dir = os.path.join(project_root, 'reports', datetime.now().strftime('%Y%m%d'))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = None
    if generate_report:
        report_path = os.path.join(output_dir, f"signal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    try:
        # Run the tracker
        logger.info(f"Running stock signal tracker on database: {db_path}")
        tracker = StockSignalTracker(db_path, create_backup=create_backup)
        
        try:
            # Update the tracking table
            tracker.update_tracking_table()
            logger.info("Updated tracking table successfully")
        except Exception as e:
            logger.warning(f"Error updating tracking table (this is expected for empty databases): {str(e)}")
            # Continue with the rest of the process even if this fails
        
        try:
            # Generate report if requested
            if generate_report:
                tracker.generate_report(report_path)
                logger.info(f"Generated report: {report_path}")
        except Exception as e:
            logger.warning(f"Error generating report: {str(e)}")
        
        try:
            # Generate visualizations if requested
            if visualize:
                viz_dir = os.path.join(output_dir, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                tracker.visualize_performance(viz_dir)
                logger.info(f"Generated visualizations in: {viz_dir}")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {str(e)}")
        
        # Send alerts if requested
        if send_alerts and report_path:
            try:
                from src.data_processing.telegram_message import send_telegram_message
                
                # Read the report summary
                with open(report_path, 'r') as f:
                    report_text = f.read()
                
                # Extract summary info (first 40 lines)
                report_summary = '\n'.join(report_text.split('\n')[:40])
                
                # Send the alert
                send_telegram_message(f"📊 Stock Signal Tracker Report Summary:\n\n{report_summary}...")
                logger.info("Sent alert with report summary")
            except Exception as e:
                logger.warning(f"Error sending alerts: {str(e)}")
        
        # Close the tracker
        tracker.close()
        
        # Perform cleanup if requested
        if cleanup_after_success:
            # TODO: Implement cleanup logic if needed
            pass
        
        logger.info("Stock signal tracker completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running stock signal tracker: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Run the tracker with default parameters
    run_tracker(visualize=True, send_alerts=True) 