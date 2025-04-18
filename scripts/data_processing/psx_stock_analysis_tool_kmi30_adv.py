import subprocess
import sys
import os
from pathlib import Path
import logging
import importlib.util
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Define paths directly
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_LOGS_DIR = DATA_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"
CONFIG_DIR = BASE_DIR / "config"
DB_PATH = DATA_DIR / "databases" / "production" / "PSX_investing_Stocks_KMI30_tracking.db"

# Configure logging with proper path
logging.basicConfig(
    filename=str(DATA_LOGS_DIR / 'psx_stock_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_and_install_dependencies(dependencies):
    """Check if dependencies are installed and install them if missing"""
    missing = []
    for package in dependencies:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
            
    if missing:
        logging.info(f"Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing dependencies: {e}")
            return False
    return True

def run_scripts(scripts: list):
    """Execute a list of scripts in sequence"""
    # Get paths
    scripts_path = SCRIPTS_DIR / 'data_processing'
    config_path = CONFIG_DIR / 'config_dashboard.py'
    
    # Ensure directories exist
    for directory in [DATA_DIR, DATA_LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        
    for script in scripts:
        if script == 'config_dashboard.py':
            script_path = config_path
        else:
            script_path = scripts_path / script
            
        logging.info(f"Executing {script}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True
            )
            if result.returncode != 0:
                logging.error(f"{script} failed with return code {result.returncode}")
                break
        except Exception as e:
            logging.error(f"Error executing {script}: {str(e)}")
            break

def run_signal_tracker(db_path: str = str(DB_PATH)) -> bool:
    """
    Run the signal tracker analysis.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Get signal tracking data
        query = """
        WITH LatestSignals AS (
            SELECT 
                Stock,
                'Buy' as Current_Signal,
                Date as Current_Date,
                Close as Current_Close,
                Signal_Date as Initial_Date,
                Signal_Close as Initial_Close
            FROM buy_stocks
            WHERE Date = (SELECT MAX(Date) FROM buy_stocks)
            
            UNION ALL
            
            SELECT 
                Stock,
                'Sell' as Current_Signal,
                Date as Current_Date,
                Close as Current_Close,
                Signal_Date as Initial_Date,
                Signal_Close as Initial_Close
            FROM sell_stocks
            WHERE Date = (SELECT MAX(Date) FROM sell_stocks)
            
            UNION ALL
            
            SELECT 
                Stock,
                'Neutral' as Current_Signal,
                Date as Current_Date,
                Close as Current_Close,
                Date as Initial_Date,
                Close as Initial_Close
            FROM neutral_stocks
            WHERE Date = (SELECT MAX(Date) FROM neutral_stocks)
        )
        SELECT * FROM LatestSignals
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Update signal tracking table
        for _, row in df.iterrows():
            update_query = """
            INSERT OR REPLACE INTO signal_tracking (
                Stock, Current_Signal, Current_Date, Current_Close,
                Initial_Date, Initial_Close, Last_Updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            conn.execute(update_query, (
                row['Stock'], row['Current_Signal'], row['Current_Date'],
                row['Current_Close'], row['Initial_Date'], row['Initial_Close'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
        
        conn.commit()
        conn.close()
        logging.info("Signal tracker analysis completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in signal tracker analysis: {str(e)}")
        return False

def main():
    """Main function to execute the stock analysis pipeline"""
    # Define script execution order
    scripts = [
        'manual_kmi_shariah_processor.py',
        'MutualFundsFavourite.py',
        'psx_database_data_download.py',
        'sql_duplicate_remover.py',
        'psx_sql_indicator.py',
        'psx_dividend_schedule.py',
        'list_weekly_rsi.py',
        'draw_indicator_trend_lines.py',
        'config/config_dashboard.py'
    ]
    
    # Required dependencies
    dependencies = [
        'pandas',
        'sqlalchemy',
        'numpy',
        'matplotlib',
        'seaborn',
        'requests',
        'beautifulsoup4',
        'python-telegram-bot',
        'streamlit'
    ]
    
    # Install dependencies if needed
    if not check_and_install_dependencies(dependencies):
        logging.error("Failed to install required dependencies")
        return
    
    # Run scripts
    run_scripts(scripts)
    
    # Run signal tracker
    if not run_signal_tracker():
        logging.error("Signal tracker analysis failed")
        return

if __name__ == "__main__":
    main()