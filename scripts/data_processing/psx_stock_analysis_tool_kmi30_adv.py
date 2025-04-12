import subprocess
import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import logging
import importlib.util
from config.paths import (
    DATA_LOGS_DIR,
    SCRIPTS_DIR,
    CONFIG_DIR,
    DATA_DIR
)

# Configure logging with proper path
logging.basicConfig(
    filename=DATA_LOGS_DIR / 'psx_stock_analysis.log',
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

if __name__ == "__main__":
    main()