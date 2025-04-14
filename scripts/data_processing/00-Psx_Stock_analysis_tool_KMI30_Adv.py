import subprocess
import os
import sys
import logging
import schedule
import time
from datetime import datetime

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('psx_analysis.log'),
        logging.StreamHandler()
    ]
)

def run_scripts(scripts: list):
    """Execute a list of scripts in sequence"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for script in scripts:
        # Split script and arguments, but keep the script name intact
        script_parts = script.split()
        script_name = script_parts[0]
        script_args = script_parts[1:] if len(script_parts) > 1 else []
        
        script_path = os.path.join(script_dir, script_name)
        logging.info(f"Executing {script}")
        
        try:
            # Build command with script path and arguments
            cmd = [sys.executable, script_path] + script_args
            result = subprocess.run(
                cmd,
                check=True
            )
            if result.returncode != 0:
                logging.error(f"{script} failed with return code {result.returncode}")
                break
        except Exception as e:
            logging.error(f"Error executing {script}: {str(e)}")
            break

def job():
    """Main job function to run all scripts"""
    logging.info("Starting scheduled PSX analysis job")
    try:
        # Define script execution order
        scripts = [
            'manual_kmi_shariah_processor.py',
            'PSXAnnouncement.py --fresh',
            '01-PSX_Database_data_download_to_SQL_db_PSX.py',
            '02-sql_duplicate_remover_ALL.py',
            '01-PSX_SQL_Indicator_PSX.py',
            '06_PSX_Dividend_Schedule.py',
            '04-List_Weekly_RSI_GT_40_BUY_SELL_KMI30_100_Weekly_v1.0_stable.py',
            '10-draw_indicator_trend_lines_with_signals_Stable_V_1.0.py',
        ]
        
        run_scripts(scripts)
        logging.info("PSX analysis job completed successfully")
    except Exception as e:
        logging.error(f"Error in scheduled job: {str(e)}")

def main():
    """Main function to set up and run the scheduler"""
    logging.info("Starting PSX analysis scheduler")
    
    # Schedule the job to run every day at 17:30
    schedule.every().day.at("17:30").do(job)
    
    # Run the job immediately on startup
    job()
    
    # Keep the script running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logging.error(f"Scheduler error: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    main()

    