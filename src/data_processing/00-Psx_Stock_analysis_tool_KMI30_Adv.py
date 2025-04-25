import subprocess
import os
import sys
import logging
import schedule
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def run_signal_tracker():
    """Run the stock signal tracker with advanced analysis and Telegram alerts"""
    try:
        logging.info("Running stock signal tracker with Telegram alerts")
        
        # Get the path to the run_stock_signal_tracker.py script
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tracker_script = os.path.join(script_dir, "scripts", "run_stock_signal_tracker.py")
        
        # Create reports directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        reports_dir = os.path.join(script_dir, "reports", f"signal_tracking_{timestamp}")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Set path to database
        db_path = os.path.join(script_dir, "data", "databases", "production", "PSX_investing_Stocks_KMI30_tracking.db")
        
        # Run the tracker script with enhanced parameters
        cmd = [
            sys.executable, 
            tracker_script,
            f"--db={db_path}",
            f"--output-dir={reports_dir}",
            "--create-backup",
            "--generate-report",
            "--send-alerts"
        ]
        
        logging.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log output
        if result.stdout:
            logging.info(f"Stock signal tracker output: {result.stdout[:500]}...")
        
        if result.stderr:
            logging.warning(f"Stock signal tracker errors: {result.stderr}")
        
        if result.returncode != 0:
            logging.error(f"Stock signal tracker failed with return code {result.returncode}")
        else:
            logging.info("Stock signal tracker completed successfully")
        
        # Add to summary notification
        if result.returncode == 0:
            try:
                from src.data_processing.telegram_message import send_telegram_message
                
                summary = f"🔄 PSX Analysis Batch Job - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                summary += f"✅ Stock signal tracker executed successfully\n"
                summary += f"📊 Reports saved to: {os.path.basename(reports_dir)}\n"
                send_telegram_message(summary)
            except Exception as e:
                logging.error(f"Error sending telegram summary: {str(e)}")
            
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Error running stock signal tracker: {str(e)}")
        return False

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
        
        start_time = time.time()
        
        # Run the analysis scripts
        run_scripts(scripts)
        
        # Run the stock signal tracker after all other scripts
        run_signal_tracker()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logging.info(f"PSX analysis job completed successfully in {execution_time:.2f} seconds")
        
        # Send completion notification
        try:
            from src.data_processing.telegram_message import send_telegram_message
            completion_msg = f"✅ PSX Stock Analysis Batch Job Completed\n"
            completion_msg += f"⏱️ Total execution time: {execution_time:.2f} seconds\n"
            completion_msg += f"🕒 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            send_telegram_message(completion_msg)
        except Exception as e:
            logging.error(f"Error sending completion notification: {str(e)}")
            
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

    