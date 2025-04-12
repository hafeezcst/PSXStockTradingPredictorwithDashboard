import subprocess
import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_scripts(scripts: list):
    """Execute a list of scripts in sequence"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for script in scripts:
        # Split script and arguments
        script_parts = script.split()
        script_name = script_parts[0]
        script_args = script_parts[1:] if len(script_parts) > 1 else []
        
        script_path = os.path.join(script_dir, script_name)
        logging.info(f"Executing {script}")
        
        try:
            # Build command with arguments
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

def main():
    # Define script execution order
    scripts = [
        'manual_kmi_shariah_processor.py',
        'PSXAnnouncement.py --fresh',
        '01-PSX_Database_data_download_to_SQL_db_PSX.py',
        '02-sql_duplicate_remover_ALL.py',
        '01-PSX_SQL_Indicator_PSX.py',
       
        #'04-portfolio_analysis.py',
        '06- PSX_Divedend_Schedule.py',
        '04-List_Weekly_RSI_GT_40_BUY_SELL_KMI30_100_Weekly_v1.0_stable.py',
        '10-draw_indicator_trend_lines_with_signals_Stable_V_1.0.py',
        #'08-PSX_MultipleTables_to_Single_Table.py'
    ]
    
    run_scripts(scripts)

if __name__ == "__main__":
    main()

    