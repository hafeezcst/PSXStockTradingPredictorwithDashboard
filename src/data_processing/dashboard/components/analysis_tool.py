"""
Analysis Tool component for the PSX dashboard.
"""

import os
import sys
import streamlit as st
import time
from datetime import datetime
from src.data_processing.dashboard.components.shared_styles import (
    apply_shared_styles,
    create_custom_header,
    create_custom_divider
)

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Define paths
BASE_DIR = project_root
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASES_DIR = os.path.join(DATA_DIR, "databases", "production")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def create_progress_bar(container, total_steps: int, current_step: int, message: str):
    """Create a progress bar with status message"""
    progress = current_step / total_steps
    container.progress(progress)
    container.markdown(f"<div style='text-align: center;'>{message}</div>", unsafe_allow_html=True)

def create_status_card(container, title: str, status: str, icon: str = "ℹ️"):
    """Create a status card with icon and message"""
    container.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
            <div style='display: flex; align-items: center;'>
                <span style='font-size: 1.5rem; margin-right: 0.5rem;'>{icon}</span>
                <div>
                    <h4 style='margin: 0;'>{title}</h4>
                    <p style='margin: 0;'>{status}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_analysis_tool_dashboard() -> None:
    """Create a dashboard for executing the PSX stock analysis tool with customization options."""
    apply_shared_styles()
    create_custom_header("PSX Stock Analysis Tool")
    create_custom_divider()
    
    # Analysis Options
    st.markdown("### Analysis Options")
    
    # Create columns for different analysis sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Analysis")
        run_basic_analysis = st.button("Run Basic Analysis", key="basic_analysis_btn")
        
        st.markdown("#### Advanced Analysis")
        run_advanced_analysis = st.button("Run Advanced Analysis", key="advanced_analysis_btn")
        
        st.markdown("#### Custom Analysis")
        custom_scripts = st.multiselect(
            "Select Scripts to Run",
            options=[
                'manual_kmi_shariah_processor.py',
                'PSXAnnouncement.py',
                '01-PSX_Database_data_download_to_SQL_db_PSX.py',
                '02-sql_duplicate_remover_ALL.py',
                '01-PSX_SQL_Indicator_PSX.py',
                '06_PSX_Dividend_Schedule.py',
                '04-List_Weekly_RSI_GT_40_BUY_SELL_KMI30_100_Weekly_v1.0_stable.py',
                '10-draw_indicator_trend_lines_with_signals_Stable_V_1.0.py'
            ],
            default=['01-PSX_Database_data_download_to_SQL_db_PSX.py', '01-PSX_SQL_Indicator_PSX.py'],
            key="custom_scripts_select"
        )
        run_custom_analysis = st.button("Run Custom Analysis", key="custom_analysis_btn")
    
    with col2:
        st.markdown("#### Analysis Settings")
        fresh_data = st.checkbox("Download Fresh Data", value=True, key="fresh_data_check")
        include_telegram = st.checkbox("Send Telegram Notifications", value=True, key="telegram_check")
        output_dir = st.text_input("Output Directory", value=REPORTS_DIR, key="output_dir_input")
        
        st.markdown("#### Schedule Analysis")
        schedule_analysis = st.checkbox("Schedule Analysis", value=False, key="schedule_check")
        if schedule_analysis:
            schedule_time = st.time_input(
                "Schedule Time", 
                value=datetime.strptime("17:30", "%H:%M").time(),
                key="schedule_time_input"
            )
            schedule_days = st.multiselect(
                "Schedule Days",
                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                key="schedule_days_select"
            )
    
    # Status and Logs Section
    st.markdown("### Analysis Status")
    status_container = st.container()
    progress_container = st.container()
    logs_container = st.container()
    
    # Execute analysis based on selection
    if run_basic_analysis or run_advanced_analysis or run_custom_analysis:
        with status_container:
            st.info("Starting analysis...")
            
            try:
                # Import the analysis tool
                from src.data_processing.psx_stock_analysis_tool_kmi30_adv import run_scripts
                from src.data_processing.dashboard.components.signal_tracker import run_signal_tracker
                
                # Set up environment variables for the analysis
                os.environ['PSX_DATA_DIR'] = DATA_DIR
                os.environ['PSX_DATABASES_DIR'] = DATABASES_DIR
                os.environ['PSX_REPORTS_DIR'] = output_dir
                
                if run_basic_analysis:
                    scripts = [
                        '01-PSX_Database_data_download_to_SQL_db_PSX.py',
                        '01-PSX_SQL_Indicator_PSX.py'
                    ]
                elif run_advanced_analysis:
                    scripts = [
                        'manual_kmi_shariah_processor.py',
                        'PSXAnnouncement.py --fresh' if fresh_data else 'PSXAnnouncement.py',
                        '01-PSX_Database_data_download_to_SQL_db_PSX.py',
                        '02-sql_duplicate_remover_ALL.py',
                        '01-PSX_SQL_Indicator_PSX.py',
                        '06_PSX_Dividend_Schedule.py',
                        '04-List_Weekly_RSI_GT_40_BUY_SELL_KMI30_100_Weekly_v1.0_stable.py',
                        '10-draw_indicator_trend_lines_with_signals_Stable_V_1.0.py'
                    ]
                else:  # Custom analysis
                    scripts = custom_scripts
                
                # Create progress tracking
                total_steps = len(scripts) + (1 if include_telegram else 0)
                current_step = 0
                
                # Run the selected scripts with progress tracking
                for script in scripts:
                    current_step += 1
                    with progress_container:
                        create_progress_bar(progress_container, total_steps, current_step, f"Running {script}...")
                        create_status_card(status_container, "Current Task", f"Executing {script}", "🔄")
                    
                    # Run the script
                    run_scripts([script])
                    
                    with status_container:
                        create_status_card(status_container, "Completed", f"Finished {script}", "✅")
                    
                    # Add a small delay to show progress
                    time.sleep(0.5)
                
                # Run signal tracker if included
                if include_telegram:
                    current_step += 1
                    with progress_container:
                        create_progress_bar(progress_container, total_steps, current_step, "Running Signal Tracker...")
                        create_status_card(status_container, "Current Task", "Running Signal Tracker", "🔄")
                    
                    run_signal_tracker()
                    
                    with status_container:
                        create_status_card(status_container, "Completed", "Signal Tracker Finished", "✅")
                
                # Show completion status
                with progress_container:
                    create_progress_bar(progress_container, total_steps, total_steps, "Analysis Completed!")
                    create_status_card(status_container, "Analysis Complete", "All tasks finished successfully!", "🎉")
                
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                with status_container:
                    create_status_card(status_container, "Error", str(e), "❌")
                st.error(f"Error during analysis: {str(e)}")
    
    # Display logs if available
    log_file = os.path.join(output_dir, 'psx_analysis.log')
    if os.path.exists(log_file):
        with logs_container:
            st.markdown("### Analysis Logs")
            with open(log_file, 'r') as f:
                logs = f.read()
                st.text_area("Logs", logs, height=200, key="logs_text_area")

def main() -> None:
    """Main function to display the analysis tool dashboard."""
    st.set_page_config(page_title="PSX Analysis Tool", layout="wide")
    create_analysis_tool_dashboard()

if __name__ == "__main__":
    main() 