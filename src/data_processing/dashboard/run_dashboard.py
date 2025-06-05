import subprocess
import sys
import os
from pathlib import Path

def run_dashboard():
    """Run the Streamlit dashboard"""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run the Streamlit app
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "main.py",
        "--server.port",
        "8502"
    ])

if __name__ == "__main__":
    run_dashboard() 