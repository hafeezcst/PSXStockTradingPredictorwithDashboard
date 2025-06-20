"""
PSX Data Download Launcher

Simple launcher script for the enhanced PSX data download system.
This script provides an easy way to run the data download process.
"""

import sys
import os
import logging
from datetime import datetime

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """Main launcher function"""
    print("[LAUNCH] PSX Data Download System - Enhanced Version 2.0")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import and run the enhanced data reader
        from enhanced_psx_data_reader import EnhancedDataReader
        
        print("Initializing Enhanced Data Reader...")
        
        # Execute the main data download process
        import runpy
        runpy.run_module('enhanced_psx_data_reader', run_name='__main__')
        
    except ImportError as e:
        print(f"[X] Import Error: {e}")
        print("Please ensure all required modules are properly installed.")
        return 1
    except FileNotFoundError as e:
        print(f"[X] File Not Found: {e}")
        print("Please ensure the enhanced_psx_data_reader.py file exists.")
        return 1
    except Exception as e:
        print(f"[X] Unexpected Error: {e}")
        logging.error(f"Launcher error: {e}")
        return 1
    
    print(f"\n[OK] Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
