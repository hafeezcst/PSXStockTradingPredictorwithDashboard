#!/usr/bin/env python3
"""
PSX Indicator Processor - Usage Guide & Launcher

This script helps you choose the right processor for your needs and
provides clear guidance on database requirements.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

# ANSI color codes
COLORS = {
    'HEADER': '\033[95m',
    'OKBLUE': '\033[94m',
    'OKCYAN': '\033[96m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def print_header():
    """Print the main header with color."""
    print(f"{COLORS['HEADER']}{'=' * 80}{COLORS['ENDC']}")
    print(f"{COLORS['BOLD']}PSX INDICATOR PROCESSOR - LAUNCHER & GUIDE{COLORS['ENDC']}")
    print(f"{COLORS['HEADER']}{'=' * 80}{COLORS['ENDC']}")

def print_options():
    """Print available processing options with color."""
    print(f"\n{COLORS['BOLD']}AVAILABLE OPTIONS:{COLORS['ENDC']}\n")
    
    print(f"{COLORS['OKGREEN']}1. DEMO MODE (No Database Required){COLORS['ENDC']}")
    print(f"   - File: {COLORS['OKBLUE']}demo_processor.py{COLORS['ENDC']}")
    print("   - Purpose: Showcase indicator calculations with sample data")
    print("   - Requirements: None (generates sample data)")
    print(f"   - Command: {COLORS['OKCYAN']}python demo_processor.py{COLORS['ENDC']}")
    
    print(f"\n{COLORS['OKGREEN']}2. SIMPLE PROCESSOR (Database Required){COLORS['ENDC']}")
    print(f"   - File: {COLORS['OKBLUE']}enhanced_psx_processor_simple.py{COLORS['ENDC']}")
    print("   - Purpose: Process real PSX data from database")
    print("   - Requirements: Valid PSX database file")
    print(f"   - Command: {COLORS['OKCYAN']}python enhanced_psx_processor_simple.py [--symbol=SYMBOL]{COLORS['ENDC']}")
    
    print(f"\n{COLORS['OKGREEN']}3. ENHANCED PROCESSOR (Database Required){COLORS['ENDC']}")
    print(f"   - File: {COLORS['OKBLUE']}enhanced_psx_indicator_processor.py{COLORS['ENDC']}")
    print("   - Purpose: Advanced processing with async capabilities")
    print("   - Requirements: Valid PSX database file + config")
    print(f"   - Command: {COLORS['OKCYAN']}python enhanced_psx_indicator_processor.py [--config=CONFIG_PATH]{COLORS['ENDC']}")
    
    print(f"\n{COLORS['OKGREEN']}4. TEST ENVIRONMENT{COLORS['ENDC']}")
    print(f"   - File: {COLORS['OKBLUE']}test_simple.py{COLORS['ENDC']}")
    print("   - Purpose: Verify installation and environment")
    print("   - Requirements: None")
    print(f"   - Command: {COLORS['OKCYAN']}python test_simple.py{COLORS['ENDC']}")

def check_database_files() -> bool:
    """Check for database files in common locations."""
    print(f"\n{COLORS['HEADER']}CHECKING FOR DATABASE FILES...{COLORS['ENDC']}\n")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Common database locations
    db_locations = [
        # User's specific database paths
        project_root / "data" / "databases" / "production" / "PSX_consolidated_data_PSX.db",
        project_root / "PSX_Stock_Data.db",
        Path("C:/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSX_consolidated_data_PSX.db"),
        Path("C:/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/PSX_Stock_Data.db"),
        # Relative paths
        Path("data/databases/production/PSX_consolidated_data_PSX.db"),
        Path("../data/databases/production/PSX_consolidated_data_PSX.db"),
        Path("../../data/databases/production/PSX_consolidated_data_PSX.db"),
        Path("../../../data/databases/production/PSX_consolidated_data_PSX.db"),
        Path("PSX_Stock_Data.db"),
        Path("../PSX_Stock_Data.db"),
        Path("../../PSX_Stock_Data.db"),
        Path("../../../PSX_Stock_Data.db"),
    ]
    
    found_databases = []
    for db_path in db_locations:
        if db_path.exists():
            found_databases.append(db_path.resolve())
    
    if found_databases:
        print(f"{COLORS['OKGREEN']}[OK] FOUND DATABASE FILES:{COLORS['ENDC']}")
        for i, db in enumerate(found_databases, 1):
            size_mb = db.stat().st_size / (1024 * 1024)
            print(f"   {i}. {COLORS['OKBLUE']}{db}{COLORS['ENDC']}")
            print(f"      Size: {COLORS['OKCYAN']}{size_mb:.1f} MB{COLORS['ENDC']}")
        
        print(f"\n{COLORS['WARNING']}[TIP] To use these databases, update your config.yaml:{COLORS['ENDC']}")
        print(f"   {COLORS['BOLD']}source_db_path: \"{found_databases[0]}\"{COLORS['ENDC']}")
        
    else:
        print(f"{COLORS['FAIL']}[X] NO DATABASE FILES FOUND{COLORS['ENDC']}")
        print(f"   {COLORS['WARNING']}[TIP] For database processing, you need:{COLORS['ENDC']}")
        print(f"   - {COLORS['BOLD']}PSX_Stock_Data.db{COLORS['ENDC']} (source database)")
        print(f"   - Or configure path in {COLORS['BOLD']}config.yaml{COLORS['ENDC']}")
        
    return len(found_databases) > 0

def interactive_launcher():
    """Interactive launcher to help user choose the right option."""
    print(f"\n{COLORS['HEADER']}==> INTERACTIVE LAUNCHER{COLORS['ENDC']}\n")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    has_database = check_database_files()
    
    print(f"\n{COLORS['HEADER']}{'-' * 60}{COLORS['ENDC']}")
    print(f"{COLORS['BOLD']}RECOMMENDATIONS BASED ON YOUR SETUP:{COLORS['ENDC']}")
    print(f"{COLORS['HEADER']}{'-' * 60}{COLORS['ENDC']}")
    
    if has_database:
        print(f"{COLORS['OKGREEN']}[OK] Database detected - All options available!{COLORS['ENDC']}")
        print(f"   Recommended: {COLORS['BOLD']}Option 2 (Simple Processor){COLORS['ENDC']} for best results")
    else:
        print(f"{COLORS['WARNING']}[!] No database detected{COLORS['ENDC']}")
        print(f"   Recommended: {COLORS['BOLD']}Option 1 (Demo Mode){COLORS['ENDC']} to see functionality")
    
    print(f"\n{COLORS['HEADER']}{'-' * 60}{COLORS['ENDC']}")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("[BYE] Goodbye!")
                return
            
            choice_num = int(choice)
            
            if choice_num == 1:
                print("\n[LAUNCH] Launching Demo Mode...")
                os.system(f"python \"{script_dir / 'demo_processor.py'}\"")
                break
                
            elif choice_num == 2:
                if has_database:
                    print("\n[LAUNCH] Launching Simple Processor...")
                    os.system(f"python \"{script_dir / 'enhanced_psx_processor_simple.py'}\"")
                else:
                    print("\n[X] Database required for this option!")
                    print("   Use Option 1 (Demo) or configure database first")
                    continue
                break
                
            elif choice_num == 3:
                if has_database:
                    print("\n[LAUNCH] Launching Enhanced Processor...")
                    os.system(f"python \"{script_dir / 'enhanced_psx_indicator_processor.py'}\"")
                else:
                    print("\n[X] Database required for this option!")
                    print("   Use Option 1 (Demo) or configure database first")
                    continue
                break
                
            elif choice_num == 4:
                print("\n[TEST] Running Environment Tests...")
                os.system(f"python \"{script_dir / 'test_simple.py'}\"")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-4 or 'q'")
                
        except ValueError:
            print("❌ Please enter a valid number (1-4) or 'q'")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return

def main():
    """Main launcher function."""
    print_header()
    print_options()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Check if running with arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['demo', '1']:
            os.system(f"python \"{script_dir / 'demo_processor.py'}\"")
        elif arg in ['simple', '2']:
            if len(sys.argv) > 2 and sys.argv[2].startswith('--symbol='):
                symbol = sys.argv[2].split('=')[1]
                os.system(f"python \"{script_dir / 'enhanced_psx_processor_simple.py'}\" --symbol {symbol}")
            else:
                os.system(f"python \"{script_dir / 'enhanced_psx_processor_simple.py'}\"")
        elif arg in ['enhanced', '3']:
            os.system(f"python \"{script_dir / 'enhanced_psx_indicator_processor.py'}\"")
        elif arg in ['test', '4']:
            os.system(f"python \"{script_dir / 'test_simple.py'}\"")
        else:
            print(f"\n❌ Unknown argument: {arg}")
            print("   Valid arguments: demo, simple [--symbol=SYMBOL], enhanced, test")
    else:
        # Interactive mode
        interactive_launcher()

if __name__ == "__main__":
    main()
