"""
Streamlit app entry point for PSX Stock Trading Predictor Dashboard
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data_processing"))

# Import the main dashboard function
from scripts.data_processing.dashboard.main import main

if __name__ == "__main__":
    main() 