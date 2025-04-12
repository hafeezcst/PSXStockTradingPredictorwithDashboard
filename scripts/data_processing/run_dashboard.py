"""
Entry point script to run the PSX dashboard.
"""

import sys
from pathlib import Path
import warnings
from fpdf import FPDF

# Configure PDF settings to handle unknown widths
FPDF.set_global("SYSTEM_TTFONTS", str(Path(__file__).parent.parent.parent / "fonts"))

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='fpdf')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing.dashboard.main import main

if __name__ == "__main__":
    main()