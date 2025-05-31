import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Add the src directory to the Python path
src_dir = project_root / 'src'
sys.path.append(str(src_dir))

# Import telegram_message first
from utils.telegram_message import send_telegram_message, send_telegram_message_with_image

# Import the analysis module
from data_processing.draw_indicator_trend_lines import *

if __name__ == "__main__":
    # The code will run automatically since we imported everything with *
    pass 