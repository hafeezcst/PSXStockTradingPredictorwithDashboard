import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

# Import telegram_message first
from utils.telegram_message import send_telegram_message, send_telegram_message_with_image

# Now import the frozen code
from data_processing.draw_indicator_trend_lines import *

if __name__ == "__main__":
    # The code will run automatically since we imported everything with *
    pass 