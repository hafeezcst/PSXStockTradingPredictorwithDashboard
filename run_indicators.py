import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Now import and run the indicators script
from src.data_processing.psx_sql_indicator_psx import main

if __name__ == "__main__":
    main() 