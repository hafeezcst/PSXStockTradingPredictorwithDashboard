"""
Script to create signal tables in the database.
"""

import logging
import os
from pathlib import Path
from manager import create_signal_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Get the database path - using absolute path from project root
        project_root = Path(__file__).parent.parent.parent.parent
        db_path = project_root / "data" / "psx_data.db"
        
        # Ensure the data directory exists
        os.makedirs(db_path.parent, exist_ok=True)
        
        logger.info(f"Creating signal tables in database: {db_path}")
        
        # Create the signal tables
        if create_signal_tables(str(db_path)):
            logger.info("Signal tables created successfully")
        else:
            logger.error("Failed to create signal tables")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 