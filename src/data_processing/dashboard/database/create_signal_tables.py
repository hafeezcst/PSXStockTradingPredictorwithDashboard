"""
Script to create signal tables in the database.
"""

import logging
import os
from pathlib import Path
from manager import create_signal_tables
from src.data_processing.dashboard.config.settings import PSX_SIGNALS_DB_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Use the database path from settings
        db_path = PSX_SIGNALS_DB_PATH
        
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