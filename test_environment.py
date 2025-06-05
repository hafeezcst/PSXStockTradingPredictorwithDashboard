import os
import sys
import logging
import yaml
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check Python version
logging.info(f"Python version: {sys.version}")

# Check current working directory
logging.info(f"Current working directory: {os.getcwd()}")

# Load configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully")
    logging.info(f"Database paths: {config['database']}")
except Exception as e:
    logging.error(f"Error loading config: {str(e)}")
    sys.exit(1)

# Test database connections
def test_db_connection(db_type, db_path):
    try:
        logging.info(f"Attempting to connect to {db_type} at {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logging.info(f"Successfully connected to {db_type}, found {len(tables)} tables")
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Error connecting to {db_type}: {str(e)}")
        return False

# Test both databases
main_db_ok = test_db_connection('main_db', config['database']['main_db'])
signals_db_ok = test_db_connection('signals_db', config['database']['signals_db'])

if not main_db_ok or not signals_db_ok:
    logging.error("Database connection test failed")
    sys.exit(1)

# Test pandas and sqlalchemy
try:
    engine = create_engine(f"sqlite:///{config['database']['main_db']}")
    logging.info("SQLAlchemy engine created successfully")
    # Try a simple query
    df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5", engine)
    logging.info(f"Successfully queried database, retrieved {len(df)} rows")
except Exception as e:
    logging.error(f"Error testing pandas/SQLAlchemy: {str(e)}")
    sys.exit(1)

logging.info("Environment test completed successfully") 