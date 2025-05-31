"""
Configuration settings for PSX data processing
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = PROJECT_ROOT / "logs"
EXCEL_DIR = DATA_DIR / "excel"

# Create necessary directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_DIR.mkdir(parents=True, exist_ok=True)

class PSXConfig:
    # URLs
    BASE_URL = "https://www.psx.com.pk/market-summary/announcements"
    COMPANY_URL = "https://www.psx.com.pk/company"
    
    # Logging
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = str(LOG_DIR / "psx_announcements.log")
    
    # Selenium settings
    SELENIUM_OPTIONS = {
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'headless': True,
        'no_sandbox': True,
        'disable_dev_shm_usage': True
    }
    
    # Database settings
    DB_PATH = str(DATA_DIR / "databases" / "production" / "PSX_investing_Stocks_KMI30.db")
    DB_DRIVER = "sqlite"
    DB_POOL_CONFIG = {
        'pool_size': 5,  # Maximum number of connections to keep in the pool
        'max_overflow': 10,  # Maximum number of connections that can be created beyond pool_size
        'pool_timeout': 30,  # Seconds to wait before giving up on getting a connection from the pool
        'pool_recycle': 1800,  # Seconds after which a connection is automatically recycled
        'pool_pre_ping': True,  # Enable connection health checks
        'echo': False  # Disable SQL query logging
    }
    
    # Cache settings
    CACHE_DIR = CACHE_DIR
    CACHE_EXPIRY = 3600  # 1 hour in seconds
    
    # Scraping settings
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30
    PAGE_LOAD_TIMEOUT = 30
    
    # File paths
    OUTPUT_DIR = DATA_DIR / "outputs" / "announcements"
    EXCEL_DIR = EXCEL_DIR
    PROJECT_ROOT = PROJECT_ROOT

    @classmethod
    def setup_directories(cls):
        """Create all necessary directories for the application"""
        # Create main directories
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        EXCEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create database directory
        db_dir = DATA_DIR / "databases" / "production"
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directories
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create cache subdirectories
        (CACHE_DIR / "announcements").mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "company_data").mkdir(parents=True, exist_ok=True)
        
        # Create log subdirectories
        (LOG_DIR / "scraping").mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "processing").mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_db_url(cls) -> str:
        """Get the database URL for SQLAlchemy connection"""
        # Ensure the database directory exists
        db_path = Path(cls.DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Return SQLAlchemy URL
        return f"{cls.DB_DRIVER}:///{cls.DB_PATH}"

    @classmethod
    def get_selenium_options(cls) -> dict:
        """Get Selenium options for WebDriver initialization"""
        return cls.SELENIUM_OPTIONS 