from pathlib import Path
from typing import Dict, Any
import os

class PSXConfig:
    """Configuration settings for PSX data processing"""
    
    # Base paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = DATA_DIR / "logs"
    CACHE_DIR = DATA_DIR / "cache"
    CSV_DIR = DATA_DIR / "csv"
    EXCEL_DIR = DATA_DIR / "announcements"
    
    # Database settings
    DB_PATH = DATA_DIR / "psx_symbols.db"
    DB_POOL_CONFIG = {
        'pool_size': 10,
        'max_overflow': 20,
        'pool_timeout': 30,
        'pool_recycle': 3600
    }
    
    # Scraping settings
    BASE_URL = "https://dps.psx.com.pk/announcements/companies"
    COMPANY_URL = "https://dps.psx.com.pk/company"
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30
    
    # Selenium settings
    SELENIUM_OPTIONS = {
        'headless': True,
        'no_sandbox': True,
        'disable_dev_shm_usage': True,
        'disable_gpu': True,
        'window_size': '1920,1080',
        'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Logging settings
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_FILE = LOGS_DIR / "psx_announcements.log"
    
    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.LOGS_DIR, cls.CACHE_DIR, cls.CSV_DIR, cls.EXCEL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_db_url(cls) -> str:
        """Get database URL"""
        return f'sqlite:///{cls.DB_PATH}'
    
    @classmethod
    def get_selenium_options(cls) -> Dict[str, Any]:
        """Get Selenium options as a dictionary"""
        return cls.SELENIUM_OPTIONS.copy() 