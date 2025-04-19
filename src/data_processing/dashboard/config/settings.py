"""
Configuration settings for the PSX dashboard application.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default KMI30 symbols
DEFAULT_KMI30_SYMBOLS = [
    "OGDC", "PPL", "LUCK", "MARI", "ENGRO", "UBL", "HBL", "MCB", "PSO", "EFERT",
    "HUBC", "POL", "BAHL", "ATRL", "FFC", "GHGL", "MTL", "UNITY", "MLCF", "DGKC",
    "KOHC", "NML", "MEBL", "ABL", "BAFL", "FABL", "AKBL", "FFBL", "CHCC", "PIOC"
]

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

# Data subdirectories
DATA_BACKUPS_DIR = DATA_DIR / "backups"
DATA_DATABASES_DIR = DATA_DIR / "databases"
DATA_EXPORTS_DIR = DATA_DIR / "exports"
DATA_EXTERNAL_DIR = DATA_DIR / "external"
DATA_LOGS_DIR = DATA_DIR / "logs"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_REPORTS_DIR = DATA_DIR / "reports"
PSX_DASHBOARDS_DIR = DATA_DIR / "dashboards" / "PSX_DASHBOARDS"
PRODUCTION_DB_DIR = DATA_DATABASES_DIR / "production"

# Database paths
PSX_DB_PATH = PRODUCTION_DB_DIR / "psx_consolidated_data_indicators_PSX.db"
PSX_SYM_PATH = PRODUCTION_DB_DIR / "psx_symbols.db"
PSX_IND_DB_PATH = PRODUCTION_DB_DIR / "psx_indicators.db"
PSX_INVESTING_DB_PATH = PRODUCTION_DB_DIR / "PSX_investing_Stocks_KMI30.db"
PSX_SIGNALS_DB_PATH = PRODUCTION_DB_DIR / "psx_consolidated_data_indicators_PSX.db"  # Same as PSX_DB_PATH since signals are stored there

# Config files
CONFIG_JSON = CONFIG_DIR / "config.json"
USER_CONFIG_JSON = CONFIG_DIR / "user_config.json"
PSX_CONFIG_JSON = CONFIG_DIR / "psx_config.json"

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "databases": {
        "main_db_path": str(PSX_DB_PATH),
        "signals_db_path": str(PSX_INVESTING_DB_PATH),
        "connection_pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 1800
    },
    "output_paths": {
        "charts_folder": str(DATA_DIR / "charts"),
        "rsi_ao_charts_folder": str(DATA_DIR / "charts/rsi_ao"),
        "dashboards_folder": str(DATA_DIR / "dashboards"),
        "psx_dashboards_folder": str(PSX_DASHBOARDS_DIR),
        "exports_folder": str(DATA_EXPORTS_DIR),
        "logs_folder": str(DATA_LOGS_DIR)
    },
    "analysis": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "neutral_threshold": 1.5,
        "max_holding_days": 60,
        "indicator_weights": {
            "rsi_score_weight": 0.2,
            "ao_score_weight": 0.2,
            "volume_score_weight": 0.2,
            "ma_score_weight": 0.2,
            "pattern_score_weight": 0.2
        }
    },
    "visualization": {
        "chart_figsize": [12, 8],
        "dashboard_figsize": [16, 10],
        "status_colors": {
            "BUY/HOLD": "#00FF00",
            "SELL": "#FF0000",
            "OPPORTUNITY": "#0000FF",
            "NEUTRAL": "#808080"
        }
    },
    "kmi30_symbols": DEFAULT_KMI30_SYMBOLS
}

def create_config_file(config_path: Path = CONFIG_JSON) -> bool:
    """Create a default config file."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Error creating config file: {str(e)}")
        return False

def load_config() -> Dict[str, Any]:
    """Load configuration from file or create default if not exists."""
    try:
        if not CONFIG_JSON.exists():
            create_config_file()
        
        with open(CONFIG_JSON, 'r') as f:
            config = json.load(f)
        
        if USER_CONFIG_JSON.exists():
            with open(USER_CONFIG_JSON, 'r') as f:
                user_config = json.load(f)
            config.update(user_config)
        
        return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return DEFAULT_CONFIG.copy()

def initialize_config() -> Dict[str, Any]:
    """Initialize configuration settings for the dashboard."""
    config = {
        # Data paths
        "data_dir": str(PROJECT_ROOT / "data"),
        "excel_dir": str(PROJECT_ROOT / "data" / "excel"),
        "csv_dir": str(PROJECT_ROOT / "data" / "csv"),
        "model_dir": str(PROJECT_ROOT / "models"),
        
        # Financial reports configuration
        "announcements_file": str(PROJECT_ROOT / "data" / "excel" / "PSX_Announcements.xlsx"),
        "reports_cache_dir": str(PROJECT_ROOT / "data" / "cache" / "reports"),
        
        # API configuration
        "api_keys": {
            "huggingface": os.getenv('HUGGINGFACE_API_KEY', ''),  # Get from environment variable instead of session state
        },
        
        # Cache settings
        "cache_dir": str(PROJECT_ROOT / "data" / "cache"),
        "cache_expiry": 3600,  # Cache expiry in seconds
        
        # Market settings
        "market_hours": {
            "regular": {
                "open": "09:32",
                "close": "15:30"
            },
            "friday": {
                "session1": {
                    "open": "09:17",
                    "close": "12:00"
                },
                "session2": {
                    "open": "14:32",
                    "close": "16:30"
                }
            }
        }
    }
    
    # Ensure cache directories exist
    for cache_dir in [config["cache_dir"], config["reports_cache_dir"]]:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    return config 