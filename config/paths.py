from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Data subdirectories
DATA_BACKUPS_DIR = DATA_DIR / "backups"
DATA_DATABASES_DIR = DATA_DIR / "databases"
DATA_EXPORTS_DIR = DATA_DIR / "exports"
DATA_EXTERNAL_DIR = DATA_DIR / "external"
DATA_LOGS_DIR = DATA_DIR / "logs"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_REPORTS_DIR = DATA_DIR / "reports"
PSX_DASHBOARDS_DIR = DATA_DIR / "dashboards" / "psx"

# Database specific paths
PRODUCTION_DB_DIR = DATA_DATABASES_DIR / "production"
PSX_DB_PATH = PRODUCTION_DB_DIR / "PSX_consolidated_data_PSX.db"
PSX_SYM_PATH = PRODUCTION_DB_DIR / "psxsymbols.db"
PSX_IND_DB_PATH = PRODUCTION_DB_DIR / "psx_consolidated_data_indicators_PSX.db"
PSX_INVESTING_DB_PATH = PRODUCTION_DB_DIR / "PSX_investing_Stocks_KMI30.db"

# Config files
SYMBOLS_FILE = CONFIG_DIR / "psxsymbols.xlsx"
CONFIG_JSON = CONFIG_DIR / "config.json"
USER_CONFIG_JSON = CONFIG_DIR / "user_config.json"
PSX_CONFIG_JSON = CONFIG_DIR / "psx_config.json"

# Ensure directories exist
for directory in [
    DATA_BACKUPS_DIR, 
    DATA_DATABASES_DIR,
    DATA_EXPORTS_DIR,
    DATA_EXTERNAL_DIR,
    DATA_LOGS_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATA_REPORTS_DIR,
    PRODUCTION_DB_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)