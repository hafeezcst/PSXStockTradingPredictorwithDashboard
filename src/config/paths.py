from pathlib import Path

# Base directories
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
DATA_DATABASES_DIR = DATA_DIR / 'databases'

# Subdirectories
DATA_LOGS_DIR = DATA_DIR / 'logs'
DATA_REPORTS_DIR = DATA_DIR / 'reports'
PRODUCTION_DB_DIR = DATA_DATABASES_DIR / 'production'

# Database paths
PSX_DB_PATH = PRODUCTION_DB_DIR / "PSX_consolidated_data_PSX.db"
PSX_SYM_PATH = PRODUCTION_DB_DIR / "PSX_symbols.db"
PSX_IND_DB_PATH = PRODUCTION_DB_DIR / "psx_consolidated_data_indicators_PSX.db"
PSX_INVESTING_DB_PATH = PRODUCTION_DB_DIR / "PSX_investing_Stocks_KMI100.db"