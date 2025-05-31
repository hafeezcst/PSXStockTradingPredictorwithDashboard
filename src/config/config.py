"""
Configuration Module for PSX Stock Trading Predictor
This module manages loading, saving, and providing default configuration settings.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Define root directory
ROOT_DIR = Path(__file__).parent.parent

DEFAULT_CONFIG = {
    "databases": {
        "main_db_path": "data/databases/production/PSX_investing_Stocks_KMI30.db",
        "signals_db_path": "data/databases/production/PSX_signals.db"
    },
    "output_paths": {
        "charts_folder": "outputs/charts/RSI_AO_CHARTS",
        "dashboards_folder": "outputs/dashboards/PSX_DASHBOARDS"
    },
    "analysis": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "neutral_threshold": 1.5,
        "max_holding_days": 180,
        "indicator_weights": {
            "rsi_score_weight": 1.0,
            "ao_score_weight": 1.0,
            "volume_score_weight": 1.0,
            "ma_score_weight": 1.0,
            "pattern_score_weight": 1.0
        }
    },
    "visualization": {
        "chart_figsize": [18, 16],
        "dashboard_figsize": [22, 18],
        "status_colors": {
            "BUY/HOLD": "green",
            "SELL": "red",
            "OPPORTUNITY": "blue"
        },
        "phase_colors": {
            "ACCUMULATION": "green",
            "DISTRIBUTION": "red",
            "NEUTRAL": "gray"
        }
    },
    "market_conditions": {
        "STRONGLY_BULLISH": 70,
        "MODERATELY_BULLISH": 60,
        "NEUTRAL": 50,
        "MODERATELY_BEARISH": 40,
        "STRONGLY_BEARISH": 30
    },
    "allocation_targets": {
        "STRONGLY_BULLISH": [80, 100],
        "MODERATELY_BULLISH": [70, 90],
        "NEUTRAL": [60, 80],
        "MODERATELY_BEARISH": [40, 60],
        "STRONGLY_BEARISH": [30, 50]
    },
    "kmi30_symbols": [
        "MEBL", "ENGRO", "OGDC", "PSO", "FFC", "LUCK", "HUBC", "MARI", "SEARL", "SYS",
        "EPCL", "PKGS", "EFERT", "UBL", "HBL", "MCB", "BAHL", "POL", "FATIMA", "GATM",
        "TRG", "UNITY", "INIL", "MUGHAL", "ATRL", "DGKC", "NML", "CHCC", "SNGP", "EFUG", "FCEPL"
    ]
}

class ConfigManager:
    def __init__(self):
        self.config_path = os.path.join(ROOT_DIR, "config", "config.json")
        self.default_config = DEFAULT_CONFIG
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return self.default_config.copy()

    def save_config(self, config: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Create a backup of the existing config
        if os.path.exists(self.config_path):
            backup_path = os.path.join(os.path.dirname(self.config_path), f"config.json.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            try:
                with open(self.config_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {str(e)}")
        
        # Save the new config
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved config to {self.config_path}")
        self.config = config

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def reset_to_defaults(self) -> None:
        self.config = self.default_config.copy()
        self.save_config(self.config)

# Add these lines at the end to provide get_config and config for import
config_manager = ConfigManager()
config = config_manager.get_config()

def get_config(section=None, key=None):
    """
    Fetch a config value by section and key, or return the whole config if not specified.
    """
    if section is None:
        return config
    if key is None:
        return config.get(section)
    return config.get(section, {}).get(key)
