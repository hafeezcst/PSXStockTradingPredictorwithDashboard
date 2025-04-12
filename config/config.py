"""
Configuration settings for PSX Stock Trading Predictor
This module centralizes all configurable parameters used in the analysis system.
"""

import os
import json
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# Default configuration
DEFAULT_CONFIG = {
    # Database settings
    "databases": {
        "main_db_path": "data/databases/production/psx_consolidated_data_indicators_PSX.db",
        "signals_db_path": "data/databases/production/PSX_investing_Stocks_KMI30.db",
    },
    
    # Output paths
    "output_paths": {
        "charts_folder": "outputs/charts/RSI_AO_CHARTS",
        "dashboards_folder": "outputs/dashboards/PSX_DASHBOARDS",
    },
    # report settings
    "report_settings": {
        "report_folder": "data/reports",
        "report_template": "templates/report_template.html",
    },
    
    # Analysis parameters
    "analysis": {
        "rsi_overbought": 60,
        "rsi_oversold": 40,
        "neutral_threshold": 1.5,
        "max_holding_days": 180,
        "indicator_weights": {
            "rsi_score_weight": 0.3,
            "ao_score_weight": 0.25, 
            "volume_score_weight": 0.2,
            "ma_score_weight": 0.15,
            "pattern_score_weight": 0.1
        }
    },
    
    # Visualization settings
    "visualization": {
        "chart_figsize": [14, 14],
        "dashboard_figsize": [20, 16],
        "status_colors": {
            "BUY/HOLD": "green",
            "SELL": "red",
            "OPPORTUNITY": "blue",
            "NEUTRAL": "gray"
        },
        "phase_colors": {
            "ACCUMULATION": "green",
            "DISTRIBUTION": "red",
            "NEUTRAL": "gray"
        }
    },
    
    # Market condition thresholds
    "market_conditions": {
        "STRONGLY_BULLISH": 65,
        "MODERATELY_BULLISH": 55,
        "NEUTRAL": 45,
        "MODERATELY_BEARISH": 35
    },
    
    # Position sizing recommendations
    "allocation_targets": {
        "STRONGLY_BULLISH": [80, 100],
        "MODERATELY_BULLISH": [70, 90],
        "NEUTRAL": [60, 80],
        "MODERATELY_BEARISH": [40, 60],
        "STRONGLY_BEARISH": [30, 50]
    },
    
    # KMI30 Symbols
    "kmi30_symbols": [
        'AICL', 'ATRL', 'BAFL', 'BAHL', 'CNERGY', 'EFERT', 'ENGRO', 
        'FFBL', 'FFC', 'FCCL', 'HUBC', 'HBL', 'ISL', 'ILP', 'LUCK', 
        'MCB', 'MARI', 'MEBL', 'MLCF', 'MTL', 'NBP', 'NML', 'OGDC', 
        'PAKT', 'PPL', 'PIOC', 'PSO', 'SNGP', 'SSGC'
    ]
}

# Configuration handling class
class ConfigManager:
    """Manages the configuration settings for the PSX analysis system"""
    
    def __init__(self, config_path=None):
        """Initialize with optional path to config file"""
        self.config = DEFAULT_CONFIG.copy()
        self.config_path = config_path or os.path.join(ROOT_DIR, "config", "user_config.json")
        self.load_config()
    
    def load_config(self):
        """Load configuration from file if it exists"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # Update config with user settings, preserving defaults for missing keys
                    self._recursive_update(self.config, user_config)
                print(f"Configuration loaded from {self.config_path}")
            else:
                print("No user configuration found, using defaults")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def restore_defaults(self):
        """Restore default configuration"""
        self.config = DEFAULT_CONFIG.copy()
        return self.save_config()
    
    def get(self, *keys):
        """Get a configuration value using nested keys"""
        result = self.config
        for key in keys:
            if key in result:
                result = result[key]
            else:
                return None
        return result
    
    def set(self, value, *keys):
        """Set a configuration value using nested keys"""
        if not keys:
            return False
        
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        return True
    
    def _recursive_update(self, d, u):
        """Recursively update a dictionary with another dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._recursive_update(d[k], v)
            else:
                d[k] = v

# Global config instance
config = ConfigManager()

# Helper functions to make accessing config easier
def get_config(*keys):
    """Get configuration value using nested keys"""
    return config.get(*keys)

def set_config(value, *keys):
    """Set configuration value using nested keys"""
    return config.set(value, *keys)

def save_config():
    """Save current configuration"""
    return config.save_config()

def restore_defaults():
    """Restore default configuration"""
    return config.restore_defaults()
