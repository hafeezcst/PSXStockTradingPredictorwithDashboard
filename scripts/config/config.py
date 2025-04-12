"""
Configuration Module for PSX Stock Trading Predictor
This module manages loading, saving, and providing default configuration settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

# Define root directory
ROOT_DIR = Path(__file__).parent.parent

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
