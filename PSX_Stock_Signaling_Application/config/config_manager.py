import json
import os
import yaml
from .path_resolver import path_resolver
from typing import Dict, Any

class ConfigManager:
    def __init__(self):
        self._configs: Dict[str, Any] = {}
        self._env = os.getenv('APP_ENV', 'development')
        
    def load_config(self, config_path: str) -> None:
        """Load configuration from file"""
        full_path = path_resolver.resolve('config', config_path)
        
        if config_path.endswith('.json'):
            with open(full_path, 'r') as f:
                self._configs.update(json.load(f))
        elif config_path.endswith(('.yaml', '.yml')):
            with open(full_path, 'r') as f:
                self._configs.update(yaml.safe_load(f))
                
    def get(self, key: str, default=None) -> Any:
        """Get config value with optional default"""
        return self._configs.get(key, default)
        
    def get_env_specific(self, base_key: str) -> Any:
        """Get environment-specific config (key_dev, key_prod, etc)"""
        env_key = f"{base_key}_{self._env}"
        return self._configs.get(env_key, self._configs.get(base_key))

# Singleton instance
config_manager = ConfigManager()