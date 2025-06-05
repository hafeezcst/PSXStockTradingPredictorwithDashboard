import json
import os
from pathlib import Path

def load_config_file(filename):
    """Load configuration from a JSON file"""
    config_path = Path(__file__).parent / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filename}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

# Load configurations
try:
    # Try to load user config first, fall back to default if not found
    config = load_config_file('user_config.json')
except FileNotFoundError:
    config = load_config_file('default_config.json')

def get_config(section, key=None):
    """Get configuration value for a given section and key
    
    Args:
        section (str): Configuration section name
        key (str, optional): Specific key within the section. If None, returns entire section
        
    Returns:
        The configuration value or entire section if key is None
    """
    if section not in config:
        raise KeyError(f"Configuration section '{section}' not found")
    
    if key is None:
        return config[section]
    
    if key not in config[section]:
        raise KeyError(f"Configuration key '{key}' not found in section '{section}'")
    
    return config[section][key] 