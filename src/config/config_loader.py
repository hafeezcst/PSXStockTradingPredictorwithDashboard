"""
Configuration loader and logger setup for PSX dashboard and analysis.
"""
import os
import yaml
import logging
from typing import Optional, Dict

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file, create default if not exists."""
    default_config = {
        'database': {
            'main_db': 'data/databases/production/PSX_investing_Stocks.db',
            'signals_db': 'data/databases/production/PSX_investing_Stocks_KMI30.db'
        },
        'output': {
            'charts_folder': 'outputs/charts/RSI_AO_CHARTS',
            'dashboards_folder': 'outputs/dashboards/PSX_DASHBOARDS'
        },
        'telegram': {
            'max_images_per_message': 10,
            'bot_token': os.environ.get('TELEGRAM_BOT_TOKEN', ''),
            'chat_id': os.environ.get('TELEGRAM_CHAT_ID', '')
        },
        'analysis': {
            'lookback_years': 10,
            'rsi_thresholds': [40, 60],
            'ma_periods': [10, 30, 50],
            'volume_ma_period': 20,
            'ao_fast_period': 5,
            'ao_slow_period': 34
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'file': 'logs/psx_analysis.log'
        }
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Ensure all required keys exist
            for section, default_values in default_config.items():
                if section not in config:
                    config[section] = default_values
                else:
                    for key, value in default_values.items():
                        if key not in config[section]:
                            config[section][key] = value
        else:
            config = default_config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Created default config file at {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config

def setup_logging(config: Optional[Dict] = None):
    """Setup logging configuration."""
    try:
        if config is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filename='logs/psx_analysis.log'
            )
            print("Using default logging configuration")
            return
        log_level = getattr(logging, config['logging']['level'])
        log_format = config['logging']['format']
        log_file = config['logging']['file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=log_file
        )
    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='logs/psx_analysis.log'
        )
        print(f"Error setting up logging: {e}") 