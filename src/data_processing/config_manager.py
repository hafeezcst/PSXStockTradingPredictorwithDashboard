"""
Configuration management for PSX data download operations.
"""

import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, List
from .exceptions import ConfigurationError

@dataclass
class DatabaseConfig:
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    main_db_path: str = "data/databases/production/PSX_consolidated_data_PSX.db"
    alt_db_path: str = "data/databases/production/PSX_consolidated_data_PSX_Alternative.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24

@dataclass
class ThreadingConfig:
    max_workers: int = 4
    max_threads: int = 8
    min_threads: int = 2
    request_timeout: int = 60
    response_time_threshold_high: float = 1.5
    response_time_threshold_low: float = 0.5

@dataclass
class APIConfig:
    history_url: str = "https://dps.psx.com.pk/historical"
    symbols_url: str = "https://dps.psx.com.pk/symbols"
    max_retries: int = 3
    pool_connections: int = 10
    pool_maxsize: int = 20
    delay_between_requests: float = 0.5

@dataclass
class DataValidationConfig:
    max_price_change_percent: float = 50.0
    min_volume: int = 0
    max_volume: int = 1000000000
    check_ohlc_relationships: bool = True
    validate_date_continuity: bool = True

@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "data_reader.log"
    format: str = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    max_file_size_mb: int = 10
    backup_count: int = 5

@dataclass
class MonitoringConfig:
    enable_metrics: bool = True
    metrics_file: str = "download_metrics.json"
    health_check_interval: int = 300

@dataclass
class SymbolsConfig:
    file_path: str = "data/databases/production/psxsymbols.xlsx"
    sheet_name: str = "KSEALL"
    max_failed_attempts: int = 5
    max_total_failures: int = 500

@dataclass
class AppConfig:
    database: DatabaseConfig
    threading: ThreadingConfig
    api: APIConfig
    data_validation: DataValidationConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    symbols: SymbolsConfig
    
    @classmethod
    def from_yaml(cls, config_path: str = None) -> 'AppConfig':
        """Load configuration from YAML file"""
        if config_path is None:
            # Default to config.yaml in the same directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'config.yaml')
        
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML config: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading config file: {e}")
        
        return cls._create_from_dict(config_data)
    
    @classmethod
    def _create_from_dict(cls, config_data: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary"""
        try:
            database_config = DatabaseConfig(**config_data.get('database', {}))
            threading_config = ThreadingConfig(**config_data.get('threading', {}))
            api_config = APIConfig(**config_data.get('api', {}))
            data_validation_config = DataValidationConfig(**config_data.get('data_validation', {}))
            logging_config = LoggingConfig(**config_data.get('logging', {}))
            monitoring_config = MonitoringConfig(**config_data.get('monitoring', {}))
            symbols_config = SymbolsConfig(**config_data.get('symbols', {}))
            
            return cls(
                database=database_config,
                threading=threading_config,
                api=api_config,
                data_validation=data_validation_config,
                logging=logging_config,
                monitoring=monitoring_config,
                symbols=symbols_config
            )
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration structure: {e}")
    
    def validate(self) -> None:
        """Validate configuration values"""
        if self.threading.min_threads > self.threading.max_threads:
            raise ConfigurationError("min_threads cannot be greater than max_threads")
        
        if self.threading.max_workers > self.threading.max_threads:
            raise ConfigurationError("max_workers cannot be greater than max_threads")
        
        if self.data_validation.min_volume < 0:
            raise ConfigurationError("min_volume cannot be negative")
        
        if self.data_validation.max_volume <= self.data_validation.min_volume:
            raise ConfigurationError("max_volume must be greater than min_volume")
        
        if not os.path.exists(os.path.dirname(self.database.main_db_path)):
            os.makedirs(os.path.dirname(self.database.main_db_path), exist_ok=True)
        
        if not os.path.exists(os.path.dirname(self.database.alt_db_path)):
            os.makedirs(os.path.dirname(self.database.alt_db_path), exist_ok=True)
