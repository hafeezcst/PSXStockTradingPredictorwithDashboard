"""
PSX Data Processing Pipeline Controller

This script integrates and orchestrates the three main PSX data processing scripts:
1. Data download (01-PSX_Database_data_download_to_SQL_db_PSX.py)
2. Indicator calculation (01-PSX_SQL_Indicator_PSX.py) 
3. Duplicate removal (02-sql_duplicate_remover_ALL.py)

Features:
- Unified configuration and logging
- Shared resource management
- Sequential execution with error handling
- Status reporting and progress tracking
"""
import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime
import sqlite3
from sqlalchemy import create_engine

# Import the individual script functions
from src.data_processing.fix_pandas_ta import ta
from src.data_processing.data_downloader import DataReader as Downloader
from src.data_processing.indicator_calculator import DataReader as IndicatorCalculator
from src.data_processing.duplicate_remover import process_database as duplicate_remover

class PSXDataPipeline:
    def __init__(self, config: Optional[dict] = None):
        """Initialize the pipeline with configuration"""
        self.config = config or self.default_config()
        self.setup_logging()
        self.db_engines = {}
        self.current_stage = "initialization"
        
    @staticmethod
    def default_config():
        """Default configuration for the pipeline"""
        return {
            "database_paths": {
                "source": "data/databases/production/psx_consolidated_data_PSX.db",
                "target": "data/databases/production/psx_consolidated_data_indicators_PSX.db",
                "alternative": "data/databases/production/PSX_consolidated_data_PSX_Alternative.db"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "filename": "psx_pipeline.log"
            },
            "max_retries": 3,
            "thread_settings": {
                "max_workers": 4,
                "max_threads": 8,
                "min_threads": 2
            }
        }

    def setup_logging(self):
        """Configure unified logging for the pipeline"""
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            filename=self.config["logging"]["filename"],
            filemode='a'
        )
        self.logger = logging.getLogger('PSX_Pipeline')
        self.logger.info("Pipeline initialized")

    def get_db_engine(self, db_key: str):
        """Get or create a shared database engine"""
        if db_key not in self.db_engines:
            path = self.config["database_paths"][db_key]
            self.db_engines[db_key] = create_engine(f'sqlite:///{path}')
            self.logger.info(f"Created engine for {db_key} database")
        return self.db_engines[db_key]

    def run_download_stage(self):
        """Execute the data download stage"""
        self.current_stage = "download"
        self.logger.info("Starting download stage")
        
        try:
            downloader = Downloader(
                db_path=self.config["database_paths"]["source"],
                alt_db_path=self.config["database_paths"]["alternative"]
            )
            downloader.check_database_integrity()
            
            # Load symbols and process data
            symbols_df = pd.read_excel('data/databases/production/psxsymbols.xlsx', 
                                     sheet_name='KSEALL')
            valid_symbols = symbols_df.iloc[:, 0].tolist()
            
            downloader.delete_unused_tables(valid_symbols)
            downloader.process_symbols(valid_symbols)
            return True
        except Exception as e:
            self.logger.error(f"Download stage failed: {str(e)}")
            return False

    def run_indicator_stage(self):
        """Execute the indicator calculation stage"""
        self.current_stage = "indicators"
        self.logger.info("Starting indicator calculation stage")
        
        try:
            calculator = IndicatorCalculator(
                source_db_path=self.config["database_paths"]["source"],
                target_db_path=self.config["database_paths"]["target"]
            )
            
            table_names = calculator.get_table_names()
            calculator.delete_unused_tables(table_names)
            
            for table_name in table_names:
                data = calculator.read_data(table_name)
                processed_data = calculator.preprocess(data)
                calculator.save_to_db(processed_data, table_name)
            return True
        except Exception as e:
            self.logger.error(f"Indicator stage failed: {str(e)}")
            return False

    def run_cleanup_stage(self):
        """Execute the duplicate removal stage"""
        self.current_stage = "cleanup"
        self.logger.info("Starting duplicate removal stage")
        
        try:
            conn = sqlite3.connect(self.config["database_paths"]["target"])
            process_database(conn, "PSX_indicators")
            return True
        except Exception as e:
            self.logger.error(f"Cleanup stage failed: {str(e)}")
            return False

    def cleanup_resources(self):
        """Clean up all shared resources"""
        self.logger.info("Cleaning up resources")
        for name, engine in self.db_engines.items():
            try:
                engine.dispose()
                self.logger.info(f"Disposed engine for {name}")
            except Exception as e:
                self.logger.error(f"Error disposing {name} engine: {str(e)}")

    def execute_pipeline(self):
        """Execute the complete pipeline with error handling"""
        start_time = datetime.now()
        self.logger.info(f"Pipeline started at {start_time}")
        
        try:
            # Execute stages in sequence
            if not self.run_download_stage():
                raise RuntimeError("Download stage failed")
            
            if not self.run_indicator_stage():
                raise RuntimeError("Indicator stage failed")
                
            if not self.run_cleanup_stage():
                raise RuntimeError("Cleanup stage failed")
                
            self.logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed at {self.current_stage} stage: {str(e)}")
            return False
            
        finally:
            self.cleanup_resources()
            duration = datetime.now() - start_time
            self.logger.info(f"Pipeline completed in {duration.total_seconds():.2f} seconds")

if __name__ == "__main__":
    pipeline = PSXDataPipeline()
    success = pipeline.execute_pipeline()
    sys.exit(0 if success else 1)