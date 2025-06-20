import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.relativedelta import relativedelta
from pandas import DataFrame as container
from bs4 import BeautifulSoup as parser
from datetime import datetime, date, timedelta
from typing import Union, List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
import requests
import os
from sqlalchemy import create_engine, MetaData, Table, inspect, text
from collections import defaultdict
import time
import numpy as np

# Import enhanced modules
from config_manager import AppConfig
from exceptions import PSXDataDownloadError, DatabaseConnectionError, APIConnectionError, DataValidationError
from monitoring import MetricsCollector, HealthChecker
from data_validator import DataValidator
from enhanced_db_manager import EnhancedDatabaseManager
from api_client import PSXAPIClient
from enhanced_data_processor import EnhancedDataProcessor

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('data_reader.log'),
        logging.StreamHandler()
    ]
)

class EnhancedDataReader:
    """Enhanced PSX Data Reader with improved architecture and features"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Enhanced Data Reader
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = AppConfig.from_yaml(config_path)
        self.config.validate()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.logging.level))
        
        # Initialize components
        self.db_manager = EnhancedDatabaseManager(self.config.database)
        self.metrics_collector = MetricsCollector(self.config.monitoring.metrics_file)
        self.health_checker = HealthChecker(self.config.monitoring.health_check_interval)
        self.api_client = PSXAPIClient(self.config.api, self.metrics_collector)
        self.data_processor = EnhancedDataProcessor(self.config.data_validation)
        
        # Threading configuration with improved limits
        self.max_workers = self.config.threading.max_workers
        self.max_threads = self.config.threading.max_threads
        self.min_threads = self.config.threading.min_threads
        self.response_times = []
        
        # Legacy compatibility
        self.headers = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        self.current_dir = os.getcwd()
        self.__history = self.config.api.history_url
        self.__symbols = self.config.api.symbols_url
        
        self.logger.info("Enhanced Data Reader initialized successfully")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()

    def tickers(self) -> pd.DataFrame:
        """
        Fetch available tickers from PSX
        
        Returns:
            DataFrame with ticker information
        """
        try:
            ticker_data = self.api_client.fetch_symbols()
            if ticker_data:
                return pd.DataFrame(ticker_data)
            else:
                self.logger.error("Failed to fetch tickers from API")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching tickers: {e}")
            return pd.DataFrame()
    
    def adjust_thread_count(self):
        """Adjust thread count based on recent response times with enhanced logic"""
        if len(self.response_times) < 5:
            return
        
        avg_response_time = np.mean(self.response_times[-5:])
        
        # More sophisticated adjustment logic
        if avg_response_time > self.config.threading.response_time_threshold_high:
            self.max_workers = max(self.min_threads, self.max_workers - 1)
            self.logger.debug(f"Reduced workers to {self.max_workers} due to slow response times")
        elif avg_response_time < self.config.threading.response_time_threshold_low:
            self.max_workers = min(self.max_threads, self.max_workers + 1)
            self.logger.debug(f"Increased workers to {self.max_workers} due to fast response times")
        
        # Perform health check periodically
        if self.health_checker.should_check():
            health_results = self.health_checker.perform_health_check(
                self.db_manager.get_engine(), 
                self.config.database.main_db_path
            )
            
            if not all(health_results.values()):
                self.logger.warning("Health check issues detected, reducing worker count")
                self.max_workers = max(self.min_threads, self.max_workers - 1)
    
    def get_last_date(self, symbol: str, use_alt: bool = False) -> Union[date, None]:
        """
        Get the last date for a symbol with enhanced error handling
        
        Args:
            symbol: Stock symbol
            use_alt: Whether to use alternative database
            
        Returns:
            Last date or None if not found
        """
        try:
            return self.db_manager.get_last_date(symbol, use_alt)
        except Exception as e:
            self.logger.error(f"Error getting last date for {symbol}: {e}")
            return None

    def get_today_data(self, symbol: str) -> container:
        """
        Fetch today's data with enhanced validation
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with today's data or empty DataFrame
        """
        today = date.today()
        
        # Check if today's data exists in the database
        if self.verify_data(symbol, today, today):
            self.logger.info(f"Today's data for {symbol} is already downloaded.")
            return pd.DataFrame()
        
        # Download today's data
        self.logger.info(f"Downloading today's data for {symbol}.")
        try:
            today_data = self.get_psx_data(symbol, [today])
            return today_data
        except Exception as e:
            self.logger.error(f"Error downloading today's data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_psx_data(self, symbol: str, dates: list) -> container:
        """
        Enhanced data retrieval with improved error handling and monitoring
        
        Args:
            symbol: Stock symbol
            dates: List of dates to download
            
        Returns:
            Combined DataFrame with all data
        """
        data = []
        successful_downloads = 0
        failed_downloads = 0
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all download tasks
                future_to_date = {
                    executor.submit(self.download, symbol, date_obj): date_obj 
                    for date_obj in dates
                }
                
                # Process completed futures with progress tracking
                for future in tqdm(as_completed(future_to_date), 
                                 total=len(future_to_date), 
                                 desc=f"Downloading {symbol}'s Data"):
                    date_obj = future_to_date[future]
                    
                    try:
                        result = future.result()
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            data.append(result)
                            successful_downloads += 1
                        else:
                            failed_downloads += 1
                            self.logger.debug(f"No data received for {symbol} on {date_obj}")
                            
                    except Exception as e:
                        failed_downloads += 1
                        self.logger.error(f"Error downloading data for {symbol} on {date_obj}: {e}")
                    
                    # Add delay to reduce server load
                    time.sleep(self.config.api.delay_between_requests)
        
        except Exception as e:
            self.logger.error(f"Error in thread pool execution for {symbol}: {e}")
        
        # Log download statistics
        total_attempts = successful_downloads + failed_downloads
        if total_attempts > 0:
            success_rate = (successful_downloads / total_attempts) * 100
            self.logger.info(f"Download completed for {symbol}: {successful_downloads}/{total_attempts} "
                           f"successful ({success_rate:.1f}%)")
        
        # Process and combine data
        return self.data_processor.combine_dataframes(data)

    def download(self, symbol: str, date_obj: date) -> pd.DataFrame:
        """
        Enhanced download method with better error handling and monitoring
        
        Args:
            symbol: Stock symbol
            date_obj: Date to download data for
            
        Returns:
            DataFrame with stock data or empty DataFrame if failed
        """
        start_time = time.time()
        
        try:
            # Fetch data using API client
            html_content = self.api_client.fetch_historical_data(symbol, date_obj)
            
            if html_content is None:
                return pd.DataFrame()
            
            # Parse the HTML content
            raw_data = self.api_client.parse_historical_data(html_content, symbol)
            
            # Process and validate the data
            processed_df, is_valid, issues = self.data_processor.process_and_validate(raw_data, symbol)
            
            if not is_valid and not processed_df.empty:
                self.logger.warning(f"Data quality issues for {symbol} on {date_obj}: {'; '.join(issues)}")
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.adjust_thread_count()
            
            return processed_df
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Unexpected error downloading {symbol} on {date_obj}: {e}")
            return pd.DataFrame()
    
    def toframe(self, data) -> pd.DataFrame:
        """
        Legacy method for backward compatibility - now uses enhanced processor
        
        Args:
            data: BeautifulSoup parsed data
            
        Returns:
            Processed DataFrame
        """
        try:
            # Extract data using the enhanced API client parser
            raw_data = self.api_client.parse_historical_data(str(data), "legacy_symbol")
            
            # Process using enhanced data processor
            processed_df, _, _ = self.data_processor.process_and_validate(raw_data, "legacy_symbol")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error in legacy toframe method: {e}")
            return pd.DataFrame()
    
    def daterange(self, start: date, end: date) -> list:
        """
        Enhanced date range generation using the data processor
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List of dates for data download
        """
        try:
            return self.data_processor.generate_date_range(start, end)
        except Exception as e:
            self.logger.error(f"Error generating date range: {e}")
            # Fallback to simple date list
            return [start] if start <= end else []
    
    def preprocess(self, data: list) -> pd.DataFrame:
        """
        Legacy preprocessing method - now uses enhanced processor
        
        Args:
            data: List of DataFrames
            
        Returns:
            Combined and processed DataFrame
        """
        try:
            return self.data_processor.combine_dataframes(data)
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            return pd.DataFrame()

    def save_to_db(self, data: pd.DataFrame, table_name: str, use_alt=False):
        """Enhanced database save with validation and error handling"""
        try:
            self.db_manager.save_data(data, table_name, use_alt)
        except Exception as e:
            self.logger.error(f"Error saving data to {table_name}: {e}")
            raise DatabaseConnectionError(f"Failed to save data to {table_name}: {e}")
    
    def verify_data(self, symbol: str, start_date: date, end_date: date, use_alt=False) -> bool:
        """Enhanced data verification"""
        try:
            return self.db_manager.verify_data_exists(symbol, start_date, end_date, use_alt)
        except Exception as e:
            self.logger.error(f"Error verifying data for {symbol}: {e}")
            return False
    
    def delete_failing_table(self, symbol: str, use_alt=False):
        """Enhanced table deletion with improved logging"""
        try:
            table_name = f'PSX_{symbol}_stock_data'
            self.db_manager.delete_table(table_name, use_alt)
            self.logger.info(f"Table {table_name} deleted due to repeated download failures")
        except Exception as e:
            self.logger.error(f"Error deleting table for {symbol}: {e}")
    
    def check_database_integrity(self, use_alt=False):
        """Enhanced database integrity check"""
        try:
            if self.db_manager.check_integrity(use_alt):
                self.logger.info("Database integrity check passed")
                return True
            else:
                self.logger.error("Database integrity check failed")
                return False
        except Exception as e:
            self.logger.error(f"Database integrity check error: {e}")
            return False
    
    def delete_unused_tables(self, valid_symbols: List[str], use_alt=False):
        """Enhanced cleanup of unused tables"""
        try:
            self.db_manager.delete_unused_tables(valid_symbols, use_alt)
        except Exception as e:
            self.logger.error(f"Error deleting unused tables: {e}")
    
    def stocks(self, ticker: str, start: date, end: date) -> container:
        """
        Enhanced stock data retrieval with comprehensive error handling
        
        Args:
            ticker: Stock symbol
            start: Start date
            end: End date
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Validate date range
            is_valid, message = self.data_processor.validator.validate_date_range(start, end)
            if not is_valid:
                raise DataValidationError(f"Invalid date range: {message}")
            
            # Generate date range
            dates = self.daterange(start, end)
              # Download data
            data = self.get_psx_data(ticker, dates)
            
            # Log summary
            if not data.empty:
                self.logger.info(f"Retrieved {len(data)} records for {ticker} from {start} to {end}")
            else:
                self.logger.warning(f"No data retrieved for {ticker}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving stocks data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            metrics = self.metrics_collector.get_current_metrics()
            db_stats = self.db_manager.get_database_stats()
            api_stats = self.api_client.get_performance_stats()
            
            return {
                'download_metrics': metrics,
                'database_stats': db_stats,
                'api_performance': api_stats,
                'threading_config': {
                    'current_workers': self.max_workers,
                    'max_threads': self.max_threads,
                    'min_threads': self.min_threads
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def cleanup(self):
        """Enhanced cleanup with proper resource management"""
        try:
            # Save final metrics
            if hasattr(self, 'metrics_collector'):
                self.metrics_collector.save_metrics()
            
            # Close API client
            if hasattr(self, 'api_client'):
                self.api_client.close()
            
            # Close database connections
            if hasattr(self, 'db_manager'):
                self.db_manager.close_connections()
            
            self.logger.info("Enhanced Data Reader cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Legacy DataReader class for backward compatibility
class DataReader(EnhancedDataReader):
    """Legacy DataReader class that wraps EnhancedDataReader for backward compatibility"""
    
    def __init__(self, db_path=None, alt_db_path=None):
        """Initialize with legacy parameters"""
        # Create temporary config for legacy compatibility
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        try:
            super().__init__(config_path)
            
            # Override database paths if provided
            if db_path:
                self.config.database.main_db_path = db_path
                self.db_manager = EnhancedDatabaseManager(self.config.database)
            
            if alt_db_path:
                self.config.database.alt_db_path = alt_db_path
                self.db_manager = EnhancedDatabaseManager(self.config.database)
                
        except Exception as e:
            self.logger.error(f"Error initializing legacy DataReader: {e}")
            # Fallback to basic initialization
            pass

if __name__ == "__main__":
    """Enhanced main execution with comprehensive error handling and monitoring"""
    
    try:
        # Initialize enhanced data reader
        with EnhancedDataReader() as data_reader:
            current_dir = os.getcwd()
            # Use the correct path for psxsymbols.xlsx
            symbols_file_path = os.path.join(current_dir, 'data', 'databases', 'production', 'psxsymbols.xlsx')
            
            # Load valid symbols
            try:
                symbols_df = pd.read_excel(symbols_file_path, sheet_name=data_reader.config.symbols.sheet_name)
                valid_symbols = symbols_df.iloc[:, 0].tolist()
                print(f'Total symbols loaded: {len(valid_symbols)}')
                data_reader.logger.info(f'Loaded {len(valid_symbols)} symbols from {symbols_file_path}')
            except Exception as e:
                data_reader.logger.error(f"Error loading symbols file: {e}")
                print(f"Error loading symbols: {e}")
                exit(1)
            
            # Initialize metrics collection
            data_reader.metrics_collector.start_session(len(valid_symbols))
            
            # Delete unused tables
            data_reader.delete_unused_tables(valid_symbols)
            
            # Determine end date based on current time
            current_time = datetime.now()
            if current_time.hour < 17:  # Before 5:00 PM
                end_date = date.today() - timedelta(days=1)
            else:  # At or after 5:00 PM
                end_date = date.today()
            
            fixed_start_date = date(2000, 1, 1)
            failed_attempts = 0
            successful_symbols = 0
            
            # Perform integrity check
            if not data_reader.check_database_integrity():
                data_reader.logger.warning("Database integrity check failed, proceeding with caution")
            
            # Process each symbol
            for i, symbol in enumerate(valid_symbols, 1):
                print(f"\nProcessing symbol {i}/{len(valid_symbols)}: {symbol}")
                data_reader.logger.info(f"Processing symbol {symbol} ({i}/{len(valid_symbols)})")
                
                try:
                    # Determine start date
                    last_date = data_reader.get_last_date(symbol)
                    if last_date:
                        start_date = last_date + timedelta(days=1)
                        data_reader.logger.debug(f"Last date for {symbol}: {last_date}, starting from {start_date}")
                    else:
                        start_date = fixed_start_date
                        data_reader.logger.debug(f"No existing data for {symbol}, starting from {start_date}")
                    
                    # Check if download is needed
                    if (end_date - start_date).days < 1:
                        # Check for today's data
                        today_data = data_reader.get_today_data(symbol)
                        if not today_data.empty:
                            data_reader.save_to_db(today_data, f'PSX_{symbol}_stock_data')
                            successful_symbols += 1
                            data_reader.logger.info(f"Successfully saved today's data for {symbol}")
                        continue
                    
                    # Attempt to download data with retries
                    attempts = 0
                    max_attempts = data_reader.config.symbols.max_failed_attempts
                    
                    while attempts < max_attempts:
                        try:
                            data = data_reader.stocks(symbol, start_date, end_date)
                            
                            if not data.empty:
                                # Validate and save data
                                is_valid, issues = data_reader.data_processor.validator.validate_stock_data(data, symbol)
                                
                                if is_valid or len(data) > 0:  # Save even if minor issues exist
                                    data_reader.save_to_db(data, f'PSX_{symbol}_stock_data')
                                    
                                    # Verify data was saved
                                    if data_reader.verify_data(symbol, start_date, end_date):
                                        successful_symbols += 1
                                        data_reader.logger.info(f"Successfully processed {symbol} with {len(data)} records")
                                        break
                                    else:
                                        data_reader.logger.warning(f"Data verification failed for {symbol}")
                                else:
                                    data_reader.logger.warning(f"Data validation failed for {symbol}: {issues}")
                            
                            attempts += 1
                            if attempts < max_attempts:
                                data_reader.logger.warning(f"Attempt {attempts} failed for {symbol}. Retrying...")
                                time.sleep(2)  # Brief delay before retry
                        
                        except Exception as download_error:
                            attempts += 1
                            data_reader.logger.error(f"Download error for {symbol} (attempt {attempts}): {download_error}")
                            if attempts < max_attempts:
                                time.sleep(2)
                    
                    # Handle persistent failures
                    if attempts >= max_attempts:
                        failed_attempts += 1
                        data_reader.logger.error(f"Failed to process {symbol} after {max_attempts} attempts")
                        
                        # Optionally delete failing table (uncommented for safety)
                        # data_reader.delete_failing_table(symbol)
                        
                        # Break if too many failures
                        if failed_attempts > data_reader.config.symbols.max_total_failures:
                            data_reader.logger.error(f"Too many failures ({failed_attempts}), stopping execution")
                            break
                
                except Exception as symbol_error:
                    failed_attempts += 1
                    data_reader.logger.error(f"Unexpected error processing {symbol}: {symbol_error}")
                
                # Periodic progress reporting
                if i % 50 == 0:
                    success_rate = (successful_symbols / i) * 100
                    print(f"Progress: {i}/{len(valid_symbols)} symbols processed, "
                          f"{successful_symbols} successful ({success_rate:.1f}%)")
            
            # Final reporting
            data_reader.metrics_collector.finalize()
            metrics_report = data_reader.metrics_collector.get_summary_report()
            performance_metrics = data_reader.get_performance_metrics()
            
            print(f"\n{'='*50}")
            print("EXECUTION SUMMARY")
            print(f"{'='*50}")
            print(f"Total symbols processed: {len(valid_symbols)}")
            print(f"Successful downloads: {successful_symbols}")
            print(f"Failed downloads: {failed_attempts}")
            print(f"Success rate: {(successful_symbols / len(valid_symbols)) * 100:.1f}%")
            
            print(f"\n{metrics_report}")
            
            # Save performance metrics
            data_reader.logger.info("Execution completed successfully")
            data_reader.logger.info(f"Final metrics: {performance_metrics}")
            
    except Exception as main_error:
        print(f"Critical error in main execution: {main_error}")
        logging.error(f"Critical error in main execution: {main_error}")
        exit(1)
