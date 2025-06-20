"""
Enhanced data processing utilities for PSX stock data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from exceptions import DataValidationError
from data_validator import DataValidator
from config_manager import DataValidationConfig

class EnhancedDataProcessor:
    """Enhanced data processor with advanced features"""
    
    def __init__(self, validation_config: DataValidationConfig):
        self.validator = DataValidator(validation_config)
        self.logger = logging.getLogger(__name__)
        self.headers = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    
    def parse_raw_data(self, raw_data: Dict[str, list], symbol: str) -> pd.DataFrame:
        """
        Parse raw data dictionary into pandas DataFrame
        
        Args:
            raw_data: Dictionary with stock data lists
            symbol: Stock symbol for logging
            
        Returns:
            Processed DataFrame
        """
        try:
            if not raw_data or not any(raw_data.values()):
                self.logger.warning(f"No data to parse for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(raw_data, columns=self.headers)
            
            if df.empty:
                return df
            
            # Process TIME column
            df = self._process_time_column(df, symbol)
            
            # Process numeric columns
            df = self._process_numeric_columns(df)
            
            # Set index and sort
            if 'TIME' in df.columns:
                df = df.set_index('TIME')
                df = df.sort_index()
            
            # Rename columns to proper case
            df = df.rename(columns=str.title)
            df.index.name = "Date"
            
            self.logger.debug(f"Parsed {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _process_time_column(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process the TIME column to convert to datetime"""
        if 'TIME' not in df.columns:
            return df
        
        processed_rows = []
        today = date.today()
        
        for index, row in df.iterrows():
            time_value = row['TIME']
            
            try:
                # Parse the date
                parsed_date = datetime.strptime(time_value, "%b %d, %Y").date()
                
                # Skip future dates
                if parsed_date > today:
                    self.logger.warning(f"Skipping future date for {symbol}: {parsed_date}")
                    continue
                
                # Update the row with parsed date
                row['TIME'] = parsed_date
                processed_rows.append(row)
                
            except ValueError as e:
                self.logger.error(f"Error parsing date '{time_value}' for {symbol}: {e}")
                continue
        
        if processed_rows:
            return pd.DataFrame(processed_rows)
        else:
            return pd.DataFrame(columns=df.columns)
    
    def _process_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process numeric columns (OHLCV)"""
        if df.empty:
            return df
        
        numeric_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        for column in numeric_columns:
            if column in df.columns:
                try:
                    # Remove commas and convert to float
                    df[column] = df[column].astype(str).str.replace(",", "").astype(float)
                except Exception as e:
                    self.logger.error(f"Error converting column {column} to float: {e}")
        
        return df
    
    def combine_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple DataFrames into one
        
        Args:
            dataframes: List of DataFrames to combine
            
        Returns:
            Combined and sorted DataFrame
        """
        # Filter out empty DataFrames
        valid_dataframes = [df for df in dataframes if not df.empty]
        
        if not valid_dataframes:
            self.logger.warning("No valid DataFrames to combine")
            return pd.DataFrame()
        
        try:
            # Concatenate all DataFrames
            combined = pd.concat(valid_dataframes, ignore_index=False)
            
            # Remove duplicates (keep last occurrence)
            combined = combined[~combined.index.duplicated(keep='last')]
            
            # Sort by date
            combined = combined.sort_index()
            
            self.logger.debug(f"Combined {len(combined)} records from {len(valid_dataframes)} DataFrames")
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining DataFrames: {e}")
            return pd.DataFrame()
    
    def generate_date_range(self, start_date: date, end_date: date) -> List[date]:
        """
        Generate optimized date range for data download
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dates (first day of each month in range)
        """
        try:
            # Validate date range
            is_valid, message = self.validator.validate_date_range(start_date, end_date)
            if not is_valid:
                raise DataValidationError(f"Invalid date range: {message}")
            
            dates = []
            current_date = start_date.replace(day=1)  # Start from first of month
            end_month = end_date.replace(day=1)
            
            while current_date <= end_month:
                dates.append(current_date)
                current_date = (current_date + relativedelta(months=1)).replace(day=1)
            
            self.logger.debug(f"Generated {len(dates)} dates from {start_date} to {end_date}")
            return dates
            
        except Exception as e:
            self.logger.error(f"Error generating date range: {e}")
            return [start_date] if start_date <= end_date else []
    
    def process_and_validate(self, raw_data: Dict[str, list], symbol: str) -> Tuple[pd.DataFrame, bool, List[str]]:
        """
        Process raw data and validate it
        
        Args:
            raw_data: Raw data dictionary
            symbol: Stock symbol
            
        Returns:
            Tuple of (processed_dataframe, is_valid, issues_list)
        """
        # Parse the data
        df = self.parse_raw_data(raw_data, symbol)
        
        if df.empty:
            return df, False, ["No data to process"]
        
        # Validate the data
        is_valid, issues = self.validator.validate_stock_data(df, symbol)
        
        # Clean the data if there are issues but it's still usable
        if not is_valid and not df.empty:
            cleaned_df = self.validator.clean_data(df, symbol)
            
            # Re-validate cleaned data
            is_valid_after_cleaning, remaining_issues = self.validator.validate_stock_data(cleaned_df, symbol)
            
            if is_valid_after_cleaning:
                return cleaned_df, True, ["Data cleaned successfully"]
            else:
                return cleaned_df, False, remaining_issues
        
        return df, is_valid, issues
    
    def calculate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic statistics for the dataset"""
        if df.empty:
            return {}
        
        stats = {}
        
        try:
            # Price statistics
            if 'Close' in df.columns:
                stats['avg_close'] = df['Close'].mean()
                stats['max_close'] = df['Close'].max()
                stats['min_close'] = df['Close'].min()
                stats['close_std'] = df['Close'].std()
            
            # Volume statistics
            if 'Volume' in df.columns:
                stats['avg_volume'] = df['Volume'].mean()
                stats['max_volume'] = df['Volume'].max()
                stats['min_volume'] = df['Volume'].min()
                stats['volume_std'] = df['Volume'].std()
            
            # Data quality metrics
            stats['total_records'] = len(df)
            stats['date_range_days'] = (df.index.max() - df.index.min()).days if len(df) > 1 else 0
            
            # Calculate returns if we have closing prices
            if 'Close' in df.columns and len(df) > 1:
                returns = df['Close'].pct_change().dropna()
                stats['avg_daily_return'] = returns.mean()
                stats['volatility'] = returns.std()
                stats['max_daily_return'] = returns.max()
                stats['min_daily_return'] = returns.min()
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
        
        return stats
    
    def check_duplicates(self, df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Check for duplicate records in the DataFrame
        
        Args:
            df: DataFrame to check for duplicates
            symbol: Stock symbol for logging
            
        Returns:
            Dictionary with duplicate analysis results
        """
        if df.empty:
            return {
                'has_duplicates': False,
                'duplicate_count': 0,
                'duplicate_indices': [],
                'duplicate_summary': "No data to check"
            }
        
        try:
            # Check for exact duplicates (all columns)
            exact_duplicates = df.duplicated()
            exact_duplicate_count = exact_duplicates.sum()
            
            # Check for date-based duplicates (same index/date)
            date_duplicates = df.index.duplicated()
            date_duplicate_count = date_duplicates.sum()
            
            # Get duplicate indices
            duplicate_indices = df.index[exact_duplicates].tolist()
            date_duplicate_indices = df.index[date_duplicates].tolist()
            
            # Create summary
            summary = {
                'has_duplicates': exact_duplicate_count > 0 or date_duplicate_count > 0,
                'exact_duplicate_count': int(exact_duplicate_count),
                'date_duplicate_count': int(date_duplicate_count),
                'duplicate_indices': duplicate_indices,
                'date_duplicate_indices': date_duplicate_indices,
                'total_records': len(df),
                'unique_records_after_removal': len(df) - exact_duplicate_count
            }
            
            if summary['has_duplicates']:
                self.logger.warning(
                    f"Found duplicates in {symbol}: "
                    f"{exact_duplicate_count} exact duplicates, "
                    f"{date_duplicate_count} date duplicates"
                )
            else:
                self.logger.debug(f"No duplicates found in {symbol}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error checking duplicates for {symbol}: {e}")
            return {
                'has_duplicates': False,
                'duplicate_count': 0,
                'duplicate_indices': [],
                'error': str(e)
            }
    
    def remove_duplicates(self, df: pd.DataFrame, symbol: str, 
                         strategy: str = 'keep_last') -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Remove duplicate records from DataFrame
        
        Args:
            df: DataFrame to remove duplicates from
            symbol: Stock symbol for logging
            strategy: How to handle duplicates ('keep_last', 'keep_first', 'remove_all')
            
        Returns:
            Tuple of (cleaned_dataframe, removal_summary)
        """
        if df.empty:
            return df, {'removed_count': 0, 'strategy': strategy}
        
        try:
            original_count = len(df)
            
            # First check what duplicates exist
            duplicate_summary = self.check_duplicates(df, symbol)
            
            if not duplicate_summary['has_duplicates']:
                return df, {
                    'removed_count': 0,
                    'strategy': strategy,
                    'original_count': original_count,
                    'final_count': original_count
                }
            
            # Remove exact duplicates first
            if strategy == 'keep_last':
                cleaned_df = df[~df.duplicated(keep='last')]
            elif strategy == 'keep_first':
                cleaned_df = df[~df.duplicated(keep='first')]
            elif strategy == 'remove_all':
                cleaned_df = df[~df.duplicated(keep=False)]
            else:
                self.logger.warning(f"Unknown strategy '{strategy}', using 'keep_last'")
                cleaned_df = df[~df.duplicated(keep='last')]
            
            # Handle date-based duplicates (same date, different data)
            # Keep the record with highest volume or latest update
            if cleaned_df.index.duplicated().any():
                cleaned_df = self._resolve_date_duplicates(cleaned_df, symbol, strategy)
            
            final_count = len(cleaned_df)
            removed_count = original_count - final_count
            
            summary = {
                'removed_count': removed_count,
                'strategy': strategy,
                'original_count': original_count,
                'final_count': final_count,
                'removal_percentage': (removed_count / original_count * 100) if original_count > 0 else 0
            }
            
            if removed_count > 0:
                self.logger.info(
                    f"Removed {removed_count} duplicate records from {symbol} "
                    f"({summary['removal_percentage']:.1f}%)"
                )
            
            return cleaned_df, summary
            
        except Exception as e:
            self.logger.error(f"Error removing duplicates from {symbol}: {e}")
            return df, {'error': str(e), 'removed_count': 0}
    
    def _resolve_date_duplicates(self, df: pd.DataFrame, symbol: str, 
                               strategy: str) -> pd.DataFrame:
        """
        Resolve duplicates that have the same date but different data
        
        Args:
            df: DataFrame with potential date duplicates
            symbol: Stock symbol for logging
            strategy: Resolution strategy
            
        Returns:
            DataFrame with date duplicates resolved
        """
        try:
            if not df.index.duplicated().any():
                return df
            
            # Group by date and resolve duplicates
            grouped = df.groupby(df.index)
            resolved_records = []
            
            for date, group in grouped:
                if len(group) == 1:
                    # No duplicates for this date
                    resolved_records.append(group.iloc[0])
                else:
                    # Multiple records for same date - resolve based on strategy
                    if strategy == 'keep_last':
                        # Keep the record that was added last (highest row number)
                        resolved_records.append(group.iloc[-1])
                    elif strategy == 'keep_first':
                        resolved_records.append(group.iloc[0])
                    elif strategy == 'remove_all':
                        # Skip all duplicates
                        continue
                    else:
                        # Default: keep record with highest volume or last record
                        if 'Volume' in group.columns:
                            best_record = group.loc[group['Volume'].idxmax()]
                        else:
                            best_record = group.iloc[-1]
                        resolved_records.append(best_record)
                    
                    self.logger.debug(
                        f"Resolved {len(group)} duplicate records for {symbol} on {date}"
                    )
            
            if resolved_records:
                result_df = pd.DataFrame(resolved_records)
                result_df.index.name = df.index.name
                return result_df.sort_index()
            else:
                return pd.DataFrame(columns=df.columns)
                
        except Exception as e:
            self.logger.error(f"Error resolving date duplicates for {symbol}: {e}")
            return df
    
    def verify_duplicate_removal(self, df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Verify that duplicates have been successfully removed
        
        Args:
            df: DataFrame to verify
            symbol: Stock symbol for logging
            
        Returns:
            Verification results dictionary
        """
        try:
            duplicate_check = self.check_duplicates(df, symbol)
            
            verification = {
                'is_clean': not duplicate_check['has_duplicates'],
                'remaining_exact_duplicates': duplicate_check.get('exact_duplicate_count', 0),
                'remaining_date_duplicates': duplicate_check.get('date_duplicate_count', 0),
                'total_records': len(df),
                'verification_passed': True
            }
            
            if verification['is_clean']:
                self.logger.info(f"Verification passed: No duplicates found in {symbol}")
            else:
                self.logger.warning(
                    f"Verification failed: {symbol} still has duplicates - "
                    f"Exact: {verification['remaining_exact_duplicates']}, "
                    f"Date: {verification['remaining_date_duplicates']}"
                )
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Error verifying duplicate removal for {symbol}: {e}")
            return {
                'is_clean': False,
                'verification_passed': False,
                'error': str(e)
            }
    
    def process_with_duplicate_removal(self, raw_data: Dict[str, list], symbol: str,
                                     remove_duplicates: bool = True,
                                     duplicate_strategy: str = 'keep_last') -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Complete processing pipeline with duplicate removal
        
        Args:
            raw_data: Raw data dictionary
            symbol: Stock symbol
            remove_duplicates: Whether to remove duplicates
            duplicate_strategy: Strategy for duplicate removal
            
        Returns:
            Tuple of (processed_dataframe, processing_summary)
        """
        summary = {
            'symbol': symbol,
            'processing_steps': [],
            'duplicate_removal': None,
            'validation': None,
            'final_status': 'success'
        }
        
        try:
            # Step 1: Parse raw data
            df = self.parse_raw_data(raw_data, symbol)
            summary['processing_steps'].append('data_parsed')
            
            if df.empty:
                summary['final_status'] = 'no_data'
                return df, summary
            
            # Step 2: Check for duplicates before processing
            initial_duplicate_check = self.check_duplicates(df, symbol)
            summary['initial_duplicates'] = initial_duplicate_check
            
            # Step 3: Remove duplicates if requested
            if remove_duplicates and initial_duplicate_check['has_duplicates']:
                df, removal_summary = self.remove_duplicates(df, symbol, duplicate_strategy)
                summary['duplicate_removal'] = removal_summary
                summary['processing_steps'].append('duplicates_removed')
                
                # Verify removal
                verification = self.verify_duplicate_removal(df, symbol)
                summary['duplicate_verification'] = verification
                
                if not verification['is_clean']:
                    summary['final_status'] = 'duplicates_remain'
            
            # Step 4: Validate processed data
            is_valid, issues = self.validator.validate_stock_data(df, symbol)
            summary['validation'] = {
                'is_valid': is_valid,
                'issues': issues
            }
            summary['processing_steps'].append('data_validated')
            
            # Step 5: Calculate final statistics
            stats = self.calculate_basic_statistics(df)
            summary['statistics'] = stats
            summary['processing_steps'].append('statistics_calculated')
            
            return df, summary
            
        except Exception as e:
            self.logger.error(f"Error in complete processing pipeline for {symbol}: {e}")
            summary['final_status'] = 'error'
            summary['error'] = str(e)
            return pd.DataFrame(), summary
    
    def check_database_duplicates(self, connection, table_name: str) -> Dict[str, any]:
        """
        Check for duplicates directly in the database table
        
        Args:
            connection: Database connection object
            table_name: Name of the table to check
            
        Returns:
            Dictionary with duplicate analysis results
        """
        try:
            # Query to find exact duplicates
            exact_duplicate_query = f"""
            SELECT Date, COUNT(*) as duplicate_count
            FROM {table_name}
            GROUP BY Date, Open, High, Low, Close, Volume
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC
            """
            
            # Query to find date duplicates (same date, different data)
            date_duplicate_query = f"""
            SELECT Date, COUNT(*) as record_count
            FROM {table_name}
            GROUP BY Date
            HAVING COUNT(*) > 1
            ORDER BY record_count DESC
            """
            
            # Execute queries
            exact_duplicates = pd.read_sql_query(exact_duplicate_query, connection)
            date_duplicates = pd.read_sql_query(date_duplicate_query, connection)
            
            # Get total record count
            total_count_query = f"SELECT COUNT(*) as total FROM {table_name}"
            total_records = pd.read_sql_query(total_count_query, connection).iloc[0]['total']
            
            summary = {
                'table_name': table_name,
                'total_records': int(total_records),
                'exact_duplicate_groups': len(exact_duplicates),
                'date_duplicate_groups': len(date_duplicates),
                'exact_duplicate_records': int(exact_duplicates['duplicate_count'].sum()) if not exact_duplicates.empty else 0,
                'date_duplicate_records': int(date_duplicates['record_count'].sum()) if not date_duplicates.empty else 0,
                'has_duplicates': len(exact_duplicates) > 0 or len(date_duplicates) > 0,
                'exact_duplicates_detail': exact_duplicates.to_dict('records') if not exact_duplicates.empty else [],
                'date_duplicates_detail': date_duplicates.to_dict('records') if not date_duplicates.empty else []
            }
            
            if summary['has_duplicates']:
                self.logger.warning(
                    f"Found duplicates in table {table_name}: "
                    f"{summary['exact_duplicate_groups']} exact duplicate groups, "
                    f"{summary['date_duplicate_groups']} date duplicate groups"
                )
            else:
                self.logger.info(f"No duplicates found in table {table_name}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error checking database duplicates for table {table_name}: {e}")
            return {
                'table_name': table_name,
                'error': str(e),
                'has_duplicates': False
            }
    
    def remove_database_duplicates(self, connection, table_name: str, 
                                 strategy: str = 'keep_latest') -> Dict[str, any]:
        """
        Remove duplicates directly from database table
        
        Args:
            connection: Database connection object
            table_name: Name of the table to clean
            strategy: Strategy for keeping records ('keep_latest', 'keep_earliest', 'keep_highest_volume')
            
        Returns:
            Summary of removal operation
        """
        try:
            # First check what duplicates exist
            duplicate_check = self.check_database_duplicates(connection, table_name)
            
            if not duplicate_check['has_duplicates']:
                return {
                    'table_name': table_name,
                    'removed_count': 0,
                    'strategy': strategy,
                    'status': 'no_duplicates_found'
                }
            
            original_count = duplicate_check['total_records']
            
            # Create a temporary table with unique records
            temp_table = f"{table_name}_temp_dedup"
            
            # Choose strategy for keeping records
            if strategy == 'keep_latest':
                # Keep the record with the maximum rowid (assuming rowid represents insertion order)
                dedup_query = f"""
                CREATE TABLE {temp_table} AS
                SELECT Date, Open, High, Low, Close, Volume
                FROM {table_name} t1
                WHERE t1.rowid = (
                    SELECT MAX(t2.rowid)
                    FROM {table_name} t2
                    WHERE t2.Date = t1.Date
                )
                """
            elif strategy == 'keep_earliest':
                dedup_query = f"""
                CREATE TABLE {temp_table} AS
                SELECT Date, Open, High, Low, Close, Volume
                FROM {table_name} t1
                WHERE t1.rowid = (
                    SELECT MIN(t2.rowid)
                    FROM {table_name} t2
                    WHERE t2.Date = t1.Date
                )
                """
            elif strategy == 'keep_highest_volume':
                dedup_query = f"""
                CREATE TABLE {temp_table} AS
                SELECT Date, Open, High, Low, Close, Volume
                FROM {table_name} t1
                WHERE t1.rowid = (
                    SELECT t2.rowid
                    FROM {table_name} t2
                    WHERE t2.Date = t1.Date
                    ORDER BY t2.Volume DESC, t2.rowid DESC
                    LIMIT 1
                )
                """
            else:
                # Default to keep_latest
                dedup_query = f"""
                CREATE TABLE {temp_table} AS
                SELECT Date, Open, High, Low, Close, Volume
                FROM {table_name} t1
                WHERE t1.rowid = (
                    SELECT MAX(t2.rowid)
                    FROM {table_name} t2
                    WHERE t2.Date = t1.Date
                )
                """
            
            # Execute the deduplication
            cursor = connection.cursor()
            
            # Create temp table with deduplicated data
            cursor.execute(dedup_query)
            
            # Get count of deduplicated records
            cursor.execute(f"SELECT COUNT(*) FROM {temp_table}")
            final_count = cursor.fetchone()[0]
            
            # Replace original table with deduplicated data
            cursor.execute(f"DELETE FROM {table_name}")
            cursor.execute(f"""
                INSERT INTO {table_name} (Date, Open, High, Low, Close, Volume)
                SELECT Date, Open, High, Low, Close, Volume FROM {temp_table}
            """)
            
            # Clean up temp table
            cursor.execute(f"DROP TABLE {temp_table}")
            
            # Commit changes
            connection.commit()
            
            removed_count = original_count - final_count
            
            summary = {
                'table_name': table_name,
                'strategy': strategy,
                'original_count': original_count,
                'final_count': final_count,
                'removed_count': removed_count,
                'removal_percentage': (removed_count / original_count * 100) if original_count > 0 else 0,
                'status': 'success'
            }
            
            self.logger.info(
                f"Removed {removed_count} duplicate records from table {table_name} "
                f"({summary['removal_percentage']:.1f}%) using strategy '{strategy}'"
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error removing database duplicates from table {table_name}: {e}")
            # Rollback in case of error
            try:
                connection.rollback()
                # Try to clean up temp table if it exists
                cursor = connection.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}_temp_dedup")
                connection.commit()
            except:
                pass
            
            return {
                'table_name': table_name,
                'error': str(e),
                'status': 'error'
            }
    
    def verify_database_cleanup(self, connection, table_name: str) -> Dict[str, any]:
        """
        Verify that database table is clean of duplicates
        
        Args:
            connection: Database connection object
            table_name: Name of the table to verify
            
        Returns:
            Verification results
        """
        try:
            # Check for remaining duplicates
            duplicate_check = self.check_database_duplicates(connection, table_name)
            
            verification = {
                'table_name': table_name,
                'is_clean': not duplicate_check['has_duplicates'],
                'total_records': duplicate_check['total_records'],
                'remaining_exact_duplicates': duplicate_check.get('exact_duplicate_groups', 0),
                'remaining_date_duplicates': duplicate_check.get('date_duplicate_groups', 0),
                'verification_passed': not duplicate_check['has_duplicates']
            }
            
            if verification['is_clean']:
                self.logger.info(f"Verification passed: Table {table_name} is clean of duplicates")
            else:
                self.logger.warning(
                    f"Verification failed: Table {table_name} still has duplicates"
                )
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Error verifying database cleanup for table {table_name}: {e}")
            return {
                'table_name': table_name,
                'verification_passed': False,
                'error': str(e)
            }
    
    def bulk_database_duplicate_cleanup(self, connection, table_names: List[str],
                                      strategy: str = 'keep_latest') -> Dict[str, any]:
        """
        Clean duplicates from multiple database tables
        
        Args:
            connection: Database connection object
            table_names: List of table names to clean
            strategy: Strategy for duplicate removal
            
        Returns:
            Summary of bulk cleanup operation
        """
        results = {
            'strategy': strategy,
            'total_tables': len(table_names),
            'successful_cleanups': 0,
            'failed_cleanups': 0,
            'total_removed_records': 0,
            'table_results': {},
            'errors': []
        }
        
        for table_name in table_names:
            try:
                self.logger.info(f"Starting duplicate cleanup for table: {table_name}")
                
                # Clean duplicates from this table
                cleanup_result = self.remove_database_duplicates(connection, table_name, strategy)
                
                if cleanup_result.get('status') == 'success':
                    results['successful_cleanups'] += 1
                    results['total_removed_records'] += cleanup_result.get('removed_count', 0)
                    
                    # Verify the cleanup
                    verification = self.verify_database_cleanup(connection, table_name)
                    cleanup_result['verification'] = verification
                    
                elif cleanup_result.get('status') == 'no_duplicates_found':
                    results['successful_cleanups'] += 1
                else:
                    results['failed_cleanups'] += 1
                    results['errors'].append(f"{table_name}: {cleanup_result.get('error', 'Unknown error')}")
                
                results['table_results'][table_name] = cleanup_result
                
            except Exception as e:
                self.logger.error(f"Error processing table {table_name}: {e}")
                results['failed_cleanups'] += 1
                results['errors'].append(f"{table_name}: {str(e)}")
                results['table_results'][table_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Overall summary
        results['success_rate'] = (results['successful_cleanups'] / results['total_tables'] * 100) if results['total_tables'] > 0 else 0
        
        self.logger.info(
            f"Bulk cleanup completed: {results['successful_cleanups']}/{results['total_tables']} tables "
            f"({results['success_rate']:.1f}% success rate), "
            f"{results['total_removed_records']} total records removed"
        )
        
        return results
