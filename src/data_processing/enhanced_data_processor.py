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
from .exceptions import DataValidationError
from .data_validator import DataValidator
from .config_manager import DataValidationConfig

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
