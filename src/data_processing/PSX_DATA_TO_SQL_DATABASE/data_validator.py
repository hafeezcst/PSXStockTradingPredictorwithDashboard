"""
Data validation utilities for PSX stock data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import List, Tuple, Optional
from exceptions import DataValidationError
from config_manager import DataValidationConfig

class DataValidator:
    """Validates stock data quality and integrity"""
    
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_stock_data(self, data: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of stock data
        
        Args:
            data: DataFrame with stock data
            symbol: Stock symbol for logging
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if data.empty:
            issues.append("Data is empty")
            return False, issues
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        if issues:  # If basic structure is wrong, don't continue
            return False, issues
        
        # Validate OHLC relationships
        if self.config.check_ohlc_relationships:
            ohlc_issues = self._validate_ohlc_relationships(data)
            issues.extend(ohlc_issues)
        
        # Validate price ranges
        price_issues = self._validate_price_ranges(data, symbol)
        issues.extend(price_issues)
        
        # Validate volume
        volume_issues = self._validate_volume(data)
        issues.extend(volume_issues)
        
        # Validate date continuity
        if self.config.validate_date_continuity:
            date_issues = self._validate_date_continuity(data)
            issues.extend(date_issues)
        
        # Check for null values
        null_issues = self._validate_null_values(data)
        issues.extend(null_issues)
        
        # Check for duplicate dates
        duplicate_issues = self._validate_duplicates(data)
        issues.extend(duplicate_issues)
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            self.logger.warning(f"Data validation failed for {symbol}: {'; '.join(issues)}")
        else:
            self.logger.debug(f"Data validation passed for {symbol}")
        
        return is_valid, issues
    
    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> List[str]:
        """Validate OHLC price relationships"""
        issues = []
        
        # High should be >= Open, Close, Low
        if (data['High'] < data['Open']).any():
            issues.append("High price is less than Open price in some records")
        if (data['High'] < data['Close']).any():
            issues.append("High price is less than Close price in some records")
        if (data['High'] < data['Low']).any():
            issues.append("High price is less than Low price in some records")
        
        # Low should be <= Open, Close, High
        if (data['Low'] > data['Open']).any():
            issues.append("Low price is greater than Open price in some records")
        if (data['Low'] > data['Close']).any():
            issues.append("Low price is greater than Close price in some records")
        
        return issues
    
    def _validate_price_ranges(self, data: pd.DataFrame, symbol: str) -> List[str]:
        """Validate price ranges for reasonableness"""
        issues = []
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (data[col] <= 0).any():
                issues.append(f"Non-positive values found in {col}")
        
        # Check for extreme price changes
        if len(data) > 1:
            for col in price_columns:
                price_changes = data[col].pct_change().abs() * 100
                extreme_changes = price_changes > self.config.max_price_change_percent
                
                if extreme_changes.any():
                    max_change = price_changes.max()
                    issues.append(f"Extreme price change in {col}: {max_change:.1f}%")
        
        return issues
    
    def _validate_volume(self, data: pd.DataFrame) -> List[str]:
        """Validate volume data"""
        issues = []
        
        # Check for negative volume
        if (data['Volume'] < self.config.min_volume).any():
            issues.append(f"Volume below minimum threshold ({self.config.min_volume})")
        
        # Check for extremely high volume
        if (data['Volume'] > self.config.max_volume).any():
            issues.append(f"Volume above maximum threshold ({self.config.max_volume})")
        
        return issues
    
    def _validate_date_continuity(self, data: pd.DataFrame) -> List[str]:
        """Validate date continuity (accounting for weekends and holidays)"""
        issues = []
        
        if len(data) < 2:
            return issues
        
        # Sort by date to ensure proper order
        data_sorted = data.sort_index()
        dates = data_sorted.index
        
        # Check for large gaps (more than 7 days, accounting for weekends)
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > 7:  # More than a week gap
                issues.append(f"Large date gap: {gap} days between {dates[i-1]} and {dates[i]}")
        
        return issues
    
    def _validate_null_values(self, data: pd.DataFrame) -> List[str]:
        """Check for null or missing values"""
        issues = []
        
        for column in data.columns:
            null_count = data[column].isnull().sum()
            if null_count > 0:
                issues.append(f"Found {null_count} null values in {column}")
        
        return issues
    
    def _validate_duplicates(self, data: pd.DataFrame) -> List[str]:
        """Check for duplicate date entries"""
        issues = []
        
        duplicate_dates = data.index.duplicated()
        if duplicate_dates.any():
            duplicate_count = duplicate_dates.sum()
            issues.append(f"Found {duplicate_count} duplicate date entries")
        
        return issues
    
    def clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean data by removing or fixing common issues
        
        Args:
            data: Raw stock data
            symbol: Stock symbol for logging
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data
        
        cleaned_data = data.copy()
        cleaning_actions = []
        
        # Remove duplicate dates (keep last occurrence)
        if cleaned_data.index.duplicated().any():
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='last')]
            after_count = len(cleaned_data)
            cleaning_actions.append(f"Removed {before_count - after_count} duplicate dates")
        
        # Remove rows with null values in critical columns
        critical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        before_count = len(cleaned_data)
        cleaned_data = cleaned_data.dropna(subset=critical_columns)
        after_count = len(cleaned_data)
        if before_count != after_count:
            cleaning_actions.append(f"Removed {before_count - after_count} rows with null values")
        
        # Remove rows with non-positive prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data[cleaned_data[col] > 0]
            after_count = len(cleaned_data)
            if before_count != after_count:
                cleaning_actions.append(f"Removed {before_count - after_count} rows with non-positive {col}")
        
        # Cap volume at maximum threshold
        if (cleaned_data['Volume'] > self.config.max_volume).any():
            over_threshold = (cleaned_data['Volume'] > self.config.max_volume).sum()
            cleaned_data.loc[cleaned_data['Volume'] > self.config.max_volume, 'Volume'] = self.config.max_volume
            cleaning_actions.append(f"Capped {over_threshold} volume values at maximum threshold")
        
        # Sort by date
        cleaned_data = cleaned_data.sort_index()
        
        if cleaning_actions:
            self.logger.info(f"Data cleaning for {symbol}: {'; '.join(cleaning_actions)}")
        
        return cleaned_data
    
    def validate_date_range(self, start_date: date, end_date: date) -> Tuple[bool, str]:
        """Validate date range parameters"""
        if start_date > end_date:
            return False, "Start date cannot be after end date"
        
        if end_date > date.today():
            return False, "End date cannot be in the future"
        
        if start_date < date(1990, 1, 1):
            return False, "Start date is too far in the past"
        
        date_range_days = (end_date - start_date).days
        if date_range_days > 10000:  # ~27 years
            return False, "Date range is too large"
        
        return True, "Date range is valid"
