"""
Security module for PSX Stock Trading Predictor Dashboard
This module provides security-related functions and utilities.
"""

import re
import logging
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityException(Exception):
    """Custom exception for security-related errors."""
    pass

def validate_symbol(symbol: str) -> bool:
    """
    Validate a stock symbol.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not symbol:
            return False
        
        # Basic validation rules
        if not isinstance(symbol, str):
            return False
        
        # Remove any whitespace
        symbol = symbol.strip()
        
        # Check length (typical PSX symbols are 3-5 characters)
        if not (2 <= len(symbol) <= 6):
            return False
        
        # Check if symbol contains only valid characters
        if not re.match(r'^[A-Z0-9]+$', symbol):
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating symbol: {str(e)}")
        return False

def sanitize_input(input_str: Any) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_str: Input string to sanitize
        
    Returns:
        str: Sanitized string
    """
    try:
        if input_str is None:
            return ""
        
        # Convert to string if not already
        input_str = str(input_str)
        
        # Remove any potentially dangerous characters
        input_str = re.sub(r'[;<>&|]', '', input_str)
        
        # Remove any SQL injection attempts
        input_str = re.sub(r'(\bSELECT\b|\bUNION\b|\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b)',
                          '', input_str, flags=re.IGNORECASE)
        
        # Remove any script tags
        input_str = re.sub(r'<script.*?>.*?</script>', '', input_str, flags=re.IGNORECASE | re.DOTALL)
        
        return input_str.strip()
    except Exception as e:
        logger.error(f"Error sanitizing input: {str(e)}")
        return ""

def validate_date(date_str: Optional[str]) -> bool:
    """
    Validate a date string.
    
    Args:
        date_str: Date string to validate (YYYY-MM-DD format)
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not date_str:
            return False
        
        # Check format
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return False
        
        # Split into components
        year, month, day = map(int, date_str.split('-'))
        
        # Basic validation
        if not (1900 <= year <= 2100):
            return False
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
        
        # Additional validation for specific months
        if month in [4, 6, 9, 11] and day > 30:
            return False
        if month == 2:
            # Check for leap year
            is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            if (is_leap and day > 29) or (not is_leap and day > 28):
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating date: {str(e)}")
        return False
