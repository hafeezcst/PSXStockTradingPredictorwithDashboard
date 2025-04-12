"""
Security module for PSX Stock Trading Predictor
Provides functions for input validation, sanitization, and rate limiting
"""

import re
import time
import logging
from typing import Any, Optional, Dict, List

# Configure logging
logger = logging.getLogger(__name__)

class SecurityException(Exception):
    """Exception raised for security-related issues."""
    pass

def validate_symbol(symbol: Any) -> str:
    """
    Validate a stock symbol to ensure it contains only valid characters.
    
    Args:
        symbol: The symbol to validate
        
    Returns:
        A clean, validated symbol string
    """
    if not isinstance(symbol, str):
        logger.warning(f"Non-string symbol provided: {type(symbol)}")
        return ""
    
    # Only allow alphanumeric and dot in symbols
    clean_symbol = re.sub(r'[^A-Za-z0-9\.]', '', symbol)
    
    # Warn if the symbol was changed
    if clean_symbol != symbol:
        logger.warning(f"Symbol was sanitized: {symbol} -> {clean_symbol}")
    
    return clean_symbol

def sanitize_input(input_str: Any) -> str:
    """
    Sanitize input strings to prevent injection attacks.
    
    Args:
        input_str: The input string to sanitize
        
    Returns:
        A sanitized string
    """
    if not isinstance(input_str, str):
        logger.warning(f"Non-string input provided: {type(input_str)}")
        return ""
    
    # Remove potentially dangerous characters
    clean_str = re.sub(r'[^\w\s\.\-_]', '', input_str)
    
    # Warn if the string was changed
    if clean_str != input_str:
        logger.warning(f"Input was sanitized: {input_str[:20]}... -> {clean_str[:20]}...")
    
    return clean_str

def sanitize_sql(sql: str) -> str:
    """
    Sanitize SQL queries to prevent SQL injection.
    
    Args:
        sql: The SQL query to sanitize
        
    Returns:
        A sanitized SQL query
    """
    # This is a basic sanitization - in production, always use parameterized queries
    if not isinstance(sql, str):
        logger.warning(f"Non-string SQL provided: {type(sql)}")
        return ""
    
    # Remove risky SQL commands
    risky_keywords = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", 
        "TRUNCATE", "EXEC", "UNION", "CREATE", "--", ";"
    ]
    
    clean_sql = sql
    for keyword in risky_keywords:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        clean_sql = pattern.sub("", clean_sql)
    
    return clean_sql

class RateLimiter:
    """Rate limiter to prevent abuse of the application."""
    
    def __init__(self, max_requests: int = 30, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times: Dict[str, List[float]] = {}  # IP -> list of request timestamps
    
    def check_rate_limit(self, ip_address: str) -> bool:
        """
        Check if a request from the given IP exceeds the rate limit.
        
        Args:
            ip_address: IP address of the requester
            
        Returns:
            bool: True if request is allowed, False if rate limit exceeded
        """
        current_time = time.time()
        
        # Initialize if this is the first request from this IP
        if ip_address not in self.request_times:
            self.request_times[ip_address] = []
        
        # Remove requests outside the time window
        self.request_times[ip_address] = [
            t for t in self.request_times[ip_address] 
            if current_time - t < self.time_window
        ]
        
        # Check if the rate limit is exceeded
        if len(self.request_times[ip_address]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for IP: {ip_address}")
            return False
        
        # Add the current request
        self.request_times[ip_address].append(current_time)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter()
