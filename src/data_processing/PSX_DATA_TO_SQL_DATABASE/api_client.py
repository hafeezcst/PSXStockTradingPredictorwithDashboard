"""
PSX API client for handling web requests and data retrieval.
"""

import requests
import logging
import time
from datetime import date
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from exceptions import APIConnectionError
from config_manager import APIConfig
from monitoring import MetricsCollector

class PSXAPIClient:
    """Handles API communication with PSX website"""
    
    def __init__(self, config: APIConfig, metrics_collector: Optional[MetricsCollector] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = metrics_collector
        
        # Initialize session with retry strategy
        self.session = self._create_session()
        
        # Performance tracking
        self.response_times = []
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
          # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize
        )
        
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': 'PSX-Data-Downloader/1.0 (Educational Purpose)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    def fetch_symbols(self) -> Optional[Dict[str, Any]]:
        """
        Fetch available symbols from PSX
        
        Returns:
            Dictionary containing symbol data or None if failed
        """
        try:
            start_time = time.time()
            response = self.session.get(
                self.config.symbols_url,
                timeout=30
            )
            response_time = time.time() - start_time
            
            response.raise_for_status()
            
            # Track metrics
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt(
                    "symbols_fetch", True, response_time
                )
            
            self.response_times.append(response_time)
            self.logger.debug(f"Successfully fetched symbols in {response_time:.2f}s")
            
            return response.json()
            
        except requests.exceptions.Timeout:
            self.logger.error("Timeout while fetching symbols")
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt("symbols_fetch", False)
            raise APIConnectionError("Request timeout", self.config.symbols_url)
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error while fetching symbols: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt("symbols_fetch", False)
            raise APIConnectionError(f"HTTP error: {e}", self.config.symbols_url, e.response.status_code)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error while fetching symbols: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt("symbols_fetch", False)
            raise APIConnectionError(f"Request failed: {e}", self.config.symbols_url)
            
        except Exception as e:
            self.logger.error(f"Unexpected error while fetching symbols: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt("symbols_fetch", False)
            raise APIConnectionError(f"Unexpected error: {e}", self.config.symbols_url)
    
    def fetch_historical_data(self, symbol: str, target_date: date) -> Optional[str]:
        """
        Fetch historical data for a symbol and date
        
        Args:
            symbol: Stock symbol
            target_date: Target date for data
            
        Returns:
            HTML response text or None if failed
        """
        post_data = {
            "month": target_date.month,
            "year": target_date.year,
            "symbol": symbol
        }
        
        try:
            start_time = time.time()
            
            response = self.session.post(
                self.config.history_url,
                data=post_data,
                timeout=self.config.max_retries * 10  # Dynamic timeout based on retries
            )
            
            response_time = time.time() - start_time
            response.raise_for_status()
            
            # Track metrics
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt(
                    f"{symbol}_{target_date}", True, response_time
                )
            
            self.response_times.append(response_time)
            self.logger.debug(f"Successfully fetched data for {symbol} ({target_date}) in {response_time:.2f}s")
            
            # Add delay to be respectful to the server
            time.sleep(self.config.delay_between_requests)
            
            return response.text
            
        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout while fetching data for {symbol} on {target_date}")
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt(f"{symbol}_{target_date}", False)
            return None
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error for {symbol} on {target_date}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt(f"{symbol}_{target_date}", False)
            return None
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {symbol} on {target_date}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt(f"{symbol}_{target_date}", False)
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected error for {symbol} on {target_date}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_download_attempt(f"{symbol}_{target_date}", False)
            return None
    
    def parse_historical_data(self, html_content: str, symbol: str) -> Dict[str, list]:
        """
        Parse HTML content to extract stock data
        
        Args:
            html_content: HTML response content
            symbol: Stock symbol for logging
            
        Returns:
            Dictionary with parsed stock data
        """
        try:
            soup = BeautifulSoup(html_content, features="html.parser")
            
            headers = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
            stocks = {header: [] for header in headers}
            
            rows = soup.select("tr")
            
            for row in rows:
                cols = [col.getText().strip() for col in row.select("td")]
                
                if len(cols) == len(headers):
                    for header, value in zip(headers, cols):
                        stocks[header].append(value)
            
            self.logger.debug(f"Parsed {len(stocks['TIME'])} records for {symbol}")
            return stocks
            
        except Exception as e:
            self.logger.error(f"Error parsing data for {symbol}: {e}")
            return {header: [] for header in ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']}
    
    def get_average_response_time(self) -> float:
        """Get average response time for recent requests"""
        if not self.response_times:
            return 0.0
        
        # Return average of last 10 requests
        recent_times = self.response_times[-10:]
        return sum(recent_times) / len(recent_times)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.response_times:
            return {
                'total_requests': 0,
                'average_response_time': 0.0,
                'min_response_time': 0.0,
                'max_response_time': 0.0
            }
        
        return {
            'total_requests': len(self.response_times),
            'average_response_time': sum(self.response_times) / len(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'recent_average': self.get_average_response_time()
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.response_times = []
        self.logger.debug("Performance statistics reset")
    
    def close(self):
        """Close the session and clean up resources"""
        try:
            self.session.close()
            self.logger.debug("API client session closed")
        except Exception as e:
            self.logger.error(f"Error closing API client session: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
