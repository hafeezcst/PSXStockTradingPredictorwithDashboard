from typing import Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import logging
from .config import PSXConfig

class SeleniumManager:
    """Manages Selenium WebDriver operations"""
    
    def __init__(self):
        """Initialize Selenium manager"""
        self.driver: Optional[webdriver.Chrome] = None
    
    def setup_driver(self) -> bool:
        """
        Initialize Selenium WebDriver
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            chrome_options = Options()
            selenium_options = PSXConfig.get_selenium_options()
            
            if selenium_options.get('headless'):
                chrome_options.add_argument("--headless")
            if selenium_options.get('no_sandbox'):
                chrome_options.add_argument("--no-sandbox")
            if selenium_options.get('disable_dev_shm_usage'):
                chrome_options.add_argument("--disable-dev-shm-usage")
            if selenium_options.get('disable_gpu'):
                chrome_options.add_argument("--disable-gpu")
            if selenium_options.get('window_size'):
                chrome_options.add_argument(f"--window-size={selenium_options['window_size']}")
            if selenium_options.get('user_agent'):
                chrome_options.add_argument(f"user-agent={selenium_options['user_agent']}")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(PSXConfig.REQUEST_TIMEOUT)
            logging.info("Selenium WebDriver initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing Selenium WebDriver: {e}")
            return False
    
    def close_driver(self) -> None:
        """Close Selenium WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("Selenium WebDriver closed")
            except Exception as e:
                logging.error(f"Error closing Selenium WebDriver: {e}")
            finally:
                self.driver = None
    
    def wait_for_element(self, by: By, value: str, timeout: int = 30) -> bool:
        """
        Wait for an element to be present
        
        Args:
            by: Selenium By locator
            value: Element locator value
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if element found, False otherwise
        """
        if not self.driver:
            logging.error("WebDriver not initialized")
            return False
            
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return True
        except TimeoutException:
            logging.warning(f"Timeout waiting for element: {value}")
            return False
        except Exception as e:
            logging.error(f"Error waiting for element {value}: {e}")
            return False
    
    def get_page_source(self, url: str) -> Optional[str]:
        """
        Get page source for a URL
        
        Args:
            url: URL to load
            
        Returns:
            Optional[str]: Page source if successful, None otherwise
        """
        if not self.driver:
            if not self.setup_driver():
                return None
                
        try:
            self.driver.get(url)
            return self.driver.page_source
        except WebDriverException as e:
            logging.error(f"Error loading URL {url}: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry"""
        self.setup_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_driver() 