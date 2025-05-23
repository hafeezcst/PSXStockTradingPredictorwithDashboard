import os
import re
import time
import json
import logging
import requests
import pandas as pd
import argparse
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime, date
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, MetaData, Table, inspect, text
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(project_root))

from src.data_processing.config import PSXConfig
from src.data_processing.database import DatabaseManager
from src.data_processing.selenium_manager import SeleniumManager
from src.data_processing.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=PSXConfig.LOG_FORMAT,
    handlers=[
        logging.FileHandler(PSXConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)

class PSXAnnouncementScraper:
    """Class for scraping PSX announcements"""
    
    def __init__(self):
        """Initialize scraper with required components"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': PSXConfig.SELENIUM_OPTIONS['user_agent']
        })
        self.db_manager = DatabaseManager()
        self.selenium_manager = SeleniumManager()
        self.data_processor = DataProcessor()
        self.cache_file = PSXConfig.CACHE_DIR / "psx_announcements_cache.json"
        self.last_scrape_time = None
    
    def scrape_announcements(self, max_retries: int = 3, scrape_company_pages: bool = True, 
                           test_companies: List[str] = None, force_fresh: bool = False) -> List[Dict]:
        """
        Scrape announcements from PSX website
        
        Args:
            max_retries: Maximum number of retries
            scrape_company_pages: Whether to scrape company-specific pages
            test_companies: List of specific companies to scrape
            force_fresh: Whether to force a fresh scrape
            
        Returns:
            List[Dict]: List of announcements
        """
        announcements = []
        
        # Try to load from cache first
        if not force_fresh:
            cached_announcements = self._load_from_cache()
            if cached_announcements:
                announcements = cached_announcements
                logging.info(f"Loaded {len(announcements)} announcements from cache")
                return announcements
        
        # Scrape general announcements
        general_announcements = self._scrape_main_announcements_page(max_retries)
        announcements.extend(general_announcements)
        logging.info(f"Scraped {len(general_announcements)} general announcements")
        
        if scrape_company_pages:
            # Get companies to scrape
            companies_to_scrape = test_companies if test_companies else list(self.data_processor.company_data.keys())
            logging.info(f"Scraping company-specific pages for {len(companies_to_scrape)} companies")
            
            # Scrape company-specific announcements
            company_announcements = []
            for symbol in companies_to_scrape:
                try:
                    if symbol in self.data_processor.company_data:
                        logging.info(f"Scraping company page for {symbol}")
                        symbol_announcements = self._scrape_company_specific_announcements(symbol)
                        company_announcements.extend(symbol_announcements)
                        logging.info(f"Scraped {len(symbol_announcements)} announcements for {symbol}")
                    else:
                        logging.warning(f"Symbol {symbol} not found in company data, skipping")
                except Exception as e:
                    logging.error(f"Error scraping company page for {symbol}: {e}")
            
            announcements.extend(company_announcements)
            logging.info(f"Scraped {len(company_announcements)} company-specific announcements")
        
        # Save to cache
        self._save_to_cache(announcements)
        
        # Filter and save announcements
        kmi100_announcements = self.data_processor.filter_kmi100_announcements(announcements)
        
        # Save to files
        if announcements:
            self.data_processor.save_to_csv(announcements, prefix="all_")
        if kmi100_announcements:
            self.data_processor.save_to_csv(kmi100_announcements, prefix="kmi100_")
        
        # Save to Excel
        excel_file = self.data_processor.save_to_excel(announcements, kmi100_announcements)
        logging.info(f"Announcements Excel file created at: {excel_file}")
        
        # Save to database
        self._save_to_db(announcements)
        
        return announcements
    
    def _scrape_main_announcements_page(self, max_retries: int = 3) -> List[Dict]:
        """Scrape announcements from the main PSX announcements page"""
        announcements = []
        
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Starting main announcements page scraping (attempt {attempt})")
                
                with self.selenium_manager as selenium:
                    if selenium.get_page_source(PSXConfig.BASE_URL):
                        soup = BeautifulSoup(selenium.driver.page_source, 'html.parser')
                        announcements = self._process_html_tables(soup)
                        
                        if announcements:
                            logging.info(f"Successfully scraped {len(announcements)} announcements from main page")
                            break
                
            except Exception as e:
                logging.error(f"Error scraping main announcements (attempt {attempt}): {e}")
                if attempt < max_retries:
                    logging.info(f"Retrying in 2 seconds...")
                    time.sleep(2)
        
        return announcements
    
    def _scrape_company_specific_announcements(self, symbol: str) -> List[Dict]:
        """Scrape announcements from company-specific page"""
        announcements = []
        company_name = self.data_processor.company_data.get(symbol, f"{symbol} Limited")
        
        try:
            company_url = f"{PSXConfig.COMPANY_URL}/{symbol}"
            logging.info(f"Scraping company-specific announcements for {symbol} from {company_url}")
            
            with self.selenium_manager as selenium:
                if selenium.get_page_source(company_url):
                    if selenium.wait_for_element(By.ID, 'announcementsTab'):
                        soup = BeautifulSoup(selenium.driver.page_source, 'html.parser')
                        announcements = self._process_company_announcements(soup, symbol, company_name)
                        
            logging.info(f"Extracted {len(announcements)} company-specific announcements for {symbol}")
            
        except Exception as e:
            logging.error(f"Error scraping company-specific announcements for {symbol}: {e}")
        
        return announcements
    
    def _process_html_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Process HTML tables to extract announcements"""
        announcements = []
        
        for table in soup.find_all('table', class_='table'):
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                try:
                    cols = row.find_all('td')
                    if len(cols) < 3:
                        continue
                    
                    # Extract data
                    date_str = cols[0].text.strip()
                    time_str = cols[1].text.strip() if len(cols) > 1 else ""
                    title = cols[2].text.strip() if len(cols) > 2 else ""
                    
                    # Extract symbol and company name
                    symbol = ""
                    company_name = ""
                    
                    # Look for symbol in title
                    symbol_match = re.search(r'\b([A-Z]{3,5})\b', title)
                    if symbol_match and symbol_match.group(1) in self.data_processor.company_data:
                        symbol = symbol_match.group(1)
                        company_name = self.data_processor.company_data[symbol]
                    
                    if not symbol:
                        symbol = "UNKNOWN"
                        company_name = "UNKNOWN"
                    
                    # Parse date
                    date_formatted = self.data_processor.parse_date(date_str)
                    
                    # Create announcement ID
                    announcement_id = self.data_processor.create_announcement_id(
                        date_formatted, title, symbol
                    )
                    
                    # Create announcement
                    announcement = {
                        'ID': announcement_id,
                        'Symbol': symbol,
                        'Company': company_name,
                        'Date': date_formatted,
                        'Time': time_str,
                        'Subject': title,
                        'URL': self._extract_url(cols),
                        'Status': 'NEW',
                        'Category': 'General'
                    }
                    
                    announcements.append(announcement)
                    
                except Exception as e:
                    logging.error(f"Error processing row: {e}")
                    continue
        
        return announcements
    
    def _process_company_announcements(self, soup: BeautifulSoup, symbol: str, company_name: str) -> List[Dict]:
        """Process company-specific announcements"""
        announcements = []
        
        announcements_tab = soup.find('div', id='announcementsTab')
        if not announcements_tab:
            return announcements
        
        for tab in announcements_tab.find_all('div', class_='tabs__panel'):
            tab_name = tab.get('data-name', 'Unknown')
            table = tab.find('table', class_='tbl')
            
            if not table:
                continue
            
            rows = table.find_all('tr')
            if len(rows) <= 1:
                continue
            
            # Get header row
            headers = [th.text.strip().upper() for th in rows[0].find_all('th')]
            
            # Determine column indices
            date_idx = next((i for i, h in enumerate(headers) if 'DATE' in h), 0)
            title_idx = next((i for i, h in enumerate(headers) if 'TITLE' in h), 1)
            document_idx = next((i for i, h in enumerate(headers) if 'DOCUMENT' in h), 2)
            
            # Process data rows
            for row in rows[1:]:
                try:
                    cols = row.find_all('td')
                    if len(cols) <= title_idx:
                        continue
                    
                    # Extract data
                    date_str = cols[date_idx].text.strip()
                    time_str = cols[1].text.strip() if len(cols) > 1 else ""
                    title = cols[title_idx].text.strip()
                    
                    # Parse date
                    date_formatted = self.data_processor.parse_date(date_str)
                    
                    # Create announcement ID
                    announcement_id = self.data_processor.create_announcement_id(
                        date_formatted, title, symbol, tab_name
                    )
                    
                    # Create announcement
                    announcement = {
                        'ID': announcement_id,
                        'Symbol': symbol,
                        'Company': company_name,
                        'Date': date_formatted,
                        'Time': time_str,
                        'Subject': f"{tab_name} - {title}",
                        'URL': self._extract_url(cols),
                        'Status': 'Active',
                        'Category': tab_name
                    }
                    
                    announcements.append(announcement)
                    
                except Exception as e:
                    logging.error(f"Error processing company announcement row: {e}")
                    continue
        
        return announcements
    
    def _extract_url(self, cols: List[Any]) -> str:
        """Extract URL from table columns"""
        url = ''
        
        # Search all columns for links
        for col in cols:
            links = col.find_all('a')
            for link in links:
                href = link.get('href', '')
                if not href.startswith('javascript:'):
                    if '/download/document/' in href and href.endswith('.pdf'):
                        url = href
                        break
                    elif '/download/attachment/' in href and href.endswith('.pdf'):
                        url = href
                        break
        
        # Add domain if needed
        if url and not url.startswith('http'):
            url = f"https://dps.psx.com.pk{url}"
        
        return url
    
    def _save_to_cache(self, data: List[Dict]) -> None:
        """Save announcements to cache file"""
        try:
            # Ensure dates are converted to strings
            serializable_data = []
            for item in data:
                item_copy = item.copy()
                if 'Date' in item_copy and not isinstance(item_copy['Date'], str):
                    if isinstance(item_copy['Date'], datetime):
                        item_copy['Date'] = item_copy['Date'].strftime("%Y-%m-%d")
                serializable_data.append(item_copy)
            
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logging.info(f"Saved {len(data)} announcements to cache")
        except Exception as e:
            logging.error(f"Error saving to cache: {e}")
    
    def _load_from_cache(self) -> List[Dict]:
        """Load announcements from cache file"""
        if not self.cache_file.exists():
            return []
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            logging.info(f"Loaded {len(data)} announcements from cache")
            return data
        except Exception as e:
            logging.error(f"Error loading from cache: {e}")
            return []
    
    def _save_to_db(self, announcements: List[Dict]) -> None:
        """Save announcements to database"""
        try:
            # Group announcements by symbol
            symbol_groups = {}
            for announcement in announcements:
                symbol = announcement.get('Symbol', 'UNKNOWN')
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(announcement)
            
            # Save each symbol's announcements
            for symbol, symbol_announcements in symbol_groups.items():
                if not symbol_announcements:
                    continue
                
                table_name = f"PSX_{symbol}_announcements"
                
                # Check if table needs to be recreated
                columns = self.db_manager.get_table_schema(table_name)
                if columns and ('ID' not in columns or 'Category' not in columns):
                    self.db_manager.recreate_table(table_name)
                
                # Save announcements
                self.db_manager.save_announcements(symbol_announcements, table_name)
            
            logging.info(f"Saved announcements for {len(symbol_groups)} symbols to database")
            
        except Exception as e:
            logging.error(f"Error saving to database: {e}")
    
    def __del__(self):
        """Clean up resources"""
        self.selenium_manager.close_driver()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='PSX Announcement Scraper')
    parser.add_argument('--fresh', action='store_true', help='Force a fresh scrape (bypass cache)')
    parser.add_argument('--company', type=str, help='Scrape a specific company page by symbol')
    parser.add_argument('--skip-company-pages', action='store_true', help='Skip scraping company-specific pages')
    parser.add_argument('--test-companies', type=str, help='Comma-separated list of symbols to test')
    args = parser.parse_args()
    
    # Setup directories
    PSXConfig.setup_directories()
    
    logging.info("Starting PSX Announcement scraper")
    try:
        scraper = PSXAnnouncementScraper()
        
        # If a specific company is requested, only scrape that company
        if args.company:
            company_announcements = scraper._scrape_company_specific_announcements(args.company.upper())
            logging.info(f"Finished scraping specific company {args.company.upper()}")
            sys.exit(0)
        
        # Parse test companies if provided
        test_companies = None
        if args.test_companies:
            test_companies = [symbol.strip().upper() for symbol in args.test_companies.split(',')]
            logging.info(f"Using test companies: {', '.join(test_companies)}")
        
        # Scrape announcements
        announcements = scraper.scrape_announcements(
            scrape_company_pages=not args.skip_company_pages,
            test_companies=test_companies,
            force_fresh=args.fresh
        )
        
        logging.info(f"Successfully scraped {len(announcements)} announcements")
        
    except Exception as e:
        logging.error(f"Error during scraping: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Finished PSX Announcement scraping")