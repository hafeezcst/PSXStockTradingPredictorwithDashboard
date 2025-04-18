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
from config.paths import (
    DATA_LOGS_DIR,
    PSX_SYM_PATH,
    SYMBOLS_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/psx_announcements.log'),
        logging.StreamHandler()
    ]
)

class DatabaseManager:
    def __init__(self, db_path=PSX_SYM_PATH, pool_config=None):
        """Initialize database manager with connection pooling"""
        if pool_config is None:
            pool_config = {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600
            }

        # Primary database with connection pooling
        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(
            f'sqlite:///{self.db_path}',
            pool_size=pool_config['pool_size'],
            max_overflow=pool_config['max_overflow'],
            pool_timeout=pool_config['pool_timeout'],
            pool_recycle=pool_config['pool_recycle']
        )
        
    def save_to_db(self, data: pd.DataFrame, table_name: str):
        """Save DataFrame to database"""
        if not data.empty:
            data.to_sql(table_name, self.engine, if_exists='append', index=True)
        else:
            logging.info(f"No data to save for table {table_name}")

    def verify_data(self, symbol: str, start_date: date, end_date: date) -> bool:
        """Verify data exists in database for given symbol and date range"""
        table_name = f'PSX_{symbol}_announcements'
        inspector = inspect(self.engine)
        if table_name in inspector.get_table_names():
            try:
                with self.engine.connect().execution_options(statement_timeout=60) as conn:
                    query = text(f"SELECT COUNT(*) FROM {table_name} WHERE Date BETWEEN :start_date AND :end_date")
                    result = conn.execute(query, {'start_date': start_date, 'end_date': end_date}).scalar()
                    return result > 0
            except Exception as e:
                logging.error(f"Error verifying data for {symbol}: {e}")
                return False
        return False

    def check_database_integrity(self):
        """Performs basic integrity checks on the primary database."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA integrity_check;"))
            logging.info("Database passed the integrity check.")
        except Exception as e:
            logging.error(f"Database integrity check failed: {e}")

class PSXAnnouncementScraper:
    """Class for scraping PSX announcements"""
    
    # Constants
    BASE_URL = "https://dps.psx.com.pk/announcements/companies"
    COMPANY_URL = "https://dps.psx.com.pk/company"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.cache_file = "data/processed/psx_announcements_cache.json"
        self.last_scrape_time = None
        self.db_manager = DatabaseManager()
        self.company_data = None
        self.company_data_file = SYMBOLS_FILE
        self.driver = None
        
    def _setup_selenium(self):
        """Initialize Selenium WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in headless mode
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # Initialize the Chrome driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logging.info("Selenium WebDriver initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Error initializing Selenium WebDriver: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
            
    def _close_selenium(self):
        """Close Selenium WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("Selenium WebDriver closed")
            except Exception as e:
                logging.error(f"Error closing Selenium WebDriver: {e}")
            finally:
                self.driver = None
                
    def scrape_announcements(self, max_retries: int = 3, scrape_company_pages: bool = True, test_companies: List[str] = None, force_fresh: bool = False) -> List[Dict]:
        """
        Scrape announcements from PSX website
        
        Args:
            max_retries (int, optional): Maximum number of retries. Defaults to 3.
            scrape_company_pages (bool, optional): Whether to scrape company-specific pages. Defaults to True.
            test_companies (List[str], optional): List of specific companies to scrape. If None, scrape all companies.
            force_fresh (bool, optional): Whether to force a fresh scrape, bypassing cache. Defaults to False.
            
        Returns:
            List[Dict]: List of announcements
        """
        announcements = []
        # Try to load from cache first if not forcing fresh scrape
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
            # Load company data
            company_data = self._load_company_data()
            symbol_set = set(company_data.keys())
            
            # Get companies to scrape
            all_symbols = list(company_data.keys())
            
            # Use test_companies if provided, otherwise use all symbols
            companies_to_scrape = test_companies if test_companies else all_symbols
            logging.info(f"Scraping company-specific pages for {len(companies_to_scrape)} companies")
            
            # Scrape company-specific announcements for each company
            company_announcements = []
            for symbol in companies_to_scrape:
                try:
                    if symbol in symbol_set:
                        logging.info(f"Scraping company page for {symbol}")
                        symbol_announcements = self._scrape_company_specific_announcements(symbol, company_data, symbol_set)
                        company_announcements.extend(symbol_announcements)
                        logging.info(f"Scraped {len(symbol_announcements)} announcements for {symbol}")
                    else:
                        logging.warning(f"Symbol {symbol} not found in company data, skipping")
                except Exception as e:
                    logging.error(f"Error scraping company page for {symbol}: {e}")
            
            # Add company-specific announcements to all announcements
            announcements.extend(company_announcements)
            logging.info(f"Scraped {len(company_announcements)} company-specific announcements")
            
        # Save all to cache
        self._save_to_cache(announcements)
        
        # Filter for KMI100 companies and test subset
        kmi100_announcements = self._filter_kmi100_announcements(announcements)
        test_subset = self._filter_test_subset_announcements(announcements)
        
        # Save to CSV
        if announcements:
            self._save_to_csv(announcements, prefix="all_")
        if kmi100_announcements:
            self._save_to_csv(kmi100_announcements, prefix="kmi100_")
        if test_subset:
            self._save_to_csv(test_subset, prefix="test_subset")
        
        # Save to Excel with two sheets
        excel_file = self._save_to_excel(announcements, kmi100_announcements)
        logging.info(f"Announcements Excel file created at: {excel_file}")
        
        # Save to database
        self._save_to_db(announcements)
        
        return announcements
    
    def scrape_specific_company(self, symbol: str) -> List[Dict]:
        """
        Scrape announcements for a specific company symbol
        
        Args:
            symbol (str): Company symbol to scrape
            
        Returns:
            List[Dict]: List of announcements for the specified company
        """
        logging.info(f"Directly scraping company page for {symbol}")
        company_data = self._load_company_data()
        symbol_set = set(company_data.keys())
        
        if symbol not in company_data:
            logging.warning(f"Symbol {symbol} not found in company data")
            return []
            
        # Scrape company-specific announcements
        announcements = self._scrape_company_specific_announcements(symbol, company_data, symbol_set)
        
        logging.info(f"Found {len(announcements)} announcements for {symbol}")
        
        # Save to CSV for inspection
        if announcements:
            csv_file = self._save_to_csv(announcements, prefix=f"{symbol}_")
            logging.info(f"Saved {symbol} announcements to {csv_file}")
            
        return announcements
    
    def _scrape_company_specific_announcements(self, symbol: str, company_data: Dict[str, str], symbol_set: Set[str]) -> List[Dict]:
        """
        Scrape announcements from company-specific page
        
        Args:
            symbol (str): Company symbol to scrape
            company_data (Dict): Dictionary of company symbols to names
            symbol_set (Set): Set of symbols for fast lookup
            
        Returns:
            List of announcement dictionaries
        """
        announcements = []
        company_name = company_data.get(symbol, f"{symbol} Limited")
        
        try:
            company_url = f"{self.COMPANY_URL}/{symbol}"
            logging.info(f"Scraping company-specific announcements for {symbol} from {company_url}")
            
            if not self.driver:
                self._setup_selenium()
                
            if not self.driver:
                logging.error("Selenium not available for company-specific scraping")
                return []
                
            # Load the company page
            self.driver.get(company_url)
            
            # Wait for the announcements tab content to load
            try:
                wait = WebDriverWait(self.driver, 25)
                wait.until(EC.presence_of_element_located((By.ID, 'announcementsTab')))
                logging.info(f"Successfully loaded company page for {symbol}")
            except TimeoutException:
                logging.warning(f"Timeout waiting for company page for {symbol}")
                return []
                
            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find the announcements tab
            announcements_tab = soup.find('div', id='announcementsTab')
            if not announcements_tab:
                logging.warning(f"No announcements tab found for {symbol}")
                return []
                
            # Find all tabs (Financial Results, Board Meetings, Others)
            tabs = announcements_tab.find_all('div', class_='tabs__panel')
            logging.info(f"Found {len(tabs)} tabs in company page for {symbol}")
            
            for tab in tabs:
                # Get the tab name
                tab_name = tab.get('data-name', 'Unknown')
                
                # Find the table in this tab
                table = tab.find('table', class_='tbl')
                if not table:
                    continue
                    
                # Get rows from the table
                rows = table.find_all('tr')
                if len(rows) <= 1:  # Skip if only header row
                    continue
                    
                # Get header row to determine column indices
                header_row = rows[0]
                headers = [th.text.strip().upper() for th in header_row.find_all('th')]
                
                # Determine column indices
                date_idx = next((i for i, h in enumerate(headers) if 'DATE' in h), 0)
                title_idx = next((i for i, h in enumerate(headers) if 'TITLE' in h), 1)
                document_idx = next((i for i, h in enumerate(headers) if 'DOCUMENT' in h), 2)
                
                # Process data rows
                for row in rows[1:]:  # Skip header row
                    cols = row.find_all('td')
                    
                    # Check if we have enough columns
                    if len(cols) <= title_idx:
                        continue
                        
                    # Extract data from columns
                    date_str = cols[date_idx].text.strip() if cols[date_idx].text else ""
                    time_str = cols[1].text.strip() if cols[1].text else ""
                    title = cols[title_idx].text.strip() if 0 <= title_idx < len(cols) and cols[title_idx].text else ""
                    
                    # Extract link - search all columns for links if needed
                    link_tag = None
                    pdf_links = []
                    
                    # First try the document column to find PDF links
                    if 0 <= document_idx < len(cols):
                        links = cols[document_idx].find_all('a')
                        for link in links:
                            href = link.get('href', '')
                            # Collect all PDF links (both document and attachment format)
                            if not href.startswith('javascript:'):
                                if '/download/document/' in href and href.endswith('.pdf'):
                                    pdf_links.append((link, 'document'))
                                elif '/download/attachment/' in href and href.endswith('.pdf'):
                                    pdf_links.append((link, 'attachment'))
                    
                    # If no PDF links found in document column, search all columns
                    if not pdf_links:
                        for col in cols:
                            links = col.find_all('a')
                            for link in links:
                                href = link.get('href', '')
                                if not href.startswith('javascript:'):
                                    if '/download/document/' in href and href.endswith('.pdf'):
                                        pdf_links.append((link, 'document'))
                                    elif '/download/attachment/' in href and href.endswith('.pdf'):
                                        pdf_links.append((link, 'attachment'))
                    
                    # Choose the best PDF link - prioritize attachment links over document links if both exist
                    url = ''
                    if pdf_links:
                        # First try to find an attachment link
                        attachment_links = [link for link, type_name in pdf_links if type_name == 'attachment']
                        if attachment_links:
                            link_tag = attachment_links[0]
                        else:
                            # Fall back to document link
                            document_links = [link for link, type_name in pdf_links if type_name == 'document']
                            if document_links:
                                link_tag = document_links[0]
                        
                        if link_tag:
                            url = link_tag.get('href', '')
                            # Add domain if needed
                            if url and not url.startswith('http'):
                                url = f"https://dps.psx.com.pk{url}"
                    
                    # Parse date
                    try:
                        date_obj = None
                        # Try different formats
                        date_formats = ['%b %d, %Y', '%B %d, %Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
                        
                        for fmt in date_formats:
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if not date_obj:
                            # Parse "Mar 3, 2025" format by extracting components
                            parts = date_str.replace(',', '').split()
                            if len(parts) == 3:
                                month_dict = {
                                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                }
                                month = month_dict.get(parts[0][:3], 1)
                                day = int(parts[1])
                                year = int(parts[2])
                                date_obj = datetime(year, month, day)
                        
                        date_formatted = date_obj.strftime('%Y-%m-%d')
                    except Exception as e:
                        logging.warning(f"Could not parse date '{date_str}': {e}")
                        date_formatted = date_str
                    
                    # Create a unique announcement ID
                    announcement_id = hashlib.md5(f"{date_formatted}_{title}_{symbol}_{tab_name}".encode()).hexdigest()
                    
                    # Create announcement data
                    announcement = {
                        'ID': announcement_id,
                        'Symbol': symbol,
                        'Company': company_name,
                        'Date': date_formatted,
                        'Time': time_str,
                        'Subject': f"{tab_name} - {title}",
                        'URL': url,
                        'Status': 'Active',
                        'Category': tab_name  # Add category based on tab name
                    }
                    
                    announcements.append(announcement)
            
            logging.info(f"Extracted {len(announcements)} company-specific announcements for {symbol}")
            
        except Exception as e:
            logging.error(f"Error scraping company-specific announcements for {symbol}: {e}")
            
        return announcements
    
    def _scrape_main_announcements_page(self, max_retries: int = 3) -> List[Dict]:
        """
        Scrape announcements from the main PSX announcements page
        
        Args:
            max_retries (int, optional): Maximum number of retries. Defaults to 3.
            
        Returns:
            List[Dict]: List of announcements
        """
        # Load company data for symbol matching
        company_data = self._load_company_data()
        symbol_set = set(company_data.keys())
        
        announcements = []
        
        # Try Selenium first, then fall back to requests
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Starting main announcements page scraping (attempt {attempt})")
                
                if self.driver is None:
                    self._setup_selenium()
                    
                if self.driver:
                    # Use Selenium for dynamic content
                    url = self.BASE_URL
                    logging.info(f"Attempting to load {url} with Selenium")
                    self.driver.get(url)
                    
                    # Wait for the page to load (table to appear)
                    try:
                        wait = WebDriverWait(self.driver, 25)
                        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'table')))
                        logging.info(f"Successfully loaded page with Selenium")
                    except TimeoutException:
                        logging.warning(f"Timeout waiting for announcements table")
                        
                    # Parse the HTML content with BeautifulSoup
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    
                    # Process HTML to extract announcements
                    announcements = self._process_html_tables(soup, company_data, symbol_set)
                    
                    if announcements:
                        logging.info(f"Successfully scraped {len(announcements)} announcements from main page")
                        break
                else:
                    # Fallback to requests (less reliable for dynamic content)
                    logging.warning(f"Selenium not available, falling back to requests")
                    response = requests.get(self.BASE_URL)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        announcements = self._process_html_tables(soup, company_data, symbol_set)
                        
                        if announcements:
                            logging.info(f"Successfully scraped {len(announcements)} announcements with requests")
                            break
                    else:
                        logging.error(f"Failed to fetch announcements: {response.status_code}")
                
            except Exception as e:
                logging.error(f"Error scraping main announcements (attempt {attempt}): {e}")
                if attempt < max_retries:
                    logging.info(f"Retrying in 2 seconds...")
                    time.sleep(2)
        
        return announcements
    
    def _process_html_tables(self, soup: BeautifulSoup, company_data: Dict[str, str], symbol_set: Set[str]) -> List[Dict]:
        """
        Process HTML tables to extract announcements
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            company_data (Dict): Dictionary of company symbols to names
            symbol_set (Set): Set of symbols for fast lookup
            
        Returns:
            List of announcement dictionaries
        """
        announcements = []
        
        logging.info("Processing announcements table")
        # Find the main announcements table
        tables = soup.find_all('table', class_='table')
        
        for table in tables:
            logging.info(f"Found announcements table with {len(table.find_all('tr'))} rows")
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cols = row.find_all('td')
                
                # Check if we have enough columns
                if len(cols) < 3:
                    continue
                
                try:
                    # Extract data from columns
                    date_str = cols[0].text.strip() if cols[0].text else ""
                    time_str = cols[1].text.strip() if len(cols) > 1 and cols[1].text else ""
                    title = cols[2].text.strip() if len(cols) > 2 and cols[2].text else ""
                    
                    # Get symbol and company name - this format is used in company-specific announcements
                    symbol = ""
                    company_name = ""
                    
                    # Look for dedicated Symbol and Name columns
                    symbol_idx = -1
                    name_idx = -1
                    title_idx = -1
                    document_idx = -1
                    
                    # Identify column indices based on header or position
                    headers = table.find_all('th')
                    for i, header in enumerate(headers):
                        header_text = header.text.strip().upper()
                        if 'SYMBOL' in header_text:
                            symbol_idx = i
                        elif 'NAME' in header_text or 'COMPANY' in header_text:
                            name_idx = i
                        elif 'SUBJECT' in header_text or 'TITLE' in header_text or 'ANNOUNCEMENT' in header_text:
                            title_idx = i
                        elif 'DOCUMENT' in header_text or 'PDF' in header_text or 'VIEW' in header_text:
                            document_idx = i
                    
                    # Fallback to positions if headers not found
                    if symbol_idx == -1 and len(cols) >= 5:
                        symbol_idx = 2
                    if name_idx == -1 and len(cols) >= 5:
                        name_idx = 3
                    if title_idx == -1:
                        title_idx = 2 if len(cols) < 5 else 4
                    if document_idx == -1:
                        document_idx = len(cols) - 1
                        
                    # Extract data using identified indices
                    if 0 <= symbol_idx < len(cols):
                        symbol = cols[symbol_idx].text.strip()
                    if 0 <= name_idx < len(cols):
                        company_name = cols[name_idx].text.strip()
                    
                    # Extract title (subject)
                    title = cols[title_idx].text.strip() if 0 <= title_idx < len(cols) and cols[title_idx].text else ""
                    
                    # Extract link - search all columns for links if needed
                    link_tag = None
                    pdf_links = []
                    
                    # First try the document column to find PDF links
                    if 0 <= document_idx < len(cols):
                        links = cols[document_idx].find_all('a')
                        for link in links:
                            href = link.get('href', '')
                            # Collect all PDF links (both document and attachment format)
                            if not href.startswith('javascript:'):
                                if '/download/document/' in href and href.endswith('.pdf'):
                                    pdf_links.append((link, 'document'))
                                elif '/download/attachment/' in href and href.endswith('.pdf'):
                                    pdf_links.append((link, 'attachment'))
                    
                    # If no PDF links found in document column, search all columns
                    if not pdf_links:
                        for col in cols:
                            links = col.find_all('a')
                            for link in links:
                                href = link.get('href', '')
                                if not href.startswith('javascript:'):
                                    if '/download/document/' in href and href.endswith('.pdf'):
                                        pdf_links.append((link, 'document'))
                                    elif '/download/attachment/' in href and href.endswith('.pdf'):
                                        pdf_links.append((link, 'attachment'))
                    
                    # Choose the best PDF link - prioritize attachment links over document links if both exist
                    url = ''
                    if pdf_links:
                        # First try to find an attachment link
                        attachment_links = [link for link, type_name in pdf_links if type_name == 'attachment']
                        if attachment_links:
                            link_tag = attachment_links[0]
                        else:
                            # Fall back to document link
                            document_links = [link for link, type_name in pdf_links if type_name == 'document']
                            if document_links:
                                link_tag = document_links[0]
                        
                        if link_tag:
                            url = link_tag.get('href', '')
                            # Add domain if needed
                            if url and not url.startswith('http'):
                                url = f"https://dps.psx.com.pk{url}"
                    
                    # Handle case where symbol is in title
                    if not symbol:
                        # Try to extract symbol from title using regex
                        symbol_match = re.search(r'\b([A-Z]{3,5})\b', title)
                        if symbol_match and symbol_match.group(1) in symbol_set:
                            symbol = symbol_match.group(1)
                    
                    # If we have a symbol but no company name, get it from company_data
                    if symbol and not company_name and symbol in company_data:
                        company_name = company_data.get(symbol, "")
                    
                    # If we still don't have a symbol, mark as unknown
                    if not symbol:
                        symbol = "UNKNOWN"
                        
                    if not company_name:
                        company_name = "UNKNOWN"
                    
                    # Parse date
                    try:
                        date_obj = None
                        # Try different formats
                        date_formats = ['%b %d, %Y', '%B %d, %Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
                        
                        for fmt in date_formats:
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if not date_obj:
                            # Parse "Mar 3, 2025" format by extracting components
                            parts = date_str.replace(',', '').split()
                            if len(parts) == 3:
                                month_dict = {
                                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                }
                                month = month_dict.get(parts[0][:3], 1)
                                day = int(parts[1])
                                year = int(parts[2])
                                date_obj = datetime(year, month, day)
                        
                        date_formatted = date_obj.strftime('%Y-%m-%d')
                    except Exception as e:
                        logging.warning(f"Could not parse date '{date_str}': {e}")
                        date_formatted = date_str
                    
                    # Create a unique announcement ID
                    announcement_id = hashlib.md5(f"{date_formatted}_{title}_{symbol}".encode()).hexdigest()
                    
                    # Create announcement dictionary
                    announcement = {
                        'ID': announcement_id,
                        'Symbol': symbol,
                        'Company': company_name,
                        'Date': date_formatted,
                        'Time': time_str,
                        'Subject': title,
                        'URL': url,
                        'Status': 'NEW',
                        'Category': 'General'
                    }
                    
                    # Check if this is a duplicate
                    is_duplicate = False
                    for existing in announcements:
                        if (existing['Symbol'] == symbol and 
                            existing['Date'] == date_formatted and 
                            existing['Subject'] == title):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        announcements.append(announcement)
                        
                except Exception as e:
                    logging.error(f"Error processing row: {e}")
                    continue
        
        logging.info(f"Extracted {len(announcements)} announcements from tables")
        return announcements
    
    def _load_company_data(self) -> Dict:
        """Load company symbols and names from Excel file, focusing on KMI100 companies"""
        company_data = {}
        try:
            # Get path to symbols file from config
            symbols_file = Path(SYMBOLS_FILE)
            
            if not symbols_file.exists():
                logging.warning(f"Symbols file not found at {symbols_file}")
                return company_data
                
            # Load Excel file with KMI100 sheet
            try:
                df = pd.read_excel(symbols_file, sheet_name='KMI100')
                logging.info(f"Loaded data from KMI100 sheet in {symbols_file}")
            except Exception as e:
                # Fallback to other sheet names if KMI100 doesn't exist
                try:
                    df = pd.read_excel(symbols_file, sheet_name='KMIALL')
                    logging.info(f"Loaded data from KMIALL sheet in {symbols_file}")
                except Exception as e2:
                    try:
                        df = pd.read_excel(symbols_file, sheet_name='KMI30')
                        logging.info(f"Loaded data from KMI30 sheet in {symbols_file}")
                    except Exception as e3:
                        # Last resort - try the first sheet
                        df = pd.read_excel(symbols_file)
                        logging.info(f"Loaded data from default sheet in {symbols_file}")
            
            # Get column names (case-insensitive)
            columns = [col.upper() for col in df.columns]
            
            # Find symbol column - usually first column
            symbol_col_idx = 0
            symbol_col = df.columns[symbol_col_idx]
            
            # Check if second column might be company name
            name_col = None
            if len(df.columns) > 1:
                name_col_candidates = [col for col in df.columns if any(x in col.upper() for x in ['NAME', 'COMPANY', 'DESC', 'TITLE'])]
                if name_col_candidates:
                    name_col = name_col_candidates[0]
                else:
                    # Just use the second column
                    name_col = df.columns[1]
            
            # Create a dictionary of symbol -> company name
            for _, row in df.iterrows():
                symbol = str(row[symbol_col]).strip().upper()
                
                # Skip empty symbols
                if not symbol or pd.isna(symbol):
                    continue
                
                # Get company name if available
                if name_col and pd.notna(row[name_col]):
                    name = str(row[name_col]).strip()
                    # Make sure name is not just a number
                    if name.replace('.', '').isdigit():
                        name = f"{symbol} Limited"
                else:
                    name = f"{symbol} Limited"
                
                company_data[symbol] = name
                
            logging.info(f"Loaded {len(company_data)} companies from Excel file")
            
        except Exception as e:
            logging.error(f"Error loading company data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
        return company_data
    
    def _save_to_cache(self, data: List[Dict]) -> None:
        """Save announcements to a local cache file"""
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / "psx_announcements_cache.json"
        
        try:
            # Ensure dates are converted to strings for JSON serialization
            serializable_data = []
            for item in data:
                item_copy = item.copy()
                if 'Date' in item_copy and not isinstance(item_copy['Date'], str):
                    if isinstance(item_copy['Date'], datetime):
                        item_copy['Date'] = item_copy['Date'].strftime("%Y-%m-%d")
                serializable_data.append(item_copy)
                
            with open(cache_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            logging.info(f"Saved {len(data)} announcements to cache")
        except Exception as e:
            logging.error(f"Error saving to cache: {e}")
    
    def _load_from_cache(self) -> List[Dict]:
        """Load announcements from local cache file"""
        cache_file = Path("data/cache/psx_announcements_cache.json")
        
        if not cache_file.exists():
            logging.warning("Cache file not found")
            return []
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            logging.info(f"Loaded {len(data)} announcements from cache")
            return data
        except Exception as e:
            logging.error(f"Error loading from cache: {e}")
            return []
    
    def _save_to_db(self, announcements: List[Dict]) -> None:
        """Save announcements to the database"""
        try:
            # Create SQLite engine
            engine = create_engine(f'sqlite:///{PSX_SYM_PATH}')
            
            # Get metadata
            metadata = MetaData()
            metadata.reflect(bind=engine)
            
            # Group announcements by symbol
            symbol_groups = {}
            for announcement in announcements:
                symbol = announcement.get('Symbol', 'UNKNOWN')
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(announcement)
            
            # Save each symbol's announcements to its own table
            for symbol, symbol_announcements in symbol_groups.items():
                if not symbol_announcements:
                    continue
                    
                table_name = f"PSX_{symbol}_announcements"
                
                # Check if table exists
                if table_name in metadata.tables:
                    table = metadata.tables[table_name]
                    
                    # Check if 'ID' column exists, if not, recreate the table
                    # Also check for 'Category' column for company-specific announcements
                    inspector = inspect(engine)
                    columns = [col['name'] for col in inspector.get_columns(table_name)]
                    
                    # If ID column or Category column doesn't exist, we need to recreate the table
                    if 'ID' not in columns or 'Category' not in columns:
                        # Create a backup of the existing table first
                        backup_table_name = f"{table_name}_backup"
                        with engine.connect() as conn:
                            # Create backup table
                            conn.execute(text(f"CREATE TABLE IF NOT EXISTS {backup_table_name} AS SELECT * FROM {table_name}"))
                            # Drop the original table
                            conn.execute(text(f"DROP TABLE {table_name}"))
                            
                        # Let metadata know the table was dropped
                        metadata.remove(table)
                        metadata.reflect(bind=engine, only=[backup_table_name])
                        
                        logging.info(f"Recreated table {table_name} with updated schema including ID and Category columns")
                else:
                    # Table doesn't exist yet
                    pass
                
                # Get dataframe from announcements
                df = pd.DataFrame(symbol_announcements)
                
                # Write to database - if table doesn't exist, it will be created with correct schema including ID
                df.to_sql(table_name, engine, if_exists='append', index=False)
            
            logging.info(f"Saved announcements for {len(symbol_groups)} symbols to database")
            
        except Exception as e:
            logging.error(f"Error saving to database: {e}")
    
    def _save_to_csv(self, announcements: List[Dict], prefix="") -> str:
        """Save announcements to a CSV file for easy viewing"""
        try:
            # Create data directory if it doesn't exist
            csv_dir = Path("data/csv")
            csv_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = csv_dir / f"{prefix}psx_announcements_{timestamp}.csv"
            
            # Convert to DataFrame
            df = pd.DataFrame(announcements)
            
            # Ensure columns are in desired order
            if len(df) > 0:
                column_order = ['Symbol', 'Company', 'Date', 'Time', 'Subject', 'URL', 'Status', 'Category']
                df = df[column_order]
            
            # Save to CSV
            df.to_csv(csv_file, index=False)
            logging.info(f"Saved {len(announcements)} announcements to CSV file: {csv_file}")
            return str(csv_file)
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
            return ""
    
    def _save_to_excel(self, all_announcements: List[Dict], kmi100_announcements: List[Dict]) -> str:
        """
        Save announcements to Excel file with two sheets
        
        Args:
            all_announcements (List[Dict]): List of all announcements
            kmi100_announcements (List[Dict]): List of filtered KMI100 announcements
            
        Returns:
            str: Path to saved Excel file
        """
        try:
            excel_dir = os.path.join(os.getcwd(), "data", "excel")
            os.makedirs(excel_dir, exist_ok=True)
            excel_file = os.path.join(excel_dir, "PSX_Announcements.xlsx")
            
            # Filter for all companies in Excel for KMIALLSHR sheet
            all_excel_companies_announcements = self._filter_all_companies_announcements(all_announcements)
            
            # Convert to DataFrames
            all_df = pd.DataFrame(all_excel_companies_announcements)
            kmi100_df = pd.DataFrame(kmi100_announcements)
            
            # Ensure columns are in desired order
            if len(all_df) > 0:
                column_order = ['Symbol', 'Company', 'Date', 'Time', 'Subject', 'URL', 'Status', 'Category']
                all_df = all_df[column_order]
                kmi100_df = kmi100_df[column_order]
            
            # Create an Excel writer
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                all_df.to_excel(writer, sheet_name='KMIALLSHRAnnouncements', index=False)
                kmi100_df.to_excel(writer, sheet_name='KMI100Announcements', index=False)
            
            logging.info(f"Saved {len(all_excel_companies_announcements)} KMIALLSHR announcements and {len(kmi100_announcements)} KMI100 announcements to Excel file: {excel_file}")
            return str(excel_file)
        except Exception as e:
            logging.error(f"Error saving to Excel: {e}")
            return ""
    
    def _filter_all_companies_announcements(self, announcements: List[Dict]) -> List[Dict]:
        """
        Filter announcements for ALL companies in the Excel sheet
        
        Args:
            announcements (List[Dict]): List of all announcements
            
        Returns:
            List[Dict]: Filtered announcements for all Excel sheet companies
        """
        # Load symbols from Excel file
        company_data = self._load_company_data()
        all_symbols = set(company_data.keys())
        
        # Filter announcements for all companies in Excel
        filtered_announcements = []
        for announcement in announcements:
            symbol = announcement.get('Symbol')
            if symbol and symbol in all_symbols:
                filtered_announcements.append(announcement)
            elif symbol == 'EPCL':  # Special check for EPCL
                logging.warning(f"EPCL announcement found but excluded: {announcement.get('Subject', '')} - Symbol not in company data")
                
        logging.info(f"Filtered {len(filtered_announcements)} announcements for {len(all_symbols)} companies from Excel")
        
        # Check for EPCL in filtered announcements
        epcl_filtered = [a for a in filtered_announcements if a.get('Symbol') == 'EPCL']
        if epcl_filtered:
            logging.info(f"EPCL announcements in filtered data: {len(epcl_filtered)}")
        else:
            logging.warning("No EPCL announcements found in filtered data")
            
        return filtered_announcements
        
    def _filter_kmi100_announcements(self, announcements: List[Dict]) -> List[Dict]:
        """
        Filter announcements for KMI100 companies (first 100 companies in Excel)
        
        Args:
            announcements (List[Dict]): List of all announcements
            
        Returns:
            List[Dict]: Filtered announcements for KMI100 companies
        """
        # Load symbols from Excel file
        company_data = self._load_company_data()
        all_symbols = list(company_data.keys())
        
        # Take first 100 symbols for KMI100 companies
        kmi100_symbols = set(all_symbols[:100])
        logging.info(f"Using first 100 symbols for KMI100 announcements")
        
        # Filter announcements for KMI100 companies only
        kmi100_announcements = []
        for announcement in announcements:
            symbol = announcement.get('Symbol')
            if symbol and symbol in kmi100_symbols:
                kmi100_announcements.append(announcement)
                
        logging.info(f"Filtered {len(kmi100_announcements)} announcements for {len(kmi100_symbols)} KMI100 companies")
        return kmi100_announcements
        
    def _filter_test_subset_announcements(self, announcements: List[Dict]) -> List[Dict]:
        """
        Filter announcements for a test subset of companies (first 30)
        
        Args:
            announcements (List[Dict]): List of all announcements
            
        Returns:
            List[Dict]: Filtered announcements for test subset
        """
        # Load symbols from Excel file
        company_data = self._load_company_data()
        all_symbols = list(company_data.keys())
        
        # Take first 30 symbols for the test subset
        test_symbols = set(all_symbols[:30])
        logging.info(f"Using first 30 symbols for focused testing")
        
        # Filter announcements for test subset
        test_announcements = []
        for announcement in announcements:
            symbol = announcement.get('Symbol')
            if symbol and symbol != 'UNKNOWN' and symbol in test_symbols:
                test_announcements.append(announcement)
                
        logging.info(f"Filtered {len(test_announcements)} announcements for {len(test_symbols)} companies")
        return test_announcements
    
    def filter_announcements_by_companies(self, announcements: List[Dict], symbols: List[str]) -> List[Dict]:
        """Filter announcements to only include those for specific companies"""
        if not symbols:
            return announcements
            
        # Convert symbols to uppercase for case-insensitive matching
        symbols_upper = [s.upper() for s in symbols]
        filtered = [a for a in announcements if a.get('Symbol', '').upper() in symbols_upper]
        
        logging.info(f"Filtered {len(filtered)} announcements for {len(symbols)} companies")
        return filtered
    
    def __del__(self):
        """Clean up resources on object destruction"""
        self._close_selenium()
    
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='PSX Announcement Scraper')
    parser.add_argument('--fresh', action='store_true', help='Force a fresh scrape (bypass cache)')
    parser.add_argument('--company', type=str, help='Scrape a specific company page by symbol')
    parser.add_argument('--skip-company-pages', action='store_true', help='Skip scraping company-specific pages')
    parser.add_argument('--test-companies', type=str, help='Comma-separated list of symbols to test (e.g., "EPCL,UBL,PSO")')
    args = parser.parse_args()
    
    logging.info("Starting PSX Announcement scraper")
    try:
        scraper = PSXAnnouncementScraper()
        
        # Load KMI100 companies
        company_data = scraper._load_company_data()
        all_symbols = list(company_data.keys())
        logging.info(f"Loaded {len(all_symbols)} symbols from KMI100 sheet")
        
        # Check if EPCL is in the list
        if 'EPCL' in all_symbols:
            logging.info(f"EPCL symbol found in position {all_symbols.index('EPCL')+1} of {len(all_symbols)}")
        else:
            logging.warning(f"EPCL symbol NOT FOUND in company list. Available symbols: {', '.join(all_symbols[:10])}...")
        
        # If a specific company is requested, only scrape that company
        if args.company:
            company_announcements = scraper.scrape_specific_company(args.company.upper())
            logging.info(f"Finished scraping specific company {args.company.upper()}")
            sys.exit(0)
        
        # Parse test companies if provided
        test_companies = None
        if args.test_companies:
            test_companies = [symbol.strip().upper() for symbol in args.test_companies.split(',')]
            logging.info(f"Using test companies: {', '.join(test_companies)}")
        
        # Limit to 30 symbols for testing if there are too many and no test companies specified
        test_symbols = all_symbols
        if len(all_symbols) > 30 and not test_companies:
            test_symbols = all_symbols[:30]
            logging.info(f"Using first 30 symbols for focused testing")
        
        # Determine if we should use cache or force fresh scrape
        force_fresh = args.fresh
        if force_fresh:
            logging.info("Forcing fresh scrape (bypassing cache)")
            # Delete the cache file if we're doing a fresh scrape
            cache_file = os.path.join("data", "cache", "psx_announcements_cache.pkl")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    logging.info(f"Deleted cache file: {cache_file}")
                except Exception as e:
                    logging.warning(f"Could not delete cache file: {e}")
        
        # Determine whether to scrape company pages
        scrape_company_pages = not args.skip_company_pages
        if args.skip_company_pages:
            logging.info("Skipping company-specific pages")
        
        # Scrape announcements
        logging.info("Starting announcement scraping")
        announcements = scraper.scrape_announcements(
            scrape_company_pages=scrape_company_pages,
            test_companies=test_companies,
            force_fresh=force_fresh
        )
        logging.info(f"Scraped {len(announcements)} announcements")
        
        if announcements:
            logging.info(f"Successfully scraped {len(announcements)} announcements")
            
            # Check for EPCL in announcements
            epcl_announcements = [a for a in announcements if a.get('Symbol') == 'EPCL']
            if epcl_announcements:
                logging.info(f"Found {len(epcl_announcements)} announcements for EPCL: {', '.join([a.get('Subject', '')[:30] + '...' for a in epcl_announcements])}")
            else:
                logging.warning("No EPCL announcements found in scraped data")
            
            # Save general announcements to CSV
            general_csv = scraper._save_to_csv(announcements, prefix="all_")
            if general_csv:
                logging.info(f"All announcements CSV created at: {general_csv}")
            
            # Filter announcements for KMI100 companies only (first 100 companies)
            kmi100_announcements = scraper._filter_kmi100_announcements(announcements)
            
            # Save KMI100 filtered announcements to CSV
            if kmi100_announcements:
                kmi100_csv = scraper._save_to_csv(kmi100_announcements, prefix="kmi100_")
                if kmi100_csv:
                    logging.info(f"KMI100 announcements CSV created at: {kmi100_csv}")
            
            # Save announcements to Excel with KMIALLSHRAnnouncements and KMI100Announcements sheets
            excel_file = scraper._save_to_excel(announcements, kmi100_announcements)
            if excel_file:
                logging.info(f"Announcements Excel file created at: {excel_file}")
            
            # Save to database
            scraper._save_to_db(announcements)
        
    except Exception as e:
        logging.error(f"Error during scraping: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Finished PSX Announcement scraping")