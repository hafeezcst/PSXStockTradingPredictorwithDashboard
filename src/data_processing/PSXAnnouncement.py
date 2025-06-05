"""
PSX Announcements scraper and database manager.
"""

import os
import re
import time
import json
import logging
import requests
import pandas as pd
import argparse
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime, date, timedelta
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, MetaData, Table, inspect, text, Column, String, DateTime, Integer, Float, ForeignKey
from sqlalchemy.orm import declarative_base
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

# Define additional paths
DATA_CACHE_DIR = project_root / "data" / "cache"
DATA_CSV_DIR = project_root / "data" / "csv"
DATA_EXCEL_DIR = project_root / "data" / "excel"
ANNOUNCEMENTS_DB_PATH = project_root / "data" / "databases" / "production" / "PSXCompanyAnnouncements.db"
SYMBOLS_DB_PATH = project_root / "data" / "databases" / "production" / "PSXSymbols.db"

# Ensure directories exist
for dir_path in [DATA_CACHE_DIR, DATA_CSV_DIR, DATA_EXCEL_DIR, ANNOUNCEMENTS_DB_PATH.parent]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging
log_file = Path(DATA_LOGS_DIR) / 'psx_announcements.log'
# Ensure log directory exists
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create SQLAlchemy Base
Base = declarative_base()

class Company(Base):
    """Company model for database."""
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    sector = Column(String(100))
    is_kmi30 = Column(Integer, default=0)  # 0 or 1
    is_kmi100 = Column(Integer, default=0)  # 0 or 1
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Announcement(Base):
    """Announcement model for database."""
    __tablename__ = 'announcements'
    
    id = Column(Integer, primary_key=True)
    announcement_id = Column(String(32), unique=True, nullable=False, index=True)  # MD5 hash
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    time = Column(String(20))
    subject = Column(String(500))
    url = Column(String(500))
    status = Column(String(20))
    category = Column(String(50))
    source = Column(String(50))  # 'main_page' or 'company_page'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AnnouncementDatabaseManager:
    """Manager for PSX Announcements database."""
    
    def __init__(self, db_path: str = str(ANNOUNCEMENTS_DB_PATH)):
        """Initialize database manager."""
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.symbols_engine = create_engine(f'sqlite:///{SYMBOLS_DB_PATH}')
        self.setup_database()
        
    def setup_database(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
            logging.info("Database tables created successfully")
        except Exception as e:
            logging.error(f"Error creating database tables: {e}")
            
    def save_to_db(self, data: pd.DataFrame, table_name: str):
        """Save DataFrame to database"""
        try:
            if not data.empty:
                data.to_sql(table_name, self.engine, if_exists='append', index=True)
                logging.info(f"Saved {len(data)} rows to table {table_name}")
            else:
                logging.info(f"No data to save for table {table_name}")
        except Exception as e:
            logging.error(f"Error saving data to table {table_name}: {e}")
            
    def upsert_company(self, symbol: str, name: str, sector: str = None, 
                      is_kmi30: bool = False, is_kmi100: bool = False) -> Optional[int]:
        """Insert or update company information."""
        try:
            with self.engine.begin() as conn:  # This automatically handles commit/rollback
                # Check if company exists
                result = conn.execute(
                    text("SELECT id FROM companies WHERE symbol = :symbol"),
                    {"symbol": symbol}
                ).fetchone()
                
                if result:
                    # Update existing company
                    conn.execute(
                        text("""
                        UPDATE companies 
                        SET name = :name, sector = :sector, 
                            is_kmi30 = :is_kmi30, is_kmi100 = :is_kmi100,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = :symbol
                        """),
                        {
                            "symbol": symbol,
                            "name": name,
                            "sector": sector,
                            "is_kmi30": int(is_kmi30),
                            "is_kmi100": int(is_kmi100)
                        }
                    )
                    return result[0]
                else:
                    # Insert new company
                    result = conn.execute(
                        text("""
                        INSERT INTO companies (symbol, name, sector, is_kmi30, is_kmi100)
                        VALUES (:symbol, :name, :sector, :is_kmi30, :is_kmi100)
                        RETURNING id
                        """),
                        {
                            "symbol": symbol,
                            "name": name,
                            "sector": sector,
                            "is_kmi30": int(is_kmi30),
                            "is_kmi100": int(is_kmi100)
                        }
                    )
                    return result.fetchone()[0]
        except Exception as e:
            logging.error(f"Error upserting company {symbol}: {e}")
            return None
            
    def save_announcement(self, announcement: Dict[str, Any], company_id: int) -> bool:
        """Save announcement to database."""
        try:
            with self.engine.begin() as conn:  # This automatically handles commit/rollback
                # Check if announcement exists
                result = conn.execute(
                    text("SELECT id FROM announcements WHERE announcement_id = :announcement_id"),
                    {"announcement_id": announcement['ID']}
                ).fetchone()
                
                if result:
                    # Update existing announcement
                    conn.execute(
                        text("""
                        UPDATE announcements 
                        SET date = :date, time = :time, subject = :subject,
                            url = :url, status = :status, category = :category,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE announcement_id = :announcement_id
                        """),
                        {
                            "announcement_id": announcement['ID'],
                            "date": announcement['Date'],
                            "time": announcement['Time'],
                            "subject": announcement['Subject'],
                            "url": announcement['URL'],
                            "status": announcement['Status'],
                            "category": announcement['Category']
                        }
                    )
                else:
                    # Insert new announcement
                    conn.execute(
                        text("""
                        INSERT INTO announcements (
                            announcement_id, company_id, date, time, subject,
                            url, status, category, source
                        ) VALUES (
                            :announcement_id, :company_id, :date, :time, :subject,
                            :url, :status, :category, :source
                        )
                        """),
                        {
                            "announcement_id": announcement['ID'],
                            "company_id": company_id,
                            "date": announcement['Date'],
                            "time": announcement['Time'],
                            "subject": announcement['Subject'],
                            "url": announcement['URL'],
                            "status": announcement['Status'],
                            "category": announcement['Category'],
                            "source": announcement.get('Source', 'main_page')
                        }
                    )
                return True
        except Exception as e:
            logging.error(f"Error saving announcement {announcement['ID']}: {e}")
            return False
            
    def get_company_announcements(self, symbol: str, start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """Get announcements for a specific company."""
        try:
            query = """
            SELECT a.*, c.symbol, c.name as company_name
            FROM announcements a
            JOIN companies c ON a.company_id = c.id
            WHERE c.symbol = :symbol
            """
            
            params = {"symbol": symbol}
            
            if start_date:
                query += " AND a.date >= :start_date"
                params["start_date"] = start_date
            if end_date:
                query += " AND a.date <= :end_date"
                params["end_date"] = end_date
                
            query += " ORDER BY a.date DESC"
            
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn, params=params)
            return df
        except Exception as e:
            logging.error(f"Error getting announcements for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_latest_announcements(self, days: int = 7) -> pd.DataFrame:
        """Get latest announcements across all companies."""
        try:
            query = """
            SELECT a.*, c.symbol, c.name as company_name
            FROM announcements a
            JOIN companies c ON a.company_id = c.id
            WHERE a.date >= date('now', :days_ago)
            ORDER BY a.date DESC, a.time DESC
            """
            
            with self.engine.connect() as conn:
                df = pd.read_sql_query(
                    text(query),
                    conn,
                    params={"days_ago": f'-{days} days'}
                )
            return df
        except Exception as e:
            logging.error(f"Error getting latest announcements: {e}")
            return pd.DataFrame()
            
    def get_announcement_stats(self) -> Dict[str, Any]:
        """Get announcement statistics."""
        try:
            stats = {
                'total_announcements': 0,
                'total_companies': 0,
                'latest_date': None,
                'category_counts': {},
                'company_counts': {}
            }
            
            with self.engine.connect() as conn:
                # Get total counts
                result = conn.execute(text("SELECT COUNT(*) FROM announcements")).fetchone()
                stats['total_announcements'] = result[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM companies")).fetchone()
                stats['total_companies'] = result[0]
                
                # Get latest date
                result = conn.execute(text("SELECT MAX(date) FROM announcements")).fetchone()
                stats['latest_date'] = result[0]
                
                # Get category counts
                result = conn.execute(text("""
                    SELECT category, COUNT(*) as count 
                    FROM announcements 
                    GROUP BY category
                """))
                stats['category_counts'] = {row[0]: row[1] for row in result}
                
                # Get company announcement counts
                result = conn.execute(text("""
                    SELECT c.symbol, COUNT(*) as count 
                    FROM announcements a
                    JOIN companies c ON a.company_id = c.id
                    GROUP BY c.symbol
                    ORDER BY count DESC
                    LIMIT 10
                """))
                stats['company_counts'] = {row[0]: row[1] for row in result}
                
            return stats
        except Exception as e:
            logging.error(f"Error getting announcement stats: {e}")
            return {}

    def get_sector_info(self, symbol: str) -> Optional[str]:
        """Get sector information for a symbol from PSXSymbols.db."""
        try:
            # Special handling for fund symbols
            if any(x in symbol.upper() for x in ['-FUNDS', 'ETF', 'FUND']):
                return 'Mutual Funds'
                
            with self.symbols_engine.connect() as conn:
                # First verify if table exists
                table_check = text("SELECT name FROM sqlite_master WHERE type='table' AND name='KSEALL'")
                if not conn.execute(table_check).fetchone():
                    logging.error("Table KSEALL does not exist in PSXSymbols.db")
                    return None
                    
                # Query sector information
                query = text("SELECT sector FROM KSEALL WHERE symbol = :symbol")
                result = conn.execute(query, {"symbol": symbol}).fetchone()
                if result:
                    logging.debug(f"Found sector information for {symbol}: {result[0]}")
                    return result[0]
                else:
                    # Try without any suffixes
                    base_symbol = symbol.split('-')[0]  # Remove any suffixes after hyphen
                    result = conn.execute(query, {"symbol": base_symbol}).fetchone()
                    if result:
                        logging.debug(f"Found sector information for {symbol} using base symbol {base_symbol}: {result[0]}")
                        return result[0]
                    else:
                        logging.warning(f"No sector information found for symbol {symbol} in KSEALL table")
                        return None
        except Exception as e:
            logging.error(f"Error getting sector info for {symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None

class PSXAnnouncementScraper:
    """Class for scraping PSX announcements"""
    
    # Constants
    BASE_URL = "https://dps.psx.com.pk/announcements/companies"
    WAIT_TIMEOUT = 120  # Increased timeout to 120 seconds
    PAGE_LOAD_TIMEOUT = 60  # Separate timeout for page load
    
    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.cache_file = DATA_CACHE_DIR / "psx_announcements_cache.json"
        self.last_scrape_time = None
        self.db_manager = AnnouncementDatabaseManager()
        self.company_data = None
        self.company_data_file = SYMBOLS_FILE
        self.driver = None
        self._setup_selenium()
        
    def _setup_selenium(self):
        """Set up Selenium WebDriver with appropriate options."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-notifications')
            chrome_options.add_argument('--disable-infobars')
            chrome_options.add_argument('--disable-logging')
            chrome_options.add_argument('--log-level=3')
            chrome_options.add_argument('--silent')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Add experimental options
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.PAGE_LOAD_TIMEOUT)
            logging.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logging.error(f"Error setting up Selenium: {e}")
            raise
            
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
                
    def scrape_announcements(self, max_retries: int = 3, force_fresh: bool = False) -> List[Dict]:
        """
        Scrape announcements from PSX website
        
        Args:
            max_retries (int, optional): Maximum number of retries. Defaults to 3.
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
        
        # Scrape announcements from main page
        announcements = self._scrape_main_announcements_page(max_retries)
        logging.info(f"Scraped {len(announcements)} announcements")
        
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
    
    def _scrape_main_announcements_page(self, max_retries: int = 3) -> List[Dict]:
        """Scrape the main announcements page."""
        for attempt in range(max_retries):
            try:
                logging.info(f"Starting main announcements page scraping (attempt {attempt + 1})")
                
                # Clear browser cache and cookies before each attempt
                self.driver.delete_all_cookies()
                
                # Load the page with explicit wait for page load
                self.driver.get(self.BASE_URL)
                
                # Wait for the page to be fully loaded
                wait = WebDriverWait(self.driver, self.WAIT_TIMEOUT)
                
                # First wait for the page to be ready
                wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
                
                # Wait for any loading indicators to disappear
                try:
                    wait.until_not(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".loading, .spinner, .progress"))
                    )
                except TimeoutException:
                    logging.warning("Loading indicator not found or didn't disappear")
                
                # Wait for the announcements table
                table = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table.tbl#announcementsTable"))
                )
                wait.until(EC.visibility_of(table))
                
                # Additional wait for table content to load
                time.sleep(5)  # Give extra time for dynamic content
                
                # Get the page source and parse with BeautifulSoup
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                table = soup.find('table', {'class': 'tbl', 'id': 'announcementsTable'})
                
                if not table:
                    logging.error("Could not find announcements table")
                    continue
                
                logging.info("Found announcements table, processing rows")
                announcements = []
                symbol_set = set()
                
                # Process the table
                rows = table.find('tbody', class_='tbl__body').find_all('tr')
                logging.info(f"Found {len(rows)} announcement rows")
                
                for row in rows:
                    try:
                        cols = row.find_all('td')
                        if len(cols) < 6:  # We expect 6 columns
                            continue
                            
                        # Extract data from columns
                        date_str = cols[0].text.strip()
                        time_str = cols[1].text.strip()
                        
                        # Get symbol and company name from links
                        symbol_link = cols[2].find('a', class_='tbl__symbol')
                        name_link = cols[3].find('a', class_='tbl__symbol')
                        
                        symbol = symbol_link.find('strong').text.strip() if symbol_link else "UNKNOWN"
                        company_name = name_link.find('strong').text.strip() if name_link else "UNKNOWN"
                        
                        # Get title
                        title = cols[4].text.strip()
                        
                        # Get PDF link
                        url = ""
                        pdf_link = cols[5].find('a', href=lambda x: x and x.endswith('.pdf'))
                        if pdf_link:
                            url = pdf_link['href']
                            if not url.startswith('http'):
                                url = f"https://dps.psx.com.pk{url}"
                        
                        # Parse date
                        try:
                            date_obj = datetime.strptime(date_str, '%b %d, %Y')
                            date_formatted = date_obj.strftime('%Y-%m-%d')
                        except ValueError as e:
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
                            logging.debug(f"Added announcement for {symbol}: {title[:50]}...")
                            
                    except Exception as e:
                        logging.error(f"Error processing row: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
                        continue
                
                if announcements:
                    logging.info(f"Successfully extracted {len(announcements)} announcements")
                    return announcements
                else:
                    logging.warning(f"No announcements extracted on attempt {attempt + 1}")
                    
            except TimeoutException:
                logging.warning(f"Timeout waiting for announcements table (attempt {attempt + 1})")
            except Exception as e:
                logging.error(f"Error scraping main page (attempt {attempt + 1}): {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
            
            if attempt < max_retries - 1:
                time.sleep(10)  # Wait before retrying
                
        logging.error("Failed to scrape announcements after all retries")
        return []
    
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
        try:
            # Ensure dates are converted to strings for JSON serialization
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
        """Load announcements from local cache file"""
        if not self.cache_file.exists():
            logging.warning("Cache file not found")
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
        """Save announcements to both the original database and the new announcements database."""
        try:
            # First save to the original database using DataFrame format
            if announcements:
                # Convert announcements list to DataFrame
                df = pd.DataFrame(announcements)
                
                # Group announcements by symbol
                symbol_groups = df.groupby('Symbol')
                
                # Create engine for original database
                original_engine = create_engine(f'sqlite:///{PSX_SYM_PATH}')
                
                # Save each symbol's announcements to its own table
                with original_engine.begin() as conn:  # This automatically handles commit/rollback
                    for symbol, group_df in symbol_groups:
                        if not group_df.empty:
                            table_name = f"PSX_{symbol}_announcements"
                            group_df.to_sql(table_name, conn, if_exists='append', index=False)
                            logging.info(f"Saved {len(group_df)} announcements for {symbol} to original database")
                
                # Then save to the new announcements database
                announcements_db = AnnouncementDatabaseManager()
                
                # Load company data once
                company_data = self._load_company_data()
                all_symbols = list(company_data.keys())
                
                # Process each announcement
                for announcement in announcements:
                    symbol = announcement.get('Symbol', 'UNKNOWN')
                    company_name = announcement.get('Company', f"{symbol} Limited")
                    
                    # Get sector information
                    sector = announcements_db.get_sector_info(symbol)
                    if sector:
                        logging.info(f"Found sector '{sector}' for {symbol}")
                    else:
                        # Try to determine sector from company name or symbol
                        company_name_upper = company_name.upper()
                        if any(x in symbol.upper() for x in ['-FUNDS', 'ETF', 'FUND']):
                            sector = 'Mutual Funds'
                            logging.info(f"Assigned sector 'Mutual Funds' for {symbol} based on symbol")
                        elif any(x in company_name_upper for x in ['BANK', 'FINANCIAL', 'INVESTMENT', 'SECURITIES']):
                            sector = 'Finance'
                            logging.info(f"Assigned sector 'Finance' for {symbol} based on company name")
                        elif any(x in company_name_upper for x in ['TEXTILE', 'SPINNING', 'WEAVING']):
                            sector = 'Consumer Non-Durables'
                            logging.info(f"Assigned sector 'Consumer Non-Durables' for {symbol} based on company name")
                        elif any(x in company_name_upper for x in ['CEMENT', 'CONCRETE']):
                            sector = 'Non-Energy Minerals'
                            logging.info(f"Assigned sector 'Non-Energy Minerals' for {symbol} based on company name")
                        elif any(x in company_name_upper for x in ['OIL', 'GAS', 'PETROLEUM']):
                            sector = 'Energy Minerals'
                            logging.info(f"Assigned sector 'Energy Minerals' for {symbol} based on company name")
                        elif any(x in company_name_upper for x in ['TELECOM', 'COMMUNICATION']):
                            sector = 'Communications'
                            logging.info(f"Assigned sector 'Communications' for {symbol} based on company name")
                        else:
                            # Default to 'Commercial Services' for unknown sectors
                            sector = 'Commercial Services'
                            logging.warning(f"No specific sector found for {symbol}, defaulting to 'Commercial Services'")
                    
                    # Determine KMI status based on symbol position
                    is_kmi30 = symbol in all_symbols[:30]
                    is_kmi100 = symbol in all_symbols[:100]
                    
                    # Save company and get company_id
                    company_id = announcements_db.upsert_company(
                        symbol=symbol,
                        name=company_name,
                        sector=sector,  # Add sector information
                        is_kmi30=is_kmi30,
                        is_kmi100=is_kmi100
                    )
                    
                    if company_id:
                        # Save announcement
                        announcements_db.save_announcement(announcement, company_id)
            
            logging.info(f"Saved {len(announcements)} announcements to both databases")
        except Exception as e:
            logging.error(f"Error saving to databases: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _save_to_csv(self, announcements: List[Dict], prefix="") -> str:
        """Save announcements to a CSV file for easy viewing"""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = DATA_CSV_DIR / f"{prefix}psx_announcements_{timestamp}.csv"
            
            # Convert to DataFrame
            df = pd.DataFrame(announcements)
            
            # Ensure columns are in desired order and format
            if len(df) > 0:
                # Define the exact column order and names as in the existing CSV
                column_order = [
                    'Symbol',
                    'Company',
                    'Date',
                    'Time',
                    'Subject',
                    'URL',
                    'Status',
                    'Category'
                ]
                
                # Ensure all required columns exist
                for col in column_order:
                    if col not in df.columns:
                        df[col] = ''
                
                # Reorder columns
                df = df[column_order]
                
                # Format date column if it exists
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                
                # Ensure all string columns are properly formatted
                string_columns = ['Symbol', 'Company', 'Subject', 'URL', 'Status', 'Category']
                for col in string_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip()
            
            # Save to CSV with specific formatting
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logging.info(f"Saved {len(announcements)} announcements to CSV file: {csv_file}")
            return str(csv_file)
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return ""
    
    def _save_to_excel(self, all_announcements: List[Dict], kmi100_announcements: List[Dict]) -> str:
        """
        Save announcements to Excel file with two sheets, appending new announcements while preserving existing ones
        
        Args:
            all_announcements (List[Dict]): List of all announcements
            kmi100_announcements (List[Dict]): List of filtered KMI100 announcements
            
        Returns:
            str: Path to saved Excel file
        """
        try:
            excel_file = DATA_EXCEL_DIR / "PSX_Announcements.xlsx"
            
            # Filter for all companies in Excel for KMIALLSHR sheet
            all_excel_companies_announcements = self._filter_all_companies_announcements(all_announcements)
            
            # Convert new announcements to DataFrames
            new_all_df = pd.DataFrame(all_excel_companies_announcements)
            new_kmi100_df = pd.DataFrame(kmi100_announcements)
            
            # Ensure columns are in desired order
            column_order = ['ID', 'Symbol', 'Company', 'Date', 'Time', 'Subject', 'URL', 'Status', 'Category']
            if len(new_all_df) > 0:
                new_all_df = new_all_df[column_order]
            if len(new_kmi100_df) > 0:
                new_kmi100_df = new_kmi100_df[column_order]
            
            # Read existing Excel file if it exists
            if excel_file.exists():
                try:
                    # Read existing data with explicit column types
                    existing_all_df = pd.read_excel(
                        excel_file, 
                        sheet_name='KMIALLSHRAnnouncements',
                        dtype={
                            'ID': str,
                            'Symbol': str,
                            'Company': str,
                            'Date': str,
                            'Time': str,
                            'Subject': str,
                            'URL': str,
                            'Status': str,
                            'Category': str
                        }
                    )
                    existing_kmi100_df = pd.read_excel(
                        excel_file, 
                        sheet_name='KMI100Announcements',
                        dtype={
                            'ID': str,
                            'Symbol': str,
                            'Company': str,
                            'Date': str,
                            'Time': str,
                            'Subject': str,
                            'URL': str,
                            'Status': str,
                            'Category': str
                        }
                    )
                    
                    # Combine existing and new announcements
                    if len(new_all_df) > 0:
                        combined_all_df = pd.concat([existing_all_df, new_all_df], ignore_index=True)
                        # Remove duplicates based on ID
                        combined_all_df = combined_all_df.drop_duplicates(subset=['ID'], keep='last')
                    else:
                        combined_all_df = existing_all_df
                        
                    if len(new_kmi100_df) > 0:
                        combined_kmi100_df = pd.concat([existing_kmi100_df, new_kmi100_df], ignore_index=True)
                        # Remove duplicates based on ID
                        combined_kmi100_df = combined_kmi100_df.drop_duplicates(subset=['ID'], keep='last')
                    else:
                        combined_kmi100_df = existing_kmi100_df
                        
                except Exception as e:
                    logging.warning(f"Error reading existing Excel file: {e}")
                    combined_all_df = new_all_df
                    combined_kmi100_df = new_kmi100_df
            else:
                combined_all_df = new_all_df
                combined_kmi100_df = new_kmi100_df
            
            # Sort by date and time
            if not combined_all_df.empty:
                # Convert date and time to datetime with explicit format
                date_format = '%Y-%m-%d'
                time_format = '%H:%M:%S'
                combined_all_df['DateTime'] = pd.to_datetime(
                    combined_all_df['Date'] + ' ' + combined_all_df['Time'],
                    format=f'{date_format} {time_format}',
                    errors='coerce'
                )
                combined_all_df = combined_all_df.sort_values('DateTime', ascending=False)
                combined_all_df = combined_all_df.drop('DateTime', axis=1)
                
            if not combined_kmi100_df.empty:
                # Convert date and time to datetime with explicit format
                date_format = '%Y-%m-%d'
                time_format = '%H:%M:%S'
                combined_kmi100_df['DateTime'] = pd.to_datetime(
                    combined_kmi100_df['Date'] + ' ' + combined_kmi100_df['Time'],
                    format=f'{date_format} {time_format}',
                    errors='coerce'
                )
                combined_kmi100_df = combined_kmi100_df.sort_values('DateTime', ascending=False)
                combined_kmi100_df = combined_kmi100_df.drop('DateTime', axis=1)
            
            # Create an Excel writer
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                combined_all_df.to_excel(writer, sheet_name='KMIALLSHRAnnouncements', index=False)
                combined_kmi100_df.to_excel(writer, sheet_name='KMI100Announcements', index=False)
            
            logging.info(f"Successfully saved {len(combined_all_df)} total announcements to Excel file")
            logging.info(f"Added {len(new_all_df)} new announcements")
            logging.info(f"Excel file saved at: {excel_file}")
            return str(excel_file)
        except Exception as e:
            logging.error(f"Error saving to Excel: {e}")
            import traceback
            logging.error(traceback.format_exc())
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
                
        logging.info(f"Filtered {len(filtered_announcements)} announcements for {len(all_symbols)} companies from Excel")
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
    
    def scrape_historical_announcements(self, start_date: str, end_date: str = None) -> List[Dict]:
        """
        Scrape historical announcements within a date range
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            List[Dict]: List of announcements
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logging.info(f"Scraping historical announcements from {start_date} to {end_date}")
        
        try:
            # Construct URL with date range
            url = f"{self.BASE_URL}?start_date={start_date}&end_date={end_date}"
            logging.info(f"Scraping with URL: {url}")
            
            # Load the page
            self.driver.get(url)
            
            # Wait for the page to be fully loaded
            wait = WebDriverWait(self.driver, self.WAIT_TIMEOUT)
            wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
            
            # Wait for any loading indicators to disappear
            try:
                wait.until_not(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".loading, .spinner, .progress"))
                )
            except TimeoutException:
                logging.warning("Loading indicator not found or didn't disappear")
            
            # Wait for the announcements table
            table = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.tbl#announcementsTable"))
            )
            wait.until(EC.visibility_of(table))
            
            # Additional wait for table content to load
            time.sleep(5)  # Give extra time for dynamic content
            
            # Get the page source and parse with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            table = soup.find('table', {'class': 'tbl', 'id': 'announcementsTable'})
            
            if not table:
                logging.error("Could not find announcements table")
                return []
            
            # Process the table
            rows = table.find('tbody', class_='tbl__body').find_all('tr')
            if not rows:
                logging.info("No announcements found")
                return []
                
            logging.info(f"Found {len(rows)} announcements")
            
            # Process announcements
            announcements = []
            for row in rows:
                try:
                    cols = row.find_all('td')
                    if len(cols) < 6:  # We expect 6 columns
                        continue
                        
                    # Extract data from columns
                    date_str = cols[0].text.strip()
                    time_str = cols[1].text.strip()
                    
                    # Get symbol and company name from links
                    symbol_link = cols[2].find('a', class_='tbl__symbol')
                    name_link = cols[3].find('a', class_='tbl__symbol')
                    
                    symbol = symbol_link.find('strong').text.strip() if symbol_link else "UNKNOWN"
                    company_name = name_link.find('strong').text.strip() if name_link else "UNKNOWN"
                    
                    # Get title
                    title = cols[4].text.strip()
                    
                    # Get PDF link
                    url = ""
                    pdf_link = cols[5].find('a', href=lambda x: x and x.endswith('.pdf'))
                    if pdf_link:
                        url = pdf_link['href']
                        if not url.startswith('http'):
                            url = f"https://dps.psx.com.pk{url}"
                    
                    # Parse date
                    try:
                        date_obj = datetime.strptime(date_str, '%b %d, %Y')
                        date_formatted = date_obj.strftime('%Y-%m-%d')
                    except ValueError as e:
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
                    
                    announcements.append(announcement)
                    logging.debug(f"Added announcement for {symbol}: {title[:50]}...")
                    
                except Exception as e:
                    logging.error(f"Error processing row: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    continue
            
            logging.info(f"Successfully processed {len(announcements)} announcements")
            return announcements
            
        except Exception as e:
            logging.error(f"Error scraping historical announcements: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []

    def scrape_latest_announcements(self, days: int = 1) -> List[Dict]:
        """
        Scrape latest announcements since last run or for specified number of days
        
        Args:
            days (int, optional): Number of days to look back. Defaults to 1.
            
        Returns:
            List[Dict]: List of announcements
        """
        # Calculate start date
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        logging.info(f"Scraping latest announcements from {start_date} to {end_date}")
        
        # Get announcements from the date range
        announcements = self.scrape_historical_announcements(start_date, end_date)
        
        if announcements:
            # Save to cache
            self._save_to_cache(announcements)
            
            # Save to database and files
            self._save_to_db(announcements)
            self._save_to_csv(announcements, prefix="latest_")
            
            # Save to Excel with KMI100 filtered announcements
            kmi100_announcements = self._filter_kmi100_announcements(announcements)
            self._save_to_excel(announcements, kmi100_announcements)
            
            logging.info(f"Saved {len(announcements)} latest announcements")
        
        return announcements

    def _process_announcement_rows(self, rows: List) -> List[Dict]:
        """Process announcement rows from the table"""
        announcements = []
        
        for row in rows:
            try:
                cols = row.find_all('td')
                if len(cols) < 6:  # We expect 6 columns
                    continue
                    
                # Extract data from columns
                date_str = cols[0].text.strip()
                time_str = cols[1].text.strip()
                
                # Get symbol and company name from links
                symbol_link = cols[2].find('a', class_='tbl__symbol')
                name_link = cols[3].find('a', class_='tbl__symbol')
                
                symbol = symbol_link.find('strong').text.strip() if symbol_link else "UNKNOWN"
                company_name = name_link.find('strong').text.strip() if name_link else "UNKNOWN"
                
                # Get title
                title = cols[4].text.strip()
                
                # Get PDF link
                url = ""
                pdf_link = cols[5].find('a', href=lambda x: x and x.endswith('.pdf'))
                if pdf_link:
                    url = pdf_link['href']
                    if not url.startswith('http'):
                        url = f"https://dps.psx.com.pk{url}"
                
                # Parse date
                try:
                    date_obj = datetime.strptime(date_str, '%b %d, %Y')
                    date_formatted = date_obj.strftime('%Y-%m-%d')
                except ValueError as e:
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
                
                announcements.append(announcement)
                logging.debug(f"Added announcement for {symbol}: {title[:50]}...")
                
            except Exception as e:
                logging.error(f"Error processing row: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        return announcements

    def _copy_csv_to_excel_and_update(self, csv_path: str) -> None:
        """
        Copy existing CSV data to Excel and update with new announcements
        
        Args:
            csv_path (str): Path to the existing CSV file
        """
        try:
            # Read the existing CSV file
            existing_df = pd.read_csv(csv_path)
            logging.info(f"Read {len(existing_df)} announcements from existing CSV: {csv_path}")
            
            # Get the latest announcements
            latest_announcements = self.scrape_latest_announcements(days=1)
            latest_df = pd.DataFrame(latest_announcements)
            
            if not latest_df.empty:
                # Combine existing and latest announcements
                combined_df = pd.concat([existing_df, latest_df], ignore_index=True)
                
                # Remove duplicates based on ID
                combined_df = combined_df.drop_duplicates(subset=['ID'], keep='last')
                
                # Sort by date and time
                combined_df['DateTime'] = pd.to_datetime(combined_df['Date'] + ' ' + combined_df['Time'])
                combined_df = combined_df.sort_values('DateTime', ascending=False)
                combined_df = combined_df.drop('DateTime', axis=1)
                
                # Filter for KMI100 companies
                kmi100_df = self._filter_kmi100_announcements(combined_df.to_dict('records'))
                kmi100_df = pd.DataFrame(kmi100_df)
                
                # Save to Excel with two sheets
                excel_file = DATA_EXCEL_DIR / "PSX_Announcements.xlsx"
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    combined_df.to_excel(writer, sheet_name='KMIALLSHRAnnouncements', index=False)
                    kmi100_df.to_excel(writer, sheet_name='KMI100Announcements', index=False)
                
                logging.info(f"Successfully updated Excel file with {len(combined_df)} total announcements")
                logging.info(f"Including {len(latest_df)} new announcements")
                logging.info(f"Excel file saved at: {excel_file}")
            else:
                logging.warning("No new announcements found to update")
                
        except Exception as e:
            logging.error(f"Error copying CSV to Excel: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def copy_csv_to_excel(self, csv_path: str) -> None:
        """
        Copy announcements from CSV to Excel file
        
        Args:
            csv_path (str): Path to the CSV file
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            logging.info(f"Read {len(df)} announcements from CSV: {csv_path}")
            
            # Filter for KMI100 companies
            kmi100_df = self._filter_kmi100_announcements(df.to_dict('records'))
            kmi100_df = pd.DataFrame(kmi100_df)
            
            # Save to Excel with two sheets
            excel_file = DATA_EXCEL_DIR / "PSX_Announcements.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='KMIALLSHRAnnouncements', index=False)
                kmi100_df.to_excel(writer, sheet_name='KMI100Announcements', index=False)
            
            logging.info(f"Successfully copied {len(df)} announcements to Excel file")
            logging.info(f"Created KMI100 sheet with {len(kmi100_df)} announcements")
            logging.info(f"Excel file saved at: {excel_file}")
            
        except Exception as e:
            logging.error(f"Error copying CSV to Excel: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def add_single_announcement(self, symbol: str, company: str, date: str, time: str, subject: str, url: str, status: str, category: str) -> None:
        """
        Add a single announcement to the Excel file, appending if not a duplicate.
        """
        try:
            # Create announcement dictionary with proper format
            announcement = {
                'ID': hashlib.md5(f"{date}_{subject}_{symbol}".encode()).hexdigest(),
                'Symbol': symbol,
                'Company': company,
                'Date': date,
                'Time': time,
                'Subject': subject,
                'URL': url,
                'Status': status,
                'Category': category
            }
            DATA_EXCEL_DIR.mkdir(parents=True, exist_ok=True)
            excel_file = DATA_EXCEL_DIR / "PSX_Announcements.xlsx"
            # Read existing Excel file if it exists
            if excel_file.exists():
                all_df = pd.read_excel(excel_file, sheet_name='KMIALLSHRAnnouncements')
                kmi100_df = pd.read_excel(excel_file, sheet_name='KMI100Announcements')
                # Only append if not a duplicate (by ID)
                if announcement['ID'] not in all_df['ID'].values:
                    all_df = pd.concat([all_df, pd.DataFrame([announcement])], ignore_index=True)
                # Check if this is a KMI100 company
                if announcement['Symbol'] in self._load_company_data():
                    if announcement['ID'] not in kmi100_df['ID'].values:
                        kmi100_df = pd.concat([kmi100_df, pd.DataFrame([announcement])], ignore_index=True)
            else:
                all_df = pd.DataFrame([announcement])
                kmi100_df = pd.DataFrame([announcement]) if announcement['Symbol'] in self._load_company_data() else pd.DataFrame()
            # Sort by date and time
            if 'Time' not in all_df.columns:
                all_df['Time'] = '00:00:00'
            all_df['Time'] = all_df['Time'].fillna('00:00:00').replace('', '00:00:00')
            all_df['DateTime'] = pd.to_datetime(all_df['Date'].astype(str) + ' ' + all_df['Time'].astype(str), errors='coerce')
            all_df = all_df.sort_values('DateTime', ascending=False)
            all_df = all_df.drop('DateTime', axis=1)
            if not kmi100_df.empty:
                if 'Time' not in kmi100_df.columns:
                    kmi100_df['Time'] = '00:00:00'
                kmi100_df['Time'] = kmi100_df['Time'].fillna('00:00:00').replace('', '00:00:00')
                kmi100_df['DateTime'] = pd.to_datetime(kmi100_df['Date'].astype(str) + ' ' + kmi100_df['Time'].astype(str), errors='coerce')
                kmi100_df = kmi100_df.sort_values('DateTime', ascending=False)
                kmi100_df = kmi100_df.drop('DateTime', axis=1)
            # Save to Excel
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                all_df.to_excel(writer, sheet_name='KMIALLSHRAnnouncements', index=False)
                kmi100_df.to_excel(writer, sheet_name='KMI100Announcements', index=False)
            logging.info(f"Successfully added announcement for {announcement['Symbol']} to Excel file (if not duplicate)")
            logging.info(f"Excel file saved at: {excel_file}")
        except Exception as e:
            logging.error(f"Error adding announcement: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def add_announcements_from_csv(self, csv_path: str) -> None:
        """
        Batch add announcements from a CSV file.
        """
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.add_single_announcement(
                    symbol=row['Symbol'],
                    company=row['Company'],
                    date=row['Date'],
                    time=row['Time'] if 'Time' in row and pd.notna(row['Time']) else "",
                    subject=row['Subject'],
                    url=row['URL'] if 'URL' in row and pd.notna(row['URL']) else "",
                    status=row['Status'],
                    category=row['Category']
                )
            logging.info(f"Successfully added {len(df)} announcements from {csv_path}")
        except Exception as e:
            logging.error(f"Error adding announcements from CSV: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def scrape_historical_announcements_chunked(self, start_date: str, end_date: str = None, chunk_size_days: int = 30) -> List[Dict]:
        """
        Scrape historical announcements in chunks to handle large date ranges efficiently.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
            chunk_size_days (int, optional): Number of days to scrape in each chunk. Defaults to 30.
            
        Returns:
            List[Dict]: List of all announcements
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_announcements = []
        current_start = start
        
        while current_start < end:
            # Calculate chunk end date
            chunk_end = min(current_start + timedelta(days=chunk_size_days), end)
            
            logging.info(f"Scraping chunk from {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            try:
                # Scrape announcements for this chunk
                chunk_announcements = self.scrape_historical_announcements(
                    start_date=current_start.strftime('%Y-%m-%d'),
                    end_date=chunk_end.strftime('%Y-%m-%d')
                )
                
                if chunk_announcements:
                    all_announcements.extend(chunk_announcements)
                    logging.info(f"Found {len(chunk_announcements)} announcements in chunk")
                    
                    # Save chunk to CSV
                    chunk_csv = self._save_to_csv(
                        chunk_announcements,
                        prefix=f"historical_{current_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}_"
                    )
                    
                    # Save chunk to database
                    self._save_to_db(chunk_announcements)
                    
                    # Save chunk to Excel
                    kmi100_chunk = self._filter_kmi100_announcements(chunk_announcements)
                    self._save_to_excel(chunk_announcements, kmi100_chunk)
                
                # Move to next chunk
                current_start = chunk_end + timedelta(days=1)
                
                # Add a small delay between chunks to avoid overwhelming the server
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"Error scraping chunk {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}: {e}")
                # Continue with next chunk even if this one failed
                current_start = chunk_end + timedelta(days=1)
                continue
        
        # Save final combined results
        if all_announcements:
            final_csv = self._save_to_csv(
                all_announcements,
                prefix=f"historical_combined_{start_date.replace('-', '')}_{end_date.replace('-', '')}_"
            )
            
            # Save final combined results to Excel
            kmi100_final = self._filter_kmi100_announcements(all_announcements)
            self._save_to_excel(all_announcements, kmi100_final)
            
            logging.info(f"Successfully scraped {len(all_announcements)} total announcements from {start_date} to {end_date}")
        
        return all_announcements

    def sync_database_to_excel(self) -> None:
        """
        Sync all announcements from the database to Excel file.
        This ensures the Excel file contains all announcements from the database.
        """
        try:
            logging.info("Starting database to Excel sync")
            
            # Query all announcements from database
            query = """
            SELECT a.*, c.symbol, c.name as company_name
            FROM announcements a
            JOIN companies c ON a.company_id = c.id
            ORDER BY a.date DESC, a.time DESC
            """
            
            with self.db_manager.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn)
            
            if df.empty:
                logging.warning("No announcements found in database")
                return
            
            logging.info(f"Found {len(df)} announcements in database")
            
            # Prepare all announcements for Excel (no filtering)
            # Map DB columns to Excel columns
            all_announcements = []
            for _, row in df.iterrows():
                all_announcements.append({
                    'ID': row['announcement_id'],
                    'Symbol': row['symbol'],
                    'Company': row['company_name'],
                    'Date': str(row['date'])[:10],
                    'Time': row['time'] if 'time' in row and pd.notna(row['time']) else '',
                    'Subject': row['subject'],
                    'URL': row['url'],
                    'Status': row['status'],
                    'Category': row['category']
                })
            
            # Filter for KMI100 companies
            kmi100_announcements = self._filter_kmi100_announcements(all_announcements)
            
            # Save to Excel (bypass _filter_all_companies_announcements)
            excel_file = DATA_EXCEL_DIR / "PSX_Announcements.xlsx"
            column_order = ['ID', 'Symbol', 'Company', 'Date', 'Time', 'Subject', 'URL', 'Status', 'Category']
            all_df = pd.DataFrame(all_announcements)[column_order]
            kmi100_df = pd.DataFrame(kmi100_announcements)[column_order] if kmi100_announcements else pd.DataFrame(columns=column_order)
            
            # Sort by date and time
            if not all_df.empty:
                all_df['DateTime'] = pd.to_datetime(all_df['Date'].astype(str) + ' ' + all_df['Time'].astype(str), errors='coerce')
                all_df = all_df.sort_values('DateTime', ascending=False)
                all_df = all_df.drop('DateTime', axis=1)
            if not kmi100_df.empty:
                kmi100_df['DateTime'] = pd.to_datetime(kmi100_df['Date'].astype(str) + ' ' + kmi100_df['Time'].astype(str), errors='coerce')
                kmi100_df = kmi100_df.sort_values('DateTime', ascending=False)
                kmi100_df = kmi100_df.drop('DateTime', axis=1)
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                all_df.to_excel(writer, sheet_name='KMIALLSHRAnnouncements', index=False)
                kmi100_df.to_excel(writer, sheet_name='KMI100Announcements', index=False)
            
            logging.info(f"Successfully synced {len(all_df)} announcements to Excel file")
            logging.info(f"Excel file saved at: {excel_file}")
            
        except Exception as e:
            logging.error(f"Error syncing database to Excel: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def verify_sync(self) -> Dict[str, Any]:
        """
        Verify and confirm sync between database and Excel file.
        Returns a dictionary with sync statistics and any discrepancies found.
        """
        try:
            logging.info("Starting sync verification between database and Excel file")
            
            # Get data from database
            db_query = """
            SELECT a.*, c.symbol, c.name as company_name
            FROM announcements a
            JOIN companies c ON a.company_id = c.id
            ORDER BY a.date DESC, a.time DESC
            """
            
            with self.db_manager.engine.connect() as conn:
                db_df = pd.read_sql_query(text(db_query), conn)
            
            if db_df.empty:
                logging.warning("No announcements found in database")
                return {"status": "error", "message": "No announcements in database"}
            
            # Get data from Excel
            excel_file = DATA_EXCEL_DIR / "PSX_Announcements.xlsx"
            if not excel_file.exists():
                logging.warning("Excel file not found")
                return {"status": "error", "message": "Excel file not found"}
            
            excel_df = pd.read_excel(excel_file, sheet_name='KMIALLSHRAnnouncements')
            
            # Basic statistics
            stats = {
                "database_count": len(db_df),
                "excel_count": len(excel_df),
                "matching_count": 0,
                "missing_in_excel": [],
                "missing_in_db": [],
                "mismatched_dates": [],
                "mismatched_subjects": []
            }
            
            # Convert IDs to sets for comparison
            db_ids = set(db_df['announcement_id'].values)
            excel_ids = set(excel_df['ID'].values)
            
            # Find missing announcements
            stats["missing_in_excel"] = list(db_ids - excel_ids)
            stats["missing_in_db"] = list(excel_ids - db_ids)
            
            # Find matching announcements
            matching_ids = db_ids.intersection(excel_ids)
            stats["matching_count"] = len(matching_ids)
            
            # Compare matching announcements for discrepancies
            for ann_id in matching_ids:
                db_row = db_df[db_df['announcement_id'] == ann_id].iloc[0]
                excel_row = excel_df[excel_df['ID'] == ann_id].iloc[0]
                
                # Check date
                if str(db_row['date']) != str(excel_row['Date']):
                    stats["mismatched_dates"].append({
                        "id": ann_id,
                        "db_date": str(db_row['date']),
                        "excel_date": str(excel_row['Date'])
                    })
                
                # Check subject
                if str(db_row['subject']) != str(excel_row['Subject']):
                    stats["mismatched_subjects"].append({
                        "id": ann_id,
                        "db_subject": str(db_row['subject']),
                        "excel_subject": str(excel_row['Subject'])
                    })
            
            # Determine sync status
            if (len(stats["missing_in_excel"]) == 0 and 
                len(stats["missing_in_db"]) == 0 and 
                len(stats["mismatched_dates"]) == 0 and 
                len(stats["mismatched_subjects"]) == 0):
                stats["status"] = "fully_synced"
                stats["message"] = "Database and Excel file are fully synchronized"
            else:
                stats["status"] = "partially_synced"
                stats["message"] = "Found discrepancies between database and Excel file"
            
            # Log results
            logging.info(f"Sync verification results:")
            logging.info(f"Database announcements: {stats['database_count']}")
            logging.info(f"Excel announcements: {stats['excel_count']}")
            logging.info(f"Matching announcements: {stats['matching_count']}")
            logging.info(f"Missing in Excel: {len(stats['missing_in_excel'])}")
            logging.info(f"Missing in Database: {len(stats['missing_in_db'])}")
            logging.info(f"Mismatched dates: {len(stats['mismatched_dates'])}")
            logging.info(f"Mismatched subjects: {len(stats['mismatched_subjects'])}")
            
            return stats
            
        except Exception as e:
            logging.error(f"Error verifying sync: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='PSX Announcement Scraper')
    parser.add_argument('--fresh', action='store_true', help='Force a fresh scrape (bypass cache)')
    parser.add_argument('--company-symbol', type=str, help='Scrape a specific company page by symbol')
    parser.add_argument('--skip-company-pages', action='store_true', help='Skip scraping company-specific pages')
    parser.add_argument('--test-companies', type=str, help='Comma-separated list of symbols to test (e.g., "EPCL,UBL,PSO")')
    parser.add_argument('--historical', action='store_true', help='Scrape historical announcements')
    parser.add_argument('--start-date', type=str, help='Start date for historical scraping (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for historical scraping (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=1, help='Number of days to look back for latest announcements')
    parser.add_argument('--copy-csv', type=str, help='Copy existing CSV to Excel and update with latest announcements')
    parser.add_argument('--csv-to-excel', type=str, help='Copy announcements from CSV to Excel file')
    parser.add_argument('--add-announcement', action='store_true', help='Add a single announcement')
    parser.add_argument('--symbol', type=str, help='Company symbol')
    parser.add_argument('--company-name', type=str, help='Company name')
    parser.add_argument('--date', type=str, help='Announcement date')
    parser.add_argument('--time', type=str, help='Announcement time')
    parser.add_argument('--subject', type=str, help='Announcement subject')
    parser.add_argument('--url', type=str, help='Announcement URL')
    parser.add_argument('--status', type=str, help='Announcement status')
    parser.add_argument('--category', type=str, help='Announcement category')
    parser.add_argument('--add-csv', type=str, help='Add all announcements from a CSV file')
    parser.add_argument('--historical-chunked', action='store_true', help='Scrape historical announcements in chunks')
    parser.add_argument('--chunk-size', type=int, default=30, help='Number of days to scrape in each chunk')
    parser.add_argument('--sync-db-to-excel', action='store_true', help='Sync all announcements from database to Excel file')
    parser.add_argument('--verify-sync', action='store_true', help='Verify sync between database and Excel file')
    args = parser.parse_args()
    
    logging.info("Starting PSX Announcement scraper")
    try:
        scraper = PSXAnnouncementScraper()
        
        if args.verify_sync:
            # Verify sync between database and Excel
            sync_stats = scraper.verify_sync()
            print("\nSync Verification Results:")
            print(f"Status: {sync_stats['status']}")
            print(f"Message: {sync_stats['message']}")
            print(f"Database announcements: {sync_stats['database_count']}")
            print(f"Excel announcements: {sync_stats['excel_count']}")
            print(f"Matching announcements: {sync_stats['matching_count']}")
            print(f"Missing in Excel: {len(sync_stats['missing_in_excel'])}")
            print(f"Missing in Database: {len(sync_stats['missing_in_db'])}")
            print(f"Mismatched dates: {len(sync_stats['mismatched_dates'])}")
            print(f"Mismatched subjects: {len(sync_stats['mismatched_subjects'])}")
        elif args.sync_db_to_excel:
            # Sync database to Excel
            scraper.sync_database_to_excel()
        elif args.add_announcement:
            # Add single announcement
            if not all([args.symbol, args.company_name, args.date, args.subject, args.url, args.status, args.category]):
                logging.error("Missing required arguments for adding announcement")
                sys.exit(1)
            scraper.add_single_announcement(
                symbol=args.symbol,
                company=args.company_name,
                date=args.date,
                time=args.time or "",
                subject=args.subject,
                url=args.url,
                status=args.status,
                category=args.category
            )
        elif args.csv_to_excel:
            # Copy CSV to Excel
            scraper.copy_csv_to_excel(args.csv_to_excel)
        elif args.copy_csv:
            # Copy existing CSV to Excel and update
            scraper._copy_csv_to_excel_and_update(args.copy_csv)
        elif args.historical_chunked:
            # Scrape historical announcements in chunks
            if not args.start_date:
                logging.error("Start date is required for historical scraping")
                sys.exit(1)
                
            announcements = scraper.scrape_historical_announcements_chunked(
                start_date=args.start_date,
                end_date=args.end_date,
                chunk_size_days=args.chunk_size
            )
        elif args.add_csv:
            scraper.add_announcements_from_csv(args.add_csv)
        else:
            # Scrape latest announcements
            announcements = scraper.scrape_latest_announcements(days=args.days)
        
        logging.info(f"Finished scraping {len(announcements) if 'announcements' in locals() else 0} announcements")
        
    except Exception as e:
        logging.error(f"Error during scraping: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Finished PSX Announcement scraping")