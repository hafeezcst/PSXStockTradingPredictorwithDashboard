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
from datetime import datetime, date
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, MetaData, Table, inspect, text, Column, String, DateTime, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
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
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.cache_file = DATA_CACHE_DIR / "psx_announcements_cache.json"
        self.last_scrape_time = None
        self.db_manager = AnnouncementDatabaseManager()
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
                    
                    # Get symbol and company name
                    symbol = ""
                    company_name = ""
                    
                    # Look for dedicated Symbol and Name columns
                    symbol_idx = -1
                    name_idx = -1
                    title_idx = -1
                    document_idx = -1
                    
                    # Identify column indices based on header
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
                    
                    # Extract data using identified indices
                    if 0 <= symbol_idx < len(cols):
                        symbol = cols[symbol_idx].text.strip()
                    if 0 <= name_idx < len(cols):
                        company_name = cols[name_idx].text.strip()
                    
                    # Extract title (subject)
                    title = cols[title_idx].text.strip() if 0 <= title_idx < len(cols) and cols[title_idx].text else ""
                    
                    # Extract PDF link
                    url = ''
                    if 0 <= document_idx < len(cols):
                        links = cols[document_idx].find_all('a')
                        for link in links:
                            href = link.get('href', '')
                            if not href.startswith('javascript:') and href.endswith('.pdf'):
                                url = href
                                if not url.startswith('http'):
                                    url = f"https://dps.psx.com.pk{url}"
                                break
                    
                    # Handle case where symbol is in title
                    if not symbol:
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
                        date_formats = ['%b %d, %Y', '%B %d, %Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
                        
                        for fmt in date_formats:
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if not date_obj:
                            # Parse "Mar 3, 2025" format
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
                        logging.warning(f"No sector information found for {symbol}")
                    
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
            excel_file = DATA_EXCEL_DIR / "PSX_Announcements.xlsx"
            
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
            if scraper.cache_file.exists():
                try:
                    scraper.cache_file.unlink()
                    logging.info(f"Deleted cache file: {scraper.cache_file}")
                except Exception as e:
                    logging.warning(f"Could not delete cache file: {e}")
        
        # Determine whether to scrape company pages
        scrape_company_pages = not args.skip_company_pages
        if args.skip_company_pages:
            logging.info("Skipping company-specific pages")
        
        # Scrape announcements
        logging.info("Starting announcement scraping")
        announcements = scraper.scrape_announcements(
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