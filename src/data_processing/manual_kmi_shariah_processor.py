import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import os
import logging
import traceback
from datetime import datetime
import json
import shutil
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Define important directories
DATA_LOGS_DIR = project_root / 'data' / 'logs'
DATA_EXPORTS_DIR = project_root / 'data' / 'exports'
PRODUCTION_DB_DIR = project_root / 'data' / 'databases' / 'production'
CONFIG_DIR = project_root / 'config'
SCRIPTS_DIR = project_root / 'src'

# Configure logging with proper path
logging.basicConfig(
    filename=DATA_LOGS_DIR / 'kmi_shariah_processor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Ensure directories exist
DATA_LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTION_DB_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

def ensure_dependencies():
    """Check and install required dependencies"""
    try:
        import importlib.util
        
        # List of packages to check/install
        packages = [
            'selenium',
            'cloudscraper',
            'bs4',
            'pandas',
            'requests'
        ]
        
        for package in packages:
            # Check if package is installed
            if importlib.util.find_spec(package) is None:
                logging.info(f"Installing missing package: {package}")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logging.info(f"Successfully installed {package}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to install dependencies: {e}")
        return False

def fetch_webpage(url, max_retries=3):
    """Fetch the webpage content with retry mechanism and improved headers"""
    import random
    import time
    
    # List of common User-Agent strings to rotate through
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36 Edg/103.0.1264.62'
    ]
    
    for attempt in range(max_retries):
        try:
            # Use a random User-Agent for each attempt
            user_agent = random.choice(user_agents)
            
            logging.info(f"Fetching data from {url} (Attempt {attempt+1}/{max_retries})")
            
            # Enhanced headers to mimic a browser request
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://sarmaaya.pk/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            }
            
            # Add a timeout to avoid hanging
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # If successful, return the content
            return response.text
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt+1} failed: {e}")
            
            if attempt < max_retries - 1:
                # Wait before retrying with exponential backoff
                wait_time = 2 ** attempt + random.uniform(0, 1)
                logging.info(f"Waiting {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to fetch webpage after {max_retries} attempts: {e}")
                return None
    
    return None

def fetch_webpage_cloudscraper(url):
    """Use cloudscraper to bypass Cloudflare protection"""
    try:
        import cloudscraper
        logging.info("Attempting to fetch data with cloudscraper")
        
        # Create a scraper instance
        scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        
        # Add additional headers
        scraper.headers.update({
            'Referer': 'https://sarmaaya.pk/',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'sec-ch-ua': '"Google Chrome";v="105", "Not)A;Brand";v="8", "Chromium";v="105"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
        })
        
        # Get the page
        response = scraper.get(url, timeout=30)
        
        if response.status_code == 200:
            logging.info("Successfully fetched data with cloudscraper")
            return response.text
        else:
            logging.error(f"Failed to fetch data with cloudscraper: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        logging.error(f"Error using cloudscraper: {e}")
        return None

def fetch_webpage_with_selenium(url):
    """Fetch the webpage using Selenium WebDriver with multiple browser options"""
    import time
    
    # Check if running in Docker
    def is_running_in_docker():
        try:
            with open('/proc/self/cgroup', 'r') as f:
                return any('docker' in line for line in f)
        except:
            # Check for .dockerenv file
            return os.path.exists('/.dockerenv')
    
    in_docker = is_running_in_docker()
    if in_docker:
        logging.info("Detected running inside Docker container")
    
    try:
        logging.info(f"Attempting to fetch data from {url} using Selenium")
        
        # OPTION 1: Try with Chrome (with explicit binary path)
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.common.exceptions import WebDriverException
            
            # Common Chrome installation paths (including Docker paths)
            chrome_paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
                "/opt/google/chrome/chrome",  # Common in Docker images
                "/headless-shell/headless-shell"  # Headless Chrome in some Docker images
            ]
            
            # Set up Chrome options with Docker-friendly settings
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")  # Required in Docker
            chrome_options.add_argument("--disable-dev-shm-usage")  # Required in Docker
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            # Disable images to save bandwidth
            chrome_options.add_argument("--blink-settings=imagesEnabled=false")
            
            # Try different paths
            found_chrome = False
            for path in chrome_paths:
                expanded_path = os.path.expandvars(path)
                if os.path.exists(expanded_path):
                    chrome_options.binary_location = expanded_path
                    found_chrome = True
                    break
            
            if found_chrome:
                logging.info(f"Using Chrome browser at: {chrome_options.binary_location}")
                try:
                    driver = webdriver.Chrome(options=chrome_options)
                    driver.get(url)
                    time.sleep(5)
                    html_content = driver.page_source
                    driver.quit()
                    return html_content
                except WebDriverException as e:
                    logging.warning(f"Chrome browser initialization failed: {e}")
                    raise
            else:
                logging.warning("Chrome browser not found, trying Firefox...")
                raise Exception("Chrome not found")
                
        except Exception as chrome_error:
            logging.warning(f"Chrome selenium attempt failed: {chrome_error}")
            
            # OPTION 2: Try with Firefox instead
            try:
                from selenium import webdriver
                from selenium.webdriver.firefox.options import Options as FirefoxOptions
                from selenium.common.exceptions import WebDriverException
                
                logging.info("Attempting with Firefox browser")
                firefox_options = FirefoxOptions()
                firefox_options.add_argument("--headless")
                firefox_options.add_argument("--no-sandbox")  # Required in Docker
                
                try:
                    driver = webdriver.Firefox(options=firefox_options)
                    driver.get(url)
                    time.sleep(5)
                    html_content = driver.page_source
                    driver.quit()
                    return html_content
                except WebDriverException as e:
                    logging.warning(f"Firefox browser initialization failed: {e}")
                    raise
                
            except Exception as firefox_error:
                logging.warning(f"Firefox selenium attempt failed: {firefox_error}")
                
                # In Docker, it's common that neither Chrome nor Firefox is available
                if in_docker:
                    logging.warning("Running in Docker and browsers failed. Trying alternative methods...")
                
                # OPTION 3: Try requests with proxy
                try:
                    logging.info("Attempting with requests using proxies")
                    # Get free proxies from a public list
                    proxy_response = requests.get('https://www.sslproxies.org/')
                    soup = BeautifulSoup(proxy_response.text, 'html.parser')
                    proxies_table = soup.find('table')
                    
                    if proxies_table:
                        proxies = []
                        for row in proxies_table.tbody.find_all('tr'):
                            cells = row.find_all('td')
                            if len(cells) >= 2:
                                proxy = f"http://{cells[0].text}:{cells[1].text}"
                                proxies.append(proxy)
                        
                        # Try different proxies
                        for proxy in proxies[:5]:  # Try first 5 proxies
                            try:
                                logging.info(f"Trying with proxy: {proxy}")
                                headers = {
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                                    'Accept-Language': 'en-US,en;q=0.5',
                                    'Referer': 'https://www.google.com/'
                                }
                                response = requests.get(url, headers=headers, proxies={'http': proxy, 'https': proxy}, timeout=10)
                                if response.status_code == 200:
                                    return response.text
                            except Exception:
                                continue
                
                except Exception as proxy_error:
                    logging.warning(f"Proxy request attempt failed: {proxy_error}")
                
                # OPTION 4: Try with cloudscraper
                try:
                    import cloudscraper
                    logging.info("Attempting with cloudscraper")
                    scraper = cloudscraper.create_scraper()
                    return scraper.get(url).text
                except Exception as cs_error:
                    logging.error(f"All browser and request methods failed: {cs_error}")
                    return None
    
    except Exception as e:
        logging.error(f"Failed to fetch webpage with any method: {e}")
        logging.error(traceback.format_exc())
        return None

def parse_kmi_shariah_data(html_content):
    """Parse the HTML to extract KMIALLSHR stocks with specific columns"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    try:
        logging.info("Parsing KMIALLSHR stocks data")
        
        # Find all tables on the page
        tables = soup.find_all('table', {'class': 'table'})
        
        # The table we want should have the specific headers we're looking for
        target_table = None
        for table in tables:
            headers = table.find_all('th')
            header_texts = [h.text.strip() for h in headers if h.text.strip()]
            
            # Check if this looks like our target table
            if any('Symbol' in h for h in header_texts) and any('Points' in h for h in header_texts):
                target_table = table
                break
        
        if not target_table:
            logging.warning("Could not find the table with KMIALLSHR data")
            return None
        
        # Extract data from table rows
        rows = target_table.find_all('tr')
        if len(rows) <= 1:  # Only header row or no rows
            logging.warning("Table has no data rows")
            return None
            
        # Identify column indices
        header_row = rows[0]
        headers = [th.text.strip() for th in header_row.find_all('th')]
        logging.info(f"Found table headers: {headers}")
        
        # Find indices for the columns we need
        symbol_idx = next((i for i, h in enumerate(headers) if 'Symbol' in h), None)
        points_idx = next((i for i, h in enumerate(headers) if 'Points' in h), None)
        weight_idx = next((i for i, h in enumerate(headers) if 'Weight' in h), None)
        current_idx = next((i for i, h in enumerate(headers) if 'Cur.' in h), None)
        change_idx = next((i for i, h in enumerate(headers) if 'Chg.' in h), None)
        change_pct_idx = next((i for i, h in enumerate(headers) if 'Chg.%' in h), None)
        high52_idx = next((i for i, h in enumerate(headers) if '52WK High' in h), None)
        low52_idx = next((i for i, h in enumerate(headers) if '52WK Low' in h), None)
        volume_idx = next((i for i, h in enumerate(headers) if 'Vol.' in h), None)
        market_cap_idx = next((i for i, h in enumerate(headers) if 'Market Cap' in h), None)
        
        if None in (symbol_idx, points_idx, current_idx):
            logging.error(f"Could not find all required columns in headers: {headers}")
            return None
            
        # Parse data rows
        data = []
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        for row in rows[1:]:  # Skip header row
            columns = row.find_all('td')
            if len(columns) > max(symbol_idx, points_idx, current_idx):
                # Extract symbol
                symbol = columns[symbol_idx].text.strip()
                
                # Extract points
                points_text = columns[points_idx].text.strip().replace(',', '')
                points = float(points_text) if points_text.replace('.', '').isdigit() else 0.0
                
                # Extract weight (percentage value)
                weight = 0.0
                if weight_idx is not None and len(columns) > weight_idx:
                    weight_text = columns[weight_idx].text.strip()
                    # Remove commas and percentage signs, then convert to float
                    weight_text = weight_text.replace(',', '').replace('%', '')
                    try:
                        weight = float(weight_text)
                    except ValueError:
                        logging.warning(f"Could not parse weight value: {columns[weight_idx].text.strip()}")
                        weight = 0.0
                
                # Extract current price
                current_text = columns[current_idx].text.strip().replace(',', '')
                current = float(current_text) if current_text.replace('.', '').isdigit() else 0.0
                
                # Extract change
                change = 0.0
                if change_idx is not None and len(columns) > change_idx:
                    change_text = columns[change_idx].text.strip().replace(',', '')
                    change = float(change_text) if change_text.replace('.', '').isdigit() else 0.0
                
                # Extract change percentage
                change_pct = 0.0
                if change_pct_idx is not None and len(columns) > change_pct_idx:
                    change_pct_text = columns[change_pct_idx].text.strip().replace('%', '').replace(',', '')
                    change_pct = float(change_pct_text) if change_pct_text.replace('.', '').isdigit() else 0.0
                
                # Extract 52-week high
                high52 = 0.0
                if high52_idx is not None and len(columns) > high52_idx:
                    high52_text = columns[high52_idx].text.strip().replace(',', '')
                    high52 = float(high52_text) if high52_text.replace('.', '').isdigit() else 0.0
                
                # Extract 52-week low
                low52 = 0.0
                if low52_idx is not None and len(columns) > low52_idx:
                    low52_text = columns[low52_idx].text.strip().replace(',', '')
                    low52 = float(low52_text) if low52_text.replace('.', '').isdigit() else 0.0
                
                # Extract volume
                volume = 0
                if volume_idx is not None and len(columns) > volume_idx:
                    volume_text = columns[volume_idx].text.strip().replace(',', '')
                    volume = int(volume_text) if volume_text.isdigit() else 0
                
                # Extract market cap (in thousands) with enhanced validation
                market_cap = 0
                if market_cap_idx is not None and len(columns) > market_cap_idx:
                    market_cap_text = columns[market_cap_idx].text.strip()
                    try:
                        # Remove commas, spaces and any non-numeric characters except B/M
                        clean_text = market_cap_text.replace(',', '').replace(' ', '')
                        
                        # Handle billions (B) and millions (M) suffixes
                        if 'B' in clean_text:
                            value = float(clean_text.replace('B', ''))
                            market_cap = int(value * 1000000)  # Convert billions to thousands
                            logging.debug(f"Parsed market cap (B): {market_cap_text} → {market_cap}")
                        elif 'M' in clean_text:
                            value = float(clean_text.replace('M', ''))
                            market_cap = int(value * 1000)  # Convert millions to thousands
                            logging.debug(f"Parsed market cap (M): {market_cap_text} → {market_cap}")
                        else:
                            # Plain numeric value (already in thousands)
                            market_cap = int(float(clean_text))
                            logging.debug(f"Parsed market cap: {market_cap_text} → {market_cap}")
                            
                    except (ValueError, AttributeError) as e:
                        logging.warning(f"Failed to parse market cap value: {market_cap_text} - {e}")
                        market_cap = 0
                
                record = {
                    'symbol': symbol,
                    'points': points,
                    'weight': weight,
                    'current_price': current,
                    'change': change,
                    'change_percent': change_pct,
                    'high_52_week': high52,
                    'low_52_week': low52,
                    'volume': volume,
                    'market_cap_000': market_cap,
                    'date_added': current_date
                }
                
                data.append(record)
        
        logging.info(f"Found {len(data)} KMIALLSHR stocks")
        return data
        
    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")
        logging.error(traceback.format_exc())
def parse_allshr_data(html_content):
    """Parse the HTML to extract ALLSHR stocks with specific columns"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    try:
        logging.info("Parsing ALLSHR stocks data")
        
        # Find all tables on the page
        tables = soup.find_all('table', {'class': 'table'})
        
        # The table we want should have the specific headers we're looking for
        target_table = None
        for table in tables:
            headers = table.find_all('th')
            header_texts = [h.text.strip() for h in headers if h.text.strip()]
            
            # Check if this looks like our target table
            if any('Symbol' in h for h in header_texts) and any('Points' in h for h in header_texts):
                target_table = table
                break
        
        if not target_table:
            logging.warning("Could not find the table with ALLSHR data")
            return None
        
        # Extract data from table rows
        rows = target_table.find_all('tr')
        if len(rows) <= 1:  # Only header row or no rows
            logging.warning("Table has no data rows")
            return None
            
        # Identify column indices
        header_row = rows[0]
        headers = [th.text.strip() for th in header_row.find_all('th')]
        logging.info(f"Found table headers: {headers}")
        
        # Find indices for the columns we need
        symbol_idx = next((i for i, h in enumerate(headers) if 'Symbol' in h), None)
        points_idx = next((i for i, h in enumerate(headers) if 'Points' in h), None)
        weight_idx = next((i for i, h in enumerate(headers) if 'Weight' in h), None)
        current_idx = next((i for i, h in enumerate(headers) if 'Cur.' in h), None)
        change_idx = next((i for i, h in enumerate(headers) if 'Chg.' in h), None)
        change_pct_idx = next((i for i, h in enumerate(headers) if 'Chg.%' in h), None)
        high52_idx = next((i for i, h in enumerate(headers) if '52WK High' in h), None)
        low52_idx = next((i for i, h in enumerate(headers) if '52WK Low' in h), None)
        volume_idx = next((i for i, h in enumerate(headers) if 'Vol.' in h), None)
        market_cap_idx = next((i for i, h in enumerate(headers) if 'Market Cap' in h), None)
        
        if None in (symbol_idx, points_idx, current_idx):
            logging.error(f"Could not find all required columns in headers: {headers}")
            return None
            
        # Parse data rows
        data = []
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        for row in rows[1:]:  # Skip header row
            columns = row.find_all('td')
            if len(columns) > max(symbol_idx, points_idx, current_idx):
                # Extract symbol
                symbol = columns[symbol_idx].text.strip()
                
                # Extract points
                points_text = columns[points_idx].text.strip().replace(',', '')
                points = float(points_text) if points_text.replace('.', '').isdigit() else 0.0
                
                # Extract weight (percentage value)
                weight = 0.0
                if weight_idx is not None and len(columns) > weight_idx:
                    weight_text = columns[weight_idx].text.strip()
                    # Remove commas and percentage signs, then convert to float
                    weight_text = weight_text.replace(',', '').replace('%', '')
                    try:
                        weight = float(weight_text)
                    except ValueError:
                        logging.warning(f"Could not parse weight value: {columns[weight_idx].text.strip()}")
                        weight = 0.0
                
                # Extract current price
                current_text = columns[current_idx].text.strip().replace(',', '')
                current = float(current_text) if current_text.replace('.', '').isdigit() else 0.0
                
                # Extract change
                change = 0.0
                if change_idx is not None and len(columns) > change_idx:
                    change_text = columns[change_idx].text.strip().replace(',', '')
                    change = float(change_text) if change_text.replace('.', '').isdigit() else 0.0
                
                # Extract change percentage
                change_pct = 0.0
                if change_pct_idx is not None and len(columns) > change_pct_idx:
                    change_pct_text = columns[change_pct_idx].text.strip().replace('%', '').replace(',', '')
                    change_pct = float(change_pct_text) if change_pct_text.replace('.', '').isdigit() else 0.0
                
                # Extract 52-week high
                high52 = 0.0
                if high52_idx is not None and len(columns) > high52_idx:
                    high52_text = columns[high52_idx].text.strip().replace(',', '')
                    high52 = float(high52_text) if high52_text.replace('.', '').isdigit() else 0.0
                
                # Extract 52-week low
                low52 = 0.0
                if low52_idx is not None and len(columns) > low52_idx:
                    low52_text = columns[low52_idx].text.strip().replace(',', '')
                    low52 = float(low52_text) if low52_text.replace('.', '').isdigit() else 0.0
                
                # Extract volume
                volume = 0
                if volume_idx is not None and len(columns) > volume_idx:
                    volume_text = columns[volume_idx].text.strip().replace(',', '')
                    volume = int(volume_text) if volume_text.isdigit() else 0
                
                # Extract market cap (in thousands) with enhanced validation
                market_cap = 0
                if market_cap_idx is not None and len(columns) > market_cap_idx:
                    market_cap_text = columns[market_cap_idx].text.strip()
                    try:
                        # Remove commas, spaces and any non-numeric characters except B/M
                        clean_text = market_cap_text.replace(',', '').replace(' ', '')
                        
                        # Handle billions (B) and millions (M) suffixes
                        if 'B' in clean_text:
                            value = float(clean_text.replace('B', ''))
                            market_cap = int(value * 1000000)  # Convert billions to thousands
                            logging.debug(f"Parsed market cap (B): {market_cap_text} → {market_cap}")
                        elif 'M' in clean_text:
                            value = float(clean_text.replace('M', ''))
                            market_cap = int(value * 1000)  # Convert millions to thousands
                            logging.debug(f"Parsed market cap (M): {market_cap_text} → {market_cap}")
                        else:
                            # Plain numeric value (already in thousands)
                            market_cap = int(float(clean_text))
                            logging.debug(f"Parsed market cap: {market_cap_text} → {market_cap}")
                            
                    except (ValueError, AttributeError) as e:
                        logging.warning(f"Failed to parse market cap value: {market_cap_text} - {e}")
                        market_cap = 0
                
                record = {
                    'symbol': symbol,
                    'points': points,
                    'weight': weight,
                    'current_price': current,
                    'change': change,
                    'change_percent': change_pct,
                    'high_52_week': high52,
                    'low_52_week': low52,
                    'volume': volume,
                    'market_cap_000': market_cap,
                    'date_added': current_date
                }
                
                data.append(record)
        
        logging.info(f"Found {len(data)} ALLSHR stocks")
        return data
        
    except Exception as e:
        logging.error(f"Error parsing HTML for ALLSHR: {e}")
        logging.error(traceback.format_exc())
        return None
        return None

def save_to_database(data, db_path=None, table_name='KMIALLSHR'):
    """Save the extracted data to SQLite database with data retention for specified table
    Args:
        data: List of dicts containing stock data with expected columns:
            - KMIALLSHR: symbol, points, weight, current_price, change, change_percent,
                        high_52_week, low_52_week, volume, market_cap_000,
                        date_added, update_date, rank
            - ALLSHR: Same as KMIALLSHR
            - KSE100: Same as KMIALLSHR
            - KMI30: Same as KMIALLSHR
            - KMI100: Same as KMIALLSHR (must have exactly 100 records)
        db_path: Path to SQLite database file
        table_name: Name of table to save to (KMIALLSHR, ALLSHR, KSE100, KMI30, KMI100)
    Returns:
        bool: True if successful, False if failed
    Raises:
        ValueError: If input data doesn't match expected schema
    """
    if db_path is None:
        db_path = Path("/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSXSymbols.db")
    
    try:
        # Validate input data schema
        required_columns = {
            'symbol': str,
            'points': (int, float),
            'weight': (int, float),
            'current_price': (int, float),
            'change': (int, float),
            'change_percent': (int, float),
            'high_52_week': (int, float),
            'low_52_week': (int, float),
            'volume': int,
            'market_cap_000': int,
            'date_added': str,
            'update_date': str,
            'rank': (int, type(None))
        }

        # Special validation for KMI100
        if table_name == 'KMI100' and len(data) != 100:
            error_msg = f"KMI100 must have exactly 100 records, got {len(data)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Validate each record
        for record in data:
            for col, col_type in required_columns.items():
                if col not in record:
                    error_msg = f"Missing required column '{col}' in record for {table_name}"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                if not isinstance(record[col], col_type):
                    error_msg = f"Invalid type for column '{col}' in {table_name}. Expected {col_type}, got {type(record[col])}"
                    logging.error(error_msg)
                    raise ValueError(error_msg)

        # Ensure the database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  - Database path: {db_path}")
        print(f"  - Database exists: {os.path.exists(db_path)}")
        
        if not os.path.exists(db_path):
            print("  - Creating new database file")
            open(db_path, 'a').close()
        
        logging.info(f"Saving {len(data)} records to {table_name} table in database: {db_path}")
        
        # Add update timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for record in data:
            record['update_date'] = timestamp
        
        # Convert to DataFrame for easier DB operations
        df = pd.DataFrame(data)
        print(f"  - DataFrame columns: {df.columns.tolist()}")
        
        # Connect to SQLite database with transaction
        with sqlite3.connect(db_path) as conn:
            # Enable foreign key constraints and WAL mode for better concurrency
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            
            # Create backup table if it doesn't exist
            conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name}_backup (
                symbol TEXT,
                points REAL,
                weight REAL,
                current_price REAL,
                change REAL,
                change_percent REAL,
                high_52_week REAL,
                low_52_week REAL,
                volume INTEGER,
                market_cap_000 INTEGER,
                date_added TEXT,
                update_date TEXT,
                rank INTEGER,
                backup_date TEXT,
                PRIMARY KEY (symbol, date_added, backup_date)
            )
            ''')
            
            # Create main table if it doesn't exist
            conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                symbol TEXT,
                points REAL,
                weight REAL,
                current_price REAL,
                change REAL,
                change_percent REAL,
                high_52_week REAL,
                low_52_week REAL,
                volume INTEGER,
                market_cap_000 INTEGER,
                date_added TEXT,
                update_date TEXT,
                rank INTEGER,
                PRIMARY KEY (symbol, date_added)
            )
            ''')
            
            # Check for missing columns and add them if needed
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            # List of all possible columns
            all_columns = [
                'symbol', 'points', 'weight', 'current_price', 'change',
                'change_percent', 'high_52_week', 'low_52_week', 'volume',
                'market_cap_000', 'date_added', 'update_date', 'rank'
            ]
            
            # Add any missing columns with error handling
            for col in all_columns:
                if col not in existing_columns:
                    try:
                        print(f"  - Adding missing column: {col}")
                        if col == 'rank':
                            col_type = 'INTEGER'
                        elif col in ['volume', 'market_cap_000']:
                            col_type = 'INTEGER'
                        elif col in ['points', 'weight', 'current_price', 'change',
                                    'change_percent', 'high_52_week', 'low_52_week']:
                            col_type = 'REAL'
                        else:
                            col_type = 'TEXT'
                        
                        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
                        print(f"    - Successfully added column {col} with type {col_type}")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" in str(e):
                            print(f"    - Column {col} already exists (skipping)")
                            continue
                        else:
                            print(f"    ❌ Error adding column {col}: {str(e)}")
                            logging.error(f"Failed to add column {col}: {str(e)}")
            
            # Begin transaction
            with conn:
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Backup existing data for today before deleting
                conn.execute(f'''
                INSERT INTO {table_name}_backup
                SELECT
                    symbol, points, weight, current_price, change,
                    change_percent, high_52_week, low_52_week, volume,
                    market_cap_000, date_added, update_date, rank, ?
                FROM {table_name}
                WHERE date_added = ?
                ''', (timestamp, today))
                
                # Delete existing records for today
                conn.execute(f"DELETE FROM {table_name} WHERE date_added = ?", (today,))
                
                # Insert new data
                df.to_sql(table_name, conn, if_exists='append', index=False)
                
                # Verify data was inserted correctly
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE date_added = ?",
                                   (today,)).fetchone()[0]
                if count != len(data):
                    raise ValueError(f"Data count mismatch: expected {len(data)}, got {count}")
            
        logging.info(f"Successfully saved {len(data)} records to {table_name} table")
        return True
    except Exception as e:
        error_msg = f"Failed to save data to {table_name}: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        print(f"  ❌ Database error: {error_msg}")
        print(f"  ❌ Check log file for details: {DATA_LOGS_DIR / 'kmi_shariah_processor.log'}")
        
        # Log schema mismatch details if available
        if "columns but" in str(e) and "values were supplied" in str(e):
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                cursor.execute(f"PRAGMA table_info({table_name}_backup)")
                backup_columns = [col[1] for col in cursor.fetchall()]
                logging.error(f"Schema mismatch - {table_name} columns: {columns}")
                logging.error(f"Schema mismatch - {table_name}_backup columns: {backup_columns}")
                logging.error(f"Data columns: {list(data[0].keys()) if data else 'No data'}")
        
        return False

def export_to_csv(data, filename=None):
    """Export the data to CSV file"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"kmi_data_{timestamp}.csv"
    
    csv_path = DATA_EXPORTS_DIR / filename
    
    try:
        # Ensure the exports directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logging.info(f"Data exported to CSV: {csv_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to export data to CSV: {e}")
        logging.error(traceback.format_exc())
        return False

def export_to_json(data, filename=None):
    """Export the data to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"kmi_data_{timestamp}.json"
    
    json_path = DATA_EXPORTS_DIR / filename
    
    try:
        # Ensure the exports directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export as JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Data exported to JSON: {json_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to export data to JSON: {e}")
        logging.error(traceback.format_exc())
        return False

def export_to_excel(data, filename=None, sheet_name='KMI100'):
    """Export the data to Excel file, preserving other sheets if the file exists"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"kmi_data_{timestamp}.xlsx"
    
    excel_path = DATA_EXPORTS_DIR / filename
    
    try:
        # Ensure the exports directory exists
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Add rank column
        df['Rank'] = range(1, len(df) + 1)
        
        print(f"\n📊 Excel Export Process:")
        print(f"  - Target file: {excel_path}")
        print(f"  - Target sheet: {sheet_name}")
        print(f"  - Data rows: {len(df)}")
        
        # Check if file exists and has other sheets
        if excel_path.exists():
            print(f"  - File exists: Yes")
            # Load existing Excel file
            with pd.ExcelFile(excel_path) as xls:
                # Get existing sheets
                existing_sheets = xls.sheet_names
                print(f"  - Existing sheets: {', '.join(existing_sheets)}")
                
                # Create a dictionary to store all sheets
                all_sheets = {}
                
                # Read all existing sheets
                for sheet in existing_sheets:
                    print(f"  - Reading sheet: {sheet}")
                    all_sheets[sheet] = pd.read_excel(xls, sheet_name=sheet)
                
                # Update or add the KMI100 sheet
                print(f"  - Updating sheet: {sheet_name} with {len(df)} rows")
                all_sheets[sheet_name] = df
                
                # Write all sheets back to the Excel file
                print(f"  - Writing all sheets back to file")
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    for sheet_name, sheet_df in all_sheets.items():
                        # Special handling for KMI100 sheet
                        if sheet_name == 'KMI100':
                            # Validate we have exactly 100 rows
                            if len(sheet_df) != 100:
                                error_msg = f"Invalid row count for KMI100: expected 100, got {len(sheet_df)}"
                                print(f"  ❌ {error_msg}")
                                logging.error(error_msg)
                                # Create new DataFrame with exactly 100 rows
                                sheet_df = sheet_df.head(100)
                                print("  - Truncated to 100 rows")
                                
                            # Clear existing data before writing
                            print(f"  - Clearing existing KMI100 data")
                            logging.info("Clearing KMI100 sheet before writing new data")
                            
                        sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    print(f"  - Successfully updated {sheet_name} sheet while preserving {len(existing_sheets)} other sheets")
                    logging.info(f"Updated {sheet_name} sheet with {len(sheet_df)} rows")
                
                # Verify the KMI100 sheet after writing
                with pd.ExcelFile(excel_path) as xls:
                    if sheet_name in xls.sheet_names:
                        verify_df = pd.read_excel(xls, sheet_name=sheet_name)
                        print(f"  - Verification: KMI100 sheet now contains {len(verify_df)} rows")
                    else:
                        print(f"  - Warning: KMI100 sheet not found after writing")
        else:
            print(f"  - File exists: No (creating new file)")
            # Create new Excel file with just the KMI100 sheet
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"  - Successfully created new file with {sheet_name} sheet")
            if sheet_name == 'KMI100':
                if len(df) != 100:
                    print(f"  ❌ Invalid row count for KMI100: expected 100, got {len(df)}")
                    logging.error(f"Invalid KMI100 row count: expected 100, got {len(df)}")
                else:
                    logging.info(f"Created new KMI100 sheet with {len(df)} rows")
            
            # Verify the KMI100 sheet after writing
            with pd.ExcelFile(excel_path) as xls:
                if sheet_name in xls.sheet_names:
                    verify_df = pd.read_excel(xls, sheet_name=sheet_name)
                    print(f"  - Verification: KMI100 sheet contains {len(verify_df)} rows")
                else:
                    print(f"  - Warning: KMI100 sheet not found after writing")
            
        logging.info(f"Data exported to Excel: {excel_path} (sheet: {sheet_name})")
        return True
    except Exception as e:
        logging.error(f"Failed to export to Excel: {e}")
        logging.error(traceback.format_exc())
        print(f"  ❌ Error exporting to Excel: {e}")
        return False

def fetch_market_data_from_api():
    """Fetch market data using an API approach"""
    try:
        logging.info("Fetching market data from API")
        # Placeholder for API implementation
        # Replace with actual API call and data parsing logic
        return []
    except Exception as e:
        logging.error(f"Failed to fetch market data from API: {e}")
        return None

def copy_missing_data_to_kmiall(db_path=None):
    """Copy missing data from KMIALLSHR to KMIALL table"""
    if db_path is None:
        db_path = PRODUCTION_DB_DIR / "PSXSymbols.db"  # Changed from PSX_KMI_data.db to PSXSymbols.db
    
    try:
        print("\n🔄 Copying missing data from KMIALLSHR to KMIALL...")
        logging.info("Copying missing data from KMIALLSHR to KMIALL")
        
        with sqlite3.connect(db_path) as conn:
            # Get all unique dates from KMIALLSHR
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT date_added FROM KMIALLSHR ORDER BY date_added")
            dates = cursor.fetchall()
            
            if not dates:
                print("  - No dates found in KMIALLSHR table")
                logging.warning("No dates found in KMIALLSHR table")
                return False
            
            print(f"  - Found {len(dates)} unique dates in KMIALLSHR")
            
            for date in dates:
                date_str = date[0]
                print(f"  - Processing date: {date_str}")
                
                # Check if data exists in KMIALL for this date
                cursor.execute("SELECT COUNT(*) FROM KMIALL WHERE date_added = ?", (date_str,))
                kmiall_count = cursor.fetchone()[0]
                
                if kmiall_count == 0:
                    print(f"    - No data found in KMIALL for {date_str}, copying from KMIALLSHR")
                    
                    # Get data from KMIALLSHR
                    cursor.execute("""
                        SELECT symbol, points, weight, current_price, change, change_percent,
                               high_52_week, low_52_week, volume, market_cap_000, date_added, update_date, rank
                        FROM KMIALLSHR 
                        WHERE date_added = ?
                    """, (date_str,))
                    data = cursor.fetchall()
                    
                    if data:
                        # Insert into KMIALL
                        cursor.executemany("""
                            INSERT INTO KMIALL (
                                symbol, points, weight, current_price, change, change_percent,
                                high_52_week, low_52_week, volume, market_cap_000, date_added, update_date, rank
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, data)
                        print(f"    - Copied {len(data)} records to KMIALL")
                        logging.info(f"Copied {len(data)} records to KMIALL for date {date_str}")
                    else:
                        print(f"    - No data found in KMIALLSHR for {date_str}")
                        logging.warning(f"No data found in KMIALLSHR for date {date_str}")
                else:
                    print(f"    - Data already exists in KMIALL for {date_str}")
            
            conn.commit()
            print("✅ Completed copying missing data")
            logging.info("Successfully completed copying missing data")
            return True
            
    except Exception as e:
        logging.error(f"Failed to copy missing data: {e}")
        logging.error(traceback.format_exc())
        print(f"❌ Error copying data: {str(e)}")
        return False

def update_kmiall_from_excel(db_path=None):
    """Update KMIALL table with only new scripts from Excel file, preserving existing data"""
    if db_path is None:
        db_path = PRODUCTION_DB_DIR / "PSXSymbols.db"
    
    try:
        print("\n🔄 Updating KMIALL table with new scripts from Excel...")
        logging.info("Updating KMIALL table with new scripts from Excel")
        
        # Read the Excel file
        excel_path = CONFIG_DIR / 'psxsymbols.xlsx'
        if not excel_path.exists():
            print(f"  ❌ Excel file not found: {excel_path}")
            logging.error(f"Excel file not found: {excel_path}")
            return False
            
        print(f"  - Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path, sheet_name='KMIALL')
        
        if df.empty:
            print("  ❌ No data found in KMIALL sheet")
            logging.error("No data found in KMIALL sheet")
            return False
            
        print(f"  - Found {len(df)} symbols in KMIALL sheet")
        
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(db_path) as conn:
            # Create KMIALL table if it doesn't exist
            conn.execute('''
            CREATE TABLE IF NOT EXISTS KMIALL (
                symbol TEXT,
                points REAL,
                weight REAL,
                current_price REAL,
                change REAL,
                change_percent REAL,
                high_52_week REAL,
                low_52_week REAL,
                volume INTEGER,
                market_cap_000 INTEGER,
                date_added TEXT,
                update_date TEXT,
                rank INTEGER,
                PRIMARY KEY (symbol, date_added)
            )
            ''')
            
            # Get existing symbols for today
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM KMIALL WHERE date_added = ?", (today,))
            existing_symbols = {row[0] for row in cursor.fetchall()}
            
            # Filter out existing symbols
            new_symbols = set(df['symbol'].unique()) - existing_symbols
            new_data = df[df['symbol'].isin(new_symbols)]
            
            if len(new_data) == 0:
                print("  - No new symbols to add")
                logging.info("No new symbols to add to KMIALL table")
                return True
                
            print(f"  - Found {len(new_data)} new symbols to add")
            
            # Add required columns if they don't exist
            if 'rank' not in new_data.columns:
                new_data['rank'] = None
            if 'date_added' not in new_data.columns:
                new_data['date_added'] = today
            if 'update_date' not in new_data.columns:
                new_data['update_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Ensure all required columns exist with NULL for missing values
            required_columns = [
                'symbol', 'points', 'weight', 'current_price', 'change', 
                'change_percent', 'high_52_week', 'low_52_week', 'volume', 
                'market_cap_000', 'date_added', 'update_date', 'rank'
            ]
            
            # Add missing columns with NULL values
            for col in required_columns:
                if col not in new_data.columns:
                    new_data[col] = None
            
            # Reorder columns to match database schema
            new_data = new_data[required_columns]
            
            # Insert new data
            new_data.to_sql('KMIALL', conn, if_exists='append', index=False)
            
            # Verify data was inserted correctly
            count = conn.execute("SELECT COUNT(*) FROM KMIALL WHERE date_added = ? AND symbol IN ({})".format(
                ','.join(['?'] * len(new_symbols))), [today] + list(new_symbols)).fetchone()[0]
            
            if count != len(new_data):
                raise ValueError(f"Data count mismatch: expected {len(new_data)}, got {count}")
            
            print(f"  ✅ Successfully added {len(new_data)} new symbols to KMIALL table")
            logging.info(f"Successfully added {len(new_data)} new symbols to KMIALL table")
            return True
            
    except Exception as e:
        logging.error(f"Failed to update KMIALL table: {e}")
        logging.error(traceback.format_exc())
        print(f"❌ Error updating KMIALL table: {str(e)}")
        return False

def main():
    """Main function to run the scraper"""
    print("\n🚀 Starting KMI Shariah Data Processor")
    print("=====================================")
    
    # Install required dependencies
    print("\n📦 Checking dependencies...")
    ensure_dependencies()
    
    # URL of the page to scrape
    kmi_url = "https://sarmaaya.pk/psx/market/KMIALLSHR"
    allshr_url = "https://sarmaaya.pk/psx/market/ALLSHR"
    kse100_url = "https://sarmaaya.pk/psx/market/KSE100"
    kmi30_url = "https://sarmaaya.pk/psx/market/KMI30"
    print(f"\n🌐 Target URLs:")
    print(f"  - KMIALLSHR: {kmi_url}")
    print(f"  - ALLSHR: {allshr_url}")
    print(f"  - KSE100: {kse100_url}")
    print(f"  - KMI30: {kmi30_url}")
    
    # Paths for output - using path constants for consistency
    db_path = Path("/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/data/databases/production/PSXSymbols.db")
    
    excel_path = CONFIG_DIR / 'psxsymbols.xlsx'  # Use the correct path to PSXSymbols.xlsx
    
    csv_path = DATA_EXPORTS_DIR / 'reports' / 'KMIALLSHR.csv'
    
    print(f"\n📂 Output paths:")
    print(f"  - Database: {db_path}")
    print(f"  - Excel: {excel_path}")
    print(f"  - CSV: {csv_path}")
    
    # Make sure the database directory exists
    os.makedirs(db_path.parent, exist_ok=True)
    # Make sure the reports directory exists
    os.makedirs(csv_path.parent, exist_ok=True)
    
    # Check if the Excel file exists and verify the KMI100 sheet
    if excel_path.exists():
        try:
            with pd.ExcelFile(excel_path) as xls:
                if 'KMI100' in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name='KMI100')
                    print(f"\n📊 Current KMI100 sheet status:")
                    print(f"  - Total rows: {len(df)}")
                    print(f"  - Columns: {', '.join(df.columns.tolist())}")
                else:
                    print(f"\n⚠️ KMI100 sheet not found in {excel_path}")
        except Exception as e:
            print(f"\n⚠️ Error reading Excel file: {e}")
    # copy the psxsymbols.xlsx file to the production directory
    shutil.copy(excel_path, PRODUCTION_DB_DIR / 'psxsymbols.xlsx')
    
    # copy the psxsymbols.xlsx file to the src/data_processing
    shutil.copy(excel_path, SCRIPTS_DIR / 'data_processing/psxsymbols.xlsx')
    kmi_html_content = None
    allshr_html_content = None
    kse100_html_content = None
    kmi30_html_content = None
    
    print("\n🔍 Fetching webpage content...")
    # Method 1: Standard requests
    print("  - Trying standard requests for KMIALLSHR...")
    kmi_html_content = fetch_webpage(kmi_url)
    print("  - Trying standard requests for ALLSHR...")
    allshr_html_content = fetch_webpage(allshr_url)
    print("  - Trying standard requests for KSE100...")
    kse100_html_content = fetch_webpage(kse100_url)
    print("  - Trying standard requests for KMI30...")
    kmi30_html_content = fetch_webpage(kmi30_url)
    
    # Method 2: Cloudscraper
    if not kmi_html_content:
        print("  - Standard requests failed for KMIALLSHR, trying cloudscraper...")
        kmi_html_content = fetch_webpage_cloudscraper(kmi_url)
    if not allshr_html_content:
        print("  - Standard requests failed for ALLSHR, trying cloudscraper...")
        allshr_html_content = fetch_webpage_cloudscraper(allshr_url)
    if not kse100_html_content:
        print("  - Standard requests failed for KSE100, trying cloudscraper...")
        kse100_html_content = fetch_webpage_cloudscraper(kse100_url)
    if not kmi30_html_content:
        print("  - Standard requests failed for KMI30, trying cloudscraper...")
        kmi30_html_content = fetch_webpage_cloudscraper(kmi30_url)
    
    # Method 3: Selenium with fallbacks
    if not kmi_html_content:
        print("  - Cloudscraper failed for KMIALLSHR, trying Selenium...")
        kmi_html_content = fetch_webpage_with_selenium(kmi_url)
    if not allshr_html_content:
        print("  - Cloudscraper failed for ALLSHR, trying Selenium...")
        allshr_html_content = fetch_webpage_with_selenium(allshr_url)
    if not kse100_html_content:
        print("  - Cloudscraper failed for KSE100, trying Selenium...")
        kse100_html_content = fetch_webpage_with_selenium(kse100_url)
    if not kmi30_html_content:
        print("  - Cloudscraper failed for KMI30, trying Selenium...")
        kmi30_html_content = fetch_webpage_with_selenium(kmi30_url)
    
    if not kmi_html_content:
        print("  ❌ All methods failed to fetch KMIALLSHR webpage content")
        logging.error("Failed to fetch KMIALLSHR webpage content by any method.")
        return False
    if not allshr_html_content:
        print("  ❌ All methods failed to fetch ALLSHR webpage content")
        logging.error("Failed to fetch ALLSHR webpage content by any method.")
        return False
    if not kse100_html_content:
        print("  ❌ All methods failed to fetch KSE100 webpage content")
        logging.error("Failed to fetch KSE100 webpage content by any method.")
        return False
    if not kmi30_html_content:
        print("  ❌ All methods failed to fetch KMI30 webpage content")
        logging.error("Failed to fetch KMI30 webpage content by any method.")
        return False
    
    print("  ✅ Successfully fetched KMIALLSHR webpage content")
    print("  ✅ Successfully fetched ALLSHR webpage content")
    
    # Parse the HTML to extract data
    print("\n🔍 Parsing KMIALLSHR data...")
    kmi_data = parse_kmi_shariah_data(kmi_html_content)
    if not kmi_data or len(kmi_data) == 0:
        print("  ❌ Could not parse KMIALLSHR data")
        logging.error("Could not parse KMIALLSHR data. Exiting.")
        return False
    
    print("\n🔍 Parsing ALLSHR data...")
    allshr_data = parse_allshr_data(allshr_html_content)
    if not allshr_data or len(allshr_data) == 0:
        print("  ❌ Could not parse ALLSHR data")
        logging.error("Could not parse ALLSHR data. Exiting.")
        return False

    print("\n🔍 Parsing KSE100 data...")
    kse100_data = parse_kmi_shariah_data(kse100_html_content)
    if not kse100_data or len(kse100_data) == 0:
        print("  ❌ Could not parse KSE100 data")
        logging.error("Could not parse KSE100 data. Exiting.")
        return False

    print("\n🔍 Parsing KMI30 data...")
    kmi30_data = parse_kmi_shariah_data(kmi30_html_content)
    if not kmi30_data or len(kmi30_data) == 0:
        print("  ❌ Could not parse KMI30 data")
        logging.error("Could not parse KMI30 data. Exiting.")
        return False
    
    print(f"  ✅ Successfully parsed {len(kmi_data)} KMIALLSHR stocks")
    print(f"  ✅ Successfully parsed {len(allshr_data)} ALLSHR stocks")
    print(f"  ✅ Successfully parsed {len(kse100_data)} KSE100 stocks")
    print(f"  ✅ Successfully parsed {len(kmi30_data)} KMI30 stocks")
        
    # Sort data by market cap in descending order
    kmi_data = sorted(kmi_data, key=lambda x: x['market_cap_000'], reverse=True)
    allshr_data = sorted(allshr_data, key=lambda x: x['market_cap_000'], reverse=True)
    kse100_data = sorted(kse100_data, key=lambda x: x['market_cap_000'], reverse=True)
    kmi30_data = sorted(kmi30_data, key=lambda x: x['market_cap_000'], reverse=True)

    # Create KMI100 data from top 100 KMIALL stocks
    kmi100_data = kmi_data[:100]
    for i, stock in enumerate(kmi100_data):
        stock['rank'] = i + 1
    print(f"  ✅ Sorted {len(kmi_data)} KMIALLSHR stocks by market capitalization")
    print(f"  ✅ Sorted {len(allshr_data)} ALLSHR stocks by market capitalization")
    print(f"  ✅ Sorted {len(kse100_data)} KSE100 stocks by market capitalization")
    print(f"  ✅ Sorted {len(kmi30_data)} KMI30 stocks by market capitalization")
    logging.info(f"Sorted {len(kmi_data)} KMIALLSHR stocks by market capitalization")
    logging.info(f"Sorted {len(allshr_data)} ALLSHR stocks by market capitalization")
    logging.info(f"Sorted {len(kse100_data)} KSE100 stocks by market capitalization")
    logging.info(f"Sorted {len(kmi30_data)} KMI30 stocks by market capitalization")
    
    # Save the data to the database
    print("\n💾 Saving data to database...")
    kmi_db_success = save_to_database(kmi_data, db_path, table_name='KMIALLSHR')
    allshr_db_success = save_to_database(allshr_data, db_path, table_name='ALLSHR')
    kse100_db_success = save_to_database(kse100_data, db_path, table_name='KSE100')
    kmi30_db_success = save_to_database(kmi30_data, db_path, table_name='KMI30')
    if kmi_db_success:
        print("  ✅ Successfully saved KMIALLSHR data to database")
    else:
        print("  ❌ Failed to save KMIALLSHR data to database")
    if allshr_db_success:
        print("  ✅ Successfully saved ALLSHR data to database")
    else:
        print("  ❌ Failed to save ALLSHR data to database")
    if kse100_db_success:
        print("  ✅ Successfully saved KSE100 data to database")
    else:
        print("  ❌ Failed to save KSE100 data to database")
    if kmi30_db_success:
        print("  ✅ Successfully saved KMI30 data to database")
    else:
        print("  ❌ Failed to save KMI30 data to database")
    
    # Export to Excel - use the specific Excel file path
    print("\n📊 Exporting data to Excel...")
    excel_path = Path("src/data_processing/psxsymbols.xlsx")  # Ensure correct path
    kmi_excel_success = export_to_excel(kmi100_data, excel_path, 'KMI100')
    allshr_excel_success = export_to_excel(allshr_data, excel_path, 'KSEALL')
    kse100_excel_success = export_to_excel(kse100_data, excel_path, 'KSE100')
    kmi30_excel_success = export_to_excel(kmi30_data, excel_path, 'KMI30')
    if kmi_excel_success:
        print("  ✅ Successfully exported KMIALLSHR data to Excel")
    else:
        print("  ❌ Failed to export KMIALLSHR data to Excel")
    if allshr_excel_success:
        print("  ✅ Successfully exported ALLSHR data to Excel")
    else:
        print("  ❌ Failed to export ALLSHR data to Excel")
    if kse100_excel_success:
        print("  ✅ Successfully exported KSE100 data to Excel")
    else:
        print("  ❌ Failed to export KSE100 data to Excel")
    if kmi30_excel_success:
        print("  ✅ Successfully exported KMI30 data to Excel")
    else:
        print("  ❌ Failed to export KMI30 data to Excel")
    
    # Export to CSV
    print("\n📊 Exporting data to CSV...")
    kmi_csv_success = export_to_csv(kmi_data, csv_path)
    allshr_csv_path = DATA_EXPORTS_DIR / 'reports' / 'ALLSHR.csv'
    allshr_csv_success = export_to_csv(allshr_data, allshr_csv_path)
    if kmi_csv_success:
        print("  ✅ Successfully exported KMIALLSHR data to CSV")
    else:
        print("  ❌ Failed to export KMIALLSHR data to CSV")
    if allshr_csv_success:
        print("  ✅ Successfully exported ALLSHR data to CSV")
    else:
        print("  ❌ Failed to export ALLSHR data to CSV")
    
    # Verify the KMI100 sheet after all operations
    if excel_path.exists():
        try:
            with pd.ExcelFile(excel_path) as xls:
                if 'KMI100' in xls.sheet_names:
                    final_df = pd.read_excel(xls, sheet_name='KMI100')
                    print(f"\n📊 Final KMI100 sheet status:")
                    print(f"  - Total rows: {len(final_df)}")
                    print(f"  - Columns: {', '.join(final_df.columns.tolist())}")
                else:
                    print(f"\n⚠️ KMI100 sheet not found in {excel_path} after processing")
        except Exception as e:
            print(f"\n⚠️ Error reading Excel file after processing: {e}")
    
    # Check if all operations completed successfully
    if (kmi_db_success and allshr_db_success and kse100_db_success and kmi30_db_success and
        kmi_excel_success and allshr_excel_success and kse100_excel_success and kmi30_excel_success and
        kmi_csv_success and allshr_csv_success):
        print("\n✅ KMIALLSHR data processing completed successfully")
        logging.info("KMIALLSHR data processing completed successfully.")
        return True
   
    else:
        print("\n⚠️ KMIALLSHR data processing completed with some issues")
        logging.warning("KMIALLSHR data processing completed with some issues.")
        return False

if __name__ == "__main__":
    # Run the main scraper
    main()
    
    # Optionally run the KMIALL update separately
    # update_kmiall_from_excel()