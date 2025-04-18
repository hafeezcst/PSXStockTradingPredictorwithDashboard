import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import os
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config.paths import (
    DATA_LOGS_DIR,
    DATA_REPORTS_DIR,
    PRODUCTION_DB_DIR,
)

# Set up logging
logging.basicConfig(
    filename=DATA_LOGS_DIR / 'mutual_funds_favorites.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
                
                # OPTION 4: Try with a different approach
                try:
                    import cloudscraper
                    scraper = cloudscraper.create_scraper()
                    response = scraper.get(url)
                    if response.status_code == 200:
                        return response.text
                except Exception as cloud_error:
                    logging.warning(f"Cloudscraper final attempt failed: {cloud_error}")
                
                # If all methods fail
                return None
    except Exception as e:
        logging.error(f"Selenium and all fallback methods failed: {e}")
        logging.error(traceback.format_exc())
        return None

def parse_mutual_funds_favorites(html_content):
    """Parse the HTML to extract mutual funds favorite stocks with specific columns"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    try:
        logging.info("Parsing mutual funds favorite stocks data")
        
        # Find all tables on the page
        tables = soup.find_all('table', {'class': 'table'})
        
        # The table we want should have the specific headers we're looking for
        target_table = None
        for table in tables:
            headers = table.find_all('th')
            header_texts = [h.text.strip() for h in headers if h.text.strip()]
            
            # Check if this looks like our target table
            if any('Symbol' in h for h in header_texts) and any('Funds Invested' in h for h in header_texts):
                target_table = table
                break
        
        if not target_table:
            logging.warning("Could not find the table with mutual funds data")
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
        funds_count_idx = next((i for i, h in enumerate(headers) if 'No of Funds' in h), None)
        rupee_invested_idx = next((i for i, h in enumerate(headers) if 'Rupee Invested' in h), None)
        
        if None in (symbol_idx, funds_count_idx, rupee_invested_idx):
            logging.error(f"Could not find all required columns in headers: {headers}")
            return None
            
        # Parse data rows
        data = []
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        for row in rows[1:]:  # Skip header row
            columns = row.find_all('td')
            if len(columns) > max(symbol_idx, funds_count_idx, rupee_invested_idx):
                # Extract symbol
                symbol = columns[symbol_idx].text.strip()
                
                # Extract number of funds invested
                funds_count_text = columns[funds_count_idx].text.strip()
                funds_count = int(funds_count_text) if funds_count_text.isdigit() else 0
                
                # Extract rupee invested (remove commas and convert to number)
                rupee_text = columns[rupee_invested_idx].text.strip()
                rupee_text = rupee_text.replace(',', '')  # Remove commas
                rupee_invested = 0
                
                # Try to extract numeric value
                import re
                numeric_match = re.search(r'(\d+(?:\.\d+)?)', rupee_text)
                if numeric_match:
                    rupee_invested = float(numeric_match.group(1))
                
                record = {
                    'symbol': symbol,
                    'no_of_funds_invested': funds_count,
                    'rupee_invested_000': rupee_invested,
                    'date_added': current_date
                }
                
                data.append(record)
        
        logging.info(f"Found {len(data)} mutual funds favorite stocks")
        return data
        
    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")
        logging.error(traceback.format_exc())
        return None

def save_to_database(data, db_path):
    """Save the extracted data to SQLite database"""
    try:
        db_path = Path(db_path)
        # Ensure the database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Saving {len(data)} records to database: {db_path}")
        
        # Add update timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for record in data:
            record['update_date'] = timestamp
        
        # Convert to DataFrame for easier DB operations
        df = pd.DataFrame(data)
        
        # Connect to SQLite database
        with sqlite3.connect(db_path) as conn:
            # Create table if it doesn't exist
            conn.execute('''
            CREATE TABLE IF NOT EXISTS MutualFundsFavourite (
                symbol TEXT,
                no_of_funds_invested INTEGER,
                rupee_invested_000 REAL,
                date_added TEXT,
                update_date TEXT,
                PRIMARY KEY (symbol, date_added)
            )
            ''')
            
            # Delete existing records for today to avoid duplicates
            today = datetime.now().strftime('%Y-%m-%d')
            conn.execute("DELETE FROM MutualFundsFavourite WHERE date_added = ?", (today,))
            
            # Insert data
            df.to_sql('MutualFundsFavourite', conn, if_exists='append', index=False)
            
        logging.info(f"Successfully saved {len(data)} records to database")
        return True
    except Exception as e:
        logging.error(f"Failed to save data to database: {e}")
        logging.error(traceback.format_exc())
        return False

def export_to_csv(data, csv_path):
    """Export the data to CSV file"""
    try:
        csv_path = Path(csv_path)
        # Ensure the directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logging.info(f"Exported data to CSV: {csv_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to export to CSV: {e}")
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

def main():
    """Main function to run the scraper"""
    # Install required dependencies
    ensure_dependencies()
    
    # URL of the page to scrape
    url = "https://sarmaaya.pk/psx/most-favorite"
    
    # Paths for output using pathlib
    db_path = PRODUCTION_DB_DIR / "PSXSymbols.db"
    csv_path = DATA_REPORTS_DIR / 'mutual_funds_favorites.csv'
    
    # Print paths for debugging
    print(f"Database path: {db_path}")
    print(f"CSV path: {csv_path}")
    
    # Make sure the directories exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try multiple methods to get the content
    html_content = None
    
    # Method 1: Standard requests
    html_content = fetch_webpage(url)
    
    # Method 2: Cloudscraper
    if not html_content:
        html_content = fetch_webpage_cloudscraper(url)
    
    # Method 3: Selenium with fallbacks
    if not html_content:
        html_content = fetch_webpage_with_selenium(url)
    
    if not html_content:
        print("ERROR: Failed to fetch webpage content by any method.")
        logging.error("Failed to fetch webpage content by any method. Exiting.")
        return False
    
    print(f"Successfully fetched webpage content - length: {len(html_content)} characters")
    
    # Parse the HTML to extract data
    data = parse_mutual_funds_favorites(html_content)
    if not data or len(data) == 0:
        print("ERROR: Could not parse mutual funds favorites data.")
        logging.error("Could not parse mutual funds favorites data. Exiting.")
        return False
    
    print(f"Successfully parsed {len(data)} mutual funds favorites stocks")
    
    # Save the data to the database
    db_success = save_to_database(data, str(db_path))
    print(f"Database save result: {'Success' if db_success else 'Failed'}")
    
    # Export to CSV
    csv_success = export_to_csv(data, str(csv_path))
    print(f"CSV export result: {'Success' if csv_success else 'Failed'}")
    
    if db_success and csv_success:
        print("Mutual funds favorites data processing completed successfully.")
        logging.info("Mutual funds favorites data processing completed successfully.")
        return True
    else:
        print("WARNING: Mutual funds favorites data processing completed with some issues.")
        logging.warning("Mutual funds favorites data processing completed with some issues.")
        return False

if __name__ == "__main__":
    main()