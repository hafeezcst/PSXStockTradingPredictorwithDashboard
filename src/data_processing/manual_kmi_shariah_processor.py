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

from config.paths import (
    DATA_LOGS_DIR,
    DATA_EXPORTS_DIR,
    PRODUCTION_DB_DIR,
    CONFIG_DIR,
    SCRIPTS_DIR
)

# Configure logging with proper path
logging.basicConfig(
    filename=DATA_LOGS_DIR / 'kmi_shariah_processor.log',
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
        return None

def save_to_database(data, db_path=None):
    """Save the extracted data to SQLite database with data retention"""
    if db_path is None:
        db_path = PRODUCTION_DB_DIR / "PSX_KMI_data.db"
    
    try:
        # Ensure the database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  - Database path: {db_path}")
        print(f"  - Database exists: {os.path.exists(db_path)}")
        
        logging.info(f"Saving {len(data)} records to database: {db_path}")
        
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
            conn.execute('''
            CREATE TABLE IF NOT EXISTS KMIALLSHR_backup (
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
                backup_date TEXT,
                PRIMARY KEY (symbol, date_added, backup_date)
            )
            ''')
            
            # Create main table if it doesn't exist
            conn.execute('''
            CREATE TABLE IF NOT EXISTS KMIALLSHR (
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
                PRIMARY KEY (symbol, date_added)
            )
            ''')
            
            # Begin transaction
            with conn:
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Backup existing data for today before deleting
                conn.execute('''
                INSERT INTO KMIALLSHR_backup
                SELECT *, ? FROM KMIALLSHR
                WHERE date_added = ?
                ''', (timestamp, today))
                
                # Delete existing records for today
                conn.execute("DELETE FROM KMIALLSHR WHERE date_added = ?", (today,))
                
                # Insert new data
                df.to_sql('KMIALLSHR', conn, if_exists='append', index=False)
                
                # Verify data was inserted correctly
                count = conn.execute("SELECT COUNT(*) FROM KMIALLSHR WHERE date_added = ?",
                                   (today,)).fetchone()[0]
                if count != len(data):
                    raise ValueError(f"Data count mismatch: expected {len(data)}, got {count}")
            
        logging.info(f"Successfully saved {len(data)} records to database with backup")
        return True
    except Exception as e:
        logging.error(f"Failed to save data to database: {e}")
        logging.error(traceback.format_exc())
        print(f"  ❌ Database error: {str(e)}")
        print(f"  ❌ Check log file for details: {DATA_LOGS_DIR / 'kmi_shariah_processor.log'}")
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
                        sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"  - Successfully updated {sheet_name} sheet while preserving {len(existing_sheets)} other sheets")
                
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

def main():
    """Main function to run the scraper"""
    print("\n🚀 Starting KMI Shariah Data Processor")
    print("=====================================")
    
    # Install required dependencies
    print("\n📦 Checking dependencies...")
    ensure_dependencies()
    
    # URL of the page to scrape
    url = "https://sarmaaya.pk/psx/market/KMIALLSHR"
    print(f"\n🌐 Target URL: {url}")
    
    # Paths for output - using path constants for consistency
    db_path = PRODUCTION_DB_DIR / 'PSXSymbols.db'
    
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
    html_content = None
    
    print("\n🔍 Fetching webpage content...")
    # Method 1: Standard requests
    print("  - Trying standard requests...")
    html_content = fetch_webpage(url)
    
    # Method 2: Cloudscraper
    if not html_content:
        print("  - Standard requests failed, trying cloudscraper...")
        html_content = fetch_webpage_cloudscraper(url)
    
    # Method 3: Selenium with fallbacks
    if not html_content:
        print("  - Cloudscraper failed, trying Selenium...")
        html_content = fetch_webpage_with_selenium(url)
    
    if not html_content:
        print("  ❌ All methods failed to fetch webpage content")
        logging.error("Failed to fetch webpage content by any method. Exiting.")
        return False
    
    print("  ✅ Successfully fetched webpage content")
    
    # Parse the HTML to extract data
    print("\n🔍 Parsing KMIALLSHR data...")
    data = parse_kmi_shariah_data(html_content)
    if not data or len(data) == 0:
        print("  ❌ Could not parse KMIALLSHR data")
        logging.error("Could not parse KMIALLSHR data. Exiting.")
        return False
    
    print(f"  ✅ Successfully parsed {len(data)} KMIALLSHR stocks")
        
    # Sort data by market cap in descending order
    data = sorted(data, key=lambda x: x['market_cap_000'], reverse=True)
    print(f"  ✅ Sorted {len(data)} stocks by market capitalization")
    logging.info(f"Sorted {len(data)} stocks by market capitalization")
    
    # Save the data to the database
    print("\n💾 Saving data to database...")
    db_success = save_to_database(data, db_path)
    if db_success:
        print("  ✅ Successfully saved data to database")
    else:
        print("  ❌ Failed to save data to database")
    
    # Export to Excel - use the specific Excel file path
    print("\n📊 Exporting data to Excel...")
    excel_success = export_to_excel(data, excel_path, 'KMI100')
    if excel_success:
        print("  ✅ Successfully exported data to Excel")
    else:
        print("  ❌ Failed to export data to Excel")
    
    # Export to CSV
    print("\n📊 Exporting data to CSV...")
    csv_success = export_to_csv(data, csv_path)
    if csv_success:
        print("  ✅ Successfully exported data to CSV")
    else:
        print("  ❌ Failed to export data to CSV")
    
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
    if db_success and excel_success and csv_success:
        print("\n✅ KMIALLSHR data processing completed successfully")
        logging.info("KMIALLSHR data processing completed successfully.")
        return True
   
    else:
        print("\n⚠️ KMIALLSHR data processing completed with some issues")
        logging.warning("KMIALLSHR data processing completed with some issues.")
        return False

if __name__ == "__main__":
    main()