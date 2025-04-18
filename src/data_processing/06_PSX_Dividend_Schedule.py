import logging
import requests
import sqlite3
import os
import time
import random
from datetime import datetime
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field  # Add import for field
import csv
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

# Telegram configuration
bot_token = '6860197701:AAESTzERZLYbqyU6gFKfAwJQL8jJ_HNKLbM'
chat_id = '-4152327824'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('psx_dividend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper functions for default factories
def default_user_agents():
    return [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0"
    ]

@dataclass
class Config:
    """Configuration settings for the scraper"""
    base_url: str = "https://www.ksestocks.com/BookClosures"
    db_path: str = "PSX_Dividend_Schedule.db"
    export_dir: str = "exports"
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    user_agents: List[str] = field(default_factory=default_user_agents)
    proxies: List[str] = field(default_factory=list)  # Empty list by default

@dataclass
class DividendRecord:
    """Data structure for dividend records"""
    symbol: str
    company_name: str
    face_value: float
    last_close: float
    bc_from: str
    bc_to: str
    dividend_amount: Optional[float]
    right_amount: Optional[float]
    payout_text: str
    data_type: str = "current"

class PSXDividendScraper:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self._configure_session()

    def _configure_session(self):
        """Configure the requests session with headers and proxies"""
        self.session.headers.update({
            'User-Agent': random.choice(self.config.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        if self.config.proxies:
            self.session.proxies.update({
                'http': random.choice(self.config.proxies),
                'https': random.choice(self.config.proxies)
            })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make a web request with retry logic"""
        try:
            response = self.session.get(
                url,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def _parse_dividend_amount(self, payout: str) -> Dict[str, float]:
        """Parse dividend and right amounts from payout string with validation"""
        info = {'dividend_amount': None, 'right_amount': None}
        
        if not payout or not isinstance(payout, str):
            return info
            
        try:
            if 'Dividend=' in payout:
                amount = float(payout.split('Dividend=')[1].rstrip('%'))
                info['dividend_amount'] = amount / 100
            elif 'Right=' in payout:
                amount = float(payout.split('Right=')[1].rstrip('%'))
                info['right_amount'] = amount / 100
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing payout {payout}: {e}")
            
        return info

    def scrape_dividend_data(self) -> List[DividendRecord]:
        """Scrape dividend announcements with error handling and validation"""
        records = []
        
        try:
            logger.info("Fetching dividend announcements")
            response = self._make_request(self.config.base_url)
            
            try:
                data_start = response.text.find('var bcs=')
                if data_start == -1:
                    raise ValueError("Could not find dividend data in response")
                    
                data_end = response.text.find('};', data_start) + 2
                js_data = response.text[data_start:data_end].replace('var bcs=', '')
                
                # Clean and parse JSON data
                js_data = js_data.strip().rstrip(';')
                data = json.loads(js_data.replace("'", '"'))
                
                # Process and validate announcements
                for announcement in data.get('cur', []):
                    if not isinstance(announcement, dict):
                        continue
                        
                    symbol = announcement.get('symbol')
                    if not symbol:
                        continue
                        
                    payout_info = self._parse_dividend_amount(announcement.get('payout', ''))
                    
                    record = DividendRecord(
                        symbol=symbol,
                        company_name=announcement.get('cname', ''),
                        face_value=float(announcement.get('faceval', 0)),
                        last_close=float(announcement.get('lc', 0)),
                        bc_from=announcement.get('bcfrom', ''),
                        bc_to=announcement.get('bcto', ''),
                        dividend_amount=payout_info['dividend_amount'],
                        right_amount=payout_info['right_amount'],
                        payout_text=announcement.get('payout', '')
                    )
                    records.append(record)
                    
                logger.info(f"Found {len(records)} valid dividend announcements")
                
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing dividend data: {e}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"Error fetching announcements: {e}")
            return []
            
        return records

class DataExporter:
    """Handles data export to various formats"""
    
    def __init__(self, export_dir: str):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_to_csv(self, records: List[DividendRecord], filename: str) -> bool:
        """Export records to CSV file"""
        try:
            filepath = self.export_dir / f"{filename}.csv"
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=DividendRecord.__annotations__.keys())
                writer.writeheader()
                writer.writerows([record.__dict__ for record in records])
            logger.info(f"Exported {len(records)} records to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def export_to_json(self, records: List[DividendRecord], filename: str) -> bool:
        """Export records to JSON file"""
        try:
            filepath = self.export_dir / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([record.__dict__ for record in records], f, indent=2)
            logger.info(f"Exported {len(records)} records to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False

class DatabaseManager:
    """Handles database operations using context manager"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._setup_database()

    def _setup_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS dividend_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                company_name TEXT,
                face_value REAL,
                last_close REAL,
                bc_from TEXT,
                bc_to TEXT,
                dividend_amount REAL,
                right_amount REAL,
                payout_text TEXT,
                data_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, bc_from, bc_to, payout_text)
            )''')

    def _get_connection(self):
        """Get database connection with context manager"""
        return sqlite3.connect(self.db_path)

    def save_records(self, records: List[DividendRecord]) -> bool:
        """Save records to database with error handling"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for record in records:
                    cursor.execute('''
                    INSERT OR REPLACE INTO dividend_schedule 
                    (symbol, company_name, face_value, last_close, bc_from, bc_to, 
                     dividend_amount, right_amount, payout_text, data_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.symbol,
                        record.company_name,
                        record.face_value,
                        record.last_close,
                        record.bc_from,
                        record.bc_to,
                        record.dividend_amount,
                        record.right_amount,
                        record.payout_text,
                        record.data_type
                    ))
                conn.commit()
            logger.info(f"Saved {len(records)} records to database")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False

def send_telegram_message(message: str, bot_token_param: str = None, chat_id_param: str = None):
    """Send a message to a Telegram chat."""
    # Use global variables, parameters, or environment variables (in that order)
    global bot_token, chat_id
    
    # Use provided parameters first, then fall back to global variables
    bot_token_to_use = bot_token_param or bot_token or os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id_to_use = chat_id_param or chat_id or os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token_to_use or not chat_id_to_use:
        logger.error("Telegram bot token or chat ID not provided")
        return False
        
    url = f"https://api.telegram.org/bot{bot_token_to_use}/sendMessage"
    max_message_length = 4096  # Telegram's maximum message length
    
    try:
        for i in range(0, len(message), max_message_length):
            payload = {
                "chat_id": chat_id_to_use,
                "text": message[i:i + max_message_length],
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
        logger.info("Message sent to Telegram successfully")
        return True
    except requests.RequestException as e:
        logger.error(f"Error sending message to Telegram: {e}")
        return False

def format_dividend_data_for_telegram(records: List[DividendRecord]) -> str:
    """Format dividend records for Telegram message"""
    if not records:
        return "No dividend announcements found."
    
    # Header
    message = "📊 *PSX DIVIDEND ANNOUNCEMENTS* 📊\n\n"
    
    # Format each record
    for i, record in enumerate(records[:20], 1):  # Limit to first 20 records
        dividend_info = f"{record.dividend_amount:.2%}" if record.dividend_amount else "N/A"
        right_info = f"{record.right_amount:.2%}" if record.right_amount else "N/A"
        
        message += f"*{i}. {record.symbol}* - {record.company_name}\n"
        message += f"   📅 BC: {record.bc_from} to {record.bc_to}\n"
        message += f"   💰 Dividend: {dividend_info}, Right: {right_info}\n"
        message += f"   💵 Last Close: Rs. {record.last_close:.2f}\n\n"
    
    # Add summary footer
    message += f"\nTotal Announcements: {len(records)}"
    if len(records) > 20:
        message += f" (showing first 20)"
    
    return message

def main():
    """Main execution function"""
    config = Config()
    
    # Initialize components
    scraper = PSXDividendScraper(config)
    db_manager = DatabaseManager(config.db_path)
    exporter = DataExporter(config.export_dir)
    
    # Scrape data
    records = scraper.scrape_dividend_data()
    if not records:
        logger.error("No dividend announcements found")
        return
        
    # Save to database
    if not db_manager.save_records(records):
        logger.error("Failed to save records to database")
        return
        
    # Export data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exporter.export_to_csv(records, f"dividends_{timestamp}")
    exporter.export_to_json(records, f"dividends_{timestamp}")
    
    # Send Telegram message with dividend information
    telegram_message = format_dividend_data_for_telegram(records)
    if send_telegram_message(telegram_message):
        logger.info("Dividend announcements sent to Telegram")
    else:
        logger.error("Failed to send dividend announcements to Telegram")

if __name__ == "__main__":
    main()