import sqlite3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import time

# Setup session with retry mechanism
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Connect to the database using a context manager
with sqlite3.connect('data/databases/production/psxsymbols.db') as conn:
    cursor = conn.cursor()

    # Add new columns to the table if they don't exist
    columns = {
        "marketCap": "TEXT",
        "shares": "TEXT",
        "freeFloatShares": "TEXT",
        "freeFloatRatio": "TEXT"
    }

    for column, col_type in columns.items():
        try:
            cursor.execute(f"ALTER TABLE KSEALL ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            print(f"Column '{column}' already exists")

    # Fetch all symbols from the database
    cursor.execute("SELECT Symbol FROM KSEALL ORDER BY Symbol ASC")
    symbols = [symbol[0].strip() for symbol in cursor.fetchall()]

    # Helper function to extract stat values based on the provided label
    def get_stat_value(label, soup):
        """Extracts the value corresponding to a stats label."""
        try:
            # Use 'contains' to match labels with span or other HTML tags inside them
            items = soup.find_all('div', class_='stats_item')
            for item in items:
                stats_label = item.find('div', class_='stats_label')
                stats_value = item.find('div', class_='stats_value')
                if stats_label and label in stats_label.get_text():
                    value = stats_value.get_text(strip=True)
                    print(f"Found {label}: {value}")  # Debugging print
                    return value
            print(f"Label '{label}' not found")
            return 'N/A'
        except Exception as e:
            print(f"Error fetching {label}: {e}")
            return 'N/A'

    # Function to fetch the equity profile for a company symbol
    def fetch_equity_profile(symbol):
        url = f"https://dps.psx.com.pk/company/{symbol}"
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data for {symbol}: {e}")
            return None  # Skip this symbol if the request fails

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract financial data based on the correct HTML structure
        market_cap = get_stat_value("Market Cap", soup) or '0'
        shares = get_stat_value("Shares", soup)
        free_float_shares = get_stat_value("Free Float", soup)

        # Handle multiple 'Free Float' entries (if applicable)
        free_float_items = soup.find_all('div', class_='stats_label', string='Free Float')
        free_float_ratio = (
            free_float_items[1].find_next_sibling('div', class_='stats_value').text.strip()
            if len(free_float_items) > 1 else 'N/A'
        )

        return {
            'Market Cap': market_cap,
            'Shares': shares,
            'Free Float Shares': free_float_shares,
            'Free Float Ratio': free_float_ratio
        }

    # Prepare data for batch update
    data_for_update = []
    for symbol in symbols:
        profile = fetch_equity_profile(symbol)
        if profile:  # Only add to the update list if data was fetched successfully
            data_for_update.append((
                profile['Market Cap'],
                profile['Shares'],
                profile['Free Float Shares'],
                profile['Free Float Ratio'],
                symbol
            ))
            print(f"Fetched profile for {symbol}: {profile}")

        # Politeness delay to avoid overwhelming the server
        time.sleep(2)

    # Perform batch update using executemany
    if data_for_update:
        cursor.executemany(
            """UPDATE KMIALL 
               SET marketCap = ?, shares = ?, freeFloatShares = ?, freeFloatRatio = ? 
               WHERE Symbol = ?""",
            data_for_update
        )
        print("Batch update completed.")
    else:
        print("No data to update.")
