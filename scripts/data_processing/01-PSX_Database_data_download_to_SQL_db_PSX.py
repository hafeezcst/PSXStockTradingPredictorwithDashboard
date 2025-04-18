import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.relativedelta import relativedelta
from pandas import DataFrame as container
from bs4 import BeautifulSoup as parser
from datetime import datetime, date, timedelta
from typing import Union
from tqdm import tqdm
import pandas as pd
import requests
import os
from sqlalchemy import create_engine, MetaData, Table, inspect, text
from collections import defaultdict
import time
import numpy as np

logging.basicConfig(filename='data_reader.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class DataReader:
    headers = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']

    def __init__(self, db_path=None, alt_db_path=None):
        self.__history = "https://dps.psx.com.pk/historical"
        self.__symbols = "https://dps.psx.com.pk/symbols"
        self.current_dir = os.getcwd()
        
        # Configure connection pooling
        pool_config = {
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }
        
        # Primary database with connection pooling
        if db_path is None:
            db_path = os.path.join(self.current_dir, 'data/databases/production/PSX_consolidated_data_PSX.db')
        self.db_path = db_path
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            pool_size=pool_config['pool_size'],
            max_overflow=pool_config['max_overflow'],
            pool_timeout=pool_config['pool_timeout'],
            pool_recycle=pool_config['pool_recycle']
        )
        
        # Alternative database with connection pooling
        if alt_db_path is None:
            alt_db_path = os.path.join(self.current_dir, 'data/databases/production/PSX_consolidated_data_PSX_Alternative.db')
        self.alt_db_path = alt_db_path
        self.alt_engine = create_engine(
            f'sqlite:///{alt_db_path}',
            pool_size=pool_config['pool_size'],
            max_overflow=pool_config['max_overflow'],
            pool_timeout=pool_config['pool_timeout'],
            pool_recycle=pool_config['pool_recycle']
        )
        
        self.metadata = MetaData()
        self.failed_scripts = 0
        
        # Configure session with retry and timeout
        self.session = requests.Session()
        retry_strategy = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount("https://", retry_strategy)
        self.session.mount("http://", retry_strategy)
        
        # Threading configuration with improved limits
        self.max_workers = 4  # Initial max workers
        self.max_threads = 8   # Reduced cap to prevent resource exhaustion
        self.min_threads = 2   # Minimum threads
        self.response_times = []  # Store response times to adjust threading dynamically
        self.request_timeout = 60  # Increased timeout for individual requests

    def tickers(self):
        try:
            return pd.read_json(self.__symbols)
        except Exception as e:
            logging.error(f"Error fetching tickers: {e}")
            return pd.DataFrame()

    def adjust_thread_count(self):
        """Adjust thread count based on recent response times."""
        if len(self.response_times) < 5:  # Adjust only after a few data points
            return

        avg_response_time = np.mean(self.response_times[-5:])  # Rolling average of the last 5 responses
        if avg_response_time > 1.5:  # Threshold to decrease threads
            self.max_workers = max(self.min_threads, self.max_workers - 1)
        elif avg_response_time < 0.5:  # Threshold to increase threads
            self.max_workers = min(self.max_threads, self.max_workers + 1)

    def get_last_date(self, symbol: str) -> Union[date, None]:
        table_name = f'PSX_{symbol}_stock_data'
        inspector = inspect(self.engine)
        if table_name in inspector.get_table_names():
            table = Table(table_name, self.metadata, autoload_with=self.engine)
            query = table.select().order_by(table.c.Date.desc()).limit(1)
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()
                if result:
                    return pd.to_datetime(result[0]).date()
        return None

    def get_today_data(self, symbol: str) -> container:
        """Fetch today's data if it does not already exist in the database."""
        today = date.today()
        # Check if today's data exists in the database
        if self.verify_data(symbol, today, today):
            logging.info(f"Today's data for {symbol} is already downloaded.")
            return pd.DataFrame()  # Return an empty DataFrame if data already exists
        
        # Otherwise, download today's data
        logging.info(f"Downloading today's data for {symbol}.")
        today_data = self.get_psx_data(symbol, [today])
        return today_data

    def get_psx_data(self, symbol: str, dates: list) -> container:
        data = []
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.download, symbol, date) for date in dates]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {symbol}'s Data"):
                    try:
                        result = future.result()
                        if isinstance(result, container):
                            data.append(result)
                    except Exception as e:
                        logging.error(f"Error downloading data for {symbol}: {e}")
                    time.sleep(0.5)  # Delay to reduce server load
        finally:
            # Ensure all futures are properly cleaned up
            for future in futures:
                future.cancel()
            # Clean up any remaining resources
            executor.shutdown(wait=False)

        return self.preprocess(data)

    def download(self, symbol: str, date: date):
        post = {"month": date.month, "year": date.year, "symbol": symbol}
        start_time = time.time()
        
        try:
            with self.session.post(
                self.__history,
                data=post,
                timeout=self.request_timeout
            ) as response:
                response.raise_for_status()
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                self.adjust_thread_count()

                # Parse response with error handling
                try:
                    data = parser(response.text, features="html.parser")
                    return self.toframe(data)
                except Exception as parse_error:
                    logging.error(f"Parse error for {symbol} on {date}: {parse_error}")
                    return pd.DataFrame()
                    
        except requests.exceptions.Timeout:
            logging.warning(f"Request timeout for {symbol} on {date}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as req_error:
            logging.error(f"Request failed for {symbol} on {date}: {req_error}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error downloading {symbol} on {date}: {e}")
            return pd.DataFrame()
        finally:
            # Explicit cleanup of resources
            if 'response' in locals():
                response.close()
            # Clean up any multiprocessing resources
            if hasattr(self, 'session'):
                self.session.close()

    def toframe(self, data):
        stocks = defaultdict(list)
        rows = data.select("tr")
        today = date.today()

        for row in rows:
            cols = [col.getText() for col in row.select("td")]
            for key, value in zip(self.headers, cols):
                if key == "TIME":
                    try:
                        parsed_date = datetime.strptime(value, "%b %d, %Y").date()
                        if parsed_date > today:
                            logging.warning(f"Skipping future date: {parsed_date} (input: {value})")
                            continue
                        logging.info(f"Parsed date: {parsed_date} from input: {value}")
                        value = parsed_date
                    except ValueError as e:
                        logging.error(f"Error parsing date {value}: {e}")
                        continue
                stocks[key].append(value)

        return pd.DataFrame(stocks, columns=self.headers).set_index("TIME")

    def daterange(self, start: date, end: date) -> list:
        period = end - start
        number_of_months = (period.days // 30) + 1
        current_date = datetime(start.year, start.month, 1).date()
        dates = [current_date]

        for _ in range(number_of_months):
            prev_date = dates[-1]
            next_date = (prev_date + relativedelta(months=1)).replace(day=1)
            if next_date <= datetime(end.year, end.month, 1).date():
                dates.append(next_date)

        return dates if len(dates) else [start]

    def preprocess(self, data: list) -> pd.DataFrame:
        data = [df for df in data if not df.empty]
        if not data:
            return pd.DataFrame()
        
        data = pd.concat(data)
        data = data.sort_index()
        data = data.rename(columns=str.title)
        data.index.name = "Date"
        data.Volume = data.Volume.str.replace(",", "")
        for column in data.columns:
            try:
                data[column] = data[column].str.replace(",", "").astype(float)
            except Exception as e:
                logging.error(f"Error converting column {column} to float: {e}")

        return data

    def save_to_db(self, data: pd.DataFrame, table_name: str, use_alt=False):
        engine = self.alt_engine if use_alt else self.engine
        if not data.empty:
            data.to_sql(table_name, engine, if_exists='append', index=True)
        else:
            logging.info(f"No data to save for table {table_name}")

    def verify_data(self, symbol: str, start_date: date, end_date: date, use_alt=False) -> bool:
        engine = self.alt_engine if use_alt else self.engine
        table_name = f'PSX_{symbol}_stock_data'
        inspector = inspect(engine)
        if table_name in inspector.get_table_names():
            try:
                with engine.connect().execution_options(statement_timeout=60) as conn:
                    query = text(f"SELECT COUNT(*) FROM {table_name} WHERE Date BETWEEN :start_date AND :end_date")
                    result = conn.execute(query, {"start_date": start_date, "end_date": end_date}).fetchone()
                    return result[0] > 0
            except Exception as e:
                logging.error(f"Database verification timeout for {table_name}: {e}")
                return False
        return False

    def delete_failing_table(self, symbol: str, use_alt=False):
        engine = self.alt_engine if use_alt else self.engine
        table_name = f'PSX_{symbol}_stock_data'
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        logging.info(f"Table {table_name} has been deleted due to repeated download failures.")

    def check_database_integrity(self):
        """Performs basic integrity checks on the primary and alternative databases."""
        try:
            with self.engine.connect() as conn:
                conn.execute("PRAGMA integrity_check;")
            with self.alt_engine.connect() as conn:
                conn.execute("PRAGMA integrity_check;")
            logging.info("Both databases passed the integrity check.")
        except Exception as e:
            logging.error(f"Database integrity check failed: {e}")

    def delete_unused_tables(self, valid_symbols):
        """Deletes tables from the database that are not listed in the provided valid symbols."""
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        valid_tables = {f"PSX_{symbol}_stock_data" for symbol in valid_symbols}

        for table in tables:
            if table not in valid_tables:
                with self.engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                logging.info(f"Deleted unused table {table} from the database.")

    def stocks(self, ticker: str, start: date, end: date) -> container:
        """Retrieve stock data for a specific ticker symbol over a date range."""
        dates = self.daterange(start, end)
        data = self.get_psx_data(ticker, dates)
        return data

data_reader = DataReader()

if __name__ == "__main__":
    current_dir = os.getcwd()
    symbols_file_path = os.path.join(current_dir, 'data/databases/production/psxsymbols.xlsx')
    
    # Load valid symbols from the 'KMI100' sheet
    symbols_df = pd.read_excel(symbols_file_path, sheet_name='KMI100')
    valid_symbols = symbols_df.iloc[:, 0].tolist()  # Get list of valid symbols from the first column
    print (f'total symbols are: {len(valid_symbols)}')
    # Delete any tables in the database that do not correspond to the valid symbols
    data_reader.delete_unused_tables(valid_symbols)
    
    end_date = date.today() - timedelta(days=1)
    fixed_start_date = date(2000, 1, 1)
    failed_attempts = 0

    data_reader.check_database_integrity()  # Perform an integrity check before starting

    for symbol in valid_symbols:
        last_date = data_reader.get_last_date(symbol)
        if last_date:
            start_date = last_date + timedelta(days=1)
        else:
            start_date = fixed_start_date

        if (end_date - start_date).days < 1:
            # Explicitly check and download today's data if needed
            today_data = data_reader.get_today_data(symbol)
            if not today_data.empty:
                data_reader.save_to_db(today_data, f'PSX_{symbol}_stock_data')
            continue

        attempts = 0
        while attempts < 5:
            data = data_reader.stocks(symbol, start_date, end_date)
            if not data.empty:
                data_reader.save_to_db(data, f'PSX_{symbol}_stock_data')
                if data_reader.verify_data(symbol, start_date, end_date):
                    logging.info(f"Data successfully saved and verified for {symbol}")
                    break
            attempts += 1
            logging.warning(f"Attempt {attempts} failed for {symbol}. Retrying...")

        if attempts >= 5:
            logging.error(f"Failed to download and save data for {symbol} after 5 attempts. Deleting the table.")
            #data_reader.delete_failing_table(symbol)
            #print (f'Failed to download and save data for {symbol} after 5 attempts. Deleting the table.')
            failed_attempts += 1

        if failed_attempts > 500:
            logging.error("More than 5 tables failed. Switching to alternative database for data integrity.")
            break
