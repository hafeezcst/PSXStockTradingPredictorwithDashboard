"""
PSX SQL Database to Indicator Converter

This module provides functionality to convert PSX stock market data from a SQL database
into technical indicators. It calculates various indicators including RSI, AO, and moving
averages across different timeframes.

Key Features:
- Reads data from SQLite database
- Calculates multiple technical indicators
- Saves processed data to a new SQLite database
- Supports daily, weekly, monthly, quarterly, and annual timeframes
- Includes comprehensive error handling and logging

Example Usage:
    data_reader = DataReader()
    table_names = data_reader.get_table_names()
    for table_name in table_names:
        data = data_reader.read_data(table_name)
        processed_data = data_reader.preprocess(data)
        data_reader.save_to_db(processed_data, table_name)
"""
import logging
import pandas as pd
import pandas_ta as ta
import os
from sqlalchemy import create_engine, inspect
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='data_reader.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataReader:
    """Handles reading, processing, and saving PSX stock market data with technical indicators.
    
    Attributes:
        source_engine (sqlalchemy.engine.Engine): Database engine for source data
        target_engine (sqlalchemy.engine.Engine): Database engine for processed data
        current_dir (str): Current working directory path
    """
    
    def __init__(self, source_db_path: str | None = None, target_db_path: str | None = None) -> None:
        """Initialize DataReader with database paths.
        
        Args:
            source_db_path: Path to source SQLite database. If None, uses default path.
            target_db_path: Path to target SQLite database. If None, uses default path.
        """
        self.current_dir = os.getcwd()
        if source_db_path is None:
            source_db_path = os.path.join(self.current_dir, 'data/databases/production/psx_consolidated_data_PSX.db')
        if target_db_path is None:
            target_db_path = os.path.join(self.current_dir, 'data/databases/production/psx_consolidated_data_indicators_PSX.db')

        self.source_engine = create_engine(f'sqlite:///{source_db_path}')
        self.target_engine = create_engine(f'sqlite:///{target_db_path}')

    def get_table_names(self) -> list[str]:
        """Get list of table names from the source database.
        
        Returns:
            list[str]: List of table names in the source database
        """
        inspector = inspect(self.source_engine)
        return inspector.get_table_names()

    def read_data(self, table_name: str) -> pd.DataFrame:
        """Read stock market data from the specified table.
        
        Args:
            table_name: Name of the table to read data from
            
        Returns:
            pd.DataFrame: DataFrame containing the stock market data, or empty DataFrame on error
        """
        try:
            data = pd.read_sql_table(table_name, self.source_engine, index_col='Date', parse_dates=['Date'])
            logging.info(f"Data read successfully from table {table_name}")
            return data
        except Exception as e:
            logging.error(f"Error reading data from table {table_name}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.Series, length: int) -> pd.Series | None:
        """Calculate the Relative Strength Index (RSI) for the given data.
        
        Args:
            data: Pandas Series containing price data (typically closing prices)
            length: Number of periods to use for RSI calculation
            
        Returns:
            pd.Series: Series containing RSI values, or None if calculation fails
            
        Raises:
            ValueError: If input data is empty or length is invalid
        """
        if data.empty:
            logging.error("Cannot calculate RSI: Input data is empty")
            return None
            
        if length <= 0:
            logging.error(f"Invalid RSI length: {length}. Must be positive integer")
            return None
            
        try:
            return ta.rsi(data, length=length)
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return None

    def _calculate_rsi_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicators across multiple timeframes.
        
        Args:
            data: DataFrame containing price data with 'Close' column
            
        Returns:
            pd.DataFrame: DataFrame with added RSI indicator columns
        """
        # Daily RSI and Average
        data['RSI_14'] = self.calculate_rsi(data['Close'], 14)
        data['RSI_14_Avg'] = ta.sma(data['RSI_14'], length=14)
        
        # Weekly RSI and Average (14 weeks * 5 trading days per week = 70 trading days)
        data['RSI_weekly'] = self.calculate_rsi(data['Close'], 70)
        data['RSI_weekly_Avg'] = ta.sma(data['RSI_weekly'], length=14)
        
        # Monthly RSI and Average (14 months * 21 trading days per month = 294 trading days)
        data['RSI_monthly'] = self.calculate_rsi(data['Close'], 294)
        data['RSI_monthly_Avg'] = ta.sma(data['RSI_monthly'], length=14)
        
        # Quarterly RSI and Average (14 quarters * 63 trading days per quarter = 882 trading days)
        data['RSI_3months'] = self.calculate_rsi(data['Close'], 882)
        data['RSI_3months_Avg'] = ta.sma(data['RSI_3months'], length=14)
        
        # semi-annual RSI and Average (14 semi-annual * 126 trading days per semi-annual = 1764 trading days)
        data['RSI_6months'] = self.calculate_rsi(data['Close'], 1764)
        data['RSI_6months_Avg'] = ta.sma(data['RSI_6months'], length=14)
        
        # Annual RSI and Average (14 annual * 252 trading days per annual = 3528 trading days)
        data['RSI_annual'] = self.calculate_rsi(data['Close'], 3528)
        data['RSI_annual_Avg'] = ta.sma(data['RSI_annual'], length=14)
        
        return data

    def _calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages for the stock data.
        
        Args:
            data: DataFrame containing price and volume data
            
        Returns:
            pd.DataFrame: DataFrame with added moving average columns
        """
        try:
            # Volume moving averages
            data['Volume_MA_20'] = ta.sma(data['Volume'], length=20)

            # Price moving averages
            data['MA_30'] = ta.sma(data['Close'], length=30)
            data['MA_30_weekly'] = ta.sma(data['Close'], length=30 * 5)
            data['MA_30_weekly_Avg'] = ta.sma(data['MA_30_weekly'], length=30)
            data['MA_50'] = ta.sma(data['Close'], length=50)
            data['MA_50_weekly'] = ta.sma(data['Close'], length=50 * 5)
            data['MA_50_weekly_Avg'] = ta.sma(data['MA_50_weekly'], length=50)
            data['MA_100'] = ta.sma(data['Close'], length=100)
            data['MA_200'] = ta.sma(data['Close'], length=200)

            # Additional RSI calculations
            data['RSI_9'] = self.calculate_rsi(data['Close'], 9)
            data['RSI_26'] = self.calculate_rsi(data['Close'], 26)

            return data
        except Exception as e:
            logging.error(f"Error calculating moving averages: {e}")
            return data

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess stock market data by calculating various technical indicators.
        
        Calculates RSI, moving averages, volume indicators, and other technical metrics
        across multiple timeframes (daily, weekly, monthly, etc.).
        
        Args:
            data: DataFrame containing raw stock market data with columns:
                - Date (index)
                - Open, High, Low, Close, Volume
                
        Returns:
            pd.DataFrame: Processed DataFrame with calculated indicators, or empty DataFrame on error
            
        Raises:
            ValueError: If required columns are missing from input data
        """
        if data.empty:
            logging.info("Empty dataframe received for preprocessing.")
            return pd.DataFrame()

        # Validate required columns
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            logging.error(f"Missing required columns: {missing}")
            return pd.DataFrame()

        # Ensure data is sorted chronologically
        data = data.sort_index()

        try:
            # Calculate RSI indicators
            data = self._calculate_rsi_indicators(data)
            # Daily RSI and Average
            data['RSI_14'] = self.calculate_rsi(data['Close'], 14)
            data['RSI_14_Avg'] = ta.sma(data['RSI_14'], length=14)
            
            # Weekly RSI and Average (14 weeks * 5 trading days per week = 70 trading days)
            data['RSI_weekly'] = self.calculate_rsi(data['Close'], 70)
            data['RSI_weekly_Avg'] = ta.sma(data['RSI_weekly'], length=14)
            
            # Monthly RSI and Average (14 months * 21 trading days per month = 294 trading days)
            data['RSI_monthly'] = self.calculate_rsi(data['Close'], 294)
            data['RSI_monthly_Avg'] = ta.sma(data['RSI_monthly'], length=14)
            
            # Quarterly RSI and Average (14 quarters * 63 trading days per quarter = 882 trading days)
            data['RSI_3months'] = self.calculate_rsi(data['Close'], 882)
            data['RSI_3months_Avg'] = ta.sma(data['RSI_3months'], length=14)
            
            # semi-annual RSI and Average (14 semi-annual * 126 trading days per semi-annual = 1764 trading days)
            data['RSI_6months'] = self.calculate_rsi(data['Close'], 1764)
            data['RSI_6months_Avg'] = ta.sma(data['RSI_6months'], length=14)
            
            # Annual RSI and Average (14 annual * 252 trading days per annual = 3528 trading days)
            data['RSI_annual'] = self.calculate_rsi(data['Close'], 3528)
            data['RSI_annual_Avg'] = ta.sma(data['RSI_annual'], length=14)
            
            # Calculate moving averages and other basic indicators
            data = self._calculate_moving_averages(data)
            data['Pct_Change'] = data['Close'].pct_change() * 100
            data['Daily_Fluctuation'] = data['High'] - data['Low']
            
            # AO (Awesome Oscillator) calculations
            hl2 = (data['High'] + data['Low']) / 2
            
            # Daily AO
            data['AO'] = ta.sma(hl2, 5) - ta.sma(hl2, 34)
            # Daily AO_AVG
            data['AO_AVG'] = ta.sma(data['AO'], 5)
            
            # Weekly AO (5 weeks and 34 weeks)
            data['AO_weekly'] = ta.sma(hl2, 25) - ta.sma(hl2, 170)
            # Weekly AO_AVG (5 weeks and 34 weeks)
            data['AO_weekly_AVG'] = ta.sma(data['AO_weekly'], 5)
            # Capture the trend of AO as an indicator

            # Check for downtrend in the last 5 weeks
            def check_downtrend(series):
                return 'Down' if (series.tail(5) == 'Down').all() else 'Up'

            data['AO_weekly_Trend'] = data['AO_weekly_Trend'].rolling(window=5).apply(check_downtrend, raw=False)
            
            # Monthly AO (5 months and 34 months)
            data['AO_monthly'] = ta.sma(hl2, 105) - ta.sma(hl2, 714)
            # Monthly AO_AVG (5 months and 34 months)
            data['AO_monthly_AVG'] = ta.sma(data['AO_monthly'], 5)
            
            # Quarterly AO (5 quarters and 34 quarters)
            data['AO_3Months'] = ta.sma(hl2, 315) - ta.sma(hl2, 2142)
            # Quarterly AO_AVG (5 quarters and 34 quarters)
            data['AO_3Months_AVG'] = ta.sma(data['AO_3Months'], 5)
            
            # semi-annual AO (5 semi-annual and 34 semi-annual)
            data['AO_6Months'] = ta.sma(hl2, 630) - ta.sma(hl2, 4284)
            # semi-annual AO_AVG (5 semi-annual and 34 semi-annual)
            data['AO_6Months_AVG'] = ta.sma(data['AO_6Months'], 5)    
            # ATR function for weekly 
            data['ATR_weekly'] = ta.atr(data['High'], data['Low'], data['Close'], length=70)
            # ATR Average for weekly
            data['ATR_weekly_Avg'] = ta.sma(data['ATR_weekly'], length=14)            
            #weekly high and low
            data['weekly_high'] = data['High'].resample('W-FRI').max()
            #weekly high and low
            data['weekly_low'] = data['Low'].resample('W-FRI').min()
            #monthly high and low
            data['monthly_high'] = data['High'].resample('ME').max()
            #monthly high and low
            data['monthly_low'] = data['Low'].resample('ME').min()
            
              
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")

        return data

    def save_to_db(self, data: pd.DataFrame, table_name: str):
        if not data.empty:
            try:
                data.to_sql(table_name, self.target_engine, if_exists='replace', index=True)
                logging.info(f"Data saved to table {table_name}")
            except Exception as e:
                logging.error(f"Error saving data to table {table_name}: {e}")
        else:
            logging.info(f"No data to save for table {table_name}")

    def delete_unused_tables(self, valid_symbols):
        """Deletes tables from the target database that are not listed in the provided valid symbols."""
        from sqlalchemy import text
        inspector = inspect(self.target_engine)
        tables = inspector.get_table_names()
        valid_tables = {f"PSX_{symbol}_stock_data" for symbol in valid_symbols}

        for table in tables:
            if table not in valid_tables:
                with self.target_engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                logging.info(f"Deleted unused table {table} from the database.")

if __name__ == "__main__":
    data_reader = DataReader()
    table_names = data_reader.get_table_names()
        # Optional: Clean up unused tables
    # valid_symbols = ["KSE100", "OGDC", "PPL"]  # Add your valid symbols here
    data_reader.delete_unused_tables(table_names)

    for table_name in tqdm(table_names, desc="Processing tables"):
        data = data_reader.read_data(table_name)
        processed_data = data_reader.preprocess(data)
        data_reader.save_to_db(processed_data, table_name)



    # Ensure database connections are closed properly
    data_reader.source_engine.dispose()
    data_reader.target_engine.dispose()
