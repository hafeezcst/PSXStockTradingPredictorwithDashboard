# PSX Database Data Download Module

## Core Functionalities

1. **DataReader Class**: 
   - Central class that manages all data operations
   - Handles connection to PSX data sources and local databases

2. **Data Retrieval**:
   - Fetches historical stock data from PSX website (`dps.psx.com.pk`)
   - Implements multi-threaded downloads with dynamic thread adjustment
   - Handles date ranges efficiently by downloading data in monthly chunks

3. **Data Processing**:
   - Parses HTML responses using BeautifulSoup
   - Converts raw data to structured pandas DataFrames
   - Cleans data (removing commas, converting to appropriate types)
   - Sorts and organizes data by date

4. **Database Management**:
   - Uses SQLite with connection pooling for performance 
   - Maintains both primary and alternative databases for redundancy
   - Includes methods to verify data integrity and perform database checks
   - Handles table creation, updates, and deletion

5. **Error Handling & Resilience**:
   - Comprehensive logging system
   - Implements retry mechanisms for failed downloads
   - Timeout handling for requests
   - Graceful failure handling with appropriate cleanup

6. **Resource Optimization**:
   - Dynamic thread count adjustment based on server response times
   - Connection pooling for database access
   - Session management for HTTP requests

## Main Workflow

When executed directly, the script:
1. Loads valid stock symbols from an Excel file
2. Cleans up unused tables in the database
3. For each symbol:
   - Determines the date range needing updates
   - Downloads missing data
   - Processes and saves the data to the database
   - Verifies data integrity
   - Retries failed downloads up to 5 times
4. Switches to an alternative database if too many failures occur

This module is designed for both one-time historical data downloads and regular updates to keep the database current with the latest PSX stock information.

# PSX SQL Indicator Module

## Core Functionalities

1. **DataReader Class**:
   - Handles reading from source database and writing to target database
   - Manages the calculation of technical indicators
   - Provides database maintenance operations

2. **Technical Indicator Calculation**:
   - **RSI (Relative Strength Index)** across multiple timeframes:
     - Daily, weekly, monthly, quarterly, semi-annual, and annual
   - **Moving Averages** at various periods:
     - 30, 50, 100, and 200-day periods
     - Weekly variations and averages
   - **Awesome Oscillator (AO)** calculations:
     - Daily, weekly, monthly, quarterly, and semi-annual periods
     - Trend analysis for oscillators
   - **ATR (Average True Range)** indicators
   - **Volume-based indicators** and moving averages
   - **Price metrics** (daily fluctuation, percentage change)

3. **Data Processing Pipeline**:
   - Reads raw stock data from SQLite database
   - Validates and preprocesses the data
   - Calculates all technical indicators
   - Handles time-based resampling for weekly/monthly metrics
   - Saves processed data to target database

4. **Error Handling & Logging**:
   - Comprehensive logging system
   - Data validation and error handling
   - Graceful handling of missing or incomplete data

5. **Database Management**:
   - Reads from source database with raw stock data
   - Writes to target database with processed indicators
   - Maintains database tables (deleting unused tables)
   - Proper connection handling and resource cleanup

## Main Workflow

When executed directly, the script:
1. Initializes database connections to source and target databases
2. Retrieves all table names from the source database
3. Optionally deletes unused tables from the target database
4. For each table (representing a stock):
   - Reads the raw price and volume data
   - Calculates all technical indicators across timeframes
   - Saves the processed data to the target database
5. Properly closes all database connections

This module serves as a critical data transformation layer, converting raw stock price data into actionable technical indicators for analysis and trading strategies.

# PSX Weekly RSI Analysis Module

## Core Functionalities

1. **Signal Generation & Stock Analysis**:
   - Analyzes technical indicators (RSI, AO, Moving Averages) across multiple timeframes
   - Categorizes stocks into Buy, Sell, and Neutral signals based on specific criteria
   - Calculates profit/loss percentages and holding days for active positions
   - Identifies trend directions and momentum patterns

2. **Data Retrieval & Processing**:
   - Connects to SQLite databases containing processed PSX stock data
   - Fetches technical indicator values from tables
   - Filters stocks based on the KMI30 and KMIALL indices
   - Retrieves additional data such as free float ratios and multibagger status

3. **Automated Signal Notifications**:
   - Formats detailed stock reports with current indicators and performance metrics
   - Sends formatted notifications to a Telegram channel using Markdown formatting
   - Handles message chunking for large reports
   - Includes AI-generated descriptions explaining signal context and meaning

4. **Dividend Information Integration**:
   - Fetches upcoming dividend details from a dedicated database
   - Calculates dividend yields based on current prices
   - Includes book closure dates and payout information in signal reports

5. **Database Management**:
   - Updates an investment tracking database with new signals
   - Prevents duplicate entries through uniqueness checking
   - Maintains historical signal data with timestamps
   - Provides signal status tracking and verification

## Main Workflow

When executed directly, the script:
1. Connects to the indicator database containing processed technical data
2. Retrieves stock data and filters by predefined technical criteria
3. Processes each stock and categorizes into Buy, Sell, or Neutral signals
4. Updates a dedicated database with newly identified signals
5. Formats detailed reports with technical analysis and AI-generated descriptions
6. Sends notification messages to Telegram channel for each signal type
7. Logs all activities and results for monitoring

This module provides actionable trading signals based on technical analysis of PSX stocks, primarily focusing on the KMI30 index components.

# PSX Technical Indicator Trend Lines and Dashboard Module

## Core Functionalities

1. **Technical Analysis Visualization**:
   - Generates detailed charts showing key technical indicators (RSI, AO, Moving Averages)
   - Creates multi-panel visualizations with overlapping indicators for pattern recognition
   - Marks buy/sell signals directly on charts with clear annotations
   - Implements dual-color visualization for positive/negative indicator values

2. **Market Phase Analysis**:
   - Identifies accumulation/distribution phases through multi-factor analysis
   - Calculates phase probabilities based on technical indicator patterns
   - Analyzes RSI trends to detect overbought/oversold conditions
   - Evaluates Awesome Oscillator momentum shifts and zero-line crossovers
   - Examines volume patterns for distribution/accumulation signs
   - Assesses price-MA relationships across multiple timeframes

3. **Dashboard Generation**:
   - Creates comprehensive market dashboards with multiple visualization panels
   - Includes market breadth indicators showing percentage of stocks in various conditions
   - Generates heat maps of market momentum and sentiment
   - Provides sector rotation analysis and decision support matrices
   - Creates HTML and CSV exports of various stock categories

4. **Signal Management & Tracking**:
   - Retrieves and displays buy/sell signals from database
   - Calculates holding days and profit/loss for active positions
   - Provides performance analysis by holding period
   - Identifies optimal exit windows based on historical performance

5. **Automated Reporting**:
   - Sends generated charts and dashboards to Telegram for monitoring
   - Formats signal data and analysis for messaging
   - Creates comprehensive market summaries and portfolio recommendations
   - Implements message chunking for large reports

## Main Workflow

When executed directly, the script:
1. Connects to the indicator database containing processed technical data
2. Retrieves recent buy and sell signals from the signals database
3. Generates individual technical charts for stocks with active buy signals
4. Creates a comprehensive market dashboard with multiple visualization panels
5. Calculates portfolio recommendations based on current market conditions
6. Sends all generated visualizations and analysis to Telegram
7. Saves outputs in organized folders for future reference

This module serves as a powerful technical analysis visualization and decision support system, providing actionable insights based on technical indicators across the PSX market.

# PSX Data Collection and Processing Modules

## 1. Free Float Data Collector (`free_float.py`)
- Fetches free float data from PSX website for listed companies
- Extracts market cap, shares outstanding, and free float ratios
- Updates SQLite database with company-specific float information
- Implements retry mechanisms and error handling for web requests
- Maintains historical free float data for analysis

## 2. KMI Shariah Data Processor (`manual_kmi_shariah_processor.py`)
- Scrapes KMI (Karachi Meezan Index) Shariah-compliant stocks data
- Processes market data including:
  - Points and weights for each stock
  - Current prices and changes
  - 52-week highs and lows
  - Market capitalization
- Maintains historical compliance data
- Exports data to both database and Excel formats
- Implements multiple web scraping fallback methods

## 3. Mutual Funds Favorite Stocks Tracker (`MutualFundsFavourite.py`)
- Tracks stocks favored by mutual funds in Pakistan
- Collects data on:
  - Number of funds invested in each stock
  - Total rupee value invested
  - Investment trends over time
- Features robust web scraping with multiple fallback methods
- Exports data to both database and CSV formats
- Implements comprehensive error handling and logging

## 4. PSX Dividend Schedule Tracker (`psx_dividend_schedule.py`)
- Monitors and collects dividend announcements from PSX
- Tracks key dividend information:
  - Announcement dates
  - Book closure periods
  - Dividend amounts and types
  - Right shares information
- Features automated Telegram notifications for new announcements
- Maintains historical dividend data
- Implements data validation and error handling
- Exports data in multiple formats (CSV, JSON)

## 5. PSX Stock Analysis Tool (`psx_stock_analysis_tool_kmi30_adv.py`)
- Main orchestrator script that coordinates execution of all data collection modules
- Manages dependencies and ensures all required packages are installed
- Executes scripts in specific order:
  1. KMI Shariah data processing
  2. Mutual funds favorites tracking
  3. Market data download
  4. Data cleaning and deduplication
  5. Technical indicator calculations
  6. Dividend schedule updates
  7. RSI analysis
  8. Technical chart generation
- Implements logging and error handling for the entire pipeline
- Ensures data integrity across different modules

## Common Features Across Modules
1. **Robust Web Scraping**:
   - Multiple fallback methods for data collection
   - Retry mechanisms with exponential backoff
   - User agent rotation and proxy support

2. **Data Validation & Storage**:
   - Comprehensive data validation before storage
   - SQLite database integration
   - Multiple export formats (CSV, JSON, Excel)

3. **Error Handling & Logging**:
   - Detailed logging of all operations
   - Graceful error handling
   - Data integrity checks

4. **Automation & Notifications**:
   - Automated data collection
   - Telegram integration for notifications
   - Scheduled execution support

This collection of modules forms a comprehensive system for collecting, processing, and analyzing PSX market data, with a focus on technical analysis and fundamental data tracking.

# PSX Configuration Dashboard Module

## Core Functionalities

1. **Configuration Management**:
   - Loads and saves configuration settings from/to JSON files
   - Provides default configuration values for all system parameters
   - Implements configuration versioning and backup
   - Allows import/export of configuration settings

2. **Database & Path Settings**:
   - Configures paths for main and signals databases
   - Sets up output directories for charts and dashboards
   - Manages report folder locations and templates
   - Validates path existence and permissions

3. **Analysis Parameters Configuration**:
   - RSI thresholds (oversold/overbought levels)
   - Market phase neutral thresholds
   - Maximum holding period settings
   - Technical indicator weights customization
   - Score calculation parameters

4. **Visualization Settings**:
   - Chart and dashboard dimension controls
   - Color scheme customization for different stock statuses
   - Phase color configuration
   - Plot style and layout settings

5. **Market Condition Controls**:
   - Threshold settings for market classifications
   - Portfolio allocation targets for different market conditions
   - Risk management parameters
   - Market phase transition settings

6. **Stock Symbol Management**:
   - KMI30 symbol list maintenance
   - Symbol validation and verification
   - Custom symbol group management
   - Symbol metadata configuration

7. **Output Visualization**:
   - Real-time preview of generated charts
   - Dashboard visualization browser
   - Report viewer with multiple format support
   - File management and organization

## Interface Features

1. **Interactive Dashboard**:
   - Multiple tabbed interface for organized settings
   - Real-time validation of inputs
   - Visual feedback for configuration changes
   - Responsive layout design

2. **File Management**:
   - File browser for charts and reports
   - Preview capabilities for various file formats
   - Bulk file operations
   - Export/import functionality

3. **Error Handling**:
   - Input validation and error checking
   - Configuration integrity verification
   - Path existence validation
   - Data type verification

4. **User Feedback**:
   - Success/error notifications
   - Configuration status updates
   - Last saved timestamps
   - Change tracking

This module serves as the central configuration management system for the PSX Stock Trading Predictor, providing a user-friendly interface for customizing all aspects of the analysis and visualization pipeline.

## Project Configuration Structure

The project's configuration system is organized as follows:

1. **Main Configuration File Location**:
   ```
   /Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/config/config.py
   ```

2. **Configuration Hierarchy**:
   - Main configuration file (`config.py`) in the config directory
   - Configuration dashboard (`config_dashboard.py`) for UI-based management
   - Local configuration overrides (if any)

3. **Configuration Access**:
   - All modules reference the main config.py file
   - Changes through the dashboard are saved to this central location
   - Configuration is loaded at runtime by each module

4. **Default Configuration Path**:
   - Base Path: `/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/`
   - Config Directory: `config/`
   - Main Config File: `config.py`

This centralized configuration approach ensures consistency across all modules and provides a single source of truth for application settings.
