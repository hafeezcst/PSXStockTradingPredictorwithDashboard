"""
PSX Stock Indicator Calculator - GUI Version

Windows-based GUI application for calculating technical indicators from PSX stock data.
Provides intuitive controls, progress feedback, and comprehensive error handling.
"""
import sys
import logging
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QLineEdit, QPushButton, QComboBox, QProgressBar,
                            QTextEdit, QGroupBox, QFileDialog, QMessageBox, QDialog,
                            QRadioButton, QCheckBox, QDialogButtonBox, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
import pandas as pd
from sqlalchemy import create_engine, inspect
from tqdm import tqdm
from src.data_processing.fix_pandas_ta import ta

class ProcessingThread(QThread):
    """Worker thread for database processing to prevent GUI freezing"""
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, data_reader, table_names):
        super().__init__()
        self.data_reader = data_reader
        self.table_names = table_names
        self._is_running = True

    def run(self):
        try:
            total_tables = len(self.table_names)
            for i, table_name in enumerate(self.table_names):
                if not self._is_running:
                    break

                self.log_message.emit(f"Processing table: {table_name}")
                data = self.data_reader.read_data(table_name)
                processed_data = self.data_reader.preprocess(data)
                self.data_reader.save_to_db(processed_data, table_name)
                
                progress = int((i + 1) / total_tables * 100)
                self.progress.emit(progress)

            self.finished.emit(True)
        except Exception as e:
            self.log_message.emit(f"Error in processing thread: {str(e)}")
            self.finished.emit(False)

    def stop(self):
        self._is_running = False

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSX Stock Indicator Calculator")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize DataReader with default paths
        self.data_reader = None
        self.processing_thread = None
        
        self.init_ui()
        self.init_logging()
        
    def init_ui(self):
        """Initialize UI components"""
        # Create menu bar
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        settings_action = QAction("&Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.setStatusTip("Configure application settings")
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.setStatusTip("Show application information")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        docs_action = QAction("&Documentation", self)
        docs_action.setStatusTip("Open documentation")
        docs_action.triggered.connect(self.show_docs)
        help_menu.addAction(docs_action)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Main content
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Database Connection Group
        db_group = QGroupBox("Database Connection")
        db_layout = QVBoxLayout()
        
        # Source DB
        source_layout = QHBoxLayout()
        source_label = QLabel("Source DB:")
        source_label.setToolTip("Path to the source SQLite database containing raw stock data")
        source_layout.addWidget(source_label)
        
        self.source_db_input = QLineEdit()
        self.source_db_input.setPlaceholderText("Path to source database")
        self.source_db_input.setToolTip("Enter path to source database or click Browse to select")
        source_layout.addWidget(self.source_db_input)
        
        self.source_browse_btn = QPushButton("Browse...")
        self.source_browse_btn.setToolTip("Browse for source database file")
        self.source_browse_btn.clicked.connect(self.browse_source_db)
        source_layout.addWidget(self.source_browse_btn)
        db_layout.addLayout(source_layout)
        
        # Target DB
        target_layout = QHBoxLayout()
        target_label = QLabel("Target DB:")
        target_label.setToolTip("Path where processed data with indicators will be saved")
        target_layout.addWidget(target_label)
        
        self.target_db_input = QLineEdit()
        self.target_db_input.setPlaceholderText("Path to target database")
        self.target_db_input.setToolTip("Enter path for target database or click Browse to select")
        target_layout.addWidget(self.target_db_input)
        
        self.target_browse_btn = QPushButton("Browse...")
        self.target_browse_btn.setToolTip("Browse for target database location")
        self.target_browse_btn.clicked.connect(self.browse_target_db)
        target_layout.addWidget(self.target_browse_btn)
        db_layout.addLayout(target_layout)
        
        # Connect Button
        self.connect_btn = QPushButton("Connect to Databases")
        self.connect_btn.setToolTip("Establish connection to both source and target databases")
        self.connect_btn.clicked.connect(self.connect_databases)
        db_layout.addWidget(self.connect_btn)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)
        
        # Processing Group
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout()
        
        # Table Selection
        table_layout = QHBoxLayout()
        table_label = QLabel("Select Tables:")
        table_label.setToolTip("Tables available in the source database")
        table_layout.addWidget(table_label)
        
        self.table_combo = QComboBox()
        self.table_combo.setPlaceholderText("Connect to database first")
        self.table_combo.setToolTip("Select one or more tables to process (hold Ctrl to select multiple)")
        self.table_combo.setEnabled(False)
        table_layout.addWidget(self.table_combo)
        process_layout.addLayout(table_layout)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        process_layout.addWidget(self.progress_bar)
        
        # Process Button
        self.process_btn = QPushButton("Process Selected Tables")
        self.process_btn.setToolTip("Calculate indicators for selected tables and save to target database")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_tables)
        process_layout.addWidget(self.process_btn)
        
        # Stop Button
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setToolTip("Cancel current processing operation")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        process_layout.addWidget(self.stop_btn)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # Log Display
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        
        # Set default paths
        self.set_default_paths()
        
    def set_default_paths(self):
        """Set default database paths"""
        current_dir = os.getcwd()
        default_source = os.path.join(current_dir, 'data/databases/production/psx_consolidated_data_PSX.db')
        default_target = os.path.join(current_dir, 'data/databases/production/psx_consolidated_data_indicators_PSX.db')
        self.source_db_input.setText(default_source)
        self.target_db_input.setText(default_target)
        
    def browse_source_db(self):
        """Browse for source database file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Source Database", "", "SQLite Database (*.db *.sqlite)")
        if path:
            self.source_db_input.setText(path)
            
    def browse_target_db(self):
        """Browse for target database file"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Select Target Database", "", "SQLite Database (*.db *.sqlite)")
        if path:
            if not path.endswith('.db'):
                path += '.db'
            self.target_db_input.setText(path)
            
    def connect_databases(self):
        """Connect to source and target databases"""
        source_path = self.source_db_input.text()
        target_path = self.target_db_input.text()
        
        if not source_path or not target_path:
            QMessageBox.warning(self, "Error", "Please specify both source and target database paths")
            return
            
        try:
            self.data_reader = DataReader(source_path, target_path)
            tables = self.data_reader.get_table_names()
            
            self.table_combo.clear()
            self.table_combo.addItems(tables)
            self.table_combo.setEnabled(True)
            self.process_btn.setEnabled(True)
            
            self.log_message(f"Successfully connected to databases\nSource: {source_path}\nTarget: {target_path}")
            self.log_message(f"Found {len(tables)} tables in source database")
            
        except Exception as e:
            self.log_message(f"Error connecting to databases: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to connect to databases:\n{str(e)}")
            
    def process_tables(self):
        """Process selected tables"""
        if not self.data_reader:
            QMessageBox.warning(self, "Error", "Please connect to databases first")
            return
            
        selected_tables = [self.table_combo.itemText(i) for i in range(self.table_combo.count())]
        
        if not selected_tables:
            QMessageBox.warning(self, "Error", "No tables selected for processing")
            return
            
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.processing_thread = ProcessingThread(self.data_reader, selected_tables)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.log_message.connect(self.log_message)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
        
    def stop_processing(self):
        """Stop the processing thread"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.log_message("Processing stopped by user")
            
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def processing_finished(self, success):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.log_message("Processing completed successfully")
            QMessageBox.information(self, "Success", "Processing completed successfully")
        else:
            self.log_message("Processing completed with errors")
            
    def log_message(self, message):
        """Add message to log display"""
        self.log_display.append(message)
        self.statusBar().showMessage(message.split('\n')[0], 5000)  # Show first line for 5 seconds

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About PSX Indicator Calculator",
                         "PSX Stock Indicator Calculator\n"
                         "Version 1.0\n\n"
                         "GUI application for calculating technical indicators\n"
                         "from Pakistan Stock Exchange data.")

    def show_docs(self):
        """Open documentation in browser"""
        QMessageBox.information(self, "Documentation",
                              "Documentation is available at:\n"
                              "https://github.com/your-repo/docs")

    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec_():
            # Apply settings
            self.apply_settings(dialog.get_settings())

    def apply_settings(self, settings):
        """Apply settings from dialog"""
        if 'theme' in settings:
            self.set_theme(settings['theme'])
        # Add other setting applications here

    def set_theme(self, theme_name):
        """Set application theme"""
        if theme_name == 'dark':
            self.setStyleSheet("""
                QMainWindow, QDialog {
                    background-color: #2D2D2D;
                    color: #FFFFFF;
                }
                QPushButton {
                    background-color: #3A3A3A;
                    border: 1px solid #5A5A5A;
                    padding: 5px;
                }
            """)
        else:
            self.setStyleSheet("")  # Reset to default light theme
        
    def init_logging(self):
        """Initialize logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[LogHandler(self.log_message)]
        )
        
    def closeEvent(self, event):
        """Handle window close event"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Processing Running',
                'Processing is still running. Are you sure you want to quit?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.processing_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

class LogHandler(logging.Handler):
    """Custom logging handler to forward logs to GUI"""
    def __init__(self, emit_signal):
        super().__init__()
        self.emit_signal = emit_signal
        
    def emit(self, record):
        msg = self.format(record)
        self.emit_signal.emit(msg)

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
            
            # Weekly high and low
            data['weekly_high'] = data['High'].resample('W-FRI').max()
            data['weekly_low'] = data['Low'].resample('W-FRI').min()
            
            # Monthly high and low
            data['monthly_high'] = data['High'].resample('ME').max()
            data['monthly_low'] = data['Low'].resample('ME').min()
              
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")

        return data

    def save_to_db(self, data: pd.DataFrame, table_name: str):
        """Save processed data to target database.
        
        Args:
            data: Processed DataFrame to save
            table_name: Name of the target table
        """
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

class SettingsDialog(QDialog):
    """Dialog for configuring application settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.settings = {}
        self.init_ui()

    def init_ui(self):
        """Initialize settings UI components"""
        layout = QVBoxLayout()

        # Theme Selection
        theme_group = QGroupBox("Interface Theme")
        theme_layout = QVBoxLayout()
        
        self.theme_light = QRadioButton("Light Theme")
        self.theme_dark = QRadioButton("Dark Theme")
        self.theme_system = QRadioButton("System Default")
        
        theme_layout.addWidget(self.theme_light)
        theme_layout.addWidget(self.theme_dark)
        theme_layout.addWidget(self.theme_system)
        theme_group.setLayout(theme_layout)

        # Processing Options
        process_group = QGroupBox("Processing Options")
        process_layout = QVBoxLayout()
        
        self.parallel_processing = QCheckBox("Enable parallel processing")
        self.parallel_processing.setToolTip("Process multiple tables simultaneously for better performance")
        
        self.enable_caching = QCheckBox("Enable result caching")
        self.enable_caching.setToolTip("Cache processed results for faster subsequent runs")
        
        process_layout.addWidget(self.parallel_processing)
        process_layout.addWidget(self.enable_caching)
        process_group.setLayout(process_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(theme_group)
        layout.addWidget(process_group)
        layout.addWidget(button_box)
        self.setLayout(layout)

        # Load current settings
        self.load_settings()

    def load_settings(self):
        """Load settings from persistent storage"""
        settings = QSettings("PSXIndicator", "GUI")
        
        theme = settings.value("theme", "light")
        if theme == "light":
            self.theme_light.setChecked(True)
        elif theme == "dark":
            self.theme_dark.setChecked(True)
        else:
            self.theme_system.setChecked(True)
            
        self.parallel_processing.setChecked(
            settings.value("parallel_processing", True, type=bool))
        self.enable_caching.setChecked(
            settings.value("enable_caching", True, type=bool))

    def get_settings(self):
        """Get current settings from dialog and save to persistent storage"""
        settings = QSettings("PSXIndicator", "GUI")
        
        if self.theme_light.isChecked():
            settings.setValue("theme", "light")
        elif self.theme_dark.isChecked():
            settings.setValue("theme", "dark")
        else:
            settings.setValue("theme", "system")
            
        settings.setValue("parallel_processing",
                         self.parallel_processing.isChecked())
        settings.setValue("enable_caching",
                         self.enable_caching.isChecked())
        
        return {
            'theme': settings.value("theme"),
            'parallel': settings.value("parallel_processing", type=bool),
            'cache': settings.value("enable_caching", type=bool)
        }

    def load_settings(self):
        """Load current settings into UI controls"""
        # Defaults
        self.theme_light.setChecked(True)
        self.parallel_processing.setChecked(True)
        self.enable_caching.setChecked(True)

    def get_settings(self):
        """Get current settings from dialog"""
        settings = {}
        
        if self.theme_light.isChecked():
            settings['theme'] = 'light'
        elif self.theme_dark.isChecked():
            settings['theme'] = 'dark'
        else:
            settings['theme'] = 'system'
            
        settings['parallel'] = self.parallel_processing.isChecked()
        settings['cache'] = self.enable_caching.isChecked()
        
        return settings

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())