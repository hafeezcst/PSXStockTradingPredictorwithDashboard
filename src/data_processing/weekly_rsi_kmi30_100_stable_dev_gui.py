import sys
import logging
from logging.handlers import RotatingFileHandler
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                            QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                            QLineEdit, QProgressBar, QTextEdit, QTableWidget,
                            QTableWidgetItem, QFileDialog, QMessageBox, QComboBox,
                            QGroupBox, QPlainTextEdit, QCheckBox, QListWidget, QListWidgetItem,
                            QCompleter, QSplitter, QSizePolicy, QFrame, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QRunnable, QThreadPool, QTimer, QUrl, QEvent
from PyQt5.QtGui import QIcon, QDesktopServices, QMovie, QFont
from PyQt5.QtCore import QCoreApplication
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
import pandas as pd
import json
import os
from pathlib import Path
import requests
from packaging import version
from sqlalchemy import text
from collections import deque
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Import DataReader from numbered file using importlib
import importlib.util
import os
module_path = os.path.join(os.path.dirname(__file__), "01-PSX_Database_data_download_to_SQL_db_PSX.py")
spec = importlib.util.spec_from_file_location("psx_data_reader", module_path)
psx_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psx_module)
DataReader = psx_module.DataReader

import tempfile  # <-- Add this at the top level, not inside any function

class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    data = pyqtSignal(pd.DataFrame)

class BatchWorker(QRunnable):
    """Worker for batch download operations"""
    def __init__(self, data_reader, symbol):
        super().__init__()
        self.data_reader = data_reader
        self.symbol = symbol
        self.signals = WorkerSignals()
        self.logger = logging.getLogger('PSXDownloader')

    def run(self):
        try:
            end_date = date.today() - timedelta(days=1)
            last_date = self.data_reader.get_last_date(self.symbol)
            if last_date and last_date >= end_date:
                msg = f"{self.symbol} is already up to date (no new data to download). Last date in DB: {last_date}."
                self.signals.status.emit(msg)
                self.logger.info(msg, extra={
                    'context': {
                        'symbol': self.symbol,
                        'last_date': str(last_date),
                        'operation': 'up_to_date'
                    }
                })
                self.signals.status.emit(f"BATCH_NODATA:{self.symbol}")
                return
            start_date = last_date + timedelta(days=1) if last_date else date(2000, 1, 1)
            self.signals.status.emit(f"Downloading {self.symbol} from {start_date} to {end_date}...")
            self.logger.info(f"Starting download for {self.symbol}", extra={
                'context': {
                    'symbol': self.symbol,
                    'start_date': str(start_date),
                    'end_date': str(end_date)
                }
            })
            data = self.data_reader.stocks(self.symbol, start_date, end_date)
            if not data.empty:
                self.logger.debug(f"Saving {len(data)} records for {self.symbol}")
                self.data_reader.save_to_db(data, f'PSX_{self.symbol}_stock_data')
                self.signals.progress.emit(1)
                first_day = data.index.min().strftime('%Y-%m-%d') if not data.empty else 'N/A'
                last_day = data.index.max().strftime('%Y-%m-%d') if not data.empty else 'N/A'
                msg = (f"Completed {self.symbol}. Downloaded {len(data)} records. "
                       f"Start: {first_day}, End: {last_day}")
                self.signals.status.emit(msg)
                self.logger.info(msg)
                self.signals.status.emit(f"BATCH_SUCCESS:{self.symbol}")
            elif last_date and start_date > end_date:
                msg = f"{self.symbol} is already up to date (no new data to download). Last date in DB: {last_date}."
                self.signals.status.emit(msg)
                self.logger.info(msg, extra={
                    'context': {
                        'symbol': self.symbol,
                        'last_date': str(last_date),
                        'operation': 'up_to_date'
                    }
                })
                self.signals.status.emit(f"BATCH_NODATA:{self.symbol}")
            else:
                msg = f"No data available for {self.symbol} in the selected date range ({start_date} to {end_date})."
                self.signals.error.emit(msg)
                self.logger.warning(msg, extra={
                    'context': {
                        'symbol': self.symbol,
                        'date_range': f"{start_date} to {end_date}",
                        'operation': 'no_data'
                    }
                })
                self.signals.status.emit(f"BATCH_NODATA:{self.symbol}")
        except Exception as e:
            msg = f"Error with {self.symbol}: {str(e)}"
            self.signals.error.emit(msg)
            self.logger.error(msg, exc_info=True, extra={
                'context': {
                    'symbol': self.symbol,
                    'operation': 'download'
                }
            })
            self.signals.status.emit(f"BATCH_FAILED:{self.symbol}")

class WorkerThread(QThread):
    def __init__(self, data_reader, symbol, start_date, end_date):
        super().__init__()
        self.data_reader = data_reader
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.signals = WorkerSignals()
        self.logger = logging.getLogger('PSXDownloader')

    def run(self):
        try:
            self.signals.status.emit(f"Downloading data for {self.symbol} from {self.start_date} to {self.end_date}...")
            logger = logging.getLogger('PSXDownloader')
            logger.info(f"Starting single download for {self.symbol}", extra={
                'context': {
                    'symbol': self.symbol,
                    'start_date': str(self.start_date),
                    'end_date': str(self.end_date)
                }
            })
            last_date = self.data_reader.get_last_date(self.symbol)
            if last_date and last_date >= self.end_date:
                msg = f"{self.symbol} is already up to date (no new data to download). Last date in DB: {last_date}."
                self.signals.status.emit(msg)
                self.logger.info(msg, extra={
                    'context': {
                        'symbol': self.symbol,
                        'last_date': str(last_date),
                        'operation': 'up_to_date'
                    }
                })
                return
            data = self.data_reader.stocks(self.symbol, self.start_date, self.end_date)
            if not data.empty:
                self.logger.debug(f"Saving {len(data)} records for {self.symbol}")
                self.data_reader.save_to_db(data, f'PSX_{self.symbol}_stock_data')
                self.signals.data.emit(data)
                first_day = data.index.min().strftime('%Y-%m-%d') if not data.empty else 'N/A'
                last_day = data.index.max().strftime('%Y-%m-%d') if not data.empty else 'N/A'
                msg = (f"Successfully downloaded {self.symbol}. Records: {len(data)}. "
                       f"Start: {first_day}, End: {last_day}")
                self.signals.status.emit(msg)
                self.logger.info(f"Completed download for {self.symbol}")
            elif self.data_reader.get_last_date(self.symbol) and self.start_date > self.end_date:
                msg = f"{self.symbol} is already up to date (no new data to download). Last date in DB: {self.data_reader.get_last_date(self.symbol)}."
                self.signals.status.emit(msg)
                self.logger.info(msg, extra={
                    'context': {
                        'symbol': self.symbol,
                        'last_date': str(self.data_reader.get_last_date(self.symbol)),
                        'operation': 'up_to_date'
                    }
                })
            else:
                msg = f"No data available for {self.symbol} in the selected date range ({self.start_date} to {self.end_date})."
                self.signals.error.emit(msg)
                self.logger.warning(msg, extra={
                    'context': {
                        'symbol': self.symbol,
                        'date_range': f"{self.start_date} to {self.end_date}",
                        'operation': 'no_data'
                    }
                })
        except Exception as e:
            msg = f"Error downloading {self.symbol}: {str(e)}"
            self.signals.error.emit(msg)
            logger = logging.getLogger('PSXDownloader')
            logger.error(msg, exc_info=True, extra={
                'context': {
                    'symbol': self.symbol,
                    'operation': 'single download'
                }
            })

class ConfigManager:
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())  # Prevent "No handlers" warnings
        else:
            self.logger = logger
        self.config_path = os.path.join(os.path.dirname(__file__), 'psx_gui_config.json')
        self.default_config = {
            'db_path': 'data/databases/production/PSX_consolidated_data_PSX.db',
            'alt_db_path': 'data/databases/production/PSX_consolidated_data_PSX_Alternative.db',
            'indicators_db_path': 'data/databases/production/psx_consolidated_data_indicators_PSX.db',
            'signals_db_path': 'data/databases/production/PSX_investing_Stocks_KMI100.db',
            'symbols_file': 'data/databases/production/psxsymbols.xlsx',
            'theme': 'dark',
            'export_path': 'exports',
            'max_threads': 4,
            'check_updates': True,
            'last_update_check': None
        }

    def load_config(self):
        try:
            config = self.default_config.copy()  # Start with defaults
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    saved_config = json.load(f)
                    # Update defaults with saved values, preserving any new defaults
                    config.update(saved_config)
                    self.logger.debug("Loaded config file", extra={
                        'context': {
                            'path': self.config_path,
                            'saved_config': saved_config,
                            'merged_config': config
                        }
                    })
            else:
                self.logger.info("Using default config")
                
            return config
        except Exception as e:
            self.logger.error("Error loading config", exc_info=True, extra={
                'context': {
                    'path': self.config_path,
                    'default_config': self.default_config
                }
            })
            return self.default_config

    def save_config(self, config):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.info("Saved config file", extra={
                'context': {
                    'path': self.config_path,
                    'config': config
                }
            })
            return True
        except Exception as e:
            self.logger.error("Error saving config", exc_info=True, extra={
                'context': {
                    'path': self.config_path,
                    'config': config
                }
            })
            return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSX Data Downloader")
        self.setGeometry(100, 100, 1200, 800)
        # Create basic logger first
        self.logger = logging.getLogger('PSXDownloader')
        # Initialize config manager with basic logger
        self.config_manager = ConfigManager(self.logger)
        self.config = self.config_manager.load_config()
        # --- Create main widget and layout FIRST ---
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        # Now setup full logging with config
        self.setup_logging()
        # Initialize DataReader
        self.data_reader = DataReader(
            db_path=self.config['db_path'],
            alt_db_path=self.config['alt_db_path']
        )

        # Load symbols file to get first symbol
        self.first_symbol = None
        try:
            symbols_df = pd.read_excel(self.config['symbols_file'], sheet_name='KMIALL')
            valid_symbols = [s for s in symbols_df.iloc[:, 0].tolist() if isinstance(s, str) and s.isalpha()]
            if valid_symbols:
                self.first_symbol = valid_symbols[0]
        except Exception as e:
            self.logger.warning(f"Could not load symbols file: {str(e)}")
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Create tabs in new order
        self.create_download_tab()
        self.create_signals_tab()
        self.create_query_tab()
        self.create_db_health_tab()
        self.create_export_tab()
        self.create_config_tab()
        self.create_help_tab()
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar, 1)
        self.progress_bar.setVisible(False)
        
        # Apply theme
        self.apply_theme(self.config['theme'])
        
        # Add counters for batch summary
        self.batch_success = 0
        self.batch_no_data = 0
        self.batch_skipped = 0
        self.batch_failed = 0
        self.batch_total = 0
        self.batch_processed = 0
        
    def setup_logging(self):
        """Configure enhanced logging with rotation and UI integration"""
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'psx_downloader.log')
        # Create logger
        self.logger = logging.getLogger('PSXDownloader')
        self.logger.setLevel(logging.DEBUG)
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1024*1024,  # 1MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        # Add handler to logger
        self.logger.addHandler(file_handler)
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        # Log startup message
        self.logger.info("Application started", extra={
            'context': {
                'version': '1.0.0',
                'config': self.config
            }
        })
        # --- Error Log Panel ---
        if not hasattr(self, 'error_log_panel'):
            self.error_log_panel = QTextEdit()
            self.error_log_panel.setReadOnly(True)
            self.error_log_panel.setMaximumHeight(80)
            self.error_log_panel.setStyleSheet("background: #330; color: #f88; font-size: 10pt;")
            self.main_layout.addWidget(self.error_log_panel)
        # Add handler for error log
        class QTextEditHandler(logging.Handler):
            def __init__(self, widget):
                super().__init__()
                self.widget = widget
            def emit(self, record):
                msg = self.format(record)
                if record.levelno >= logging.WARNING:
                    # Only append if widget still exists and is not deleted
                    if self.widget is not None and hasattr(self.widget, 'append'):
                        try:
                            self.widget.append(msg)
                        except RuntimeError:
                            # Widget was deleted, ignore
                            pass
        error_handler = QTextEditHandler(self.error_log_panel)
        error_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)

    def create_download_tab(self):
        self.download_tab = QWidget()
        self.tab_widget.addTab(self.download_tab, "PSX Data")
        layout = QVBoxLayout()
        self.download_tab.setLayout(layout)
        # --- Download History Panel ---
        self.download_history_panel = QTextEdit()
        self.download_history_panel.setReadOnly(True)
        self.download_history_panel.setMaximumHeight(100)
        self.download_history_panel.setVisible(False)
        layout.addWidget(self.download_history_panel)
        self.toggle_history_button = QPushButton("Show Download History")
        self.toggle_history_button.setCheckable(True)
        self.toggle_history_button.setToolTip("Show/hide recent download history")
        self.toggle_history_button.toggled.connect(lambda checked: self.download_history_panel.setVisible(checked))
        layout.addWidget(self.toggle_history_button)
        # Symbol selection
        symbol_layout = QHBoxLayout()
        symbol_label = QLabel("Symbol:")
        symbol_label.setToolTip("Select the stock symbol to download")
        symbol_layout.addWidget(symbol_label)
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.setInsertPolicy(QComboBox.NoInsert)
        self.symbol_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.symbol_combo.setToolTip("Type or select a stock symbol")
        symbol_layout.addWidget(self.symbol_combo)
        self.reload_symbols_button = QPushButton("Reload Symbols")
        self.reload_symbols_button.setToolTip("Reload the list of available symbols")
        self.reload_symbols_button.clicked.connect(self.load_symbols_into_combo)
        symbol_layout.addWidget(self.reload_symbols_button)
        # Add Preview Available Data button
        self.preview_button = QPushButton("Preview Available Data")
        self.preview_button.setToolTip("Preview data for the selected symbol and date range")
        self.preview_button.clicked.connect(self.preview_available_data)
        symbol_layout.addWidget(self.preview_button)
        # Add Refresh Preview button
        self.refresh_preview_button = QPushButton("Refresh Preview")
        self.refresh_preview_button.setToolTip("Refresh the data preview table")
        self.refresh_preview_button.clicked.connect(self.preview_available_data)
        symbol_layout.addWidget(self.refresh_preview_button)
        layout.addLayout(symbol_layout)
        # Date range
        date_layout = QHBoxLayout()
        start_label = QLabel("Start Date:")
        start_label.setToolTip("Enter the start date (YYYY-MM-DD)")
        date_layout.addWidget(start_label)
        self.start_date_input = QLineEdit()
        self.start_date_input.setPlaceholderText("YYYY-MM-DD")
        self.start_date_input.setToolTip("Enter the start date for download (YYYY-MM-DD)")
        fixed_start_date = '2020-01-01'
        today_str = date.today().strftime('%Y-%m-%d')
        self.start_date_input.setText(fixed_start_date)
        date_layout.addWidget(self.start_date_input)
        end_label = QLabel("End Date:")
        end_label.setToolTip("Enter the end date (YYYY-MM-DD)")
        date_layout.addWidget(end_label)
        self.end_date_input = QLineEdit()
        self.end_date_input.setPlaceholderText("YYYY-MM-DD")
        self.end_date_input.setToolTip("Enter the end date for download (YYYY-MM-DD)")
        self.end_date_input.setText(today_str)
        date_layout.addWidget(self.end_date_input)
        layout.addLayout(date_layout)
        # Connect symbol/date changes to preview refresh
        self.symbol_combo.currentIndexChanged.connect(self.preview_available_data)
        self.symbol_combo.lineEdit().editingFinished.connect(self.preview_available_data)
        self.start_date_input.editingFinished.connect(self.preview_available_data)
        self.end_date_input.editingFinished.connect(self.preview_available_data)
        # Buttons
        button_layout = QHBoxLayout()
        self.download_button = QPushButton("Download")
        self.download_button.setToolTip("Download data for the selected symbol and date range")
        self.download_button.clicked.connect(self.start_download)
        button_layout.addWidget(self.download_button)
        self.batch_button = QPushButton("Batch Download")
        self.batch_button.setToolTip("Download data for all symbols in the list")
        self.batch_button.clicked.connect(self.start_batch_download)
        button_layout.addWidget(self.batch_button)
        layout.addLayout(button_layout)
        # Data preview
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(6)
        self.preview_table.setHorizontalHeaderLabels(["Date", "Open", "High", "Low", "Close", "Volume"])
        self.preview_table.setSortingEnabled(True)
        self.preview_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStyleSheet("font-weight: bold;")
        self.preview_table.setToolTip("Preview of the downloaded data")
        # Add filter controls
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        filter_label.setToolTip("Filter rows in the preview table")
        filter_layout.addWidget(filter_label)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter text to filter rows...")
        self.filter_input.setToolTip("Type to filter the preview table")
        self.filter_input.textChanged.connect(self.filter_table)
        filter_layout.addWidget(self.filter_input)
        clear_button = QPushButton("Clear")
        clear_button.setToolTip("Clear the filter and show all rows")
        clear_button.clicked.connect(self.clear_filters)
        filter_layout.addWidget(clear_button)
        layout.addLayout(filter_layout)
        layout.addWidget(self.preview_table)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setToolTip("Log of download and preview operations")
        layout.addWidget(self.log_area)
        # --- Statistics & Analysis Panel ---
        self.analysis_panel = QWidget()
        analysis_layout = QVBoxLayout()
        self.analysis_panel.setLayout(analysis_layout)
        self.stats_label = QLabel("<b>Statistics:</b> No data loaded.")
        self.stats_label.setToolTip("Summary statistics for the previewed data")
        analysis_layout.addWidget(self.stats_label)
        self.analysis_figure = plt.Figure(figsize=(5, 2.5))
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        analysis_layout.addWidget(self.analysis_canvas)
        self.export_chart_button = QPushButton("Export Chart as PNG")
        self.export_chart_button.setToolTip("Export the statistics chart as a PNG image")
        self.export_chart_button.clicked.connect(self.export_analysis_chart)
        analysis_layout.addWidget(self.export_chart_button)
        layout.addWidget(self.analysis_panel)
        # --- Progress Spinner Overlay ---
        self.spinner_overlay = QLabel(self.download_tab)
        self.spinner_overlay.setAlignment(Qt.AlignCenter)
        self.spinner_overlay.setStyleSheet("background: rgba(30,30,30,0.5); border-radius: 10px;")
        self.spinner_overlay.setVisible(False)
        self.spinner_movie = QMovie(os.path.join(os.path.dirname(__file__), "spinner.gif"))
        self.spinner_overlay.setMovie(self.spinner_movie)
        self.spinner_overlay.setFixedSize(120, 120)
        self.spinner_overlay.move(self.download_tab.width()//2-60, self.download_tab.height()//2-60)
        self.download_tab.resizeEvent = lambda event: self.spinner_overlay.move(self.download_tab.width()//2-60, self.download_tab.height()//2-60)
        # --- Auto-complete for symbol input ---
        self.load_symbols_into_combo()
        if self.symbol_combo.count() > 0:
            self.symbol_combo.setCurrentIndex(0)
        completer = QCompleter([self.symbol_combo.itemText(i) for i in range(self.symbol_combo.count())])
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.symbol_combo.setCompleter(completer)
        
    def load_symbols_into_combo(self):
        self.symbol_combo.clear()
        symbols = []
        # Try to load from database first
        try:
            with self.data_reader.engine.connect() as conn:
                result = conn.execute(text(r"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'PSX\_%\_stock\_data'"))
                for row in result.fetchall():
                    tname = row[0]
                    if tname.startswith('PSX_') and tname.endswith('_stock_data'):
                        symbol = tname[len('PSX_'):-len('_stock_data')]
                        if symbol.isalpha():
                            symbols.append(symbol)
        except Exception as e:
            self.logger.warning(f"Could not load symbols from DB: {e}")
        # If DB fails, fallback to symbols file
        if not symbols:
            try:
                symbols_df = pd.read_excel(self.config['symbols_file'], sheet_name='KMIALL')
                symbols = [str(s) for s in symbols_df.iloc[:, 0].tolist() if isinstance(s, str) and s.isalpha()]
            except Exception as e:
                self.logger.warning(f"Could not load symbols from file: {e}")
        self.symbol_combo.addItems(sorted(set(symbols)))

    def preview_available_data(self):
        symbol = self.symbol_combo.currentText().strip().upper()
        if not symbol or not symbol.isalpha():
            self.show_error("Please select a valid stock symbol (letters only)")
            self.preview_table.setRowCount(0)
            return
        # Parse date range
        try:
            start_date = datetime.strptime(self.start_date_input.text(), "%Y-%m-%d").date()
        except Exception:
            start_date = None
        try:
            end_date = datetime.strptime(self.end_date_input.text(), "%Y-%m-%d").date()
        except Exception:
            end_date = None
        if not start_date or not end_date or start_date > end_date:
            self.show_error("Please enter a valid date range")
            self.preview_table.setRowCount(0)
            return
        try:
            table_name = f'PSX_{symbol}_stock_data'
            with self.data_reader.engine.connect() as conn:
                query = text(f"SELECT * FROM {table_name} WHERE Date >= :start AND Date <= :end ORDER BY Date DESC")
                self.logger.info(f"Preview Query: {query} | Params: start={start_date}, end={end_date}")
                result = conn.execute(query, {"start": str(start_date), "end": str(end_date)})
                rows = result.fetchall()
                columns = result.keys()
                if not rows:
                    msg = f"No data found in the database for {symbol} in the selected date range."
                    if hasattr(self, 'log_area'):
                        self.log_area.append(msg)
                    self.preview_table.setRowCount(0)
                    return
                import pandas as pd
                df = pd.DataFrame(rows, columns=columns)
                # Do NOT set index, just pass df
                self.show_data(df)
                msg = f"Previewed {len(df)} records for {symbol} from {df['Date'].min()} to {df['Date'].max()}"
                if hasattr(self, 'log_area'):
                    self.log_area.append(msg)
        except Exception as e:
            if 'no such table' in str(e):
                msg = f"Table for symbol '{symbol}' does not exist in the database."
                self.show_error(msg)
                if hasattr(self, 'log_area'):
                    self.log_area.append(msg)
                self.preview_table.setRowCount(0)
            else:
                msg = f"Error previewing data for {symbol}: {str(e)}"
                self.show_error(msg)
                if hasattr(self, 'log_area'):
                    self.log_area.append(msg)
                self.preview_table.setRowCount(0)
        
    def create_config_tab(self):
        self.config_tab = QWidget()
        self.tab_widget.addTab(self.config_tab, "Configuration")
        layout = QVBoxLayout()
        self.config_tab.setLayout(layout)
        # --- DB Paths Group ---
        db_group = QGroupBox("Database Paths")
        db_layout = QVBoxLayout()
        db_group.setLayout(db_layout)
        # Main DB
        db_hbox = QHBoxLayout()
        db_hbox.addWidget(QLabel("Main DB Path:"))
        self.db_path_input = QLineEdit(self.config['db_path'])
        db_hbox.addWidget(self.db_path_input)
        browse_db_button = QPushButton("Browse")
        browse_db_button.clicked.connect(self.browse_db_path)
        db_hbox.addWidget(browse_db_button)
        test_main_db_button = QPushButton("Test Connection")
        test_main_db_button.clicked.connect(lambda: self.test_db_connection(self.db_path_input.text()))
        db_hbox.addWidget(test_main_db_button)
        db_layout.addLayout(db_hbox)
        # Alt DB
        alt_db_hbox = QHBoxLayout()
        alt_db_hbox.addWidget(QLabel("Alt DB Path:"))
        self.alt_db_path_input = QLineEdit(self.config['alt_db_path'])
        alt_db_hbox.addWidget(self.alt_db_path_input)
        browse_alt_db_button = QPushButton("Browse")
        browse_alt_db_button.clicked.connect(self.browse_alt_db_path)
        alt_db_hbox.addWidget(browse_alt_db_button)
        test_alt_db_button = QPushButton("Test Connection")
        test_alt_db_button.clicked.connect(lambda: self.test_db_connection(self.alt_db_path_input.text()))
        alt_db_hbox.addWidget(test_alt_db_button)
        db_layout.addLayout(alt_db_hbox)
        # Indicators DB
        indicators_db_hbox = QHBoxLayout()
        indicators_db_hbox.addWidget(QLabel("Indicators DB Path:"))
        self.indicators_db_path_input = QLineEdit(self.config.get('indicators_db_path', self.config_manager.default_config['indicators_db_path']))
        indicators_db_hbox.addWidget(self.indicators_db_path_input)
        browse_indicators_db_button = QPushButton("Browse")
        browse_indicators_db_button.clicked.connect(self.browse_indicators_db_path)
        indicators_db_hbox.addWidget(browse_indicators_db_button)
        test_indicators_db_button = QPushButton("Test Connection")
        test_indicators_db_button.clicked.connect(lambda: self.test_db_connection(self.indicators_db_path_input.text()))
        indicators_db_hbox.addWidget(test_indicators_db_button)
        db_layout.addLayout(indicators_db_hbox)
        # Signals DB
        signals_db_hbox = QHBoxLayout()
        signals_db_hbox.addWidget(QLabel("Signals DB Path:"))
        self.signals_db_path_input = QLineEdit(self.config.get('signals_db_path', self.config_manager.default_config['signals_db_path']))
        signals_db_hbox.addWidget(self.signals_db_path_input)
        browse_signals_db_button = QPushButton("Browse")
        browse_signals_db_button.clicked.connect(self.browse_signals_db_path)
        signals_db_hbox.addWidget(browse_signals_db_button)
        test_signals_db_button = QPushButton("Test Connection")
        test_signals_db_button.clicked.connect(lambda: self.test_db_connection(self.signals_db_path_input.text()))
        signals_db_hbox.addWidget(test_signals_db_button)
        db_layout.addLayout(signals_db_hbox)
        layout.addWidget(db_group)
        # --- Symbols Group ---
        symbols_group = QGroupBox("Symbols File")
        symbols_layout = QHBoxLayout()
        symbols_group.setLayout(symbols_layout)
        symbols_layout.addWidget(QLabel("Symbols File:"))
        self.symbols_file_input = QLineEdit(self.config.get('symbols_file', 'data/symbols/default_symbols.json'))
        symbols_layout.addWidget(self.symbols_file_input)
        browse_symbols_button = QPushButton("Browse")
        browse_symbols_button.clicked.connect(self.browse_symbols_file)
        symbols_layout.addWidget(browse_symbols_button)
        layout.addWidget(symbols_group)
        # --- Theme & Font Group ---
        theme_group = QGroupBox("Theme & Font")
        theme_layout = QHBoxLayout()
        theme_group.setLayout(theme_layout)
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light", "solarized", "high-contrast", "green", "blue", "windows-default"])
        self.theme_combo.setCurrentText(self.config.get('theme', 'dark'))
        self.theme_combo.currentTextChanged.connect(lambda t: self.apply_theme(t))
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addWidget(QLabel("Font:"))
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Segoe UI", "Arial", "Roboto", "Inter", "Fira Sans", "JetBrains Mono", "Courier New"])
        self.font_combo.setCurrentText(self.config.get('font_family', 'Segoe UI'))
        self.font_combo.currentTextChanged.connect(lambda f: self.apply_font(f, self.font_size_spin.value()))
        theme_layout.addWidget(self.font_combo)
        theme_layout.addWidget(QLabel("Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 32)
        self.font_size_spin.setValue(self.config.get('font_size', 12))
        self.font_size_spin.valueChanged.connect(lambda s: self.apply_font(self.font_combo.currentText(), s))
        theme_layout.addWidget(self.font_size_spin)
        layout.addWidget(theme_group)
        # --- Telegram Alerts Group ---
        telegram_group = QGroupBox("Telegram Alerts")
        telegram_layout = QHBoxLayout()
        telegram_group.setLayout(telegram_layout)
        self.telegram_enable_checkbox = QCheckBox("Enable Telegram Alerts")
        self.telegram_enable_checkbox.setChecked(self.config.get('telegram_enabled', False))
        telegram_layout.addWidget(self.telegram_enable_checkbox)
        telegram_layout.addWidget(QLabel("Bot Token:"))
        self.telegram_token_input = QLineEdit(self.config.get('telegram_token', ''))
        telegram_layout.addWidget(self.telegram_token_input)
        telegram_layout.addWidget(QLabel("Chat ID:"))
        self.telegram_chatid_input = QLineEdit(self.config.get('telegram_chat_id', ''))
        telegram_layout.addWidget(self.telegram_chatid_input)
        layout.addWidget(telegram_group)
        # --- Actions Group ---
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout()
        actions_group.setLayout(actions_layout)
        save_button = QPushButton("Save Configuration")
        save_button.clicked.connect(self.save_configuration)
        actions_layout.addWidget(save_button)
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.reset_config_to_defaults)
        actions_layout.addWidget(reset_button)
        self.config_feedback_label = QLabel()
        actions_layout.addWidget(self.config_feedback_label)
        layout.addWidget(actions_group)

    def test_db_connection(self, db_path):
        from sqlalchemy import create_engine
        try:
            engine = create_engine(f'sqlite:///{db_path}')
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            QMessageBox.information(self, "DB Connection", f"Successfully connected to {db_path}")
        except Exception as e:
            QMessageBox.critical(self, "DB Connection", f"Failed to connect to {db_path}: {str(e)}")

    def reset_config_to_defaults(self):
        self.db_path_input.setText(self.config_manager.default_config['db_path'])
        self.alt_db_path_input.setText(self.config_manager.default_config['alt_db_path'])
        self.indicators_db_path_input.setText(self.config_manager.default_config['indicators_db_path'])
        self.signals_db_path_input.setText(self.config_manager.default_config['signals_db_path'])
        self.symbols_file_input.setText(self.config_manager.default_config['symbols_file'])
        self.theme_combo.setCurrentText(self.config_manager.default_config['theme'])
        self.font_combo.setCurrentText(self.config_manager.default_config['font_family'])
        self.font_size_spin.setValue(self.config_manager.default_config['font_size'])
        self.telegram_enable_checkbox.setChecked(self.config_manager.default_config['telegram_enabled'])
        self.telegram_token_input.setText(self.config_manager.default_config['telegram_token'])
        self.telegram_chatid_input.setText(self.config_manager.default_config['telegram_chat_id'])
        self.config_feedback_label.setText("Reset to defaults.")

    def save_configuration(self):
        self.config = {
            'db_path': self.db_path_input.text(),
            'alt_db_path': self.alt_db_path_input.text(),
            'indicators_db_path': self.indicators_db_path_input.text(),
            'signals_db_path': self.signals_db_path_input.text(),
            'symbols_file': self.symbols_file_input.text(),
            'theme': self.theme_combo.currentText(),
            'export_path': self.config['export_path'],
            'max_threads': self.config['max_threads'],
            'font_family': self.font_combo.currentText(),
            'font_size': self.font_size_spin.value(),
            'telegram_enabled': self.telegram_enable_checkbox.isChecked(),
            'telegram_token': self.telegram_token_input.text(),
            'telegram_chat_id': self.telegram_chatid_input.text(),
        }
        if self.config_manager.save_config(self.config):
            self.apply_theme(self.config['theme'])
            self.config_feedback_label.setText("<span style='color:green'>Configuration saved successfully</span>")
        else:
            self.config_feedback_label.setText("<span style='color:red'>Failed to save configuration</span>")

    def create_query_tab(self):
        self.query_tab = QWidget()
        self.tab_widget.addTab(self.query_tab, "Stock Analysis")
        layout = QVBoxLayout()
        self.query_tab.setLayout(layout)
        
        # --- Symbol Selection for Queries ---
        symbol_select_layout = QHBoxLayout()
        symbol_select_layout.addWidget(QLabel("Symbol:"))
        self.query_symbol_combo = QComboBox()
        self.query_symbol_combo.setEditable(True)
        self.query_symbol_combo.setInsertPolicy(QComboBox.NoInsert)
        self.query_symbol_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        symbol_select_layout.addWidget(self.query_symbol_combo)
        self.query_reload_symbols_button = QPushButton("Reload Symbols")
        self.query_reload_symbols_button.clicked.connect(self.load_symbols_into_query_combo)
        symbol_select_layout.addWidget(self.query_reload_symbols_button)
        layout.addLayout(symbol_select_layout)
        self.load_symbols_into_query_combo()
        
        # --- Advanced Analysis Queries Section ---
        analysis_group = QGroupBox("Advanced Analysis Queries")
        analysis_layout = QVBoxLayout()
        analysis_group.setLayout(analysis_layout)
        
        # Query categories
        categories_layout = QHBoxLayout()
        
        # Technical Analysis Queries
        tech_analysis_layout = QVBoxLayout()
        tech_analysis_layout.addWidget(QLabel("<b>Technical Analysis:</b>"))
        rsi_button = QPushButton("RSI Analysis (Oversold/Overbought)")
        rsi_button.clicked.connect(lambda: self.load_advanced_query("rsi"))
        tech_analysis_layout.addWidget(rsi_button)
        ma_button = QPushButton("Moving Average Crossovers")
        ma_button.clicked.connect(lambda: self.load_advanced_query("moving_averages"))
        tech_analysis_layout.addWidget(ma_button)
        volume_button = QPushButton("Volume Analysis")
        volume_button.clicked.connect(lambda: self.load_advanced_query("volume"))
        tech_analysis_layout.addWidget(volume_button)
        patterns_button = QPushButton("Price Patterns & Breakouts")
        patterns_button.clicked.connect(lambda: self.load_advanced_query("patterns"))
        tech_analysis_layout.addWidget(patterns_button)
        # --- In create_query_tab, under Technical Analysis ---
        weekly_ao_button = QPushButton("Weekly AO (Awesome Oscillator)")
        weekly_ao_button.clicked.connect(lambda: self.load_advanced_query("weekly_ao"))
        tech_analysis_layout.addWidget(weekly_ao_button)
        monthly_ao_button = QPushButton("Monthly AO (Awesome Oscillator)")
        monthly_ao_button.clicked.connect(lambda: self.load_advanced_query("monthly_ao"))
        tech_analysis_layout.addWidget(monthly_ao_button)
        categories_layout.addLayout(tech_analysis_layout)
        
        # Fundamental Analysis Queries
        fundamental_layout = QVBoxLayout()
        fundamental_layout.addWidget(QLabel("<b>Fundamental Analysis:</b>"))
        volatility_button = QPushButton("Volatility Analysis")
        volatility_button.clicked.connect(lambda: self.load_advanced_query("volatility"))
        fundamental_layout.addWidget(volatility_button)
        performance_button = QPushButton("Performance Comparison")
        performance_button.clicked.connect(lambda: self.load_advanced_query("performance"))
        fundamental_layout.addWidget(performance_button)
        risk_button = QPushButton("Risk Assessment")
        risk_button.clicked.connect(lambda: self.load_advanced_query("risk"))
        fundamental_layout.addWidget(risk_button)
        categories_layout.addLayout(fundamental_layout)
        
        # Trading Signals
        signals_layout = QVBoxLayout()
        signals_layout.addWidget(QLabel("<b>Trading Signals:</b>"))
        buy_signals_button = QPushButton("Strong Buy Signals")
        buy_signals_button.clicked.connect(lambda: self.load_advanced_query("buy_signals"))
        signals_layout.addWidget(buy_signals_button)
        sell_signals_button = QPushButton("Strong Sell Signals")
        sell_signals_button.clicked.connect(lambda: self.load_advanced_query("sell_signals"))
        signals_layout.addWidget(sell_signals_button)
        hold_analysis_button = QPushButton("Hold Analysis")
        hold_analysis_button.clicked.connect(lambda: self.load_advanced_query("hold"))
        signals_layout.addWidget(hold_analysis_button)
        categories_layout.addLayout(signals_layout)
        analysis_layout.addLayout(categories_layout)
        
        # Quick Analysis for specific symbol
        symbol_analysis_layout = QHBoxLayout()
        symbol_analysis_layout.addWidget(QLabel("Quick Analysis for Symbol:"))
        self.quick_symbol_input = QLineEdit()
        self.quick_symbol_input.setPlaceholderText("Enter symbol (e.g., PSO)")
        symbol_analysis_layout.addWidget(self.quick_symbol_input)
        quick_analysis_button = QPushButton("Run Quick Analysis")
        quick_analysis_button.clicked.connect(self.run_quick_analysis)
        symbol_analysis_layout.addWidget(quick_analysis_button)
        analysis_layout.addLayout(symbol_analysis_layout)
        layout.addWidget(analysis_group)
        
        # Query input with history
        query_group = QGroupBox("SQL Query")
        query_layout = QVBoxLayout()
        query_group.setLayout(query_layout)
        self.query_input = QPlainTextEdit()
        self.query_input.setPlaceholderText("Enter your SQL query here...")
        query_layout.addWidget(self.query_input)
        # Query history
        self.query_history = []
        self.query_history_combo = QComboBox()
        self.query_history_combo.setEditable(False)
        self.query_history_combo.setInsertPolicy(QComboBox.NoInsert)
        self.query_history_combo.currentIndexChanged.connect(self.load_query_from_history)
        query_layout.addWidget(QLabel("Query History:"))
        query_layout.addWidget(self.query_history_combo)
        layout.addWidget(query_group)
        # Execute and export buttons
        button_layout = QHBoxLayout()
        execute_button = QPushButton("Execute Query")
        execute_button.clicked.connect(self.execute_query)
        button_layout.addWidget(execute_button)
        export_query_button = QPushButton("Export Results")
        export_query_button.clicked.connect(self.export_query_results)
        button_layout.addWidget(export_query_button)
        self.visualize_query_button = QPushButton("Visualize Results")
        self.visualize_query_button.clicked.connect(self.visualize_query_results)
        self.visualize_query_button.setEnabled(False)
        button_layout.addWidget(self.visualize_query_button)
        clear_results_button = QPushButton("Clear Results")
        clear_results_button.clicked.connect(self.clear_query_results)
        button_layout.addWidget(clear_results_button)
        layout.addLayout(button_layout)
        # Results table (with max height and scroll)
        from PyQt5.QtWidgets import QScrollArea
        self.results_table = QTableWidget()
        self.results_table.setMaximumHeight(350)
        self.results_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        self.results_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.results_table)
        # Error/feedback area
        self.query_feedback_label = QLabel()
        layout.addWidget(self.query_feedback_label)

        # --- In create_query_tab, above the main query symbol combo ---
        signal_symbol_layout = QHBoxLayout()
        signal_symbol_layout.addWidget(QLabel("Signal Symbol (from KMI30 Tracking DB):"))
        self.signal_symbol_combo = QComboBox()
        self.signal_symbol_combo.setEditable(True)
        self.signal_symbol_combo.setInsertPolicy(QComboBox.NoInsert)
        self.signal_symbol_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        signal_symbol_layout.addWidget(self.signal_symbol_combo)
        self.signal_reload_symbols_button = QPushButton("Reload Signal Symbols")
        self.signal_reload_symbols_button.clicked.connect(self.load_signal_symbols_into_combo)
        signal_symbol_layout.addWidget(self.signal_reload_symbols_button)
        layout.addLayout(signal_symbol_layout)
        self.load_signal_symbols_into_combo()
        # Add a note for the user
        layout.addWidget(QLabel("<i>Select a symbol from the KMI30 tracking DB to analyze active buy/sell signals and confirm their strength with other indicators.</i>"))

        # --- Add Show Active Signals button ---
        active_signals_button = QPushButton("Show Active Signals for Selected Symbol")
        active_signals_button.clicked.connect(lambda: self.load_advanced_query("active_signals"))
        layout.addWidget(active_signals_button)

    def load_symbols_into_query_combo(self):
        self.query_symbol_combo.clear()
        self.query_symbol_combo.addItem("All Symbols")
        symbols = []
        try:
            with self.data_reader.engine.connect() as conn:
                result = conn.execute(text(r"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'PSX\_%\_stock\_data'"))
                for row in result.fetchall():
                    tname = row[0]
                    if tname.startswith('PSX_') and tname.endswith('_stock_data'):
                        symbol = tname[len('PSX_'):-len('_stock_data')]
                        if symbol.isalpha():
                            symbols.append(symbol)
        except Exception as e:
            self.logger.warning(f"Could not load symbols from DB: {e}")
        if not symbols:
            try:
                symbols_df = pd.read_excel(self.config['symbols_file'], sheet_name='KMIALL')
                symbols = [str(s) for s in symbols_df.iloc[:, 0].tolist() if isinstance(s, str) and s.isalpha()]
            except Exception as e:
                self.logger.warning(f"Could not load symbols from file: {e}")
        self.query_symbol_combo.addItems(sorted(set(symbols)))

    def clear_query_results(self):
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        self.query_feedback_label.setText("")

    def load_advanced_query(self, query_type):
        symbol = self.query_symbol_combo.currentText().strip().upper()
        all_symbols = symbol == "ALL SYMBOLS"
        queries = {}
        # Helper: single symbol table
        def table(sym):
            return f"PSX_{sym}_stock_data"
        # RSI
        if query_type == "rsi":
            if all_symbols:
                queries["rsi"] = f"""
-- Weekly Average RSI Analysis: Find stocks with oversold (RSI < 30) or overbought (RSI > 70) conditions using weekly average RSI
WITH all_data AS (
    {{SYMBOL_UNION}}
),
daily_rsi AS (
    SELECT
        symbol,
        Date,
        Close,
        LAG(Close, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_close
    FROM all_data
    WHERE Date >= date('now', '-180 days')
),
rsi_calc AS (
    SELECT
        symbol,
        Date,
        Close,
        CASE WHEN prev_close > Close THEN prev_close - Close ELSE 0 END as loss,
        CASE WHEN Close > prev_close THEN Close - prev_close ELSE 0 END as gain,
        AVG(CASE WHEN prev_close > Close THEN prev_close - Close ELSE 0 END) OVER (PARTITION BY symbol ORDER BY Date ROWS 13 PRECEDING) as avg_loss_14,
        AVG(CASE WHEN Close > prev_close THEN Close - prev_close ELSE 0 END) OVER (PARTITION BY symbol ORDER BY Date ROWS 13 PRECEDING) as avg_gain_14
    FROM daily_rsi
),
rsi_final AS (
    SELECT
        symbol,
        Date,
        Close,
        ROUND(
            CASE
                WHEN avg_loss_14 = 0 THEN 100
                ELSE 100 - (100 / (1 + (avg_gain_14 / avg_loss_14)))
            END, 2
        ) as RSI
    FROM rsi_calc
),
weekly_rsi AS (
    SELECT
        symbol,
        strftime('%Y-%W', Date) as week,
        MAX(Date) as week_end,
        AVG(RSI) as avg_weekly_rsi
    FROM rsi_final
    GROUP BY symbol, week
),
weekly_close AS (
    SELECT symbol, Date as week_end, Close as Close_Price
    FROM rsi_final
    WHERE (symbol, Date) IN (SELECT symbol, MAX(Date) FROM rsi_final GROUP BY symbol, strftime('%Y-%W', Date))
)
SELECT
    wr.symbol,
    wr.week_end as Date,
    wc.Close_Price,
    ROUND(wr.avg_weekly_rsi, 2) as Weekly_Avg_RSI,
    CASE
        WHEN wr.avg_weekly_rsi < 30 THEN 'Oversold (Buy Signal)'
        WHEN wr.avg_weekly_rsi > 70 THEN 'Overbought (Sell Signal)'
        ELSE 'Neutral'
    END as Signal
FROM weekly_rsi wr
LEFT JOIN weekly_close wc ON wr.symbol = wc.symbol AND wr.week_end = wc.week_end
WHERE wr.week_end = (SELECT MAX(week_end) FROM weekly_rsi WHERE symbol = wr.symbol)
ORDER BY Weekly_Avg_RSI DESC
LIMIT 50;
"""
                # Build union for all symbols
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close FROM {table(s)}" for s in all_syms])
                queries["rsi"] = queries["rsi"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["rsi"] = f"""
-- Weekly Average RSI Analysis for {symbol}
WITH daily_rsi AS (
    SELECT
        Date,
        Close,
        LAG(Close, 1) OVER (ORDER BY Date) as prev_close
    FROM {table(symbol)}
    WHERE Date >= date('now', '-180 days')
),
rsi_calc AS (
    SELECT
        Date,
        Close,
        CASE WHEN prev_close > Close THEN prev_close - Close ELSE 0 END as loss,
        CASE WHEN Close > prev_close THEN Close - prev_close ELSE 0 END as gain,
        AVG(CASE WHEN prev_close > Close THEN prev_close - Close ELSE 0 END) OVER (ORDER BY Date ROWS 13 PRECEDING) as avg_loss_14,
        AVG(CASE WHEN Close > prev_close THEN Close - prev_close ELSE 0 END) OVER (ORDER BY Date ROWS 13 PRECEDING) as avg_gain_14
    FROM daily_rsi
),
rsi_final AS (
    SELECT
        Date,
        Close,
        ROUND(
            CASE
                WHEN avg_loss_14 = 0 THEN 100
                ELSE 100 - (100 / (1 + (avg_gain_14 / avg_loss_14)))
            END, 2
        ) as RSI
    FROM rsi_calc
),
weekly_rsi AS (
    SELECT
        strftime('%Y-%W', Date) as week,
        MAX(Date) as week_end,
        AVG(RSI) as avg_weekly_rsi
    FROM rsi_final
    GROUP BY week
),
weekly_close AS (
    SELECT Date as week_end, Close as Close_Price
    FROM rsi_final
    WHERE Date IN (SELECT MAX(Date) FROM rsi_final GROUP BY strftime('%Y-%W', Date))
)
SELECT
    wr.week_end as Date,
    wc.Close_Price,
    ROUND(wr.avg_weekly_rsi, 2) as Weekly_Avg_RSI,
    CASE
        WHEN wr.avg_weekly_rsi < 30 THEN 'Oversold (Buy Signal)'
        WHEN wr.avg_weekly_rsi > 70 THEN 'Overbought (Sell Signal)'
        ELSE 'Neutral'
    END as Signal
FROM weekly_rsi wr
LEFT JOIN weekly_close wc ON wr.week_end = wc.week_end
ORDER BY Date DESC
LIMIT 10;
"""
        # Moving Averages
        if query_type == "moving_averages":
            if all_symbols:
                queries["moving_averages"] = f"""
-- Moving Average Crossovers: Find stocks with bullish/bearish MA crossovers
WITH ma_data AS (
    SELECT 
        symbol,
        Date,
        Close,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 9 PRECEDING) as MA_10,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as MA_20,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 49 PRECEDING) as MA_50
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-60 days')
),
ma_signals AS (
    SELECT 
        symbol,
        Date,
        Close,
        MA_10,
        MA_20,
        MA_50,
        LAG(MA_10, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_MA_10,
        LAG(MA_20, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_MA_20,
        LAG(MA_50, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_MA_50
    FROM ma_data
)
SELECT 
    symbol,
    Date,
    ROUND(Close, 2) as Close_Price,
    ROUND(MA_10, 2) as MA_10,
    ROUND(MA_20, 2) as MA_20,
    ROUND(MA_50, 2) as MA_50,
    CASE 
        WHEN MA_10 > MA_20 AND prev_MA_10 <= prev_MA_20 THEN 'Golden Cross (10/20) - Bullish'
        WHEN MA_10 < MA_20 AND prev_MA_10 >= prev_MA_20 THEN 'Death Cross (10/20) - Bearish'
        WHEN MA_20 > MA_50 AND prev_MA_20 <= prev_MA_50 THEN 'Golden Cross (20/50) - Strong Bullish'
        WHEN MA_20 < MA_50 AND prev_MA_20 >= prev_MA_50 THEN 'Death Cross (20/50) - Strong Bearish'
        WHEN MA_10 > MA_20 AND MA_20 > MA_50 THEN 'Bullish Alignment'
        WHEN MA_10 < MA_20 AND MA_20 < MA_50 THEN 'Bearish Alignment'
        ELSE 'Mixed Signals'
    END as Signal
FROM ma_signals
WHERE Date = (SELECT MAX(Date) FROM ma_signals)
ORDER BY 
    CASE 
        WHEN MA_10 > MA_20 AND MA_20 > MA_50 THEN 1
        WHEN MA_10 < MA_20 AND MA_20 < MA_50 THEN 3
        ELSE 2
    END,
    symbol
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close FROM {table(s)}" for s in all_syms])
                queries["moving_averages"] = queries["moving_averages"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["moving_averages"] = f"""
-- Moving Average Crossovers for {symbol}
WITH ma_data AS (
    SELECT Date, Close,
        AVG(Close) OVER (ORDER BY Date ROWS 9 PRECEDING) as MA_10,
        AVG(Close) OVER (ORDER BY Date ROWS 19 PRECEDING) as MA_20,
        AVG(Close) OVER (ORDER BY Date ROWS 49 PRECEDING) as MA_50
    FROM {table(symbol)}
    WHERE Date >= date('now', '-60 days')
),
ma_signals AS (
    SELECT Date, Close, MA_10, MA_20, MA_50,
        LAG(MA_10, 1) OVER (ORDER BY Date) as prev_MA_10,
        LAG(MA_20, 1) OVER (ORDER BY Date) as prev_MA_20,
        LAG(MA_50, 1) OVER (ORDER BY Date) as prev_MA_50
    FROM ma_data
)
SELECT Date, ROUND(Close, 2) as Close_Price, ROUND(MA_10, 2) as MA_10, ROUND(MA_20, 2) as MA_20, ROUND(MA_50, 2) as MA_50,
    CASE 
        WHEN MA_10 > MA_20 AND prev_MA_10 <= prev_MA_20 THEN 'Golden Cross (10/20) - Bullish'
        WHEN MA_10 < MA_20 AND prev_MA_10 >= prev_MA_20 THEN 'Death Cross (10/20) - Bearish'
        WHEN MA_20 > MA_50 AND prev_MA_20 <= prev_MA_50 THEN 'Golden Cross (20/50) - Strong Bullish'
        WHEN MA_20 < MA_50 AND prev_MA_20 >= prev_MA_50 THEN 'Death Cross (20/50) - Strong Bearish'
        WHEN MA_10 > MA_20 AND MA_20 > MA_50 THEN 'Bullish Alignment'
        WHEN MA_10 < MA_20 AND MA_20 < MA_50 THEN 'Bearish Alignment'
        ELSE 'Mixed Signals'
    END as Signal
FROM ma_signals
WHERE Date = (SELECT MAX(Date) FROM ma_signals)
ORDER BY 
    CASE 
        WHEN MA_10 > MA_20 AND MA_20 > MA_50 THEN 1
        WHEN MA_10 < MA_20 AND MA_20 < MA_50 THEN 3
        ELSE 2
    END
"""
        # Volume Analysis
        if query_type == "volume":
            if all_symbols:
                queries["volume"] = f"""
-- Volume Analysis: High volume breakouts and unusual volume patterns
WITH volume_data AS (
    SELECT 
        symbol,
        Date,
        Close,
        Volume,
        AVG(Volume) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as avg_volume_20,
        LAG(Close, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_close
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-30 days')
)
SELECT 
    symbol,
    Date,
    ROUND(Close, 2) as Close_Price,
    Volume,
    ROUND(avg_volume_20, 0) as Avg_Volume_20d,
    ROUND((Volume * 100.0 / avg_volume_20), 1) as Volume_Ratio,
    ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as Price_Change_Percent,
    CASE 
        WHEN Volume > avg_volume_20 * 2 AND Close > prev_close THEN 'High Volume Breakout (Bullish)'
        WHEN Volume > avg_volume_20 * 2 AND Close < prev_close THEN 'High Volume Breakdown (Bearish)'
        WHEN Volume > avg_volume_20 * 1.5 AND Close > prev_close THEN 'Above Average Volume Rally'
        WHEN Volume > avg_volume_20 * 1.5 AND Close < prev_close THEN 'Above Average Volume Decline'
        WHEN Volume < avg_volume_20 * 0.5 THEN 'Low Volume (Caution)'
        ELSE 'Normal Volume'
    END as Volume_Signal
FROM volume_data
WHERE Date = (SELECT MAX(Date) FROM volume_data)
ORDER BY Volume_Ratio DESC
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close, Volume FROM {table(s)}" for s in all_syms])
                queries["volume"] = queries["volume"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["volume"] = f"""
-- Volume Analysis for {symbol}
WITH volume_data AS (
    SELECT Date, Close, Volume,
        AVG(Volume) OVER (ORDER BY Date ROWS 19 PRECEDING) as avg_volume_20,
        LAG(Close, 1) OVER (ORDER BY Date) as prev_close
    FROM {table(symbol)}
    WHERE Date >= date('now', '-30 days')
)
SELECT Date, ROUND(Close, 2) as Close_Price, Volume, ROUND(avg_volume_20, 0) as Avg_Volume_20d,
    ROUND((Volume * 100.0 / avg_volume_20), 1) as Volume_Ratio,
    ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as Price_Change_Percent,
    CASE 
        WHEN Volume > avg_volume_20 * 2 AND Close > prev_close THEN 'High Volume Breakout (Bullish)'
        WHEN Volume > avg_volume_20 * 2 AND Close < prev_close THEN 'High Volume Breakdown (Bearish)'
        WHEN Volume > avg_volume_20 * 1.5 AND Close > prev_close THEN 'Above Average Volume Rally'
        WHEN Volume > avg_volume_20 * 1.5 AND Close < prev_close THEN 'Above Average Volume Decline'
        WHEN Volume < avg_volume_20 * 0.5 THEN 'Low Volume (Caution)'
        ELSE 'Normal Volume'
    END as Volume_Signal
FROM volume_data
WHERE Date = (SELECT MAX(Date) FROM volume_data)
ORDER BY Volume_Ratio DESC
"""
        # Patterns
        if query_type == "patterns":
            if all_symbols:
                queries["patterns"] = f"""
-- Price Patterns and Breakouts: Support/Resistance levels and pattern recognition
WITH price_data AS (
    SELECT 
        symbol,
        Date,
        Open,
        High,
        Low,
        Close,
        Volume,
        LAG(High, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_high,
        LAG(Low, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_low,
        LAG(Close, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_close,
        LAG(Close, 20) OVER (PARTITION BY symbol ORDER BY Date) as close_20d_ago
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-30 days')
)
SELECT 
    symbol,
    Date,
    ROUND(Open, 2) as Open,
    ROUND(High, 2) as High,
    ROUND(Low, 2) as Low,
    ROUND(Close, 2) as Close,
    ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as Daily_Return,
    ROUND(((Close - close_20d_ago) * 100.0 / close_20d_ago), 2) as Return_20d,
    CASE 
        WHEN Close > High AND High > prev_high THEN 'Breakout Above Resistance (Bullish)'
        WHEN Close < Low AND Low < prev_low THEN 'Breakdown Below Support (Bearish)'
        WHEN Close > prev_close * 1.05 THEN 'Strong Upward Move (>5%)'
        WHEN Close < prev_close * 0.95 THEN 'Strong Downward Move (>5%)'
        WHEN Close > close_20d_ago * 1.1 THEN '20-Day High (Bullish)'
        WHEN Close < close_20d_ago * 0.9 THEN '20-Day Low (Bearish)'
        WHEN High = Low THEN 'Doji Pattern (Indecision)'
        WHEN Close > Open AND (Close - Open) > (High - Close) * 2 THEN 'Hammer Pattern (Bullish)'
        WHEN Close < Open AND (Open - Close) > (Close - Low) * 2 THEN 'Shooting Star (Bearish)'
        ELSE 'Normal Price Action'
    END as Pattern_Signal
FROM price_data
WHERE Date = (SELECT MAX(Date) FROM price_data)
ORDER BY 
    CASE 
        WHEN Close > High AND High > prev_high THEN 1
        WHEN Close < Low AND Low < prev_low THEN 2
        WHEN Close > prev_close * 1.05 THEN 3
        WHEN Close < prev_close * 0.95 THEN 4
        ELSE 5
    END,
    symbol
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Open, High, Low, Close, Volume FROM {table(s)}" for s in all_syms])
                queries["patterns"] = queries["patterns"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["patterns"] = f"""
-- Price Patterns and Breakouts for {symbol}
WITH price_data AS (
    SELECT Date, Open, High, Low, Close, Volume,
        LAG(High, 1) OVER (ORDER BY Date) as prev_high,
        LAG(Low, 1) OVER (ORDER BY Date) as prev_low,
        LAG(Close, 1) OVER (ORDER BY Date) as prev_close,
        LAG(Close, 20) OVER (ORDER BY Date) as close_20d_ago
    FROM {table(symbol)}
    WHERE Date >= date('now', '-30 days')
)
SELECT Date, ROUND(Open, 2) as Open, ROUND(High, 2) as High, ROUND(Low, 2) as Low, ROUND(Close, 2) as Close,
    ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as Daily_Return,
    ROUND(((Close - close_20d_ago) * 100.0 / close_20d_ago), 2) as Return_20d,
    CASE 
        WHEN Close > High AND High > prev_high THEN 'Breakout Above Resistance (Bullish)'
        WHEN Close < Low AND Low < prev_low THEN 'Breakdown Below Support (Bearish)'
        WHEN Close > prev_close * 1.05 THEN 'Strong Upward Move (>5%)'
        WHEN Close < prev_close * 0.95 THEN 'Strong Downward Move (>5%)'
        WHEN Close > close_20d_ago * 1.1 THEN '20-Day High (Bullish)'
        WHEN Close < close_20d_ago * 0.9 THEN '20-Day Low (Bearish)'
        WHEN High = Low THEN 'Doji Pattern (Indecision)'
        WHEN Close > Open AND (Close - Open) > (High - Close) * 2 THEN 'Hammer Pattern (Bullish)'
        WHEN Close < Open AND (Open - Close) > (Close - Low) * 2 THEN 'Shooting Star (Bearish)'
        ELSE 'Normal Price Action'
    END as Pattern_Signal
FROM price_data
WHERE Date = (SELECT MAX(Date) FROM price_data)
ORDER BY 
    CASE 
        WHEN Close > High AND High > prev_high THEN 1
        WHEN Close < Low AND Low < prev_low THEN 2
        WHEN Close > prev_close * 1.05 THEN 3
        WHEN Close < prev_close * 0.95 THEN 4
        ELSE 5
    END
"""
        # Volatility
        if query_type == "volatility":
            if all_symbols:
                queries["volatility"] = f"""
-- Volatility Analysis: Historical volatility and risk assessment
WITH volatility_data AS (
    SELECT 
        symbol,
        Date,
        Close,
        LAG(Close, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_close
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-60 days')
),
volatility_calc AS (
    SELECT 
        symbol,
        Date,
        Close,
        ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as daily_return,
        AVG(ABS(((Close - prev_close) * 100.0 / prev_close))) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as avg_daily_volatility_20d
    FROM volatility_data
)
SELECT 
    symbol,
    Date,
    ROUND(Close, 2) as Close_Price,
    daily_return as Daily_Return_Percent,
    ROUND(avg_daily_volatility_20d, 2) as Avg_Daily_Volatility_20d
FROM volatility_calc
WHERE Date = (SELECT MAX(Date) FROM volatility_calc)
ORDER BY avg_daily_volatility_20d DESC
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close FROM {table(s)}" for s in all_syms])
                queries["volatility"] = queries["volatility"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["volatility"] = f"""
-- Volatility Analysis for {symbol}
WITH volatility_data AS (
    SELECT Date, Close, LAG(Close, 1) OVER (ORDER BY Date) as prev_close
    FROM {table(symbol)}
    WHERE Date >= date('now', '-60 days')
),
volatility_calc AS (
    SELECT Date, Close, ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as daily_return,
        AVG(ABS(((Close - prev_close) * 100.0 / prev_close))) OVER (ORDER BY Date ROWS 19 PRECEDING) as avg_daily_volatility_20d
    FROM volatility_data
)
SELECT Date, ROUND(Close, 2) as Close_Price, daily_return as Daily_Return_Percent, ROUND(avg_daily_volatility_20d, 2) as Avg_Daily_Volatility_20d
FROM volatility_calc
WHERE Date = (SELECT MAX(Date) FROM volatility_calc)
ORDER BY avg_daily_volatility_20d DESC
"""
        # Performance
        if query_type == "performance":
            if all_symbols:
                queries["performance"] = f"""
-- Performance Comparison: Relative strength and momentum analysis
WITH performance_data AS (
    SELECT 
        symbol,
        Date,
        Close,
        LAG(Close, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_close,
        LAG(Close, 5) OVER (PARTITION BY symbol ORDER BY Date) as close_5d_ago,
        LAG(Close, 20) OVER (PARTITION BY symbol ORDER BY Date) as close_20d_ago
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-90 days')
)
SELECT 
    symbol,
    Date,
    ROUND(Close, 2) as Close_Price,
    ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as Daily_Return,
    ROUND(((Close - close_5d_ago) * 100.0 / close_5d_ago), 2) as Return_5d,
    ROUND(((Close - close_20d_ago) * 100.0 / close_20d_ago), 2) as Return_20d,
    CASE 
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > 10 THEN 'Strong Momentum (Buy)'
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > 5 THEN 'Positive Momentum (Hold/Buy)'
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > 0 THEN 'Slight Positive (Hold)'
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > -5 THEN 'Slight Negative (Hold)'
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > -10 THEN 'Negative Momentum (Hold/Sell)'
        ELSE 'Strong Negative (Sell)'
    END as Momentum_Signal
FROM performance_data
WHERE Date = (SELECT MAX(Date) FROM performance_data)
ORDER BY Return_20d DESC
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close FROM {table(s)}" for s in all_syms])
                queries["performance"] = queries["performance"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["performance"] = f"""
-- Performance Comparison for {symbol}
WITH performance_data AS (
    SELECT Date, Close, LAG(Close, 1) OVER (ORDER BY Date) as prev_close,
        LAG(Close, 5) OVER (ORDER BY Date) as close_5d_ago,
        LAG(Close, 20) OVER (ORDER BY Date) as close_20d_ago
    FROM {table(symbol)}
    WHERE Date >= date('now', '-90 days')
)
SELECT Date, ROUND(Close, 2) as Close_Price, ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as Daily_Return,
    ROUND(((Close - close_5d_ago) * 100.0 / close_5d_ago), 2) as Return_5d,
    ROUND(((Close - close_20d_ago) * 100.0 / close_20d_ago), 2) as Return_20d,
    CASE 
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > 10 THEN 'Strong Momentum (Buy)'
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > 5 THEN 'Positive Momentum (Hold/Buy)'
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > 0 THEN 'Slight Positive (Hold)'
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > -5 THEN 'Slight Negative (Hold)'
        WHEN ((Close - close_20d_ago) * 100.0 / close_20d_ago) > -10 THEN 'Negative Momentum (Hold/Sell)'
        ELSE 'Strong Negative (Sell)'
    END as Momentum_Signal
FROM performance_data
WHERE Date = (SELECT MAX(Date) FROM performance_data)
ORDER BY Return_20d DESC
"""
        # Risk
        if query_type == "risk":
            if all_symbols:
                queries["risk"] = f"""
-- Risk Assessment: Drawdown analysis and risk metrics
WITH risk_data AS (
    SELECT 
        symbol,
        Date,
        Close,
        LAG(Close, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_close,
        MAX(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS UNBOUNDED PRECEDING) as peak_price,
        MIN(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS UNBOUNDED PRECEDING) as trough_price
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-90 days')
),
risk_metrics AS (
    SELECT 
        symbol,
        Date,
        Close,
        ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as daily_return,
        ROUND(((peak_price - Close) * 100.0 / peak_price), 2) as current_drawdown
    FROM risk_data
)
SELECT 
    symbol,
    Date,
    ROUND(Close, 2) as Close_Price,
    daily_return as Daily_Return_Percent,
    current_drawdown as Current_Drawdown_Percent
FROM risk_metrics
WHERE Date = (SELECT MAX(Date) FROM risk_metrics)
ORDER BY current_drawdown DESC
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close FROM {table(s)}" for s in all_syms])
                queries["risk"] = queries["risk"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["risk"] = f"""
-- Risk Assessment for {symbol}
WITH risk_data AS (
    SELECT Date, Close, LAG(Close, 1) OVER (ORDER BY Date) as prev_close,
        MAX(Close) OVER (ORDER BY Date ROWS UNBOUNDED PRECEDING) as peak_price,
        MIN(Close) OVER (ORDER BY Date ROWS UNBOUNDED PRECEDING) as trough_price
    FROM {table(symbol)}
    WHERE Date >= date('now', '-90 days')
),
risk_metrics AS (
    SELECT Date, Close, ROUND(((Close - prev_close) * 100.0 / prev_close), 2) as daily_return,
        ROUND(((peak_price - Close) * 100.0 / peak_price), 2) as current_drawdown
    FROM risk_data
)
SELECT Date, ROUND(Close, 2) as Close_Price, daily_return as Daily_Return_Percent, current_drawdown as Current_Drawdown_Percent
FROM risk_metrics
WHERE Date = (SELECT MAX(Date) FROM risk_metrics)
ORDER BY current_drawdown DESC
"""
        # Buy Signals
        if query_type == "buy_signals":
            if all_symbols:
                queries["buy_signals"] = f"""
-- Strong Buy Signals: Multiple technical indicators showing bullish signals
WITH buy_signals AS (
    SELECT 
        symbol,
        Date,
        Close,
        Volume,
        LAG(Close, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_close,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 9 PRECEDING) as ma_10,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as ma_20,
        AVG(Volume) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as avg_volume_20
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-30 days')
)
SELECT 
    symbol,
    Date,
    ROUND(Close, 2) as Close_Price,
    ROUND(ma_10, 2) as MA_10,
    ROUND(ma_20, 2) as MA_20,
    ROUND((Volume * 100.0 / avg_volume_20), 1) as Volume_Ratio
FROM buy_signals
WHERE Date = (SELECT MAX(Date) FROM buy_signals)
    AND Close > ma_10
ORDER BY Volume_Ratio DESC
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close FROM {table(s)}" for s in all_syms])
                queries["buy_signals"] = queries["buy_signals"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["buy_signals"] = f"""
-- Strong Buy Signals for {symbol}
WITH buy_signals AS (
    SELECT Date, Close, Volume,
        AVG(Close) OVER (ORDER BY Date ROWS 9 PRECEDING) as ma_10,
        AVG(Close) OVER (ORDER BY Date ROWS 19 PRECEDING) as ma_20,
        AVG(Volume) OVER (ORDER BY Date ROWS 19 PRECEDING) as avg_volume_20,
        LAG(Close, 1) OVER (ORDER BY Date) as prev_close
    FROM {table(symbol)}
    WHERE Date >= date('now', '-30 days')
)
SELECT Date, ROUND(Close, 2) as Close_Price, ROUND(ma_10, 2) as MA_10, ROUND(ma_20, 2) as MA_20,
    ROUND((Volume * 100.0 / avg_volume_20), 1) as Volume_Ratio
FROM buy_signals
WHERE Date = (SELECT MAX(Date) FROM buy_signals)
    AND Close > ma_10
ORDER BY Volume_Ratio DESC
"""
        # Sell Signals
        if query_type == "sell_signals":
            if all_symbols:
                queries["sell_signals"] = f"""
-- Strong Sell Signals: Multiple technical indicators showing bearish signals
WITH sell_signals AS (
    SELECT 
        symbol,
        Date,
        Close,
        Volume,
        LAG(Close, 1) OVER (PARTITION BY symbol ORDER BY Date) as prev_close,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 9 PRECEDING) as ma_10,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as ma_20,
        AVG(Volume) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as avg_volume_20
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-30 days')
)
SELECT 
    symbol,
    Date,
    ROUND(Close, 2) as Close_Price,
    ROUND(ma_10, 2) as MA_10,
    ROUND(ma_20, 2) as MA_20,
    ROUND((Volume * 100.0 / avg_volume_20), 1) as Volume_Ratio
FROM sell_signals
WHERE Date = (SELECT MAX(Date) FROM sell_signals)
    AND Close < ma_10
ORDER BY Volume_Ratio DESC
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close FROM {table(s)}" for s in all_syms])
                queries["sell_signals"] = queries["sell_signals"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["sell_signals"] = f"""
-- Strong Sell Signals for {symbol}
WITH sell_signals AS (
    SELECT Date, Close, Volume,
        AVG(Close) OVER (ORDER BY Date ROWS 9 PRECEDING) as ma_10,
        AVG(Close) OVER (ORDER BY Date ROWS 19 PRECEDING) as ma_20,
        AVG(Volume) OVER (ORDER BY Date ROWS 19 PRECEDING) as avg_volume_20,
        LAG(Close, 1) OVER (ORDER BY Date) as prev_close
    FROM {table(symbol)}
    WHERE Date >= date('now', '-30 days')
)
SELECT Date, ROUND(Close, 2) as Close_Price, ROUND(ma_10, 2) as MA_10, ROUND(ma_20, 2) as MA_20,
    ROUND((Volume * 100.0 / avg_volume_20), 1) as Volume_Ratio
FROM sell_signals
WHERE Date = (SELECT MAX(Date) FROM sell_signals)
    AND Close < ma_10
ORDER BY Volume_Ratio DESC
"""
        # Hold
        if query_type == "hold":
            if all_symbols:
                queries["hold"] = f"""
-- Hold Analysis: Stocks with neutral signals or mixed indicators
WITH hold_analysis AS (
    SELECT 
        symbol,
        Date,
        Close,
        Volume,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 9 PRECEDING) as ma_10,
        AVG(Close) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as ma_20,
        AVG(Volume) OVER (PARTITION BY symbol ORDER BY Date ROWS 19 PRECEDING) as avg_volume_20
    FROM (
        {{SYMBOL_UNION}}
    ) all_data
    WHERE Date >= date('now', '-30 days')
)
SELECT 
    symbol,
    Date,
    ROUND(Close, 2) as Close_Price,
    ROUND(ma_10, 2) as MA_10,
    ROUND(ma_20, 2) as MA_20,
    ROUND((Volume * 100.0 / avg_volume_20), 1) as Volume_Ratio
FROM hold_analysis
WHERE Date = (SELECT MAX(Date) FROM hold_analysis)
    AND ABS(Close - ma_10) / ma_10 < 0.05
ORDER BY Volume_Ratio DESC
"""
                all_syms = [self.query_symbol_combo.itemText(i) for i in range(1, self.query_symbol_combo.count())]
                union = "\n        UNION ALL ".join([f"SELECT '{s}' as symbol, Date, Close FROM {table(s)}" for s in all_syms])
                queries["hold"] = queries["hold"].replace("{{SYMBOL_UNION}}", union)
            else:
                queries["hold"] = f"""
-- Hold Analysis for {symbol}
WITH hold_analysis AS (
    SELECT Date, Close, Volume,
        AVG(Close) OVER (ORDER BY Date ROWS 9 PRECEDING) as ma_10,
        AVG(Close) OVER (ORDER BY Date ROWS 19 PRECEDING) as ma_20,
        AVG(Volume) OVER (ORDER BY Date ROWS 19 PRECEDING) as avg_volume_20
    FROM {table(symbol)}
    WHERE Date >= date('now', '-30 days')
)
SELECT Date, ROUND(Close, 2) as Close_Price, ROUND(ma_10, 2) as MA_10, ROUND(ma_20, 2) as MA_20,
    ROUND((Volume * 100.0 / avg_volume_20), 1) as Volume_Ratio
FROM hold_analysis
WHERE Date = (SELECT MAX(Date) FROM hold_analysis)
    AND ABS(Close - ma_10) / ma_10 < 0.05
ORDER BY Volume_Ratio DESC
"""
        # Weekly AO
        if query_type == "weekly_ao":
            if all_symbols:
                self.query_feedback_label.setText("<span style='color:red'>Weekly AO for all symbols is not supported yet.</span>")
                return
            queries["weekly_ao"] = f"""
-- Weekly Awesome Oscillator (AO) with Crossover Signal and AO_SMA3 for {symbol}
WITH base AS (
    SELECT
        Date,
        (High + Low) / 2.0 AS median_price
    FROM {table(symbol)}
),
weekly AS (
    SELECT
        strftime('%Y-%W', Date) AS week,
        MAX(Date) AS week_end,
        AVG(median_price) AS weekly_median
    FROM base
    GROUP BY week
),
ao_calc AS (
    SELECT
        week_end AS Date,
        weekly_median,
        AVG(weekly_median) OVER (ORDER BY week_end ROWS 4 PRECEDING) AS sma5,
        AVG(weekly_median) OVER (ORDER BY week_end ROWS 33 PRECEDING) AS sma34
    FROM weekly
),
ao_final AS (
    SELECT
        Date,
        ROUND(weekly_median, 2) AS Weekly_Median,
        ROUND(sma5, 2) AS SMA5,
        ROUND(sma34, 2) AS SMA34,
        ROUND(sma5 - sma34, 2) AS AO,
        LAG(ROUND(sma5 - sma34, 2), 1) OVER (ORDER BY Date) AS prev_AO
    FROM ao_calc
),
ao_smooth AS (
    SELECT *,
        AVG(AO) OVER (ORDER BY Date ROWS 2 PRECEDING) AS AO_SMA3
    FROM ao_final
)
SELECT
    Date,
    Weekly_Median,
    SMA5,
    SMA34,
    AO,
    ROUND(AO_SMA3, 2) AS AO_SMA3,
    prev_AO,
    CASE
        WHEN AO > 0 AND prev_AO <= 0 THEN 'Bullish Crossover'
        WHEN AO < 0 AND prev_AO >= 0 THEN 'Bearish Crossover'
        ELSE 'No Crossover'
    END AS Crossover_Signal
FROM ao_smooth
ORDER BY Date DESC
LIMIT 30;
"""
        # Monthly AO
        if query_type == "monthly_ao":
            if all_symbols:
                self.query_feedback_label.setText("<span style='color:red'>Monthly AO for all symbols is not supported yet.</span>")
                return
            queries["monthly_ao"] = f"""
-- Monthly Awesome Oscillator (AO) with Crossover Signal and AO_SMA3 for {symbol}
WITH base AS (
    SELECT
        Date,
        (High + Low) / 2.0 AS median_price
    FROM {table(symbol)}
),
monthly AS (
    SELECT
        strftime('%Y-%m', Date) AS month,
        MAX(Date) AS month_end,
        AVG(median_price) AS monthly_median
    FROM base
    GROUP BY month
),
ao_calc AS (
    SELECT
        month_end AS Date,
        monthly_median,
        AVG(monthly_median) OVER (ORDER BY month_end ROWS 4 PRECEDING) AS sma5,
        AVG(monthly_median) OVER (ORDER BY month_end ROWS 33 PRECEDING) AS sma34
    FROM monthly
),
ao_final AS (
    SELECT
        Date,
        ROUND(monthly_median, 2) AS Monthly_Median,
        ROUND(sma5, 2) AS SMA5,
        ROUND(sma34, 2) AS SMA34,
        ROUND(sma5 - sma34, 2) AS AO,
        LAG(ROUND(sma5 - sma34, 2), 1) OVER (ORDER BY Date) AS prev_AO
    FROM ao_calc
),
ao_smooth AS (
    SELECT *,
        AVG(AO) OVER (ORDER BY Date ROWS 2 PRECEDING) AS AO_SMA3
    FROM ao_final
)
SELECT
    Date,
    Monthly_Median,
    SMA5,
    SMA34,
    AO,
    ROUND(AO_SMA3, 2) AS AO_SMA3,
    prev_AO,
    CASE
        WHEN AO > 0 AND prev_AO <= 0 THEN 'Bullish Crossover'
        WHEN AO < 0 AND prev_AO >= 0 THEN 'Bearish Crossover'
        ELSE 'No Crossover'
    END AS Crossover_Signal
FROM ao_smooth
ORDER BY Date DESC
LIMIT 30;
"""
        # Set query in input
        if query_type in queries:
            self.query_input.setPlainText(queries[query_type])
            self.query_feedback_label.setText(f"<span style='color:green'>Loaded {query_type.replace('_', ' ').title()} query for {'all symbols' if all_symbols else symbol}</span>")
        else:
            self.query_feedback_label.setText("<span style='color:red'>Unknown query type</span>")

        # --- In load_advanced_query, add support for 'active_signals' ---
        if query_type == "active_signals":
            selected_signal_symbol = self.signal_symbol_combo.currentText().strip().upper()
            queries["active_signals"] = f"""
-- Show all active BUY/SELL/NEUTRAL signals for {selected_signal_symbol} from KMI100 DB
SELECT Stock, Date, 'BUY' AS signal_type, * FROM buy_stocks WHERE Stock = '{selected_signal_symbol}'
UNION ALL
SELECT Stock, Date, 'SELL' AS signal_type, * FROM sell_stocks WHERE Stock = '{selected_signal_symbol}'
UNION ALL
SELECT Stock, Date, 'NEUTRAL' AS signal_type, * FROM neutral_stocks WHERE Stock = '{selected_signal_symbol}'
ORDER BY Date DESC
LIMIT 50;
"""

    def load_query_from_history(self):
        idx = self.query_history_combo.currentIndex()
        if 0 <= idx < len(self.query_history):
            self.query_input.setPlainText(self.query_history[idx])

    def execute_query(self):
        self.current_data = None
        query = self.query_input.toPlainText().strip()
        if not query:
            msg = "Please enter a query"
            self.query_feedback_label.setText(f"<span style='color:red'>{msg}</span>")
            self.logger.warning(msg)
            return
        try:
            with self.data_reader.engine.connect() as conn:
                self.logger.debug("Executing query", extra={
                    'context': {
                        'query': query,
                        'connection': str(conn.engine.url)
                    }
                })
                result = conn.execute(text(query))
                data = result.fetchall()
                columns = result.keys()
                self.results_table.setRowCount(len(data))
                self.results_table.setColumnCount(len(columns))
                self.results_table.setHorizontalHeaderLabels(columns)
                for row_idx, row in enumerate(data):
                    for col_idx, value in enumerate(row):
                        self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
                self.logger.info("Query executed successfully", extra={
                    'context': {
                        'query': query,
                        'result_count': len(data),
                        'columns': columns
                    }
                })
                self.query_feedback_label.setText(f"<span style='color:green'>Query executed successfully ({len(data)} rows)</span>")
                # Add to history
                if query not in self.query_history:
                    self.query_history.append(query)
                    self.query_history_combo.addItem(query[:60] + ("..." if len(query) > 60 else ""))
                # Enable visualize if numeric/time series
                import pandas as pd
                df = pd.DataFrame(data, columns=columns)
                self.query_result_df = df
                if not df.empty and (df.select_dtypes(include=['number']).shape[1] > 0 or 'Date' in df.columns):
                    self.visualize_query_button.setEnabled(True)
                else:
                    self.visualize_query_button.setEnabled(False)
        except Exception as e:
            error_msg = f"Query error: {str(e)}"
            self.query_feedback_label.setText(f"<span style='color:red'>{error_msg}</span>")
            self.logger.error(error_msg, exc_info=True)
            self.visualize_query_button.setEnabled(False)

    def export_query_results(self):
        if not hasattr(self, 'query_result_df') or self.query_result_df.empty:
            self.query_feedback_label.setText("<span style='color:red'>No query results to export</span>")
            return
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Export Query Results", "query_results.csv", "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)")
        if path:
            try:
                if path.endswith('.csv'):
                    self.query_result_df.to_csv(path, index=False)
                elif path.endswith('.xlsx'):
                    self.query_result_df.to_excel(path, index=False)
                elif path.endswith('.json'):
                    self.query_result_df.to_json(path, orient='records')
                self.query_feedback_label.setText(f"<span style='color:green'>Results exported to {path}</span>")
            except Exception as e:
                self.query_feedback_label.setText(f"<span style='color:red'>Export failed: {str(e)}</span>")

    def visualize_query_results(self):
        # Defensive: Try to use the latest data from the results table if query_result_df is empty
        if (not hasattr(self, 'query_result_df') or self.query_result_df.empty) and self.results_table.rowCount() > 0:
            columns = [self.results_table.horizontalHeaderItem(i).text() for i in range(self.results_table.columnCount())]
            data = []
            for row in range(self.results_table.rowCount()):
                data.append([
                    self.results_table.item(row, col).text() if self.results_table.item(row, col) else ""
                    for col in range(self.results_table.columnCount())
                ])
            import pandas as pd
            self.query_result_df = pd.DataFrame(data, columns=columns)

        if not hasattr(self, 'query_result_df') or self.query_result_df.empty:
            self.query_feedback_label.setText("<span style='color:red'>No query results to visualize</span>")
            return
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QTabWidget, QWidget

        df = self.query_result_df

        dialog = QDialog(self)
        dialog.setWindowTitle("Query Visualization & Data")
        vbox = QVBoxLayout()
        dialog.setLayout(vbox)

        # --- Plot Tab ---
        fig = plt.Figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        if 'Date' in df.columns and df.select_dtypes(include=['number']).shape[1] > 0:
            ycol = df.select_dtypes(include=['number']).columns[0]
            ax.plot(df['Date'], df[ycol], label=ycol)
            ax.set_xlabel('Date')
            ax.set_ylabel(ycol)
            ax.set_title(f'{ycol} over Date')
        elif df.shape[1] >= 2:
            ax.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'{df.columns[1]}')
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.set_title(f'{df.columns[1]} vs {df.columns[0]}')
        else:
            ax.text(0.5, 0.5, "Not enough data to plot.", ha='center', va='center')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        canvas = FigureCanvas(fig)

        # --- Table Tab ---
        table = QTableWidget()
        preview_df = df.head(100)  # Limit to 100 rows for preview
        table.setRowCount(len(preview_df))
        table.setColumnCount(len(preview_df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in preview_df.columns])
        for row_idx, row in preview_df.iterrows():
            for col_idx, col in enumerate(preview_df.columns):
                table.setItem(row_idx, col_idx, QTableWidgetItem(str(row[col])))

        # --- Tabs ---
        tabs = QTabWidget()
        plot_tab = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(canvas)
        plot_tab.setLayout(plot_layout)
        tabs.addTab(plot_tab, "Plot")

        table_tab = QWidget()
        table_layout = QVBoxLayout()
        table_layout.addWidget(table)
        table_tab.setLayout(table_layout)
        tabs.addTab(table_tab, "Data Preview")

        vbox.addWidget(tabs)
        dialog.resize(800, 600)
        dialog.exec_()

    def show_data(self, data):
        self.current_data = data
        self.preview_table.setRowCount(len(data))
        for row_idx, row in data.iterrows():
            # Format the date as string (YYYY-MM-DD if possible)
            date_str = str(row['Date'])
            try:
                import pandas as pd
                if isinstance(row['Date'], (pd.Timestamp, )):
                    date_str = row['Date'].strftime('%Y-%m-%d')
                elif isinstance(row['Date'], str) and len(row['Date']) > 10:
                    date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
            except Exception:
                pass
            self.preview_table.setItem(row_idx, 0, QTableWidgetItem(date_str))
            self.preview_table.setItem(row_idx, 1, QTableWidgetItem(str(row['Open'])))
            self.preview_table.setItem(row_idx, 2, QTableWidgetItem(str(row['High'])))
            self.preview_table.setItem(row_idx, 3, QTableWidgetItem(str(row['Low'])))
            self.preview_table.setItem(row_idx, 4, QTableWidgetItem(str(row['Close'])))
            self.preview_table.setItem(row_idx, 5, QTableWidgetItem(str(row['Volume'])))
        # --- Update statistics and chart ---
        if not data.empty:
            stats = data[['Open', 'High', 'Low', 'Close', 'Volume']].agg(['mean', 'median', 'min', 'max', 'std'])
            stats_html = '<b>Statistics:</b><br>' + stats.round(2).to_html()
            self.stats_label.setText(stats_html)
            # Plot Close price
            self.analysis_figure.clf()
            ax = self.analysis_figure.add_subplot(111)
            # Get symbol for title
            symbol = None
            if 'symbol' in data.columns:
                symbol = data['symbol'].iloc[0]
            elif hasattr(self, 'symbol_combo'):
                symbol = self.symbol_combo.currentText().strip().upper()
            # Sort data by Date ascending for correct time axis
            data_sorted = data.sort_values('Date', ascending=True)
            ax.plot(data_sorted['Date'], data_sorted['Close'], label='Close', color='royalblue')
            if symbol:
                ax.set_title(f'Close Price Over Time: {symbol}')
            else:
                ax.set_title('Close Price Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            self.analysis_figure.tight_layout()
            self.analysis_canvas.draw()
            # --- Export tab preview and columns ---
            if hasattr(self, 'export_preview_table') and hasattr(self, 'columns_list_widget'):
                self.export_preview_table.setRowCount(len(data))
                self.export_preview_table.setColumnCount(len(data.columns))
                self.export_preview_table.setHorizontalHeaderLabels(list(data.columns))
                for row_idx, row in data.iterrows():
                    for col_idx, col in enumerate(data.columns):
                        self.export_preview_table.setItem(row_idx, col_idx, QTableWidgetItem(str(row[col])))
                self.columns_list_widget.clear()
                for col in data.columns:
                    item = QListWidgetItem(col)
                    item.setCheckState(Qt.Checked)
                    self.columns_list_widget.addItem(item)
        else:
            self.stats_label.setText('<b>Statistics:</b> No data loaded.')
            self.analysis_figure.clf()
            self.analysis_canvas.draw()
            if hasattr(self, 'export_preview_table'):
                self.export_preview_table.setRowCount(0)
            if hasattr(self, 'columns_list_widget'):
                self.columns_list_widget.clear()
        # Always update preview after showing data
        self.preview_table.repaint()

    def export_analysis_chart(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Chart As", "chart.png", "PNG Files (*.png)")
        if path:
            self.analysis_figure.savefig(path)

    def filter_table(self, text):
        """Filter the preview table based on input text"""
        if not hasattr(self, 'preview_table'):
            return
            
        text = text.lower()
        for row in range(self.preview_table.rowCount()):
            match = False
            for col in range(self.preview_table.columnCount()):
                item = self.preview_table.item(row, col)
                if item and text in item.text().lower():
                    match = True
                    break
            self.preview_table.setRowHidden(row, not match)
            
    def clear_filters(self):
        """Clear all filters from the preview table"""
        if hasattr(self, 'preview_table'):
            for row in range(self.preview_table.rowCount()):
                self.preview_table.setRowHidden(row, False)
            if hasattr(self, 'filter_input'):
                self.filter_input.clear()

    def browse_db_path(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select Main Database", "", "SQLite Database (*.db)")
            if path:
                self.db_path_input.setText(path)
                self.logger.debug("Selected main database path", extra={
                    'context': {
                        'path': path
                    }
                })
        except Exception as e:
            self.logger.error("Failed to browse for main database", exc_info=True)
            
    def browse_alt_db_path(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select Alternative Database", "", "SQLite Database (*.db)")
            if path:
                self.alt_db_path_input.setText(path)
                self.logger.debug("Selected alternative database path", extra={
                    'context': {
                        'path': path
                    }
                })
        except Exception as e:
            self.logger.error("Failed to browse for alternative database", exc_info=True)
            
    def browse_symbols_file(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select Symbols File", "", "Excel Files (*.xlsx *.xls)")
            if path:
                self.symbols_file_input.setText(path)
                self.logger.debug("Selected symbols file path", extra={
                    'context': {
                        'path': path
                    }
                })
        except Exception as e:
            self.logger.error("Failed to browse for symbols file", exc_info=True)
            
    def browse_export_path(self):
        try:
            path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if path:
                self.export_path_input.setText(path)
                self.logger.debug("Selected export directory", extra={
                'context': {
                        'path': path
                    }
                })
        except Exception as e:
            self.logger.error("Failed to browse for export directory", exc_info=True)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        if hasattr(self, 'log_area'):
            self.log_area.append(f"<span style='color:red'>{message}</span>")

    def update_status(self, message):
        self.status_label.setText(message)
        if hasattr(self, 'log_area'):
            self.log_area.append(message)
        # --- Download History Logging ---
        if hasattr(self, 'download_history_panel'):
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.download_history_panel.append(f"[{ts}] {message}")
            # Keep only last 50 lines
            lines = self.download_history_panel.toPlainText().splitlines()
            if len(lines) > 50:
                self.download_history_panel.setPlainText('\n'.join(lines[-50:]))
        if message.startswith("BATCH_SUCCESS:"):
            self.batch_success += 1
            self.batch_processed += 1
            self.preview_available_data()  # Auto-refresh preview
        elif message.startswith("BATCH_NODATA:"):
            self.batch_no_data += 1
            self.batch_processed += 1
            self.preview_available_data()
        elif message.startswith("BATCH_FAILED:"):
            self.batch_failed += 1
            self.batch_processed += 1
            self.preview_available_data()
        # When all processed, show summary
        if self.batch_total > 0 and self.batch_processed == self.batch_total:
            self.show_batch_summary()
            self.show_spinner(False)
        
    def update_batch_progress(self, value):
        self.progress_bar.setValue(value)
        if hasattr(self, 'log_area'):
            self.log_area.append(f"Progress: {value}/{self.progress_bar.maximum()}")
        
    def parse_date(self, date_str):
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
            if parsed > date.today():
                self.show_error("Date cannot be in the future")
                return None
            return parsed
        except ValueError:
            return None
            
    def start_download(self):
        """Start single symbol download with validation and logging"""
        self.show_spinner(True)
        symbol = self.symbol_combo.currentText().strip().upper()
        
        # Validate symbol format (letters only)
        if not symbol or not symbol.isalpha():
            error_msg = "Please enter a valid stock symbol (letters only)"
            self.show_error(error_msg)
            self.logger.warning(error_msg, extra={
                'context': {
                    'input': self.symbol_combo.currentText(),
                    'operation': 'symbol validation'
                }
            })
            return
            
        start_date = self.parse_date(self.start_date_input.text())
        end_date = self.parse_date(self.end_date_input.text())
        
        if not start_date or not end_date:
            error_msg = "Please enter valid dates in YYYY-MM-DD format"
            self.show_error(error_msg)
            self.logger.warning(error_msg, extra={
                'context': {
                    'start_date_input': self.start_date_input.text(),
                    'end_date_input': self.end_date_input.text(),
                    'operation': 'date validation'
                }
            })
            return
            
        if start_date > end_date:
            error_msg = "Start date cannot be after end date"
            self.show_error(error_msg)
            self.logger.warning(error_msg, extra={
                'context': {
                    'start_date': str(start_date),
                    'end_date': str(end_date),
                    'operation': 'date validation'
                }
            })
            return
            
        if end_date > date.today():
            error_msg = "End date cannot be in the future"
            self.show_error(error_msg)
            self.logger.warning(error_msg, extra={
                'context': {
                    'end_date': str(end_date),
                    'current_date': str(date.today()),
                    'operation': 'date validation'
                }
            })
            return
            
        self.logger.info("Starting single symbol download", extra={
            'context': {
                'symbol': symbol,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'days': (end_date - start_date).days
            }
        })
            
        self.batch_total = 0
        self.batch_processed = 0
        
        self.worker = WorkerThread(self.data_reader, symbol, start_date, end_date)
        self.worker.signals.data.connect(self.show_data)
        self.worker.signals.error.connect(self.show_error)
        self.worker.signals.status.connect(self.update_status)
        self.worker.finished.connect(lambda: self.show_spinner(False))
        self.worker.start()
        
    def start_batch_download(self):
        """Download data for all symbols in the symbols file, strictly limiting to 5 concurrent downloads"""
        self.show_spinner(True)
        try:
            self.logger.info("Starting batch download", extra={
                'context': {
                    'symbols_file': self.config['symbols_file'],
                    'max_threads': 5  # Hardcoded to 5
                }
            })
            # Load symbols from Excel
            try:
                symbols_df = pd.read_excel(self.config['symbols_file'], sheet_name='KMIALL')
                valid_symbols = symbols_df.iloc[:, 0].tolist()
                self.logger.debug("Loaded symbols from file", extra={
                    'context': {
                        'file': self.config['symbols_file'],
                        'symbol_count': len(valid_symbols),
                        'first_symbol': valid_symbols[0] if valid_symbols else None
                    }
                })
            except Exception as e:
                error_msg = f"Failed to load symbols file: {str(e)}"
                self.show_error(error_msg)
                self.logger.error(error_msg, exc_info=True, extra={
                    'context': {
                        'file': self.config['symbols_file'],
                        'operation': 'symbols file loading'
                    }
                })
                return
            if not valid_symbols:
                error_msg = "No valid symbols found in file"
                self.show_error(error_msg)
                self.logger.warning(error_msg, extra={
                    'context': {
                        'file': self.config['symbols_file'],
                        'operation': 'symbol validation'
                    }
                })
                return
            # Setup progress tracking
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(valid_symbols))
            self.progress_bar.setValue(0)
            # Prepare queue and counters
            self.batch_success = 0
            self.batch_no_data = 0
            self.batch_skipped = 0
            self.batch_failed = 0
            self.batch_total = len(valid_symbols)
            self.batch_processed = 0
            self.batch_symbol_queue = deque()
            self.batch_active_workers = 0
            self.batch_max_workers = 5
            # Filter and queue valid symbols
            for i, symbol in enumerate(valid_symbols):
                if not symbol or not isinstance(symbol, str) or not symbol.isalpha():
                    self.logger.warning("Invalid symbol skipped", extra={
                        'context': {
                            'symbol': symbol,
                            'position': i,
                            'operation': 'symbol validation'
                        }
                    })
                    self.batch_skipped += 1
                    self.batch_processed += 1
                    continue
                self.batch_symbol_queue.append(symbol)
            self.thread_pool = QThreadPool()
            self.thread_pool.setMaxThreadCount(self.batch_max_workers)
            self.logger.info("Batch download started (strict 5 concurrent)", extra={
                'context': {
                    'valid_symbols': len(self.batch_symbol_queue),
                    'invalid_symbols': self.batch_skipped
                }
            })
            # Start up to 5 workers
            for _ in range(min(self.batch_max_workers, len(self.batch_symbol_queue))):
                self._start_next_batch_worker()
        except Exception as e:
            error_msg = f"Failed to start batch download: {str(e)}"
            self.show_error(error_msg)
            self.logger.error(error_msg, exc_info=True, extra={
                'context': {
                    'symbols_file': self.config['symbols_file'],
                    'max_threads': self.batch_max_workers,
                    'operation': 'batch download'
                }
            })

    def _start_next_batch_worker(self):
        if self.batch_symbol_queue and self.batch_active_workers < self.batch_max_workers:
            symbol = self.batch_symbol_queue.popleft()
            worker = BatchWorker(self.data_reader, symbol)
            worker.signals.progress.connect(self.update_batch_progress)
            worker.signals.status.connect(self._batch_worker_status_handler)
            worker.signals.error.connect(self.show_error)
            self.batch_active_workers += 1
            self.thread_pool.start(worker)

    def _batch_worker_status_handler(self, message):
        # Update status and handle worker completion
        self.update_status(message)
        # Only count as finished if it's a terminal status
        if (message.startswith("BATCH_SUCCESS:") or
            message.startswith("BATCH_NODATA:") or
            message.startswith("BATCH_FAILED:")):
            self.batch_active_workers -= 1
            # Start next worker if any remain
            self._start_next_batch_worker()

    def show_batch_summary(self):
        if hasattr(self, 'log_area'):
            summary = (f"<b>Batch download complete.</b><br>"
                       f"Success: {self.batch_success}<br>"
                       f"No Data: {self.batch_no_data}<br>"
                       f"Skipped: {self.batch_skipped}<br>"
                       f"Failed: {self.batch_failed}<br>"
                       f"Total: {self.batch_total}")
            self.log_area.append(summary)
            QMessageBox.information(self, "Batch Complete", summary.replace('<br>', '\n'))

    def apply_theme(self, theme):
        try:
            if theme == "dark":
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: #23272e;
                        color: #f8f8f2;
                        font-family: 'Segoe UI', 'Arial', sans-serif;
                        font-size: 12pt;
                    }
                    QTabWidget::pane {
                        border: 1px solid #444;
                        background: #23272e;
                        border-radius: 8px;
                    }
                    QTabBar::tab {
                        background: #2d323b;
                        color: #f8f8f2;
                        padding: 8px 16px;
                        border-top-left-radius: 8px;
                        border-top-right-radius: 8px;
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected {
                        background: #3a3f4b;
                        font-weight: bold;
                    }
                    QLineEdit, QTextEdit, QComboBox {
                        background: #2d323b;
                        color: #f8f8f2;
                        border: 1px solid #555;
                        border-radius: 6px;
                        padding: 4px;
                    }
                    QTableWidget {
                        background: #2d323b;
                        color: #f8f8f2;
                        gridline-color: #555;
                        border-radius: 6px;
                        alternate-background-color: #23272e;
                    }
                    QHeaderView::section {
                        background-color: #3a3f4b;
                        color: #f8f8f2;
                        font-weight: bold;
                        border-radius: 6px;
                    }
                    QPushButton {
                        background-color: #3a3f4b;
                        color: #f8f8f2;
                        border-radius: 6px;
                        padding: 6px 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #4e5462;
                    }
                    QLabel {
                        color: #f8f8f2;
                    }
                """)
                self.logger.info("Applied dark theme")
            elif theme == "light":
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: #f5f6fa;
                        color: #23272e;
                        font-family: 'Segoe UI', 'Arial', sans-serif;
                        font-size: 12pt;
                    }
                    QTabWidget::pane {
                        border: 1px solid #bbb;
                        background: #f5f6fa;
                        border-radius: 8px;
                    }
                    QTabBar::tab {
                        background: #e1e3ea;
                        color: #23272e;
                        padding: 8px 16px;
                        border-top-left-radius: 8px;
                        border-top-right-radius: 8px;
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected {
                        background: #d1d3db;
                        font-weight: bold;
                    }
                    QLineEdit, QTextEdit, QComboBox {
                        background: #e1e3ea;
                        color: #23272e;
                        border: 1px solid #bbb;
                        border-radius: 6px;
                        padding: 4px;
                    }
                    QTableWidget {
                        background: #e1e3ea;
                        color: #23272e;
                        gridline-color: #bbb;
                        border-radius: 6px;
                        alternate-background-color: #f5f6fa;
                    }
                    QHeaderView::section {
                        background-color: #d1d3db;
                        color: #23272e;
                        font-weight: bold;
                        border-radius: 6px;
                    }
                    QPushButton {
                        background-color: #d1d3db;
                        color: #23272e;
                        border-radius: 6px;
                        padding: 6px 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #bfc2cc;
                    }
                    QLabel {
                        color: #23272e;
                    }
                """)
                self.logger.info("Applied light theme")
            elif theme == "solarized":
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: #fdf6e3;
                        color: #657b83;
                        font-family: 'Fira Sans', 'Segoe UI', 'Arial', sans-serif;
                        font-size: 12pt;
                    }
                    QTabWidget::pane {
                        border: 1px solid #eee8d5;
                        background: #fdf6e3;
                        border-radius: 8px;
                    }
                    QTabBar::tab {
                        background: #eee8d5;
                        color: #657b83;
                        padding: 8px 16px;
                        border-top-left-radius: 8px;
                        border-top-right-radius: 8px;
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected {
                        background: #e1dbcd;
                        font-weight: bold;
                    }
                    QLineEdit, QTextEdit, QComboBox {
                        background: #eee8d5;
                        color: #657b83;
                        border: 1px solid #93a1a1;
                        border-radius: 6px;
                        padding: 4px;
                    }
                    QTableWidget {
                        background: #eee8d5;
                        color: #657b83;
                        gridline-color: #93a1a1;
                        border-radius: 6px;
                        alternate-background-color: #fdf6e3;
                    }
                    QHeaderView::section {
                        background-color: #e1dbcd;
                        color: #657b83;
                        font-weight: bold;
                        border-radius: 6px;
                    }
                    QPushButton {
                        background-color: #e1dbcd;
                        color: #657b83;
                        border-radius: 6px;
                        padding: 6px 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #d6cbb2;
                    }
                    QLabel {
                        color: #657b83;
                    }
                """)
                self.logger.info("Applied solarized theme")
            elif theme == "high-contrast":
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: #000000;
                        color: #ffffff;
                        font-family: 'Arial', 'Segoe UI', sans-serif;
                        font-size: 13pt;
                    }
                    QTabWidget::pane {
                        border: 2px solid #fff;
                        background: #000;
                        border-radius: 8px;
                    }
                    QTabBar::tab {
                        background: #222;
                        color: #fff;
                        padding: 8px 16px;
                        border-top-left-radius: 8px;
                        border-top-right-radius: 8px;
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected {
                        background: #444;
                        font-weight: bold;
                    }
                    QLineEdit, QTextEdit, QComboBox {
                        background: #111;
                        color: #fff;
                        border: 2px solid #fff;
                        border-radius: 6px;
                        padding: 4px;
                    }
                    QTableWidget {
                        background: #111;
                        color: #fff;
                        gridline-color: #fff;
                        border-radius: 6px;
                        alternate-background-color: #222;
                    }
                    QHeaderView::section {
                        background-color: #444;
                        color: #fff;
                        font-weight: bold;
                        border-radius: 6px;
                    }
                    QPushButton {
                        background-color: #444;
                        color: #fff;
                        border-radius: 6px;
                        padding: 6px 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #666;
                    }
                    QLabel {
                        color: #fff;
                    }
                """)
                self.logger.info("Applied high-contrast theme")
            elif theme == "green":
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: #e8f5e9;
                        color: #1b5e20;
                        font-family: 'Segoe UI', 'Arial', sans-serif;
                        font-size: 12pt;
                    }
                    QTabWidget::pane {
                        border: 1px solid #388e3c;
                        background: #e8f5e9;
                        border-radius: 8px;
                    }
                    QTabBar::tab {
                        background: #a5d6a7;
                        color: #1b5e20;
                        padding: 8px 16px;
                        border-top-left-radius: 8px;
                        border-top-right-radius: 8px;
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected {
                        background: #66bb6a;
                        font-weight: bold;
                    }
                    QLineEdit, QTextEdit, QComboBox {
                        background: #c8e6c9;
                        color: #1b5e20;
                        border: 1px solid #388e3c;
                        border-radius: 6px;
                        padding: 4px;
                    }
                    QTableWidget {
                        background: #c8e6c9;
                        color: #1b5e20;
                        gridline-color: #388e3c;
                        border-radius: 6px;
                        alternate-background-color: #e8f5e9;
                    }
                    QHeaderView::section {
                        background-color: #66bb6a;
                        color: #1b5e20;
                        font-weight: bold;
                        border-radius: 6px;
                    }
                    QPushButton {
                        background-color: #66bb6a;
                        color: #1b5e20;
                        border-radius: 6px;
                        padding: 6px 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #388e3c;
                        color: #fff;
                    }
                    QLabel {
                        color: #1b5e20;
                    }
                """)
                self.logger.info("Applied green theme")
            elif theme == "blue":
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: #e3f2fd;
                        color: #0d47a1;
                        font-family: 'Segoe UI', 'Arial', sans-serif;
                        font-size: 12pt;
                    }
                    QTabWidget::pane {
                        border: 1px solid #1976d2;
                        background: #e3f2fd;
                        border-radius: 8px;
                    }
                    QTabBar::tab {
                        background: #90caf9;
                        color: #0d47a1;
                        padding: 8px 16px;
                        border-top-left-radius: 8px;
                        border-top-right-radius: 8px;
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected {
                        background: #42a5f5;
                        font-weight: bold;
                    }
                    QLineEdit, QTextEdit, QComboBox {
                        background: #bbdefb;
                        color: #0d47a1;
                        border: 1px solid #1976d2;
                        border-radius: 6px;
                        padding: 4px;
                    }
                    QTableWidget {
                        background: #bbdefb;
                        color: #0d47a1;
                        gridline-color: #1976d2;
                        border-radius: 6px;
                        alternate-background-color: #e3f2fd;
                    }
                    QHeaderView::section {
                        background-color: #42a5f5;
                        color: #0d47a1;
                        font-weight: bold;
                        border-radius: 6px;
                    }
                    QPushButton {
                        background-color: #42a5f5;
                        color: #0d47a1;
                        border-radius: 6px;
                        padding: 6px 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #1976d2;
                        color: #fff;
                    }
                    QLabel {
                        color: #0d47a1;
                    }
                """)
                self.logger.info("Applied blue theme")
            elif theme == "windows-default":
                self.setStyleSheet("")
                from PyQt5.QtGui import QFont
                QApplication.setFont(QFont())
                self.logger.info("Applied Windows/system default theme and font")
            else:
                self.setStyleSheet("")
            # Save theme to config
            self.config['theme'] = theme
            self.config_manager.save_config(self.config)
        except Exception as e:
            self.logger.error("Failed to apply theme", exc_info=True, extra={
                'context': {
                    'theme': theme,
                    'config': self.config
                }
            })

    def apply_font(self, font_family, font_size):
        from PyQt5.QtGui import QFont
        font = QFont(font_family, font_size)
        QApplication.setFont(font)
        self.config['font_family'] = font_family
        self.config['font_size'] = font_size
        self.config_manager.save_config(self.config)
        self.logger.info(f"Applied font: {font_family} {font_size}pt")

    def create_export_tab(self):
        self.export_tab = QWidget()
        self.tab_widget.addTab(self.export_tab, "Export")
        layout = QVBoxLayout()
        self.export_tab.setLayout(layout)
        # --- Export Options Group ---
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        # Format
        format_hbox = QHBoxLayout()
        format_hbox.addWidget(QLabel("Export Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "Excel", "JSON"])
        format_hbox.addWidget(self.format_combo)
        options_layout.addLayout(format_hbox)
        # Export path
        path_hbox = QHBoxLayout()
        path_hbox.addWidget(QLabel("Export Path:"))
        self.export_path_input = QLineEdit(self.config['export_path'])
        path_hbox.addWidget(self.export_path_input)
        browse_export_button = QPushButton("Browse")
        browse_export_button.clicked.connect(self.browse_export_path)
        path_hbox.addWidget(browse_export_button)
        options_layout.addLayout(path_hbox)
        layout.addWidget(options_group)
        # --- Data Preview Group ---
        preview_group = QGroupBox("Data Preview & Column Selection")
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)
        self.export_preview_table = QTableWidget()
        preview_layout.addWidget(self.export_preview_table)
        self.column_checkboxes = []
        preview_layout.addWidget(QLabel("Select columns to export:"))
        self.columns_list_widget = QListWidget()
        preview_layout.addWidget(self.columns_list_widget)
        layout.addWidget(preview_group)
        # --- Export Button & Feedback ---
        export_button = QPushButton("Export Data")
        export_button.clicked.connect(self.export_data)
        layout.addWidget(export_button)
        self.export_feedback_label = QLabel()
        layout.addWidget(self.export_feedback_label)

    def export_data(self):
        # Use current_data for export
        if not hasattr(self, 'current_data') or self.current_data.empty:
            msg = "No data available to export"
            self.export_feedback_label.setText(f"<span style='color:red'>{msg}</span>")
            self.logger.warning(msg)
            return
        export_path = self.export_path_input.text()
        if not export_path:
            msg = "Please select an export path"
            self.export_feedback_label.setText(f"<span style='color:red'>{msg}</span>")
            self.logger.warning(msg)
            return
        os.makedirs(export_path, exist_ok=True)
        format = self.format_combo.currentText().lower()
        filename = f"psx_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        full_path = os.path.join(export_path, filename)
        # Get selected columns
        selected_columns = [item.text() for item in self.columns_list_widget.selectedItems()]
        data_to_export = self.current_data[selected_columns] if selected_columns else self.current_data
        try:
            if format == "csv":
                full_path += ".csv"
                data_to_export.to_csv(full_path, index=False)
            elif format == "excel":
                full_path += ".xlsx"
                data_to_export.to_excel(full_path, index=False)
            elif format == "json":
                full_path += ".json"
                data_to_export.to_json(full_path, orient='records')
            success_msg = f"Data exported successfully to {full_path}"
            self.export_feedback_label.setText(f"<span style='color:green'>{success_msg}</span>")
            self.logger.info("Data exported", extra={
                    'context': {
                    'path': full_path,
                    'format': format,
                    'record_count': len(data_to_export)
                    }
                })
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.export_feedback_label.setText(f"<span style='color:red'>{error_msg}</span>")
            self.logger.error(error_msg, exc_info=True, extra={
                    'context': {
                    'path': full_path,
                    'format': format,
                    'data_size': len(data_to_export) if hasattr(self, 'current_data') else 0
                }
            })

    def check_database_health(self):
        import re
        from datetime import date, timedelta
        try:
            with self.data_reader.engine.connect() as conn:
                # Get all tables
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result.fetchall()]
                stock_tables = [t for t in tables if re.match(r'^PSX_[A-Z]+_stock_data$', t)]
                health_data = []
                healthy = 0
                outdated = 0
                nodata = 0
                yesterday = date.today() - timedelta(days=1)
                for table in stock_tables:
                    symbol = table.replace('PSX_', '').replace('_stock_data', '')
                    try:
                        res = conn.execute(text(f"SELECT COUNT(*) as cnt, MIN(Date) as min_date, MAX(Date) as max_date FROM {table}"))
                        row = res.fetchone()
                        count = row[0] if row else 0
                        min_date = row[1] if row and row[1] else None
                        max_date = row[2] if row and row[2] else None
                        if count == 0 or not min_date or not max_date:
                            up_to_date = "No"
                            status = "No data"
                            nodata += 1
                        else:
                            # Check if max_date is up to yesterday
                            try:
                                max_date_obj = pd.to_datetime(max_date).date()
                                if max_date_obj >= yesterday:
                                    up_to_date = "Yes"
                                    status = "Healthy"
                                    healthy += 1
                                else:
                                    up_to_date = "No"
                                    status = "Outdated"
                                    outdated += 1
                            except Exception:
                                up_to_date = "?"
                                status = "Error"
                                nodata += 1
                        health_data.append((symbol, count, min_date, max_date, up_to_date, status))
                    except Exception as e:
                        health_data.append((symbol, 0, None, None, "?", f"Error: {str(e)}"))
                        nodata += 1
                # Show in table
                self.health_table.setRowCount(len(health_data))
                for i, (symbol, count, min_date, max_date, up_to_date, status) in enumerate(health_data):
                    self.health_table.setItem(i, 0, QTableWidgetItem(str(symbol)))
                    self.health_table.setItem(i, 1, QTableWidgetItem(str(count)))
                    self.health_table.setItem(i, 2, QTableWidgetItem(str(min_date) if min_date else "-"))
                    self.health_table.setItem(i, 3, QTableWidgetItem(str(max_date) if max_date else "-"))
                    up_to_date_item = QTableWidgetItem(str(up_to_date))
                    status_item = QTableWidgetItem(str(status))
                    # Color code
                    if status == "Healthy":
                        up_to_date_item.setBackground(Qt.green)
                        status_item.setBackground(Qt.green)
                    elif status == "Outdated":
                        up_to_date_item.setBackground(Qt.yellow)
                        status_item.setBackground(Qt.yellow)
                    else:
                        up_to_date_item.setBackground(Qt.red)
                        status_item.setBackground(Qt.red)
                    self.health_table.setItem(i, 4, up_to_date_item)
                    self.health_table.setItem(i, 5, status_item)
                    # Add Download button
                    btn = QPushButton("Download")
                    btn.clicked.connect(lambda _, sym=symbol: self.download_symbol_from_health_tab(sym))
                    self.health_table.setCellWidget(i, 6, btn)
                # Summary
                total = len(health_data)
                summary = (f"Total symbols: {total} | Healthy: {healthy} | Outdated: {outdated} | No data/Error: {nodata}")
                self.health_summary_label.setText(summary)
                # Update pie chart
                self.update_health_pie_chart(healthy, outdated, nodata)
        except Exception as e:
            self.health_summary_label.setText(f"Error checking database health: {str(e)}")
            self.update_health_pie_chart(0, 0, 0)

    def update_health_pie_chart(self, healthy, outdated, nodata):
        ax = self.health_pie_canvas.figure.subplots()
        self.health_pie_canvas.figure.clf()
        ax = self.health_pie_canvas.figure.add_subplot(111)
        labels = []
        sizes = []
        colors = []
        if healthy > 0:
            labels.append('Healthy')
            sizes.append(healthy)
            colors.append('green')
        if outdated > 0:
            labels.append('Outdated')
            sizes.append(outdated)
            colors.append('gold')
        if nodata > 0:
            labels.append('No Data/Error')
            sizes.append(nodata)
            colors.append('red')
        if not sizes:
            labels = ['No Data']
            sizes = [1]
            colors = ['grey']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90, counterclock=False)
        ax.set_title('Database Health')
        self.health_pie_canvas.draw()

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About PSX Data Downloader",
            f"""<b>PSX Data Downloader</b><br>
            Version 1.0.0<br><br>
            A tool for downloading and analyzing PSX stock data<br><br>
            © 2025 All rights reserved""")

    def run_quick_analysis(self):
        symbol = self.quick_symbol_input.text().strip().upper()
        if not symbol or not symbol.isalpha():
            self.query_feedback_label.setText("<span style='color:red'>Please enter a valid symbol for quick analysis.</span>")
            return
        # Set the symbol in the query tab
        self.query_symbol_combo.setCurrentText(symbol)
        # Load RSI analysis for this symbol
        self.load_advanced_query("rsi")
        # Execute the query
        self.execute_query()

    def create_help_tab(self):
        self.help_tab = QWidget()
        self.tab_widget.addTab(self.help_tab, "Help")
        layout = QVBoxLayout()
        self.help_tab.setLayout(layout)

        # --- Interactive navigation links ---
        nav_links = (
            '<h2>PSX Data Downloader - Help & User Guide</h2>'
            '<p><b>Quick Navigation:</b> '
            '<a href="tab:psxdata">PSX Data</a> | '
            '<a href="tab:stockanalysis">Stock Analysis</a> | '
            '<a href="tab:signals">Signals</a> | '
            '<a href="tab:export">Export</a> | '
            '<a href="tab:dbhealth">DB Health</a> | '
            '<a href="tab:config">Configuration</a> | '
            '<a href="action:about">About</a>'
            '</p>'
        )
        nav_label = QLabel(nav_links)
        nav_label.setTextFormat(Qt.RichText)
        nav_label.setOpenExternalLinks(False)
        nav_label.linkActivated.connect(self._help_link_activated)
        layout.addWidget(nav_label)

        # --- Main help text ---
        help_text = QLabel(
            """
            <ul>
                <li><b>PSX Data Tab:</b>
                    <ul>
                        <li>Select a stock symbol (auto-complete supported) and date range to preview and download data.</li>
                        <li>Batch Download: Download data for all symbols in the list with progress tracking and download history.</li>
                        <li>Preview Available Data: See a table of available data for the selected symbol and range.</li>
                        <li>Statistics & Chart: View summary statistics and a Close price chart for the previewed data.</li>
                        <li>Filter: Quickly filter previewed data by text.</li>
                        <li>Download History: View recent download events in a collapsible panel.</li>
                        <li>Progress Spinner: Visual feedback during downloads.</li>
                    </ul>
                </li>
                <li><b>Stock Analysis Tab:</b>
                    <ul>
                        <li>Run advanced SQL queries on your stock data.</li>
                        <li>Choose a symbol or run queries for all symbols.</li>
                        <li>Use built-in advanced queries: RSI (weekly avg), moving averages, volume, price patterns, volatility, performance, risk, buy/sell/hold signals, and more.</li>
                        <li>Quick Analysis: Instantly run an analysis for a specific symbol.</li>
                        <li>Query History: Access and reuse previous queries.</li>
                        <li>Visualize Results: Plot query results with interactive charts and data preview.</li>
                        <li>Export Results: Save query results as CSV, Excel, or JSON.</li>
                        <li>Clear Results: Quickly clear the results table.</li>
                        <li>Signal Symbol Analysis: Analyze buy/sell/neutral signals from the KMI100 DB.</li>
                    </ul>
                </li>
                <li><b>DB Health Tab:</b>
                    <ul>
                        <li>Check the health of all stock data tables in your database.</li>
                        <li>See which symbols are up-to-date, outdated, or missing data.</li>
                        <li>Pie chart visualization of health status.</li>
                        <li>One-click Download: Download missing/outdated data directly from this tab.</li>
                        <li>Refresh Health Check: Update the health status at any time.</li>
                    </ul>
                </li>
                <li><b>Export Tab:</b>
                    <ul>
                        <li>Export previewed data to CSV, Excel, or JSON.</li>
                        <li>Select which columns to export and preview the data before saving.</li>
                        <li>Choose export path and format.</li>
                        <li>Export feedback and error messages shown inline.</li>
                    </ul>
                </li>
                <li><b>Configuration Tab:</b>
                    <ul>
                        <li>Set database paths, symbols file, and export directory.</li>
                        <li>Test database connections and browse for files.</li>
                        <li>Choose application theme (Dark, Light, Solarized, High Contrast).</li>
                        <li>Select font family and size for the entire app.</li>
                        <li>Save or reset configuration to defaults.</li>
                    </ul>
                </li>
                <li><b>General Features:</b>
                    <ul>
                        <li>Modern, theme-aware interface with live theme/font switching.</li>
                        <li>Persistent configuration and user preferences.</li>
                        <li>Comprehensive error log panel for troubleshooting.</li>
                        <li>Keyboard shortcuts for common actions (where available).</li>
                        <li>Tooltips on all major controls for guidance.</li>
                        <li>Accessibility: High-contrast mode and scalable fonts.</li>
                    </ul>
                </li>
            </ul>
            <h3>Need More Help?</h3>
            <ul>
                <li>Refer to the <b>user guide</b> in the documentation folder for detailed walkthroughs.</li>
                <li>Contact support or open an issue on GitHub for further assistance.</li>
            </ul>
            """
        )
        help_text.setWordWrap(True)
        help_text.setTextFormat(Qt.RichText)
        layout.addWidget(help_text)

    def _help_link_activated(self, link):
        # Map link targets to tab indices or actions
        tab_map = {
            'psxdata': 'PSX Data',
            'stockanalysis': 'Stock Analysis',
            'signals': 'Signals',
            'export': 'Export',
            'dbhealth': 'DB Health',
            'config': 'Configuration',
        }
        if link.startswith('tab:'):
            tab_name = tab_map.get(link[4:], None)
            if tab_name:
                for i in range(self.tab_widget.count()):
                    if self.tab_widget.tabText(i) == tab_name:
                        self.tab_widget.setCurrentIndex(i)
                        break
        elif link == 'action:about':
            self.show_about()

    def create_db_health_tab(self):
        self.db_health_tab = QWidget()
        self.tab_widget.addTab(self.db_health_tab, "DB Health")
        layout = QVBoxLayout()
        self.db_health_tab.setLayout(layout)

        self.health_table = QTableWidget()
        self.health_table.setColumnCount(7)
        self.health_table.setHorizontalHeaderLabels([
            "Symbol", "Records", "Min Date", "Max Date", "Up-to-date", "Status", "Download"
        ])
        layout.addWidget(self.health_table)

        self.health_summary_label = QLabel()
        layout.addWidget(self.health_summary_label)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt
        self.health_pie_canvas = FigureCanvas(plt.Figure(figsize=(3, 3)))
        layout.addWidget(self.health_pie_canvas)

        refresh_button = QPushButton("Refresh Health Check")
        refresh_button.clicked.connect(self.check_database_health)
        layout.addWidget(refresh_button)

        # Initial check
        self.check_database_health()

    def download_symbol_from_health_tab(self, symbol):
        # Switch to Download tab
        self.tab_widget.setCurrentWidget(self.download_tab)
        # Set the symbol in the Download tab
        self.symbol_combo.setCurrentText(symbol)
        # Optionally, set date range to default or last available
        # Start the download using existing logic
        self.start_download()

    def browse_indicators_db_path(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select Indicators Database", "", "SQLite Database (*.db)")
            if path:
                self.indicators_db_path_input.setText(path)
                self.logger.debug("Selected indicators database path", extra={
                    'context': {
                        'path': path
                    }
                })
        except Exception as e:
            self.logger.error("Failed to browse for indicators database", exc_info=True)
            
    def browse_signals_db_path(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select Signals Database", "", "SQLite Database (*.db)")
            if path:
                self.signals_db_path_input.setText(path)
                self.logger.debug("Selected signals database path", extra={
                    'context': {
                        'path': path
                    }
                })
        except Exception as e:
            self.logger.error("Failed to browse for signals database", exc_info=True)

    def load_signal_symbols_into_combo(self):
        self.signal_symbol_combo.clear()
        import sqlite3
        db_path = 'data/databases/production/PSX_investing_Stocks_KMI100.db'
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            symbols = set()
            for t in ['buy_stocks', 'sell_stocks', 'neutral_stocks']:
                if t in tables:
                    cursor.execute(f"SELECT DISTINCT Stock FROM {t}")
                    symbols.update([row[0] for row in cursor.fetchall() if row[0]])
            conn.close()
            self.signal_symbol_combo.addItems(sorted(symbols))
        except Exception as e:
            self.signal_symbol_combo.addItem("(Error loading symbols)")
            print(f"Error loading signal symbols: {e}")
            if hasattr(self, 'log_area'):
                self.log_area.append(f"<span style='color:red'>Error loading signal symbols: {e}</span>")

    def show_spinner(self, show):
        if hasattr(self, 'spinner_overlay'):
            self.spinner_overlay.setVisible(show)
            if show:
                self.spinner_movie.start()
            else:
                self.spinner_movie.stop()

    def create_signals_tab(self):
        import sqlite3
        from PyQt5.QtWidgets import QTabWidget
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        import plotly.graph_objs as go
        import plotly.express as px
        import tempfile
        import os
        from PyQt5.QtWebChannel import QWebChannel
        from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
        from PyQt5.QtWidgets import QGridLayout, QComboBox, QLabel, QPushButton, QHBoxLayout, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QSizePolicy
        from PyQt5.QtWidgets import QFrame
        self.signals_tab = QWidget()
        self.tab_widget.addTab(self.signals_tab, "Signals")
        layout = QVBoxLayout()
        self.signals_tab.setLayout(layout)
        # --- Symbol selection ---
        symbol_layout = QHBoxLayout()
        symbol_label = QLabel("Symbol:")
        symbol_layout.addWidget(symbol_label)
        self.signals_symbol_combo = QComboBox()
        self.signals_symbol_combo.setEditable(True)
        self.signals_symbol_combo.addItem("All Symbols")
        db_path = self.config.get('signals_db_path', 'data/databases/production/PSX_investing_Stocks_KMI100.db')
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            symbols = set()
            for t in ['buy_stocks', 'sell_stocks', 'neutral_stocks']:
                cursor.execute(f"SELECT DISTINCT Stock FROM {t}")
                symbols.update([row[0] for row in cursor.fetchall() if row[0]])
            conn.close()
            for s in sorted(symbols):
                self.signals_symbol_combo.addItem(s)
        except Exception as e:
            self.signals_symbol_combo.addItem("(Error loading symbols)")
        symbol_layout.addWidget(self.signals_symbol_combo)
        reload_btn = QPushButton("Reload Symbols")
        reload_btn.clicked.connect(self.create_signals_tab)
        symbol_layout.addWidget(reload_btn)
        layout.addLayout(symbol_layout)
        # --- Date range filter ---
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Start Date:"))
        self.signals_start_date = QLineEdit()
        self.signals_start_date.setPlaceholderText("YYYY-MM-DD")
        date_layout.addWidget(self.signals_start_date)
        date_layout.addWidget(QLabel("End Date:"))
        self.signals_end_date = QLineEdit()
        self.signals_end_date.setPlaceholderText("YYYY-MM-DD")
        date_layout.addWidget(self.signals_end_date)
        filter_btn = QPushButton("Apply Filter")
        filter_btn.clicked.connect(self.load_signals_data)
        date_layout.addWidget(filter_btn)
        layout.addLayout(date_layout)
        # --- Text filter ---
        text_filter_layout = QHBoxLayout()
        text_filter_layout.addWidget(QLabel("Filter:"))
        self.signals_text_filter = QLineEdit()
        self.signals_text_filter.setPlaceholderText("Type to filter signals table...")
        self.signals_text_filter.textChanged.connect(self.filter_signals_table)
        text_filter_layout.addWidget(self.signals_text_filter)
        clear_filter_btn = QPushButton("Clear")
        clear_filter_btn.clicked.connect(lambda: self.signals_text_filter.setText(""))
        text_filter_layout.addWidget(clear_filter_btn)
        layout.addLayout(text_filter_layout)
        # --- Unified signals table ---
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(19)
        self.signals_table.setHorizontalHeaderLabels([
            "Stock", "Signal Type", "Date", "Close", "Volume", "RSI_Weekly_Avg", "AO_Weekly", "MA_30", "Multibagger", "FreeFloatRatio", "% P/L", "Signal_Date", "Signal_Close", "Holding_Days", "Status", "Update_Date", "Trend_Direction", "Score", "Details"
        ])
        self.signals_table.setSortingEnabled(True)
        self.signals_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.signals_table.setAlternatingRowColors(True)
        self.signals_table.cellDoubleClicked.connect(self.show_signal_details)
        layout.addWidget(self.signals_table)
        # --- New: Signal History/Transitions Tab ---
        self.signal_history_tab = QWidget()
        self.signal_history_layout = QVBoxLayout()
        self.signal_history_tab.setLayout(self.signal_history_layout)
        self.signal_history_table = QTableWidget()
        self.signal_history_table.setColumnCount(8)
        self.signal_history_table.setHorizontalHeaderLabels([
            "Stock", "From Signal", "To Signal", "Entry Date", "Exit Date", "Holding Days", "% P/L", "Details"
        ])
        self.signal_history_layout.addWidget(self.signal_history_table)
        self.signal_history_refresh_btn = QPushButton("Refresh Signal History")
        self.signal_history_refresh_btn.clicked.connect(self.load_signal_history)
        self.signal_history_layout.addWidget(self.signal_history_refresh_btn)
        # Add the new tab to the signals_chart_tabs if it exists, else to layout
        if hasattr(self, 'signals_chart_tabs'):
            self.signals_chart_tabs.addTab(self.signal_history_tab, "Signal History")
        else:
            layout.addWidget(self.signal_history_tab)
        # --- Analytics and Visualizations ---
        analytics_group = QGroupBox("Signal Analytics & Backtest")
        analytics_layout = QVBoxLayout()
        analytics_group.setLayout(analytics_layout)
        self.signals_summary_label = QLabel()
        analytics_layout.addWidget(self.signals_summary_label)
        # Backtest summary
        self.signals_backtest_label = QLabel()
        analytics_layout.addWidget(self.signals_backtest_label)
        # Tabbed charts
        self.signals_chart_tabs = QTabWidget()
        # Chart 1: Distribution
        self.signals_chart_figure = plt.Figure(figsize=(4,2))
        self.signals_chart_canvas = FigureCanvas(self.signals_chart_figure)
        chart1 = QWidget()
        chart1_layout = QVBoxLayout()
        chart1_layout.addWidget(self.signals_chart_canvas)
        chart1.setLayout(chart1_layout)
        self.signals_chart_tabs.addTab(chart1, "Signal Distribution")
        # Chart 2: P/L over time
        self.signals_pl_figure = plt.Figure(figsize=(4,2))
        self.signals_pl_canvas = FigureCanvas(self.signals_pl_figure)
        chart2 = QWidget()
        chart2_layout = QVBoxLayout()
        chart2_layout.addWidget(self.signals_pl_canvas)
        chart2.setLayout(chart2_layout)
        self.signals_chart_tabs.addTab(chart2, "P/L Over Time")
        # Chart 3: Score histogram
        self.signals_score_figure = plt.Figure(figsize=(4,2))
        self.signals_score_canvas = FigureCanvas(self.signals_score_figure)
        chart3 = QWidget()
        chart3_layout = QVBoxLayout()
        chart3_layout.addWidget(self.signals_score_canvas)
        chart3.setLayout(chart3_layout)
        self.signals_chart_tabs.addTab(chart3, "Score Histogram")
        analytics_layout.addWidget(self.signals_chart_tabs)
        layout.addWidget(analytics_group)
        # --- Alerts Group ---
        alerts_group = QGroupBox("Alerts & Notifications")
        alerts_layout = QHBoxLayout()
        alerts_group.setLayout(alerts_layout)
        self.alert_new_signal_checkbox = QCheckBox("Alert on New Signal")
        self.alert_new_signal_checkbox.setChecked(True)
        alerts_layout.addWidget(self.alert_new_signal_checkbox)
        self.alert_high_score_checkbox = QCheckBox("Alert on High Score (>8)")
        self.alert_high_score_checkbox.setChecked(False)
        alerts_layout.addWidget(self.alert_high_score_checkbox)
        self.alert_large_pl_checkbox = QCheckBox("Alert on Large P/L (>10%)")
        self.alert_large_pl_checkbox.setChecked(False)
        alerts_layout.addWidget(self.alert_large_pl_checkbox)
        layout.addWidget(alerts_group)
        # --- Interactive Plotly Dashboard Tab ---
        self.plotly_tab = QWidget()
        # --- Persistent Filter Chips ---
        self.filter_chips_frame = QFrame()
        self.filter_chips_layout = QHBoxLayout()
        self.filter_chips_frame.setLayout(self.filter_chips_layout)
        self.filter_chips_frame.setMaximumHeight(40)
        self.filter_chips_frame.setFrameShape(QFrame.NoFrame)
        # Add to dashboard layout
        plotly_layout = QVBoxLayout()
        plotly_layout.addWidget(self.filter_chips_frame)
        # --- Dashboard Controls ---
        dashboard_controls = QHBoxLayout()
        dashboard_controls.addWidget(QLabel("Layout:"))
        self.dashboard_layout_combo = QComboBox()
        self.dashboard_layout_combo.addItems(["2x2 Grid", "Single Row", "Single Column"])
        dashboard_controls.addWidget(self.dashboard_layout_combo)
        dashboard_controls.addWidget(QLabel("Group By:"))
        self.plotly_groupby_combo = QComboBox()
        dashboard_controls.addWidget(self.plotly_groupby_combo)
        dashboard_controls.addWidget(QLabel("Group By 2:"))
        self.plotly_groupby2_combo = QComboBox()
        dashboard_controls.addWidget(self.plotly_groupby2_combo)
        dashboard_controls.addWidget(QLabel("Aggregations:"))
        self.plotly_agg_multi_combo = QComboBox()
        self.plotly_agg_multi_combo.addItems(["mean", "sum", "count", "min", "max", "std"])
        self.plotly_agg_multi_combo.setEditable(True)
        dashboard_controls.addWidget(self.plotly_agg_multi_combo)
        self.plotly_update_btn = QPushButton("Update Dashboard")
        dashboard_controls.addWidget(self.plotly_update_btn)
        # --- Export Button ---
        self.dashboard_export_btn = QPushButton("Export Filtered/Aggregated Data")
        dashboard_controls.addWidget(self.dashboard_export_btn)
        # Add controls to layout
        plotly_layout.addLayout(dashboard_controls)
        # --- Dashboard Grid ---
        self.dashboard_grid = QGridLayout()
        plotly_layout.addLayout(self.dashboard_grid)
        # --- Chart WebViews and Chart Type Controls ---
        self.dashboard_charts = []
        self.dashboard_chart_type_combos = []
        chart_names = ["Distribution", "Rolling P/L", "Score Histogram", "Heatmap"]
        for i in range(4):
            chart_type_combo = QComboBox()
            chart_type_combo.addItems(["Bar", "Line", "Scatter", "Heatmap", "Box", "Violin", "Pie", "Area"])
            chart_type_combo.setCurrentText(chart_names[i] if chart_names[i] in chart_type_combo.itemText(i) else "Bar")
            chart_type_combo.setMaximumWidth(100)
            self.dashboard_chart_type_combos.append(chart_type_combo)
            chart_type_combo.currentTextChanged.connect(self.update_dashboard_charts)
            chart_type_layout = QHBoxLayout()
            chart_type_layout.addWidget(QLabel(f"Chart {i+1}:"))
            chart_type_layout.addWidget(chart_type_combo)
            chart_type_widget = QWidget()
            chart_type_widget.setLayout(chart_type_layout)
            plotly_layout.addWidget(chart_type_widget)
            webview = QWebEngineView()
            webview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.dashboard_charts.append(webview)
        # --- Mini-table for drill-down ---
        self.plotly_drill_table = QTableWidget()
        self.plotly_drill_table.setMaximumHeight(180)
        plotly_layout.addWidget(self.plotly_drill_table)
        self.plotly_tab.setLayout(plotly_layout)
        self.signals_chart_tabs.addTab(self.plotly_tab, "Interactive Dashboard")
        # --- QWebChannel bridges for each chart ---
        self.dashboard_bridges = []
        self.dashboard_channels = []
        for i, webview in enumerate(self.dashboard_charts):
            class DrillBridge(QObject):
                drillSignal = pyqtSignal(str, str, str, int)
                @pyqtSlot(str, str, str, int)
                def drill(self, x, y, chart, chart_idx):
                    self.drillSignal.emit(x, y, chart, chart_idx)
            bridge = DrillBridge()
            bridge.drillSignal.connect(self.handle_dashboard_drill)
            channel = QWebChannel()
            channel.registerObject('drillBridge', bridge)
            webview.page().setWebChannel(channel)
            self.dashboard_bridges.append(bridge)
            self.dashboard_channels.append(channel)
        # --- Shared filter state ---
        self.dashboard_filter = {}
        # --- Connect controls ---
        self.plotly_update_btn.clicked.connect(self.update_dashboard_layout)
        self.dashboard_layout_combo.currentTextChanged.connect(self.update_dashboard_layout)
        self.plotly_groupby_combo.currentTextChanged.connect(self.update_dashboard_layout)
        self.plotly_groupby2_combo.currentTextChanged.connect(self.update_dashboard_layout)
        self.plotly_agg_multi_combo.currentTextChanged.connect(self.update_dashboard_layout)
        self.dashboard_export_btn.clicked.connect(self.export_dashboard_data)
        # --- Initial dashboard ---
        self.update_dashboard_layout()
        # --- Load data initially ---
        self.load_signals_data()

    def update_filter_chips(self):
        # Remove all chips
        for i in reversed(range(self.filter_chips_layout.count())):
            widget = self.filter_chips_layout.itemAt(i).widget()
            if widget:
                self.filter_chips_layout.removeWidget(widget)
                widget.setParent(None)
        # Add chips for each filter
        for k, v in self.dashboard_filter.items():
            chip = QPushButton(f"{k}: {v} ✕")
            chip.setStyleSheet("background:#e0e0e0; border-radius:10px; padding:2px 8px; margin:2px;")
            chip.setMaximumHeight(28)
            chip.clicked.connect(lambda _, key=k: self.remove_filter_chip(key))
            self.filter_chips_layout.addWidget(chip)
        self.filter_chips_layout.addStretch(1)

    def remove_filter_chip(self, key):
        if key in self.dashboard_filter:
            del self.dashboard_filter[key]
            self.update_dashboard_charts()

    def export_dashboard_data(self):
        import pandas as pd
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        df = getattr(self, 'signals_df', pd.DataFrame())
        # Apply shared filter
        dff = df.copy()
        for k, v in self.dashboard_filter.items():
            if k in dff:
                dff = dff[dff[k].astype(str) == v]
        # Grouping and aggregation
        groupby = self.plotly_groupby_combo.currentText()
        groupby2 = self.plotly_groupby2_combo.currentText()
        aggfuncs = [a.strip() for a in self.plotly_agg_multi_combo.currentText().split(',') if a.strip()]
        if groupby != "None" and groupby2 != "None" and groupby != groupby2 and aggfuncs:
            grouped = dff.groupby([groupby, groupby2])
            agg_df = grouped.agg(aggfuncs).reset_index()
        elif groupby != "None" and aggfuncs:
            grouped = dff.groupby([groupby])
            agg_df = grouped.agg(aggfuncs).reset_index()
        else:
            agg_df = None
        # Ask user for file path
        path, _ = QFileDialog.getSaveFileName(self.plotly_tab, "Export Data", "dashboard_export.csv", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if path:
            try:
                if path.endswith('.csv'):
                    dff.to_csv(path, index=False)
                    if agg_df is not None:
                        agg_df.to_csv(path.replace('.csv', '_agg.csv'), index=False)
                elif path.endswith('.xlsx'):
                    with pd.ExcelWriter(path) as writer:
                        dff.to_excel(writer, sheet_name='Filtered', index=False)
                        if agg_df is not None:
                            agg_df.to_excel(writer, sheet_name='Aggregated', index=False)
                QMessageBox.information(self.plotly_tab, "Export", f"Exported to {path}")
            except Exception as e:
                QMessageBox.critical(self.plotly_tab, "Export", f"Export failed: {e}")

    def update_dashboard_layout(self):
        # Remove all widgets from grid
        for i in reversed(range(self.dashboard_grid.count())):
            widget = self.dashboard_grid.itemAt(i).widget()
            if widget:
                self.dashboard_grid.removeWidget(widget)
                widget.setParent(None)
        # Layout selection
        layout_type = self.dashboard_layout_combo.currentText()
        if layout_type == "2x2 Grid":
            for idx, webview in enumerate(self.dashboard_charts):
                self.dashboard_grid.addWidget(webview, idx // 2, idx % 2)
        elif layout_type == "Single Row":
            for idx, webview in enumerate(self.dashboard_charts):
                self.dashboard_grid.addWidget(webview, 0, idx)
        elif layout_type == "Single Column":
            for idx, webview in enumerate(self.dashboard_charts):
                self.dashboard_grid.addWidget(webview, idx, 0)
        # Update chart combos
        df = getattr(self, 'signals_df', None)
        if df is not None and not df.empty:
            all_cols = list(df.columns)
            if self.plotly_groupby_combo.count() == 0:
                self.plotly_groupby_combo.addItems(["None"] + all_cols)
            if self.plotly_groupby2_combo.count() == 0:
                self.plotly_groupby2_combo.addItems(["None"] + all_cols)
        # Update all charts
        self.update_dashboard_charts()

    def update_dashboard_charts(self):
        import plotly.graph_objs as go
        import plotly.express as px
        import pandas as pd
        df = getattr(self, 'signals_df', pd.DataFrame())
        # Apply shared filter
        dff = df.copy()
        for k, v in self.dashboard_filter.items():
            if k in dff:
                dff = dff[dff[k].astype(str) == v]
        # Grouping and aggregation
        groupby = self.plotly_groupby_combo.currentText()
        groupby2 = self.plotly_groupby2_combo.currentText()
        aggfuncs = [a.strip() for a in self.plotly_agg_multi_combo.currentText().split(',') if a.strip()]
        # Validate groupby columns
        valid_cols = set(dff.columns)
        groupby_valid = groupby and groupby != "None" and groupby in valid_cols
        groupby2_valid = groupby2 and groupby2 != "None" and groupby2 in valid_cols and groupby2 != groupby
        if groupby != "None" and groupby2 != "None" and groupby != groupby2:
            if groupby_valid and groupby2_valid:
                grouped = dff.groupby([groupby, groupby2])
            else:
                grouped = None
        elif groupby != "None":
            if groupby_valid:
                grouped = dff.groupby([groupby])
            else:
                grouped = None
        else:
            grouped = None
        # --- Custom Chart Types for each slot ---
        figs = []
        for idx, chart_type_combo in enumerate(self.dashboard_chart_type_combos):
            chart_type = chart_type_combo.currentText()
            if idx == 0:  # Distribution/Bar/Box/Pie/Area
                if grouped is not None and aggfuncs:
                    # Only aggregate numeric columns
                    numeric_cols = dff.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        agg_df = grouped[numeric_cols].agg(aggfuncs).reset_index()
                    else:
                        agg_df = grouped.agg(aggfuncs).reset_index()
                    if chart_type == "Bar":
                        if groupby in agg_df and aggfuncs[0] in agg_df:
                            fig = px.bar(agg_df, x=groupby, y=aggfuncs[0], color=groupby2 if groupby2 != "None" and groupby2 in agg_df else None)
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    elif chart_type == "Box":
                        if groupby in agg_df and aggfuncs[0] in agg_df:
                            fig = px.box(agg_df, x=groupby, y=aggfuncs[0], color=groupby2 if groupby2 != "None" and groupby2 in agg_df else None)
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    elif chart_type == "Violin":
                        if groupby in agg_df and aggfuncs[0] in agg_df:
                            fig = px.violin(agg_df, x=groupby, y=aggfuncs[0], color=groupby2 if groupby2 != "None" and groupby2 in agg_df else None, box=True, points="all")
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    elif chart_type == "Pie":
                        if groupby in agg_df and aggfuncs[0] in agg_df:
                            fig = px.pie(agg_df, names=groupby, values=aggfuncs[0])
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    elif chart_type == "Area":
                        if groupby in agg_df and aggfuncs[0] in agg_df:
                            fig = px.area(agg_df, x=groupby, y=aggfuncs[0], color=groupby2 if groupby2 != "None" and groupby2 in agg_df else None)
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    else:
                        fig = go.Figure()
                        fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                else:
                    # Defensive: check for required columns
                    has_stock = 'Stock' in dff
                    has_score = 'Score' in dff
                    has_signal_type = 'Signal_Type' in dff
                    if chart_type == "Bar":
                        if has_stock and has_score and has_signal_type:
                            fig = px.bar(dff, x='Stock', y='Score', color='Signal_Type')
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    elif chart_type == "Box":
                        if has_stock and has_score and has_signal_type:
                            fig = px.box(dff, x='Stock', y='Score', color='Signal_Type')
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    elif chart_type == "Violin":
                        if has_stock and has_score and has_signal_type:
                            fig = px.violin(dff, x='Stock', y='Score', color='Signal_Type', box=True, points="all")
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    elif chart_type == "Pie":
                        if has_stock and has_score:
                            fig = px.pie(dff, names='Stock', values='Score')
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    elif chart_type == "Area":
                        if has_stock and has_score and has_signal_type:
                            fig = px.area(dff, x='Stock', y='Score', color='Signal_Type')
                        else:
                            fig = go.Figure()
                            fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
                    else:
                        fig = go.Figure()
                        fig.add_annotation(text="Required columns missing", xref="paper", yref="paper", showarrow=False, font=dict(size=18))
            elif idx == 1:  # Rolling P/L/Line/Area
                if 'Date' in dff and '% P/L' in dff:
                    dff_sorted = dff.sort_values('Date')
                    dff_sorted['Rolling_PL'] = dff_sorted['% P/L'].astype(float).rolling(window=20, min_periods=1).mean()
                    if chart_type == "Line":
                        fig = px.line(dff_sorted, x='Date', y='Rolling_PL', color='Signal_Type')
                    elif chart_type == "Area":
                        fig = px.area(dff_sorted, x='Date', y='Rolling_PL', color='Signal_Type')
                    elif chart_type == "Scatter":
                        fig = px.scatter(dff_sorted, x='Date', y='Rolling_PL', color='Signal_Type')
                    else:
                        fig = px.line(dff_sorted, x='Date', y='Rolling_PL', color='Signal_Type')
                else:
                    fig = go.Figure()
            elif idx == 2:  # Score Histogram/Box/Violin
                if 'Score' in dff:
                    if chart_type == "Histogram" or chart_type == "Bar":
                        fig = px.histogram(dff, x='Score', color='Signal_Type')
                    elif chart_type == "Box":
                        fig = px.box(dff, x='Signal_Type', y='Score', color='Signal_Type')
                    elif chart_type == "Violin":
                        fig = px.violin(dff, x='Signal_Type', y='Score', color='Signal_Type', box=True, points="all")
                    else:
                        fig = px.histogram(dff, x='Score', color='Signal_Type')
                else:
                    fig = go.Figure()
            elif idx == 3:  # Heatmap
                if 'Date' in dff and 'Signal_Type' in dff:
                    dff['Month'] = pd.to_datetime(dff['Date'], errors='coerce').dt.to_period('M').astype(str)
                    heatmap_data = pd.crosstab(dff['Month'], dff['Signal_Type'])
                    fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=[str(m) for m in heatmap_data.index]))
                else:
                    fig = go.Figure()
            else:
                fig = go.Figure()
            figs.append(fig)
        # Add JS for cross-filtering
        for idx, (fig, webview) in enumerate(zip(figs, self.dashboard_charts)):
            fig.update_layout(clickmode='event+select')
            custom_js = f'''
            <script type="text/javascript">
            new QWebChannel(qt.webChannelTransport, function(channel) {{
                var bridge = channel.objects.drillBridge;
                var plot = document.getElementsByClassName('js-plotly-plot')[0];
                plot.on('plotly_click', function(data) {{
                    if(data.points && data.points.length > 0) {{
                        var pt = data.points[0];
                        var x = pt.x;
                        var y = pt.y;
                        var chart = pt.data.name || '';
                        bridge.drill(String(x), String(y), String(chart), {idx});
                    }}
                }});
            }});
            </script>
            '''
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                html = fig.to_html(include_plotlyjs='cdn', full_html=True)
                html = html.replace('</body>', custom_js + '</body>')
                f.write(html)
                html_path = f.name
            webview.load(QUrl.fromLocalFile(html_path))
        # Show preview in mini-table (filtered data)
        self.update_drill_table(dff)
        # Update main table
        self.update_signals_table(dff)
        # Update persistent filter chips
        self.update_filter_chips()

    def handle_dashboard_drill(self, x, y, chart, chart_idx):
        # Update shared filter state based on which chart was clicked
        # For demo: filter by x (could be extended for y, chart type, etc.)
        if chart_idx == 0:
            # Distribution: filter by groupby
            groupby = self.plotly_groupby_combo.currentText()
            if groupby != "None":
                self.dashboard_filter[groupby] = x
        elif chart_idx == 1:
            # Rolling P/L: filter by date
            self.dashboard_filter['Date'] = x
        elif chart_idx == 2:
            # Score Histogram: filter by score
            self.dashboard_filter['Score'] = x
        elif chart_idx == 3:
            # Heatmap: filter by month
            self.dashboard_filter['Month'] = x
        self.update_dashboard_charts()

    def load_signals_data(self):
        import sqlite3
        import numpy as np
        db_path = self.config.get('signals_db_path', 'data/databases/production/PSX_investing_Stocks_KMI100.db')
        symbol = self.signals_symbol_combo.currentText()
        start_date = self.signals_start_date.text().strip()
        end_date = self.signals_end_date.text().strip()
        filter_text = self.signals_text_filter.text().strip().lower()
        base_cols = [
            'Stock', 'Date', 'Close', 'Volume', 'RSI_Weekly_Avg', 'AO_Weekly', 'MA_30', 'Multibagger', 'FreeFloatRatio',
            '"% P/L"', 'Signal_Date', 'Signal_Close', 'Holding_Days', 'Status', 'Update_Date'
        ]
        buy_cols = ', '.join(base_cols + ['Success', 'NULL as Trend_Direction', "'Buy' as Signal_Type"])
        sell_cols = ', '.join(base_cols + ['Success', 'NULL as Trend_Direction', "'Sell' as Signal_Type"])
        neutral_cols = ', '.join([
            'Stock', 'Date', 'Close', 'Volume', 'RSI_Weekly_Avg', 'AO_Weekly', 'MA_30', 'Multibagger', 'FreeFloatRatio',
            'NULL as "% P/L"', 'NULL as Signal_Date', 'NULL as Signal_Close', 'NULL as Holding_Days', 'Status', 'Update_Date', 'NULL as Success', 'Trend_Direction', "'Neutral' as Signal_Type"
        ])
        query = f"""
SELECT * FROM (
    SELECT {buy_cols} FROM buy_stocks
    UNION ALL
    SELECT {sell_cols} FROM sell_stocks
    UNION ALL
    SELECT {neutral_cols} FROM neutral_stocks
) WHERE 1=1
"""
        params = []
        if symbol and symbol != "All Symbols":
            query += " AND Stock = ?"
            params.append(symbol)
        if start_date:
            query += " AND Date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND Date <= ?"
            params.append(end_date)
        query += " ORDER BY Date DESC LIMIT 500"
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
        except Exception as e:
            df = pd.DataFrame()
            self.signals_summary_label.setText(f"<span style='color:red'>Error loading signals: {e}</span>")
        if not df.empty and filter_text:
            df = df[df.apply(lambda row: filter_text in str(row.values).lower(), axis=1)]
        # --- Enhanced Signal scoring ---
        def score_row(row):
            score = 0
            try:
                rsi = float(row.get('RSI_Weekly_Avg', 0) or 0)
                ao = float(row.get('AO_Weekly', 0) or 0)
                pl = float(row.get('% P/L', 0) or 0)
                vol = float(row.get('Volume', 0) or 0)
                multibagger = str(row.get('Multibagger', '') or '').lower()
                freefloat = row.get('FreeFloatRatio', '')
                try:
                    freefloat = float(freefloat)
                except Exception:
                    freefloat = 0
                # Recency bonus (signals in last 30 days get +1)
                date = pd.to_datetime(row.get('Date', None), errors='coerce')
                if pd.notnull(date):
                    days_ago = (pd.Timestamp.now() - date).days
                    if days_ago <= 30:
                        score += 1
                    elif days_ago <= 90:
                        score += 0.5
                # RSI, AO, P/L as before
                score += (rsi - 50) / 10  # -5 to +5
                score += ao / 5           # AO moderate impact
                score += pl / 20          # P/L moderate impact
                # Penalize high negative P/L
                if pl < -10:
                    score -= 1
                # Penalize very low volume
                if vol < 1000:
                    score -= 0.5
                # Bonus for Multibagger
                if 'yes' in multibagger:
                    score += 1
                # Bonus for high FreeFloatRatio
                if freefloat and freefloat > 0.5:
                    score += 0.5
            except Exception:
                pass
            return round(score, 2)
        if not df.empty:
            df['Score'] = df.apply(score_row, axis=1)
        self.signals_df = df
        # --- Alerts logic ---
        self.check_alerts(df)
        self.update_signals_table()
        self.update_signals_analytics()
    def check_alerts(self, df):
        from PyQt5.QtWidgets import QMessageBox
        # Only check the most recent row (top of table)
        if df.empty:
            return
        latest = df.iloc[0]
        alerts = []
        if hasattr(self, 'alert_new_signal_checkbox') and self.alert_new_signal_checkbox.isChecked():
            alerts.append(f"New {latest.get('Signal_Type','')} signal for {latest.get('Stock','')} on {latest.get('Date','')}")
        if hasattr(self, 'alert_high_score_checkbox') and self.alert_high_score_checkbox.isChecked():
            try:
                if float(latest.get('Score', 0)) > 8:
                    alerts.append(f"High Score Alert: {latest.get('Stock','')} ({latest.get('Score','')})")
            except Exception:
                pass
        if hasattr(self, 'alert_large_pl_checkbox') and self.alert_large_pl_checkbox.isChecked():
            try:
                if abs(float(latest.get('% P/L', 0))) > 10:
                    alerts.append(f"Large P/L Alert: {latest.get('Stock','')} ({latest.get('% P/L','')}%)")
            except Exception:
                pass
        for msg in alerts:
            QMessageBox.information(self.signals_tab, "Alert", msg)
            # --- Telegram integration ---
            if hasattr(self, 'telegram_enable_checkbox') and self.telegram_enable_checkbox.isChecked():
                token = self.telegram_token_input.text().strip()
                chat_id = self.telegram_chatid_input.text().strip()
                if token and chat_id:
                    self.send_telegram_alert(token, chat_id, msg)
    def send_telegram_alert(self, token, chat_id, message):
        import requests
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {"chat_id": chat_id, "text": message}
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")

    def update_signals_analytics(self):
        df = getattr(self, 'signals_df', pd.DataFrame())
        if df.empty:
            self.signals_summary_label.setText("No signals to display.")
            self.signals_chart_figure.clf()
            self.signals_chart_canvas.draw()
            self.signals_pl_figure.clf()
            self.signals_pl_canvas.draw()
            self.signals_score_figure.clf()
            self.signals_score_canvas.draw()
            self.signals_backtest_label.setText("")
            return
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from PyQt5.QtWidgets import QPushButton, QComboBox, QHBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem, QDialog, QVBoxLayout, QLabel
        # --- Advanced Analytics ---
        # Confusion-matrix-like summary: Signal_Type vs. Success
        confusion = None
        if 'Signal_Type' in df and 'Success' in df:
            confusion = pd.crosstab(df['Signal_Type'], df['Success'], dropna=False)
        # Rolling win-rate and rolling avg P/L
        window_size = getattr(self, 'rolling_window_size', 30)
        if not hasattr(self, 'rolling_window_combo'):
            self.rolling_window_combo = QComboBox()
            for w in [10, 20, 30, 50, 100]:
                self.rolling_window_combo.addItem(str(w))
            self.rolling_window_combo.setCurrentText(str(window_size))
            self.rolling_window_combo.currentTextChanged.connect(self._set_rolling_window_size)
            self.signals_chart_tabs.widget(1).layout().addWidget(self.rolling_window_combo)
        # Rolling calculations
        pl_col = df['% P/L'].dropna().astype(float)
        win_col = (pl_col > 0).astype(int)
        rolling_win = win_col.rolling(window=window_size, min_periods=1).mean() * 100
        rolling_pl = pl_col.rolling(window=window_size, min_periods=1).mean()
        # Most improved/deteriorated stocks
        if 'Date' in df and 'Stock' in df and 'Score' in df:
            df_sorted = df.sort_values(['Stock', 'Date'])
            grouped = df_sorted.groupby('Stock')['Score']
            first_score = grouped.first()
            last_score = grouped.last()
            score_change = (last_score - first_score).sort_values(ascending=False)
            most_improved = score_change.head(3)
            most_deteriorated = score_change.tail(3)
        else:
            most_improved = most_deteriorated = pd.Series(dtype=float)
        # Heatmap of signal counts by month and type
        if 'Date' in df and 'Signal_Type' in df:
            df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.to_period('M')
            heatmap_data = pd.crosstab(df['Month'], df['Signal_Type'])
        else:
            heatmap_data = pd.DataFrame()
        # --- Summary label ---
        buy_count = (df['Signal_Type'] == 'Buy').sum()
        sell_count = (df['Signal_Type'] == 'Sell').sum()
        neutral_count = (df['Signal_Type'] == 'Neutral').sum()
        avg_holding = df['Holding_Days'].dropna().astype(float).mean() if 'Holding_Days' in df else None
        pl_col = df['% P/L'].dropna().astype(float)
        win_rate = (pl_col > 0).sum() / len(pl_col) * 100 if len(pl_col) else 0
        total_pl = pl_col.sum() if len(pl_col) else 0
        avg_pl = pl_col.mean() if len(pl_col) else 0
        median_pl = pl_col.median() if len(pl_col) else 0
        holding_col = df['Holding_Days'].dropna().astype(float) if 'Holding_Days' in df else pd.Series(dtype=float)
        best_holding = holding_col.max() if not holding_col.empty else None
        worst_holding = holding_col.min() if not holding_col.empty else None
        most_freq_signal = df['Signal_Type'].mode()[0] if not df.empty else ''
        top_scores = df.groupby('Stock')['Score'].mean().sort_values(ascending=False).head(5)
        summary = f"<b>Buy:</b> {buy_count} | <b>Sell:</b> {sell_count} | <b>Neutral:</b> {neutral_count}"
        if avg_holding:
            summary += f" | <b>Avg Holding Days:</b> {avg_holding:.1f}"
        summary += f" | <b>Median P/L:</b> {median_pl:.2f}"
        if best_holding is not None:
            summary += f" | <b>Best Holding:</b> {best_holding:.0f}d"
        if worst_holding is not None:
            summary += f" | <b>Worst Holding:</b> {worst_holding:.0f}d"
        summary += f" | <b>Most Frequent Signal:</b> {most_freq_signal}"
        summary += f"<br><b>Top 5 Stocks by Avg Score:</b> " + ', '.join(f"{stock} ({score:.2f})" for stock, score in top_scores.items())
        if not most_improved.empty:
            summary += f"<br><b>Most Improved:</b> " + ', '.join(f"{stock} ({change:.2f})" for stock, change in most_improved.items())
        if not most_deteriorated.empty:
            summary += f"<br><b>Most Deteriorated:</b> " + ', '.join(f"{stock} ({change:.2f})" for stock, change in most_deteriorated.items())
        self.signals_summary_label.setText(summary)
        self.signals_backtest_label.setText(f"<b>Backtest Summary:</b> Total P/L: {total_pl:.2f} | Win Rate: {win_rate:.1f}% | Avg P/L: {avg_pl:.2f}")
        # --- Confusion Matrix Table (interactive) ---
        if confusion is not None:
            if not hasattr(self, 'confusion_table'):
                self.confusion_table = QTableWidget()
                self.signals_tab.layout().insertWidget(2, self.confusion_table)
            self.confusion_table.setRowCount(confusion.shape[0])
            self.confusion_table.setColumnCount(confusion.shape[1])
            self.confusion_table.setHorizontalHeaderLabels([str(c) for c in confusion.columns])
            self.confusion_table.setVerticalHeaderLabels([str(i) for i in confusion.index])
            for i, idx in enumerate(confusion.index):
                for j, col in enumerate(confusion.columns):
                    val = confusion.loc[idx, col]
                    item = QTableWidgetItem(str(val))
                    item.setToolTip(f"Filter: {idx} & {col}")
                    self.confusion_table.setItem(i, j, item)
            def on_cell_clicked(row, col):
                sig = confusion.index[row]
                succ = confusion.columns[col]
                # Filter table
                self.signals_text_filter.setText(f"{sig} {succ}")
            self.confusion_table.cellClicked.connect(on_cell_clicked)
            self.confusion_table.setMaximumHeight(120)
            self.confusion_table.setVisible(True)
        # --- Chart 1: Distribution (interactive) ---
        self.signals_chart_figure.clf()
        ax1 = self.signals_chart_figure.add_subplot(111)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df1 = df.dropna(subset=['Date'])
        counts = df1.groupby([df1['Date'].dt.to_period('M'), 'Signal_Type']).size().unstack(fill_value=0)
        bars = counts.plot(kind='bar', stacked=True, ax=ax1, picker=True)
        ax1.set_title('Signal Distribution Over Time')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Count')
        self.signals_chart_figure.tight_layout()
        self.signals_chart_canvas.draw()
        def on_pick(event):
            if hasattr(event, 'ind') and event.ind:
                bar_idx = event.ind[0]
                month = counts.index[bar_idx].strftime('%Y-%m')
                self.signals_text_filter.setText(month)
        self.signals_chart_canvas.mpl_connect('pick_event', on_pick)
        # --- Chart 2: Rolling Win Rate & Avg P/L ---
        self.signals_pl_figure.clf()
        ax2 = self.signals_pl_figure.add_subplot(111)
        if not df1.empty and '% P/L' in df1:
            df1_sorted = df1.sort_values('Date')
            x = np.arange(len(df1_sorted))
            y_pl = df1_sorted['% P/L'].astype(float).rolling(window=window_size, min_periods=1).mean()
            y_win = (df1_sorted['% P/L'].astype(float) > 0).rolling(window=window_size, min_periods=1).mean() * 100
            l1 = ax2.plot(df1_sorted['Date'], y_pl, label=f'Rolling Avg P/L ({window_size})', color='blue')[0]
            l2 = ax2.plot(df1_sorted['Date'], y_win, label=f'Rolling Win Rate (%) ({window_size})', color='green')[0]
            ax2.set_title('Rolling Win Rate & Avg P/L')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value')
            ax2.legend()
            # Tooltip on hover
            annot = ax2.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)
            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax2:
                    for line, label in zip([l1, l2], ['Avg P/L', 'Win Rate']):
                        cont, ind = line.contains(event)
                        if cont:
                            xdata, ydata = line.get_data()
                            idx = ind["ind"][0]
                            annot.xy = (xdata[idx], ydata[idx])
                            annot.set_text(f"{label}\n{xdata[idx].strftime('%Y-%m-%d')}: {ydata[idx]:.2f}")
                            annot.set_visible(True)
                            self.signals_pl_canvas.draw_idle()
                            break
                    else:
                        if vis:
                            annot.set_visible(False)
                            self.signals_pl_canvas.draw_idle()
            self.signals_pl_canvas.mpl_connect("motion_notify_event", hover)
        self.signals_pl_figure.tight_layout()
        self.signals_pl_canvas.draw()
        # --- Chart 3: Score histogram (with save button) ---
        self.signals_score_figure.clf()
        ax3 = self.signals_score_figure.add_subplot(111)
        if 'Score' in df:
            n, bins, patches = ax3.hist(df['Score'].astype(float), bins=20, color='skyblue', edgecolor='black', picker=True)
            ax3.set_title('Signal Score Histogram')
            ax3.set_xlabel('Score')
            ax3.set_ylabel('Frequency')
            annot = ax3.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)
            def hover_hist(event):
                vis = annot.get_visible()
                if event.inaxes == ax3:
                    for patch in patches:
                        if patch.contains(event)[0]:
                            x = patch.get_x() + patch.get_width()/2
                            y = patch.get_height()
                            annot.xy = (x, y)
                            annot.set_text(f"Score: {x:.2f}\nCount: {int(y)}")
                            annot.set_visible(True)
                            self.signals_score_canvas.draw_idle()
                            break
                    else:
                        if vis:
                            annot.set_visible(False)
                            self.signals_score_canvas.draw_idle()
            self.signals_score_canvas.mpl_connect("motion_notify_event", hover_hist)
        self.signals_score_figure.tight_layout()
        self.signals_score_canvas.draw()
        # --- Chart 4: Heatmap of signal counts ---
        if not hasattr(self, 'signals_heatmap_figure'):
            self.signals_heatmap_figure = plt.Figure(figsize=(4,2))
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            self.signals_heatmap_canvas = FigureCanvas(self.signals_heatmap_figure)
            # Only add the tab if it hasn't already been added
            if all(self.signals_chart_tabs.widget(i) is not self.signals_heatmap_canvas for i in range(self.signals_chart_tabs.count())):
                self.signals_chart_tabs.addTab(self.signals_heatmap_canvas, "Signal Heatmap")
        self.signals_heatmap_figure.clf()
        if not heatmap_data.empty:
            ax4 = self.signals_heatmap_figure.add_subplot(111)
            im = ax4.imshow(heatmap_data.values, aspect='auto', cmap='YlGnBu')
            ax4.set_xticks(np.arange(heatmap_data.shape[1]))
            ax4.set_yticks(np.arange(heatmap_data.shape[0]))
            ax4.set_xticklabels(heatmap_data.columns)
            ax4.set_yticklabels([str(m) for m in heatmap_data.index])
            for i in range(heatmap_data.shape[0]):
                for j in range(heatmap_data.shape[1]):
                    ax4.text(j, i, str(heatmap_data.iloc[i, j]), ha='center', va='center', color='black')
            ax4.set_title('Signal Count Heatmap')
            self.signals_heatmap_figure.tight_layout()
            self.signals_heatmap_canvas.draw()
        # --- Add Reset Filters Button ---
        if not hasattr(self, 'signals_reset_filters_btn'):
            self.signals_reset_filters_btn = QPushButton("Reset All Filters")
            self.signals_reset_filters_btn.clicked.connect(lambda: self._reset_signals_filters())
            self.signals_tab.layout().insertWidget(0, self.signals_reset_filters_btn)
        # --- Add Export Analytics Button ---
        if not hasattr(self, 'signals_export_analytics_btn'):
            self.signals_export_analytics_btn = QPushButton("Export Analytics Report")
            self.signals_export_analytics_btn.clicked.connect(lambda: self._export_signals_analytics_report(df, confusion, heatmap_data))
            self.signals_tab.layout().insertWidget(1, self.signals_export_analytics_btn)
        # --- Save chart buttons (as before) ---
        if not hasattr(self, 'signals_save_chart_btns'):
            self.signals_save_chart_btns = []
        for btn in self.signals_save_chart_btns:
            btn.setParent(None)
        self.signals_save_chart_btns = []
        chart_widgets = [self.signals_chart_canvas, self.signals_pl_canvas, self.signals_score_canvas]
        chart_names = ["Signal Distribution", "Rolling Win Rate & Avg P/L", "Score Histogram"]
        for i, (canvas, name) in enumerate(zip(chart_widgets, chart_names)):
            btn = QPushButton(f"Save {name} as Image")
            def save_chart(c=canvas, n=name):
                path, _ = QFileDialog.getSaveFileName(self, f"Save {n}", f"{n.replace(' ', '_').lower()}.png", "PNG Files (*.png)")
                if path:
                    c.figure.savefig(path)
            btn.clicked.connect(save_chart)
            self.signals_chart_tabs.widget(i).layout().addWidget(btn)
            self.signals_save_chart_btns.append(btn)

    def _set_rolling_window_size(self, val):
        try:
            self.rolling_window_size = int(val)
        except Exception:
            self.rolling_window_size = 30
        self.update_signals_analytics()

    def _reset_signals_filters(self):
        self.signals_symbol_combo.setCurrentIndex(0)
        self.signals_start_date.setText("")
        self.signals_end_date.setText("")
        self.signals_text_filter.setText("")
        self.load_signals_data()

    def _export_signals_analytics_report(self, df, confusion, heatmap_data):
        path, _ = QFileDialog.getSaveFileName(self, "Export Analytics Report", "signals_analytics_report.xlsx", "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if path:
            try:
                with pd.ExcelWriter(path) as writer:
                    df.to_excel(writer, sheet_name='Signals', index=False)
                    if confusion is not None:
                        confusion.to_excel(writer, sheet_name='ConfusionMatrix')
                    if not heatmap_data.empty:
                        heatmap_data.to_excel(writer, sheet_name='SignalHeatmap')
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Export", f"Analytics report exported to {path}")
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Export", f"Export failed: {e}")

    def filter_signals_table(self):
        self.load_signals_data()

    def export_signals_data(self):
        df = getattr(self, 'signals_df', pd.DataFrame())
        if df.empty:
            QMessageBox.warning(self, "Export", "No signals to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Signals", "signals_export.csv", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if path:
            try:
                if path.endswith('.csv'):
                    df.to_csv(path, index=False)
                elif path.endswith('.xlsx'):
                    df.to_excel(path, index=False)
                QMessageBox.information(self, "Export", f"Signals exported to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export", f"Export failed: {e}")

    def show_signal_details(self, row, col):
        df = getattr(self, 'signals_df', pd.DataFrame())
        if df.empty or row >= len(df):
            return
        signal = df.iloc[row]
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Signal Details: {signal['Stock']} ({signal['Signal_Type']})")
        vbox = QVBoxLayout()
        for col in df.columns:
            vbox.addWidget(QLabel(f"<b>{col}:</b> {signal[col]}"))
        # Mini chart
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            fig = plt.Figure(figsize=(4,2))
            ax = fig.add_subplot(111)
            if 'Date' in df.columns and 'Close' in df.columns:
                plot_df = df[df['Stock'] == signal['Stock']].sort_values('Date')
                ax.plot(pd.to_datetime(plot_df['Date']), plot_df['Close'], label='Close')
                if 'RSI_Weekly_Avg' in plot_df:
                    ax.plot(pd.to_datetime(plot_df['Date']), plot_df['RSI_Weekly_Avg'], label='RSI_Weekly_Avg')
                if 'Score' in plot_df:
                    ax2 = ax.twinx()
                    ax2.plot(pd.to_datetime(plot_df['Date']), plot_df['Score'], color='orange', label='Score', linestyle='dashed')
                    ax2.set_ylabel('Score')
                ax.legend()
                ax.set_title('Close, RSI Weekly Avg & Score')
            canvas = FigureCanvas(fig)
            vbox.addWidget(canvas)
        except Exception:
            pass
        dialog.setLayout(vbox)
        dialog.resize(500, 400)
        dialog.exec_()

    def update_plotly_chart(self):
        import plotly.graph_objs as go
        import plotly.express as px
        import pandas as pd
        df = getattr(self, 'signals_df', pd.DataFrame())
        if df.empty:
            self.plotly_webview.setHtml("<h3>No data to display</h3>")
            return
        # Populate combos if empty
        numeric_cols = df.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
        all_cols = list(df.columns)
        if self.plotly_x_combo.count() == 0:
            self.plotly_x_combo.addItems(all_cols)
        if self.plotly_y_combo.count() == 0:
            self.plotly_y_combo.addItems(numeric_cols)
        x = self.plotly_x_combo.currentText() or all_cols[0]
        y = self.plotly_y_combo.currentText() or (numeric_cols[0] if numeric_cols else all_cols[0])
        chart_type = self.plotly_type_combo.currentText()
        fig = None
        if chart_type == "Bar":
            fig = px.bar(df, x=x, y=y, color='Signal_Type', hover_data=df.columns)
        elif chart_type == "Line":
            fig = px.line(df, x=x, y=y, color='Signal_Type', hover_data=df.columns)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x, y=y, color='Signal_Type', hover_data=df.columns)
        elif chart_type == "Heatmap":
            if pd.api.types.is_numeric_dtype(df[y]):
                z = y
            else:
                z = None
            fig = px.density_heatmap(df, x=x, y=y, z=z, histfunc='avg' if z else None, color_continuous_scale='Viridis')
        # Add click event for drill-down
        fig.update_layout(clickmode='event+select')
        # Save to temp HTML and load in QWebEngineView
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
            fig.write_html(f.name, include_plotlyjs='cdn', full_html=True)
            html_path = f.name
        self.plotly_webview.load(QUrl.fromLocalFile(html_path))
        # Drill-down: listen for click events via JavaScript bridge
        # (PyQt5 QWebChannel/JS bridge is complex; for now, provide instructions for user to filter manually)
        self.plotly_drill_table.setRowCount(0)
        self.plotly_drill_table.setColumnCount(len(df.columns))
        self.plotly_drill_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        # Optionally, show top 10 rows as preview
        preview = df.head(10)
        self.plotly_drill_table.setRowCount(len(preview))
        for row_idx, row in preview.iterrows():
            for col_idx, col in enumerate(df.columns):
                self.plotly_drill_table.setItem(row_idx, col_idx, QTableWidgetItem(str(row[col])))
        # Clean up temp file on next update
        import atexit
        def cleanup():
            try:
                os.remove(html_path)
            except Exception:
                pass
        atexit.register(cleanup)

    def update_drill_table(self, df):
        """Update the mini drill-down table in the dashboard with the given DataFrame."""
        if not hasattr(self, 'plotly_drill_table'):
            return
        table = self.plotly_drill_table
        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        table.setRowCount(min(len(df), 100))  # Limit to 100 rows for preview
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for row_idx, row in enumerate(df.head(100).itertuples(index=False)):
            for col_idx, value in enumerate(row):
                table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    def update_signals_table(self, df=None):
        """Update the signals_table widget with the given DataFrame."""
        if df is None:
            df = getattr(self, 'signals_df', None)
        if df is None or df.empty:
            self.signals_table.setRowCount(0)
            return
        self.signals_table.setRowCount(len(df))
        self.signals_table.setColumnCount(len(df.columns))
        self.signals_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for row_idx, row in df.iterrows():
            for col_idx, col in enumerate(df.columns):
                self.signals_table.setItem(row_idx, col_idx, QTableWidgetItem(str(row[col])))

    def load_signal_history(self):
        import sqlite3
        import pandas as pd
        db_path = self.config.get('signals_db_path', 'data/databases/production/PSX_investing_Stocks_KMI100.db')
        try:
            conn = sqlite3.connect(db_path)
            # Example: reconstruct transitions for each stock
            query = '''
            SELECT Stock, Date, "Signal_Type", "Close", "% P/L", "Holding_Days" FROM (
                SELECT Stock, Date, 'Buy' as Signal_Type, Close, "% P/L", Holding_Days FROM buy_stocks
                UNION ALL
                SELECT Stock, Date, 'Sell' as Signal_Type, Close, "% P/L", Holding_Days FROM sell_stocks
                UNION ALL
                SELECT Stock, Date, 'Neutral' as Signal_Type, Close, "% P/L", Holding_Days FROM neutral_stocks
            ) ORDER BY Stock, Date
            '''
            df = pd.read_sql_query(query, conn)
            conn.close()
            # Build transitions
            transitions = []
            for stock, group in df.groupby('Stock'):
                group = group.sort_values('Date')
                prev_signal = None
                prev_date = None
                prev_close = None
                for idx, row in group.iterrows():
                    if prev_signal is not None and row['Signal_Type'] != prev_signal:
                        holding_days = (pd.to_datetime(row['Date']) - pd.to_datetime(prev_date)).days
                        pl = row['Close'] - prev_close if prev_close is not None else None
                        transitions.append([
                            stock, prev_signal, row['Signal_Type'], prev_date, row['Date'], holding_days, pl, ""
                        ])
                    prev_signal = row['Signal_Type']
                    prev_date = row['Date']
                    prev_close = row['Close']
            self.signal_history_table.setRowCount(len(transitions))
            for i, t in enumerate(transitions):
                for j, val in enumerate(t):
                    self.signal_history_table.setItem(i, j, QTableWidgetItem(str(val)))
        except Exception as e:
            self.signal_history_table.setRowCount(0)
            self.signal_history_table.setColumnCount(1)
            self.signal_history_table.setHorizontalHeaderLabels(["Error"])
            self.signal_history_table.setItem(0, 0, QTableWidgetItem(str(e)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())