import sys
import os
import asyncio
import threading
import pandas as pd
import aiosqlite
import matplotlib
import numpy as np
import time
import logging
from pathlib import Path

# Configure logging first
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "psx_dashboard.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug("Starting ui_manager.py imports")
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTableView, QComboBox, 
    QPushButton, QLabel, QDateEdit, QLineEdit, QStatusBar, QMenuBar, QMenu, 
    QFileDialog, QTabWidget, QGroupBox, QGraphicsDropShadowEffect, QGridLayout
)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal, QAbstractTableModel, QModelIndex, QTimer
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem, QAction, QColor, QIntValidator
from typing import Optional, List, Dict, Any
from src.data_processing.stock_analysis import db_handler
from collections import defaultdict
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# Optional TensorFlow imports (for AI prediction features)
try:
    logging.debug("Attempting TensorFlow imports")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
    logging.debug("TensorFlow imports successful")
except ImportError:
    logging.debug("TensorFlow not available")
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    LSTM = None
    Dense = None

# Supported technical indicators
VALID_INDICATORS = ["bollinger", "rsi", "macd"]

import numpy as np
from typing import Tuple, Optional

class PlotlyWidget(QWidget):
    """Enhanced interactive Plotly chart widget with technical indicators."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = logger.getChild(self.__class__.__name__)
        self.webview = QWebEngineView()
        self.indicators = []  # Track active indicators
        self.df = None  # Store current data
        layout = QVBoxLayout(self)
        layout.addWidget(self.webview)
        
    def plot(self, df: pd.DataFrame, x: str, y: str, title: str = ""):
        """Plot data with interactive controls."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        if x not in df.columns or y not in df.columns:
            raise ValueError(f"Columns {x} or {y} not found in DataFrame")
        if not all(isinstance(df[col].iloc[0], (int, float)) for col in [x, y]):
            raise ValueError("Plot columns must contain numeric data")
            
        self.df = df.copy()
        fig = go.Figure()
        # Main price line
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[y],
            mode='lines+markers',
            name='Price',
            line=dict(color='#1f77b4')
        ))
        
        # Layout configuration
        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title=y,
            hovermode='x unified',
            dragmode='pan',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
        )
        
        # Apply any active indicators
        for indicator in self.indicators:
            self._apply_indicator(fig, indicator)
            
        # Set the HTML content for the webview
        self.webview.setHtml(fig.to_html(include_plotlyjs='cdn'))
            
        
    def add_indicator(self, indicator_type: str, **params):
        """Add technical indicator to the chart."""
        if not isinstance(indicator_type, str):
            raise TypeError("Indicator type must be a string")
        if indicator_type not in VALID_INDICATORS:  # Add constant at top
            raise ValueError(f"Invalid indicator type: {indicator_type}")
        if indicator_type not in self.indicators:
            self.indicators.append((indicator_type, params))
            if self.df is not None:
                self.plot(self.df, self.df.columns[0], self.df.columns[1])
                
    def remove_indicator(self, indicator_type: str):
        """Remove technical indicator from chart."""
        self.indicators = [i for i in self.indicators if i[0] != indicator_type]
        if self.df is not None:
            self.plot(self.df, self.df.columns[0], self.df.columns[1])
            
    def _apply_indicator(self, fig, indicator):
        """Apply specific indicator to figure."""
        indicator_type, params = indicator
        if indicator_type == "bollinger":
            window = params.get('window', 20)
            std_dev = params.get('std_dev', 2)
            self._add_bollinger_bands(fig, window, std_dev)
        elif indicator_type == "rsi":
            window = params.get('window', 14)
            self._add_rsi(fig, window)
        elif indicator_type == "macd":
            self._add_macd(fig)
            
    def _add_bollinger_bands(self, fig, window=20, std_dev=2):
        """Add Bollinger Bands to figure."""
        if 'Close' not in self.df.columns:
            return
            
        rolling_mean = self.df['Close'].rolling(window=window).mean()
        rolling_std = self.df['Close'].rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=upper_band,
            name=f'Upper BB ({window},{std_dev})',
            line=dict(color='rgba(255, 0, 0, 0.5)')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=lower_band,
            name=f'Lower BB ({window},{std_dev})',
            line=dict(color='rgba(0, 255, 0, 0.5)'),
            fill='tonexty',
            fillcolor='rgba(0, 100, 80, 0.1)'
        ))

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    def plot(self, df: pd.DataFrame, x: str, y: str, title: str = ""):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(df[x], df[y], marker='o')
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        self.canvas.draw()
class AIPredictor:
    """AI-based stock prediction class with LSTM and ARIMA models."""
    
    def __init__(self):
        self.logger = logger.getChild(self.__class__.__name__)
        self.lstm_model = None
        self.arima_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        
    def train_lstm(self, data: pd.Series) -> bool:
        """Train LSTM model on historical data."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Install tensorflow to use LSTM predictions.")
            return False
            
        try:
            # Prepare data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, 60)
            
            if len(X) < 10:  # Need minimum data
                return False
                
            # Build model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            
            self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            self.lstm_model.fit(X, y, batch_size=1, epochs=1, verbose=0)
            
            return True
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}", exc_info=True)
            return False
        
    
    def train_arima(self, data: pd.Series) -> bool:
        """Train ARIMA model on historical data."""
        try:
            # Use automatic ARIMA order selection
            self.arima_model = ARIMA(data, order=(5, 1, 0))
            self.arima_model = self.arima_model.fit()
            return True
        except Exception as e:
            self.logger.error(f"Error training ARIMA model: {str(e)}", exc_info=True)
            return False
        
    
    def predict(self, model_type: str, data: pd.Series, days: int = 30) -> Optional[pd.Series]:
        """Generate predictions using specified model."""
        try:
            if model_type.lower() == "lstm" and self.lstm_model is not None:
                return self._predict_lstm(data, days)
            elif model_type.lower() == "arima" and self.arima_model is not None:
                return self._predict_arima(days)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}", exc_info=True)
            return None
            
    
    def _create_sequences(self, data, seq_length):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def _predict_lstm(self, data: pd.Series, days: int) -> pd.Series:
        """Generate LSTM predictions."""
        if not TENSORFLOW_AVAILABLE or self.lstm_model is None:
            return None
            
        # Prepare last 60 days of data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        last_60_days = scaled_data[-60:]
        
        predictions = []
        current_batch = last_60_days.reshape((1, 60, 1))
        
        for _ in range(days):
            pred = self.lstm_model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            
            # Update batch for next prediction
            current_batch = np.append(current_batch[:, 1:, :], 
                                    [[pred]], axis=1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create date range for predictions
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )
        
        return pd.Series(predictions.flatten(), index=future_dates)
    
    def _predict_arima(self, days: int) -> pd.Series:
        """Generate ARIMA predictions."""
        if self.arima_model is None:
            return None
            
        forecast = self.arima_model.forecast(steps=days)
        
        # Create date range for predictions
        last_date = pd.Timestamp.now()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )
        
        return pd.Series(forecast, index=future_dates)

class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(list, list)  # rows, headers
    error = pyqtSignal(str)

    def __init__(self, db_path: str = "data/databases/production/PSX_investing_Stocks_KMI100.db"):
        if not isinstance(db_path, str):
            raise TypeError("db_path must be a string")
        if not db_path.endswith('.db'):
            raise ValueError("Database path must end with .db extension")
        super().__init__()
        self.logger = logger.getChild(self.__class__.__name__)
        self.db_path = db_path
        self.signal_tables = [
            ("buy_stocks", "BUY"),
            ("sell_stocks", "SELL"),
            ("neutral_stocks", "NEUTRAL")
        ]

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            data, headers = loop.run_until_complete(self.fetch_all_signals_with_transitions())
            self.data_loaded.emit(data, headers)
        except Exception as e:
            self.error.emit(str(e))

    async def fetch_all_signals_with_transitions(self):
        all_rows = []
        all_headers = set()
        stock_entries = defaultdict(list)
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for table, signal_type in self.signal_tables:
                    try:
                        async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                            columns = [row[1] for row in await pragma_cursor.fetchall()]
                        if not columns:
                            continue
                        select_cols = ', '.join([f'"{col}"' for col in columns])
                        async with db.execute(f"SELECT {select_cols} FROM {table}") as cursor:
                            rows = await cursor.fetchall()
                            if not rows:
                                self.logger.warning(f"No data found in table {table}")
                                continue
                            for row in rows:
                                row_dict = dict(zip(columns, row))
                                row_dict['signal_type'] = signal_type
                                # Use 'Stock' or 'symbol' as the key (try both)
                                stock_key = row_dict.get('Stock') or row_dict.get('symbol')
                                date_val = row_dict.get('Date') or row_dict.get('date')
                                if stock_key and date_val:
                                    stock_entries[stock_key].append((date_val, row_dict))
                                all_headers.update(columns)
                    except aiosqlite.Error as e:
                        self.logger.error(f"Database error in table {table}: {str(e)}", exc_info=True)
                        continue  # skip problematic tables
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}", exc_info=True)
            raise
        # Sort entries for each stock by date
        for stock, entries in stock_entries.items():
            entries.sort(key=lambda x: x[0])
            prev_signal = None
            prev_date = None
            for idx, (date_val, row_dict) in enumerate(entries):
                row_dict['previous_signal_type'] = prev_signal if prev_signal else ''
                row_dict['transition_date'] = date_val if prev_signal and prev_signal != row_dict['signal_type'] else ''
                prev_signal = row_dict['signal_type']
                prev_date = date_val
                all_rows.append(row_dict)
        all_headers = list(all_headers)
        for extra_col in ['signal_type', 'previous_signal_type', 'transition_date']:
            if extra_col not in all_headers:
                all_headers.append(extra_col)
        # Ensure all rows have all headers
        final_rows = []
        for row in all_rows:
            final_rows.append([row.get(h, '') for h in all_headers])
        return final_rows, all_headers

class MainWindow(QMainWindow):
    """Main application window for the Stock Analysis GUI."""
    signal_data_ready = pyqtSignal(list, list)
    prediction_ready = pyqtSignal(str, pd.Series)  # model_type, predictions
    
    THEMES = {
        "Light": """
            QWidget { background: #f7f7f7; color: #222; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #f7f7f7; }
            QTableView { background: #fff; border-radius: 8px; padding: 4px; }
            QPushButton { background: #e0e0e0; border-radius: 6px; padding: 6px 16px; border: 1px solid #ccc; }
            QPushButton:hover { background: #d0d0ff; }
            QComboBox, QLineEdit, QDateEdit { background: #fff; border-radius: 6px; border: 1px solid #bbb; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #bbb; border-radius: 8px; }
            QTabBar::tab { background: #e0e0e0; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #d0d0ff; }
        """,
        "Dark": """
            QWidget { background: #232629; color: #f0f0f0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #232629; }
            QTableView { background: #1e1e1e; border-radius: 8px; padding: 4px; color: #f0f0f0; }
            QPushButton { background: #444; color: #fff; border-radius: 6px; padding: 6px 16px; border: 1px solid #666; }
            QPushButton:hover { background: #5a5a8a; }
            QComboBox, QLineEdit, QDateEdit { background: #333; color: #fff; border-radius: 6px; border: 1px solid #666; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #444; border-radius: 8px; }
            QTabBar::tab { background: #444; color: #fff; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #5a5a8a; }
        """,
        "Blue": """
            QWidget { background: #eaf3fb; color: #1a237e; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #eaf3fb; }
            QTableView { background: #fff; border-radius: 8px; padding: 4px; }
            QPushButton { background: #1976d2; color: #fff; border-radius: 6px; padding: 6px 16px; border: 1px solid #1565c0; }
            QPushButton:hover { background: #1565c0; }
            QComboBox, QLineEdit, QDateEdit { background: #fff; border-radius: 6px; border: 1px solid #90caf9; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #90caf9; border-radius: 8px; }
            QTabBar::tab { background: #bbdefb; color: #1a237e; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #1976d2; color: #fff; }
        """,
        "Green": """
            QWidget { background: #e8f5e9; color: #1b5e20; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #e8f5e9; }
            QTableView { background: #fff; border-radius: 8px; padding: 4px; }
            QPushButton { background: #388e3c; color: #fff; border-radius: 6px; padding: 6px 16px; border: 1px solid #2e7d32; }
            QPushButton:hover { background: #2e7d32; }
            QComboBox, QLineEdit, QDateEdit { background: #fff; border-radius: 6px; border: 1px solid #a5d6a7; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #a5d6a7; border-radius: 8px; }
            QTabBar::tab { background: #c8e6c9; color: #1b5e20; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #388e3c; color: #fff; }
        """,
        "Purple": """
            QWidget { background: #f3e5f5; color: #4a148c; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #f3e5f5; }
            QTableView { background: #fff; border-radius: 8px; padding: 4px; }
            QPushButton { background: #8e24aa; color: #fff; border-radius: 6px; padding: 6px 16px; border: 1px solid #6a1b9a; }
            QPushButton:hover { background: #6a1b9a; }
            QComboBox, QLineEdit, QDateEdit { background: #fff; border-radius: 6px; border: 1px solid #ce93d8; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #ce93d8; border-radius: 8px; }
            QTabBar::tab { background: #e1bee7; color: #4a148c; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #8e24aa; color: #fff; }
        """
    }
    def __init__(self):
        logger.debug("MainWindow.__init__ started")
        super().__init__()
        logger.debug("QMainWindow super().__init__ completed")
        
        # Initialize logger for this instance
        self.logger = logger.getChild(self.__class__.__name__)
        
        try:
            self.ai_predictor = AIPredictor()  # Initialize AI predictor
            logger.debug("AI predictor initialized")
            
            self.setWindowTitle("PSX Stock Analysis Dashboard")
            self.setGeometry(100, 100, 1400, 900)
            logger.debug("Window title and geometry set")
            
            self.db_path = "data/databases/production/PSX_investing_Stocks_KMI100.db"
            self.portfolio_data = []  # Data store for portfolio
            self.current_stock_data = None  # Initialize current stock data
            
            logger.debug("Initializing menu")
            self._init_menu()
            logger.debug("Menu initialized")
            
            logger.debug("Initializing tabs")
            self._init_tabs()
            logger.debug("Tabs initialized")
            
            logger.debug("Initializing status bar")
            self._init_status_bar()
            logger.debug("Status bar initialized")
            
            logger.debug("Applying dark mode")
            self._apply_dark_mode(False)
            logger.debug("Dark mode applied")
            
            logger.debug("Loading initial data")
            self.load_data()  # Load data after status bar is initialized
            self.signal_data_ready.connect(self.on_signal_data_loaded)
            logger.debug("MainWindow initialization complete")
            
        except Exception as e:
            logger.error(f"Error during MainWindow initialization: {str(e)}", exc_info=True)
            raise

    def _init_menu(self):
        menubar = QMenuBar(self)
        file_menu = QMenu("File", self)
        export_action = QAction("Export", self)
        settings_menu = QMenu("Settings", self)
        # Theme submenu
        theme_menu = QMenu("Theme", self)
        self.theme_actions = {}
        for theme_name in self.THEMES:
            action = QAction(theme_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, t=theme_name: self.set_theme(t))
            theme_menu.addAction(action)
            self.theme_actions[theme_name] = action
        settings_menu.addMenu(theme_menu)
        dark_mode_action = QAction("Toggle Dark/Light Mode", self)
        help_menu = QMenu("Help", self)
        about_action = QAction("About", self)
        file_menu.addAction(export_action)
        settings_menu.addAction(dark_mode_action)
        help_menu.addAction(about_action)
        menubar.addMenu(file_menu)
        menubar.addMenu(settings_menu)
        menubar.addMenu(help_menu)
        self.setMenuBar(menubar)
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        export_action.triggered.connect(self.export_data)
        about_action.triggered.connect(self.show_about)
        # Set default theme
        self.set_theme("Dark")

    def _init_tabs(self):
        self.tabs = QTabWidget()
        # --- Signals Tab ---
        self.signals_tab = QWidget()
        self._init_signals_tab()
        self.tabs.addTab(self.signals_tab, "Signals")
        # --- Performance Tab ---
        self.performance_tab = QWidget()
        self._init_performance_tab()
        self.tabs.addTab(self.performance_tab, "Performance")
        # --- Correlation Tab ---
        self.correlation_tab = QWidget()
        self._init_correlation_tab()
        self.tabs.addTab(self.correlation_tab, "Correlation Analysis")
        # --- Portfolio Tab ---
        self.portfolio_tab = QWidget()
        self._init_portfolio_tab()
        self.tabs.addTab(self.portfolio_tab, "Portfolio")
        self.setCentralWidget(self.tabs)

    def _init_signals_tab(self):
        """Initializes the Signals tab with a modern, card-based layout and adds graphical representation."""
        layout = QVBoxLayout()

        # --- Filters GroupBox ---
        filters_group = QGroupBox("Filters")
        filters_layout = QGridLayout()
        filters_group.setLayout(filters_layout)
        filters_group.setGraphicsEffect(self.create_shadow_effect())

        self.symbol_selector = QComboBox()
        self.load_symbols()
        filters_layout.addWidget(QLabel("Symbol:"), 0, 0)
        filters_layout.addWidget(self.symbol_selector, 0, 1)

        self.start_date_edit = QDateEdit(QDate.currentDate().addMonths(-1))
        self.end_date_edit = QDateEdit(QDate.currentDate())
        filters_layout.addWidget(QLabel("Start Date:"), 1, 0)
        filters_layout.addWidget(self.start_date_edit, 1, 1)
        filters_layout.addWidget(QLabel("End Date:"), 1, 2)
        filters_layout.addWidget(self.end_date_edit, 1, 3)

        self.signal_type_selector = QComboBox()
        self.signal_type_selector.addItems(["All", "BUY", "SELL"])
        filters_layout.addWidget(QLabel("Signal Type:"), 0, 2)
        filters_layout.addWidget(self.signal_type_selector, 0, 3)

        self.fetch_button = QPushButton("Fetch Signals")
        self.fetch_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.fetch_button.clicked.connect(self.load_signal_data)
        filters_layout.addWidget(self.fetch_button, 2, 0, 1, 4)

        layout.addWidget(filters_group)

        # --- Results GroupBox ---
        results_group = QGroupBox("Signal Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        results_group.setGraphicsEffect(self.create_shadow_effect())

        self.table_view = QTableView()
        results_layout.addWidget(self.table_view)
        layout.addWidget(results_group)

        # --- Graphical Representation GroupBox ---
        graph_group = QGroupBox("Signal State & Transitions")
        graph_layout = QVBoxLayout()
        graph_group.setLayout(graph_layout)
        graph_group.setGraphicsEffect(self.create_shadow_effect())

        self.signals_plotly_widget = PlotlyWidget()
        graph_layout.addWidget(self.signals_plotly_widget)
        layout.addWidget(graph_group)

        self.signals_tab.setLayout(layout)

    def update_signals_graph(self, data, headers):
        """Update the graphical representation in the Signals tab."""
        import pandas as pd
        import plotly.graph_objects as go
        if not data or not headers:
            self.signals_plotly_widget.webview.setHtml("<b>No data to display</b>")
            return
        df = pd.DataFrame(data, columns=headers)
        # Current position counts
        if 'signal_type' in df.columns:
            counts = df['signal_type'].value_counts().reindex(['BUY', 'SELL', 'NEUTRAL'], fill_value=0)
        else:
            counts = pd.Series([0,0,0], index=['BUY','SELL','NEUTRAL'])
        # Transitions
        if 'previous_signal_type' in df.columns and 'signal_type' in df.columns:
            transitions = df.groupby(['previous_signal_type', 'signal_type']).size().reset_index(name='count')
            transitions = transitions[transitions['previous_signal_type'].notna() & transitions['signal_type'].notna()]
        else:
            transitions = pd.DataFrame(columns=['previous_signal_type','signal_type','count'])
        # Bar chart for current positions
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color=['green','red','gray']))
        bar_fig.update_layout(title="Current Stock Positions", xaxis_title="Position", yaxis_title="Count")
        # Sankey diagram for transitions
        sankey_fig = go.Figure()
        if not transitions.empty:
            labels = list(set(transitions['previous_signal_type'].dropna().unique()) | set(transitions['signal_type'].dropna().unique()))
            label_map = {k: i for i, k in enumerate(labels)}
            sankey_fig.add_trace(go.Sankey(
                node=dict(label=labels),
                link=dict(
                    source=[label_map[s] for s in transitions['previous_signal_type']],
                    target=[label_map[t] for t in transitions['signal_type']],
                    value=transitions['count'],
                    label=[f"{s}→{t}" for s, t in zip(transitions['previous_signal_type'], transitions['signal_type'])]
                )
            ))
            sankey_fig.update_layout(title_text="Signal State Transitions (from→to)")
        # Combine both charts in HTML
        html = bar_fig.to_html(include_plotlyjs='cdn')
        if not transitions.empty:
            html += sankey_fig.to_html(include_plotlyjs=False)
        self.signals_plotly_widget.webview.setHtml(html)

    def _apply_indicator(self, fig, indicator):
        """Apply specific indicator to figure."""
        indicator_type, params = indicator
        if indicator_type == "bollinger":
            window = params.get('window', 20)
            std_dev = params.get('std_dev', 2)
            self._add_bollinger_bands(fig, window, std_dev)
        elif indicator_type == "rsi":
            window = params.get('window', 14)
            self._add_rsi(fig, window)
        elif indicator_type == "macd":
            self._add_macd(fig)
            
    def _add_bollinger_bands(self, fig, window=20, std_dev=2):
        """Add Bollinger Bands to figure."""
        if 'Close' not in self.df.columns:
            return
            
        rolling_mean = self.df['Close'].rolling(window=window).mean()
        rolling_std = self.df['Close'].rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=upper_band,
            name=f'Upper BB ({window},{std_dev})',
            line=dict(color='rgba(255, 0, 0, 0.5)')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=lower_band,
            name=f'Lower BB ({window},{std_dev})',
            line=dict(color='rgba(0, 255, 0, 0.5)'),
            fill='tonexty',
            fillcolor='rgba(0, 100, 80, 0.1)'
        ))

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    def plot(self, df: pd.DataFrame, x: str, y: str, title: str = ""):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(df[x], df[y], marker='o')
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        self.canvas.draw()
class AIPredictor:
    """AI-based stock prediction class with LSTM and ARIMA models."""
    
    def __init__(self):
        self.logger = logger.getChild(self.__class__.__name__)
        self.lstm_model = None
        self.arima_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        
    def train_lstm(self, data: pd.Series) -> bool:
        """Train LSTM model on historical data."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Install tensorflow to use LSTM predictions.")
            return False
            
        try:
            # Prepare data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, 60)
            
            if len(X) < 10:  # Need minimum data
                return False
                
            # Build model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            
            self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            self.lstm_model.fit(X, y, batch_size=1, epochs=1, verbose=0)
            
            return True
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}", exc_info=True)
            return False
        
    
    def train_arima(self, data: pd.Series) -> bool:
        """Train ARIMA model on historical data."""
        try:
            # Use automatic ARIMA order selection
            self.arima_model = ARIMA(data, order=(5, 1, 0))
            self.arima_model = self.arima_model.fit()
            return True
        except Exception as e:
            self.logger.error(f"Error training ARIMA model: {str(e)}", exc_info=True)
            return False
        
    
    def predict(self, model_type: str, data: pd.Series, days: int = 30) -> Optional[pd.Series]:
        """Generate predictions using specified model."""
        try:
            if model_type.lower() == "lstm" and self.lstm_model is not None:
                return self._predict_lstm(data, days)
            elif model_type.lower() == "arima" and self.arima_model is not None:
                return self._predict_arima(days)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}", exc_info=True)
            return None
            
    
    def _create_sequences(self, data, seq_length):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def _predict_lstm(self, data: pd.Series, days: int) -> pd.Series:
        """Generate LSTM predictions."""
        if not TENSORFLOW_AVAILABLE or self.lstm_model is None:
            return None
            
        # Prepare last 60 days of data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        last_60_days = scaled_data[-60:]
        
        predictions = []
        current_batch = last_60_days.reshape((1, 60, 1))
        
        for _ in range(days):
            pred = self.lstm_model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            
            # Update batch for next prediction
            current_batch = np.append(current_batch[:, 1:, :], 
                                    [[pred]], axis=1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create date range for predictions
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )
        
        return pd.Series(predictions.flatten(), index=future_dates)
    
    def _predict_arima(self, days: int) -> pd.Series:
        """Generate ARIMA predictions."""
        if self.arima_model is None:
            return None
            
        forecast = self.arima_model.forecast(steps=days)
        
        # Create date range for predictions
        last_date = pd.Timestamp.now()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )
        
        return pd.Series(forecast, index=future_dates)

class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(list, list)  # rows, headers
    error = pyqtSignal(str)

    def __init__(self, db_path: str = "data/databases/production/PSX_investing_Stocks_KMI100.db"):
        if not isinstance(db_path, str):
            raise TypeError("db_path must be a string")
        if not db_path.endswith('.db'):
            raise ValueError("Database path must end with .db extension")
        super().__init__()
        self.logger = logger.getChild(self.__class__.__name__)
        self.db_path = db_path
        self.signal_tables = [
            ("buy_stocks", "BUY"),
            ("sell_stocks", "SELL"),
            ("neutral_stocks", "NEUTRAL")
        ]

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            data, headers = loop.run_until_complete(self.fetch_all_signals_with_transitions())
            self.data_loaded.emit(data, headers)
        except Exception as e:
            self.error.emit(str(e))

    async def fetch_all_signals_with_transitions(self):
        all_rows = []
        all_headers = set()
        stock_entries = defaultdict(list)
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for table, signal_type in self.signal_tables:
                    try:
                        async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                            columns = [row[1] for row in await pragma_cursor.fetchall()]
                        if not columns:
                            continue
                        select_cols = ', '.join([f'"{col}"' for col in columns])
                        async with db.execute(f"SELECT {select_cols} FROM {table}") as cursor:
                            rows = await cursor.fetchall()
                            if not rows:
                                self.logger.warning(f"No data found in table {table}")
                                continue
                            for row in rows:
                                row_dict = dict(zip(columns, row))
                                row_dict['signal_type'] = signal_type
                                # Use 'Stock' or 'symbol' as the key (try both)
                                stock_key = row_dict.get('Stock') or row_dict.get('symbol')
                                date_val = row_dict.get('Date') or row_dict.get('date')
                                if stock_key and date_val:
                                    stock_entries[stock_key].append((date_val, row_dict))
                                all_headers.update(columns)
                    except aiosqlite.Error as e:
                        self.logger.error(f"Database error in table {table}: {str(e)}", exc_info=True)
                        continue  # skip problematic tables
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}", exc_info=True)
            raise
        # Sort entries for each stock by date
        for stock, entries in stock_entries.items():
            entries.sort(key=lambda x: x[0])
            prev_signal = None
            prev_date = None
            for idx, (date_val, row_dict) in enumerate(entries):
                row_dict['previous_signal_type'] = prev_signal if prev_signal else ''
                row_dict['transition_date'] = date_val if prev_signal and prev_signal != row_dict['signal_type'] else ''
                prev_signal = row_dict['signal_type']
                prev_date = date_val
                all_rows.append(row_dict)
        all_headers = list(all_headers)
        for extra_col in ['signal_type', 'previous_signal_type', 'transition_date']:
            if extra_col not in all_headers:
                all_headers.append(extra_col)
        # Ensure all rows have all headers
        final_rows = []
        for row in all_rows:
            final_rows.append([row.get(h, '') for h in all_headers])
        return final_rows, all_headers

class MainWindow(QMainWindow):
    """Main application window for the Stock Analysis GUI."""
    signal_data_ready = pyqtSignal(list, list)
    prediction_ready = pyqtSignal(str, pd.Series)  # model_type, predictions
    
    THEMES = {
        "Light": """
            QWidget { background: #f7f7f7; color: #222; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #f7f7f7; }
            QTableView { background: #fff; border-radius: 8px; padding: 4px; }
            QPushButton { background: #e0e0e0; border-radius: 6px; padding: 6px 16px; border: 1px solid #ccc; }
            QPushButton:hover { background: #d0d0ff; }
            QComboBox, QLineEdit, QDateEdit { background: #fff; border-radius: 6px; border: 1px solid #bbb; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #bbb; border-radius: 8px; }
            QTabBar::tab { background: #e0e0e0; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #d0d0ff; }
        """,
        "Dark": """
            QWidget { background: #232629; color: #f0f0f0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #232629; }
            QTableView { background: #1e1e1e; border-radius: 8px; padding: 4px; color: #f0f0f0; }
            QPushButton { background: #444; color: #fff; border-radius: 6px; padding: 6px 16px; border: 1px solid #666; }
            QPushButton:hover { background: #5a5a8a; }
            QComboBox, QLineEdit, QDateEdit { background: #333; color: #fff; border-radius: 6px; border: 1px solid #666; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #444; border-radius: 8px; }
            QTabBar::tab { background: #444; color: #fff; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #5a5a8a; }
        """,
        "Blue": """
            QWidget { background: #eaf3fb; color: #1a237e; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #eaf3fb; }
            QTableView { background: #fff; border-radius: 8px; padding: 4px; }
            QPushButton { background: #1976d2; color: #fff; border-radius: 6px; padding: 6px 16px; border: 1px solid #1565c0; }
            QPushButton:hover { background: #1565c0; }
            QComboBox, QLineEdit, QDateEdit { background: #fff; border-radius: 6px; border: 1px solid #90caf9; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #90caf9; border-radius: 8px; }
            QTabBar::tab { background: #bbdefb; color: #1a237e; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #1976d2; color: #fff; }
        """,
        "Green": """
            QWidget { background: #e8f5e9; color: #1b5e20; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #e8f5e9; }
            QTableView { background: #fff; border-radius: 8px; padding: 4px; }
            QPushButton { background: #388e3c; color: #fff; border-radius: 6px; padding: 6px 16px; border: 1px solid #2e7d32; }
            QPushButton:hover { background: #2e7d32; }
            QComboBox, QLineEdit, QDateEdit { background: #fff; border-radius: 6px; border: 1px solid #a5d6a7; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #a5d6a7; border-radius: 8px; }
            QTabBar::tab { background: #c8e6c9; color: #1b5e20; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #388e3c; color: #fff; }
        """,
        "Purple": """
            QWidget { background: #f3e5f5; color: #4a148c; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QMainWindow { background: #f3e5f5; }
            QTableView { background: #fff; border-radius: 8px; padding: 4px; }
            QPushButton { background: #8e24aa; color: #fff; border-radius: 6px; padding: 6px 16px; border: 1px solid #6a1b9a; }
            QPushButton:hover { background: #6a1b9a; }
            QComboBox, QLineEdit, QDateEdit { background: #fff; border-radius: 6px; border: 1px solid #ce93d8; padding: 4px 8px; }
            QTabWidget::pane { border: 1px solid #ce93d8; border-radius: 8px; }
            QTabBar::tab { background: #e1bee7; color: #4a148c; border-radius: 6px; padding: 6px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #8e24aa; color: #fff; }
        """
    }
    def __init__(self):
        logger.debug("MainWindow.__init__ started")
        super().__init__()
        logger.debug("QMainWindow super().__init__ completed")
        
        # Initialize logger for this instance
        self.logger = logger.getChild(self.__class__.__name__)
        
        try:
            self.ai_predictor = AIPredictor()  # Initialize AI predictor
            logger.debug("AI predictor initialized")
            
            self.setWindowTitle("PSX Stock Analysis Dashboard")
            self.setGeometry(100, 100, 1400, 900)
            logger.debug("Window title and geometry set")
            
            self.db_path = "data/databases/production/PSX_investing_Stocks_KMI100.db"
            self.portfolio_data = []  # Data store for portfolio
            self.current_stock_data = None  # Initialize current stock data
            
            logger.debug("Initializing menu")
            self._init_menu()
            logger.debug("Menu initialized")
            
            logger.debug("Initializing tabs")
            self._init_tabs()
            logger.debug("Tabs initialized")
            
            logger.debug("Initializing status bar")
            self._init_status_bar()
            logger.debug("Status bar initialized")
            
            logger.debug("Applying dark mode")
            self._apply_dark_mode(False)
            logger.debug("Dark mode applied")
            
            logger.debug("Loading initial data")
            self.load_data()  # Load data after status bar is initialized
            self.signal_data_ready.connect(self.on_signal_data_loaded)
            logger.debug("MainWindow initialization complete")
            
        except Exception as e:
            logger.error(f"Error during MainWindow initialization: {str(e)}", exc_info=True)
            raise

    def _init_menu(self):
        menubar = QMenuBar(self)
        file_menu = QMenu("File", self)
        export_action = QAction("Export", self)
        settings_menu = QMenu("Settings", self)
        # Theme submenu
        theme_menu = QMenu("Theme", self)
        self.theme_actions = {}
        for theme_name in self.THEMES:
            action = QAction(theme_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, t=theme_name: self.set_theme(t))
            theme_menu.addAction(action)
            self.theme_actions[theme_name] = action
        settings_menu.addMenu(theme_menu)
        dark_mode_action = QAction("Toggle Dark/Light Mode", self)
        help_menu = QMenu("Help", self)
        about_action = QAction("About", self)
        file_menu.addAction(export_action)
        settings_menu.addAction(dark_mode_action)
        help_menu.addAction(about_action)
        menubar.addMenu(file_menu)
        menubar.addMenu(settings_menu)
        menubar.addMenu(help_menu)
        self.setMenuBar(menubar)
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        export_action.triggered.connect(self.export_data)
        about_action.triggered.connect(self.show_about)
        # Set default theme
        self.set_theme("Dark")

    def _init_tabs(self):
        self.tabs = QTabWidget()
        # --- Signals Tab ---
        self.signals_tab = QWidget()
        self._init_signals_tab()
        self.tabs.addTab(self.signals_tab, "Signals")
        # --- Performance Tab ---
        self.performance_tab = QWidget()
        self._init_performance_tab()
        self.tabs.addTab(self.performance_tab, "Performance")
        # --- Correlation Tab ---
        self.correlation_tab = QWidget()
        self._init_correlation_tab()
        self.tabs.addTab(self.correlation_tab, "Correlation Analysis")
        # --- Portfolio Tab ---
        self.portfolio_tab = QWidget()
        self._init_portfolio_tab()
        self.tabs.addTab(self.portfolio_tab, "Portfolio")
        self.setCentralWidget(self.tabs)

    def _init_signals_tab(self):
        """Initializes the Signals tab with a modern, card-based layout and adds graphical representation."""
        layout = QVBoxLayout()

        # --- Filters GroupBox ---
        filters_group = QGroupBox("Filters")
        filters_layout = QGridLayout()
        filters_group.setLayout(filters_layout)
        filters_group.setGraphicsEffect(self.create_shadow_effect())

        self.symbol_selector = QComboBox()
        self.load_symbols()
        filters_layout.addWidget(QLabel("Symbol:"), 0, 0)
        filters_layout.addWidget(self.symbol_selector, 0, 1)

        self.start_date_edit = QDateEdit(QDate.currentDate().addMonths(-1))
        self.end_date_edit = QDateEdit(QDate.currentDate())
        filters_layout.addWidget(QLabel("Start Date:"), 1, 0)
        filters_layout.addWidget(self.start_date_edit, 1, 1)
        filters_layout.addWidget(QLabel("End Date:"), 1, 2)
        filters_layout.addWidget(self.end_date_edit, 1, 3)

        self.signal_type_selector = QComboBox()
        self.signal_type_selector.addItems(["All", "BUY", "SELL"])
        filters_layout.addWidget(QLabel("Signal Type:"), 0, 2)
        filters_layout.addWidget(self.signal_type_selector, 0, 3)

        self.fetch_button = QPushButton("Fetch Signals")
        self.fetch_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.fetch_button.clicked.connect(self.load_signal_data)
        filters_layout.addWidget(self.fetch_button, 2, 0, 1, 4)

        layout.addWidget(filters_group)

        # --- Results GroupBox ---
        results_group = QGroupBox("Signal Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        results_group.setGraphicsEffect(self.create_shadow_effect())

        self.table_view = QTableView()
        results_layout.addWidget(self.table_view)
        layout.addWidget(results_group)

        # --- Graphical Representation GroupBox ---
        graph_group = QGroupBox("Signal State & Transitions")
        graph_layout = QVBoxLayout()
        graph_group.setLayout(graph_layout)
        graph_group.setGraphicsEffect(self.create_shadow_effect())

        self.signals_plotly_widget = PlotlyWidget()
        graph_layout.addWidget(self.signals_plotly_widget)
        layout.addWidget(graph_group)

        self.signals_tab.setLayout(layout)

    def update_signals_graph(self, data, headers):
        """Update the graphical representation in the Signals tab."""
        import pandas as pd
        import plotly.graph_objects as go
        if not data or not headers:
            self.signals_plotly_widget.webview.setHtml("<b>No data to display</b>")
            return
        df = pd.DataFrame(data, columns=headers)
        # Current position counts
        if 'signal_type' in df.columns:
            counts = df['signal_type'].value_counts().reindex(['BUY', 'SELL', 'NEUTRAL'], fill_value=0)
        else:
            counts = pd.Series([0,0,0], index=['BUY','SELL','NEUTRAL'])
        # Transitions
        if 'previous_signal_type' in df.columns and 'signal_type' in df.columns:
            transitions = df.groupby(['previous_signal_type', 'signal_type']).size().reset_index(name='count')
            transitions = transitions[transitions['previous_signal_type'].notna() & transitions['signal_type'].notna()]
        else:
            transitions = pd.DataFrame(columns=['previous_signal_type','signal_type','count'])
        # Bar chart for current positions
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color=['green','red','gray']))
        bar_fig.update_layout(title="Current Stock Positions", xaxis_title="Position", yaxis_title="Count")
        # Sankey diagram for transitions
        sankey_fig = go.Figure()
        if not transitions.empty:
            labels = list(set(transitions['previous_signal_type'].dropna().unique()) | set(transitions['signal_type'].dropna().unique()))
            label_map = {k: i for i, k in enumerate(labels)}
            sankey_fig.add_trace(go.Sankey(
                node=dict(label=labels),
                link=dict(
                    source=[label_map[s] for s in transitions['previous_signal_type']],
                    target=[label_map[t] for t in transitions['signal_type']],
                    value=transitions['count'],
                    label=[f"{s}→{t}" for s, t in zip(transitions['previous_signal_type'], transitions['signal_type'])]
                )
            ))
            sankey_fig.update_layout(title_text="Signal State Transitions (from→to)")
        # Combine both charts in HTML
        html = bar_fig.to_html(include_plotlyjs='cdn')
        if not transitions.empty:
            html += sankey_fig.to_html(include_plotlyjs=False)
        self.signals_plotly_widget.webview.setHtml(html)

    def _init_performance_tab(self):
        """Initializes the Performance tab with a modern, card-based layout."""
        layout = QVBoxLayout()
        
        # --- AI Prediction Controls ---
        prediction_group = QGroupBox("AI Prediction")
        prediction_layout = QGridLayout()
        prediction_group.setLayout(prediction_layout)
        prediction_group.setGraphicsEffect(self.create_shadow_effect())
        
        self.model_selector = QComboBox()
        self.model_selector.addItems(["LSTM", "ARIMA"])
        prediction_layout.addWidget(QLabel("Model:"), 0, 0)
        prediction_layout.addWidget(self.model_selector, 0, 1)
        
        self.prediction_days = QLineEdit("30")
        self.prediction_days.setValidator(QIntValidator(1, 365))
        prediction_layout.addWidget(QLabel("Days to Predict:"), 0, 2)
        prediction_layout.addWidget(self.prediction_days, 0, 3)
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_prediction_model)
        prediction_layout.addWidget(self.train_button, 1, 0, 1, 2)
        
        self.predict_button = QPushButton("Generate Prediction")
        self.predict_button.clicked.connect(self.generate_prediction)
        prediction_layout.addWidget(self.predict_button, 1, 2, 1, 2)
        
        layout.addWidget(prediction_group)

        # --- Controls GroupBox ---
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout()
        controls_group.setLayout(controls_layout)
        controls_group.setGraphicsEffect(self.create_shadow_effect())

        self.performance_stock_selector = QComboBox()
        self.populate_performance_stock_selector()
        controls_layout.addWidget(QLabel("Select Stock:"), 0, 0)
        controls_layout.addWidget(self.performance_stock_selector, 0, 1, 1, 3)
        self.performance_stock_selector.currentIndexChanged.connect(self.on_performance_stock_selected)
        
        layout.addWidget(controls_group)

        # --- Analytics GroupBox ---
        analytics_group = QGroupBox("Performance Analytics")
        analytics_layout = QHBoxLayout()
        analytics_group.setLayout(analytics_layout)
        analytics_group.setGraphicsEffect(self.create_shadow_effect())

        self.performance_stats_table = QTableView()
        self.performance_stats_model = QStandardItemModel()
        self.performance_stats_table.setModel(self.performance_stats_model)
        analytics_layout.addWidget(self.performance_stats_table)

        self.performance_chart = PlotlyWidget() # Use interactive Plotly chart
        analytics_layout.addWidget(self.performance_chart)
        
        analytics_layout.setStretch(0, 1) # Give table 1/3 of space
        analytics_layout.setStretch(1, 2) # Give chart 2/3 of space

        layout.addWidget(analytics_group)
        self.performance_tab.setLayout(layout)

    def populate_performance_stock_selector(self):
        # Fetch all unique stock symbols from all three tables
        async def fetch_symbols():
            db_path = "data/databases/production/PSX_investing_Stocks_KMI100.db"
            symbols = set()
            async with aiosqlite.connect(db_path) as db:
                for table in ["buy_stocks", "sell_stocks", "neutral_stocks"]:
                    try:
                        async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                            columns = [row[1] for row in await pragma_cursor.fetchall()]
                        symbol_col = 'Stock' if 'Stock' in columns else 'symbol' if 'symbol' in columns else None
                        if not symbol_col:
                            continue
                        async with db.execute(f"SELECT DISTINCT {symbol_col} FROM {table}") as cursor:
                            rows = await cursor.fetchall()
                            for row in rows:
                                if row[0]:
                                    symbols.add(row[0])
                    except Exception:
                        continue
            return sorted(symbols)
        
        def on_symbols(symbols):
            if hasattr(self, 'performance_stock_selector') and self.performance_stock_selector:
                self.performance_stock_selector.blockSignals(True)
                self.performance_stock_selector.clear()
                self.performance_stock_selector.addItem("Select Stock")
                for s in symbols:
                    self.performance_stock_selector.addItem(str(s))
                self.performance_stock_selector.blockSignals(False)
        
        def run_fetch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            symbols = loop.run_until_complete(fetch_symbols())
            # Use QTimer to safely update UI from thread
            QTimer.singleShot(0, lambda: on_symbols(symbols))
        
        threading.Thread(target=run_fetch, daemon=True).start()

    def show_signal_performance_analysis(self, df: pd.DataFrame):
        """Analyze signal performance and visualize metrics."""
        # Calculate success rate for buy and sell signals
        df['Success'] = (df['Close'] > df['Close'].shift(1)) & (df['type'] == 'buy') | (df['Close'] < df['Close'].shift(1)) & (df['type'] == 'sell')
        success_rate = df.groupby('type')['Success'].mean().round(2) * 100

        # Volume analysis
        volume_analysis = df.groupby('type')['Volume'].mean().round(2)

        # RSI impact analysis
        rsi_correlation = df[['rsi_weekly', 'rsi_monthly', 'rsi_quarterly', 'Success']].corr().round(2)

        # Score-based analysis
        score_correlation = df[['technical_score', 'fundamental_score', 'Success']].corr().round(2)

        # Update performance table
        self.performance_stats_model.clear()
        self.performance_stats_model.setHorizontalHeaderLabels(['Metric', 'Buy', 'Sell', 'Neutral'])
        self.performance_stats_model.appendRow([QStandardItem('Success Rate (%)'), QStandardItem(str(success_rate.get('buy', 0))), QStandardItem(str(success_rate.get('sell', 0))), QStandardItem(str(success_rate.get('neutral', 0)))])
        self.performance_stats_model.appendRow([QStandardItem('Avg Volume'), QStandardItem(str(volume_analysis.get('buy', 0))), QStandardItem(str(volume_analysis.get('sell', 0))), QStandardItem(str(volume_analysis.get('neutral', 0)))])
        self.performance_stats_table.resizeColumnsToContents()

        # Plot RSI correlation heatmap
        self.performance_chart.figure.clear()
        ax = self.performance_chart.figure.add_subplot(121)
        cax = ax.matshow(rsi_correlation, cmap="coolwarm")
        self.performance_chart.figure.colorbar(cax)
        ax.set_xticks(range(len(rsi_correlation.columns)))
        ax.set_xticklabels(rsi_correlation.columns, rotation=90)
        ax.set_yticks(range(len(rsi_correlation.index)))
        ax.set_yticklabels(rsi_correlation.index)

        # Plot score correlation heatmap
        ax2 = self.performance_chart.figure.add_subplot(122)
        cax2 = ax2.matshow(score_correlation, cmap="coolwarm")
        self.performance_chart.figure.colorbar(cax2)
        ax2.set_xticks(range(len(score_correlation.columns)))
        ax2.set_xticklabels(score_correlation.columns, rotation=90)
        ax2.set_yticks(range(len(score_correlation.index)))
        ax2.set_yticklabels(score_correlation.index)

        self.performance_chart.canvas.draw()

    def on_performance_stock_selected(self, idx):
        symbol = self.performance_stock_selector.currentText()
        self.current_stock_data = None  # Reset when new stock selected
        if symbol == "Select Stock" or not symbol:
            self.performance_stats_model.clear()
            self.performance_chart.figure.clear()
            self.performance_chart.canvas.draw()
            return
        # Fetch and analyze data for the selected stock
        def fetch_and_analyze():
            db_path = "data/databases/production/PSX_investing_Stocks_KMI100.db"
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(self.fetch_stock_data(symbol, db_path))
            if df is not None and not df.empty:
                self.show_signal_performance_analysis(df)
        threading.Thread(target=fetch_and_analyze, daemon=True).start()

    async def fetch_stock_data(self, symbol: str, db_path: str) -> pd.DataFrame:
        dfs = []
        async with aiosqlite.connect(db_path) as db:
            for table, signal_type in [("buy_stocks", "buy"), ("sell_stocks", "sell"), ("neutral_stocks", "neutral")]:
                try:
                    async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                        columns = [row[1] for row in await pragma_cursor.fetchall()]
                    symbol_col = 'Stock' if 'Stock' in columns else 'symbol' if 'symbol' in columns else None
                    date_col = 'Date' if 'Date' in columns else 'date' if 'date' in columns else None
                    close_col = 'Close' if 'Close' in columns else 'close' if 'close' in columns else None
                    volume_col = 'Volume' if 'Volume' in columns else 'volume' if 'volume' in columns else None
                    if not symbol_col or not date_col or not close_col:
                        continue
                    async with db.execute(f"SELECT {date_col}, {close_col}, {volume_col} FROM {table} WHERE {symbol_col} = ?", (symbol,)) as cursor:
                        rows = await cursor.fetchall()
                        if rows:
                            df = pd.DataFrame(rows, columns=[date_col, close_col, volume_col])
                            df['source_table'] = table
                            df['type'] = signal_type  # Add the 'type' column
                            dfs.append(df)
                except Exception:
                    continue
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df = df.dropna(subset=[df.columns[0], df.columns[1]])
            df = df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Close', df.columns[2]: 'Volume'})
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')
            return df
        if not df.empty:
            self.current_stock_data = df['Close']  # Store for predictions
        return df

    def show_performance_analysis(self, df: pd.DataFrame, symbol: str):
        df['Month'] = df['Date'].dt.to_period('M')
        monthly = df.groupby('Month').agg({
            'Close': ['first', 'last', 'mean'],
            'Volume': 'sum'
        }).reset_index()
        monthly.columns = ['Month', 'First Close', 'Last Close', 'Avg Close', 'Total Volume']
        monthly['Return %'] = ((monthly['Last Close'] - monthly['First Close']) / monthly['First Close'] * 100).round(2)

        # Calculate signal performance
        df['Year'] = df['Date'].dt.year
        yearly = df.groupby('Year').agg({
            'Close': ['first', 'last', 'mean'],
            'Volume': 'sum'
        }).reset_index()
        yearly.columns = ['Year', 'First Close', 'Last Close', 'Avg Close', 'Total Volume']
        yearly['Return %'] = ((yearly['Last Close'] - yearly['First Close']) / yearly['First Close'] * 100).round(2)

        # Add signal performance metrics
        monthly['Total Signals'] = df.groupby(df['Date'].dt.to_period('M')).size().values
        monthly['Success Signals'] = df[df['Close'] > df['Close'].shift(1)].groupby(df['Date'].dt.to_period('M')).size().reindex(monthly['Month'], fill_value=0).values
        monthly['Profit'] = (monthly['Success Signals'] / monthly['Total Signals'] * 100).round(2)

        yearly['Total Signals'] = df.groupby(df['Date'].dt.year).size().values
        yearly['Success Signals'] = df[df['Close'] > df['Close'].shift(1)].groupby(df['Date'].dt.year).size().reindex(yearly['Year'], fill_value=0).values
        yearly['Profit'] = (yearly['Success Signals'] / yearly['Total Signals'] * 100).round(2)

        # Show monthly in table by default
        self.performance_stats_model.clear()
        self.performance_stats_model.setHorizontalHeaderLabels(list(monthly.columns))
        for _, row in monthly.iterrows():
            items = [QStandardItem(str(row[col])) for col in monthly.columns]
            self.performance_stats_model.appendRow(items)
        self.performance_stats_table.resizeColumnsToContents()

        # Plot monthly close price
        self.performance_chart.plot(monthly, x='Month', y='Avg Close', title=f"{symbol} Monthly Avg Close")
        self.plotly_widget.plot(monthly, x='Month', y='Profit', title=f"{symbol} Monthly Profit %")

    def _init_correlation_tab(self):
        """Initializes the Correlation Analysis tab with a modern, card-based layout."""
        layout = QVBoxLayout()

        # --- Controls GroupBox ---
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout()
        controls_group.setLayout(controls_layout)
        controls_group.setGraphicsEffect(self.create_shadow_effect())

        self.correlation_stock_selector = QComboBox()
        self.populate_correlation_stock_selector()
        controls_layout.addWidget(QLabel("Select Stock:"), 0, 0)
        controls_layout.addWidget(self.correlation_stock_selector, 0, 1)

        self.external_data_selector = QComboBox()
        self.external_data_selector.addItems(["Gold", "Oil", "KSE100"])
        controls_layout.addWidget(QLabel("External Data:"), 0, 2)
        controls_layout.addWidget(self.external_data_selector, 0, 3)

        self.analyze_correlation_button = QPushButton("Analyze Correlation")
        self.analyze_correlation_button.setIcon(QIcon.fromTheme("system-search"))
        self.analyze_correlation_button.clicked.connect(self.on_analyze_correlation)
        controls_layout.addWidget(self.analyze_correlation_button, 1, 0, 1, 4)

        layout.addWidget(controls_group)

        # --- Results GroupBox ---
        results_group = QGroupBox("Correlation Results")
        results_layout = QHBoxLayout()
        results_group.setLayout(results_layout)
        results_group.setGraphicsEffect(self.create_shadow_effect())

        self.correlation_table = QTableView()
        self.correlation_table_model = QStandardItemModel()
        self.correlation_table.setModel(self.correlation_table_model)
        results_layout.addWidget(self.correlation_table)

        self.correlation_heatmap = PlotlyWidget()
        results_layout.addWidget(self.correlation_heatmap)
        
        results_layout.setStretch(0, 1)
        results_layout.setStretch(1, 2)

        layout.addWidget(results_group)
        self.correlation_tab.setLayout(layout)

    def populate_correlation_stock_selector(self):
        """Populate the stock selector in the Correlation tab."""
        async def fetch_symbols():
            symbols = set()
            async with aiosqlite.connect(self.db_path) as db:
                for table in ["buy_stocks", "sell_stocks", "neutral_stocks"]:
                    try:
                        async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                            columns = [row[1] for row in await pragma_cursor.fetchall()]
                        symbol_col = 'Stock' if 'Stock' in columns else 'symbol' if 'symbol' in columns else None
                        if not symbol_col:
                            continue
                        async with db.execute(f"SELECT DISTINCT {symbol_col} FROM {table}") as cursor:
                            rows = await cursor.fetchall()
                            for row in rows:
                                if row[0]:
                                    symbols.add(row[0])
                    except Exception:
                        continue
            return sorted(symbols)

        def on_symbols(symbols):
            if hasattr(self, 'correlation_stock_selector') and self.correlation_stock_selector:
                self.correlation_stock_selector.clear()
                self.correlation_stock_selector.addItem("Select Stock")
                for s in symbols:
                    self.correlation_stock_selector.addItem(str(s))

        def run_fetch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            symbols = loop.run_until_complete(fetch_symbols())
            # Use QTimer to safely update UI from thread
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: on_symbols(symbols))

        threading.Thread(target=run_fetch, daemon=True).start()

    def on_analyze_correlation(self):
        """Analyze correlation between stock signals and external data."""
        stock = self.correlation_stock_selector.currentText()
        external_data = self.external_data_selector.currentText()

        if not stock or not external_data:
            self.status.showMessage("Select both stock and external data.")
            return

        # Placeholder: Perform correlation analysis
        self.status.showMessage(f"Analyzing correlation for {stock} with {external_data}...")
        threading.Thread(target=self.perform_correlation_analysis, daemon=True).start()

    def perform_correlation_analysis(self):
        stock = self.correlation_stock_selector.currentText()
        external_data = self.external_data_selector.currentText()

        # Simulate time