import sys
import os
import asyncio
import threading
import pandas as pd
import aiosqlite
import matplotlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTableView, QComboBox, 
    QPushButton, QLabel, QDateEdit, QLineEdit, QStatusBar, QMenuBar, QMenu, 
    QFileDialog, QTabWidget, QGroupBox, QGraphicsDropShadowEffect, QGridLayout
)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal, QAbstractTableModel, QModelIndex, QTimer
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem, QAction, QColor
from typing import Optional, List, Dict, Any
from src.data_processing.stock_analysis import db_handler
from collections import defaultdict
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go

class PlotlyWidget(QWidget):
    """Interactive Plotly chart widget using QWebEngineView."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.webview = QWebEngineView()
        layout = QVBoxLayout(self)
        layout.addWidget(self.webview)
        
    def plot(self, df: pd.DataFrame, x: str, y: str, title: str = ""):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='lines+markers'))
        fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
        self.webview.setHtml(fig.to_html(include_plotlyjs='cdn'))

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

class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(list, list)  # rows, headers
    error = pyqtSignal(str)

    def __init__(self, db_path: str = "data/databases/production/PSX_investing_Stocks_KMI100.db"):
        super().__init__()
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
                        for row in rows:
                            row_dict = dict(zip(columns, row))
                            row_dict['signal_type'] = signal_type
                            # Use 'Stock' or 'symbol' as the key (try both)
                            stock_key = row_dict.get('Stock') or row_dict.get('symbol')
                            date_val = row_dict.get('Date') or row_dict.get('date')
                            if stock_key and date_val:
                                stock_entries[stock_key].append((date_val, row_dict))
                            all_headers.update(columns)
                except Exception as e:
                    continue  # skip missing tables
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
        """
    }
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSX Stock Analysis Dashboard - Modern Edition")
        self.setGeometry(100, 100, 1400, 900)
        self.db_path = "data/databases/production/PSX_investing_Stocks_KMI100.db"
        self.portfolio_data = []
        
        self._init_menu()
        self._init_tabs()
        self._init_status_bar()
        self.set_theme("Dark")
        self.load_data()
        self.signal_data_ready.connect(self.on_signal_data_loaded)

    def _init_menu(self):
        menubar = QMenuBar(self)
        
        # File menu
        file_menu = QMenu("File", self)
        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        # Settings menu
        settings_menu = QMenu("Settings", self)
        theme_menu = QMenu("Theme", self)
        self.theme_actions = {}
        for theme_name in self.THEMES:
            action = QAction(theme_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, t=theme_name: self.set_theme(t))
            theme_menu.addAction(action)
            self.theme_actions[theme_name] = action
        settings_menu.addMenu(theme_menu)
        
        # Help menu
        help_menu = QMenu("Help", self)
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        menubar.addMenu(file_menu)
        menubar.addMenu(settings_menu)
        menubar.addMenu(help_menu)
        self.setMenuBar(menubar)

    def _init_tabs(self):
        self.tabs = QTabWidget()
        
        # Signals Tab
        self.signals_tab = QWidget()
        self._init_signals_tab()
        self.tabs.addTab(self.signals_tab, "📈 Signals")
        
        # Performance Tab
        self.performance_tab = QWidget()
        self._init_performance_tab()
        self.tabs.addTab(self.performance_tab, "📊 Performance")
        
        # Portfolio Tab
        self.portfolio_tab = QWidget()
        self._init_portfolio_tab()
        self.tabs.addTab(self.portfolio_tab, "💼 Portfolio")
        
        self.setCentralWidget(self.tabs)

    def _init_signals_tab(self):
        """Initializes the Signals tab with a modern, card-based layout."""
        layout = QVBoxLayout()

        # Filters GroupBox
        filters_group = QGroupBox("🔍 Filters")
        filters_layout = QGridLayout()
        filters_group.setLayout(filters_layout)
        filters_group.setGraphicsEffect(self.create_shadow_effect())

        self.symbol_selector = QComboBox()
        self.symbol_selector.addItem("All Symbols")
        self.load_symbols()
        filters_layout.addWidget(QLabel("Symbol:"), 0, 0)
        filters_layout.addWidget(self.symbol_selector, 0, 1)

        self.start_date_edit = QDateEdit(QDate.currentDate().addMonths(-1))
        self.start_date_edit.setCalendarPopup(True)
        self.end_date_edit = QDateEdit(QDate.currentDate())
        self.end_date_edit.setCalendarPopup(True)
        filters_layout.addWidget(QLabel("Start Date:"), 1, 0)
        filters_layout.addWidget(self.start_date_edit, 1, 1)
        filters_layout.addWidget(QLabel("End Date:"), 1, 2)
        filters_layout.addWidget(self.end_date_edit, 1, 3)

        self.signal_type_selector = QComboBox()
        self.signal_type_selector.addItems(["All", "BUY", "SELL"])
        filters_layout.addWidget(QLabel("Signal Type:"), 0, 2)
        filters_layout.addWidget(self.signal_type_selector, 0, 3)

        self.fetch_button = QPushButton("🔄 Fetch Signals")
        self.fetch_button.clicked.connect(self.load_signal_data)
        filters_layout.addWidget(self.fetch_button, 2, 0, 1, 4)

        layout.addWidget(filters_group)

        # Results GroupBox
        results_group = QGroupBox("📋 Signal Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        results_group.setGraphicsEffect(self.create_shadow_effect())

        self.table_view = QTableView()
        results_layout.addWidget(self.table_view)
        layout.addWidget(results_group)

        self.signals_tab.setLayout(layout)

    def _init_performance_tab(self):
        """Initializes the Performance tab with a modern, card-based layout."""
        layout = QVBoxLayout()

        # Controls GroupBox
        controls_group = QGroupBox("⚙️ Controls")
        controls_layout = QGridLayout()
        controls_group.setLayout(controls_layout)
        controls_group.setGraphicsEffect(self.create_shadow_effect())

        self.performance_stock_selector = QComboBox()
        self.performance_stock_selector.addItem("Select Stock")
        self.populate_performance_stock_selector()
        controls_layout.addWidget(QLabel("Select Stock:"), 0, 0)
        controls_layout.addWidget(self.performance_stock_selector, 0, 1, 1, 3)
        self.performance_stock_selector.currentIndexChanged.connect(self.on_performance_stock_selected)
        
        layout.addWidget(controls_group)

        # Analytics GroupBox
        analytics_group = QGroupBox("📈 Performance Analytics")
        analytics_layout = QHBoxLayout()
        analytics_group.setLayout(analytics_layout)
        analytics_group.setGraphicsEffect(self.create_shadow_effect())

        self.performance_stats_table = QTableView()
        self.performance_stats_model = QStandardItemModel()
        self.performance_stats_table.setModel(self.performance_stats_model)
        analytics_layout.addWidget(self.performance_stats_table)

        self.performance_chart = PlotlyWidget()
        analytics_layout.addWidget(self.performance_chart)
        
        analytics_layout.setStretch(0, 1)
        analytics_layout.setStretch(1, 2)

        layout.addWidget(analytics_group)
        self.performance_tab.setLayout(layout)

    def _init_portfolio_tab(self):
        """Initializes the Portfolio tab with a modern, card-based layout."""
        layout = QVBoxLayout()

        # Manage Portfolio GroupBox
        manage_group = QGroupBox("💰 Manage Portfolio")
        manage_layout = QGridLayout()
        manage_group.setLayout(manage_layout)
        manage_group.setGraphicsEffect(self.create_shadow_effect())

        self.portfolio_stock_selector = QComboBox()
        self.populate_portfolio_stock_selector()
        manage_layout.addWidget(QLabel("Select Stock:"), 0, 0)
        manage_layout.addWidget(self.portfolio_stock_selector, 0, 1)

        self.quantity_input = QLineEdit("100")
        self.quantity_input.setPlaceholderText("Enter quantity")
        manage_layout.addWidget(QLabel("Quantity:"), 0, 2)
        manage_layout.addWidget(self.quantity_input, 0, 3)

        self.add_stock_button = QPushButton("➕ Add to Portfolio")
        self.add_stock_button.clicked.connect(self.add_stock_to_portfolio)
        manage_layout.addWidget(self.add_stock_button, 1, 0, 1, 2)

        self.remove_stock_button = QPushButton("➖ Remove Selected")
        self.remove_stock_button.clicked.connect(self.remove_stock_from_portfolio)
        manage_layout.addWidget(self.remove_stock_button, 1, 2, 1, 2)

        layout.addWidget(manage_group)

        # Holdings GroupBox
        holdings_group = QGroupBox("📊 My Holdings")
        holdings_layout = QVBoxLayout()
        holdings_group.setLayout(holdings_layout)
        holdings_group.setGraphicsEffect(self.create_shadow_effect())

        self.portfolio_table = QTableView()
        self.portfolio_model = QStandardItemModel()
        self.portfolio_model.setHorizontalHeaderLabels(["Symbol", "Quantity", "Purchase Price", "Current Price", "Gain/Loss"])
        self.portfolio_table.setModel(self.portfolio_model)
        holdings_layout.addWidget(self.portfolio_table)

        layout.addWidget(holdings_group)
        self.portfolio_tab.setLayout(layout)

    def _init_status_bar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready - Modern PSX Dashboard")

    def create_shadow_effect(self):
        """Create a shadow effect for widgets."""
        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(15)
        effect.setXOffset(0)
        effect.setYOffset(4)
        effect.setColor(QColor(0, 0, 0, 160))
        return effect

    def load_symbols(self):
        """Load stock symbols from the database for selection."""
        async def fetch_symbols():
            symbols = set()
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    for table in ["buy_stocks", "sell_stocks", "neutral_stocks"]:
                        try:
                            async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                                columns = [row[1] for row in await pragma_cursor.fetchall()]
                            symbol_col = 'Stock' if 'Stock' in columns else 'symbol'
                            if symbol_col in columns:
                                async with db.execute(f"SELECT DISTINCT {symbol_col} FROM {table}") as cursor:
                                    rows = await cursor.fetchall()
                                    for row in rows:
                                        if row[0]:
                                            symbols.add(row[0])
                        except Exception:
                            continue
            except Exception:
                pass
            return sorted(symbols)

        def on_symbols(symbols):
            if hasattr(self, 'symbol_selector') and self.symbol_selector:
                current_items = [self.symbol_selector.itemText(i) for i in range(self.symbol_selector.count())]
                for symbol in symbols:
                    if symbol not in current_items:
                        self.symbol_selector.addItem(str(symbol))

        def run_fetch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            symbols = loop.run_until_complete(fetch_symbols())
            QTimer.singleShot(0, lambda: on_symbols(symbols))

        threading.Thread(target=run_fetch, daemon=True).start()

    def populate_performance_stock_selector(self):
        """Populate performance stock selector."""
        async def fetch_symbols():
            symbols = set()
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    for table in ["buy_stocks", "sell_stocks", "neutral_stocks"]:
                        try:
                            async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                                columns = [row[1] for row in await pragma_cursor.fetchall()]
                            symbol_col = 'Stock' if 'Stock' in columns else 'symbol'
                            if symbol_col in columns:
                                async with db.execute(f"SELECT DISTINCT {symbol_col} FROM {table}") as cursor:
                                    rows = await cursor.fetchall()
                                    for row in rows:
                                        if row[0]:
                                            symbols.add(row[0])
                        except Exception:
                            continue
            except Exception:
                pass
            return sorted(symbols)

        def on_symbols(symbols):
            if hasattr(self, 'performance_stock_selector') and self.performance_stock_selector:
                for symbol in symbols:
                    self.performance_stock_selector.addItem(str(symbol))

        def run_fetch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            symbols = loop.run_until_complete(fetch_symbols())
            QTimer.singleShot(0, lambda: on_symbols(symbols))

        threading.Thread(target=run_fetch, daemon=True).start()

    def populate_portfolio_stock_selector(self):
        """Populate portfolio stock selector."""
        async def fetch_symbols():
            symbols = set()
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    for table in ["buy_stocks", "sell_stocks", "neutral_stocks"]:
                        try:
                            async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                                columns = [row[1] for row in await pragma_cursor.fetchall()]
                            symbol_col = 'Stock' if 'Stock' in columns else 'symbol'
                            if symbol_col in columns:
                                async with db.execute(f"SELECT DISTINCT {symbol_col} FROM {table}") as cursor:
                                    rows = await cursor.fetchall()
                                    for row in rows:
                                        if row[0]:
                                            symbols.add(row[0])
                        except Exception:
                            continue
            except Exception:
                pass
            return sorted(symbols)

        def on_symbols(symbols):
            if hasattr(self, 'portfolio_stock_selector') and self.portfolio_stock_selector:
                for symbol in symbols:
                    self.portfolio_stock_selector.addItem(str(symbol))

        def run_fetch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            symbols = loop.run_until_complete(fetch_symbols())
            QTimer.singleShot(0, lambda: on_symbols(symbols))

        threading.Thread(target=run_fetch, daemon=True).start()

    def load_signal_data(self):
        """Fetch signal data from the database based on selected filters."""
        symbol = self.symbol_selector.currentText()
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        signal_type = self.signal_type_selector.currentText()

        self.status.showMessage(f"Fetching signals for {symbol}...")

        def run_fetch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            data, headers = loop.run_until_complete(
                self.fetch_signal_data(symbol, start_date, end_date, signal_type)
            )
            self.signal_data_ready.emit(data, headers)

        threading.Thread(target=run_fetch, daemon=True).start()

    async def fetch_signal_data(self, symbol, start_date, end_date, signal_type):
        """Asynchronously fetch signal data from the database."""
        all_rows_dicts = []
        all_headers = set()
        
        tables_to_query = []
        if signal_type == 'BUY':
            tables_to_query.append(("buy_stocks", "BUY"))
        elif signal_type == 'SELL':
            tables_to_query.append(("sell_stocks", "SELL"))
        else:  # ALL
            tables_to_query.extend([("buy_stocks", "BUY"), ("sell_stocks", "SELL"), ("neutral_stocks", "NEUTRAL")])

        try:
            async with aiosqlite.connect(self.db_path) as db:
                for table, sig_type in tables_to_query:
                    try:
                        async with db.execute(f"PRAGMA table_info({table})") as pragma_cursor:
                            columns = [row[1] for row in await pragma_cursor.fetchall()]
                        if not columns:
                            continue
                        
                        symbol_col = 'Stock' if 'Stock' in columns else 'symbol'
                        date_col = 'Date' if 'Date' in columns else 'date'

                        query = f"SELECT * FROM {table} WHERE {date_col} BETWEEN ? AND ?"
                        q_params = [start_date, end_date]

                        if symbol != "All Symbols" and symbol_col in columns:
                            query += f" AND {symbol_col} = ?"
                            q_params.append(symbol)

                        async with db.execute(query, q_params) as cursor:
                            rows = await cursor.fetchall()
                            for row in rows:
                                row_dict = dict(zip(columns, row))
                                row_dict['signal_type'] = sig_type
                                all_rows_dicts.append(row_dict)
                                all_headers.update(columns)
                    except Exception:
                        continue
        except Exception:
            pass
        
        headers = sorted(list(all_headers))
        if 'signal_type' not in headers:
            headers.append('signal_type')

        final_rows = []
        for row in all_rows_dicts:
            final_rows.append([row.get(h, '') for h in headers])

        return final_rows, headers

    def on_signal_data_loaded(self, data, headers):
        """Update the signals table view with new data."""
        self.status.showMessage(f"Loaded {len(data)} signals.")
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(headers)
        for row_data in data:
            items = [QStandardItem(str(cell)) for cell in row_data]
            model.appendRow(items)
        
        self.table_view.setModel(model)
        self.table_view.resizeColumnsToContents()

    def on_performance_stock_selected(self, idx):
        """Handle performance stock selection."""
        symbol = self.performance_stock_selector.currentText()
        if symbol == "Select Stock" or not symbol:
            self.performance_stats_model.clear()
            return
        self.status.showMessage(f"Analyzing performance for {symbol}...")

    def add_stock_to_portfolio(self):
        """Add a stock to the portfolio."""
        symbol = self.portfolio_stock_selector.currentText()
        quantity = self.quantity_input.text()
        
        if not symbol or not quantity.isdigit() or int(quantity) <= 0:
            self.status.showMessage("Please select a valid symbol and enter a positive quantity.", 3000)
            return

        # Simulate adding to portfolio with dummy price
        import random
        purchase_price = random.uniform(50, 500)
        
        self.portfolio_data.append({
            "Symbol": symbol,
            "Quantity": int(quantity),
            "Purchase Price": purchase_price,
            "Current Price": purchase_price * random.uniform(0.95, 1.05),
            "Gain/Loss": 0
        })
        
        self.update_portfolio_table()
        self.status.showMessage(f"Added {symbol} to portfolio.", 3000)

    def remove_stock_from_portfolio(self):
        """Remove selected stock from portfolio."""
        selected_indexes = self.portfolio_table.selectionModel().selectedRows()
        if not selected_indexes:
            self.status.showMessage("Please select a stock to remove.", 3000)
            return
        
        for index in sorted(selected_indexes, reverse=True):
            self.portfolio_data.pop(index.row())
        
        self.update_portfolio_table()
        self.status.showMessage("Removed selected stock(s).", 3000)

    def update_portfolio_table(self):
        """Update the portfolio table with current data."""
        self.portfolio_model.clear()
        self.portfolio_model.setHorizontalHeaderLabels(["Symbol", "Quantity", "Purchase Price", "Current Price", "Gain/Loss"])
        
        for item in self.portfolio_data:
            gain_loss = (item['Current Price'] - item['Purchase Price']) * item['Quantity']
            item['Gain/Loss'] = gain_loss
            
            row = [
                QStandardItem(str(item["Symbol"])),
                QStandardItem(str(item["Quantity"])),
                QStandardItem(f"{item['Purchase Price']:.2f}"),
                QStandardItem(f"{item['Current Price']:.2f}"),
                QStandardItem(f"{gain_loss:.2f}")
            ]
            
            # Color coding for gain/loss
            if gain_loss > 0:
                row[-1].setForeground(QColor('green'))
            elif gain_loss < 0:
                row[-1].setForeground(QColor('red'))
            
            self.portfolio_model.appendRow(row)
        
        self.portfolio_table.resizeColumnsToContents()

    def load_data(self):
        """Load initial data."""
        self.status.showMessage("Loading data...")
        self.loader_thread = DataLoaderThread()
        self.loader_thread.data_loaded.connect(self.on_data_loaded)
        self.loader_thread.error.connect(self.on_data_error)
        self.loader_thread.start()

    def on_data_loaded(self, data, headers):
        """Handle loaded data."""
        self.status.showMessage(f"Loaded {len(data)} records.")
        # Initialize table model if needed
        if not hasattr(self, 'table_model') or self.table_model is None:
            self.table_model = QStandardItemModel()
            self.table_view.setModel(self.table_model)
        
        self.table_model.clear()
        self.table_model.setHorizontalHeaderLabels(headers)
        for row in data:
            items = [QStandardItem(str(cell)) for cell in row]
            self.table_model.appendRow(items)
        self.table_view.resizeColumnsToContents()

    def on_data_error(self, error: str):
        """Handle data loading errors."""
        self.status.showMessage(f"Error loading data: {error}")

    def set_theme(self, theme_name: str):
        """Set the application theme."""
        for name, action in self.theme_actions.items():
            action.setChecked(name == theme_name)
        self.setStyleSheet(self.THEMES.get(theme_name, ""))

    def export_data(self):
        """Export data functionality."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if file_path:
            self.status.showMessage(f"Export to {file_path} - Feature coming soon!")

    def show_about(self):
        """Show about dialog."""
        self.status.showMessage("PSX Stock Analysis Dashboard v2.0 - Modern Edition with Enhanced UI")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
