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
                            QRadioButton, QCheckBox, QDialogButtonBox, QAction, QListWidget, QListWidgetItem, QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget, QSizePolicy, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer
from PyQt5.QtGui import QIcon
import pandas as pd
from sqlalchemy import create_engine, inspect
from tqdm import tqdm
from src.data_processing.fix_pandas_ta import ta
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import mplcursors

class ProcessingThread(QThread):
    """Worker thread for database processing to prevent GUI freezing"""
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, data_reader, table_names, selected_indicators):
        super().__init__()
        self.data_reader = data_reader
        self.table_names = table_names
        self.selected_indicators = selected_indicators
        self._is_running = True

    def run(self):
        try:
            total_tables = len(self.table_names)
            for i, table_name in enumerate(self.table_names):
                if not self._is_running:
                    break

                self.log_message.emit(f"Processing table: {table_name}")
                data = self.data_reader.read_data(table_name)
                processed_data = self.data_reader.preprocess(data, self.selected_indicators)
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
        self.setMinimumSize(1100, 700)
        self.resize(1400, 900)
        self.theme = 'dark'  # Default theme (will be overwritten by settings)
        self.data_reader = None
        self.processing_thread = None
        self.init_logging()
        self.init_ui()
        self.load_theme_from_settings()
        # Auto-connect to databases and populate dashboard after UI is ready
        QTimer.singleShot(100, self.auto_connect_and_populate_dashboard)
        
    def init_ui(self):
        """Initialize UI components"""
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
        
        # --- Tabbed Interface ---
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)
        self.tabs.setStyleSheet("")  # Will be set by apply_theme
        
        # Add tabs
        self.dashboard_tab = self.create_dashboard_tab()
        self.tabs.addTab(self.dashboard_tab, "  Dashboard  ")
        self.process_tab = self.create_process_tab()
        self.tabs.addTab(self.process_tab, "  Process Data  ")
        self.visualize_tab = self.create_visualize_tab()
        self.tabs.addTab(self.visualize_tab, "  Visualize  ")
        self.export_tab = self.create_export_tab()
        self.tabs.addTab(self.export_tab, "  Export  ")
        self.settings_tab = self.create_settings_tab()
        self.tabs.addTab(self.settings_tab, "  Settings  ")
        self.help_tab = self.create_help_tab()
        self.tabs.addTab(self.help_tab, "  Help  ")
        
        # --- Set main widget ---
        self.setCentralWidget(self.tabs)
        # Modern main window stylesheet
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #181C20;
                color: #F5F5F5;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 13pt;
            }
            QTabWidget::pane {
                border: 1.5px solid #1976D2;
                background: #23272e;
                border-radius: 12px;
            }
            QTabBar::tab {
                background: #23272e;
                color: #90caf9;
                padding: 12px 36px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                margin-right: 6px;
                font-size: 17px;
                font-weight: bold;
                min-width: 140px;
                max-width: 320px;
            }
            QTabBar::tab:selected {
                background: #1976D2;
                color: #fff;
                font-weight: bold;
            }
            QLineEdit, QTextEdit, QComboBox, QListWidget {
                background: #2d323b;
                color: #f8f8f2;
                border: 1.5px solid #1976D2;
                border-radius: 8px;
                padding: 6px;
            }
            QTableWidget {
                background: #2d323b;
                color: #f8f8f2;
                gridline-color: #1976D2;
                border-radius: 8px;
                alternate-background-color: #23272e;
            }
            QHeaderView::section {
                background-color: #1976D2;
                color: #f8f8f2;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #1976D2;
                color: #fff;
                border-radius: 8px;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 15px;
                min-width: 100px;
                max-width: 1000px;
                text-align: center;
            }
            QPushButton:disabled {
                background-color: #B0BEC5;
                color: #ECEFF1;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QLabel {
                color: #f8f8f2;
            }
            QGroupBox {
                background-color: #23272B;
                color: #F5F5F5;
                border-radius: 12px;
                border: 1.5px solid #1976D2;
                padding: 12px;
            }
        """)
        # Set default paths
        self.set_default_paths()
        
        # Populate visualize tab combos after DB connect
        self.vis_table_combo.currentTextChanged.connect(self.on_vis_table_changed)
        
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
            
            self.table_list.clear()
            for t in tables:
                item = QListWidgetItem(t)
                self.table_list.addItem(item)
            self.table_list.setEnabled(True)
            self.process_btn.setEnabled(True)
            
            self.log_message(f"Successfully connected to databases\nSource: {source_path}\nTarget: {target_path}")
            self.log_message(f"Found {len(tables)} tables in source database")
            self.update_dashboard_cards()
            
            # Update visualize tab table combo with tables from the TARGET database
            target_tables = []
            if hasattr(self.data_reader.target_engine, 'table_names'):
                target_tables = self.data_reader.target_engine.table_names()
            if not target_tables:
                # fallback: use inspector
                try:
                    from sqlalchemy import inspect as sa_inspect
                    inspector = sa_inspect(self.data_reader.target_engine)
                    target_tables = inspector.get_table_names()
                except Exception:
                    target_tables = []
            self.vis_table_combo.clear()
            self.vis_table_combo.addItems(target_tables)
            # Auto-select first table and populate indicator list
            if target_tables:
                self.vis_table_combo.setCurrentIndex(0)
                self.on_vis_table_changed(target_tables[0])
            
            # Update export tab table list (from target DB)
            self.export_table_list.clear()
            for t in target_tables:
                self.export_table_list.addItem(QListWidgetItem(t))
            
        except Exception as e:
            self.log_message(f"Error connecting to databases: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to connect to databases:\n{str(e)}")
            self.update_dashboard_cards()
            
    def process_tables(self):
        """Process selected tables"""
        if not self.data_reader:
            QMessageBox.warning(self, "Error", "Please connect to databases first")
            return
            
        selected_tables = [item.text() for item in self.table_list.selectedItems()]
        
        if not selected_tables:
            QMessageBox.warning(self, "Error", "No tables selected for processing")
            return
        # Gather selected indicators and parameters
        selected_indicators = {}
        for name, cb in self.indicator_checkboxes.items():
            if cb.isChecked():
                param = self.indicator_params[name]
                if isinstance(param, QSpinBox):
                    selected_indicators[name] = param.value()
                elif isinstance(param, tuple):
                    selected_indicators[name] = tuple(p.value() for p in param)
                else:
                    selected_indicators[name] = None
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_dashboard_cards()
        # Pass selected_indicators to ProcessingThread
        self.processing_thread = ProcessingThread(self.data_reader, selected_tables, selected_indicators)
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
        self.update_dashboard_cards()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.update_dashboard_cards()
        
    def processing_finished(self, success):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.log_message("Processing completed successfully")
            QMessageBox.information(self, "Success", "Processing completed successfully")
        else:
            self.log_message("Processing completed with errors")
        self.update_dashboard_cards()
        
    def log_message(self, message):
        """Add message to log display and update last status card"""
        self.log_display.append(message)
        self.statusBar().showMessage(message.split('\n')[0], 5000)
        self.card_last_status.value_label.setText(message.split('\n')[0])
        # Also update dashboard log
        if hasattr(self, 'dashboard_log_display'):
            self.dashboard_log_display.append(message)
        # Add to dashboard activity table if it's a process or visualize event
        if hasattr(self, 'dashboard_activity_table'):
            if any(x in message.lower() for x in ["process", "visualize", "export", "connected", "error"]):
                self.add_dashboard_activity("Log", message.split('\n')[0])

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
            self.theme = settings['theme']
            self.apply_theme(self.theme)
        # Add other setting applications here

    def apply_theme(self, theme):
        """Apply dark or light theme to the entire app, matching the reference GUI."""
        # Modernized stylesheet
        if theme == "dark":
            self.setStyleSheet("""
                QMainWindow, QDialog {
                    background-color: #181C20;
                    color: #F5F5F5;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    font-size: 13pt;
                }
                QTabWidget::pane {
                    border: 1.5px solid #1976D2;
                    background: #23272e;
                    border-radius: 12px;
                }
                QTabBar::tab {
                    background: #23272e;
                    color: #90caf9;
                    padding: 12px 36px;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    margin-right: 6px;
                    font-size: 17px;
                    font-weight: bold;
                    min-width: 140px;
                    max-width: 320px;
                }
                QTabBar::tab:selected {
                    background: #1976D2;
                    color: #fff;
                    font-weight: bold;
                }
                QLineEdit, QTextEdit, QComboBox, QListWidget {
                    background: #2d323b;
                    color: #f8f8f2;
                    border: 1.5px solid #1976D2;
                    border-radius: 8px;
                    padding: 6px;
                }
                QTableWidget {
                    background: #2d323b;
                    color: #f8f8f2;
                    gridline-color: #1976D2;
                    border-radius: 8px;
                    alternate-background-color: #23272e;
                }
                QHeaderView::section {
                    background-color: #1976D2;
                    color: #f8f8f2;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton {
                    background-color: #1976D2;
                    color: #fff;
                    border-radius: 8px;
                    padding: 10px 24px;
                    font-weight: bold;
                    font-size: 15px;
                    min-width: 100px;
                    max-width: 1000px;
                    text-align: center;
                }
                QPushButton:disabled {
                    background-color: #B0BEC5;
                    color: #ECEFF1;
                }
                QPushButton:hover {
                    background-color: #1565C0;
                }
                QLabel {
                    color: #f8f8f2;
                }
                QGroupBox {
                    background-color: #23272B;
                    color: #F5F5F5;
                    border-radius: 12px;
                    border: 1.5px solid #1976D2;
                    padding: 12px;
                }
            """)
            if hasattr(self, 'vis_timeframe_label'):
                self.vis_timeframe_label.setStyleSheet("font-size: 13px; color: #90caf9; margin-top: 8px;")
        elif theme == "light":
            self.setStyleSheet("""
                QMainWindow, QDialog {
                    background-color: #f5f6fa;
                    color: #23272e;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    font-size: 13pt;
                }
                QTabWidget::pane {
                    border: 1.5px solid #1976D2;
                    background: #f5f6fa;
                    border-radius: 12px;
                }
                QTabBar::tab {
                    background: #e1e3ea;
                    color: #1976D2;
                    padding: 12px 36px;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    margin-right: 6px;
                    font-size: 17px;
                    font-weight: bold;
                    min-width: 140px;
                    max-width: 320px;
                }
                QTabBar::tab:selected {
                    background: #1976D2;
                    color: #fff;
                    font-weight: bold;
                }
                QLineEdit, QTextEdit, QComboBox, QListWidget {
                    background: #e1e3ea;
                    color: #23272e;
                    border: 1.5px solid #1976D2;
                    border-radius: 8px;
                    padding: 6px;
                }
                QTableWidget {
                    background: #e1e3ea;
                    color: #23272e;
                    gridline-color: #1976D2;
                    border-radius: 8px;
                    alternate-background-color: #f5f6fa;
                }
                QHeaderView::section {
                    background-color: #1976D2;
                    color: #fff;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton {
                    background-color: #1976D2;
                    color: #fff;
                    border-radius: 8px;
                    padding: 10px 24px;
                    font-weight: bold;
                    font-size: 15px;
                    min-width: 100px;
                    max-width: 1000px;
                    text-align: center;
                }
                QPushButton:disabled {
                    background-color: #B0BEC5;
                    color: #ECEFF1;
                }
                QPushButton:hover {
                    background-color: #1565C0;
                }
                QLabel {
                    color: #23272e;
                }
                QGroupBox {
                    background-color: #ffffff;
                    color: #23272e;
                    border-radius: 12px;
                    border: 1.5px solid #1976D2;
                    padding: 12px;
                }
            """)
            if hasattr(self, 'vis_timeframe_label'):
                self.vis_timeframe_label.setStyleSheet("font-size: 13px; color: #1976D2; margin-top: 8px;")
        elif theme == "blue":
            self.setStyleSheet("""
                QMainWindow, QDialog {
                    background-color: #e3f2fd;
                    color: #0d47a1;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    font-size: 13pt;
                }
                QTabWidget::pane {
                    border: 1.5px solid #1976D2;
                    background: #e3f2fd;
                    border-radius: 12px;
                }
                QTabBar::tab {
                    background: #90caf9;
                    color: #0d47a1;
                    padding: 12px 36px;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    margin-right: 6px;
                    font-size: 17px;
                    font-weight: bold;
                    min-width: 140px;
                    max-width: 320px;
                }
                QTabBar::tab:selected {
                    background: #42a5f5;
                    color: #fff;
                    font-weight: bold;
                }
                QLineEdit, QTextEdit, QComboBox, QListWidget {
                    background: #bbdefb;
                    color: #0d47a1;
                    border: 1.5px solid #1976D2;
                    border-radius: 8px;
                    padding: 6px;
                }
                QTableWidget {
                    background: #bbdefb;
                    color: #0d47a1;
                    gridline-color: #1976D2;
                    border-radius: 8px;
                    alternate-background-color: #e3f2fd;
                }
                QHeaderView::section {
                    background-color: #42a5f5;
                    color: #0d47a1;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton {
                    background-color: #42a5f5;
                    color: #0d47a1;
                    border-radius: 8px;
                    padding: 10px 24px;
                    font-weight: bold;
                    font-size: 15px;
                    min-width: 100px;
                    max-width: 1000px;
                    text-align: center;
                }
                QPushButton:disabled {
                    background-color: #B0BEC5;
                    color: #ECEFF1;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                    color: #fff;
                }
                QLabel {
                    color: #0d47a1;
                }
                QGroupBox {
                    background-color: #e3f2fd;
                    color: #0d47a1;
                    border-radius: 12px;
                    border: 1.5px solid #1976D2;
                    padding: 12px;
                }
            """)
            if hasattr(self, 'vis_timeframe_label'):
                self.vis_timeframe_label.setStyleSheet("font-size: 13px; color: #0d47a1; margin-top: 8px;")
        elif theme == "green":
            self.setStyleSheet("""
                QMainWindow, QDialog {
                    background-color: #e8f5e9;
                    color: #1b5e20;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    font-size: 13pt;
                }
                QTabWidget::pane {
                    border: 1.5px solid #388e3c;
                    background: #e8f5e9;
                    border-radius: 12px;
                }
                QTabBar::tab {
                    background: #a5d6a7;
                    color: #1b5e20;
                    padding: 12px 36px;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    margin-right: 6px;
                    font-size: 17px;
                    font-weight: bold;
                    min-width: 140px;
                    max-width: 320px;
                }
                QTabBar::tab:selected {
                    background: #66bb6a;
                    color: #fff;
                    font-weight: bold;
                }
                QLineEdit, QTextEdit, QComboBox, QListWidget {
                    background: #c8e6c9;
                    color: #1b5e20;
                    border: 1.5px solid #388e3c;
                    border-radius: 8px;
                    padding: 6px;
                }
                QTableWidget {
                    background: #c8e6c9;
                    color: #1b5e20;
                    gridline-color: #388e3c;
                    border-radius: 8px;
                    alternate-background-color: #e8f5e9;
                }
                QHeaderView::section {
                    background-color: #66bb6a;
                    color: #1b5e20;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton {
                    background-color: #66bb6a;
                    color: #1b5e20;
                    border-radius: 8px;
                    padding: 10px 24px;
                    font-weight: bold;
                    font-size: 15px;
                    min-width: 100px;
                    max-width: 1000px;
                    text-align: center;
                }
                QPushButton:disabled {
                    background-color: #B0BEC5;
                    color: #ECEFF1;
                }
                QPushButton:hover {
                    background-color: #388e3c;
                    color: #fff;
                }
                QLabel {
                    color: #1b5e20;
                }
                QGroupBox {
                    background-color: #e8f5e9;
                    color: #1b5e20;
                    border-radius: 12px;
                    border: 1.5px solid #388e3c;
                    padding: 12px;
                }
            """)
            if hasattr(self, 'vis_timeframe_label'):
                self.vis_timeframe_label.setStyleSheet("font-size: 13px; color: #388E3C; margin-top: 8px;")
        else:
            # fallback to dark
            self.apply_theme("dark")

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

    def create_indicators_group(self):
        # Reuse the indicators group creation logic from previous steps
        indicators_group = QGroupBox("Indicators")
        indicators_group.setStyleSheet("QGroupBox { border-radius: 8px; border: 1px solid #aaa; padding: 8px; }")
        indicators_layout = QFormLayout()
        indicators_layout.setSpacing(8)
        self.indicator_checkboxes = {}
        self.indicator_params = {}
        indicator_defs = [
            ("RSI", "Relative Strength Index", 14),
            ("MACD", "Moving Average Convergence Divergence", (12, 26, 9)),
            ("Bollinger Bands", "Bollinger Bands", 20),
            ("Stochastic", "Stochastic Oscillator", (14, 3)),
            ("EMA", "Exponential Moving Average", 20),
            ("VWAP", "Volume Weighted Average Price", None),
            ("ADX", "Average Directional Index", 14),
            ("CCI", "Commodity Channel Index", 20),
            ("SMA", "Simple Moving Average", 20),
            ("ATR", "Average True Range", 14),
            ("AO", "Awesome Oscillator", None),
        ]
        for name, desc, default in indicator_defs:
            cb = QCheckBox(name)
            cb.setToolTip(f"{desc}")
            self.indicator_checkboxes[name] = cb
            if name == "RSI":
                spin = QSpinBox()
                spin.setRange(2, 100)
                spin.setValue(default)
                spin.setToolTip("RSI period length")
                self.indicator_params[name] = spin
                indicators_layout.addRow(cb, spin)
            elif name == "MACD":
                fast = QSpinBox(); fast.setRange(2, 50); fast.setValue(default[0]); fast.setToolTip("MACD fast period")
                slow = QSpinBox(); slow.setRange(2, 100); slow.setValue(default[1]); slow.setToolTip("MACD slow period")
                signal = QSpinBox(); signal.setRange(1, 50); signal.setValue(default[2]); signal.setToolTip("MACD signal period")
                macd_layout = QHBoxLayout(); macd_layout.addWidget(fast); macd_layout.addWidget(slow); macd_layout.addWidget(signal)
                macd_widget = QWidget(); macd_widget.setLayout(macd_layout)
                self.indicator_params[name] = (fast, slow, signal)
                indicators_layout.addRow(cb, macd_widget)
            elif name == "Bollinger Bands":
                spin = QSpinBox(); spin.setRange(2, 100); spin.setValue(default); spin.setToolTip("Bollinger Bands window")
                self.indicator_params[name] = spin
                indicators_layout.addRow(cb, spin)
            elif name == "Stochastic":
                k = QSpinBox(); k.setRange(2, 50); k.setValue(default[0]); k.setToolTip("Stochastic %K period")
                d = QSpinBox(); d.setRange(1, 50); d.setValue(default[1]); d.setToolTip("Stochastic %D period")
                stoch_layout = QHBoxLayout(); stoch_layout.addWidget(k); stoch_layout.addWidget(d)
                stoch_widget = QWidget(); stoch_widget.setLayout(stoch_layout)
                self.indicator_params[name] = (k, d)
                indicators_layout.addRow(cb, stoch_widget)
            elif name in ("EMA", "ADX", "CCI", "SMA", "ATR"):
                spin = QSpinBox(); spin.setRange(2, 100); spin.setValue(default); spin.setToolTip(f"{name} period length")
                self.indicator_params[name] = spin
                indicators_layout.addRow(cb, spin)
            else:
                self.indicator_params[name] = None
                indicators_layout.addRow(cb)
        indicators_group.setLayout(indicators_layout)
        return indicators_group

    def create_processing_layout(self):
        process_layout = QVBoxLayout()
        table_layout = QHBoxLayout()
        table_label = QLabel("Select Tables:")
        table_label.setToolTip("Tables available in the source database. Hold Ctrl or Shift to select multiple.")
        table_layout.addWidget(table_label)
        self.table_list = QListWidget()
        self.table_list.setSelectionMode(QListWidget.MultiSelection)
        self.table_list.setToolTip("Select one or more tables to process (Ctrl/Shift for multi-select)")
        self.table_list.setEnabled(False)
        table_layout.addWidget(self.table_list)
        process_layout.addLayout(table_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setStyleSheet("QProgressBar { border-radius: 8px; height: 24px; }")
        process_layout.addWidget(self.progress_bar)
        self.process_btn = QPushButton("Process Selected Tables")
        self.process_btn.setToolTip("Calculate indicators for selected tables and save to target database")
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumWidth(220)
        self.process_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.process_btn.clicked.connect(self.process_tables)
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setToolTip("Cancel current processing operation")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumWidth(180)
        self.stop_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_btn.clicked.connect(self.stop_processing)
        process_layout.addWidget(self.process_btn)
        process_layout.addWidget(self.stop_btn)
        return process_layout

    def on_theme_changed(self, table_name):
        self.theme = table_name
        self.apply_theme(self.theme)
        # Optionally persist theme to a config file or QSettings

    def create_dashboard_card(self, title, value):
        card = QGroupBox()
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        card.setStyleSheet("""
            QGroupBox {
                background-color: rgba(58, 63, 75, 0.95);
                border-radius: 12px;
                border: 1.5px solid #1976D2;
                padding: 12px 18px 12px 18px;
                margin: 0px;
            }
        """)
        vbox = QVBoxLayout()
        vbox.setSpacing(2)
        label_title = QLabel(title)
        label_title.setStyleSheet("font-size: 13px; color: #90caf9; font-weight: 500;")
        label_value = QLabel(str(value))
        label_value.setObjectName("card_value")
        label_value.setStyleSheet("font-size: 28px; font-weight: bold; color: #fff; margin-top: 2px;")
        vbox.addWidget(label_title)
        vbox.addWidget(label_value)
        card.setLayout(vbox)
        card.value_label = label_value  # For easy updating
        return card

    def update_dashboard_cards(self):
        # Update number of tables found
        tables = getattr(self.data_reader, 'source_engine', None)
        table_count = 0
        if self.data_reader:
            try:
                table_count = len(self.data_reader.get_table_names())
            except Exception:
                table_count = 0
        self.card_tables_found.value_label.setText(str(table_count))
        # Update number of tables selected
        selected_tables = [item.text() for item in self.table_list.selectedItems()] if hasattr(self, 'table_list') else []
        self.card_tables_selected.value_label.setText(str(len(selected_tables)))
        # Update number of indicators selected
        selected_indicators = [name for name, cb in self.indicator_checkboxes.items() if cb.isChecked()] if hasattr(self, 'indicator_checkboxes') else []
        self.card_indicators_selected.value_label.setText(str(len(selected_indicators)))
        # Last status is updated elsewhere

    def on_vis_table_changed(self, table_name):
        # Update indicator list with columns from the selected table
        if not self.data_reader or not table_name:
            self.vis_indicator_list.clear()
            return
        try:
            df = self.data_reader.read_data(table_name, use_target=True)
            all_cols = list(df.columns)
            indicator_cols = [col for col in all_cols if col not in ("Open", "High", "Low", "Close", "Volume")]
            self.vis_indicator_list.clear()
            if not indicator_cols:
                self.vis_indicator_list.addItem(QListWidgetItem("[No indicators found]"))
                self.vis_indicator_list.setEnabled(False)
                self.log_message(f"[DEBUG] No indicator columns found in table '{table_name}'. Columns: {all_cols}")
            else:
                for col in indicator_cols:
                    item = QListWidgetItem(col)
                    self.vis_indicator_list.addItem(item)
                self.vis_indicator_list.setEnabled(True)
                self.log_message(f"[DEBUG] Indicator columns for '{table_name}': {indicator_cols}")
                # Default select RSI_weekly, AO_weekly, AO_monthly if present
                default_inds = {"RSI_weekly", "AO_weekly", "AO_monthly"}
                for i in range(self.vis_indicator_list.count()):
                    item = self.vis_indicator_list.item(i)
                    if item.text() in default_inds:
                        item.setSelected(True)
        except Exception as e:
            self.vis_indicator_list.clear()
            self.vis_indicator_list.addItem(QListWidgetItem("[Error loading indicators]"))
            self.vis_indicator_list.setEnabled(False)
            self.log_message(f"[DEBUG] Error loading indicators for '{table_name}': {e}")

    def on_plot_visualization(self):
        table = self.vis_table_combo.currentText()
        selected_items = self.vis_indicator_list.selectedItems()
        indicators = [item.text() for item in selected_items]
        if not self.data_reader or not table or not indicators:
            QMessageBox.warning(self, "Visualization", "Please select a table and at least one indicator.")
            return
        try:
            df = self.data_reader.read_data(table, use_target=True)
            if df.empty:
                QMessageBox.warning(self, "Visualization", f"No data for table '{table}'.")
                return
            self.vis_figure.clear()
            ax = self.vis_figure.add_subplot(111)
            # Color palette for indicators
            color_palette = [
                '#1976D2', '#43A047', '#FBC02D', '#E53935', '#8E24AA', '#00897B', '#F57C00', '#3949AB', '#C2185B', '#0097A7',
                '#7CB342', '#F06292', '#FFA000', '#5C6BC0', '#D81B60', '#388E3C', '#0288D1', '#F4511E', '#6D4C41', '#C0CA33'
            ]
            # AO color mapping for different timeframes
            ao_colors = {
                'AO_weekly': {'pos': '#43A047', 'neg': '#E53935'},  # green/red
                'AO_monthly': {'pos': '#1976D2', 'neg': '#FFA000'}, # blue/orange
            }
            # Plot price (Close) with a thick, distinct line
            ax.plot(df.index, df['Close'], label='Close', color='#90caf9', linewidth=2.5, linestyle='-', zorder=2)
            # Plot each selected indicator
            for i, indicator in enumerate(indicators):
                if indicator in df.columns:
                    if indicator in ao_colors:
                        ao_data = df[indicator]
                        above_zero = ao_data.where(ao_data >= 0)
                        below_zero = ao_data.where(ao_data < 0)
                        ax.plot(df.index, above_zero, label=f"{indicator} (+)", color=ao_colors[indicator]['pos'], linewidth=2.2, zorder=3)
                        ax.plot(df.index, below_zero, label=f"{indicator} (-)", color=ao_colors[indicator]['neg'], linewidth=2.2, zorder=3)
                    else:
                        ax.plot(df.index, df[indicator], label=indicator, linewidth=2.2, color=color_palette[i % len(color_palette)], zorder=3)
            # --- Buy/Sell Signal Markers ---
            # Buy Start: AO_weekly_AVG crosses from below to above zero
            if 'AO_weekly_AVG' in df.columns:
                ao_weekly_avg = df['AO_weekly_AVG']
                cross_up = (ao_weekly_avg.shift(1) < 0) & (ao_weekly_avg >= 0)
                buy_dates = df.index[cross_up]
                buy_prices = df['Close'][cross_up]
                ax.scatter(buy_dates, buy_prices, marker='^', color='#43A047', s=120, label='Buy Start (AO_weekly_AVG)', zorder=10, edgecolor='black')
                # Sell: AO_weekly_AVG crosses from above to below zero
                cross_down = (ao_weekly_avg.shift(1) > 0) & (ao_weekly_avg <= 0)
                sell_dates = df.index[cross_down]
                sell_prices = df['Close'][cross_down]
                ax.scatter(sell_dates, sell_prices, marker='v', color='#E53935', s=120, label='Sell (AO_weekly_AVG)', zorder=10, edgecolor='black')
            # Strong Buy: AO_monthly crosses from below to above zero
            if 'AO_monthly' in df.columns:
                ao_monthly = df['AO_monthly']
                cross_up = (ao_monthly.shift(1) < 0) & (ao_monthly >= 0)
                strong_buy_dates = df.index[cross_up]
                strong_buy_prices = df['Close'][cross_up]
                ax.scatter(strong_buy_dates, strong_buy_prices, marker='*', color='#1976D2', s=180, label='Strong Buy (AO_monthly)', zorder=11, edgecolor='black')
                # Strong Sell: AO_monthly crosses from above to below zero
                cross_down = (ao_monthly.shift(1) > 0) & (ao_monthly <= 0)
                strong_sell_dates = df.index[cross_down]
                strong_sell_prices = df['Close'][cross_down]
                ax.scatter(strong_sell_dates, strong_sell_prices, marker='*', color='#FFA000', s=180, label='Strong Sell (AO_monthly)', zorder=11, edgecolor='black')
            ax.set_title(f"Indicators for {table}", fontsize=15, fontweight='bold', color='#1976D2', pad=16)
            ax.set_facecolor('#23272e')
            ax.grid(True, color='#444', alpha=0.3, linestyle='--', linewidth=1)
            legend = ax.legend(frameon=True, facecolor='#23272e', edgecolor='#1976D2', fontsize=11, loc='best')
            for text in legend.get_texts():
                text.set_color('#f8f8f2')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color('#f8f8f2')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1976D2')
            ax.spines['bottom'].set_color('#1976D2')
            self.vis_figure.tight_layout()
            self.vis_canvas.draw()
            # Add interactive tooltips for all lines and markers
            mplcursors.cursor(ax.lines + ax.collections, hover=True).connect(
                "add", lambda sel: sel.annotation.set_text(
                    f"{df.index[int(sel.target.index)] if hasattr(sel.target, 'index') else ''}\nValue: {sel.target[1]:.2f}"))
            # Update chart time frame label
            if not df.empty:
                start_date = str(df.index.min())[:10]
                end_date = str(df.index.max())[:10]
                self.vis_timeframe_label.setText(f"Chart Time Frame: {start_date} to {end_date}")
            else:
                self.vis_timeframe_label.setText("")
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Failed to plot: {str(e)}")

    def on_clear_indicator_selection(self):
        self.vis_indicator_list.clearSelection()

    def on_export_tables(self):
        selected_items = self.export_table_list.selectedItems()
        tables = [item.text() for item in selected_items]
        if not tables:
            self.export_status_label.setText("<span style='color:red'>No tables selected for export.</span>")
            return
        # Choose export format
        export_format = 'csv' if self.radio_csv.isChecked() else 'excel'
        # Ask user for export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            self.export_status_label.setText("<span style='color:red'>Export cancelled.</span>")
            return
        # Export each table
        errors = []
        for table in tables:
            try:
                df = pd.read_sql_table(table, self.data_reader.target_engine, index_col='Date', parse_dates=['Date'])
                if export_format == 'csv':
                    out_path = os.path.join(export_dir, f"{table}.csv")
                    df.to_csv(out_path)
                else:
                    out_path = os.path.join(export_dir, f"{table}.xlsx")
                    df.to_excel(out_path)
            except Exception as e:
                errors.append(f"{table}: {str(e)}")
        if errors:
            self.export_status_label.setText(f"<span style='color:red'>Some tables failed to export:<br>{'<br>'.join(errors)}</span>")
        else:
            self.export_status_label.setText(f"<span style='color:green'>Exported {len(tables)} table(s) to {export_dir}</span>")

    def create_dashboard_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(18)
        layout.setContentsMargins(30, 30, 30, 30)
        # --- Top Controls: Refresh and Theme Switcher ---
        top_controls = QHBoxLayout()
        # Quick Theme Switcher
        self.dashboard_theme_combo = QComboBox()
        self.dashboard_theme_combo.addItems(["dark", "light", "blue", "green"])
        self.dashboard_theme_combo.setCurrentText(self.theme)
        self.dashboard_theme_combo.setToolTip("Quickly change the app theme")
        self.dashboard_theme_combo.currentTextChanged.connect(self.on_dashboard_theme_changed)
        top_controls.addWidget(QLabel("Theme:"))
        top_controls.addWidget(self.dashboard_theme_combo)
        # Refresh Button
        self.dashboard_refresh_btn = QPushButton("Refresh")
        self.dashboard_refresh_btn.setToolTip("Refresh dashboard stats and logs")
        self.dashboard_refresh_btn.setMinimumWidth(100)
        self.dashboard_refresh_btn.clicked.connect(self.on_dashboard_refresh)
        top_controls.addWidget(self.dashboard_refresh_btn)
        top_controls.addStretch(1)
        layout.addLayout(top_controls)
        # --- Summary Cards ---
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(24)
        self.card_tables_found = self.create_dashboard_card("Tables Found", "0")
        self.card_tables_selected = self.create_dashboard_card("Tables Selected", "0")
        self.card_indicators_selected = self.create_dashboard_card("Indicators Selected", "0")
        self.card_last_status = self.create_dashboard_card("Last Status", "-")
        for card in [self.card_tables_found, self.card_tables_selected, self.card_indicators_selected, self.card_last_status]:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cards_layout.addWidget(card)
        layout.addLayout(cards_layout)
        # --- Recent Activity Table ---
        self.dashboard_activity_table = QTableWidget(0, 3)
        self.dashboard_activity_table.setHorizontalHeaderLabels(["Time", "Action", "Details"])
        self.dashboard_activity_table.horizontalHeader().setStretchLastSection(True)
        self.dashboard_activity_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.dashboard_activity_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.dashboard_activity_table.setMaximumHeight(140)
        self.dashboard_activity_table.setStyleSheet("font-size: 13px;")
        layout.addWidget(QLabel("Recent Activity:"))
        layout.addWidget(self.dashboard_activity_table)
        # --- Recent logs ---
        log_group = QGroupBox("Recent Activity Log")
        log_group.setStyleSheet("QGroupBox { border-radius: 8px; border: 1px solid #aaa; padding: 8px; }")
        log_layout = QVBoxLayout(log_group)
        self.dashboard_log_display = QTextEdit()
        self.dashboard_log_display.setReadOnly(True)
        self.dashboard_log_display.setMaximumHeight(120)
        self.dashboard_log_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        log_layout.addWidget(self.dashboard_log_display)
        layout.addWidget(log_group)
        # Quick actions
        quick_actions = QHBoxLayout()
        btn_process = QPushButton("Process Data")
        btn_process.setToolTip("Go to Process Data tab")
        btn_process.setMinimumWidth(140)
        btn_process.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_process.clicked.connect(lambda: self.tabs.setCurrentWidget(self.process_tab))
        btn_visualize = QPushButton("Visualize")
        btn_visualize.setToolTip("Go to Visualize tab")
        btn_visualize.setMinimumWidth(120)
        btn_visualize.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_visualize.clicked.connect(lambda: self.tabs.setCurrentWidget(self.visualize_tab))
        btn_export = QPushButton("Export")
        btn_export.setToolTip("Go to Export tab")
        btn_export.setMinimumWidth(120)
        btn_export.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_export.clicked.connect(lambda: self.tabs.setCurrentWidget(self.export_tab))
        for btn in [btn_process, btn_visualize, btn_export]:
            quick_actions.addWidget(btn)
        quick_actions.addStretch(1)
        layout.addLayout(quick_actions)
        layout.addStretch(1)
        tab.setLayout(layout)
        return tab

    def on_dashboard_theme_changed(self, theme):
        self.theme = theme
        self.apply_theme(theme)
        if hasattr(self, 'theme_combo'):
            self.theme_combo.setCurrentText(theme)

    def on_dashboard_refresh(self):
        self.update_dashboard_cards()
        # Optionally refresh logs/activity table if needed

    def add_dashboard_activity(self, action, details):
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.dashboard_activity_table.insertRow(0)
        self.dashboard_activity_table.setItem(0, 0, QTableWidgetItem(now))
        self.dashboard_activity_table.setItem(0, 1, QTableWidgetItem(action))
        self.dashboard_activity_table.setItem(0, 2, QTableWidgetItem(details))
        # Keep only last 5
        while self.dashboard_activity_table.rowCount() > 5:
            self.dashboard_activity_table.removeRow(5)

    def create_process_tab(self):
        process_tab = QWidget()
        process_layout = QVBoxLayout(process_tab)
        process_layout.setSpacing(16)
        process_layout.setContentsMargins(20, 20, 20, 20)
        # Database Connection Group
        db_group = QGroupBox("Database Connection")
        db_group.setStyleSheet("QGroupBox { border-radius: 8px; border: 1px solid #aaa; padding: 8px; }")
        db_layout = QVBoxLayout(db_group)
        db_layout.setSpacing(8)
        # Source DB
        source_layout = QHBoxLayout()
        source_label = QLabel("Source DB:")
        source_label.setToolTip("Path to the source SQLite database containing raw stock data")
        source_layout.addWidget(source_label)
        self.source_db_input = QLineEdit()
        self.source_db_input.setPlaceholderText("Path to source database")
        self.source_db_input.setToolTip("Enter path to source database or click Browse to select")
        self.source_db_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_layout.addWidget(self.source_db_input)
        self.source_browse_btn = QPushButton("Browse...")
        self.source_browse_btn.setToolTip("Browse for source database file")
        self.source_browse_btn.setMinimumWidth(100)
        self.source_browse_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self.target_db_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        target_layout.addWidget(self.target_db_input)
        self.target_browse_btn = QPushButton("Browse...")
        self.target_browse_btn.setToolTip("Browse for target database location")
        self.target_browse_btn.setMinimumWidth(100)
        self.target_browse_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.target_browse_btn.clicked.connect(self.browse_target_db)
        target_layout.addWidget(self.target_browse_btn)
        db_layout.addLayout(target_layout)
        # Connect Button
        self.connect_btn = QPushButton("Connect to Databases")
        self.connect_btn.setToolTip("Establish connection to both source and target databases")
        self.connect_btn.setMinimumWidth(200)
        self.connect_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.connect_btn.clicked.connect(self.connect_databases)
        db_layout.addWidget(self.connect_btn)
        db_group.setLayout(db_layout)
        process_layout.addWidget(db_group)
        # Indicators Group
        indicators_group = self.create_indicators_group()
        indicators_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        process_layout.addWidget(indicators_group)
        # Processing Group
        process_group = QGroupBox("Processing")
        process_group.setStyleSheet("QGroupBox { border-radius: 8px; border: 1px solid #aaa; padding: 8px; }")
        process_group.setLayout(self.create_processing_layout())
        process_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        process_layout.addWidget(process_group)
        # Log Display
        log_group = QGroupBox("Log")
        log_group.setStyleSheet("QGroupBox { border-radius: 8px; border: 1px solid #aaa; padding: 8px; }")
        log_layout = QVBoxLayout(log_group)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_display)
        process_layout.addWidget(log_group, stretch=1)
        process_tab.setLayout(process_layout)
        return process_tab

    def create_visualize_tab(self):
        visualize_tab = QWidget()
        vis_layout = QVBoxLayout(visualize_tab)
        vis_layout.setSpacing(16)
        vis_layout.setContentsMargins(20, 20, 20, 20)
        vis_controls_layout = QHBoxLayout()
        vis_controls_layout.setSpacing(12)
        vis_controls_layout.addWidget(QLabel("Table:"))
        self.vis_table_combo = QComboBox()
        self.vis_table_combo.setMinimumWidth(200)
        self.vis_table_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.vis_table_combo.setEditable(True)  # Allow user to type/filter
        vis_controls_layout.addWidget(self.vis_table_combo)
        vis_controls_layout.addWidget(QLabel("Indicator(s):"))
        self.vis_indicator_list = QListWidget()
        self.vis_indicator_list.setSelectionMode(QListWidget.MultiSelection)
        self.vis_indicator_list.setMinimumWidth(220)
        self.vis_indicator_list.setMaximumHeight(120)
        self.vis_indicator_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vis_controls_layout.addWidget(self.vis_indicator_list)
        self.vis_plot_btn = QPushButton("Plot")
        self.vis_plot_btn.setToolTip("Plot selected indicators for the chosen table")
        self.vis_plot_btn.setMinimumWidth(100)
        self.vis_plot_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.vis_plot_btn.clicked.connect(self.on_plot_visualization)
        vis_controls_layout.addWidget(self.vis_plot_btn)
        # Add Clear Selection button
        self.vis_clear_btn = QPushButton("Clear Selection")
        self.vis_clear_btn.setToolTip("Clear all selected indicators")
        self.vis_clear_btn.setMinimumWidth(120)
        self.vis_clear_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.vis_clear_btn.clicked.connect(self.on_clear_indicator_selection)
        vis_controls_layout.addWidget(self.vis_clear_btn)
        # Add Export Chart button
        self.vis_export_btn = QPushButton("Export Chart")
        self.vis_export_btn.setToolTip("Export the current chart as a PNG file")
        self.vis_export_btn.setMinimumWidth(120)
        self.vis_export_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.vis_export_btn.clicked.connect(self.on_export_chart)
        vis_controls_layout.addWidget(self.vis_export_btn)
        vis_layout.addLayout(vis_controls_layout)
        self.vis_figure = plt.Figure(figsize=(8, 4))
        self.vis_canvas = FigureCanvas(self.vis_figure)
        self.vis_canvas.setStyleSheet("background-color: #23272e; border-radius: 8px;")
        self.vis_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vis_layout.addWidget(self.vis_canvas, stretch=1)
        # Add chart time frame label
        self.vis_timeframe_label = QLabel("")
        self.vis_timeframe_label.setAlignment(Qt.AlignCenter)
        self.vis_timeframe_label.setStyleSheet("font-size: 13px; color: #90caf9; margin-top: 8px;")
        vis_layout.addWidget(self.vis_timeframe_label)
        visualize_tab.setLayout(vis_layout)
        return visualize_tab

    def on_export_chart(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Chart As", "chart.png", "PNG Files (*.png)")
        if file_path:
            self.vis_figure.savefig(file_path, format='png', bbox_inches='tight')
            QMessageBox.information(self, "Export Chart", f"Chart saved as: {file_path}")

    def create_export_tab(self):
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        export_layout.setSpacing(16)
        export_layout.setContentsMargins(20, 20, 20, 20)
        export_layout.addWidget(QLabel("Select tables to export:"))
        self.export_table_list = QListWidget()
        self.export_table_list.setSelectionMode(QListWidget.MultiSelection)
        self.export_table_list.setMinimumHeight(120)
        self.export_table_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        export_layout.addWidget(self.export_table_list)
        format_group = QGroupBox("Export Format")
        format_layout = QHBoxLayout(format_group)
        self.radio_csv = QRadioButton("CSV")
        self.radio_excel = QRadioButton("Excel (.xlsx)")
        self.radio_csv.setChecked(True)
        format_layout.addWidget(self.radio_csv)
        format_layout.addWidget(self.radio_excel)
        format_group.setLayout(format_layout)
        export_layout.addWidget(format_group)
        self.export_btn = QPushButton("Export Selected Tables")
        self.export_btn.setToolTip("Export selected tables to CSV or Excel format")
        self.export_btn.setMinimumWidth(220)
        self.export_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.export_btn.clicked.connect(self.on_export_tables)
        export_layout.addWidget(self.export_btn)
        self.export_status_label = QLabel("")
        export_layout.addWidget(self.export_status_label)
        export_tab.setLayout(export_layout)
        return export_tab

    def create_settings_tab(self):
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        settings_label = QLabel("Settings")
        settings_label.setAlignment(Qt.AlignLeft)
        settings_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 12px;")
        settings_layout.addWidget(settings_label)
        theme_group = QGroupBox("Theme")
        theme_layout = QHBoxLayout()
        theme_group.setLayout(theme_layout)
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light", "blue", "green"])
        self.theme_combo.setCurrentText(self.theme)
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        theme_layout.addWidget(self.theme_combo)
        settings_layout.addWidget(theme_group)
        # Add Save button
        self.save_theme_btn = QPushButton("Save")
        self.save_theme_btn.setToolTip("Save the selected theme as default")
        self.save_theme_btn.setMinimumWidth(100)
        self.save_theme_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.save_theme_btn.clicked.connect(self.on_save_theme)
        settings_layout.addWidget(self.save_theme_btn)
        settings_tab.setLayout(settings_layout)
        return settings_tab

    def on_save_theme(self):
        from PyQt5.QtCore import QSettings
        settings = QSettings("PSXIndicator", "GUI")
        settings.setValue("theme", self.theme)
        QMessageBox.information(self, "Settings", f"Theme '{self.theme}' saved as default.")

    def load_theme_from_settings(self):
        from PyQt5.QtCore import QSettings
        settings = QSettings("PSXIndicator", "GUI")
        theme = settings.value("theme", "dark")
        self.theme = theme
        if hasattr(self, 'theme_combo'):
            self.theme_combo.setCurrentText(theme)
        self.apply_theme(theme)

    def create_help_tab(self):
        help_tab = QWidget()
        help_layout = QVBoxLayout()
        help_label = QLabel("[Help and documentation coming soon]")
        help_label.setAlignment(Qt.AlignCenter)
        help_layout.addWidget(help_label)
        help_tab.setLayout(help_layout)
        return help_tab

    def auto_connect_and_populate_dashboard(self):
        """Automatically connect to default databases and populate dashboard on load."""
        try:
            self.connect_databases()
        except Exception as e:
            self.log_message(f"Auto-connect failed: {e}")

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

    def read_data(self, table_name: str, use_target: bool = False) -> pd.DataFrame:
        """Read stock market data from the specified table.
        
        Args:
            table_name: Name of the table to read data from
            use_target: If True, read from the target (processed) database; else from source
        Returns:
            pd.DataFrame: DataFrame containing the stock market data, or empty DataFrame on error
        """
        try:
            engine = self.target_engine if use_target else self.source_engine
            data = pd.read_sql_table(table_name, engine, index_col='Date', parse_dates=['Date'])
            logging.info(f"Data read successfully from table {table_name} ({'target' if use_target else 'source'})")
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

    def preprocess(self, data: pd.DataFrame, selected_indicators: dict) -> pd.DataFrame:
        """
        Preprocess stock market data by calculating all standard indicators (for compatibility),
        and add new advanced indicators only if selected by the user.
        Args:
            data: DataFrame containing raw stock market data with columns: Date (index), Open, High, Low, Close, Volume
            selected_indicators: dict of {indicator_name: parameter(s)} as selected by the user
        Returns:
            pd.DataFrame: Processed DataFrame with all standard and any requested advanced indicators
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

        # --- Always calculate all existing indicators for compatibility ---
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
            logging.error(f"Error in preprocessing data (standard indicators): {e}")

        # --- Add new advanced indicators only if selected ---
        try:
            for name, param in selected_indicators.items():
                # Only add if not already part of the standard indicators
                try:
                    if name == "MACD":
                        fast, slow, signal = param
                        macd = ta.macd(data['Close'], fast=fast, slow=slow, signal=signal)
                        if macd is not None:
                            for col in macd.columns:
                                data[f'MACD_{col}_{fast}_{slow}_{signal}'] = macd[col]
                    elif name == "Bollinger Bands":
                        bb = ta.bbands(data['Close'], length=param)
                        if bb is not None:
                            for col in bb.columns:
                                data[f'BB_{col}_{param}'] = bb[col]
                    elif name == "Stochastic":
                        k, d = param
                        stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=k, d=d)
                        if stoch is not None:
                            for col in stoch.columns:
                                data[f'STOCH_{col}_{k}_{d}'] = stoch[col]
                    elif name == "EMA":
                        data[f'EMA_{param}'] = ta.ema(data['Close'], length=param)
                    elif name == "VWAP":
                        vwap = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
                        if vwap is not None:
                            data['VWAP'] = vwap
                    elif name == "ADX":
                        adx = ta.adx(data['High'], data['Low'], data['Close'], length=param)
                        if adx is not None:
                            for col in adx.columns:
                                data[f'ADX_{col}_{param}'] = adx[col]
                    elif name == "CCI":
                        data[f'CCI_{param}'] = ta.cci(data['High'], data['Low'], data['Close'], length=param)
                    # Do not add RSI, SMA, ATR, AO here (they are always calculated above)
                except Exception as ind_e:
                    logging.error(f"Error calculating {name}: {ind_e}")
        except Exception as e:
            logging.error(f"Error in preprocessing data (advanced indicators): {e}")

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