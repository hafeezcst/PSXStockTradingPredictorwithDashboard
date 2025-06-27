import sys
import os
import logging
import sqlite3
import requests
import time
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QProgressBar, QTabWidget, 
    QGroupBox, QLineEdit, QComboBox, QFileDialog, QMessageBox,
    QScrollArea, QTableWidget, QTableWidgetItem, QSplitter,
    QFrame, QCheckBox, QSpinBox, QDoubleSpinBox, QTextBrowser,
    QHeaderView, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QBrush, QColor
import importlib.util
from threading import Event

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Optional: QDarkStyle for modern dark theme
try:
    # import qdarkstyle  # Remove or comment out this line
    DARK_THEME = True
except ImportError:
    DARK_THEME = False

# Dynamically import the main script with dashes in filename
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '04-List_Weekly_RSI_GT_40_BUY_SELL_KMI30_100_Weekly_v1.0_stable_Development.py')
SCRIPT_MODULE_NAME = 'analysis_module_temp'
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE_NAME, SCRIPT_PATH)
analysis_module = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE_NAME] = analysis_module
spec.loader.exec_module(analysis_module)

class AnalysisWorker(QThread):
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path

    def run(self):
        def log_callback(msg):
            self.log_signal.emit(msg)
        def result_callback(result):
            self.result_signal.emit(result)
        results = analysis_module.run_full_analysis(self.db_path, log_callback, result_callback)
        self.finished_signal.emit(results)

class EmailSender(QThread):
    finished_signal = pyqtSignal(bool, str)
    log_signal = pyqtSignal(str)

    def __init__(self, smtp_server, smtp_port, sender_email, sender_password, recipient_email, subject, message):
        super().__init__()
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.subject = subject
        self.message = message

    def run(self):
        try:
            self.log_signal.emit("Connecting to SMTP server...")
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = self.subject
            msg.attach(MIMEText(self.message, 'plain'))
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=15)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            self.log_signal.emit("Sending email...")
            server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
            server.quit()
            self.finished_signal.emit(True, "Email sent successfully!")
        except Exception as e:
            self.finished_signal.emit(False, f"Email send failed: {str(e)}")

class TelegramSender(QThread):
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, int)  # current, total
    log_signal = pyqtSignal(str)

    def __init__(self, token, chat_id, message, cancel_event=None, max_retries=5):
        super().__init__()
        self.token = token
        self.chat_id = chat_id
        self.message = message
        self.cancel_event = cancel_event or Event()
        self.max_retries = max_retries

    def run(self):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        max_length = 4096
        messages = [m.strip() for m in self.message.split('\n\n') if m.strip()]
        total_chunks = sum(len([msg[i:i+max_length] for i in range(0, len(msg), max_length)]) for msg in messages)
        chunk_counter = 0
        all_success = True
        error_msgs = []
        for idx, msg in enumerate(messages):
            if self.cancel_event.is_set():
                self.log_signal.emit(f"[Message {idx+1}] Sending cancelled by user.")
                self.finished_signal.emit(False, "Sending cancelled by user.")
                return
            chunks = [msg[i:i+max_length] for i in range(0, len(msg), max_length)]
            for chunk_idx, chunk in enumerate(chunks):
                if self.cancel_event.is_set():
                    self.log_signal.emit(f"[Message {idx+1} - Chunk {chunk_idx+1}] Sending cancelled by user.")
                    self.finished_signal.emit(False, "Sending cancelled by user.")
                    return
                payload = {
                    'chat_id': self.chat_id,
                    'text': chunk,
                    'parse_mode': 'MarkdownV2',
                    'disable_web_page_preview': True
                }
                retries = 0
                while retries < self.max_retries:
                    try:
                        self.log_signal.emit(f"[Message {idx+1} - Chunk {chunk_idx+1}] Sending (attempt {retries+1})...")
                        response = requests.post(url, json=payload, timeout=15)
                        if response.status_code == 200:
                            self.log_signal.emit(f"[Message {idx+1} - Chunk {chunk_idx+1}] Success.")
                            break
                        elif response.status_code == 429:
                            retry_after = int(response.headers.get('Retry-After', '5'))
                            self.log_signal.emit(f"[Message {idx+1} - Chunk {chunk_idx+1}] Rate limit hit. Waiting {retry_after}s before retry...")
                            time.sleep(retry_after + 1)
                            retries += 1
                        else:
                            all_success = False
                            err = f"[Message {idx+1} - Chunk {chunk_idx+1}] Failed: {response.text}"
                            self.log_signal.emit(err)
                            error_msgs.append(err)
                            break
                    except Exception as e:
                        all_success = False
                        err = f"[Message {idx+1} - Chunk {chunk_idx+1}] Error: {str(e)}"
                        self.log_signal.emit(err)
                        error_msgs.append(err)
                        break
                chunk_counter += 1
                self.progress_signal.emit(chunk_counter, total_chunks)
        if all_success:
            self.finished_signal.emit(True, "All messages sent successfully!")
        else:
            self.finished_signal.emit(False, "Some messages failed to send:\n" + '\n'.join(error_msgs))

class AnalysisTab(QWidget):
    analysis_requested = pyqtSignal(str)
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create scroll area for the entire tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Summary Card ---
        self.summary_group = QGroupBox()
        self.summary_group.setTitle(" Summary")
        self.summary_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #2196F3;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 12px;
                background-color: #fafbfc;
            }
            QGroupBox::title {
                color: #2196F3;
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
            }
        """)
        summary_layout = QHBoxLayout()
        self.total_signals_label = QLabel("Total Signals: 0")
        self.total_signals_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.last_update_label = QLabel("Last Analysis: -")
        self.last_update_label.setFont(QFont("Segoe UI", 11))
        summary_layout.addWidget(self.total_signals_label)
        summary_layout.addSpacing(30)
        summary_layout.addWidget(self.last_update_label)
        summary_layout.addStretch()
        self.summary_group.setLayout(summary_layout)
        layout.addWidget(self.summary_group)

        # --- Database Selection Group ---
        db_group = QGroupBox("Database Configuration")
        db_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        db_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #90caf9;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f5faff;
            }
            QGroupBox::title {
                color: #1976D2;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        db_layout = QVBoxLayout()
        db_input_layout = QHBoxLayout()
        db_input_layout.addWidget(QLabel("Database Path:"))
        self.db_path = QLineEdit()
        self.db_path.setText("data/databases/production/psx_consolidated_data_indicators_PSX.db")
        self.db_path.setPlaceholderText("Select database file...")
        self.db_path.setMinimumHeight(32)
        db_input_layout.addWidget(self.db_path, 1)
        self.db_browse = QPushButton("📂 Browse...")
        self.db_browse.setMinimumHeight(32)
        self.db_browse.setMinimumWidth(100)
        self.db_browse.clicked.connect(self.browse_db)
        db_input_layout.addWidget(self.db_browse)
        db_layout.addLayout(db_input_layout)
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        # --- Analysis Controls Group ---
        controls_group = QGroupBox("Analysis Controls")
        controls_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        controls_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4caf50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f7fff7;
            }
            QGroupBox::title {
                color: #388e3c;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        controls_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("🚀 Run Analysis")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.run_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.run_btn.clicked.connect(self.run_analysis)
        btn_layout.addWidget(self.run_btn)
        self.export_btn = QPushButton("📊 Export to Excel")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.export_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.export_btn.clicked.connect(self.export_to_excel)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        controls_layout.addLayout(btn_layout)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setMinimumHeight(24)
        self.progress.setStyleSheet("border-radius: 6px;")
        controls_layout.addWidget(self.progress)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # --- Results Group ---
        results_group = QGroupBox("Analysis Results")
        results_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        results_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ba68c8;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #faf5ff;
            }
            QGroupBox::title {
                color: #8e24aa;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        results_layout = QVBoxLayout()
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setMaximumHeight(320)
        self.result_box.setFont(QFont("Consolas", 11))
        self.result_box.setStyleSheet("background: #f3e5f5; border-radius: 8px; padding: 10px; color: #4a148c;")
        results_layout.addWidget(self.result_box)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)
        # Set the scroll area as the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self.latest_results = None
        self.latest_dfs = {}
        self.worker = None
        self.parent_window = parent

    def browse_db(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Database", "", "Database Files (*.db)")
        if file:
            self.db_path.setText(file)

    def run_analysis(self):
        db_path = self.db_path.text().strip()
        if not db_path:
            QMessageBox.warning(self, "No Database Selected", "Please select a database file.")
            return
        self.result_box.clear()
        self.progress.setValue(0)
        self.export_btn.setEnabled(False)
        self.latest_results = None
        self.latest_dfs = {}
        # Start threaded analysis
        self.worker = AnalysisWorker(db_path)
        self.worker.log_signal.connect(self.append_log)
        self.worker.result_signal.connect(self.handle_result)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()
        if self.parent_window:
            self.parent_window.status.showMessage("Running analysis...")
        # Update summary card
        self.last_update_label.setText(f"Last Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def append_log(self, msg):
        if self.parent_window:
            self.parent_window.logs_tab.append_log(msg)

    def handle_result(self, result):
        # Show the latest message in the result box
        if 'message' in result:
            self.result_box.setPlainText(result['message'])
        # Store DataFrames for export
        if 'type' in result and 'df' in result:
            self.latest_dfs[result['type']] = result['df']
        # Update summary card with total signals if possible
        if 'df' in result and hasattr(result['df'], 'shape'):
            total = result['df'].shape[0]
            self.total_signals_label.setText(f"Total Signals: {total}")

    def analysis_finished(self, results):
        self.export_btn.setEnabled(True)
        self.latest_results = results
        if self.parent_window:
            self.parent_window.status.showMessage("Analysis complete.")
        # Show the first available message in the result box
        for key in ['buy', 'sell', 'neutral', 'breakout']:
            if results.get(key) is not None:
                self.result_box.setPlainText(str(results[key]))
                break
        # Prepare Telegram message(s) for preview
        telegram_messages = []
        # Collect all formatted messages from result_callback (as in main script)
        for result in results.get('log', []):
            if result.startswith('🟢') or result.startswith('🔴') or result.startswith('🟡') or result.startswith('🚀'):
                telegram_messages.append(result)
        if self.parent_window:
            self.parent_window.telegram_tab.set_message('\n\n'.join(telegram_messages))

    def export_to_excel(self):
        if not self.latest_dfs:
            QMessageBox.warning(self, "No Results", "No analysis results to export.")
            return
        file, _ = QFileDialog.getSaveFileName(self, "Export to Excel", "results.xlsx", "Excel Files (*.xlsx)")
        if file:
            try:
                # Export all DataFrames to separate sheets
                with pd.ExcelWriter(file) as writer:
                    for key, df in self.latest_dfs.items():
                        df.to_excel(writer, sheet_name=key, index=False)
                QMessageBox.information(self, "Exported", f"Results exported to {file}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

class LogsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create scroll area for the entire tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Log Controls Group ---
        controls_group = QGroupBox("Log Controls")
        controls_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        controls_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ff9800;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fff8f0;
            }
            QGroupBox::title {
                color: #f57c00;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        controls_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑️ Clear Logs")
        self.clear_btn.setMinimumHeight(36)
        self.clear_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.clear_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.clear_btn.clicked.connect(self.clear_logs)
        controls_layout.addWidget(self.clear_btn)
        
        self.export_log_btn = QPushButton("📄 Export Logs")
        self.export_log_btn.setMinimumHeight(36)
        self.export_log_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.export_log_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.export_log_btn.clicked.connect(self.export_logs)
        controls_layout.addWidget(self.export_log_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # --- Log Display Group ---
        log_group = QGroupBox("Log Output")
        log_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        log_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #607d8b;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f5f7fa;
            }
            QGroupBox::title {
                color: #455a64;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        log_layout = QVBoxLayout()
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 10))
        self.log_box.setStyleSheet("""
            background: #263238;
            color: #eceff1;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #455a64;
        """)
        log_layout.addWidget(self.log_box)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)
        
        # Set the scroll area as the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def append_log(self, text):
        self.log_box.append(text)

    def clear_logs(self):
        self.log_box.clear()

    def export_logs(self):
        file, _ = QFileDialog.getSaveFileName(self, "Export Logs", "analysis_logs.txt", "Text Files (*.txt)")
        if file:
            try:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(self.log_box.toPlainText())
                QMessageBox.information(self, "Success", f"Logs exported to {file}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

class TelegramTab(QWidget):
    send_message_requested = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create scroll area for the entire tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Telegram Configuration Group ---
        telegram_group = QGroupBox("Telegram Configuration")
        telegram_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        telegram_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f0f8ff;
            }
            QGroupBox::title {
                color: #1976D2;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        telegram_layout = QVBoxLayout()
        
        # Bot credentials - more compact layout
        cred_layout = QHBoxLayout()
        cred_layout.addWidget(QLabel("Bot Token:"))
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Enter bot token...")
        self.token_input.setMinimumHeight(32)
        cred_layout.addWidget(self.token_input, 1)
        
        cred_layout.addWidget(QLabel("Chat ID:"))
        self.chatid_input = QLineEdit()
        self.chatid_input.setPlaceholderText("Enter chat ID...")
        self.chatid_input.setMinimumHeight(32)
        cred_layout.addWidget(self.chatid_input, 1)
        telegram_layout.addLayout(cred_layout)
        
        # Load Telegram credentials from .env if available
        if DOTENV_AVAILABLE:
            dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path)
                token = os.getenv('TELEGRAM_BOT_TOKEN')
                chat_id = os.getenv('TELEGRAM_CHAT_ID')
                if token:
                    self.token_input.setText(token)
                if chat_id:
                    self.chatid_input.setText(chat_id)
        
        telegram_group.setLayout(telegram_layout)
        layout.addWidget(telegram_group)

        # --- Message Preview Group ---
        preview_group = QGroupBox("Message Preview")
        preview_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        preview_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4caf50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f1f8e9;
            }
            QGroupBox::title {
                color: #388e3c;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        preview_layout = QVBoxLayout()
        
        self.message_preview = QTextEdit()
        self.message_preview.setReadOnly(True)
        self.message_preview.setMaximumHeight(200)
        self.message_preview.setFont(QFont("Segoe UI", 10))
        self.message_preview.setStyleSheet("""
            background: #e8f5e8;
            color: #2e7d32;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #81c784;
        """)
        preview_layout.addWidget(self.message_preview)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # --- Sending Controls Group ---
        sending_group = QGroupBox("Sending Controls")
        sending_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        sending_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ff9800;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fff8f0;
            }
            QGroupBox::title {
                color: #f57c00;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        sending_layout = QVBoxLayout()
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton("📤 Send Message")
        self.send_btn.setMinimumHeight(40)
        self.send_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.send_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.send_btn.clicked.connect(self.send_message)
        btn_layout.addWidget(self.send_btn)
        
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.cancel_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.cancel_btn.clicked.connect(self.cancel_sending)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        
        self.export_log_btn = QPushButton("📄 Export Log")
        self.export_log_btn.setMinimumHeight(40)
        self.export_log_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.export_log_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.export_log_btn.clicked.connect(self.export_log)
        btn_layout.addWidget(self.export_log_btn)
        
        btn_layout.addStretch()
        sending_layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setMinimumHeight(24)
        self.progress.setStyleSheet("border-radius: 6px;")
        sending_layout.addWidget(self.progress)
        sending_group.setLayout(sending_layout)
        layout.addWidget(sending_group)

        # --- Sending Log Group ---
        log_group = QGroupBox("Sending Log")
        log_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        log_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #607d8b;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f5f7fa;
            }
            QGroupBox::title {
                color: #455a64;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            background: #263238;
            color: #eceff1;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #455a64;
        """)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # --- Summary Table Group ---
        summary_group = QGroupBox("Message Summary")
        summary_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        summary_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #9c27b0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #faf5ff;
            }
            QGroupBox::title {
                color: #8e24aa;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        summary_layout = QVBoxLayout()
        
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Type", "Count", "Status"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setMaximumHeight(120)
        self.summary_table.setAlternatingRowColors(True)
        summary_layout.addWidget(self.summary_table)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)
        
        # Set the scroll area as the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        self.telegram_sender = None
        self.cancel_event = Event()
        self.current_message = ""

    def set_message(self, message):
        self.message_preview.setPlainText(message)
        self.current_message = message

    def send_message(self):
        token = self.token_input.text().strip()
        chat_id = self.chatid_input.text().strip()
        message = self.message_preview.toPlainText()
        if not token or not chat_id or not message:
            QMessageBox.warning(self, "Missing Info", "Please provide bot token, chat ID, and message.")
            return
        self.send_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.log_text.clear()
        self.summary_table.setRowCount(0)
        self.telegram_sender = TelegramSender(token, chat_id, message, self.cancel_event)
        self.telegram_sender.finished_signal.connect(self.handle_send_result)
        self.telegram_sender.progress_signal.connect(self.update_progress)
        self.telegram_sender.log_signal.connect(self.append_log)
        self.telegram_sender.start()

    def cancel_sending(self):
        if self.cancel_event:
            self.cancel_event.set()
        self.send_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def export_log(self):
        log_text = self.log_text.toPlainText()
        if not log_text.strip():
            QMessageBox.warning(self, "No Log", "There is no log to export.")
            return
        file, _ = QFileDialog.getSaveFileName(self, "Export Log", "telegram_log.txt", "Text Files (*.txt)")
        if file:
            try:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(log_text)
                QMessageBox.information(self, "Success", f"Log exported to {file}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def update_progress(self, current, total):
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress.setValue(percentage)

    def append_log(self, msg):
        self.log_text.append(msg)
        # Update summary table with message info
        import re
        # Extract message number and chunk number from log message
        msg_match = re.search(r'Message (\d+)', msg)
        chunk_match = re.search(r'Chunk (\d+)', msg)
        status_match = re.search(r'(Success|Failed|Error|Sending cancelled by user)', msg)
        
        if msg_match and chunk_match and status_match:
            msg_num = msg_match.group(1)
            chunk_num = chunk_match.group(1)
            status = status_match.group(1)
            
            # Add to summary table
            row = self.summary_table.rowCount()
            self.summary_table.insertRow(row)
            self.summary_table.setItem(row, 0, QTableWidgetItem(f"Message {msg_num}"))
            self.summary_table.setItem(row, 1, QTableWidgetItem(f"Chunk {chunk_num}"))
            self.summary_table.setItem(row, 2, QTableWidgetItem(status))

    def handle_send_result(self, success, msg):
        self.send_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if success:
            QMessageBox.information(self, "Success", msg)
        else:
            QMessageBox.critical(self, "Error", msg)

class SignalLogicTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create scroll area for the entire tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Stock Selection Group ---
        stock_group = QGroupBox("Stock Selection")
        stock_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        stock_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f0f8ff;
            }
            QGroupBox::title {
                color: #1976D2;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        stock_layout = QHBoxLayout()
        
        stock_layout.addWidget(QLabel("Database:"))
        self.db_path_label = QLabel("data/databases/production/psx_consolidated_data_indicators_PSX.db")
        self.db_path_label.setStyleSheet("background: #e3f2fd; padding: 8px; border-radius: 6px; color: #1565c0; font-weight: bold;")
        stock_layout.addWidget(self.db_path_label, 1)
        
        self.load_stocks_btn = QPushButton("📊 Load Stocks")
        self.load_stocks_btn.setMinimumHeight(36)
        self.load_stocks_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.load_stocks_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.load_stocks_btn.clicked.connect(self.load_available_stocks)
        stock_layout.addWidget(self.load_stocks_btn)
        
        stock_group.setLayout(stock_layout)
        layout.addWidget(stock_group)

        # --- Stock Selection Controls ---
        stock_controls_group = QGroupBox("Stock Selection Controls")
        stock_controls_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        stock_controls_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4caf50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f1f8e9;
            }
            QGroupBox::title {
                color: #388e3c;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        stock_controls_layout = QHBoxLayout()
        
        stock_controls_layout.addWidget(QLabel("Select Stock:"))
        self.stock_combo = QComboBox()
        self.stock_combo.setMinimumHeight(32)
        self.stock_combo.setMinimumWidth(200)
        self.stock_combo.currentTextChanged.connect(self.on_stock_selected)
        stock_controls_layout.addWidget(self.stock_combo)
        
        self.load_stock_data_btn = QPushButton("📈 Load Stock Data")
        self.load_stock_data_btn.setMinimumHeight(36)
        self.load_stock_data_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.load_stock_data_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.load_stock_data_btn.clicked.connect(self.load_selected_stock_data)
        self.load_stock_data_btn.setEnabled(False)
        stock_controls_layout.addWidget(self.load_stock_data_btn)
        
        stock_controls_layout.addStretch()
        stock_controls_group.setLayout(stock_controls_layout)
        layout.addWidget(stock_controls_group)

        # --- Parameter Tuning Group ---
        params_group = QGroupBox("Parameter Tuning")
        params_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        params_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ff9800;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fff8f0;
            }
            QGroupBox::title {
                color: #f57c00;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        params_layout = QHBoxLayout()
        
        # RSI Parameters
        rsi_layout = QVBoxLayout()
        rsi_layout.addWidget(QLabel("RSI Thresholds:"))
        
        rsi_buy_layout = QHBoxLayout()
        rsi_buy_layout.addWidget(QLabel("Buy >"))
        self.rsi_buy_spin = QSpinBox()
        self.rsi_buy_spin.setRange(20, 80)
        self.rsi_buy_spin.setValue(40)
        self.rsi_buy_spin.setMinimumHeight(28)
        rsi_buy_layout.addWidget(self.rsi_buy_spin)
        rsi_layout.addLayout(rsi_buy_layout)
        
        rsi_sell_layout = QHBoxLayout()
        rsi_sell_layout.addWidget(QLabel("Sell <"))
        self.rsi_sell_spin = QSpinBox()
        self.rsi_sell_spin.setRange(10, 70)
        self.rsi_sell_spin.setValue(30)
        self.rsi_sell_spin.setMinimumHeight(28)
        rsi_sell_layout.addWidget(self.rsi_sell_spin)
        rsi_layout.addLayout(rsi_sell_layout)
        
        params_layout.addLayout(rsi_layout)
        
        # Volume Parameters
        volume_layout = QVBoxLayout()
        volume_layout.addWidget(QLabel("Volume Threshold:"))
        
        volume_threshold_layout = QHBoxLayout()
        volume_threshold_layout.addWidget(QLabel("Min Volume >"))
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(1000, 1000000)
        self.volume_spin.setValue(50000)
        self.volume_spin.setSuffix(" shares")
        self.volume_spin.setMinimumHeight(28)
        volume_threshold_layout.addWidget(self.volume_spin)
        volume_layout.addLayout(volume_threshold_layout)
        
        params_layout.addLayout(volume_layout)
        
        # AO Parameters
        ao_layout = QVBoxLayout()
        ao_layout.addWidget(QLabel("AO Thresholds:"))
        
        ao_buy_layout = QHBoxLayout()
        ao_buy_layout.addWidget(QLabel("Buy >"))
        self.ao_buy_spin = QDoubleSpinBox()
        self.ao_buy_spin.setRange(-10, 10)
        self.ao_buy_spin.setValue(0)
        self.ao_buy_spin.setDecimals(2)
        self.ao_buy_spin.setMinimumHeight(28)
        ao_buy_layout.addWidget(self.ao_buy_spin)
        ao_layout.addLayout(ao_buy_layout)
        
        ao_sell_layout = QHBoxLayout()
        ao_sell_layout.addWidget(QLabel("Sell <"))
        self.ao_sell_spin = QDoubleSpinBox()
        self.ao_sell_spin.setRange(-10, 10)
        self.ao_sell_spin.setValue(0)
        self.ao_sell_spin.setDecimals(2)
        self.ao_sell_spin.setMinimumHeight(28)
        ao_sell_layout.addWidget(self.ao_sell_spin)
        ao_layout.addLayout(ao_sell_layout)
        
        params_layout.addLayout(ao_layout)
        
        # New Advanced Parameters
        advanced_layout = QVBoxLayout()
        advanced_layout.addWidget(QLabel("Advanced Settings:"))
        
        # MA Period
        ma_layout = QHBoxLayout()
        ma_layout.addWidget(QLabel("MA Period:"))
        self.ma_period_spin = QSpinBox()
        self.ma_period_spin.setRange(5, 200)
        self.ma_period_spin.setValue(30)
        self.ma_period_spin.setMinimumHeight(28)
        ma_layout.addWidget(self.ma_period_spin)
        advanced_layout.addLayout(ma_layout)
        
        # Price Change Threshold
        price_change_layout = QHBoxLayout()
        price_change_layout.addWidget(QLabel("Price Change %:"))
        self.price_change_spin = QDoubleSpinBox()
        self.price_change_spin.setRange(0.1, 50.0)
        self.price_change_spin.setValue(5.0)
        self.price_change_spin.setSuffix("%")
        self.price_change_spin.setDecimals(1)
        self.price_change_spin.setMinimumHeight(28)
        price_change_layout.addWidget(self.price_change_spin)
        advanced_layout.addLayout(price_change_layout)
        
        params_layout.addLayout(advanced_layout)
        params_layout.addStretch()
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # --- Advanced Signal Logic Controls Group ---
        advanced_controls_group = QGroupBox("Advanced Signal Logic Controls")
        advanced_controls_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        advanced_controls_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #e91e63;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fce4ec;
            }
            QGroupBox::title {
                color: #c2185b;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        advanced_controls_layout = QHBoxLayout()
        
        self.optimize_btn = QPushButton("🎯 Optimize Parameters")
        self.optimize_btn.setMinimumHeight(40)
        self.optimize_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.optimize_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.optimize_btn.clicked.connect(self.optimize_parameters)
        advanced_controls_layout.addWidget(self.optimize_btn)
        
        self.export_signals_btn = QPushButton("📈 Export Signals")
        self.export_signals_btn.setMinimumHeight(40)
        self.export_signals_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.export_signals_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.export_signals_btn.clicked.connect(self.export_signals)
        advanced_controls_layout.addWidget(self.export_signals_btn)
        
        self.visualize_btn = QPushButton("📊 Visualize Signals")
        self.visualize_btn.setMinimumHeight(40)
        self.visualize_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.visualize_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.visualize_btn.clicked.connect(self.visualize_signals)
        advanced_controls_layout.addWidget(self.visualize_btn)
        
        self.compare_btn = QPushButton("⚖️ Compare Strategies")
        self.compare_btn.setMinimumHeight(40)
        self.compare_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.compare_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.compare_btn.clicked.connect(self.compare_strategies)
        advanced_controls_layout.addWidget(self.compare_btn)
        
        advanced_controls_layout.addStretch()
        advanced_controls_group.setLayout(advanced_controls_layout)
        layout.addWidget(advanced_controls_group)

        # --- Signal Logic Controls Group ---
        controls_group = QGroupBox("Signal Logic Controls")
        controls_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        controls_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #9c27b0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #faf5ff;
            }
            QGroupBox::title {
                color: #8e24aa;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        controls_layout = QHBoxLayout()
        
        self.save_logic_btn = QPushButton("💾 Save Logic")
        self.save_logic_btn.setMinimumHeight(40)
        self.save_logic_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.save_logic_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.save_logic_btn.clicked.connect(self.save_logic)
        controls_layout.addWidget(self.save_logic_btn)
        
        self.load_logic_btn = QPushButton("📂 Load Logic")
        self.load_logic_btn.setMinimumHeight(40)
        self.load_logic_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.load_logic_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.load_logic_btn.clicked.connect(self.load_logic)
        controls_layout.addWidget(self.load_logic_btn)
        
        self.reset_logic_btn = QPushButton("🔄 Reset to Default")
        self.reset_logic_btn.setMinimumHeight(40)
        self.reset_logic_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.reset_logic_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.reset_logic_btn.clicked.connect(self.reset_to_default)
        controls_layout.addWidget(self.reset_logic_btn)
        
        self.test_logic_btn = QPushButton("🧪 Test Logic")
        self.test_logic_btn.setMinimumHeight(40)
        self.test_logic_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.test_logic_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.test_logic_btn.clicked.connect(self.test_logic)
        controls_layout.addWidget(self.test_logic_btn)
        
        self.backtest_btn = QPushButton("📊 Backtest")
        self.backtest_btn.setMinimumHeight(40)
        self.backtest_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.backtest_btn.setStyleSheet("padding: 8px 24px; border-radius: 6px;")
        self.backtest_btn.clicked.connect(self.run_backtest)
        controls_layout.addWidget(self.backtest_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # --- Buy Signal Logic Group ---
        buy_group = QGroupBox("🟢 Buy Signal Logic")
        buy_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        buy_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4caf50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f1f8e9;
            }
            QGroupBox::title {
                color: #388e3c;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        buy_layout = QVBoxLayout()
        
        buy_info = QLabel("Define conditions for BUY signals. Use variables: row['RSI_Weekly_Avg'], row['AO_Weekly'], row['Close'], row['MA_30'], row['Volume']")
        buy_info.setWordWrap(True)
        buy_info.setFont(QFont("Segoe UI", 10))
        buy_info.setStyleSheet("color: #2e7d32; padding: 8px; background: #e8f5e8; border-radius: 6px;")
        buy_layout.addWidget(buy_info)
        
        self.buy_logic_edit = QTextEdit()
        self.buy_logic_edit.setFont(QFont("Consolas", 10))
        self.buy_logic_edit.setMaximumHeight(150)
        self.buy_logic_edit.setStyleSheet("""
            background: #f1f8e9;
            color: #2e7d32;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #81c784;
        """)
        self.buy_logic_edit.setPlaceholderText("""# Buy Signal Logic
# Return True for buy signal, False otherwise
def check_buy_signal(row):
    return (row['RSI_Weekly_Avg'] > 40 and 
            row['AO_Weekly'] > 0 and 
            row['Close'] > row['MA_30'] and
            row['Volume'] > 50000)""")
        buy_layout.addWidget(self.buy_logic_edit)
        buy_group.setLayout(buy_layout)
        layout.addWidget(buy_group)

        # --- Sell Signal Logic Group ---
        sell_group = QGroupBox("🔴 Sell Signal Logic")
        sell_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        sell_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #f44336;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffebee;
            }
            QGroupBox::title {
                color: #d32f2f;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        sell_layout = QVBoxLayout()
        
        sell_info = QLabel("Define conditions for SELL signals. Use variables: row['RSI_Weekly_Avg'], row['AO_Weekly'], row['Close'], row['MA_30'], row['Volume']")
        sell_info.setWordWrap(True)
        sell_info.setFont(QFont("Segoe UI", 10))
        sell_info.setStyleSheet("color: #c62828; padding: 8px; background: #ffcdd2; border-radius: 6px;")
        sell_layout.addWidget(sell_info)
        
        self.sell_logic_edit = QTextEdit()
        self.sell_logic_edit.setFont(QFont("Consolas", 10))
        self.sell_logic_edit.setMaximumHeight(150)
        self.sell_logic_edit.setStyleSheet("""
            background: #ffebee;
            color: #c62828;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #ef9a9a;
        """)
        self.sell_logic_edit.setPlaceholderText("""# Sell Signal Logic
# Return True for sell signal, False otherwise
def check_sell_signal(row):
    return (row['RSI_Weekly_Avg'] < 30 and 
            row['AO_Weekly'] < 0 and 
            row['Close'] < row['MA_30'])""")
        sell_layout.addWidget(self.sell_logic_edit)
        sell_group.setLayout(sell_layout)
        layout.addWidget(sell_group)

        # --- Neutral Signal Logic Group ---
        neutral_group = QGroupBox("🟡 Neutral Signal Logic")
        neutral_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        neutral_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ff9800;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fff8e1;
            }
            QGroupBox::title {
                color: #f57c00;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        neutral_layout = QVBoxLayout()
        
        neutral_info = QLabel("Define conditions for NEUTRAL signals. Use variables: row['RSI_Weekly_Avg'], row['AO_Weekly'], row['Close'], row['MA_30'], row['Volume']")
        neutral_info.setWordWrap(True)
        neutral_info.setFont(QFont("Segoe UI", 10))
        neutral_info.setStyleSheet("color: #ef6c00; padding: 8px; background: #ffe0b2; border-radius: 6px;")
        neutral_layout.addWidget(neutral_info)
        
        self.neutral_logic_edit = QTextEdit()
        self.neutral_logic_edit.setFont(QFont("Consolas", 10))
        self.neutral_logic_edit.setMaximumHeight(150)
        self.neutral_logic_edit.setStyleSheet("""
            background: #fff8e1;
            color: #ef6c00;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #ffcc02;
        """)
        self.neutral_logic_edit.setPlaceholderText("""# Neutral Signal Logic
# Return True for neutral signal, False otherwise
def check_neutral_signal(row):
    return (30 <= row['RSI_Weekly_Avg'] <= 40 and 
            -2 <= row['AO_Weekly'] <= 2 and
            abs(row['Close'] - row['MA_30']) / row['MA_30'] < 0.05)""")
        neutral_layout.addWidget(self.neutral_logic_edit)
        neutral_group.setLayout(neutral_layout)
        layout.addWidget(neutral_group)

        # --- Logic Status Group ---
        status_group = QGroupBox("Logic Status & Testing")
        status_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        status_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #607d8b;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f5f7fa;
            }
            QGroupBox::title {
                color: #455a64;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        status_layout = QVBoxLayout()
        
        self.logic_status_label = QLabel("Status: Ready to test")
        self.logic_status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 12px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
        status_layout.addWidget(self.logic_status_label)
        
        self.logic_output = QTextEdit()
        self.logic_output.setReadOnly(True)
        self.logic_output.setFont(QFont("Consolas", 9))
        self.logic_output.setMaximumHeight(150)
        self.logic_output.setStyleSheet("""
            background: #263238;
            color: #eceff1;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #455a64;
        """)
        self.logic_output.setPlaceholderText("Logic test results will appear here...")
        status_layout.addWidget(self.logic_output)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)
        
        # Set the scroll area as the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        # Initialize data storage
        self.available_stocks = []
        self.current_stock_data = None
        self.selected_stock = None
        
        # Load default logic
        self.load_default_logic()
        
        # Connect parameter changes to logic updates
        self.rsi_buy_spin.valueChanged.connect(self.update_logic_from_params)
        self.rsi_sell_spin.valueChanged.connect(self.update_logic_from_params)
        self.volume_spin.valueChanged.connect(self.update_logic_from_params)
        self.ao_buy_spin.valueChanged.connect(self.update_logic_from_params)
        self.ao_sell_spin.valueChanged.connect(self.update_logic_from_params)
        self.ma_period_spin.valueChanged.connect(self.update_logic_from_params)
        self.price_change_spin.valueChanged.connect(self.update_logic_from_params)

    def load_available_stocks(self):
        """Load available stocks from the database"""
        try:
            db_path = "data/databases/production/psx_consolidated_data_indicators_PSX.db"
            if not os.path.exists(db_path):
                QMessageBox.warning(self, "Database Not Found", f"Database file not found: {db_path}")
                return
            
            # Connect to database and get table names
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_stock_data'")
            tables = cursor.fetchall()
            conn.close()
            
            # Extract stock symbols from table names
            self.available_stocks = []
            for table in tables:
                stock_symbol = table[0].replace('PSX_', '').replace('_stock_data', '').strip().upper()
                self.available_stocks.append(stock_symbol)
            
            # Update combo box
            self.stock_combo.clear()
            self.stock_combo.addItems(sorted(self.available_stocks))
            
            self.logic_status_label.setText(f"Status: Loaded {len(self.available_stocks)} stocks")
            self.logic_status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 12px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
            
            QMessageBox.information(self, "Success", f"Loaded {len(self.available_stocks)} stocks from database")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load stocks: {str(e)}")

    def on_stock_selected(self, stock_symbol):
        """Handle stock selection"""
        self.selected_stock = stock_symbol
        self.load_stock_data_btn.setEnabled(bool(stock_symbol))

    def load_selected_stock_data(self):
        """Load data for the selected stock"""
        if not self.selected_stock:
            QMessageBox.warning(self, "No Stock Selected", "Please select a stock first.")
            return
        
        try:
            db_path = "data/databases/production/psx_consolidated_data_indicators_PSX.db"
            table_name = f"PSX_{self.selected_stock}_stock_data"
            
            # Load stock data
            self.current_stock_data = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY Date DESC LIMIT 100", f"sqlite:///{db_path}")
            
            self.logic_status_label.setText(f"Status: Loaded {len(self.current_stock_data)} records for {self.selected_stock}")
            self.logic_status_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 12px; padding: 8px; background: #e3f2fd; border-radius: 6px;")
            
            QMessageBox.information(self, "Success", f"Loaded {len(self.current_stock_data)} records for {self.selected_stock}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load stock data: {str(e)}")

    def update_logic_from_params(self):
        """Update logic code from parameter values"""
        rsi_buy = self.rsi_buy_spin.value()
        rsi_sell = self.rsi_sell_spin.value()
        volume = self.volume_spin.value()
        ao_buy = self.ao_buy_spin.value()
        ao_sell = self.ao_sell_spin.value()
        ma_period = self.ma_period_spin.value()
        price_change = self.price_change_spin.value()
        
        # Update buy logic with dynamic MA column
        ma_column = f'MA_{ma_period}'
        buy_logic = f"""# Buy Signal Logic
# Return True for buy signal, False otherwise
def check_buy_signal(row):
    return (row['RSI_Weekly_Avg'] > {rsi_buy} and 
            row['AO_Weekly'] > {ao_buy} and 
            row['Close'] > row['{ma_column}'] and
            row['Volume'] > {volume} and
            ((row['Close'] - row['{ma_column}']) / row['{ma_column}'] * 100) > {price_change})"""
        
        # Update sell logic
        sell_logic = f"""# Sell Signal Logic
# Return True for sell signal, False otherwise
def check_sell_signal(row):
    return (row['RSI_Weekly_Avg'] < {rsi_sell} and 
            row['AO_Weekly'] < {ao_sell} and 
            row['Close'] < row['{ma_column}'])"""
        
        # Update neutral logic
        neutral_logic = f"""# Neutral Signal Logic
# Return True for neutral signal, False otherwise
def check_neutral_signal(row):
    return ({rsi_sell} <= row['RSI_Weekly_Avg'] <= {rsi_buy} and 
            {ao_sell} <= row['AO_Weekly'] <= {ao_buy} and
            abs(row['Close'] - row['{ma_column}']) / row['{ma_column}'] < {price_change/100})"""
        
        self.buy_logic_edit.setPlainText(buy_logic)
        self.sell_logic_edit.setPlainText(sell_logic)
        self.neutral_logic_edit.setPlainText(neutral_logic)

    def load_default_logic(self):
        """Load default signal logic"""
        ma_column = f'MA_{self.ma_period_spin.value()}'
        price_change = self.price_change_spin.value()
        
        self.buy_logic_edit.setPlainText(f"""# Buy Signal Logic
# Return True for buy signal, False otherwise
def check_buy_signal(row):
    return (row['RSI_Weekly_Avg'] > 40 and 
            row['AO_Weekly'] > 0 and 
            row['Close'] > row['{ma_column}'] and
            row['Volume'] > 50000 and
            ((row['Close'] - row['{ma_column}']) / row['{ma_column}'] * 100) > {price_change})""")
        
        self.sell_logic_edit.setPlainText(f"""# Sell Signal Logic
# Return True for sell signal, False otherwise
def check_sell_signal(row):
    return (row['RSI_Weekly_Avg'] < 30 and 
            row['AO_Weekly'] < 0 and 
            row['Close'] < row['{ma_column}'])""")
        
        self.neutral_logic_edit.setPlainText(f"""# Neutral Signal Logic
# Return True for neutral signal, False otherwise
def check_neutral_signal(row):
    return (30 <= row['RSI_Weekly_Avg'] <= 40 and 
            -2 <= row['AO_Weekly'] <= 2 and
            abs(row['Close'] - row['{ma_column}']) / row['{ma_column}'] < {price_change/100})""")

    def save_logic(self):
        """Save current logic to file"""
        try:
            file, _ = QFileDialog.getSaveFileName(
                self, "Save Signal Logic", "signal_logic.py", "Python Files (*.py)"
            )
            if file:
                logic_content = f"""# Signal Logic Configuration
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Parameters: RSI_Buy={self.rsi_buy_spin.value()}, RSI_Sell={self.rsi_sell_spin.value()}, 
# Volume={self.volume_spin.value()}, AO_Buy={self.ao_buy_spin.value()}, AO_Sell={self.ao_sell_spin.value()}

{self.buy_logic_edit.toPlainText()}

{self.sell_logic_edit.toPlainText()}

{self.neutral_logic_edit.toPlainText()}
"""
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(logic_content)
                QMessageBox.information(self, "Success", f"Logic saved to {file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save logic: {str(e)}")

    def load_logic(self):
        """Load logic from file"""
        try:
            file, _ = QFileDialog.getOpenFileName(
                self, "Load Signal Logic", "", "Python Files (*.py)"
            )
            if file:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the content to extract logic sections
                # This is a simple parser - you might want to make it more robust
                sections = content.split('\n\n')
                
                for section in sections:
                    if 'check_buy_signal' in section:
                        self.buy_logic_edit.setPlainText(section.strip())
                    elif 'check_sell_signal' in section:
                        self.sell_logic_edit.setPlainText(section.strip())
                    elif 'check_neutral_signal' in section:
                        self.neutral_logic_edit.setPlainText(section.strip())
                
                QMessageBox.information(self, "Success", f"Logic loaded from {file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load logic: {str(e)}")

    def reset_to_default(self):
        """Reset logic to default values"""
        reply = QMessageBox.question(self, "Reset Logic", 
                                   "Are you sure you want to reset to default logic?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.load_default_logic()
            self.logic_status_label.setText("Status: Reset to default logic")
            self.logic_status_label.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 12px; padding: 8px; background: #fff8e1; border-radius: 6px;")

    def test_logic(self):
        """Test the current logic with sample data"""
        if self.current_stock_data is None or self.current_stock_data.empty:
            QMessageBox.warning(self, "No Data", "Please load stock data first.")
            return
        
        try:
            # Get the logic functions
            buy_logic = self.buy_logic_edit.toPlainText()
            sell_logic = self.sell_logic_edit.toPlainText()
            neutral_logic = self.neutral_logic_edit.toPlainText()
            
            # Create a test environment
            test_env = {}
            exec(buy_logic, test_env)
            exec(sell_logic, test_env)
            exec(neutral_logic, test_env)
            
            check_buy_signal = test_env['check_buy_signal']
            check_sell_signal = test_env['check_sell_signal']
            check_neutral_signal = test_env['check_neutral_signal']
            
            # Test with recent data
            results = []
            buy_count = 0
            sell_count = 0
            neutral_count = 0
            
            for idx, row in self.current_stock_data.head(10).iterrows():
                buy_signal = check_buy_signal(row)
                sell_signal = check_sell_signal(row)
                neutral_signal = check_neutral_signal(row)
                
                if buy_signal:
                    buy_count += 1
                    signal_type = "BUY"
                elif sell_signal:
                    sell_count += 1
                    signal_type = "SELL"
                elif neutral_signal:
                    neutral_count += 1
                    signal_type = "NEUTRAL"
                else:
                    signal_type = "NONE"
                
                results.append(f"Date: {row['Date']} | Signal: {signal_type} | RSI: {row['RSI_Weekly_Avg']:.2f} | AO: {row['AO_Weekly']:.2f} | Close: {row['Close']:.2f}")
            
            # Display results
            output = f"""Logic Test Results for {self.selected_stock}:
Recent 10 records analysis:

{chr(10).join(results)}

Summary:
- Buy Signals: {buy_count}
- Sell Signals: {sell_count}
- Neutral Signals: {neutral_count}
- No Signal: {10 - buy_count - sell_count - neutral_count}

Logic functions loaded successfully!"""
            
            self.logic_output.setPlainText(output)
            self.logic_status_label.setText("Status: Logic test completed successfully")
            self.logic_status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 12px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
            
        except Exception as e:
            error_msg = f"Logic test failed: {str(e)}"
            self.logic_output.setPlainText(error_msg)
            self.logic_status_label.setText("Status: Logic test failed")
            self.logic_status_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 12px; padding: 8px; background: #ffebee; border-radius: 6px;")

    def run_backtest(self):
        """Run backtest on the current logic"""
        if self.current_stock_data is None or self.current_stock_data.empty:
            QMessageBox.warning(self, "No Data", "Please load stock data first.")
            return
        
        try:
            # Get the logic functions
            buy_logic = self.buy_logic_edit.toPlainText()
            sell_logic = self.sell_logic_edit.toPlainText()
            neutral_logic = self.neutral_logic_edit.toPlainText()
            
            # Create a test environment
            test_env = {}
            exec(buy_logic, test_env)
            exec(sell_logic, test_env)
            exec(neutral_logic, test_env)
            
            check_buy_signal = test_env['check_buy_signal']
            check_sell_signal = test_env['check_sell_signal']
            check_neutral_signal = test_env['check_neutral_signal']
            
            # Run backtest on all data
            signals = []
            buy_count = 0
            sell_count = 0
            neutral_count = 0
            
            for idx, row in self.current_stock_data.iterrows():
                buy_signal = check_buy_signal(row)
                sell_signal = check_sell_signal(row)
                neutral_signal = check_neutral_signal(row)
                
                if buy_signal:
                    buy_count += 1
                    signal_type = "BUY"
                elif sell_signal:
                    sell_count += 1
                    signal_type = "SELL"
                elif neutral_signal:
                    neutral_count += 1
                    signal_type = "NEUTRAL"
                else:
                    signal_type = "NONE"
                
                signals.append({
                    'date': row['Date'],
                    'signal': signal_type,
                    'close': row['Close'],
                    'rsi': row['RSI_Weekly_Avg'],
                    'ao': row['AO_Weekly']
                })
            
            # Calculate performance metrics
            total_records = len(self.current_stock_data)
            signal_rate = (buy_count + sell_count + neutral_count) / total_records * 100
            
            # Display backtest results
            output = f"""Backtest Results for {self.selected_stock}:
Period: {self.current_stock_data['Date'].min()} to {self.current_stock_data['Date'].max()}
Total Records: {total_records}

Signal Distribution:
- Buy Signals: {buy_count} ({buy_count/total_records*100:.1f}%)
- Sell Signals: {sell_count} ({sell_count/total_records*100:.1f}%)
- Neutral Signals: {neutral_count} ({neutral_count/total_records*100:.1f}%)
- No Signal: {total_records - buy_count - sell_count - neutral_count} ({(total_records - buy_count - sell_count - neutral_count)/total_records*100:.1f}%)

Signal Rate: {signal_rate:.1f}%

Recent Signals (Last 10):
"""
            
            # Add recent signals
            recent_signals = [s for s in signals if s['signal'] != 'NONE'][-10:]
            for signal in recent_signals:
                output += f"- {signal['date']}: {signal['signal']} | Close: {signal['close']:.2f} | RSI: {signal['rsi']:.2f} | AO: {signal['ao']:.2f}\n"
            
            self.logic_output.setPlainText(output)
            self.logic_status_label.setText("Status: Backtest completed successfully")
            self.logic_status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 12px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
            
        except Exception as e:
            error_msg = f"Backtest failed: {str(e)}"
            self.logic_output.setPlainText(error_msg)
            self.logic_status_label.setText("Status: Backtest failed")
            self.logic_status_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 12px; padding: 8px; background: #ffebee; border-radius: 6px;")

    def get_signal_logic(self):
        """Get the current signal logic as functions"""
        try:
            buy_logic = self.buy_logic_edit.toPlainText()
            sell_logic = self.sell_logic_edit.toPlainText()
            neutral_logic = self.neutral_logic_edit.toPlainText()
            
            # Create a namespace for the functions
            logic_env = {}
            exec(buy_logic, logic_env)
            exec(sell_logic, logic_env)
            exec(neutral_logic, logic_env)
            
            return {
                'check_buy_signal': logic_env['check_buy_signal'],
                'check_sell_signal': logic_env['check_sell_signal'],
                'check_neutral_signal': logic_env['check_neutral_signal']
            }
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compile signal logic: {str(e)}")
            return None

    def optimize_parameters(self):
        """Optimize parameters using grid search"""
        if self.current_stock_data is None or self.current_stock_data.empty:
            QMessageBox.warning(self, "No Data", "Please load stock data first.")
            return
        
        try:
            self.logic_status_label.setText("Status: Optimizing parameters...")
            self.logic_status_label.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 12px; padding: 8px; background: #fff8e1; border-radius: 6px;")
            
            # Simple grid search optimization
            best_score = 0
            best_params = {}
            
            rsi_buy_range = range(35, 46, 5)
            rsi_sell_range = range(25, 36, 5)
            ao_buy_range = [-0.5, 0, 0.5]
            ao_sell_range = [-0.5, 0, 0.5]
            
            total_combinations = len(rsi_buy_range) * len(rsi_sell_range) * len(ao_buy_range) * len(ao_sell_range)
            current_combination = 0
            
            for rsi_buy in rsi_buy_range:
                for rsi_sell in rsi_sell_range:
                    for ao_buy in ao_buy_range:
                        for ao_sell in ao_sell_range:
                            current_combination += 1
                            
                            # Test this combination
                            score = self._evaluate_parameters(rsi_buy, rsi_sell, ao_buy, ao_sell)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'rsi_buy': rsi_buy,
                                    'rsi_sell': rsi_sell,
                                    'ao_buy': ao_buy,
                                    'ao_sell': ao_sell
                                }
            
            # Apply best parameters
            if best_params:
                self.rsi_buy_spin.setValue(best_params['rsi_buy'])
                self.rsi_sell_spin.setValue(best_params['rsi_sell'])
                self.ao_buy_spin.setValue(best_params['ao_buy'])
                self.ao_sell_spin.setValue(best_params['ao_sell'])
                
                output = f"""Parameter Optimization Complete!
Tested {total_combinations} combinations.

Best Parameters Found:
- RSI Buy Threshold: {best_params['rsi_buy']}
- RSI Sell Threshold: {best_params['rsi_sell']}
- AO Buy Threshold: {best_params['ao_buy']}
- AO Sell Threshold: {best_params['ao_sell']}

Best Score: {best_score:.2f}

Parameters have been automatically applied."""
                
                self.logic_output.setPlainText(output)
                self.logic_status_label.setText("Status: Parameters optimized successfully")
                self.logic_status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 12px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
            
        except Exception as e:
            error_msg = f"Parameter optimization failed: {str(e)}"
            self.logic_output.setPlainText(error_msg)
            self.logic_status_label.setText("Status: Optimization failed")
            self.logic_status_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 12px; padding: 8px; background: #ffebee; border-radius: 6px;")

    def _evaluate_parameters(self, rsi_buy, rsi_sell, ao_buy, ao_sell):
        """Evaluate parameter combination and return a score"""
        try:
            # Create test logic with these parameters
            buy_logic = f"""def check_buy_signal(row):
    return (row['RSI_Weekly_Avg'] > {rsi_buy} and 
            row['AO_Weekly'] > {ao_buy} and 
            row['Close'] > row['MA_30'] and
            row['Volume'] > {self.volume_spin.value()})"""
            
            sell_logic = f"""def check_sell_signal(row):
    return (row['RSI_Weekly_Avg'] < {rsi_sell} and 
            row['AO_Weekly'] < {ao_sell} and 
            row['Close'] < row['MA_30'])"""
            
            # Execute logic
            test_env = {}
            exec(buy_logic, test_env)
            exec(sell_logic, test_env)
            
            check_buy_signal = test_env['check_buy_signal']
            check_sell_signal = test_env['check_sell_signal']
            
            # Test on data
            buy_count = 0
            sell_count = 0
            
            for idx, row in self.current_stock_data.iterrows():
                if check_buy_signal(row):
                    buy_count += 1
                elif check_sell_signal(row):
                    sell_count += 1
            
            # Calculate score (balance between buy and sell signals)
            total_signals = buy_count + sell_count
            if total_signals == 0:
                return 0
            
            # Prefer balanced signals with reasonable frequency
            balance_score = 1 - abs(buy_count - sell_count) / total_signals
            frequency_score = min(total_signals / len(self.current_stock_data), 0.3) / 0.3
            
            return balance_score * 0.7 + frequency_score * 0.3
            
        except Exception:
            return 0

    def export_signals(self):
        """Export signal analysis to CSV"""
        if self.current_stock_data is None or self.current_stock_data.empty:
            QMessageBox.warning(self, "No Data", "Please load stock data first.")
            return
        
        try:
            file, _ = QFileDialog.getSaveFileName(
                self, "Export Signals", f"{self.selected_stock}_signals.csv", "CSV Files (*.csv)"
            )
            if file:
                # Generate signals
                signals_df = self._generate_signals_dataframe()
                signals_df.to_csv(file, index=False)
                
                QMessageBox.information(self, "Success", f"Signals exported to {file}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export signals: {str(e)}")

    def _generate_signals_dataframe(self):
        """Generate a dataframe with signal analysis"""
        try:
            # Get the logic functions
            buy_logic = self.buy_logic_edit.toPlainText()
            sell_logic = self.sell_logic_edit.toPlainText()
            neutral_logic = self.neutral_logic_edit.toPlainText()
            
            # Create a test environment
            test_env = {}
            exec(buy_logic, test_env)
            exec(sell_logic, test_env)
            exec(neutral_logic, test_env)
            
            check_buy_signal = test_env['check_buy_signal']
            check_sell_signal = test_env['check_sell_signal']
            check_neutral_signal = test_env['check_neutral_signal']
            
            # Generate signals
            signals_data = []
            for idx, row in self.current_stock_data.iterrows():
                buy_signal = check_buy_signal(row)
                sell_signal = check_sell_signal(row)
                neutral_signal = check_neutral_signal(row)
                
                if buy_signal:
                    signal_type = "BUY"
                elif sell_signal:
                    signal_type = "SELL"
                elif neutral_signal:
                    signal_type = "NEUTRAL"
                else:
                    signal_type = "NONE"
                
                signals_data.append({
                    'Date': row['Date'],
                    'Symbol': self.selected_stock,
                    'Signal': signal_type,
                    'Close': row['Close'],
                    'Volume': row['Volume'],
                    'RSI_Weekly_Avg': row['RSI_Weekly_Avg'],
                    'AO_Weekly': row['AO_Weekly'],
                    'MA_30': row['MA_30'],
                    'Price_Change_Pct': ((row['Close'] - row['MA_30']) / row['MA_30']) * 100
                })
            
            return pd.DataFrame(signals_data)
            
        except Exception as e:
            raise Exception(f"Failed to generate signals dataframe: {str(e)}")

    def visualize_signals(self):
        """Create signal visualization"""
        if self.current_stock_data is None or self.current_stock_data.empty:
            QMessageBox.warning(self, "No Data", "Please load stock data first.")
            return
        
        try:
            # Generate signals data
            signals_df = self._generate_signals_dataframe()
            
            # Create visualization
            output = f"""Signal Visualization for {self.selected_stock}:

📊 Signal Distribution:
{signals_df['Signal'].value_counts().to_string()}

📈 Price Analysis:
- Average Close Price: {signals_df['Close'].mean():.2f}
- Price Range: {signals_df['Close'].min():.2f} - {signals_df['Close'].max():.2f}
- Price Volatility: {signals_df['Close'].std():.2f}

📉 RSI Analysis:
- Average RSI: {signals_df['RSI_Weekly_Avg'].mean():.2f}
- RSI Range: {signals_df['RSI_Weekly_Avg'].min():.2f} - {signals_df['RSI_Weekly_Avg'].max():.2f}

📊 AO Analysis:
- Average AO: {signals_df['AO_Weekly'].mean():.2f}
- AO Range: {signals_df['AO_Weekly'].min():.2f} - {signals_df['AO_Weekly'].max():.2f}

🎯 Signal Performance:
- Buy Signals: {len(signals_df[signals_df['Signal'] == 'BUY'])} 
  (Avg RSI: {signals_df[signals_df['Signal'] == 'BUY']['RSI_Weekly_Avg'].mean():.2f})
- Sell Signals: {len(signals_df[signals_df['Signal'] == 'SELL'])}
  (Avg RSI: {signals_df[signals_df['Signal'] == 'SELL']['RSI_Weekly_Avg'].mean():.2f})
- Neutral Signals: {len(signals_df[signals_df['Signal'] == 'NEUTRAL'])}
  (Avg RSI: {signals_df[signals_df['Signal'] == 'NEUTRAL']['RSI_Weekly_Avg'].mean():.2f})

📅 Recent Signals (Last 10):
"""
            
            recent_signals = signals_df[signals_df['Signal'] != 'NONE'].tail(10)
            for _, signal in recent_signals.iterrows():
                output += f"- {signal['Date']}: {signal['Signal']} | Close: {signal['Close']:.2f} | RSI: {signal['RSI_Weekly_Avg']:.2f} | AO: {signal['AO_Weekly']:.2f}\n"
            
            self.logic_output.setPlainText(output)
            self.logic_status_label.setText("Status: Visualization completed")
            self.logic_status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 12px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
            
        except Exception as e:
            error_msg = f"Visualization failed: {str(e)}"
            self.logic_output.setPlainText(error_msg)
            self.logic_status_label.setText("Status: Visualization failed")
            self.logic_status_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 12px; padding: 8px; background: #ffebee; border-radius: 6px;")

    def compare_strategies(self):
        """Compare different signal strategies"""
        if self.current_stock_data is None or self.current_stock_data.empty:
            QMessageBox.warning(self, "No Data", "Please load stock data first.")
            return
        
        try:
            # Define different strategies
            strategies = {
                'Conservative': {'rsi_buy': 45, 'rsi_sell': 25, 'ao_buy': 0.5, 'ao_sell': -0.5},
                'Moderate': {'rsi_buy': 40, 'rsi_sell': 30, 'ao_buy': 0, 'ao_sell': 0},
                'Aggressive': {'rsi_buy': 35, 'rsi_sell': 35, 'ao_buy': -0.5, 'ao_sell': 0.5}
            }
            
            results = {}
            
            for strategy_name, params in strategies.items():
                # Test strategy
                buy_logic = f"""def check_buy_signal(row):
    return (row['RSI_Weekly_Avg'] > {params['rsi_buy']} and 
            row['AO_Weekly'] > {params['ao_buy']} and 
            row['Close'] > row['MA_30'] and
            row['Volume'] > {self.volume_spin.value()})"""
                
                sell_logic = f"""def check_sell_signal(row):
    return (row['RSI_Weekly_Avg'] < {params['rsi_sell']} and 
            row['AO_Weekly'] < {params['ao_sell']} and 
            row['Close'] < row['MA_30'])"""
                
                # Execute logic
                test_env = {}
                exec(buy_logic, test_env)
                exec(sell_logic, test_env)
                
                check_buy_signal = test_env['check_buy_signal']
                check_sell_signal = test_env['check_sell_signal']
                
                # Count signals
                buy_count = sum(1 for _, row in self.current_stock_data.iterrows() if check_buy_signal(row))
                sell_count = sum(1 for _, row in self.current_stock_data.iterrows() if check_sell_signal(row))
                total_signals = buy_count + sell_count
                
                results[strategy_name] = {
                    'buy_signals': buy_count,
                    'sell_signals': sell_count,
                    'total_signals': total_signals,
                    'signal_rate': total_signals / len(self.current_stock_data) * 100,
                    'balance': 1 - abs(buy_count - sell_count) / max(total_signals, 1)
                }
            
            # Display comparison
            output = f"""Strategy Comparison for {self.selected_stock}:

{'Strategy':<15} {'Buy':<8} {'Sell':<8} {'Total':<8} {'Rate%':<8} {'Balance':<8}
{'-' * 60}
"""
            
            for strategy_name, result in results.items():
                output += f"{strategy_name:<15} {result['buy_signals']:<8} {result['sell_signals']:<8} {result['total_signals']:<8} {result['signal_rate']:<8.1f} {result['balance']:<8.2f}\n"
            
            output += f"""
Analysis:
- Conservative: Fewer signals, higher RSI thresholds
- Moderate: Balanced approach with current settings
- Aggressive: More signals, lower RSI thresholds

Current Strategy: Moderate
Parameters: RSI Buy={self.rsi_buy_spin.value()}, RSI Sell={self.rsi_sell_spin.value()}, AO Buy={self.ao_buy_spin.value()}, AO Sell={self.ao_sell_spin.value()}
"""
            
            self.logic_output.setPlainText(output)
            self.logic_status_label.setText("Status: Strategy comparison completed")
            self.logic_status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 12px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
            
        except Exception as e:
            error_msg = f"Strategy comparison failed: {str(e)}"
            self.logic_output.setPlainText(error_msg)
            self.logic_status_label.setText("Status: Comparison failed")
            self.logic_status_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 12px; padding: 8px; background: #ffebee; border-radius: 6px;")

class DatabaseAnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create scroll area for the entire tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Database Connection Group ---
        db_group = QGroupBox("Database Connection")
        db_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        db_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f0f8ff;
            }
            QGroupBox::title {
                color: #1976D2;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        db_layout = QHBoxLayout()
        
        db_layout.addWidget(QLabel("Database:"))
        self.db_path_label = QLabel("data/databases/production/PSX_investing_Stocks_KMI100.db")
        self.db_path_label.setStyleSheet("background: #e3f2fd; padding: 8px; border-radius: 6px; color: #1565c0; font-weight: bold;")
        db_layout.addWidget(self.db_path_label, 1)
        
        self.refresh_btn = QPushButton("🔄 Refresh Data")
        self.refresh_btn.setMinimumHeight(36)
        self.refresh_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.refresh_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.refresh_btn.clicked.connect(self.load_database_data)
        db_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("📊 Export Analysis")
        self.export_btn.setMinimumHeight(36)
        self.export_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.export_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.export_btn.clicked.connect(self.export_analysis)
        db_layout.addWidget(self.export_btn)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        # --- Statistics Group ---
        stats_group = QGroupBox("Signal Statistics")
        stats_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        stats_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4caf50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f1f8e9;
            }
            QGroupBox::title {
                color: #388e3c;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        stats_layout = QHBoxLayout()
        
        self.buy_count_label = QLabel("Buy Signals: 0")
        self.buy_count_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 13px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
        stats_layout.addWidget(self.buy_count_label)
        
        self.sell_count_label = QLabel("Sell Signals: 0")
        self.sell_count_label.setStyleSheet("color: #F44336; font-weight: bold; font-size: 13px; padding: 8px; background: #ffebee; border-radius: 6px;")
        stats_layout.addWidget(self.sell_count_label)
        
        self.neutral_count_label = QLabel("Neutral Signals: 0")
        self.neutral_count_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 13px; padding: 8px; background: #fff8e1; border-radius: 6px;")
        stats_layout.addWidget(self.neutral_count_label)
        
        self.total_count_label = QLabel("Total: 0")
        self.total_count_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 13px; padding: 8px; background: #e3f2fd; border-radius: 6px;")
        stats_layout.addWidget(self.total_count_label)
        
        stats_layout.addStretch()
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # --- Signal Type Selection Group ---
        signal_group = QGroupBox("Signal Type Selection")
        signal_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        signal_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ff9800;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fff8f0;
            }
            QGroupBox::title {
                color: #f57c00;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        signal_layout = QHBoxLayout()
        
        signal_layout.addWidget(QLabel("Signal Type:"))
        self.signal_type_combo = QComboBox()
        self.signal_type_combo.addItems(["Buy Signals", "Sell Signals", "Neutral Signals", "All Signals"])
        self.signal_type_combo.setMinimumHeight(32)
        self.signal_type_combo.currentIndexChanged.connect(self.on_signal_type_changed)
        signal_layout.addWidget(self.signal_type_combo)
        
        signal_layout.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter text to filter...")
        self.filter_input.setMinimumHeight(32)
        self.filter_input.textChanged.connect(self.apply_filter)
        signal_layout.addWidget(self.filter_input, 1)
        
        self.clear_filter_btn = QPushButton("Clear Filter")
        self.clear_filter_btn.setMinimumHeight(32)
        self.clear_filter_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.clear_filter_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        signal_layout.addWidget(self.clear_filter_btn)
        
        signal_group.setLayout(signal_layout)
        layout.addWidget(signal_group)

        # --- Data Table Group ---
        table_group = QGroupBox("Signal Data")
        table_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        table_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #607d8b;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f5f7fa;
            }
            QGroupBox::title {
                color: #455a64;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        table_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        self.data_table.setSortingEnabled(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setMaximumHeight(400)
        self.data_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: #ffffff;
                alternate-background-color: #f9f9f9;
                color: #333333;
                border-radius: 8px;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                color: #333333;
                padding: 8px;
                border: 1px solid #e0e0e0;
                font-weight: bold;
                border-radius: 4px;
            }
        """)
        table_layout.addWidget(self.data_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # --- Analysis Group ---
        analysis_group = QGroupBox("Signal Analysis")
        analysis_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        analysis_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #9c27b0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #faf5ff;
            }
            QGroupBox::title {
                color: #8e24aa;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        analysis_layout = QVBoxLayout()
        
        # Analysis controls
        analysis_controls = QHBoxLayout()
        analysis_controls.addWidget(QLabel("Analysis Type:"))
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["Signal Distribution", "Performance by Stock", "Date Analysis", "RSI Analysis"])
        self.analysis_type_combo.setMinimumHeight(32)
        self.analysis_type_combo.currentIndexChanged.connect(self.run_analysis)
        analysis_controls.addWidget(self.analysis_type_combo)
        
        self.run_analysis_btn = QPushButton("📈 Run Analysis")
        self.run_analysis_btn.setMinimumHeight(36)
        self.run_analysis_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.run_analysis_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        analysis_controls.addWidget(self.run_analysis_btn)
        
        analysis_controls.addStretch()
        analysis_layout.addLayout(analysis_controls)
        
        # Analysis results
        self.analysis_output = QTextEdit()
        self.analysis_output.setReadOnly(True)
        self.analysis_output.setFont(QFont("Consolas", 9))
        self.analysis_output.setMaximumHeight(200)
        self.analysis_output.setStyleSheet("""
            background: #f3e5f5;
            color: #4a148c;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #ba68c8;
        """)
        self.analysis_output.setPlaceholderText("Analysis results will appear here...")
        analysis_layout.addWidget(self.analysis_output)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)
        
        # Set the scroll area as the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        # Initialize data storage
        self.buy_data = None
        self.sell_data = None
        self.neutral_data = None
        self.current_data = None
        
        # Load initial data
        self.load_database_data()

    def load_database_data(self):
        """Load data from the PSX_investing_Stocks_KMI100.db database"""
        try:
            db_path = "data/databases/production/PSX_investing_Stocks_KMI100.db"
            
            if not os.path.exists(db_path):
                QMessageBox.warning(self, "Database Not Found", f"Database file not found: {db_path}")
                return
            
            # Load data from all tables
            self.buy_data = pd.read_sql("SELECT * FROM buy_stocks", f"sqlite:///{db_path}")
            self.sell_data = pd.read_sql("SELECT * FROM sell_stocks", f"sqlite:///{db_path}")
            self.neutral_data = pd.read_sql("SELECT * FROM neutral_stocks", f"sqlite:///{db_path}")
            
            # Update statistics
            self.update_statistics()
            
            # Set initial data
            self.on_signal_type_changed()
            
            QMessageBox.information(self, "Success", "Database data loaded successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load database data: {str(e)}")

    def update_statistics(self):
        """Update the statistics labels"""
        buy_count = len(self.buy_data) if self.buy_data is not None else 0
        sell_count = len(self.sell_data) if self.sell_data is not None else 0
        neutral_count = len(self.neutral_data) if self.neutral_data is not None else 0
        total_count = buy_count + sell_count + neutral_count
        
        self.buy_count_label.setText(f"Buy Signals: {buy_count}")
        self.sell_count_label.setText(f"Sell Signals: {sell_count}")
        self.neutral_count_label.setText(f"Neutral Signals: {neutral_count}")
        self.total_count_label.setText(f"Total: {total_count}")

    def on_signal_type_changed(self):
        """Handle signal type selection change"""
        signal_type = self.signal_type_combo.currentText()
        
        if signal_type == "Buy Signals" and self.buy_data is not None:
            self.current_data = self.buy_data
        elif signal_type == "Sell Signals" and self.sell_data is not None:
            self.current_data = self.sell_data
        elif signal_type == "Neutral Signals" and self.neutral_data is not None:
            self.current_data = self.neutral_data
        elif signal_type == "All Signals":
            # Combine all data
            all_data = []
            if self.buy_data is not None:
                buy_with_type = self.buy_data.copy()
                buy_with_type['Signal_Type'] = 'Buy'
                all_data.append(buy_with_type)
            if self.sell_data is not None:
                sell_with_type = self.sell_data.copy()
                sell_with_type['Signal_Type'] = 'Sell'
                all_data.append(sell_with_type)
            if self.neutral_data is not None:
                neutral_with_type = self.neutral_data.copy()
                neutral_with_type['Signal_Type'] = 'Neutral'
                all_data.append(neutral_with_type)
            
            if all_data:
                self.current_data = pd.concat(all_data, ignore_index=True)
            else:
                self.current_data = None
        else:
            self.current_data = None
        
        self.update_table()
        self.apply_filter()

    def update_table(self):
        """Update the data table with current data"""
        if self.current_data is None or self.current_data.empty:
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            return
        
        # Set table dimensions
        self.data_table.setRowCount(len(self.current_data))
        self.data_table.setColumnCount(len(self.current_data.columns))
        
        # Set headers
        self.data_table.setHorizontalHeaderLabels(self.current_data.columns)
        
        # Populate table
        for i, row in self.current_data.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)
        
        # Resize columns to content
        self.data_table.resizeColumnsToContents()

    def apply_filter(self):
        """Apply text filter to the table"""
        filter_text = self.filter_input.text().lower()
        
        if not filter_text or self.current_data is None:
            self.update_table()
            return
        
        # Filter the data
        filtered_data = self.current_data[
            self.current_data.astype(str).apply(
                lambda x: x.str.lower().str.contains(filter_text, na=False)
            ).any(axis=1)
        ]
        
        # Update table with filtered data
        self.data_table.setRowCount(len(filtered_data))
        self.data_table.setColumnCount(len(filtered_data.columns))
        self.data_table.setHorizontalHeaderLabels(filtered_data.columns)
        
        for i, row in filtered_data.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)
        
        self.data_table.resizeColumnsToContents()

    def clear_filter(self):
        """Clear the filter"""
        self.filter_input.clear()
        self.update_table()

    def run_analysis(self):
        """Run the selected analysis"""
        if self.current_data is None or self.current_data.empty:
            QMessageBox.warning(self, "No Data", "No data available for analysis.")
            return
        
        analysis_type = self.analysis_type_combo.currentText()
        
        try:
            if analysis_type == "Signal Distribution":
                self.analyze_signal_distribution()
            elif analysis_type == "Performance by Stock":
                self.analyze_performance_by_stock()
            elif analysis_type == "Date Analysis":
                self.analyze_by_date()
            elif analysis_type == "RSI Analysis":
                self.analyze_rsi()
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Error running analysis: {str(e)}")

    def analyze_signal_distribution(self):
        """Analyze signal distribution"""
        if 'Signal_Type' not in self.current_data.columns:
            # If we don't have signal type, we need to determine it from the data source
            if self.current_data is self.buy_data:
                signal_type = "Buy"
            elif self.current_data is self.sell_data:
                signal_type = "Sell"
            elif self.current_data is self.neutral_data:
                signal_type = "Neutral"
            else:
                signal_type = "Unknown"
            
            analysis_result = f"""Signal Distribution Analysis:
Signal Type: {signal_type}
Total Records: {len(self.current_data)}

Column Analysis:
{self.current_data.describe().to_string()}

Sample Data (First 5 rows):
{self.current_data.head().to_string()}
"""
        else:
            # We have signal types in the data
            signal_counts = self.current_data['Signal_Type'].value_counts()
            analysis_result = f"""Signal Distribution Analysis:
Total Records: {len(self.current_data)}

Signal Type Distribution:
{signal_counts.to_string()}

Column Analysis:
{self.current_data.describe().to_string()}

Sample Data (First 5 rows):
{self.current_data.head().to_string()}
"""
        
        self.analysis_output.setPlainText(analysis_result)

    def analyze_performance_by_stock(self):
        """Analyze performance by stock"""
        if 'Stock' not in self.current_data.columns:
            QMessageBox.warning(self, "No Stock Column", "Stock column not found in data.")
            return
        
        stock_counts = self.current_data['Stock'].value_counts()
        analysis_result = f"""Performance by Stock Analysis:
Total Records: {len(self.current_data)}

Top 10 Stocks by Signal Count:
{stock_counts.head(10).to_string()}

Stock Statistics:
{stock_counts.describe().to_string()}

Sample Data by Stock:
{self.current_data.groupby('Stock').size().head(10).to_string()}
"""
        
        self.analysis_output.setPlainText(analysis_result)

    def analyze_by_date(self):
        """Analyze signals by date"""
        date_columns = [col for col in self.current_data.columns if 'date' in col.lower() or 'Date' in col]
        
        if not date_columns:
            QMessageBox.warning(self, "No Date Column", "No date column found in data.")
            return
        
        date_col = date_columns[0]
        analysis_result = f"""Date Analysis:
Total Records: {len(self.current_data)}
Date Column: {date_col}

Date Range:
{self.current_data[date_col].min()} to {self.current_data[date_col].max()}

Signals by Date:
{self.current_data[date_col].value_counts().head(10).to_string()}

Recent Signals (Last 10):
{self.current_data.sort_values(date_col, ascending=False).head(10)[['Stock', date_col]].to_string()}
"""
        
        self.analysis_output.setPlainText(analysis_result)

    def analyze_rsi(self):
        """Analyze RSI patterns"""
        rsi_columns = [col for col in self.current_data.columns if 'rsi' in col.lower() or 'RSI' in col]
        
        if not rsi_columns:
            QMessageBox.warning(self, "No RSI Column", "No RSI column found in data.")
            return
        
        rsi_col = rsi_columns[0]
        analysis_result = f"""RSI Analysis:
Total Records: {len(self.current_data)}
RSI Column: {rsi_col}

RSI Statistics:
{self.current_data[rsi_col].describe().to_string()}

RSI Distribution:
{self.current_data[rsi_col].value_counts(bins=10).to_string()}

Sample RSI Data:
{self.current_data[['Stock', rsi_col]].head(10).to_string()}
"""
        
        self.analysis_output.setPlainText(analysis_result)

    def export_analysis(self):
        """Export the current analysis and data"""
        try:
            file, _ = QFileDialog.getSaveFileName(
                self, "Export Analysis", "signal_analysis.xlsx", "Excel Files (*.xlsx)"
            )
            if file:
                with pd.ExcelWriter(file) as writer:
                    # Export current data
                    if self.current_data is not None:
                        self.current_data.to_excel(writer, sheet_name='Current_Data', index=False)
                    
                    # Export all data
                    if self.buy_data is not None:
                        self.buy_data.to_excel(writer, sheet_name='Buy_Signals', index=False)
                    if self.sell_data is not None:
                        self.sell_data.to_excel(writer, sheet_name='Sell_Signals', index=False)
                    if self.neutral_data is not None:
                        self.neutral_data.to_excel(writer, sheet_name='Neutral_Signals', index=False)
                    
                    # Export statistics
                    stats_data = {
                        'Metric': ['Buy Signals', 'Sell Signals', 'Neutral Signals', 'Total'],
                        'Count': [
                            len(self.buy_data) if self.buy_data is not None else 0,
                            len(self.sell_data) if self.sell_data is not None else 0,
                            len(self.neutral_data) if self.neutral_data is not None else 0,
                            (len(self.buy_data) if self.buy_data is not None else 0) +
                            (len(self.sell_data) if self.sell_data is not None else 0) +
                            (len(self.neutral_data) if self.neutral_data is not None else 0)
                        ]
                    }
                    pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)
                
                QMessageBox.information(self, "Success", f"Analysis exported to {file}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export analysis: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KMI30/100 Weekly RSI Stock Analysis Tool (GUI)")
        self.setWindowIcon(QIcon())
        self.resize(1400, 1000)
        self.setMinimumSize(1200, 800)
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Create toolbar for theme selection
        self.create_toolbar()
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.analysis_tab = AnalysisTab(parent=self)
        self.logs_tab = LogsTab()
        self.telegram_tab = TelegramTab()
        self.signal_logic_tab = SignalLogicTab()
        self.database_analysis_tab = DatabaseAnalysisTab()
        self.psx_announcements_tab = PSXAnnouncementsTab()
        self.tabs.addTab(self.analysis_tab, "📊 Analysis")
        self.tabs.addTab(self.logs_tab, "📝 Logs")
        self.tabs.addTab(self.telegram_tab, "📱 Telegram & Notifications")
        self.tabs.addTab(self.signal_logic_tab, "⚙️ Signal Logic")
        self.tabs.addTab(self.database_analysis_tab, "🗄️ Database Analysis")
        self.tabs.addTab(self.psx_announcements_tab, "📢 PSX Announcements")
        
        # Status bar
        self.status = self.statusBar()
        self.status_label = QLabel("Ready")
        self.status.addPermanentWidget(self.status_label)
        
        # Progress bar in status bar
        self.status_progress = QProgressBar()
        self.status_progress.setMaximumWidth(200)
        self.status.addPermanentWidget(self.status_progress)
        self.status_progress.setVisible(False)
        
        # Apply default light theme
        self.apply_theme("light")

    def create_toolbar(self):
        """Create toolbar with theme selection"""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        # Theme selection
        toolbar.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "Blue", "Green", "Purple"])
        self.theme_combo.setCurrentText("Light")  # Set light as default
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        toolbar.addWidget(self.theme_combo)
        
        toolbar.addSeparator()
        
        # Add some spacing
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

    def on_theme_changed(self, theme_name):
        """Handle theme selection change"""
        self.apply_theme(theme_name.lower())

    def apply_theme(self, theme):
        """Apply the selected theme to the application"""
        if theme == "light":
            self.apply_light_theme()
        elif theme == "dark":
            self.apply_dark_theme()
        elif theme == "blue":
            self.apply_blue_theme()
        elif theme == "green":
            self.apply_green_theme()
        elif theme == "purple":
            self.apply_purple_theme()
        else:
            self.apply_light_theme()  # Default to light theme

    def apply_light_theme(self):
        """Apply light theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                color: #333333;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333333;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #2196F3;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLineEdit, QTextEdit, QComboBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 6px;
                background-color: #ffffff;
                color: #333333;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 2px solid #2196F3;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4e5462;
                border-radius: 3px;
            }
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: #ffffff;
                alternate-background-color: #f9f9f9;
                color: #333333;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                color: #333333;
                padding: 6px;
                border: 1px solid #e0e0e0;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background-color: #e3f2fd;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #90caf9;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #64b5f6;
            }
            QStatusBar {
                background-color: #e3f2fd;
                color: #1565c0;
            }
            QToolBar {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                spacing: 3px;
            }
        """)

    def apply_dark_theme(self):
        """Apply dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d323b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #4a4a4a;
                background-color: #2d323b;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2d323b;
                color: #64b5f6;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #4a4a4a;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #2d323b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
            QLineEdit, QTextEdit, QComboBox {
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 6px;
                background-color: #3a3a3a;
                color: #ffffff;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 2px solid #64b5f6;
            }
            QProgressBar {
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                text-align: center;
                background-color: #3a3a3a;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 3px;
            }
            QTableWidget {
                gridline-color: #4a4a4a;
                background-color: #2d323b;
                alternate-background-color: #3a3a3a;
                color: #ffffff;
            }
            QTableWidget::item:selected {
                background-color: #1976d2;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #3a3a3a;
                color: #ffffff;
                padding: 6px;
                border: 1px solid #4a4a4a;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background-color: #3a3a3a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #888888;
            }
            QStatusBar {
                background-color: #2d323b;
                color: #ffffff;
            }
            QToolBar {
                background-color: #2d323b;
                border: 1px solid #4a4a4a;
                spacing: 3px;
            }
        """)

    def apply_blue_theme(self):
        """Apply blue theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e3f2fd;
                color: #1565c0;
            }
            QTabWidget::pane {
                border: 1px solid #90caf9;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #bbdefb;
                color: #1565c0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #1976d2;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #e3f2fd;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #90caf9;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #1565c0;
            }
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #bbdefb;
                color: #90caf9;
            }
            QLineEdit, QTextEdit, QComboBox {
                border: 1px solid #90caf9;
                border-radius: 4px;
                padding: 6px;
                background-color: #ffffff;
                color: #1565c0;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 2px solid #2196f3;
            }
            QProgressBar {
                border: 1px solid #90caf9;
                border-radius: 4px;
                text-align: center;
                background-color: #e3f2fd;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
                border-radius: 3px;
            }
            QTableWidget {
                gridline-color: #e3f2fd;
                background-color: #ffffff;
                alternate-background-color: #f3f8ff;
                color: #1565c0;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #1565c0;
            }
            QHeaderView::section {
                background-color: #bbdefb;
                color: #1565c0;
                padding: 6px;
                border: 1px solid #90caf9;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background-color: #e3f2fd;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #90caf9;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #64b5f6;
            }
            QStatusBar {
                background-color: #e3f2fd;
                color: #1565c0;
            }
            QToolBar {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                spacing: 3px;
            }
        """)

    def apply_green_theme(self):
        """Apply green theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e8f5e8;
                color: #2e7d32;
            }
            QTabWidget::pane {
                border: 1px solid #81c784;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #c8e6c9;
                color: #2e7d32;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #388e3c;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #e8f5e8;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #81c784;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2e7d32;
            }
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:pressed {
                background-color: #2e7d32;
            }
            QPushButton:disabled {
                background-color: #c8e6c9;
                color: #81c784;
            }
            QLineEdit, QTextEdit, QComboBox {
                border: 1px solid #81c784;
                border-radius: 4px;
                padding: 6px;
                background-color: #ffffff;
                color: #2e7d32;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 2px solid #4caf50;
            }
            QProgressBar {
                border: 1px solid #81c784;
                border-radius: 4px;
                text-align: center;
                background-color: #e8f5e8;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 3px;
            }
            QTableWidget {
                gridline-color: #e8f5e8;
                background-color: #ffffff;
                alternate-background-color: #f1f8e9;
                color: #2e7d32;
            }
            QTableWidget::item:selected {
                background-color: #e8f5e8;
                color: #2e7d32;
            }
            QHeaderView::section {
                background-color: #c8e6c9;
                color: #2e7d32;
                padding: 6px;
                border: 1px solid #81c784;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background-color: #e8f5e8;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #81c784;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #66bb6a;
            }
            QStatusBar {
                background-color: #e8f5e8;
                color: #2e7d32;
            }
            QToolBar {
                background-color: #e8f5e8;
                border: 1px solid #81c784;
                spacing: 3px;
            }
        """)

    def apply_purple_theme(self):
        """Apply purple theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f3e5f5;
                color: #6a1b9a;
            }
            QTabWidget::pane {
                border: 1px solid #ba68c8;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e1bee7;
                color: #6a1b9a;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #8e24aa;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #f3e5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ba68c8;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #6a1b9a;
            }
            QPushButton {
                background-color: #9c27b0;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8e24aa;
            }
            QPushButton:pressed {
                background-color: #6a1b9a;
            }
            QPushButton:disabled {
                background-color: #e1bee7;
                color: #ba68c8;
            }
            QLineEdit, QTextEdit, QComboBox {
                border: 1px solid #ba68c8;
                border-radius: 4px;
                padding: 6px;
                background-color: #ffffff;
                color: #6a1b9a;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 2px solid #9c27b0;
            }
            QProgressBar {
                border: 1px solid #ba68c8;
                border-radius: 4px;
                text-align: center;
                background-color: #f3e5f5;
            }
            QProgressBar::chunk {
                background-color: #9c27b0;
                border-radius: 3px;
            }
            QTableWidget {
                gridline-color: #f3e5f5;
                background-color: #ffffff;
                alternate-background-color: #faf5ff;
                color: #6a1b9a;
            }
            QTableWidget::item:selected {
                background-color: #f3e5f5;
                color: #6a1b9a;
            }
            QHeaderView::section {
                background-color: #e1bee7;
                color: #6a1b9a;
                padding: 6px;
                border: 1px solid #ba68c8;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background-color: #f3e5f5;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #ba68c8;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #ab47bc;
            }
            QStatusBar {
                background-color: #f3e5f5;
                color: #6a1b9a;
            }
            QToolBar {
                background-color: #f3e5f5;
                border: 1px solid #ba68c8;
                spacing: 3px;
            }
        """)

class PSXAnnouncementsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create scroll area for the entire tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Database Connection Group ---
        db_group = QGroupBox("Database Connection")
        db_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        db_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f0f8ff;
            }
            QGroupBox::title {
                color: #1976D2;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        db_layout = QHBoxLayout()
        
        db_layout.addWidget(QLabel("Database:"))
        self.db_path_label = QLabel("data/databases/production/PSXCompanyAnnouncements.db")
        self.db_path_label.setStyleSheet("background: #e3f2fd; padding: 8px; border-radius: 6px; color: #1565c0; font-weight: bold;")
        db_layout.addWidget(self.db_path_label, 1)
        
        self.refresh_btn = QPushButton("🔄 Refresh Data")
        self.refresh_btn.setMinimumHeight(36)
        self.refresh_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.refresh_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.refresh_btn.clicked.connect(self.load_announcements_data)
        db_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("📊 Export Announcements")
        self.export_btn.setMinimumHeight(36)
        self.export_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.export_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.export_btn.clicked.connect(self.export_announcements)
        db_layout.addWidget(self.export_btn)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        # --- Statistics Group ---
        stats_group = QGroupBox("Announcement Statistics")
        stats_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        stats_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4caf50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f1f8e9;
            }
            QGroupBox::title {
                color: #388e3c;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        stats_layout = QHBoxLayout()
        
        self.total_count_label = QLabel("Total Announcements: 0")
        self.total_count_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 13px; padding: 8px; background: #e3f2fd; border-radius: 6px;")
        stats_layout.addWidget(self.total_count_label)
        
        self.today_count_label = QLabel("Today: 0")
        self.today_count_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 13px; padding: 8px; background: #e8f5e8; border-radius: 6px;")
        stats_layout.addWidget(self.today_count_label)
        
        self.week_count_label = QLabel("This Week: 0")
        self.week_count_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 13px; padding: 8px; background: #fff8e1; border-radius: 6px;")
        stats_layout.addWidget(self.week_count_label)
        
        self.month_count_label = QLabel("This Month: 0")
        self.month_count_label.setStyleSheet("color: #9C27B0; font-weight: bold; font-size: 13px; padding: 8px; background: #f3e5f5; border-radius: 6px;")
        stats_layout.addWidget(self.month_count_label)
        
        stats_layout.addStretch()
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # --- Filter Controls Group ---
        filter_group = QGroupBox("Filter & Search Controls")
        filter_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        filter_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ff9800;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fff8f0;
            }
            QGroupBox::title {
                color: #f57c00;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Company:"))
        self.company_combo = QComboBox()
        self.company_combo.addItem("All Companies")
        self.company_combo.setMinimumHeight(32)
        self.company_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.company_combo)
        
        filter_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("All Categories")
        self.category_combo.setMinimumHeight(32)
        self.category_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.category_combo)
        
        filter_layout.addWidget(QLabel("Date Range:"))
        self.date_combo = QComboBox()
        self.date_combo.addItems(["All Time", "Today", "This Week", "This Month", "Last 3 Months", "Last 6 Months", "Last Year"])
        self.date_combo.setMinimumHeight(32)
        self.date_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.date_combo)
        
        filter_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in title or content...")
        self.search_input.setMinimumHeight(32)
        self.search_input.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.search_input, 1)
        
        self.clear_filters_btn = QPushButton("Clear Filters")
        self.clear_filters_btn.setMinimumHeight(32)
        self.clear_filters_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.clear_filters_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.clear_filters_btn.clicked.connect(self.clear_filters)
        filter_layout.addWidget(self.clear_filters_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # --- Announcements Table Group ---
        table_group = QGroupBox("Announcements Data")
        table_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        table_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #607d8b;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f5f7fa;
            }
            QGroupBox::title {
                color: #455a64;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        table_layout = QVBoxLayout()
        
        self.announcements_table = QTableWidget()
        self.announcements_table.setColumnCount(6)
        self.announcements_table.setHorizontalHeaderLabels([
            "Date", "Company", "Category", "Title", "Content", "Link"
        ])
        self.announcements_table.horizontalHeader().setStretchLastSection(True)
        self.announcements_table.setAlternatingRowColors(True)
        self.announcements_table.setMaximumHeight(400)
        self.announcements_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: #ffffff;
                alternate-background-color: #f9f9f9;
                color: #333333;
                border-radius: 8px;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                color: #333333;
                padding: 8px;
                border: 1px solid #e0e0e0;
                font-weight: bold;
                border-radius: 4px;
            }
        """)
        self.announcements_table.cellDoubleClicked.connect(self.open_announcement_link)
        table_layout.addWidget(self.announcements_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # --- Analysis Group ---
        analysis_group = QGroupBox("Announcement Analysis")
        analysis_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        analysis_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #9c27b0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #faf5ff;
            }
            QGroupBox::title {
                color: #8e24aa;
                font-size: 13px;
                left: 10px;
                padding: 0 6px 0 6px;
            }
        """)
        analysis_layout = QVBoxLayout()
        
        analysis_controls = QHBoxLayout()
        analysis_controls.addWidget(QLabel("Analysis Type:"))
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Company Distribution", 
            "Category Analysis", 
            "Date Distribution", 
            "Trend Analysis",
            "Keyword Analysis"
        ])
        self.analysis_type_combo.setMinimumHeight(32)
        self.analysis_type_combo.currentIndexChanged.connect(self.run_analysis)
        analysis_controls.addWidget(self.analysis_type_combo)
        
        self.run_analysis_btn = QPushButton("📈 Run Analysis")
        self.run_analysis_btn.setMinimumHeight(36)
        self.run_analysis_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.run_analysis_btn.setStyleSheet("padding: 6px 16px; border-radius: 6px;")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        analysis_controls.addWidget(self.run_analysis_btn)
        
        analysis_controls.addStretch()
        analysis_layout.addLayout(analysis_controls)
        
        self.analysis_output = QTextEdit()
        self.analysis_output.setReadOnly(True)
        self.analysis_output.setFont(QFont("Consolas", 9))
        self.analysis_output.setMaximumHeight(200)
        self.analysis_output.setStyleSheet("""
            background: #f3e5f5;
            color: #4a148c;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #ba68c8;
        """)
        self.analysis_output.setPlaceholderText("Analysis results will appear here...")
        analysis_layout.addWidget(self.analysis_output)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)
        
        # Set the scroll area as the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        # Initialize data storage
        self.announcements_data = None
        self.filtered_data = None
        self.current_table_name = None
        
        # Load initial data
        self.load_announcements_data()

    def load_announcements_data(self):
        """Load announcements data from the database"""
        try:
            db_path = "data/databases/production/PSXCompanyAnnouncements.db"
            
            # Check if database exists
            if not os.path.exists(db_path):
                QMessageBox.warning(self, "Database Not Found", 
                                  f"Database file not found: {db_path}\n\nPlease ensure the database file exists in the specified location.")
                self.set_empty_state()
                return
            
            # Connect to database and get table names
            conn = None
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                if not tables:
                    QMessageBox.warning(self, "No Tables", "No tables found in the announcements database.")
                    self.set_empty_state()
                    return
                
                # Find the best table to use
                table_name = self.find_best_table(tables)
                self.current_table_name = table_name
                
                # Check table structure
                column_info = self.get_table_structure(conn, table_name)
                if not column_info:
                    QMessageBox.warning(self, "Table Error", f"Could not read structure of table '{table_name}'")
                    self.set_empty_state()
                    return
                
                # Load data
                self.announcements_data = self.load_table_data(conn, table_name)
                
                if self.announcements_data is None or self.announcements_data.empty:
                    QMessageBox.information(self, "No Data", f"No data found in table '{table_name}'")
                    self.set_empty_state()
                    return
                
                # Clean and prepare data
                self.clean_announcements_data()
                
                # Update UI
                self.update_statistics()
                self.update_filter_options()
                self.update_table()
                
                QMessageBox.information(self, "Success", 
                                      f"Loaded {len(self.announcements_data)} announcements from table '{table_name}'")
                
            except sqlite3.Error as e:
                QMessageBox.critical(self, "Database Error", f"SQLite error: {str(e)}")
                self.set_empty_state()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load announcements data: {str(e)}")
                self.set_empty_state()
            finally:
                if conn:
                    conn.close()
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
            self.set_empty_state()

    def set_empty_state(self):
        """Set the tab to empty state when no data is available"""
        self.announcements_data = pd.DataFrame()
        self.filtered_data = None
        self.current_table_name = None
        
        # Update UI elements
        self.total_count_label.setText("Total Announcements: 0")
        self.today_count_label.setText("Today: 0")
        self.week_count_label.setText("This Week: 0")
        self.month_count_label.setText("This Month: 0")
        
        # Clear filter options
        self.company_combo.clear()
        self.company_combo.addItem("All Companies")
        self.category_combo.clear()
        self.category_combo.addItem("All Categories")
        
        # Clear table
        self.announcements_table.setRowCount(0)
        
        # Clear analysis output
        self.analysis_output.clear()

    def find_best_table(self, tables):
        """Find the best table to use from available tables"""
        table_names = [table[0] for table in tables]
        
        # Look for common announcement table names
        preferred_names = ['announcements', 'company_announcements', 'psx_announcements', 'announcement']
        for name in preferred_names:
            for table_name in table_names:
                if name in table_name.lower():
                    return table_name
        
        # If no preferred name found, use the first table
        return table_names[0]

    def get_table_structure(self, conn, table_name):
        """Get the structure of a table"""
        try:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            return columns
        except Exception as e:
            print(f"Error getting table structure: {str(e)}")
            return None

    def load_table_data(self, conn, table_name):
        """Load data from a table with error handling"""
        try:
            # Try with ORDER BY first
            query = f"SELECT * FROM {table_name} ORDER BY Date DESC"
            return pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error with ORDER BY, trying without: {str(e)}")
            try:
                # Try without ORDER BY
                query = f"SELECT * FROM {table_name}"
                return pd.read_sql(query, conn)
            except Exception as e2:
                print(f"Error loading table data: {str(e2)}")
                return None

    def clean_announcements_data(self):
        """Clean and prepare the announcements data"""
        if self.announcements_data is None or self.announcements_data.empty:
            return
        
        try:
            # Handle missing columns by creating them with default values
            default_columns = {
                'Date': pd.Timestamp.now(),
                'Company': 'Unknown',
                'Category': 'General',
                'Title': 'No Title',
                'Content': '',
                'Link': ''
            }
            
            for col, default_value in default_columns.items():
                if col not in self.announcements_data.columns:
                    self.announcements_data[col] = default_value
            
            # Clean date column
            if 'Date' in self.announcements_data.columns:
                try:
                    self.announcements_data['Date'] = pd.to_datetime(self.announcements_data['Date'], errors='coerce')
                    # Fill NaT values with current date
                    self.announcements_data['Date'] = self.announcements_data['Date'].fillna(pd.Timestamp.now())
                except Exception as e:
                    print(f"Error cleaning dates: {str(e)}")
                    self.announcements_data['Date'] = pd.Timestamp.now()
            
            # Clean string columns
            string_columns = ['Company', 'Category', 'Title', 'Content', 'Link']
            for col in string_columns:
                if col in self.announcements_data.columns:
                    try:
                        self.announcements_data[col] = self.announcements_data[col].astype(str).fillna('')
                        # Remove problematic characters
                        self.announcements_data[col] = self.announcements_data[col].str.replace('\x00', '', regex=False)
                        # Replace 'nan' strings with empty string
                        self.announcements_data[col] = self.announcements_data[col].replace('nan', '')
                    except Exception as e:
                        print(f"Error cleaning column {col}: {str(e)}")
                        self.announcements_data[col] = ''
            
            # Remove completely empty rows
            self.announcements_data = self.announcements_data.dropna(how='all')
            
        except Exception as e:
            print(f"Error cleaning data: {str(e)}")

    def update_statistics(self):
        """Update statistics display"""
        try:
            if self.announcements_data is None or self.announcements_data.empty:
                self.total_count_label.setText("Total Announcements: 0")
                self.today_count_label.setText("Today: 0")
                self.week_count_label.setText("This Week: 0")
                self.month_count_label.setText("This Month: 0")
                return
            
            total_count = len(self.announcements_data)
            self.total_count_label.setText(f"Total Announcements: {total_count}")
            
            # Calculate date-based statistics
            if 'Date' in self.announcements_data.columns:
                try:
                    dates = pd.to_datetime(self.announcements_data['Date'], errors='coerce')
                    valid_dates = dates.dropna()
                    
                    if len(valid_dates) > 0:
                        today = pd.Timestamp.now().date()
                        
                        today_count = len(valid_dates[valid_dates.dt.date == today])
                        self.today_count_label.setText(f"Today: {today_count}")
                        
                        week_start = today - pd.Timedelta(days=today.weekday())
                        week_count = len(valid_dates[valid_dates.dt.date >= week_start])
                        self.week_count_label.setText(f"This Week: {week_count}")
                        
                        month_start = today.replace(day=1)
                        month_count = len(valid_dates[valid_dates.dt.date >= month_start])
                        self.month_count_label.setText(f"This Month: {month_count}")
                    else:
                        self.today_count_label.setText("Today: 0")
                        self.week_count_label.setText("This Week: 0")
                        self.month_count_label.setText("This Month: 0")
                except Exception as e:
                    print(f"Error calculating date statistics: {str(e)}")
                    self.today_count_label.setText("Today: N/A")
                    self.week_count_label.setText("This Week: N/A")
                    self.month_count_label.setText("This Month: N/A")
            else:
                self.today_count_label.setText("Today: N/A")
                self.week_count_label.setText("This Week: N/A")
                self.month_count_label.setText("This Month: N/A")
        except Exception as e:
            print(f"Error updating statistics: {str(e)}")
            self.total_count_label.setText("Total Announcements: 0")
            self.today_count_label.setText("Today: N/A")
            self.week_count_label.setText("This Week: N/A")
            self.month_count_label.setText("This Month: N/A")

    def update_filter_options(self):
        """Update filter dropdown options"""
        try:
            if self.announcements_data is None or self.announcements_data.empty:
                return
            
            # Update company filter
            if 'Company' in self.announcements_data.columns:
                try:
                    companies = self.announcements_data['Company'].dropna().unique()
                    companies = [str(c).strip() for c in companies if str(c).strip() and str(c).strip().lower() != 'nan']
                    companies = sorted(list(set(companies)))  # Remove duplicates
                    self.company_combo.clear()
                    self.company_combo.addItem("All Companies")
                    self.company_combo.addItems(companies)
                except Exception as e:
                    print(f"Error updating company filter: {str(e)}")
                    self.company_combo.clear()
                    self.company_combo.addItem("All Companies")
            
            # Update category filter
            if 'Category' in self.announcements_data.columns:
                try:
                    categories = self.announcements_data['Category'].dropna().unique()
                    categories = [str(c).strip() for c in categories if str(c).strip() and str(c).strip().lower() != 'nan']
                    categories = sorted(list(set(categories)))  # Remove duplicates
                    self.category_combo.clear()
                    self.category_combo.addItem("All Categories")
                    self.category_combo.addItems(categories)
                except Exception as e:
                    print(f"Error updating category filter: {str(e)}")
                    self.category_combo.clear()
                    self.category_combo.addItem("All Categories")
        except Exception as e:
            print(f"Error updating filter options: {str(e)}")

    def apply_filters(self):
        """Apply filters to the data"""
        try:
            if self.announcements_data is None or self.announcements_data.empty:
                return
            
            # Start with all data
            filtered_data = self.announcements_data.copy()
            
            # Apply company filter
            if self.company_combo.currentText() != "All Companies":
                filtered_data = filtered_data[filtered_data['Company'] == self.company_combo.currentText()]
            
            # Apply category filter
            if self.category_combo.currentText() != "All Categories":
                filtered_data = filtered_data[filtered_data['Category'] == self.category_combo.currentText()]
            
            # Apply date filter
            if self.date_combo.currentText() != "All Time":
                try:
                    if 'Date' in filtered_data.columns:
                        dates = pd.to_datetime(filtered_data['Date'], errors='coerce')
                        valid_dates = dates.dropna()
                        
                        if len(valid_dates) > 0:
                            today = pd.Timestamp.now().date()
                            
                            if self.date_combo.currentText() == "Today":
                                filtered_data = filtered_data[dates.dt.date == today]
                            elif self.date_combo.currentText() == "This Week":
                                week_start = today - pd.Timedelta(days=today.weekday())
                                filtered_data = filtered_data[dates.dt.date >= week_start]
                            elif self.date_combo.currentText() == "This Month":
                                month_start = today.replace(day=1)
                                filtered_data = filtered_data[dates.dt.date >= month_start]
                            elif self.date_combo.currentText() == "Last 3 Months":
                                three_months_ago = today - pd.Timedelta(days=90)
                                filtered_data = filtered_data[dates.dt.date >= three_months_ago]
                            elif self.date_combo.currentText() == "Last 6 Months":
                                six_months_ago = today - pd.Timedelta(days=180)
                                filtered_data = filtered_data[dates.dt.date >= six_months_ago]
                            elif self.date_combo.currentText() == "Last Year":
                                one_year_ago = today - pd.Timedelta(days=365)
                                filtered_data = filtered_data[dates.dt.date >= one_year_ago]
                except Exception as e:
                    print(f"Error applying date filter: {str(e)}")
            
            # Apply search filter
            search_text = self.search_input.text().strip().lower()
            if search_text:
                try:
                    search_mask = pd.Series([False] * len(filtered_data))
                    for col in ['Title', 'Content']:
                        if col in filtered_data.columns:
                            col_mask = filtered_data[col].astype(str).str.lower().str.contains(search_text, na=False)
                            search_mask |= col_mask
                    filtered_data = filtered_data[search_mask]
                except Exception as e:
                    print(f"Error applying search filter: {str(e)}")
            
            self.filtered_data = filtered_data
            self.update_table()
        except Exception as e:
            print(f"Error applying filters: {str(e)}")
            self.filtered_data = self.announcements_data
            self.update_table()

    def clear_filters(self):
        """Clear all filters"""
        try:
            self.company_combo.setCurrentText("All Companies")
            self.category_combo.setCurrentText("All Categories")
            self.date_combo.setCurrentText("All Time")
            self.search_input.clear()
            self.filtered_data = self.announcements_data
            self.update_table()
        except Exception as e:
            print(f"Error clearing filters: {str(e)}")

    def update_table(self):
        """Update the announcements table"""
        try:
            data_to_show = self.filtered_data if self.filtered_data is not None else self.announcements_data
            
            if data_to_show is None or data_to_show.empty:
                self.announcements_table.setRowCount(0)
                return
            
            # Limit to first 1000 rows for performance
            display_data = data_to_show.head(1000)
            
            self.announcements_table.setRowCount(len(display_data))
            
            for row_idx, (_, row) in enumerate(display_data.iterrows()):
                # Date
                if 'Date' in row:
                    try:
                        date_value = row['Date']
                        if pd.isna(date_value):
                            date_text = "N/A"
                        else:
                            date_text = str(date_value)
                        date_item = QTableWidgetItem(date_text)
                        self.announcements_table.setItem(row_idx, 0, date_item)
                    except Exception:
                        self.announcements_table.setItem(row_idx, 0, QTableWidgetItem("N/A"))
                
                # Company
                if 'Company' in row:
                    try:
                        company_text = str(row['Company']) if not pd.isna(row['Company']) else "Unknown"
                        company_item = QTableWidgetItem(company_text)
                        self.announcements_table.setItem(row_idx, 1, company_item)
                    except Exception:
                        self.announcements_table.setItem(row_idx, 1, QTableWidgetItem("Unknown"))
                
                # Category
                if 'Category' in row:
                    try:
                        category_text = str(row['Category']) if not pd.isna(row['Category']) else "General"
                        category_item = QTableWidgetItem(category_text)
                        self.announcements_table.setItem(row_idx, 2, category_item)
                    except Exception:
                        self.announcements_table.setItem(row_idx, 2, QTableWidgetItem("General"))
                
                # Title
                if 'Title' in row:
                    try:
                        title_text = str(row['Title']) if not pd.isna(row['Title']) else "No Title"
                        if len(title_text) > 100:
                            title_text = title_text[:97] + "..."
                        title_item = QTableWidgetItem(title_text)
                        self.announcements_table.setItem(row_idx, 3, title_item)
                    except Exception:
                        self.announcements_table.setItem(row_idx, 3, QTableWidgetItem("No Title"))
                
                # Content
                if 'Content' in row:
                    try:
                        content_text = str(row['Content']) if not pd.isna(row['Content']) else ""
                        if len(content_text) > 150:
                            content_text = content_text[:147] + "..."
                        content_item = QTableWidgetItem(content_text)
                        self.announcements_table.setItem(row_idx, 4, content_item)
                    except Exception:
                        self.announcements_table.setItem(row_idx, 4, QTableWidgetItem(""))
                
                # Link (support both 'Link' and 'URL' columns)
                link_text = ""
                if 'Link' in row and not pd.isna(row['Link']) and row['Link']:
                    link_text = str(row['Link'])
                elif 'URL' in row and not pd.isna(row['URL']) and row['URL']:
                    link_text = str(row['URL'])
                link_item = QTableWidgetItem(link_text)
                if link_text:
                    link_item.setForeground(QBrush(QColor(33, 150, 243)))  # Blue color
                    font = link_item.font()
                    font.setUnderline(True)
                    link_item.setFont(font)
                    link_item.setToolTip("Double-click to open link")
                self.announcements_table.setItem(row_idx, 5, link_item)
        except Exception as e:
            print(f"Error updating table: {str(e)}")
            self.announcements_table.setRowCount(0)

    def open_announcement_link(self, row, column):
        """Open announcement link in default browser"""
        if column == 5:  # Link column
            try:
                link_item = self.announcements_table.item(row, column)
                if link_item and link_item.text() and link_item.text() != "nan" and link_item.text().strip():
                    import webbrowser
                    webbrowser.open(link_item.text().strip())
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open link: {str(e)}")

    def run_analysis(self):
        """Run analysis on the announcements data"""
        if self.announcements_data is None or self.announcements_data.empty:
            QMessageBox.warning(self, "No Data", "Please load announcements data first.")
            return
        
        analysis_type = self.analysis_type_combo.currentText()
        
        try:
            if analysis_type == "Company Distribution":
                self.analyze_company_distribution()
            elif analysis_type == "Category Analysis":
                self.analyze_categories()
            elif analysis_type == "Date Distribution":
                self.analyze_date_distribution()
            elif analysis_type == "Trend Analysis":
                self.analyze_trends()
            elif analysis_type == "Keyword Analysis":
                self.analyze_keywords()
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.analysis_output.setPlainText(error_msg)
            print(f"Analysis error: {str(e)}")

    def analyze_company_distribution(self):
        """Analyze announcement distribution by company"""
        try:
            if 'Company' not in self.announcements_data.columns:
                self.analysis_output.setPlainText("Company column not found in data.")
                return
            
            # Clean company data
            company_data = self.announcements_data['Company'].dropna()
            company_data = company_data[company_data.astype(str).str.strip() != '']
            company_data = company_data[company_data.astype(str).str.lower() != 'nan']
            
            if len(company_data) == 0:
                self.analysis_output.setPlainText("No valid company data found.")
                return
            
            company_counts = company_data.value_counts()
            
            output = f"""Company Distribution Analysis:
Total Companies: {len(company_counts)}
Total Announcements: {len(company_data)}

Top 10 Companies by Announcement Count:
{company_counts.head(10).to_string()}

Bottom 10 Companies by Announcement Count:
{company_counts.tail(10).to_string()}

Statistics:
- Average announcements per company: {company_counts.mean():.2f}
- Median announcements per company: {company_counts.median():.2f}
- Most active company: {company_counts.index[0]} ({company_counts.iloc[0]} announcements)
- Least active company: {company_counts.index[-1]} ({company_counts.iloc[-1]} announcements)
"""
            
            self.analysis_output.setPlainText(output)
        except Exception as e:
            self.analysis_output.setPlainText(f"Company distribution analysis failed: {str(e)}")

    def analyze_categories(self):
        """Analyze announcement categories"""
        try:
            if 'Category' not in self.announcements_data.columns:
                self.analysis_output.setPlainText("Category column not found in data.")
                return
            
            # Clean category data
            category_data = self.announcements_data['Category'].dropna()
            category_data = category_data[category_data.astype(str).str.strip() != '']
            category_data = category_data[category_data.astype(str).str.lower() != 'nan']
            
            if len(category_data) == 0:
                self.analysis_output.setPlainText("No valid category data found.")
                return
            
            category_counts = category_data.value_counts()
            
            output = f"""Category Analysis:
Total Categories: {len(category_counts)}
Total Announcements: {len(category_data)}

Category Distribution:
{category_counts.to_string()}

Most Common Categories:
{category_counts.head(10).to_string()}

Category Statistics:
- Most frequent category: {category_counts.index[0]} ({category_counts.iloc[0]} announcements)
- Least frequent category: {category_counts.index[-1]} ({category_counts.iloc[-1]} announcements)
- Average announcements per category: {category_counts.mean():.2f}
"""
            
            self.analysis_output.setPlainText(output)
        except Exception as e:
            self.analysis_output.setPlainText(f"Category analysis failed: {str(e)}")

    def analyze_date_distribution(self):
        """Analyze announcement distribution by date"""
        try:
            if 'Date' not in self.announcements_data.columns:
                self.analysis_output.setPlainText("Date column not found in data.")
                return
            
            # Clean date data
            dates = pd.to_datetime(self.announcements_data['Date'], errors='coerce')
            valid_dates = dates.dropna()
            
            if len(valid_dates) == 0:
                self.analysis_output.setPlainText("No valid date data found.")
                return
            
            date_counts = valid_dates.dt.date.value_counts().sort_index()
            
            output = f"""Date Distribution Analysis:
Date Range: {date_counts.index[0]} to {date_counts.index[-1]}
Total Days with Announcements: {len(date_counts)}
Total Announcements: {len(valid_dates)}

Recent Activity (Last 10 Days):
{date_counts.tail(10).to_string()}

Most Active Days:
{date_counts.nlargest(10).to_string()}

Statistics:
- Average announcements per day: {date_counts.mean():.2f}
- Most announcements in a day: {date_counts.max()} on {date_counts.idxmax()}
- Days with no announcements: {len(date_counts[date_counts == 0]) if len(date_counts[date_counts == 0]) > 0 else 'None'}
"""
            
            self.analysis_output.setPlainText(output)
        except Exception as e:
            self.analysis_output.setPlainText(f"Date distribution analysis failed: {str(e)}")

    def analyze_trends(self):
        """Analyze announcement trends over time"""
        try:
            if 'Date' not in self.announcements_data.columns:
                self.analysis_output.setPlainText("Date column not found in data.")
                return
            
            # Clean date data
            dates = pd.to_datetime(self.announcements_data['Date'], errors='coerce')
            valid_dates = dates.dropna()
            
            if len(valid_dates) == 0:
                self.analysis_output.setPlainText("No valid date data found.")
                return
            
            # Monthly trends
            monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
            
            # Weekly trends
            weekly_counts = valid_dates.dt.to_period('W').value_counts().sort_index()
            
            if len(monthly_counts) == 0:
                self.analysis_output.setPlainText("No valid monthly data found.")
                return
            
            output = f"""Trend Analysis:
Analysis Period: {monthly_counts.index[0]} to {monthly_counts.index[-1]}

Monthly Trends (Last 12 Months):
{monthly_counts.tail(12).to_string()}

Weekly Trends (Last 8 Weeks):
{weekly_counts.tail(8).to_string()}

Trend Statistics:
- Average announcements per month: {monthly_counts.mean():.2f}
- Average announcements per week: {weekly_counts.mean():.2f}
- Most active month: {monthly_counts.idxmax()} ({monthly_counts.max()} announcements)
- Most active week: {weekly_counts.idxmax()} ({weekly_counts.max()} announcements)

Recent Trend:
- Last month: {monthly_counts.iloc[-1] if len(monthly_counts) > 0 else 'N/A'} announcements
- Previous month: {monthly_counts.iloc[-2] if len(monthly_counts) > 1 else 'N/A'} announcements
"""
            
            self.analysis_output.setPlainText(output)
        except Exception as e:
            self.analysis_output.setPlainText(f"Trend analysis failed: {str(e)}")

    def analyze_keywords(self):
        """Analyze keywords in announcement titles and content"""
        try:
            if 'Title' not in self.announcements_data.columns and 'Content' not in self.announcements_data.columns:
                self.analysis_output.setPlainText("Title or Content columns not found in data.")
                return
            
            import re
            from collections import Counter
            
            # Combine title and content for keyword analysis
            text_data = ""
            if 'Title' in self.announcements_data.columns:
                title_data = self.announcements_data['Title'].dropna()
                title_data = title_data[title_data.astype(str).str.strip() != '']
                title_data = title_data[title_data.astype(str).str.lower() != 'nan']
                text_data += " " + title_data.astype(str).str.cat(sep=" ")
            
            if 'Content' in self.announcements_data.columns:
                content_data = self.announcements_data['Content'].dropna()
                content_data = content_data[content_data.astype(str).str.strip() != '']
                content_data = content_data[content_data.astype(str).str.lower() != 'nan']
                text_data += " " + content_data.astype(str).str.cat(sep=" ")
            
            if not text_data.strip():
                self.analysis_output.setPlainText("No valid text data found for keyword analysis.")
                return
            
            # Extract words (simple approach)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text_data.lower())
            
            if len(words) == 0:
                self.analysis_output.setPlainText("No valid words found for keyword analysis.")
                return
            
            # Remove common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an', 'as', 'from', 'not', 'no', 'yes', 'all', 'any', 'each', 'every', 'some', 'such', 'than', 'too', 'very', 'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose'}
            words = [word for word in words if word not in stop_words]
            
            if len(words) == 0:
                self.analysis_output.setPlainText("No meaningful words found after filtering stop words.")
                return
            
            # Count word frequencies
            word_counts = Counter(words)
            
            output = f"""Keyword Analysis:
Total Words Analyzed: {len(words)}
Unique Words: {len(word_counts)}

Most Common Keywords (Top 20):
"""
            
            for word, count in word_counts.most_common(20):
                output += f"- {word}: {count} occurrences\n"
            
            output += f"""
Keyword Statistics:
- Average word frequency: {sum(word_counts.values()) / len(word_counts):.2f}
- Most frequent word: '{word_counts.most_common(1)[0][0]}' ({word_counts.most_common(1)[0][1]} occurrences)
- Words appearing only once: {sum(1 for count in word_counts.values() if count == 1)}
"""
            
            self.analysis_output.setPlainText(output)
        except Exception as e:
            self.analysis_output.setPlainText(f"Keyword analysis failed: {str(e)}")

    def export_announcements(self):
        """Export announcements data to CSV"""
        if self.announcements_data is None or self.announcements_data.empty:
            QMessageBox.warning(self, "No Data", "Please load announcements data first.")
            return
        
        try:
            file, _ = QFileDialog.getSaveFileName(
                self, "Export Announcements", "psx_announcements.csv", "CSV Files (*.csv)"
            )
            if file:
                # Export filtered data if available, otherwise export all data
                data_to_export = self.filtered_data if self.filtered_data is not None else self.announcements_data
                
                # Clean data before export
                export_data = data_to_export.copy()
                for col in export_data.columns:
                    if export_data[col].dtype == 'object':
                        export_data[col] = export_data[col].astype(str).fillna('')
                
                export_data.to_csv(file, index=False, encoding='utf-8')
                
                QMessageBox.information(self, "Success", f"Announcements exported to {file}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export announcements: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 