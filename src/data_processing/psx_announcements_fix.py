import os
import sqlite3
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class PSXAnnouncementsTabFixed(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.announcements_data = None
        self.filtered_data = None
        self.current_table_name = None
        self.setup_ui()
        self.load_announcements_data()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # Database connection section
        db_group = QGroupBox("Database Connection")
        db_layout = QHBoxLayout()
        db_layout.addWidget(QLabel("Database:"))
        self.db_path_label = QLabel("data/databases/production/PSXCompanyAnnouncements.db")
        db_layout.addWidget(self.db_path_label, 1)
        
        self.refresh_btn = QPushButton("🔄 Refresh Data")
        self.refresh_btn.clicked.connect(self.load_announcements_data)
        db_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("📊 Export")
        self.export_btn.clicked.connect(self.export_announcements)
        db_layout.addWidget(self.export_btn)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        # Statistics section
        stats_group = QGroupBox("Statistics")
        stats_layout = QHBoxLayout()
        
        self.total_count_label = QLabel("Total: 0")
        stats_layout.addWidget(self.total_count_label)
        
        self.today_count_label = QLabel("Today: 0")
        stats_layout.addWidget(self.today_count_label)
        
        self.week_count_label = QLabel("This Week: 0")
        stats_layout.addWidget(self.week_count_label)
        
        self.month_count_label = QLabel("This Month: 0")
        stats_layout.addWidget(self.month_count_label)
        
        stats_layout.addStretch()
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Filter section
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Company:"))
        self.company_combo = QComboBox()
        self.company_combo.addItem("All Companies")
        self.company_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.company_combo)
        
        filter_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("All Categories")
        self.category_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.category_combo)
        
        filter_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.search_input, 1)
        
        self.clear_filters_btn = QPushButton("Clear")
        self.clear_filters_btn.clicked.connect(self.clear_filters)
        filter_layout.addWidget(self.clear_filters_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Table
        self.announcements_table = QTableWidget()
        self.announcements_table.setColumnCount(6)
        self.announcements_table.setHorizontalHeaderLabels([
            "Date", "Company", "Category", "Title", "Content", "Link"
        ])
        self.announcements_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.announcements_table)

        # Analysis section
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()
        
        analysis_controls = QHBoxLayout()
        analysis_controls.addWidget(QLabel("Analysis:"))
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Company Distribution", "Category Analysis", "Date Distribution"
        ])
        analysis_controls.addWidget(self.analysis_type_combo)
        
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        analysis_controls.addWidget(self.run_analysis_btn)
        
        analysis_controls.addStretch()
        analysis_layout.addLayout(analysis_controls)
        
        self.analysis_output = QTextEdit()
        self.analysis_output.setReadOnly(True)
        self.analysis_output.setMaximumHeight(150)
        analysis_layout.addWidget(self.analysis_output)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        self.setLayout(layout)

    def load_announcements_data(self):
        """Load announcements data with robust error handling"""
        try:
            db_path = "data/databases/production/PSXCompanyAnnouncements.db"
            
            if not os.path.exists(db_path):
                QMessageBox.warning(self, "Database Not Found", f"Database not found: {db_path}")
                self.set_empty_state()
                return
            
            # Connect and get table names
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            if not tables:
                QMessageBox.warning(self, "No Tables", "No tables found in database")
                self.set_empty_state()
                return
            
            # Find best table
            table_name = self.find_best_table(tables)
            self.current_table_name = table_name
            
            # Load data
            try:
                self.announcements_data = pd.read_sql(f"SELECT * FROM {table_name}", f"sqlite:///{db_path}")
            except Exception as e:
                print(f"Error loading data: {e}")
                QMessageBox.warning(self, "Load Error", f"Could not load data: {str(e)}")
                self.set_empty_state()
                return
            
            if self.announcements_data.empty:
                QMessageBox.information(self, "No Data", "No data found in table")
                self.set_empty_state()
                return
            
            # Clean data
            self.clean_data()
            
            # Update UI
            self.update_statistics()
            self.update_filter_options()
            self.update_table()
            
            QMessageBox.information(self, "Success", f"Loaded {len(self.announcements_data)} announcements")
            
        except Exception as e:
            print(f"Error in load_announcements_data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.set_empty_state()

    def set_empty_state(self):
        """Set empty state when no data"""
        self.announcements_data = pd.DataFrame()
        self.filtered_data = None
        self.total_count_label.setText("Total: 0")
        self.today_count_label.setText("Today: 0")
        self.week_count_label.setText("This Week: 0")
        self.month_count_label.setText("This Month: 0")
        self.company_combo.clear()
        self.company_combo.addItem("All Companies")
        self.category_combo.clear()
        self.category_combo.addItem("All Categories")
        self.announcements_table.setRowCount(0)
        self.analysis_output.clear()

    def find_best_table(self, tables):
        """Find the best table to use"""
        table_names = [t[0] for t in tables]
        preferred = ['announcements', 'company_announcements', 'psx_announcements']
        
        for name in preferred:
            for table_name in table_names:
                if name in table_name.lower():
                    return table_name
        
        return table_names[0]

    def clean_data(self):
        """Clean the loaded data"""
        if self.announcements_data is None or self.announcements_data.empty:
            return
        
        try:
            # Handle missing columns
            columns_needed = ['Date', 'Company', 'Category', 'Title', 'Content', 'Link']
            for col in columns_needed:
                if col not in self.announcements_data.columns:
                    self.announcements_data[col] = ''
            
            # Clean date column
            if 'Date' in self.announcements_data.columns:
                try:
                    self.announcements_data['Date'] = pd.to_datetime(self.announcements_data['Date'], errors='coerce')
                    self.announcements_data['Date'] = self.announcements_data['Date'].fillna(pd.Timestamp.now())
                except:
                    self.announcements_data['Date'] = pd.Timestamp.now()
            
            # Clean string columns
            string_cols = ['Company', 'Category', 'Title', 'Content', 'Link']
            for col in string_cols:
                if col in self.announcements_data.columns:
                    self.announcements_data[col] = self.announcements_data[col].astype(str).fillna('')
                    self.announcements_data[col] = self.announcements_data[col].replace('nan', '')
            
            # Remove empty rows
            self.announcements_data = self.announcements_data.dropna(how='all')
            
        except Exception as e:
            print(f"Error cleaning data: {e}")

    def update_statistics(self):
        """Update statistics display"""
        try:
            if self.announcements_data is None or self.announcements_data.empty:
                self.total_count_label.setText("Total: 0")
                self.today_count_label.setText("Today: 0")
                self.week_count_label.setText("This Week: 0")
                self.month_count_label.setText("This Month: 0")
                return
            
            total = len(self.announcements_data)
            self.total_count_label.setText(f"Total: {total}")
            
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
                except:
                    self.today_count_label.setText("Today: N/A")
                    self.week_count_label.setText("This Week: N/A")
                    self.month_count_label.setText("This Month: N/A")
            else:
                self.today_count_label.setText("Today: N/A")
                self.week_count_label.setText("This Week: N/A")
                self.month_count_label.setText("This Month: N/A")
                
        except Exception as e:
            print(f"Error updating statistics: {e}")

    def update_filter_options(self):
        """Update filter dropdown options"""
        try:
            if self.announcements_data is None or self.announcements_data.empty:
                return
            
            # Update company filter
            if 'Company' in self.announcements_data.columns:
                companies = self.announcements_data['Company'].dropna().unique()
                companies = [str(c).strip() for c in companies if str(c).strip() and str(c).strip().lower() != 'nan']
                companies = sorted(list(set(companies)))
                self.company_combo.clear()
                self.company_combo.addItem("All Companies")
                self.company_combo.addItems(companies)
            
            # Update category filter
            if 'Category' in self.announcements_data.columns:
                categories = self.announcements_data['Category'].dropna().unique()
                categories = [str(c).strip() for c in categories if str(c).strip() and str(c).strip().lower() != 'nan']
                categories = sorted(list(set(categories)))
                self.category_combo.clear()
                self.category_combo.addItem("All Categories")
                self.category_combo.addItems(categories)
                
        except Exception as e:
            print(f"Error updating filter options: {e}")

    def apply_filters(self):
        """Apply filters to the data"""
        try:
            if self.announcements_data is None or self.announcements_data.empty:
                return
            
            filtered_data = self.announcements_data.copy()
            
            # Company filter
            if self.company_combo.currentText() != "All Companies":
                filtered_data = filtered_data[filtered_data['Company'] == self.company_combo.currentText()]
            
            # Category filter
            if self.category_combo.currentText() != "All Categories":
                filtered_data = filtered_data[filtered_data['Category'] == self.category_combo.currentText()]
            
            # Search filter
            search_text = self.search_input.text().strip().lower()
            if search_text:
                search_mask = pd.Series([False] * len(filtered_data))
                for col in ['Title', 'Content']:
                    if col in filtered_data.columns:
                        col_mask = filtered_data[col].astype(str).str.lower().str.contains(search_text, na=False)
                        search_mask |= col_mask
                filtered_data = filtered_data[search_mask]
            
            self.filtered_data = filtered_data
            self.update_table()
            
        except Exception as e:
            print(f"Error applying filters: {e}")
            self.filtered_data = self.announcements_data
            self.update_table()

    def clear_filters(self):
        """Clear all filters"""
        try:
            self.company_combo.setCurrentText("All Companies")
            self.category_combo.setCurrentText("All Categories")
            self.search_input.clear()
            self.filtered_data = self.announcements_data
            self.update_table()
        except Exception as e:
            print(f"Error clearing filters: {e}")

    def update_table(self):
        """Update the announcements table"""
        try:
            data_to_show = self.filtered_data if self.filtered_data is not None else self.announcements_data
            
            if data_to_show is None or data_to_show.empty:
                self.announcements_table.setRowCount(0)
                return
            
            # Limit rows for performance
            display_data = data_to_show.head(500)
            
            self.announcements_table.setRowCount(len(display_data))
            
            for row_idx, (_, row) in enumerate(display_data.iterrows()):
                # Date
                try:
                    date_text = str(row.get('Date', 'N/A')) if not pd.isna(row.get('Date')) else "N/A"
                    self.announcements_table.setItem(row_idx, 0, QTableWidgetItem(date_text))
                except:
                    self.announcements_table.setItem(row_idx, 0, QTableWidgetItem("N/A"))
                
                # Company
                try:
                    company_text = str(row.get('Company', 'Unknown')) if not pd.isna(row.get('Company')) else "Unknown"
                    self.announcements_table.setItem(row_idx, 1, QTableWidgetItem(company_text))
                except:
                    self.announcements_table.setItem(row_idx, 1, QTableWidgetItem("Unknown"))
                
                # Category
                try:
                    category_text = str(row.get('Category', 'General')) if not pd.isna(row.get('Category')) else "General"
                    self.announcements_table.setItem(row_idx, 2, QTableWidgetItem(category_text))
                except:
                    self.announcements_table.setItem(row_idx, 2, QTableWidgetItem("General"))
                
                # Title
                try:
                    title_text = str(row.get('Title', 'No Title')) if not pd.isna(row.get('Title')) else "No Title"
                    if len(title_text) > 50:
                        title_text = title_text[:47] + "..."
                    self.announcements_table.setItem(row_idx, 3, QTableWidgetItem(title_text))
                except:
                    self.announcements_table.setItem(row_idx, 3, QTableWidgetItem("No Title"))
                
                # Content
                try:
                    content_text = str(row.get('Content', '')) if not pd.isna(row.get('Content')) else ""
                    if len(content_text) > 100:
                        content_text = content_text[:97] + "..."
                    self.announcements_table.setItem(row_idx, 4, QTableWidgetItem(content_text))
                except:
                    self.announcements_table.setItem(row_idx, 4, QTableWidgetItem(""))
                
                # Link
                try:
                    link_text = str(row.get('Link', '')) if not pd.isna(row.get('Link')) else ""
                    self.announcements_table.setItem(row_idx, 5, QTableWidgetItem(link_text))
                except:
                    self.announcements_table.setItem(row_idx, 5, QTableWidgetItem(""))
                    
        except Exception as e:
            print(f"Error updating table: {e}")
            self.announcements_table.setRowCount(0)

    def run_analysis(self):
        """Run analysis on the data"""
        if self.announcements_data is None or self.announcements_data.empty:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return
        
        analysis_type = self.analysis_type_combo.currentText()
        
        try:
            if analysis_type == "Company Distribution":
                self.analyze_companies()
            elif analysis_type == "Category Analysis":
                self.analyze_categories()
            elif analysis_type == "Date Distribution":
                self.analyze_dates()
        except Exception as e:
            self.analysis_output.setPlainText(f"Analysis failed: {str(e)}")

    def analyze_companies(self):
        """Analyze company distribution"""
        try:
            if 'Company' not in self.announcements_data.columns:
                self.analysis_output.setPlainText("Company column not found")
                return
            
            company_data = self.announcements_data['Company'].dropna()
            company_data = company_data[company_data.astype(str).str.strip() != '']
            company_data = company_data[company_data.astype(str).str.lower() != 'nan']
            
            if len(company_data) == 0:
                self.analysis_output.setPlainText("No valid company data")
                return
            
            company_counts = company_data.value_counts()
            
            output = f"Company Distribution:\n"
            output += f"Total Companies: {len(company_counts)}\n"
            output += f"Total Announcements: {len(company_data)}\n\n"
            output += f"Top 10 Companies:\n{company_counts.head(10).to_string()}\n"
            
            self.analysis_output.setPlainText(output)
            
        except Exception as e:
            self.analysis_output.setPlainText(f"Company analysis failed: {str(e)}")

    def analyze_categories(self):
        """Analyze categories"""
        try:
            if 'Category' not in self.announcements_data.columns:
                self.analysis_output.setPlainText("Category column not found")
                return
            
            category_data = self.announcements_data['Category'].dropna()
            category_data = category_data[category_data.astype(str).str.strip() != '']
            category_data = category_data[category_data.astype(str).str.lower() != 'nan']
            
            if len(category_data) == 0:
                self.analysis_output.setPlainText("No valid category data")
                return
            
            category_counts = category_data.value_counts()
            
            output = f"Category Analysis:\n"
            output += f"Total Categories: {len(category_counts)}\n"
            output += f"Total Announcements: {len(category_data)}\n\n"
            output += f"Category Distribution:\n{category_counts.to_string()}\n"
            
            self.analysis_output.setPlainText(output)
            
        except Exception as e:
            self.analysis_output.setPlainText(f"Category analysis failed: {str(e)}")

    def analyze_dates(self):
        """Analyze date distribution"""
        try:
            if 'Date' not in self.announcements_data.columns:
                self.analysis_output.setPlainText("Date column not found")
                return
            
            dates = pd.to_datetime(self.announcements_data['Date'], errors='coerce')
            valid_dates = dates.dropna()
            
            if len(valid_dates) == 0:
                self.analysis_output.setPlainText("No valid date data")
                return
            
            date_counts = valid_dates.dt.date.value_counts().sort_index()
            
            output = f"Date Distribution:\n"
            output += f"Date Range: {date_counts.index[0]} to {date_counts.index[-1]}\n"
            output += f"Total Days: {len(date_counts)}\n"
            output += f"Total Announcements: {len(valid_dates)}\n\n"
            output += f"Recent Activity (Last 10 Days):\n{date_counts.tail(10).to_string()}\n"
            
            self.analysis_output.setPlainText(output)
            
        except Exception as e:
            self.analysis_output.setPlainText(f"Date analysis failed: {str(e)}")

    def export_announcements(self):
        """Export data to CSV"""
        if self.announcements_data is None or self.announcements_data.empty:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return
        
        try:
            file, _ = QFileDialog.getSaveFileName(
                self, "Export Announcements", "psx_announcements.csv", "CSV Files (*.csv)"
            )
            if file:
                data_to_export = self.filtered_data if self.filtered_data is not None else self.announcements_data
                data_to_export.to_csv(file, index=False, encoding='utf-8')
                QMessageBox.information(self, "Success", f"Exported to {file}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}") 