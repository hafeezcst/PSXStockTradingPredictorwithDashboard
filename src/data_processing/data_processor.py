from typing import List, Dict, Set
import pandas as pd
from datetime import datetime
import hashlib
import logging
from pathlib import Path
from .config import PSXConfig

class DataProcessor:
    """Handles data processing and transformations for PSX announcements"""
    
    def __init__(self):
        """Initialize data processor"""
        self.company_data: Dict[str, str] = {}
        self.load_company_data()
    
    def load_company_data(self) -> None:
        """Load company data from Excel file"""
        try:
            symbols_file = Path(PSXConfig.PROJECT_ROOT) / "src" / "data_processing" / "psxsymbols.xlsx"
            
            if not symbols_file.exists():
                logging.warning(f"Symbols file not found at {symbols_file}")
                return
                
            # Read the KMI100 sheet
            try:
                df = pd.read_excel(symbols_file, sheet_name='KMI100')
                logging.info(f"Loaded data from KMI100 sheet")
            except Exception as e:
                logging.error(f"Error reading KMI100 sheet: {e}")
                return
            
            # Get column names (case-insensitive)
            columns = [col.upper() for col in df.columns]
            
            # Find symbol and name columns
            symbol_col = df.columns[0]  # Usually first column
            name_col = None
            
            if len(df.columns) > 1:
                name_col_candidates = [col for col in df.columns if any(x in col.upper() for x in ['NAME', 'COMPANY', 'DESC', 'TITLE'])]
                name_col = name_col_candidates[0] if name_col_candidates else df.columns[1]
            
            # Create dictionary
            for _, row in df.iterrows():
                symbol = str(row[symbol_col]).strip().upper()
                if not symbol or pd.isna(symbol):
                    continue
                    
                if name_col and pd.notna(row[name_col]):
                    name = str(row[name_col]).strip()
                    if name.replace('.', '').isdigit():
                        name = f"{symbol} Limited"
                else:
                    name = f"{symbol} Limited"
                
                self.company_data[symbol] = name
                
            logging.info(f"Loaded {len(self.company_data)} companies from Excel file")
            
        except Exception as e:
            logging.error(f"Error loading company data: {e}")
    
    def create_announcement_id(self, date: str, title: str, symbol: str, category: str = '') -> str:
        """
        Create a unique announcement ID
        
        Args:
            date: Announcement date
            title: Announcement title
            symbol: Company symbol
            category: Announcement category
            
        Returns:
            str: Unique announcement ID
        """
        return hashlib.md5(f"{date}_{title}_{symbol}_{category}".encode()).hexdigest()
    
    def parse_date(self, date_str: str) -> str:
        """
        Parse date string into YYYY-MM-DD format
        
        Args:
            date_str: Date string to parse
            
        Returns:
            str: Date in YYYY-MM-DD format
        """
        try:
            date_formats = ['%b %d, %Y', '%B %d, %Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
            
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # Try parsing "Mar 3, 2025" format
            parts = date_str.replace(',', '').split()
            if len(parts) == 3:
                month_dict = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month = month_dict.get(parts[0][:3], 1)
                day = int(parts[1])
                year = int(parts[2])
                return datetime(year, month, day).strftime('%Y-%m-%d')
                
        except Exception as e:
            logging.warning(f"Could not parse date '{date_str}': {e}")
            
        return date_str
    
    def filter_announcements(self, announcements: List[Dict], symbols: List[str] = None) -> List[Dict]:
        """
        Filter announcements by company symbols
        
        Args:
            announcements: List of announcements to filter
            symbols: List of symbols to filter by (None for all)
            
        Returns:
            List[Dict]: Filtered announcements
        """
        if not symbols:
            return announcements
            
        symbols_upper = [s.upper() for s in symbols]
        filtered = [a for a in announcements if a.get('Symbol', '').upper() in symbols_upper]
        
        logging.info(f"Filtered {len(filtered)} announcements for {len(symbols)} companies")
        return filtered
    
    def filter_kmi100_announcements(self, announcements: List[Dict]) -> List[Dict]:
        """
        Filter announcements for KMI100 companies
        
        Args:
            announcements: List of all announcements
            
        Returns:
            List[Dict]: Filtered announcements for KMI100 companies
        """
        all_symbols = list(self.company_data.keys())
        kmi100_symbols = set(all_symbols[:100])
        
        filtered = [a for a in announcements if a.get('Symbol') in kmi100_symbols]
        logging.info(f"Filtered {len(filtered)} announcements for {len(kmi100_symbols)} KMI100 companies")
        return filtered
    
    def save_to_csv(self, announcements: List[Dict], prefix: str = "") -> str:
        """
        Save announcements to CSV file
        
        Args:
            announcements: List of announcements to save
            prefix: Prefix for filename
            
        Returns:
            str: Path to saved CSV file
        """
        try:
            if not announcements:
                logging.info("No announcements to save to CSV")
                return ""
                
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = PSXConfig.CSV_DIR / f"{prefix}psx_announcements_{timestamp}.csv"
            
            # Convert to DataFrame
            df = pd.DataFrame(announcements)
            
            # Ensure columns are in desired order
            column_order = ['Symbol', 'Company', 'Date', 'Time', 'Subject', 'URL', 'Status', 'Category']
            df = df[column_order]
            
            # Save to CSV
            df.to_csv(csv_file, index=False)
            logging.info(f"Saved {len(announcements)} announcements to CSV file: {csv_file}")
            return str(csv_file)
            
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
            return ""
    
    def save_to_excel(self, all_announcements: List[Dict], kmi100_announcements: List[Dict]) -> str:
        """
        Save announcements to Excel file with two sheets
        
        Args:
            all_announcements: List of all announcements
            kmi100_announcements: List of KMI100 announcements
            
        Returns:
            str: Path to saved Excel file
        """
        try:
            excel_file = PSXConfig.EXCEL_DIR / "PSX_Announcements.xlsx"
            
            # Convert to DataFrames
            all_df = pd.DataFrame(all_announcements)
            kmi100_df = pd.DataFrame(kmi100_announcements)
            
            # Define expected columns
            expected_columns = ['Symbol', 'Company', 'Date', 'Time', 'Subject', 'URL', 'Status', 'Category']
            
            # Ensure all expected columns exist in both DataFrames
            for df in [all_df, kmi100_df]:
                if not df.empty:
                    # Add any missing columns with empty values
                    for col in expected_columns:
                        if col not in df.columns:
                            df[col] = ''
                    
                    # Reorder columns to match expected order
                    df = df[expected_columns]
            
            # Create Excel writer
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                all_df.to_excel(writer, sheet_name='PSX_Announcements', index=False)
                kmi100_df.to_excel(writer, sheet_name='KMI100Announcements', index=False)
            
            logging.info(f"Saved {len(all_announcements)} announcements and {len(kmi100_announcements)} KMI100 announcements to Excel file: {excel_file}")
            return str(excel_file)
            
        except Exception as e:
            logging.error(f"Error saving to Excel: {e}")
            return "" 