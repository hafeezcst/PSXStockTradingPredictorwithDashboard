"""
Stock Status Management Utility
Helps manage stock status, mergers, delistings, and renames for PSX stocks.
"""

import logging
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enhanced_db_manager import EnhancedDatabaseManager, StockStatus
from config_manager import AppConfig

class StockStatusManager:
    """Utility class for managing stock status and changes"""
    
    def __init__(self, db_manager: EnhancedDatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Known problematic stocks (this can be updated based on market knowledge)
        self.known_issues = {
            # Examples of common issues in PSX
            'IGIIL': {'status': StockStatus.DELISTED, 'reason': 'Delisted from PSX', 'date': '2024-01-01'},
            # Add more known issues here
        }
    
    def initialize_known_stock_issues(self):
        """Initialize database with known stock issues"""
        self.logger.info("Initializing known stock issues...")
        
        for symbol, info in self.known_issues.items():
            try:
                status_date = datetime.strptime(info['date'], '%Y-%m-%d').date() if info.get('date') else date.today()
                
                self.db_manager.update_stock_status(
                    symbol=symbol,
                    status=info['status'],
                    reason=info['reason'],
                    last_trading_date=status_date
                )
                
                self.logger.info(f"Updated status for {symbol}: {info['status']}")
                
            except Exception as e:
                self.logger.error(f"Failed to update status for {symbol}: {e}")
    
    def import_stock_changes_from_csv(self, csv_path: str):
        """Import stock changes from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['old_symbol', 'new_symbol', 'change_type', 'change_date']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            for _, row in df.iterrows():
                change_date = datetime.strptime(row['change_date'], '%Y-%m-%d').date()
                
                self.db_manager.record_symbol_change(
                    old_symbol=row['old_symbol'],
                    new_symbol=row['new_symbol'],
                    change_type=row['change_type'],
                    change_date=change_date,
                    notes=row.get('notes', '')
                )
            
            self.logger.info(f"Imported {len(df)} stock changes from {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import stock changes from CSV: {e}")
    
    def export_problematic_stocks_report(self, output_path: str = None):
        """Export problematic stocks report to CSV"""
        if output_path is None:
            output_path = f"problematic_stocks_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            report = self.db_manager.get_problematic_stocks_report()
            
            # Create a consolidated DataFrame
            all_issues = []
            
            for category, stocks in report.items():
                for stock in stocks:
                    issue_info = {
                        'symbol': stock['symbol'],
                        'category': category,
                        'reason': stock.get('reason', ''),
                        'last_trading_date': stock.get('last_trading_date', ''),
                        'merged_into': stock.get('merged_into', ''),
                        'renamed_to': stock.get('renamed_to', ''),
                        'failure_count': stock.get('failure_count', 0)
                    }
                    all_issues.append(issue_info)
            
            df = pd.DataFrame(all_issues)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Exported problematic stocks report to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return None
    
    def suggest_stock_fixes(self) -> Dict[str, List[str]]:
        """Suggest fixes for problematic stocks"""
        suggestions = {
            'manual_investigation_needed': [],
            'use_alternative_symbol': [],
            'mark_as_delisted': [],
            'check_for_merger': []
        }
        
        try:
            report = self.db_manager.get_problematic_stocks_report()
            
            # Analyze frequent failures
            for stock in report.get('frequent_failures', []):
                symbol = stock['symbol']
                failure_count = stock['failure_count']
                
                if failure_count >= 10:
                    suggestions['manual_investigation_needed'].append(
                        f"{symbol}: {failure_count} failures - needs manual investigation"
                    )
                elif failure_count >= 5:
                    suggestions['check_for_merger'].append(
                        f"{symbol}: {failure_count} failures - check for merger/rename"
                    )
            
            # Check for stocks with alternative symbols
            for category in ['renamed', 'merged']:
                for stock in report.get(category, []):
                    symbol = stock['symbol']
                    alternative = stock.get('renamed_to') or stock.get('merged_into')
                    if alternative:
                        suggestions['use_alternative_symbol'].append(
                            f"{symbol} -> {alternative}"
                        )
            
        except Exception as e:
            self.logger.error(f"Failed to generate suggestions: {e}")
        
        return suggestions
    
    def bulk_update_stock_status(self, updates: List[Dict]):
        """Bulk update stock statuses"""
        success_count = 0
        
        for update in updates:
            try:
                self.db_manager.update_stock_status(
                    symbol=update['symbol'],
                    status=update['status'],
                    **{k: v for k, v in update.items() if k not in ['symbol', 'status']}
                )
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to update {update['symbol']}: {e}")
        
        self.logger.info(f"Successfully updated {success_count}/{len(updates)} stock statuses")
        return success_count
    
    def clean_symbols_list(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        """Clean symbols list by removing problematic stocks and suggesting alternatives"""
        clean_symbols = []
        problematic_symbols = []
        
        for symbol in symbols:
            should_skip, reason = self.db_manager.should_skip_download(symbol)
            
            if should_skip:
                problematic_symbols.append(f"{symbol}: {reason}")
                
                # Try to find alternative
                alternative = self.db_manager.get_alternative_symbol(symbol)
                if alternative and alternative not in symbols:
                    clean_symbols.append(alternative)
                    self.logger.info(f"Replaced {symbol} with {alternative}")
            else:
                clean_symbols.append(symbol)
        
        return clean_symbols, problematic_symbols
    
    def validate_symbols_against_known_issues(self, symbols_file: str) -> Dict:
        """Validate symbols file against known issues"""
        try:
            # Read symbols file
            if symbols_file.endswith('.xlsx'):
                df = pd.read_excel(symbols_file)
            else:
                df = pd.read_csv(symbols_file)
            
            # Assume first column contains symbols
            symbols = df.iloc[:, 0].tolist()
            
            # Clean and validate
            clean_symbols, problematic = self.clean_symbols_list(symbols)
            
            validation_result = {
                'total_symbols': len(symbols),
                'clean_symbols': len(clean_symbols),
                'problematic_symbols': len(problematic),
                'problematic_details': problematic,
                'replacement_suggestions': []
            }
            
            # Generate replacement file
            if clean_symbols:
                clean_df = pd.DataFrame({'Symbol': clean_symbols})
                output_file = symbols_file.replace('.xlsx', '_cleaned.xlsx').replace('.csv', '_cleaned.csv')
                
                if output_file.endswith('.xlsx'):
                    clean_df.to_excel(output_file, index=False)
                else:
                    clean_df.to_csv(output_file, index=False)
                
                validation_result['cleaned_file'] = output_file
                self.logger.info(f"Created cleaned symbols file: {output_file}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate symbols file: {e}")
            return {'error': str(e)}

def main():
    """CLI interface for stock status management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Status Management Utility')
    parser.add_argument('--init', action='store_true', help='Initialize known stock issues')
    parser.add_argument('--report', type=str, help='Generate problematic stocks report')
    parser.add_argument('--validate', type=str, help='Validate symbols file')
    parser.add_argument('--import-changes', type=str, help='Import stock changes from CSV')
    parser.add_argument('--suggestions', action='store_true', help='Show fix suggestions')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize
    config = AppConfig()
    db_manager = EnhancedDatabaseManager(config.database)
    status_manager = StockStatusManager(db_manager)
    
    try:
        if args.init:
            status_manager.initialize_known_stock_issues()
            print("✓ Known stock issues initialized")
        
        if args.report:
            output_path = status_manager.export_problematic_stocks_report(args.report)
            if output_path:
                print(f"✓ Report exported to: {output_path}")
        
        if args.validate:
            result = status_manager.validate_symbols_against_known_issues(args.validate)
            print("\n📊 Validation Results:")
            print(f"Total symbols: {result.get('total_symbols', 0)}")
            print(f"Clean symbols: {result.get('clean_symbols', 0)}")
            print(f"Problematic symbols: {result.get('problematic_symbols', 0)}")
            
            if result.get('problematic_details'):
                print("\n⚠️  Problematic symbols:")
                for issue in result['problematic_details']:
                    print(f"  - {issue}")
            
            if result.get('cleaned_file'):
                print(f"\n✓ Cleaned file created: {result['cleaned_file']}")
        
        if args.import_changes:
            status_manager.import_stock_changes_from_csv(args.import_changes)
            print("✓ Stock changes imported")
        
        if args.suggestions:
            suggestions = status_manager.suggest_stock_fixes()
            print("\n💡 Fix Suggestions:")
            for category, items in suggestions.items():
                if items:
                    print(f"\n{category.replace('_', ' ').title()}:")
                    for item in items:
                        print(f"  - {item}")
    
    finally:
        db_manager.close_connections()

if __name__ == "__main__":
    main()
