#!/usr/bin/env python3
"""
Migration Script: Old PSX Indicator Processor to Enhanced Version

This script helps migrate from the old indicator processor to the new enhanced version.
It provides utilities to:
- Compare processing results
- Migrate existing databases
- Validate indicator calculations
- Performance benchmarking

Usage:
    python migrate_to_enhanced.py --action compare
    python migrate_to_enhanced.py --action benchmark
    python migrate_to_enhanced.py --action migrate --source old_db.db --target new_db.db
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Import both old and new processors
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_psx_indicator_processor import (
    EnhancedPSXIndicatorProcessor,
    ProcessorConfig
)

# Assuming the old processor is available
try:
    from ..fix_pandas_ta import ta
    from sqlalchemy import create_engine, inspect
    
    class OldDataReader:
        """Simplified version of the old DataReader for comparison."""
        
        def __init__(self, source_db_path: str, target_db_path: str):
            self.source_engine = create_engine(f'sqlite:///{source_db_path}')
            self.target_engine = create_engine(f'sqlite:///{target_db_path}')
        
        def get_table_names(self) -> List[str]:
            inspector = inspect(self.source_engine)
            return inspector.get_table_names()
        
        def read_data(self, table_name: str) -> pd.DataFrame:
            try:
                return pd.read_sql_table(table_name, self.source_engine, index_col='Date', parse_dates=['Date'])
            except Exception as e:
                logging.error(f"Error reading {table_name}: {e}")
                return pd.DataFrame()
        
        def calculate_rsi(self, data: pd.Series, length: int) -> pd.Series:
            try:
                return ta.rsi(data, length=length)
            except Exception:
                return pd.Series(index=data.index, dtype=float)
        
        def preprocess_old(self, data: pd.DataFrame) -> pd.DataFrame:
            """Old preprocessing logic for comparison."""
            if data.empty:
                return pd.DataFrame()
            
            # Basic RSI calculations (simplified)
            data['RSI_14'] = self.calculate_rsi(data['Close'], 14)
            data['RSI_14_Avg'] = ta.sma(data['RSI_14'], length=14)
            
            # Basic moving averages
            data['MA_30'] = ta.sma(data['Close'], length=30)
            data['MA_50'] = ta.sma(data['Close'], length=50)
            data['MA_100'] = ta.sma(data['Close'], length=100)
            data['MA_200'] = ta.sma(data['Close'], length=200)
            
            # Basic AO
            hl2 = (data['High'] + data['Low']) / 2
            data['AO'] = ta.sma(hl2, 5) - ta.sma(hl2, 34)
            
            return data

except ImportError:
    logging.warning("Old processor modules not available. Some features may be limited.")
    OldDataReader = None


class MigrationTool:
    """Migration and comparison tool for PSX processors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def create_sample_data(self, num_days: int = 365) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic stock data
        base_price = 100
        price_walk = np.cumsum(np.random.normal(0, 1, num_days)) + base_price
        
        data = pd.DataFrame({
            'Open': price_walk + np.random.normal(0, 0.5, num_days),
            'High': price_walk + np.abs(np.random.normal(2, 1, num_days)),
            'Low': price_walk - np.abs(np.random.normal(2, 1, num_days)),
            'Close': price_walk + np.random.normal(0, 0.5, num_days),
            'Volume': np.random.randint(1000, 10000, num_days)
        }, index=dates)
        
        # Ensure logical relationships
        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
        
        return data
    
    def benchmark_processors(self, data_sizes: List[int] = [100, 500, 1000, 2000]) -> Dict[str, Any]:
        """Benchmark old vs new processor performance."""
        results = {
            'data_sizes': data_sizes,
            'old_times': [],
            'new_times': [],
            'speedup_ratios': []
        }
        
        self.logger.info("Starting performance benchmark...")
        
        for size in data_sizes:
            self.logger.info(f"Benchmarking with {size} data points...")
            
            # Create test data
            test_data = self.create_sample_data(size)
            
            # Benchmark old processor (if available)
            old_time = None
            if OldDataReader:
                try:
                    old_reader = OldDataReader("dummy.db", "dummy.db")
                    start_time = time.time()
                    old_reader.preprocess_old(test_data.copy())
                    old_time = time.time() - start_time
                    results['old_times'].append(old_time)
                except Exception as e:
                    self.logger.warning(f"Old processor benchmark failed: {e}")
                    results['old_times'].append(None)
            else:
                results['old_times'].append(None)
            
            # Benchmark new processor
            try:
                config = ProcessorConfig(
                    calculate_advanced_indicators=True,
                    include_ml_features=True
                )
                from enhanced_psx_indicator_processor import IndicatorCalculator
                calculator = IndicatorCalculator(config)
                
                start_time = time.time()
                calculator.calculate_comprehensive_indicators(test_data.copy())
                new_time = time.time() - start_time
                results['new_times'].append(new_time)
                
                # Calculate speedup ratio
                if old_time and new_time:
                    ratio = old_time / new_time
                    results['speedup_ratios'].append(ratio)
                    self.logger.info(f"Size {size}: Old={old_time:.3f}s, New={new_time:.3f}s, Speedup={ratio:.2f}x")
                else:
                    results['speedup_ratios'].append(None)
                    self.logger.info(f"Size {size}: New={new_time:.3f}s")
                    
            except Exception as e:
                self.logger.error(f"New processor benchmark failed: {e}")
                results['new_times'].append(None)
                results['speedup_ratios'].append(None)
        
        return results
    
    def compare_indicators(self, symbol: str = "TEST") -> Dict[str, Any]:
        """Compare indicator calculations between old and new processors."""
        self.logger.info(f"Comparing indicators for symbol: {symbol}")
        
        # Create test data
        test_data = self.create_sample_data(500)
        
        comparison_results = {
            'symbol': symbol,
            'data_points': len(test_data),
            'indicators_compared': [],
            'differences': {},
            'correlation_scores': {},
            'summary': {}
        }
        
        # Process with old method (if available)
        old_results = None
        if OldDataReader:
            try:
                old_reader = OldDataReader("dummy.db", "dummy.db")
                old_results = old_reader.preprocess_old(test_data.copy())
                self.logger.info(f"Old processor generated {len(old_results.columns)} columns")
            except Exception as e:
                self.logger.warning(f"Old processor failed: {e}")
        
        # Process with new method
        try:
            config = ProcessorConfig()
            from enhanced_psx_indicator_processor import IndicatorCalculator
            calculator = IndicatorCalculator(config)
            new_results = calculator.calculate_comprehensive_indicators(test_data.copy())
            self.logger.info(f"New processor generated {len(new_results.columns)} columns")
        except Exception as e:
            self.logger.error(f"New processor failed: {e}")
            return comparison_results
        
        # Compare common indicators
        if old_results is not None:
            common_columns = set(old_results.columns).intersection(set(new_results.columns))
            self.logger.info(f"Found {len(common_columns)} common indicators")
            
            for col in common_columns:
                if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    continue  # Skip original data columns
                
                old_values = old_results[col].dropna()
                new_values = new_results[col].dropna()
                
                if len(old_values) > 0 and len(new_values) > 0:
                    # Align the data
                    min_len = min(len(old_values), len(new_values))
                    old_aligned = old_values.iloc[-min_len:]
                    new_aligned = new_values.iloc[-min_len:]
                    
                    # Calculate correlation
                    try:
                        correlation = old_aligned.corr(new_aligned)
                        comparison_results['correlation_scores'][col] = correlation
                        
                        # Calculate mean absolute difference
                        diff = np.abs(old_aligned - new_aligned).mean()
                        comparison_results['differences'][col] = diff
                        
                        comparison_results['indicators_compared'].append(col)
                        
                        self.logger.info(f"{col}: Correlation={correlation:.4f}, MAD={diff:.6f}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to compare {col}: {e}")
        
        # Generate summary
        if comparison_results['correlation_scores']:
            avg_correlation = np.mean(list(comparison_results['correlation_scores'].values()))
            high_correlation_count = sum(1 for c in comparison_results['correlation_scores'].values() if c > 0.95)
            
            comparison_results['summary'] = {
                'total_indicators_compared': len(comparison_results['indicators_compared']),
                'average_correlation': avg_correlation,
                'high_correlation_indicators': high_correlation_count,
                'new_indicators_count': len(new_results.columns) - len(test_data.columns)
            }
            
            self.logger.info(f"Summary: {comparison_results['summary']}")
        
        return comparison_results
    
    async def migrate_database(self, source_db: str, target_db: str, 
                             symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Migrate existing database to new format."""
        self.logger.info(f"Migrating database from {source_db} to {target_db}")
        
        migration_results = {
            'source_db': source_db,
            'target_db': target_db,
            'symbols_processed': 0,
            'symbols_failed': 0,
            'processing_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Setup enhanced processor
            config = ProcessorConfig(
                source_db_path=source_db,
                target_db_path=target_db,
                calculate_advanced_indicators=True,
                include_ml_features=True,
                enable_data_validation=True
            )
            
            processor = EnhancedPSXIndicatorProcessor(config)
            
            # Process symbols
            if symbols:
                results = await processor.process_symbols_async(symbols)
            else:
                results = await processor.process_all_symbols()
            
            migration_results.update({
                'symbols_processed': results['successful'],
                'symbols_failed': results['failed'],
                'processing_time': results['processing_time']
            })
            
            self.logger.info(f"Migration completed: {results['successful']} successful, {results['failed']} failed")
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            migration_results['errors'].append(str(e))
        
        migration_results['processing_time'] = time.time() - start_time
        return migration_results
    
    def generate_migration_report(self, benchmark_results: Dict[str, Any], 
                                comparison_results: Dict[str, Any]) -> str:
        """Generate a comprehensive migration report."""
        report = []
        report.append("=" * 60)
        report.append("PSX INDICATOR PROCESSOR MIGRATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Performance section
        report.append("PERFORMANCE BENCHMARK:")
        report.append("-" * 30)
        
        if benchmark_results['new_times']:
            for i, size in enumerate(benchmark_results['data_sizes']):
                new_time = benchmark_results['new_times'][i]
                old_time = benchmark_results['old_times'][i]
                
                if new_time:
                    report.append(f"Data size {size:4d}: New processor = {new_time:.3f}s")
                    if old_time:
                        speedup = old_time / new_time
                        report.append(f"                 Old processor = {old_time:.3f}s (Speedup: {speedup:.2f}x)")
                    report.append("")
        
        # Comparison section
        if comparison_results.get('summary'):
            report.append("INDICATOR COMPARISON:")
            report.append("-" * 30)
            summary = comparison_results['summary']
            report.append(f"Indicators compared: {summary['total_indicators_compared']}")
            report.append(f"Average correlation: {summary['average_correlation']:.4f}")
            report.append(f"High correlation (>0.95): {summary['high_correlation_indicators']}")
            report.append(f"New indicators added: {summary['new_indicators_count']}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 30)
        
        if comparison_results.get('summary', {}).get('average_correlation', 0) > 0.9:
            report.append("✅ High correlation between old and new calculations")
            report.append("✅ Migration recommended - results are consistent")
        else:
            report.append("⚠️  Some differences found in calculations")
            report.append("⚠️  Review specific indicators before migration")
        
        if benchmark_results.get('speedup_ratios'):
            avg_speedup = np.mean([r for r in benchmark_results['speedup_ratios'] if r])
            if avg_speedup > 1.5:
                report.append("✅ Significant performance improvement expected")
            elif avg_speedup > 1.0:
                report.append("✅ Moderate performance improvement expected")
        
        report.append("")
        report.append("ENHANCED FEATURES:")
        report.append("-" * 30)
        report.append("✅ Async processing for better concurrency")
        report.append("✅ Data validation and quality scoring")
        report.append("✅ 50+ technical indicators")
        report.append("✅ Machine learning features")
        report.append("✅ Multiple export formats")
        report.append("✅ GPU acceleration support")
        report.append("✅ Rich progress monitoring")
        report.append("✅ Configuration management")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


async def main():
    """Main CLI interface for migration tool."""
    parser = argparse.ArgumentParser(description="PSX Processor Migration Tool")
    parser.add_argument("--action", choices=["compare", "benchmark", "migrate", "report"], 
                       required=True, help="Action to perform")
    parser.add_argument("--source", help="Source database path")
    parser.add_argument("--target", help="Target database path")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to process")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    tool = MigrationTool()
    
    if args.action == "benchmark":
        print("Running performance benchmark...")
        benchmark_results = tool.benchmark_processors()
        print(f"Benchmark completed. Results: {benchmark_results}")
        
    elif args.action == "compare":
        print("Comparing indicator calculations...")
        comparison_results = tool.compare_indicators()
        print(f"Comparison completed. Results: {comparison_results}")
        
    elif args.action == "migrate":
        if not args.source or not args.target:
            print("Error: --source and --target required for migration")
            return
        
        print(f"Migrating database from {args.source} to {args.target}...")
        migration_results = await tool.migrate_database(args.source, args.target, args.symbols)
        print(f"Migration completed. Results: {migration_results}")
        
    elif args.action == "report":
        print("Generating comprehensive migration report...")
        benchmark_results = tool.benchmark_processors()
        comparison_results = tool.compare_indicators()
        
        report = tool.generate_migration_report(benchmark_results, comparison_results)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)


if __name__ == "__main__":
    asyncio.run(main())
