"""
Test suite for Enhanced PSX Indicator Processor

Basic tests to verify functionality and performance.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_psx_indicator_processor import (
    EnhancedPSXIndicatorProcessor,
    ProcessorConfig,
    DataValidator,
    IndicatorCalculator
)


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_valid_data(self):
        """Test validation of valid OHLCV data."""
        # Create sample valid data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(200, 250, len(dates)),
            'Low': np.random.uniform(50, 100, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure High >= Low and other logical relationships
        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
        
        validator = DataValidator()
        results = validator.validate_ohlcv_data(data)
        
        assert results['is_valid'] == True
        assert results['quality_score'] > 80
        assert len(results['issues']) == 0
    
    def test_invalid_data(self):
        """Test validation of invalid data."""
        # Create sample invalid data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        data = pd.DataFrame({
            'Open': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            'High': [90, 100, 110, 120, 130, 140, 150, 160, 170, 180],  # High < Open (invalid)
            'Low': [200, 210, 220, 230, 240, 250, 260, 270, 280, 290],  # Low > High (invalid)
            'Close': [105, 115, 125, 135, 145, 155, 165, 175, 185, 195],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)
        
        validator = DataValidator()
        results = validator.validate_ohlcv_data(data)
        
        assert results['is_valid'] == False
        assert results['quality_score'] < 100
        assert len(results['issues']) > 0


class TestIndicatorCalculator:
    """Test indicator calculation functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = ProcessorConfig()
        self.calculator = IndicatorCalculator(self.config)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        self.data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(200, 250, len(dates)),
            'Low': np.random.uniform(50, 100, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure logical price relationships
        self.data['High'] = np.maximum(self.data['High'], self.data[['Open', 'Close']].max(axis=1))
        self.data['Low'] = np.minimum(self.data['Low'], self.data[['Open', 'Close']].min(axis=1))
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        result = self.calculator._calculate_rsi_suite(self.data.copy())
        
        # Check if RSI columns are created
        assert 'RSI_14' in result.columns
        assert 'RSI_14_SMA' in result.columns
        
        # Check RSI values are in valid range (0-100)
        rsi_values = result['RSI_14'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_moving_averages(self):
        """Test moving average calculation."""
        result = self.calculator._calculate_moving_averages(self.data.copy())
        
        # Check if MA columns are created
        for period in self.config.ma_periods:
            assert f'SMA_{period}' in result.columns
            assert f'EMA_{period}' in result.columns
        
        # Check that MA values are reasonable
        assert not result['SMA_20'].isna().all()
        assert not result['EMA_20'].isna().all()
    
    def test_comprehensive_indicators(self):
        """Test comprehensive indicator calculation."""
        result = self.calculator.calculate_comprehensive_indicators(self.data.copy())
        
        # Should have more columns than original data
        assert len(result.columns) > len(self.data.columns)
        
        # Should contain RSI
        assert 'RSI_14' in result.columns
        
        # Should contain moving averages
        assert 'SMA_20' in result.columns
        assert 'EMA_20' in result.columns


class TestProcessorConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ProcessorConfig()
        
        assert config.max_workers > 0
        assert config.batch_size > 0
        assert isinstance(config.rsi_periods, list)
        assert isinstance(config.ma_periods, list)
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        config = ProcessorConfig(
            max_workers=4,
            batch_size=500,
            rsi_periods=[14, 21],
            ma_periods=[20, 50]
        )
        
        assert config.max_workers == 4
        assert config.batch_size == 500
        assert config.rsi_periods == [14, 21]
        assert config.ma_periods == [20, 50]


def create_sample_database():
    """Create a sample database for testing."""
    from sqlalchemy import create_engine
    import tempfile
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    engine = create_engine(f'sqlite:///{temp_db.name}')
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(200, 250, len(dates)),
        'Low': np.random.uniform(50, 100, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Ensure logical relationships
    sample_data['High'] = np.maximum(sample_data['High'], sample_data[['Open', 'Close']].max(axis=1))
    sample_data['Low'] = np.minimum(sample_data['Low'], sample_data[['Open', 'Close']].min(axis=1))
    
    # Save to database
    sample_data.to_sql('PSX_TEST_stock_data', engine, index=False)
    
    return temp_db.name


@pytest.mark.asyncio
async def test_processor_basic_functionality():
    """Test basic processor functionality with sample data."""
    # Create sample database
    db_path = create_sample_database()
    
    try:
        # Configure processor with sample database
        config = ProcessorConfig(
            source_db_path=db_path,
            target_db_path=db_path.replace('.db', '_indicators.db'),
            max_workers=1,  # Use single worker for testing
            calculate_advanced_indicators=False,  # Faster for testing
            include_ml_features=False
        )
        
        processor = EnhancedPSXIndicatorProcessor(config)
        
        # Test table name retrieval
        table_names = processor.get_table_names()
        assert len(table_names) > 0
        assert 'PSX_TEST_stock_data' in table_names
        
        # Test single symbol processing
        result = processor.process_single_symbol('PSX_TEST_stock_data')
        assert result['success'] == True
        assert result['row_count'] > 0
        
    finally:
        # Cleanup
        import os
        try:
            os.unlink(db_path)
            os.unlink(db_path.replace('.db', '_indicators.db'))
        except:
            pass


def run_performance_test():
    """Run basic performance test."""
    print("Running performance test...")
    
    # Create larger dataset for performance testing
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    large_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(200, 250, len(dates)),
        'Low': np.random.uniform(50, 100, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure logical relationships
    large_data['High'] = np.maximum(large_data['High'], large_data[['Open', 'Close']].max(axis=1))
    large_data['Low'] = np.minimum(large_data['Low'], large_data[['Open', 'Close']].min(axis=1))
    
    config = ProcessorConfig()
    calculator = IndicatorCalculator(config)
    
    import time
    start_time = time.time()
    
    result = calculator.calculate_comprehensive_indicators(large_data)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processed {len(large_data)} rows in {processing_time:.2f} seconds")
    print(f"Processing rate: {len(large_data)/processing_time:.0f} rows/second")
    print(f"Generated {len(result.columns)} indicator columns")
    
    return processing_time


if __name__ == "__main__":
    # Run individual tests
    print("Testing Data Validator...")
    validator_test = TestDataValidator()
    validator_test.test_valid_data()
    validator_test.test_invalid_data()
    print("✅ Data Validator tests passed")
    
    print("\nTesting Indicator Calculator...")
    calc_test = TestIndicatorCalculator()
    calc_test.setup_method()
    calc_test.test_rsi_calculation()
    calc_test.test_moving_averages()
    calc_test.test_comprehensive_indicators()
    print("✅ Indicator Calculator tests passed")
    
    print("\nTesting Processor Config...")
    config_test = TestProcessorConfig()
    config_test.test_default_config()
    config_test.test_config_validation()
    print("✅ Processor Config tests passed")
    
    print("\nRunning performance test...")
    perf_time = run_performance_test()
    print("✅ Performance test completed")
    
    print("\nAll tests completed successfully! 🎉")
