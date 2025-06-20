#!/usr/bin/env python3
"""
Simple test script for the Enhanced PSX Processor

This tests the basic functionality without complex dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_imports():
    """Test if we can import the processor."""
    print("[TEST] Testing imports...")
    try:
        from enhanced_psx_processor_simple import PSXIndicatorProcessor, ProcessorConfig, DataValidator
        print("[OK] Imports successful")
        return True
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_config():
    """Test configuration creation."""
    print("[TEST] Testing configuration...")
    try:
        from enhanced_psx_processor_simple import ProcessorConfig
        config = ProcessorConfig()
        print(f"[OK] Config created with {config.max_workers} workers")
        return True
    except Exception as e:
        print(f"[ERROR] Config test failed: {e}")
        return False

def test_data_validator():
    """Test data validation."""
    print("[TEST] Testing data validator...")
    try:
        from enhanced_psx_processor_simple import DataValidator
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 250, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure logical relationships
        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
        
        validator = DataValidator()
        results = validator.validate_ohlcv_data(data)
        
        print(f"[OK] Data validation: Valid={results['is_valid']}, Quality={results['quality_score']:.1f}")
        return results['is_valid']
    except Exception as e:
        print(f"[ERROR] Data validator test failed: {e}")
        return False

def test_indicator_calculator():
    """Test indicator calculations."""
    print("[TEST] Testing indicator calculator...")
    try:
        from enhanced_psx_processor_simple import IndicatorCalculator, ProcessorConfig
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)  # For reproducible results
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 200),
            'High': np.random.uniform(200, 250, 200),
            'Low': np.random.uniform(50, 100, 200),
            'Close': np.random.uniform(100, 200, 200),
            'Volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        # Ensure logical relationships
        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
        
        config = ProcessorConfig()
        calculator = IndicatorCalculator(config)
        
        result = calculator.calculate_comprehensive_indicators(data.copy())
        
        print(f"[OK] Indicators calculated: {len(result.columns)} total columns")
        
        # Check for key indicators
        expected_indicators = ['RSI_14', 'SMA_20', 'Price_Change']
        found_indicators = [ind for ind in expected_indicators if ind in result.columns]
        print(f"[OK] Found indicators: {found_indicators}")
        
        return len(result.columns) > len(data.columns)
    except Exception as e:
        print(f"[ERROR] Indicator calculator test failed: {e}")
        return False

def test_processor_init():
    """Test processor initialization."""
    print("[TEST] Testing processor initialization...")
    try:
        from enhanced_psx_processor_simple import PSXIndicatorProcessor
        
        processor = PSXIndicatorProcessor()
        stats = processor.get_processing_stats()
        
        print(f"[OK] Processor initialized")
        print(f"     CPU cores: {stats['system']['cpu_count']}")
        print(f"     TA available: {stats['system']['ta_available']}")
        print(f"     Max workers: {stats['config']['max_workers']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Processor init test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("PSX INDICATOR PROCESSOR TESTS")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_validator,
        test_indicator_calculator,
        test_processor_init
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    print("=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test.__name__}")
    
    print(f"\nSUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed!")
        return True
    else:
        print("[WARNING] Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
