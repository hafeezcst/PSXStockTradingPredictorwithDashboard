#!/usr/bin/env python3
"""
Quick test for the enhanced processor fixes
"""

import sys
import traceback
import pandas as pd
import numpy as np

try:
    print("🧪 Testing Enhanced PSX Indicator Processor fixes...")
    
    # Test imports
    from enhanced_psx_indicator_processor import EnhancedPSXIndicatorProcessor, ProcessorConfig
    print("✅ Imports successful")
    
    # Test processor creation
    config = ProcessorConfig()
    processor = EnhancedPSXIndicatorProcessor(config)
    print("✅ Processor creation successful")
    
    # Test the safe TA call method with sample data
    calculator = processor.calculator
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=50),
        'Open': np.random.uniform(100, 110, 50),
        'High': np.random.uniform(110, 120, 50),
        'Low': np.random.uniform(90, 100, 50),
        'Close': np.random.uniform(95, 105, 50),
        'Volume': np.random.uniform(1000, 5000, 50)
    })
    sample_data.set_index('Date', inplace=True)
    
    # Test safe MACD calculation
    import pandas_ta as ta
    macd_result = calculator._safe_ta_call(ta.macd, sample_data['Close'])
    print(f"✅ Safe MACD call successful: {type(macd_result)}")
    
    # Test safe concat
    test_data = calculator._safe_concat(sample_data, macd_result, "MACD")
    print(f"✅ Safe concat successful: {len(test_data.columns)} columns")
    
    print("🎉 All tuple handling tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    traceback.print_exc()
