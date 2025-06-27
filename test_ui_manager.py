#!/usr/bin/env python
"""
Test script for ui_manager.py to verify that all imports and basic functionality work.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test if all imports work correctly."""
    try:
        print("Testing basic imports...")
        import PyQt6
        print("✓ PyQt6 imported successfully")
        
        import pandas as pd
        print("✓ pandas imported successfully")
        
        import numpy as np
        print("✓ numpy imported successfully")
        
        import matplotlib
        print("✓ matplotlib imported successfully")
        
        import plotly
        print("✓ plotly imported successfully")
        
        import sklearn
        print("✓ sklearn imported successfully")
        
        import statsmodels
        print("✓ statsmodels imported successfully")
        
        print("\n✓ All required dependencies are available")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_ui_manager_import():
    """Test if ui_manager can be imported."""
    try:
        print("\nTesting ui_manager import...")
        from src.data_processing.stock_analysis.ui_manager import MainWindow, AIPredictor, PlotlyWidget
        print("✓ MainWindow, AIPredictor, and PlotlyWidget imported successfully")
        
        # Test class instantiation without UI
        print("Testing class instantiation...")
        predictor = AIPredictor()
        print("✓ AIPredictor instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ ui_manager import error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("PSX Stock Analysis UI Manager - Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test imports
    test_results.append(test_imports())
    
    # Test ui_manager
    test_results.append(test_ui_manager_import())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    if all(test_results):
        print("✓ All tests passed! ui_manager.py is working correctly.")
        print("You can now run the GUI with: python src/data_processing/stock_analysis/ui_manager.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    
    return all(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
