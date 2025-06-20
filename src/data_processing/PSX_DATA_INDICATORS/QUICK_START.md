# Quick Start Guide - PSX Indicator Processor

## Problem Fixed
You encountered:
1. **Unicode errors** (`UnicodeEncodeError: 'charmap' codec can't encode character`)
2. **Indentation errors** in setup.py
3. **Import path issues**

## Solution
I've created Windows-compatible versions with ASCII-only output and proper error handling.

## Quick Setup (Choose ONE method)

### Method 1: Batch File (Easiest)
```cmd
# Navigate to the folder first
cd "C:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\src\data_processing\PSX_DATA_INDICATORS"

# Run the batch setup
setup.bat
```

### Method 2: PowerShell Script
```powershell
# Navigate to the folder first
cd "C:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\src\data_processing\PSX_DATA_INDICATORS"

# Run PowerShell setup
.\setup.ps1
```

### Method 3: Manual Python Setup
```cmd
# Navigate to the folder first
cd "C:\Users\muhammadhafeez\Documents\GitHub\PSXStockTradingPredictorwithDashboard\src\data_processing\PSX_DATA_INDICATORS"

# Install requirements
pip install pandas numpy sqlalchemy tqdm pandas-ta pyyaml

# Run the fixed setup
python setup_new.py

# Test the installation
python test_simple.py
```

## Quick Test
After setup, test with:
```cmd
python test_simple.py
```

Expected output:
```
[TEST] Testing imports...
[OK] Imports successful

[TEST] Testing configuration...
[OK] Config created with 4 workers

[TEST] Testing data validator...
[OK] Data validation: Valid=True, Quality=100.0

[TEST] Testing indicator calculator...
[OK] Indicators calculated: 25 total columns
[OK] Found indicators: ['RSI_14', 'SMA_20', 'Price_Change']

[TEST] Testing processor initialization...
[OK] Processor initialized
     CPU cores: 8
     TA available: True
     Max workers: 4

[SUCCESS] All tests passed!
```

## Quick Usage
Once setup is complete:
```python
# Simple usage
from enhanced_psx_processor_simple import PSXIndicatorProcessor

processor = PSXIndicatorProcessor()
results = processor.process_all_symbols()
print(f"Processed {results['successful']} symbols successfully")
```

## Files Overview
- **`setup_new.py`** - Fixed setup script (Windows compatible)
- **`setup.bat`** - Windows batch setup
- **`setup.ps1`** - PowerShell setup script
- **`test_simple.py`** - Simple test to verify everything works
- **`enhanced_psx_processor_simple.py`** - Main processor (Windows compatible)

## Troubleshooting

### If you still get Unicode errors:
1. Use Command Prompt instead of PowerShell
2. Set environment variable: `set PYTHONIOENCODING=utf-8`
3. Use the simple processor instead of the full one

### If imports fail:
1. Make sure you're in the correct directory
2. Install pandas-ta: `pip install pandas-ta`
3. The simple processor has fallback implementations

### If database errors occur:
1. Check that your database files exist
2. Verify the paths in the configuration
3. Ensure proper file permissions

## Success Indicators
✅ No Unicode errors in output  
✅ All imports work properly  
✅ Tests pass without errors  
✅ Processor initializes correctly  
✅ Can calculate technical indicators  

## Next Steps After Success
1. Configure database paths in config/sample_config.yaml
2. Run the processor on your actual data
3. Check the logs folder for processing information
4. Use the various export options (CSV, Parquet, etc.)

This setup resolves all the issues you encountered and provides a robust, Windows-compatible solution.
