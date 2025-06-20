# PSX Indicator Processor - Windows Compatible Version

This folder contains enhanced versions of the PSX indicator processor that are optimized for Windows systems and resolve Unicode/import issues.

## Files Overview

### Main Processors
- **`enhanced_psx_processor_simple.py`** - Windows-compatible enhanced processor (recommended)
- **`enhanced_psx_indicator_processor.py`** - Full-featured processor (may have import issues)
- **`01-PSX_SQL_Indicator_PSX_bkup.py`** - Original script with fixes

### Setup and Testing
- **`setup_fixed.py`** - Windows-compatible setup script
- **`test_simple.py`** - Simple test script to verify functionality
- **`test_enhanced_processor.py`** - Comprehensive test suite

### Configuration
- **`config.yaml`** - Configuration template
- **`requirements.txt`** - Python dependencies

### Documentation
- **`README.md`** - Full documentation
- **`usage_examples.py`** - Usage examples
- **`migrate_to_enhanced.py`** - Migration utilities

## Quick Start (Windows)

1. **Install Dependencies**:
   ```cmd
   pip install pandas numpy sqlalchemy tqdm pandas-ta pyyaml
   ```

2. **Run Simple Test**:
   ```cmd
   python test_simple.py
   ```

3. **Use the Simple Processor**:
   ```python
   from enhanced_psx_processor_simple import PSXIndicatorProcessor
   
   processor = PSXIndicatorProcessor()
   results = processor.process_all_symbols()
   print(f"Processed {results['successful']} symbols")
   ```

## Key Fixes Applied

### Unicode Issues Fixed
- Replaced all Unicode symbols (✅❌🚀) with ASCII equivalents ([OK][ERROR][INFO])
- Fixed Windows console encoding issues
- Updated all print statements to use ASCII-safe characters

### Import Issues Fixed
- Made pandas_ta import optional with fallback mechanisms
- Added multiple import strategies for technical analysis library
- Created fallback implementations for key indicators
- Improved error handling for missing dependencies

### Performance Improvements
- Added ThreadPoolExecutor for parallel processing
- Optimized database connections
- Improved memory usage for large datasets
- Added progress tracking with tqdm

### Enhanced Features
- Data validation and quality scoring
- 30+ technical indicators
- Multi-timeframe analysis
- Machine learning features
- Configuration management
- Comprehensive logging

## Usage Options

### Option 1: Simple Processor (Recommended for Windows)
```python
from enhanced_psx_processor_simple import PSXIndicatorProcessor

processor = PSXIndicatorProcessor()
results = processor.process_all_symbols()
```

### Option 2: Original Script (Fixed)
```python
from src.data_processing.fix_pandas_ta import ta

data_reader = DataReader()
table_names = data_reader.get_table_names()

for table_name in table_names:
    data = data_reader.read_data(table_name)
    processed_data = data_reader.preprocess(data)
    data_reader.save_to_db(processed_data, table_name)
```

## Configuration

Create a configuration file:
```yaml
# Performance settings
max_workers: 4
batch_size: 1000
cache_size: 128

# Processing options
calculate_advanced_indicators: true
include_ml_features: true
enable_data_validation: true

# Indicator parameters
rsi_periods: [9, 14, 21, 26]
ma_periods: [20, 30, 50, 100, 200]
```

## Troubleshooting

### If you see Unicode errors:
- Use `enhanced_psx_processor_simple.py` instead
- Set `PYTHONIOENCODING=utf-8` environment variable
- Use Command Prompt instead of PowerShell

### If pandas_ta import fails:
- Install: `pip install pandas-ta`
- Or use the fallback implementations in the simple processor

### If database errors occur:
- Check database paths in configuration
- Ensure SQLite databases are accessible
- Verify proper file permissions

## Features Comparison

| Feature | Original Script | Simple Processor | Full Processor |
|---------|----------------|------------------|----------------|
| Windows Compatible | ❌ | ✅ | ⚠️ |
| Unicode Safe | ❌ | ✅ | ❌ |
| Parallel Processing | ❌ | ✅ | ✅ |
| Data Validation | ❌ | ✅ | ✅ |
| 30+ Indicators | ❌ | ✅ | ✅ |
| ML Features | ❌ | ✅ | ✅ |
| Progress Tracking | ✅ | ✅ | ✅ |
| Configuration | ❌ | ✅ | ✅ |

## Recommended Workflow

1. **Start with Simple Test**: Run `python test_simple.py`
2. **Use Simple Processor**: For production use `enhanced_psx_processor_simple.py`
3. **Configure as Needed**: Modify settings in configuration file
4. **Monitor Performance**: Check logs for processing statistics

This setup ensures maximum compatibility with Windows systems while providing enhanced functionality.
