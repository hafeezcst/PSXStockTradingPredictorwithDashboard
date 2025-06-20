# PSX Data to SQL Database - Enhanced Module

The enhanced PSX data download and processing functionality has been reorganized into a dedicated module for better maintainability and future enhancements.

## 📁 New Location

The enhanced PSX data processing system is now located at:
```
src/data_processing/PSX_DATA_TO_SQL_DATABASE/
```

## 🚀 Quick Start

Navigate to the enhanced module directory:
```bash
cd src/data_processing/PSX_DATA_TO_SQL_DATABASE
```

Run the enhanced data reader:
```bash
python enhanced_psx_data_reader.py
```

Or use the launcher script:
```bash
python run_psx_download.py
```

## 📋 Key Features

- **Enhanced Architecture**: Modular design with separated concerns
- **Configuration Management**: YAML-based centralized configuration
- **Data Validation**: Comprehensive OHLCV validation and anomaly detection
- **Performance Monitoring**: Real-time metrics and health monitoring
- **Error Handling**: Robust retry mechanisms and custom exceptions
- **Database Management**: Advanced connection pooling and backup features

## 🔗 Migration

The original file `01-PSX_Database_data_download_to_SQL_db_PSX.py` remains in the main data_processing directory for backward compatibility, but the enhanced version is now in the dedicated module folder.

## 📖 Documentation

For detailed documentation, see:
- [Enhanced Module README](PSX_DATA_TO_SQL_DATABASE/README.md)
- [Migration Guide](PSX_DATA_TO_SQL_DATABASE/MIGRATION_GUIDE.md)

## 🛠️ Requirements

The enhanced module has its own requirements file:
```bash
pip install -r PSX_DATA_TO_SQL_DATABASE/requirements.txt
```

---

*This organization allows for better code management, easier testing, and streamlined future enhancements.*
