# PSX Data Downloader GUI

A modern Windows GUI application for downloading PSX stock data to SQLite databases.

## Features

- Modern PyQt5-based interface with dark/light themes
- Multi-threaded downloads with progress tracking
- Configurable database connections
- Data preview and export (CSV, Excel, JSON)
- System tray integration
- Scheduled downloads
- Auto-update functionality

## Installation

1. Install Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. Install requirements:
   ```bash
   pip install -r psx_gui_requirements.txt
   ```
3. Run the application:
   ```bash
   python 01-PSX_Database_data_download_to_SQL_db_PSX_GUI.py
   ```

## Configuration

Edit `psx_gui_config.json` to customize:
- Database paths
- Threading settings
- Theme preferences
- Export locations

## Packaging

Create a standalone executable using PyInstaller:
```bash
pyinstaller --onefile --windowed 01-PSX_Database_data_download_to_SQL_db_PSX_GUI.py
```

## Usage

1. Select a stock symbol from the dropdown
2. Set date range (defaults to 2000-01-01 to today)
3. Click "Download" to start
4. Use "Preview" to view existing data
5. Export data using the "Export" button

## Troubleshooting

- Check `data_reader.log` for errors
- Ensure database paths in config are correct
- Verify internet connection to PSX servers