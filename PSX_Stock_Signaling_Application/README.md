# PSX Stock Signaling Application

## Overview
Standalone executable application for PSX stock analysis and trading signals with Telegram notifications.

## Project Structure
```
PSX_Stock_Signaling_Application/
├── config/               # Configuration files
├── data/                 # Database files
├── docs/                 # Documentation
├── logs/                 # Application logs
├── src/                  # Source code
│   ├── core/             # Core application logic
│   ├── modules/          # Functional modules
│   └── main.py           # Application entry point
├── scripts/              # Utility scripts
├── setup.py              # Setup script
└── file_mapping.json     # File organization mapping
```

## Setup Instructions
1. Install Python 3.8+
2. Run setup:
   ```bash
   python setup.py
   ```
3. Configure your settings in `config/app_config.yaml`
4. Add API keys in `config/secrets/api_keys.json`

## Configuration
Key configuration options:
- `database.main`: Path to main stock database
- `telegram.bot_token`: Telegram bot token
- `telegram.chat_id`: Target chat ID for notifications
- `analysis.rsi_period`: RSI calculation period (default: 14)

## Usage
Run the application:
```bash
python src/main.py
```

## Scheduling
For automated execution, set up a cron job (Linux/macOS) or Task Scheduler (Windows) to run:
```bash
python src/main.py --auto
```

## Requirements
- Python 3.8+
- Required packages: pandas, numpy, requests, python-telegram-bot, sqlalchemy