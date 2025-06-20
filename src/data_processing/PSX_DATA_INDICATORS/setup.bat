@echo off
echo [SETUP] PSX Indicator Processor Windows Setup
echo ================================================

echo [INFO] Checking current directory...
cd /d "%~dp0"
echo [OK] Current directory: %CD%

echo [INFO] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ first.
    pause
    exit /b 1
)

echo [INFO] Installing requirements...
python -m pip install --upgrade pip
python -m pip install pandas numpy sqlalchemy tqdm pandas-ta pyyaml

echo [INFO] Creating directories...
if not exist "exports\csv" mkdir "exports\csv"
if not exist "exports\parquet" mkdir "exports\parquet"
if not exist "exports\json" mkdir "exports\json"
if not exist "logs" mkdir "logs"
if not exist "config" mkdir "config"

echo [INFO] Running setup script...
python setup_new.py

echo [INFO] Running tests...
python test_simple.py

echo [SUCCESS] Setup completed!
echo.
echo Next steps:
echo 1. Run: python enhanced_psx_processor_simple.py
echo 2. Check logs folder for processing logs
echo 3. See README_WINDOWS.md for documentation

pause
