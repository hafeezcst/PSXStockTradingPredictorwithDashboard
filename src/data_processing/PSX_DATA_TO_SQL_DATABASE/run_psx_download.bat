@echo off
REM PSX Data Download - Windows Batch Launcher
REM This script provides an easy way to run the PSX data download system on Windows

echo ===============================================
echo PSX Data to SQL Database - Enhanced v2.0
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later
    pause
    exit /b 1
)

REM Change to the script directory
cd /d "%~dp0"

echo Current Directory: %cd%
echo.

REM Check if setup has been run
if not exist "data\databases\production" (
    echo Running initial setup...
    python setup.py
    if %errorlevel% neq 0 (
        echo ERROR: Setup failed
        pause
        exit /b 1
    )
    echo.
)

REM Run the main data download
echo Starting PSX data download...
echo.
python run_psx_download.py

if %errorlevel% equ 0 (
    echo.
    echo ===============================================
    echo Data download completed successfully!
    echo ===============================================
) else (
    echo.
    echo ===============================================
    echo Data download encountered errors
    echo Check the logs for more information
    echo ===============================================
)

echo.
echo Press any key to exit...
pause >nul
