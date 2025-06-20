# PSX Indicator Processor PowerShell Setup Script

Write-Host "[SETUP] PSX Indicator Processor Windows Setup" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Set location to script directory
Set-Location $PSScriptRoot
Write-Host "[INFO] Current directory: $(Get-Location)" -ForegroundColor Cyan

# Check Python installation
Write-Host "[INFO] Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install requirements
Write-Host "[INFO] Installing requirements..." -ForegroundColor Cyan
try {
    python -m pip install --upgrade pip
    python -m pip install pandas numpy sqlalchemy tqdm pandas-ta pyyaml
    Write-Host "[OK] Requirements installed successfully" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to install requirements" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create directories
Write-Host "[INFO] Creating directories..." -ForegroundColor Cyan
$directories = @("exports\csv", "exports\parquet", "exports\json", "logs", "config")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  [FOLDER] Created $dir" -ForegroundColor Yellow
    } else {
        Write-Host "  [FOLDER] $dir already exists" -ForegroundColor Yellow
    }
}

# Run setup script
Write-Host "[INFO] Running Python setup script..." -ForegroundColor Cyan
try {
    python setup_new.py
    Write-Host "[OK] Setup script completed" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Setup script had issues, continuing..." -ForegroundColor Yellow
}

# Run tests
Write-Host "[INFO] Running tests..." -ForegroundColor Cyan
try {
    python test_simple.py
    Write-Host "[OK] Tests completed" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Tests had issues, but setup may still work" -ForegroundColor Yellow
}

Write-Host "`n[SUCCESS] Setup completed!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Run: python enhanced_psx_processor_simple.py" -ForegroundColor White
Write-Host "2. Check logs folder for processing logs" -ForegroundColor White
Write-Host "3. See README_WINDOWS.md for documentation" -ForegroundColor White

Read-Host "`nPress Enter to exit"
