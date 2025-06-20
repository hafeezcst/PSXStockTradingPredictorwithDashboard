#!/usr/bin/env python3
"""
Setup script for Enhanced PSX Indicator Processor

This script helps set up the enhanced processor environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"[ERROR] Python 3.9+ required, found {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_requirements():
    """Install required packages."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"[ERROR] Requirements file not found: {requirements_file}")
        return False
    
    command = f'"{sys.executable}" -m pip install -r "{requirements_file}"'
    return run_command(command, "Installing requirements")

def setup_directories():
    """Create necessary directories."""
    base_dir = Path(__file__).parent
    directories = [
        base_dir / "exports" / "csv",
        base_dir / "exports" / "parquet", 
        base_dir / "exports" / "json",
        base_dir / "logs",
        base_dir / "config"
    ]
    
    print("[INFO] Creating directories...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  [FOLDER] {directory}")
    
    print("[OK] Directories created successfully")
    return True

def create_sample_config():
    """Create a sample configuration file if it doesn't exist."""
    config_path = Path(__file__).parent / "config" / "sample_config.yaml"
    
    if config_path.exists():
        print("[OK] Sample configuration already exists")
        return True
    
    sample_config = """# Enhanced PSX Indicator Processor Configuration
# Copy this file and modify as needed

# Performance settings
max_workers: 4  # Adjust based on your CPU cores
batch_size: 1000
use_gpu: false  # Set to true if you have CUDA setup
cache_size: 128

# Processing options
calculate_advanced_indicators: true
include_ml_features: true
enable_data_validation: true

# Export formats
export_formats:
  - sqlite
  # - csv      # Uncomment to export CSV
  # - parquet  # Uncomment to export Parquet

# Indicator parameters
rsi_periods: [9, 14, 21, 26]
ma_periods: [20, 30, 50, 100, 200]
bollinger_period: 20
bollinger_std: 2.0

# Database paths (null = use defaults)
source_db_path: null
target_db_path: null
"""
    
    try:
        with open(config_path, 'w') as f:
            f.write(sample_config)
        print(f"[OK] Sample configuration created: {config_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create sample configuration: {e}")
        return False

def test_installation():
    """Test if the installation works."""
    print("[INFO] Testing installation...")
    
    test_script = '''
import sys
import pandas as pd
import numpy as np

try:
    from enhanced_psx_processor_simple import PSXIndicatorProcessor, ProcessorConfig
    print("[OK] Main modules imported successfully")
    
    # Test configuration
    config = ProcessorConfig()
    print("[OK] Configuration created successfully")
    
    # Test basic functionality
    processor = PSXIndicatorProcessor(config)
    stats = processor.get_processing_stats()
    print("[OK] Processor initialized successfully")
    print(f"   CPU cores detected: {stats['system']['cpu_count']}")
    print(f"   TA available: {stats['system']['ta_available']}")
    
    print("[OK] Installation test passed!")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    sys.exit(1)
'''
    
    try:
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Installation test failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Main setup function."""
    print("[SETUP] Enhanced PSX Indicator Processor Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup directories
    if not setup_directories():
        return False
    
    # Install requirements
    if not install_requirements():
        print("[ERROR] Failed to install requirements. Please install manually:")
        print(f"   pip install -r {Path(__file__).parent / 'requirements.txt'}")
        return False
    
    # Create sample configuration
    if not create_sample_config():
        return False
    
    # Test installation
    if not test_installation():
        return False
    
    print("\n[SUCCESS] Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy and modify config/sample_config.yaml as needed")
    print("2. Run: python enhanced_psx_processor_simple.py")
    print("3. Or run: python test_simple.py")
    print("\nFor advanced usage:")
    print("1. See usage_examples.py for examples")
    print("2. Check README_WINDOWS.md for documentation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
