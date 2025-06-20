"""
Setup and Installation Script for PSX Data to SQL Database Module

This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating necessary directories...")
    
    directories = [
        "data/databases/production",
        "logs",
        "backups"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ Created: {directory}")
        except Exception as e:
            print(f"  ❌ Failed to create {directory}: {e}")

def validate_installation():
    """Validate the installation"""
    print("🔍 Validating installation...")
    
    try:
        # Test imports
        import pandas
        import numpy
        import requests
        import yaml
        import sqlalchemy
        print("  ✅ All core packages imported successfully")
        
        # Test configuration loading
        from config_manager import AppConfig
        print("  ✅ Configuration manager loaded")
        
        # Test other modules
        from exceptions import PSXDataDownloadError
        from monitoring import MetricsCollector
        from data_validator import DataValidator
        print("  ✅ All custom modules loaded successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 PSX Data to SQL Database - Setup Script")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        return 1
    
    # Create directories
    create_directories()
    
    # Validate installation
    if not validate_installation():
        print("❌ Setup failed during validation")
        return 1
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review and update config.yaml if needed")
    print("2. Run: python run_psx_download.py")
    print("3. Or run tests: python test_enhanced_reader.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
