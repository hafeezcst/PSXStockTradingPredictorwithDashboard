import sys
import subprocess
from pathlib import Path
from scripts.file_copier import FileCopier

def check_dependencies():
    """Verify required Python packages are installed"""
    required = ['pandas', 'numpy', 'requests', 'python-telegram-bot', 'sqlalchemy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
            
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

def initialize_application():
    """Set up application structure"""
    print("Initializing PSX Stock Signaling Application...")
    
    # Copy files to new structure
    copier = FileCopier()
    copier.copy_files()
    
    # Create additional required directories
    Path("logs").mkdir(exist_ok=True)
    Path("temp").mkdir(exist_ok=True)
    
    print("Setup completed successfully")

if __name__ == "__main__":
    check_dependencies()
    initialize_application()