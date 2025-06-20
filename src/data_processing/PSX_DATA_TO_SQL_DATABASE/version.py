"""
Version information for PSX Data to SQL Database module
"""

__version__ = "2.0.0"
__version_info__ = (2, 0, 0)
__author__ = "PSX Data Team"
__email__ = "data@psx.com"
__description__ = "Enhanced PSX Data Download and Processing System"
__url__ = "https://github.com/psx-data/enhanced-downloader"

# Release information
RELEASE_DATE = "2025-06-19"
RELEASE_NAME = "Enhanced Production Release"

# Compatibility information
PYTHON_REQUIRES = ">=3.8"
REQUIRES = [
    "pandas>=2.2.1",
    "numpy>=1.26.4",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.3",
    "sqlalchemy>=2.0.27",
    "pyyaml>=6.0.1",
    "python-dateutil>=2.8.2",
    "tqdm>=4.66.1"
]

# Feature flags
FEATURES = {
    "enhanced_monitoring": True,
    "data_validation": True,
    "health_checks": True,
    "metrics_collection": True,
    "automatic_backup": True,
    "threading_optimization": True,
    "connection_pooling": True
}

# Component versions
COMPONENTS = {
    "data_reader": "2.0.0",
    "api_client": "1.5.0",
    "database_manager": "1.3.0",
    "data_validator": "1.2.0",
    "monitoring": "1.1.0",
    "config_manager": "1.0.0"
}

def get_version_string():
    """Get formatted version string"""
    return f"PSX Data to SQL Database v{__version__} ({RELEASE_NAME})"

def get_system_info():
    """Get system and version information"""
    import sys
    import platform
    
    return {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "features": FEATURES,
        "components": COMPONENTS
    }

if __name__ == "__main__":
    print(get_version_string())
    print(f"Release Date: {RELEASE_DATE}")
    print(f"Python Required: {PYTHON_REQUIRES}")
    print(f"Author: {__author__}")
    
    info = get_system_info()
    print("\nSystem Information:")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
