import os
import sys
from pathlib import Path

class PathResolver:
    def __init__(self):
        self.base_path = self._get_base_path()
        
    def _get_base_path(self):
        """Resolve the base application path dynamically"""
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            return Path(sys.executable).parent
        else:
            # Running as script
            return Path(__file__).parent.parent
            
    def resolve(self, *path_parts):
        """Resolve path relative to application root"""
        return str(self.base_path.joinpath(*path_parts))
        
    def ensure_path_exists(self, path):
        """Ensure directory exists for given path"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return path

# Singleton instance for application-wide use
path_resolver = PathResolver()