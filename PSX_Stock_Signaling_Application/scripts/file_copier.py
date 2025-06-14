import json
import shutil
from pathlib import Path
from config.path_resolver import path_resolver

class FileCopier:
    def __init__(self, mapping_file="file_mapping.json"):
        self.mapping = self._load_mapping(mapping_file)
        
    def _load_mapping(self, mapping_file):
        """Load the file mapping configuration"""
        mapping_path = path_resolver.resolve(mapping_file)
        with open(mapping_path, 'r') as f:
            return json.load(f)
            
    def copy_files(self):
        """Execute all file copies according to mapping"""
        for category, mappings in self.mapping.items():
            for src, dest in mappings.items():
                self._copy_file(src, dest)
                
    def _copy_file(self, src, dest):
        """Copy single file with path resolution"""
        src_path = Path(src)
        dest_path = path_resolver.resolve(dest)
        
        # Ensure destination directory exists
        path_resolver.ensure_path_exists(dest_path)
        
        # Copy file
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")

if __name__ == "__main__":
    copier = FileCopier()
    copier.copy_files()