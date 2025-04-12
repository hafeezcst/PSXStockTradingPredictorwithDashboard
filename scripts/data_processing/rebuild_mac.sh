#!/bin/bash

# Save the original directory
ORIGINAL_DIR=$(pwd)
# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Get the project root directory (assuming script is in scripts/data_processing)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Clean up existing PySimpleGUI installation
pip uninstall -y PySimpleGUI
pip cache purge

# Install correct PySimpleGUI version
pip install --force-reinstall --extra-index-url https://PySimpleGUI.net/install PySimpleGUI

# Install other dependencies
pip install -r requirements.txt

# Ensure database directories exist in multiple potential locations
echo "Creating database directories..."
# In project root
mkdir -p "$PROJECT_ROOT/data/database"
chmod 755 "$PROJECT_ROOT/data" "$PROJECT_ROOT/data/database"

# In script directory
mkdir -p "$SCRIPT_DIR/data/database"
chmod 755 "$SCRIPT_DIR/data" "$SCRIPT_DIR/data/database"

# In current directory
mkdir -p "data/database"
chmod 755 "data" "data/database"

# Run the database download script from the project root
echo "Running database download script from project root..."
cd "$PROJECT_ROOT"
echo "Current directory: $(pwd)"
python "$SCRIPT_DIR/psx_database_data_download.py"
DB_RESULT=$?

# If it failed, try from the script directory
if [ $DB_RESULT -ne 0 ]; then
    echo "Failed when run from project root, trying from script directory..."
    cd "$SCRIPT_DIR"
    echo "Current directory: $(pwd)"
    python psx_database_data_download.py
    DB_RESULT=$?
fi

# Check final result
if [ $DB_RESULT -ne 0 ]; then
    echo "Database download failed after multiple attempts - aborting build"
    exit 1
fi

# Return to original directory
cd "$ORIGINAL_DIR"

# Clean previous build artifacts
rm -rf build dist __pycache__
mkdir -p dist

# Rebuild the application with clean flags
pyinstaller psx_analysis.spec --noconfirm --clean

# Only create DMG if .app bundle exists
if [ -d "dist/PSX Analysis.app" ]; then
    # Remove any resource forks/Finder info
    xattr -cr "dist/PSX Analysis.app"
    
    # Create DMG package
    hdiutil create -volname "PSX Analysis" -srcfolder dist/PSX\ Analysis.app -ov -format UDZO dist/PSX_Analysis.dmg
    echo "Rebuild complete. DMG package created in dist/PSX_Analysis.dmg"
else
    echo "Rebuild failed - PSX Analysis.app bundle not created"
    exit 1
fi