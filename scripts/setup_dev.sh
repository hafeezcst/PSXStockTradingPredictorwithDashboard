#!/bin/bash

# Exit on error
set -e

echo "Setting up development environment..."

# Create necessary directories
mkdir -p data/raw data/processed logs outputs exports

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch logs/.gitkeep
touch outputs/.gitkeep
touch exports/.gitkeep

# Install pre-commit hooks
pre-commit install

# Run initial linting
echo "Running initial linting..."
make lint

echo "Development environment setup complete!" 