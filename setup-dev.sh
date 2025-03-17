#!/bin/bash

# Exit on error
set -e

echo "Setting up development environment..."

# Create a virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python -m venv env
fi

# Activate the virtual environment
source env/bin/activate

# Install dependencies
echo "Installing development dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

echo "Development environment setup complete!"
echo "You can now run 'pre-commit run --all-files' to check all files in the repository."
