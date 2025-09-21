#!/bin/bash

# Exit on error
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install system dependencies for pytesseract if needed
# (These are usually handled by Render's default Python environment)

echo "Build completed successfully!"