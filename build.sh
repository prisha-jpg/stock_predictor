#!/bin/bash

# Exit on error
set -o errexit

echo "Starting build process..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Build completed successfully!"