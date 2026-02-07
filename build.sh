#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Build Frontend
echo "Building Frontend..."
cd frontend
npm install
CI=false npm run build
cd ..

# 2. Install Backend Dependencies
echo "Installing Backend Dependencies..."
pip install -r requirements.txt
