#!/bin/bash
# Setup script for Credit Risk Escalation System (macOS/Linux)
# This script sets up the environment and installs all dependencies

set -e  # Exit on error

echo "=================================================="
echo "Credit Risk Escalation System - Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if uom_venv exists
if [ -d "uom_venv" ]; then
    echo ""
    echo "✓ Found existing virtual environment 'uom_venv'"
    echo "  Using existing environment..."
    VENV_DIR="uom_venv"
else
    echo ""
    echo "Creating new virtual environment '.venv'..."
    python3 -m venv .venv
    VENV_DIR=".venv"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "Installing required packages..."
pip install -r requirements.txt --quiet

echo ""
echo "=================================================="
echo "✅ Setup completed successfully!"
echo "=================================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To test the system:"
echo "  1. Command line: python predict_new_loan.py --interactive"
echo "  2. Web app:      streamlit run app.py"
echo ""
echo "Note: Make sure you have trained models in results/models/"
echo "      Run the training notebooks first if needed."
echo ""
