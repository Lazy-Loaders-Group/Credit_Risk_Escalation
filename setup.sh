#!/bin/bash
python3 -m venv uom_venv
source uom_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete! To activate your environment, run: source uom_venv/bin/activate"