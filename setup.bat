@echo off
REM Setup script for Credit Risk Escalation System (Windows)
REM This script sets up the environment and installs all dependencies

echo ==================================================
echo Credit Risk Escalation System - Setup
echo ==================================================
echo.

REM Check Python version
echo Checking Python version...
python --version

REM Check if uom_venv exists
if exist "uom_venv\" (
    echo.
    echo Found existing virtual environment 'uom_venv'
    echo Using existing environment...
    set VENV_DIR=uom_venv
) else (
    echo.
    echo Creating new virtual environment '.venv'...
    python -m venv .venv
    set VENV_DIR=.venv
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install requirements
echo.
echo Installing required packages...
pip install -r requirements.txt --quiet

echo.
echo ==================================================
echo Setup completed successfully!
echo ==================================================
echo.
echo To activate the environment, run:
echo   %VENV_DIR%\Scripts\activate.bat
echo.
echo To test the system:
echo   1. Command line: python predict_new_loan.py --interactive
echo   2. Web app:      streamlit run app.py
echo.
echo Note: Make sure you have trained models in results\models\
echo       Run the training notebooks first if needed.
echo.
pause
