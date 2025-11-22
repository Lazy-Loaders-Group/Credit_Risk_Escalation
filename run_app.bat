@echo off
REM Script to run the Credit Risk Assessment Streamlit Application

echo Starting Credit Risk Assessment Application...
echo ================================================
echo.
echo Activating virtual environment...
call uom_venv\Scripts\activate.bat

echo Launching Streamlit app...
echo The app will open in your default web browser.
echo.
echo Press Ctrl+C to stop the application.
echo.

streamlit run app.py
