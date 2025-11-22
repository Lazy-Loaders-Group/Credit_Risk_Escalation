@echo off
REM Smoke test for Credit Risk Escalation System (Windows)
REM Tests the full application using saved models (no retraining)

echo ==================================================
echo Credit Risk Escalation - Smoke Test
echo ==================================================
echo.

REM Track test results
set TESTS_PASSED=0
set TESTS_FAILED=0

REM Determine which venv to use
if exist "uom_venv\" (
    set VENV_DIR=uom_venv
) else if exist ".venv\" (
    set VENV_DIR=.venv
) else (
    echo Error: No virtual environment found
    echo Please run setup.bat first
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
echo.

REM Test 1: Check if models exist
echo Test 1: Checking for saved models...
if exist "results\models\bootstrap_ensemble.pkl" (
    if exist "results\models\preprocessor.pkl" (
        if exist "results\models\escalation_system.pkl" (
            echo [PASS] Models exist
            set /a TESTS_PASSED+=1
        ) else (
            goto :models_missing
        )
    ) else (
        goto :models_missing
    )
) else (
    :models_missing
    echo [FAIL] Models not found
    echo.
    echo Models need to be trained first. Run:
    echo   python train_and_save.py
    set /a TESTS_FAILED+=1
    exit /b 1
)
echo.

REM Test 2: Check model metadata
echo Test 2: Checking model metadata...
if exist "results\models\model_metadata.json" (
    echo [PASS] Metadata exists
    set /a TESTS_PASSED+=1
) else (
    echo [WARN] Metadata file not found (optional)
)
echo.

REM Test 3: Run prediction on example file
echo Test 3: Running prediction on example_new_loans.csv...
set OUTPUT_FILE=results\reports\predictions_smoke.csv

REM Ensure reports directory exists
if not exist "results\reports\" mkdir results\reports

REM Run prediction
python predict_and_decide.py --input example_new_loans.csv --output %OUTPUT_FILE%
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Prediction script executed
    set /a TESTS_PASSED+=1
    
    REM Check if output file was created
    if exist "%OUTPUT_FILE%" (
        echo [PASS] Output file created
        set /a TESTS_PASSED+=1
        
        REM Show sample output
        echo.
        echo Sample output (first 5 rows):
        powershell -Command "Get-Content %OUTPUT_FILE% -TotalCount 5"
    ) else (
        echo [FAIL] Output file not created
        set /a TESTS_FAILED+=1
    )
) else (
    echo [FAIL] Prediction script failed
    set /a TESTS_FAILED+=1
)
echo.

REM Test 4: Test decision logic import
echo Test 4: Testing decision logic import...
python -c "from src.escalation_system import EscalationSystem; es = EscalationSystem(); print('OK')" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Decision logic imports correctly
    set /a TESTS_PASSED+=1
) else (
    echo [FAIL] Cannot import decision logic
    set /a TESTS_FAILED+=1
)
echo.

REM Test 5: Verify Streamlit app can be imported
echo Test 5: Checking Streamlit app...
python -c "import app" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Streamlit app imports correctly
    set /a TESTS_PASSED+=1
) else (
    echo [WARN] Streamlit app import failed (may need manual testing)
)
echo.

REM Print summary
echo ==================================================
echo Smoke Test Summary
echo ==================================================
echo Tests Passed: %TESTS_PASSED%
echo Tests Failed: %TESTS_FAILED%
echo ==================================================
echo.

if %TESTS_FAILED% EQU 0 (
    echo All smoke tests passed!
    echo.
    echo The system is ready to use:
    echo   - Run predictions: python predict_and_decide.py --input example_new_loans.csv
    echo   - Start web UI:    streamlit run app.py
    echo   - Run tests:       python -m pytest tests\ -v
    echo.
    exit /b 0
) else (
    echo Some smoke tests failed
    echo.
    echo Please check the errors above and fix them before using the system.
    echo.
    exit /b 1
)
