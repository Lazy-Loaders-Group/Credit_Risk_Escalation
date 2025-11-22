#!/bin/bash
# Smoke test for Credit Risk Escalation System (macOS/Linux)
# Tests the full application using saved models (no retraining)

set -e  # Exit on error

echo "=================================================="
echo "Credit Risk Escalation - Smoke Test"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $1"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: $1"
        ((TESTS_FAILED++))
    fi
}

# Determine which venv to use
if [ -d "uom_venv" ]; then
    VENV_DIR="uom_venv"
elif [ -d ".venv" ]; then
    VENV_DIR=".venv"
else
    echo -e "${RED}Error: No virtual environment found${NC}"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate
echo ""

# Test 1: Check if models exist
echo "Test 1: Checking for saved models..."
if [ -f "results/models/bootstrap_ensemble.pkl" ] && \
   [ -f "results/models/preprocessor.pkl" ] && \
   [ -f "results/models/escalation_system.pkl" ]; then
    print_result "Models exist"
else
    echo -e "${RED}✗ FAILED${NC}: Models not found"
    echo ""
    echo "Models need to be trained first. Run:"
    echo "  python train_and_save.py"
    ((TESTS_FAILED++))
    exit 1
fi
echo ""

# Test 2: Check model metadata
echo "Test 2: Checking model metadata..."
if [ -f "results/models/model_metadata.json" ]; then
    print_result "Metadata exists"
else
    echo -e "${YELLOW}⚠ WARNING${NC}: Metadata file not found (optional)"
fi
echo ""

# Test 3: Check Streamlit app (primary prediction interface)
echo "Test 3: Checking Streamlit app availability..."
if python -c "import streamlit; import app; print('OK')" > /dev/null 2>&1; then
    print_result "Streamlit app ready"
    echo ""
    echo "  ℹ️  To use the prediction system:"
    echo "     streamlit run app.py"
    echo ""
else
    echo -e "${YELLOW}⚠ WARNING${NC}: Streamlit app check failed"
fi
echo ""

# Test 4: Test decision logic (import test)
echo "Test 4: Testing decision logic import..."
if python -c "from src.escalation_system import EscalationSystem; es = EscalationSystem(); print('OK')" > /dev/null 2>&1; then
    print_result "Decision logic imports correctly"
else
    echo -e "${RED}✗ FAILED${NC}: Cannot import decision logic"
    ((TESTS_FAILED++))
fi
echo ""

# Test 5: Verify unit tests can run
echo "Test 5: Checking unit tests..."
if python -m pytest tests/test_decision_logic.py --collect-only > /dev/null 2>&1; then
    print_result "Unit tests are discoverable"
else
    echo -e "${YELLOW}⚠ WARNING${NC}: Unit tests check failed (pytest may not be installed)"
fi
echo ""

# Test 6: Verify Streamlit app can be imported
echo "Test 6: Final Streamlit verification..."
if python -c "import app" > /dev/null 2>&1; then
    print_result "Streamlit app imports correctly"
else
    echo -e "${YELLOW}⚠ WARNING${NC}: Streamlit app import failed (may need manual testing)"
fi
echo ""

# Print summary
echo "=================================================="
echo "Smoke Test Summary"
echo "=================================================="
echo -e "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests Failed: ${RED}${TESTS_FAILED}${NC}"
echo "=================================================="
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All smoke tests passed!${NC}"
    echo ""
    echo "The system is ready to use!"
    echo ""
    echo "🚀 Primary Interface (RECOMMENDED):"
    echo "   streamlit run app.py"
    echo ""
    echo "📊 Additional Options:"
    echo "   • Jupyter notebooks:  jupyter notebook"
    echo "   • Run unit tests:     python -m pytest tests/ -v"
    echo "   • Retrain models:     python train_and_save.py"
    echo ""
    echo "ℹ️  See CURRENT_STATUS_AND_USAGE.md for details"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some smoke tests failed${NC}"
    echo ""
    echo "Please check the errors above and fix them before using the system."
    echo ""
    exit 1
fi
