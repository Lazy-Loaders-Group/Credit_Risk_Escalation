# 🚀 Setup Guide - Credit Risk Escalation System

**Complete installation and environment setup guide**

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Setup](#quick-setup)
3. [Detailed Setup Instructions](#detailed-setup-instructions)
4. [Verification](#verification)
5. [Dataset Setup](#dataset-setup)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- [ ] Python 3.8+ installed
- [ ] Git installed (to clone repository)
- [ ] At least 4GB free RAM
- [ ] At least 2GB free disk space
- [ ] Internet connection (for downloading packages)

### Check Python Version
```bash
python --version
# Should show Python 3.8 or higher
```

---

## Quick Setup

### macOS/Linux - One Command Setup

```bash
# Navigate to project directory
cd /Users/nadunhettiarachchi/Documents/Credit_Risk_Escalation

# Run setup script (creates venv, installs packages)
bash setup.sh

# Activate virtual environment
source uom_venv/bin/activate

# Verify installation
python -c "import pandas, sklearn, xgboost, shap; print('✅ Setup complete!')"
```

### Windows - Quick Setup

```cmd
REM Navigate to project directory
cd C:\path\to\Credit_Risk_Escalation

REM Run setup script
setup.bat

REM Activate virtual environment
uom_venv\Scripts\activate

REM Verify installation
python -c "import pandas, sklearn, xgboost, shap; print('✅ Setup complete!')"
```

---

## Detailed Setup Instructions

### Step 1: Clone or Download Project

#### Option A: Clone from GitHub
```bash
git clone https://github.com/Lazy-Loaders-Group/Credit_Risk_Escalation.git
cd Credit_Risk_Escalation
```

#### Option B: If Already Downloaded
```bash
# Navigate to project folder
cd path/to/Credit_Risk_Escalation
```

---

### Step 2: Create Virtual Environment

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv uom_venv

# Activate virtual environment
source uom_venv/bin/activate

# You should see (uom_venv) in your terminal prompt
```

#### Windows
```cmd
REM Create virtual environment
python -m venv uom_venv

REM Activate virtual environment
uom_venv\Scripts\activate

REM You should see (uom_venv) in your command prompt
```

---

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This will install:
- **Data Processing:** pandas (2.3.3), numpy (2.3.4)
- **Machine Learning:** scikit-learn (1.7.2), xgboost (3.0.5)
- **Visualization:** matplotlib (3.10.7), seaborn (0.13.2)
- **Notebooks:** jupyter (1.1.1)
- **Interpretability:** shap (0.49.1), lime (0.2.0.1)
- **Imbalanced Data:** imbalanced-learn (0.14.0)

**Expected output:**
```
Successfully installed pandas-2.3.3 numpy-2.3.4 scikit-learn-1.7.2 ...
```

---

## Verification

### Verify Package Installation

```bash
# Test import of key packages
python -c "
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
from imblearn.over_sampling import SMOTE
print('✅ All packages imported successfully!')
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'scikit-learn version: {sklearn.__version__}')
print(f'xgboost version: {xgb.__version__}')
"
```

**Expected output:**
```
✅ All packages imported successfully!
pandas version: 2.3.3
numpy version: 2.3.4
scikit-learn version: 1.7.2
xgboost version: 3.0.5
```

### Verify Jupyter Installation

```bash
# Check Jupyter
jupyter --version

# Should show:
# jupyter core     : 1.1.1
# jupyter-notebook : 7.x.x
```

### Start Jupyter (Optional Test)

```bash
# Launch Jupyter Notebook
jupyter notebook

# This should open http://localhost:8888 in your browser
# Press Ctrl+C in terminal to stop Jupyter when done testing
```

---

## Dataset Setup

### Extract Dataset

The dataset is in ZIP format and needs to be extracted:

```bash
# Navigate to data directory
cd data/raw

# Extract the dataset (macOS/Linux)
unzip LC_loans_granting_model_dataset.csv.zip

# OR on Windows, right-click the ZIP file and select "Extract Here"

# Verify extraction
ls -lh LC_loans_granting_model_dataset.csv
# Should show ~100MB+ file
```

### Verify Dataset

```bash
# Quick check of dataset
python -c "
import pandas as pd
import os

data_file = 'data/raw/LC_loans_granting_model_dataset.csv'
if os.path.exists(data_file):
    df = pd.read_csv(data_file, nrows=5)
    print(f'✅ Dataset found!')
    print(f'Shape (first 5 rows): {df.shape}')
    print(f'Columns: {list(df.columns)}')
else:
    print('❌ Dataset not found. Please extract the ZIP file.')
"
```

**Expected output:**
```
✅ Dataset found!
Shape (first 5 rows): (5, 15)
Columns: ['id', 'FICO.Score', 'Employment.Length', ...]
```

---

## Troubleshooting

### Issue 1: "python not found"

**Solution:**
```bash
# Try python3 instead
python3 --version

# If still not working, install Python from python.org
# Download from: https://www.python.org/downloads/
```

---

### Issue 2: "pip not found" or "pip install fails"

**Solution:**
```bash
# macOS/Linux: Upgrade pip
python3 -m pip install --upgrade pip

# Windows: Upgrade pip
python -m pip install --upgrade pip

# If still failing, install pip manually:
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

---

### Issue 3: "Permission denied" when creating virtual environment

**Solution (macOS/Linux):**
```bash
# Use sudo (if needed)
sudo python3 -m venv uom_venv

# Or change directory permissions
sudo chmod -R 755 /path/to/Credit_Risk_Escalation
```

**Solution (Windows):**
- Run Command Prompt as Administrator
- Then run the setup commands

---

### Issue 4: "Module not found" after installation

**Solution:**
```bash
# Make sure virtual environment is activated
# You should see (uom_venv) in your prompt

# If not activated:
source uom_venv/bin/activate  # macOS/Linux
uom_venv\Scripts\activate     # Windows

# Reinstall packages
pip install -r requirements.txt
```

---

### Issue 5: Jupyter doesn't start

**Solution:**
```bash
# Reinstall Jupyter
pip uninstall jupyter notebook
pip install jupyter notebook

# Try starting again
jupyter notebook

# Alternative: Use JupyterLab
pip install jupyterlab
jupyter lab
```

---

### Issue 6: "No space left on device" during installation

**Solution:**
```bash
# Check available disk space
df -h  # macOS/Linux
wmic logicaldisk get size,freespace,caption  # Windows

# Free up space:
# - Delete temporary files
# - Clear pip cache: pip cache purge
# - Uninstall unused packages
```

---

### Issue 7: Installation is very slow

**Solution:**
```bash
# Use faster mirrors (if outside US)
pip install -r requirements.txt --index-url https://pypi.org/simple

# Or install packages individually:
pip install pandas numpy scikit-learn xgboost
pip install matplotlib seaborn jupyter
pip install shap lime imbalanced-learn
```

---

### Issue 8: "SSL Certificate Error"

**Solution:**
```bash
# Option 1: Update certificates (recommended)
pip install --upgrade certifi

# Option 2: Use trusted host (temporary fix)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

## Environment Management

### Activate Virtual Environment

**Every time you work on the project**, activate the environment:

```bash
# macOS/Linux
source uom_venv/bin/activate

# Windows
uom_venv\Scripts\activate

# You should see (uom_venv) in your prompt
```

### Deactivate Virtual Environment

```bash
# When done working
deactivate

# Your prompt will return to normal (no (uom_venv))
```

### Update Packages

```bash
# Activate environment first
source uom_venv/bin/activate

# Update a specific package
pip install --upgrade package_name

# Example: Update pandas
pip install --upgrade pandas

# Update all packages (use with caution)
pip list --outdated
pip install --upgrade pandas numpy scikit-learn
```

---

## Quick Reference Commands

### Daily Workflow Commands

```bash
# 1. Navigate to project
cd /path/to/Credit_Risk_Escalation

# 2. Activate environment
source uom_venv/bin/activate  # macOS/Linux
uom_venv\Scripts\activate     # Windows

# 3. Start Jupyter
jupyter notebook

# 4. When done, deactivate
deactivate
```

### Check Installed Packages

```bash
# List all installed packages
pip list

# Check specific package version
pip show pandas

# Export current environment
pip freeze > requirements_current.txt
```

### Clean Installation (Fresh Start)

```bash
# Remove virtual environment
rm -rf uom_venv  # macOS/Linux
rmdir /s uom_venv  # Windows

# Create new environment
python3 -m venv uom_venv

# Activate and reinstall
source uom_venv/bin/activate
pip install -r requirements.txt
```

---

## System Requirements

### Minimum Requirements
- **CPU:** Dual-core processor (2+ GHz)
- **RAM:** 4GB
- **Storage:** 2GB free space
- **OS:** macOS, Linux, or Windows 10+
- **Python:** 3.8+

### Recommended Requirements
- **CPU:** Quad-core processor (2.5+ GHz)
- **RAM:** 8GB (for faster processing)
- **Storage:** 5GB free space
- **OS:** macOS, Linux, or Windows 10+
- **Python:** 3.10+

### Execution Time Estimates

| Task | Minimum Specs | Recommended Specs |
|------|--------------|-------------------|
| Setup | 10-15 min | 5-10 min |
| Notebook 2 | 30-40 min | 20-30 min |
| Notebook 3 | 60-90 min | 40-60 min |
| Notebook 4 | 20-25 min | 15-20 min |
| Notebook 5 | 40-50 min | 30-40 min |

---

## Next Steps

### After Successful Setup:

1. ✅ **Environment is ready** - Virtual environment activated
2. ✅ **Packages installed** - All imports working
3. ✅ **Dataset extracted** - CSV file in data/raw/
4. ✅ **Jupyter working** - Can launch notebooks

### What to Do Next:

1. **Read PROGRESS.md** - See current project status and results
2. **Review notebooks** - Start with `01_data_exploration_executed.ipynb`
3. **Run the pipeline** - Execute notebooks 02 → 03 → 04 → 05
4. **Check results** - View outputs in `results/` folder

### Quick Start Commands:

```bash
# 1. Activate environment
source uom_venv/bin/activate

# 2. Launch Jupyter
jupyter notebook

# 3. Open and run notebooks in order:
#    - 01_data_exploration_executed.ipynb (review only)
#    - 02_baseline_model.ipynb (run first)
#    - 03_uncertainty_quantification.ipynb (run second)
#    - 04_escalation_system.ipynb (run third)
#    - 05_comprehensive_evaluation.ipynb (run fourth)
```

---

## Additional Resources

### Python Package Documentation
- **pandas:** https://pandas.pydata.org/docs/
- **scikit-learn:** https://scikit-learn.org/stable/
- **XGBoost:** https://xgboost.readthedocs.io/
- **SHAP:** https://shap.readthedocs.io/
- **Jupyter:** https://jupyter.org/documentation

### Helpful Tutorials
- Virtual environments: https://docs.python.org/3/tutorial/venv.html
- pip usage: https://pip.pypa.io/en/stable/user_guide/
- Jupyter notebooks: https://jupyter-notebook.readthedocs.io/

---

## Success Checklist

Mark off as you complete:

- [ ] Python 3.8+ installed and verified
- [ ] Virtual environment created (`uom_venv`)
- [ ] Virtual environment activated (see `(uom_venv)` in prompt)
- [ ] pip upgraded to latest version
- [ ] All packages installed from `requirements.txt`
- [ ] Package imports verified (no errors)
- [ ] Jupyter Notebook working
- [ ] Dataset extracted from ZIP file
- [ ] Dataset verified (can load with pandas)
- [ ] Ready to run notebooks!

---

## Contact & Support

**Questions?**
- Check troubleshooting section above
- Review README.md for project overview
- Read PROGRESS.md for current status
- Consult PROJECT_GUIDE.md for detailed plan

**Team:** Lazy Loaders Group  
**Project:** Credit Risk Escalation System  
**Last Updated:** November 2025

---

**🎉 Setup Complete! You're ready to build an AI system!** 🚀
