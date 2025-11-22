# 💳 Credit Risk Escalation System

**An intelligent ML system that automates loan decisions with uncertainty-aware escalation to human reviewers**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This system uses **machine learning with uncertainty quantification** to:
- ✅ **Automate 70-85%** of loan decisions with high confidence
- 🔴 **Escalate 15-30%** of uncertain cases to human agents
- 📊 **Achieve >85%** accuracy on automated decisions
- 💰 **Save ~20%** costs through intelligent automation

### Key Innovation
Unlike traditional ML systems that force predictions on all cases, this system **knows when it doesn't know** and routes uncertain cases to human experts.

---

## ⚡ Quick Start

### For First-Time Setup:

- **⭐ [ACTION_PLAN.md](ACTION_PLAN.md)** - **YOUR STEP-BY-STEP GUIDE!** Clear action items to complete the project
- **🔧 [SETUP.md](SETUP.md)** - Complete setup and installation guide (if you need to reinstall)
- **📖 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - Comprehensive project reference and methodology

### What's Included:

- ✅ Step-by-step setup instructions (10 minutes)
- ✅ Complete execution workflow (2-3 hours)
- ✅ Troubleshooting for common issues
- ✅ How to verify your results
- ✅ How to use the trained system on new data
- ✅ Git repository optimization guide
- ✅ Documentation reorganization notes

---

## 📋 Initial Setup

### Prerequisites

- Python 3.8 or higher
- At least 4GB RAM
- At least 2GB free disk space

### macOS/Linux

```bash
# 1. Clone repository
git clone https://github.com/Lazy-Loaders-Group/Credit_Risk_Escalation.git
cd Credit_Risk_Escalation

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Activate environment
source uom_venv/bin/activate  # or .venv/bin/activate

# 4. Start Jupyter
jupyter notebook
```

### To Use Existing Setup:

```bash
# Activate environment
cd Credit_Risk_Escalation
source uom_venv/bin/activate

# Run evaluation notebook
jupyter notebook notebooks/05_comprehensive_evaluation.ipynb
```

### Windows

```cmd
REM 1. Clone repository (if not already done)
git clone https://github.com/Lazy-Loaders-Group/Credit_Risk_Escalation.git
cd Credit_Risk_Escalation

REM 2. Run setup script
setup.bat

REM 3. Activate virtual environment
uom_venv\Scripts\activate

REM 4. Verify installation
python -c "import pandas, sklearn, xgboost, shap; print('✅ Setup complete!')"
```

**⚠️ Important:** The dataset is in `data/raw` folder in .zip format. Extract it before running the notebooks.

If you don't have the dataset locally, download it from the Google Drive folder below and place the extracted files into the `data/raw/` directory:

- Download link: [Google Drive - Dataset](https://drive.google.com/drive/folders/1hAWSmFFxp2n_5ET2MR2HfL5O41otZ47Y?usp=sharing)

Example:

```bash
# after downloading and extracting, ensure the CSV(s) live in data/raw/
ls data/raw
```

---

## 📊 System Architecture

```
New Loan Application
        ↓
[1] Data Preprocessing
    • Clean & encode data
    • Scale features
        ↓
[2] Bootstrap Ensemble (30 models)
    • Each model votes
    • Calculate uncertainty from variance
        ↓
[3] Uncertainty Quantification
    • High disagreement = High uncertainty
    • Low disagreement = High confidence
        ↓
[4] Escalation Decision
    IF high uncertainty OR low confidence:
        → 🔴 ESCALATE to human
    ELSE:
        → ✅ AUTOMATED decision
        ↓
[5] Output: Decision + Confidence + Explanation
```

---

## 🚀 Features

- **Bootstrap Ensemble**: 30-model ensemble for robust predictions
- **Uncertainty Quantification**: Measure prediction uncertainty through model variance
- **Intelligent Escalation**: Automatic routing based on confidence thresholds
- **Cost Optimization**: Balance automation vs. human review costs
- **Full Explainability**: SHAP values for model interpretability
- **Performance Tracking**: Comprehensive metrics and visualizations

---

## 📁 Project Structure

```
Credit_Risk_Escalation/
├── notebooks/                # Jupyter notebooks (main interface)
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_uncertainty_quantification.ipynb
│   ├── 04_escalation_system.ipynb
│   └── 05_comprehensive_evaluation.ipynb ⭐
│
├── src/                      # Source code modules
│   ├── data_preprocessing.py
│   ├── uncertainty_quantification.py
│   └── escalation_system.py
│
├── results/models/           # Trained models (generated)
│   ├── preprocessor.pkl
│   ├── bootstrap_ensemble.pkl
│   └── escalation_system.pkl
│
├── data/                     # Data files
│   ├── raw/                  # Original dataset
│   └── splits/               # Train/val/test splits
│
├── predict_new_loan.py       # CLI prediction script (in development)
├── simple_predict.py         # Works with preprocessed data
├── app.py                    # Streamlit web app (in development)
├── setup.sh / setup.bat      # Setup scripts
└── requirements.txt          # Python dependencies
```

---

## 📈 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Automation Rate** | 70-85% | Decisions handled automatically |
| **Escalation Rate** | 15-30% | Cases needing human review |
| **Automated Accuracy** | >85% | Quality of automated decisions |
| **Cost Savings** | ~20% | Reduction in manual review costs |
| **Models** | 30 | Bootstrap ensemble size |

| Metric                     | Target   | Status        |
| -------------------------- | -------- | ------------- |
| **Baseline AUC-ROC**       | >0.75    | 🎯 To achieve |
| **Automation Rate**        | 70-85%   | 🎯 To achieve |
| **Automated Accuracy**     | >85%     | 🎯 To achieve |
| **Cost Savings**           | Positive | 🎯 To achieve |
| **Uncertainty Validation** | Strong   | 🎯 To achieve |

**Potential Business Impact:**

- 💰 ~$678 saved per 210K applications (~21% reduction)
- ⚡ ~78% of decisions automated (only ~22% need human review)
- 🎯 ~89% accuracy on automated decisions (vs ~79% baseline)
- 📊 Full explainability with SHAP values

---

## 🎓 How It Works

### 1. Uncertainty Quantification

The system uses a **bootstrap ensemble** of 30 models:
- Each model is trained on a different sample of data
- For each prediction, all 30 models vote
- **High agreement** → Low uncertainty → Automate
- **Low agreement** → High uncertainty → Escalate

### 2. Escalation Criteria

A case is escalated if ANY of these apply:
- **High Uncertainty**: `uncertainty > 0.1` (models disagree)
- **Low Confidence**: `confidence < 0.7` (not sure either way)
- **Borderline Probability**: `0.4 < probability < 0.6` (near decision boundary)

### 3. Example Results

**Automated Approval** ✅
```
Probability of Default: 25%
Confidence: 85%
Uncertainty: 0.03
→ AUTOMATED APPROVE (no review needed)
```

**Escalated Case** 🔴
```
Probability of Default: 52%
Confidence: 52%
Uncertainty: 0.15
→ ESCALATE TO AGENT (requires human judgment)
```

---

## 📚 Documentation

### Main Guides:

- **[CURRENT_STATUS.md](CURRENT_STATUS.md)** - What's working now ⭐ START HERE
- **[SETUP_README.md](SETUP_README.md)** - Complete setup instructions
- **[HOW_TO_USE.md](HOW_TO_USE.md)** - Usage guide
- **[PREDICTION_GUIDE.md](PREDICTION_GUIDE.md)** - Detailed prediction documentation
- **[SETUP.md](SETUP.md)** - Complete setup and installation guide
- **[PROGRESS.md](PROGRESS.md)** - Project progress and changes tracker
- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - Comprehensive 6-phase project plan
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Implementation summary

### Technical Reports:

- **[results/reports/FINAL_PROJECT_REPORT.md](results/reports/FINAL_PROJECT_REPORT.md)** - Complete technical report
- **[results/reports/phase1_data_quality_report.md](results/reports/phase1_data_quality_report.md)** - Data quality analysis

### Archived Documentation:

- **[archived_docs/](archived_docs/)** - Old documentation files (consolidated into main guides)

---

## 🔧 Prerequisites

- **Python 3.10-3.12** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads/))
- **4GB RAM** (minimum)
- **2GB free disk space**

### Common Issues:

**"ModuleNotFoundError"**

```bash
# Make sure virtual environment is activated
source uom_venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

**"Kernel died" in Jupyter**

```bash
# Reduce ensemble size in notebook 3:
n_models = 10  # Instead of 30
```

**"Dataset not found"**

```bash
# Extract the ZIP file in data/raw/
cd data/raw
unzip LC_loans_granting_model_dataset.csv.zip
```

**More help:** See the Troubleshooting section in [QUICKSTART.md](QUICKSTART.md)

---

## 💻 Usage

### Primary Method: Jupyter Notebooks (Recommended)

The fully functional system is available through Jupyter notebooks:

```bash
# Activate environment
source uom_venv/bin/activate

# Open Jupyter
jupyter notebook

# Run: notebooks/05_comprehensive_evaluation.ipynb
```

This notebook provides:
- ✅ Complete prediction pipeline
- ✅ Uncertainty quantification
- ✅ Escalation analysis
- ✅ Performance metrics
- ✅ Visual charts
- ✅ Detailed reports

### Alternative: Command Line (For Preprocessed Data)

```bash
# Predict on pre-processed data
python simple_predict.py --input data/splits/X_test.csv --output predictions.csv --limit 1000
```

### What You'll Learn:

- ✅ Building production ML pipelines
- ✅ Uncertainty quantification with bootstrap ensembles
- ✅ Cost-benefit optimization for business decisions
- ✅ Model interpretability with SHAP
- ✅ Handling class imbalance (SMOTE)
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ Model calibration (Platt scaling)

### Technologies Used:

- **Python 3.12** - Core programming language
- **pandas & numpy** - Data manipulation
- **scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting models
- **SHAP** - Model explainability
- **matplotlib & seaborn** - Visualizations
- **Jupyter** - Interactive development

---

## 🛠️ Development Status

### ✅ Completed & Working:
- [x] ML models trained and optimized
- [x] Bootstrap ensemble implementation
- [x] Uncertainty quantification system
- [x] Intelligent escalation logic  
- [x] Performance evaluation
- [x] Visualization and reporting
- [x] Complete documentation
- [x] Setup scripts for any computer

### 🔄 In Development:
- [ ] Standalone preprocessing pipeline
- [ ] Command-line tool for raw data
- [ ] Web application interface

**Note**: Core functionality is complete and working through notebooks!

---

## 🤝 Contributing

This project was developed by the **Lazy Loaders Team** for credit risk assessment research.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For questions or contributions, please:

1. Review the [QUICKSTART.md](QUICKSTART.md) guide
2. Check [PROGRESS.md](PROGRESS.md) for current status
3. Open an issue on GitHub

---

## 📄 License

This project is for educational purposes. See LICENSE file for details.

---

## 🆘 Support

- **Documentation**: Check the `docs/` folder and markdown files
- **Issues**: Open an issue on GitHub
- **Quick Help**: Read [CURRENT_STATUS.md](CURRENT_STATUS.md)

---

## 🎯 Use Cases

### Financial Institutions
- Automate routine loan approvals
- Route complex cases to experienced officers
- Reduce processing time and costs

### Risk Management
- Identify high-uncertainty decisions
- Maintain human oversight on edge cases
- Improve decision quality

### Research & Education
- Study uncertainty quantification methods
- Learn production ML system design
- Explore cost-benefit optimization

---

## 🏆 Key Results

From actual training on Lending Club dataset:

- **210,000+ loans** analyzed
- **78% automation rate** achieved
- **89% accuracy** on automated decisions
- **$678 savings** demonstrated per batch
- **21% cost reduction** in total review process

---

## 📞 Contact

- **Repository**: https://github.com/Lazy-Loaders-Group/Credit_Risk_Escalation
- **Team**: Lazy Loaders
- **Email**: [Contact maintainers]

---

## 🙏 Acknowledgments

- **Dataset**: Lending Club Loan Data
- **Libraries**: scikit-learn, XGBoost, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: Jupyter

---

## 🚀 Get Started Now

```bash
# Quick start (3 commands)
git clone https://github.com/Lazy-Loaders-Group/Credit_Risk_Escalation.git
cd Credit_Risk_Escalation
./setup.sh && source uom_venv/bin/activate && jupyter notebook
```

Then open: `notebooks/05_comprehensive_evaluation.ipynb`

---

**Last Updated**: November 22, 2024  
**Version**: 1.0.0  
**Status**: ✅ Production Ready (via Jupyter notebooks)

**Developed by:** Lazy Loaders Team  
**Last Updated:** November 5, 2025  
**Status:** ✅ All 6 phases complete - Production ready!

---

**⭐ Star this repo if you find it useful!**

**📖 Read [CURRENT_STATUS.md](CURRENT_STATUS.md) to see what's working now.**
