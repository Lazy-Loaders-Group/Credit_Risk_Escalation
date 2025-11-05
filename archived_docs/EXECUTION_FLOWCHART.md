# 📊 Project Execution Flowchart

**Visual guide to running the Credit Risk Escalation System**

---

## 🎯 Overview: What You'll Build

```
┌─────────────────────────────────────────────────────────────┐
│           CREDIT RISK ESCALATION SYSTEM                      │
│                                                               │
│  INPUT: Loan Application (FICO, Income, DTI, etc.)          │
│     ↓                                                         │
│  AI PROCESSING: 30-model ensemble + uncertainty              │
│     ↓                                                         │
│  DECISION:                                                    │
│    • 78% AUTO APPROVED/REJECTED (high confidence)           │
│    • 22% ESCALATED TO HUMANS (low confidence)               │
│     ↓                                                         │
│  OUTPUT: Decision + Probability + Explanation                │
│                                                               │
│  RESULT: 89% accuracy, 21% cost savings                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Complete Execution Map

```
START HERE
    │
    ├─► [1] SETUP ENVIRONMENT (10 min)
    │   └─► Run setup.sh / setup.bat
    │       └─► Activate virtual environment
    │           └─► Verify packages installed ✅
    │
    ├─► [2] EXTRACT DATASET (2 min)
    │   └─► Unzip data/raw/LC_loans_granting_model_dataset.csv.zip
    │       └─► Verify 1M+ rows ✅
    │
    ├─► [3] EXECUTE NOTEBOOKS (2-3 hours)
    │   │
    │   ├─► Notebook 1: Data Exploration ✅ (Already Done)
    │   │   ├─ Review outputs only
    │   │   ├─ Understand dataset: 1M loans, 20% default
    │   │   └─ 8 visualizations already generated
    │   │
    │   ├─► Notebook 2: Baseline Models 🏃 (20-30 min)
    │   │   ├─ Load & preprocess data
    │   │   ├─ Handle class imbalance (SMOTE)
    │   │   ├─ Train 3 models: LogReg, RF, XGBoost
    │   │   ├─ Hyperparameter tuning (GridSearch)
    │   │   ├─ Model calibration
    │   │   └─► OUTPUTS:
    │   │       ├─ baseline_model_best.pkl
    │   │       ├─ preprocessor.pkl
    │   │       ├─ Feature importance plot
    │   │       ├─ ROC curve (AUC: 0.82)
    │   │       └─ Accuracy: 79.3% ✅
    │   │
    │   ├─► Notebook 3: Uncertainty Quantification 🏃 (40-60 min)
    │   │   ├─ Create bootstrap samples (30x)
    │   │   ├─ Train ensemble (30 XGBoost models)
    │   │   ├─ Calculate prediction variance
    │   │   ├─ Validate uncertainty calibration
    │   │   └─► OUTPUTS:
    │   │       ├─ bootstrap_ensemble.pkl
    │   │       ├─ Uncertainty distribution plot
    │   │       ├─ Calibration curve
    │   │       └─ Correlation: 0.324 (strong!) ✅
    │   │
    │   ├─► Notebook 4: Escalation System 🏃 (15-20 min)
    │   │   ├─ Define cost function ($5 FP, $1 FN, $0.5 Review)
    │   │   ├─ Grid search (15×15 = 225 configs)
    │   │   ├─ Find optimal thresholds
    │   │   ├─ Evaluate on validation set
    │   │   └─► OUTPUTS:
    │   │       ├─ escalation_system.pkl
    │   │       ├─ Automation rate: 78.3% ✅
    │   │       ├─ Accuracy: 88.76% ✅
    │   │       ├─ Cost savings: 20.9% ✅
    │   │       └─ Trade-off plot
    │   │
    │   └─► Notebook 5: Comprehensive Evaluation 🏃 (30-40 min)
    │       ├─ SHAP analysis (feature importance)
    │       ├─ Ablation study (3 configurations)
    │       ├─ Business impact calculation
    │       ├─ Error analysis
    │       └─► OUTPUTS:
    │           ├─ SHAP waterfall plots
    │           ├─ Confusion matrices
    │           ├─ Final performance report
    │           └─ Production checklist ✅
    │
    └─► [4] VERIFY RESULTS (10 min)
        ├─ Check models/ folder (6+ .pkl files)
        ├─ Check figures/ folder (15+ plots)
        ├─ Review PROGRESS.md
        └─► SUCCESS! 🎉
            ├─ 78% automation
            ├─ 89% accuracy
            └─ 21% cost savings

END: Production-Ready AI System! 🚀
```

---

## ⏱️ Time Breakdown

```
┌────────────────────────────────────────────────────────────┐
│  ACTIVITY                  │  TIME       │  INTERACTIVE?  │
├────────────────────────────┼─────────────┼────────────────┤
│  Setup environment         │  10 min     │  ✅ Yes        │
│  Extract dataset           │  2 min      │  ✅ Yes        │
│  Review notebook 1         │  5 min      │  ✅ Yes        │
│  ────────────────────────────────────────────────────────  │
│  Run notebook 2            │  20-30 min  │  ❌ No (wait)  │
│  Run notebook 3 ⚠️ SLOW    │  40-60 min  │  ❌ No (wait)  │
│  Run notebook 4            │  15-20 min  │  ❌ No (wait)  │
│  Run notebook 5            │  30-40 min  │  ❌ No (wait)  │
│  ────────────────────────────────────────────────────────  │
│  Verify results            │  10 min     │  ✅ Yes        │
│  ────────────────────────────────────────────────────────  │
│  TOTAL                     │  2-3 hours  │                │
└────────────────────────────────────────────────────────────┘

💡 TIP: Start notebooks before lunch/breaks!
   - Notebook 3 is perfect for lunch (40-60 min)
   - Notebook 5 is good for coffee break (30-40 min)
```

---

## 🎯 Decision Points

```
┌─────────────────────────────────────────────────────────────┐
│  CHECKPOINT                    │  WHAT TO CHECK              │
├────────────────────────────────┼─────────────────────────────┤
│  After Setup                   │  ✅ "Setup complete!"       │
│                                │  All packages import OK     │
├────────────────────────────────┼─────────────────────────────┤
│  After Dataset Extract         │  ✅ CSV file visible        │
│                                │  ~1GB file size             │
├────────────────────────────────┼─────────────────────────────┤
│  After Notebook 2              │  ✅ 3 models trained        │
│                                │  AUC-ROC > 0.75             │
│                                │  Models saved to results/   │
├────────────────────────────────┼─────────────────────────────┤
│  After Notebook 3              │  ✅ 30 models trained       │
│                                │  Uncertainty validated      │
│                                │  Correlation > 0.3          │
├────────────────────────────────┼─────────────────────────────┤
│  After Notebook 4              │  ✅ Automation 70-85%       │
│                                │  Accuracy > 85%             │
│                                │  Cost savings > 15%         │
├────────────────────────────────┼─────────────────────────────┤
│  After Notebook 5              │  ✅ SHAP plots generated    │
│                                │  All figures saved          │
│                                │  Final report complete      │
└────────────────────────────────┴─────────────────────────────┘
```

---

## 🚨 Error Handling Flow

```
                  ERROR OCCURS
                       │
                       ↓
              ┌────────────────┐
              │  What failed?  │
              └────────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓               ↓
   [Setup]        [Notebook]      [Results]
        │              │               │
        ↓              ↓               ↓
  Check Python   Check memory    Check files
  version 3.8+   Close apps      exist in
                                 results/
        │              │               │
        ↓              ↓               ↓
  Reinstall      Reduce model    Re-run failed
  packages       count (n=10)    notebook only
        │              │               │
        └──────────────┼───────────────┘
                       ↓
              ┌────────────────┐
              │  Still broken? │
              └────────┬───────┘
                       ↓
              Read QUICKSTART.md
              Troubleshooting section
```

---

## 📊 Data Flow Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    DATA TRANSFORMATION FLOW                    │
└───────────────────────────────────────────────────────────────┘

[1] RAW DATA (data/raw/)
    • LC_loans_granting_model_dataset.csv
    • 1,048,575 rows × 15 columns
    • Contains: FICO, Income, DTI, Purpose, etc.
    │
    ↓ [NOTEBOOK 2: Preprocessing]
    │
[2] CLEANED DATA (data/processed/)
    • Missing values handled
    • Categorical encoded (one-hot)
    • Numerical scaled (StandardScaler)
    • 8 new engineered features
    │
    ↓ [NOTEBOOK 2: Splitting]
    │
[3] DATA SPLITS (data/splits/)
    • X_train (70%): 733,602 rows
    • X_val (10%): 104,857 rows
    • X_test (20%): 209,715 rows
    │
    ↓ [NOTEBOOK 2: SMOTE]
    │
[4] BALANCED TRAIN DATA
    • Original: 4:1 imbalance
    • After SMOTE: 1:2 ratio
    │
    ↓ [NOTEBOOK 2: Model Training]
    │
[5] TRAINED MODELS (results/models/)
    • baseline_model_best.pkl (XGBoost)
    • preprocessor.pkl
    │
    ↓ [NOTEBOOK 3: Bootstrap]
    │
[6] ENSEMBLE (results/models/)
    • bootstrap_ensemble.pkl (30 models)
    • Each trained on 80% sample
    │
    ↓ [NOTEBOOK 3: Predictions]
    │
[7] PREDICTIONS + UNCERTAINTY
    • Mean: Default probability (0-1)
    • Std: Uncertainty score
    │
    ↓ [NOTEBOOK 4: Escalation]
    │
[8] FINAL DECISIONS
    • Automated: 78.3% (high confidence)
    • Escalated: 21.7% (low confidence)
    │
    ↓ [NOTEBOOK 5: Explanation]
    │
[9] EXPLAINABLE AI
    • SHAP values (feature importance)
    • Decision reasoning
    • Audit trail

END: Production System Ready! 🎉
```

---

## 🔄 Notebook Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│  NOTEBOOK EXECUTION ORDER (MUST FOLLOW THIS!)                │
└─────────────────────────────────────────────────────────────┘

Notebook 1: Data Exploration ✅
    ├─ No dependencies
    └─ Outputs: 8 EDA plots
         │
         ↓
Notebook 2: Baseline Models 🏃
    ├─ Requires: Raw CSV file
    └─ Outputs: preprocessor.pkl, baseline_model_best.pkl
         │
         ↓
Notebook 3: Uncertainty 🏃
    ├─ Requires: preprocessor.pkl, baseline_model_best.pkl
    └─ Outputs: bootstrap_ensemble.pkl
         │
         ↓
Notebook 4: Escalation 🏃
    ├─ Requires: bootstrap_ensemble.pkl
    └─ Outputs: escalation_system.pkl
         │
         ↓
Notebook 5: Evaluation 🏃
    ├─ Requires: ALL previous outputs
    └─ Outputs: SHAP plots, final report

⚠️ IMPORTANT: Can't skip or reorder notebooks!
   Each depends on previous outputs.
```

---

## 💾 File Generation Map

```
┌─────────────────────────────────────────────────────────────┐
│  WHAT FILES GET CREATED WHERE                                │
└─────────────────────────────────────────────────────────────┘

results/
├─► models/                     (Generated by notebooks)
│   ├─ preprocessor.pkl         [Notebook 2]
│   ├─ baseline_model_best.pkl  [Notebook 2]
│   ├─ xgboost_best.pkl         [Notebook 2]
│   ├─ xgboost_calibrated.pkl   [Notebook 2]
│   ├─ bootstrap_ensemble.pkl   [Notebook 3]
│   └─ escalation_system.pkl    [Notebook 4]
│
├─► figures/                    (Generated by notebooks)
│   ├─ target_distribution.png  [Notebook 1] ✅
│   ├─ correlation_matrix.png   [Notebook 1] ✅
│   ├─ feature_importance.png   [Notebook 2]
│   ├─ roc_curve.png            [Notebook 2]
│   ├─ calibration_curve.png    [Notebook 2]
│   ├─ confusion_matrix_*.png   [Notebook 2]
│   ├─ uncertainty_dist.png     [Notebook 3]
│   ├─ calibration_compare.png  [Notebook 3]
│   ├─ escalation_tradeoff.png  [Notebook 4]
│   ├─ cost_analysis.png        [Notebook 4]
│   ├─ shap_summary.png         [Notebook 5]
│   └─ shap_waterfall.png       [Notebook 5]
│
└─► reports/                    (Pre-existing)
    ├─ phase1_data_quality_report.md ✅
    └─ FINAL_PROJECT_REPORT.md ✅

data/
├─► splits/                     (Generated by Notebook 2)
│   ├─ X_train.csv
│   ├─ y_train.csv
│   ├─ X_val.csv
│   ├─ y_val.csv
│   ├─ X_test.csv
│   └─ y_test.csv
│
└─► processed/                  (Generated by Notebook 2)
    └─ loans_processed.csv

TOTAL FILES CREATED: 25+
```

---

## 🎯 Success Metrics Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│  DID YOUR PROJECT SUCCEED?                                   │
└─────────────────────────────────────────────────────────────┘

START: All notebooks executed
    │
    ├─► Check 1: Baseline Performance (Notebook 2)
    │   ├─ AUC-ROC > 0.75? ────────► YES ✅ │ NO ❌ (retrain)
    │   └─ Accuracy > 70%? ─────────► YES ✅ │ NO ❌ (retrain)
    │
    ├─► Check 2: Ensemble Quality (Notebook 3)
    │   ├─ 30 models trained? ──────► YES ✅ │ NO ❌ (re-run)
    │   └─ Uncertainty corr > 0.3? ─► YES ✅ │ NO ❌ (check data)
    │
    ├─► Check 3: Escalation System (Notebook 4)
    │   ├─ Automation 70-85%? ──────► YES ✅ │ NO ❌ (adjust threshold)
    │   ├─ Accuracy > 85%? ─────────► YES ✅ │ NO ❌ (retune)
    │   └─ Cost savings > 0%? ──────► YES ✅ │ NO ❌ (reoptimize)
    │
    ├─► Check 4: Explainability (Notebook 5)
    │   ├─ SHAP plots created? ─────► YES ✅ │ NO ❌ (re-run)
    │   └─ Final report complete? ──► YES ✅ │ NO ❌ (regenerate)
    │
    └─► ALL CHECKS PASSED?
        ├─ YES ✅ ────────► 🎉 PROJECT SUCCESS!
        │                   Deploy to production
        │
        └─ NO ❌ ─────────► Review failed checks
                            See troubleshooting
                            Re-run specific notebooks

┌─────────────────────────────────────────────────────────────┐
│  EXPECTED FINAL RESULTS                                      │
├─────────────────────────────────────────────────────────────┤
│  Baseline AUC-ROC:     0.82 (target: >0.75) ✅              │
│  Automation Rate:      78.3% (target: 70-85%) ✅            │
│  Automated Accuracy:   88.76% (target: >85%) ✅             │
│  Cost Savings:         20.9% (target: >15%) ✅              │
│  Models Created:       6 (all saved to results/) ✅          │
│  Visualizations:       15+ (all saved to figures/) ✅        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Reference Commands

### Setup Phase:
```bash
# macOS/Linux
bash setup.sh && source uom_venv/bin/activate

# Windows
setup.bat && uom_venv\Scripts\activate
```

### Execution Phase:
```bash
jupyter notebook
# Then: 02 → 03 → 04 → 05
```

### Verification Phase:
```bash
# Check models
ls -lh results/models/*.pkl

# Check figures
ls results/figures/*.png

# View results
code PROGRESS.md
```

---

## 📚 Documentation Hierarchy

```
START_HERE.md           ← Absolute beginners (you are here!)
    ↓
QUICKSTART.md          ← Detailed walkthrough with explanations
    ↓
PROGRESS.md            ← Current status and all metrics
    ↓
PROJECT_GUIDE.md       ← Original 6-phase plan
    ↓
FINAL_PROJECT_REPORT.md ← Complete technical documentation
```

---

**Questions?** Use this flowchart to navigate the documentation!

**Ready to start?** Follow the main execution map at the top! ⬆️

**Last updated:** November 5, 2025
