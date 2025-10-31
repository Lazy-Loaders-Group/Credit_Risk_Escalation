# Phase 2 Quick Start Guide

**Phase 2: Baseline Model Development**  
**Start Date:** November 1, 2025  
**Duration:** 4-6 days  
**Status:** Ready to begin

---

## 🎯 Phase 2 Objectives

Build and evaluate baseline machine learning models for credit risk prediction:
1. Preprocess the full dataset
2. Train multiple baseline models
3. Optimize hyperparameters
4. Evaluate and compare models
5. Select best model for Phase 3 (uncertainty quantification)

---

## 📋 Day-by-Day Plan

### Day 1: Data Preprocessing & Splitting
**Time:** 2-3 hours

```python
# Create notebook: 02_baseline_model.ipynb

# 1. Load data
import pandas as pd
from src.data_preprocessing import CreditDataPreprocessor

df = pd.read_csv('data/raw/LC_loans_granting_model_dataset.csv')

# 2. Preprocess
preprocessor = CreditDataPreprocessor()
df_processed = preprocessor.fit_transform(df, target_col='Default')

# 3. Split data (stratified)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
    df_processed, target_col='Default'
)

# 4. Handle class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 5. Save splits
from src.data_preprocessing import save_splits
save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
```

**Deliverable:** Preprocessed data saved to `data/splits/`

---

### Day 2: Train Baseline Models
**Time:** 3-4 hours

```python
# Train 3 baseline models

# Model 1: Logistic Regression (simple baseline)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)

# Model 2: Random Forest (robust)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

# Model 3: XGBoost (state-of-the-art)
import xgboost as xgb
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

**Deliverable:** 3 trained models

---

### Day 3: Initial Evaluation
**Time:** 2-3 hours

```python
# Evaluate on validation set
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

results = []
for name, model in models.items():
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Metrics
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1-Score': f1_score(y_val, y_pred),
        'AUC-ROC': roc_auc_score(y_val, y_pred_proba)
    })
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n{name} Confusion Matrix:\n{cm}")

# Compare models
results_df = pd.DataFrame(results)
print(results_df)
```

**Deliverable:** Model comparison table

---

### Day 4: Hyperparameter Tuning
**Time:** 3-4 hours

```python
# Tune best performing model (likely XGBoost or Random Forest)
from sklearn.model_selection import GridSearchCV

# Example for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")
```

**Deliverable:** Optimized model

---

### Day 5: Feature Importance & Calibration
**Time:** 2-3 hours

```python
# Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('results/figures/feature_importance.png')
plt.show()

# Calibration Curve
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_val, best_model.predict_proba(X_val)[:, 1], n_bins=10
)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curve')
plt.legend()
plt.savefig('results/figures/calibration_curve.png')
plt.show()

# Temperature Scaling (if needed)
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
calibrated_model.fit(X_train, y_train)
```

**Deliverable:** Feature importance analysis + calibrated model

---

### Day 6: Final Evaluation & Documentation
**Time:** 2-3 hours

```python
# Final evaluation on test set (ONLY ONCE!)
y_test_pred = calibrated_model.predict(X_test)
y_test_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

# All metrics
print("Final Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_test_pred_proba):.4f}")

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {roc_auc_score(y_test, y_test_pred_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Model')
plt.legend()
plt.savefig('results/figures/roc_curve.png')
plt.show()

# Save best model
import joblib
joblib.dump(calibrated_model, 'results/models/baseline_model_best.pkl')
joblib.dump(preprocessor, 'results/models/preprocessor.pkl')
```

**Deliverable:** Final model saved + performance report

---

## 📊 Expected Results

### Baseline Performance Targets

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| **Accuracy** | >70% | >80% | >85% |
| **Precision** | >50% | >60% | >70% |
| **Recall** | >45% | >55% | >65% |
| **F1-Score** | >45% | >55% | >65% |
| **AUC-ROC** | >0.75 | >0.80 | >0.85 |

### Success Criteria
- ✅ At least one model achieves AUC-ROC >0.75
- ✅ Probabilities are reasonably calibrated (ECE <0.1)
- ✅ Clear feature importance insights
- ✅ Model saved and ready for Phase 3

---

## 🔧 Key Python Functions to Use

### Preprocessing
```python
from src.data_preprocessing import CreditDataPreprocessor
preprocessor = CreditDataPreprocessor()
df_processed = preprocessor.fit_transform(df)
```

### Class Imbalance
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

### Evaluation
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
```

### Model Saving
```python
import joblib
joblib.dump(model, 'path/to/model.pkl')
model = joblib.load('path/to/model.pkl')
```

---

## 📁 Files to Create

### Notebooks
- [ ] `notebooks/02_baseline_model.ipynb` - Main modeling notebook

### Python Modules (if needed)
- [ ] `src/models.py` - Model training utilities
- [ ] `src/evaluation.py` - Evaluation metrics and plots

### Results
- [ ] `results/figures/feature_importance.png`
- [ ] `results/figures/confusion_matrix_*.png`
- [ ] `results/figures/roc_curve.png`
- [ ] `results/figures/calibration_curve.png`
- [ ] `results/models/baseline_model_best.pkl`
- [ ] `results/models/preprocessor.pkl`
- [ ] `results/reports/phase2_baseline_model_report.md`

---

## ⚠️ Common Pitfalls to Avoid

1. **Don't touch test set until final evaluation**
   - Use validation set for all tuning
   - Test set = final evaluation only

2. **Handle class imbalance**
   - Use SMOTE or class_weight='balanced'
   - Don't rely on accuracy alone

3. **Check probability calibration**
   - Use calibration curves
   - Apply temperature scaling if needed

4. **Avoid data leakage**
   - Fit preprocessing on train set only
   - Transform val/test sets separately

5. **Use stratified splits**
   - Maintain class balance across splits
   - Use `stratify=y` parameter

---

## 📚 Quick Reference

### Model Parameters

**Logistic Regression:**
```python
LogisticRegression(max_iter=1000, class_weight='balanced')
```

**Random Forest:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

**XGBoost:**
```python
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
```

### Evaluation Template
```python
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall: {recall_score(y, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y, y_pred_proba):.4f}")
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc_roc': roc_auc_score(y, y_pred_proba)
    }
```

---

## ✅ Phase 2 Checklist

### Week 2 Tasks
- [ ] Day 1: Preprocess data and create splits
- [ ] Day 2: Train 3 baseline models
- [ ] Day 3: Evaluate on validation set
- [ ] Day 4: Hyperparameter tuning
- [ ] Day 5: Feature importance + calibration
- [ ] Day 6: Final test set evaluation

### Deliverables
- [ ] Preprocessed data saved
- [ ] 3 baseline models trained
- [ ] Best model optimized and saved
- [ ] Feature importance analysis
- [ ] Calibration curve
- [ ] ROC curve
- [ ] Confusion matrix
- [ ] Phase 2 report written
- [ ] PROGRESS.md updated

---

**Ready to start Phase 2!** 🚀

**Next Action:** Create `notebooks/02_baseline_model.ipynb` and begin Day 1 tasks.

---

**Prepared By:** Lazy Loaders Team  
**Date:** October 31, 2025  
**For:** Phase 2 - Baseline Model Development
