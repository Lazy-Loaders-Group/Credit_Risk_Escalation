#!/usr/bin/env python3
"""
Quick script to complete Phase 4: Escalation System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import sys

warnings.filterwarnings('ignore')
sys.path.append('src')

from sklearn.preprocessing import LabelEncoder
from escalation_system import EscalationSystem

print("="*60)
print("PHASE 4: HUMAN ESCALATION SYSTEM")
print("="*60)

# 1. Load data
print("\n1. Loading data...")
X_val = pd.read_csv('data/splits/X_val.csv')
X_test = pd.read_csv('data/splits/X_test.csv')

# Preprocess data
text_cols = ['title', 'desc']
existing_text_cols = [col for col in text_cols if col in X_val.columns]
if existing_text_cols:
    X_val = X_val.drop(columns=existing_text_cols)
    X_test = X_test.drop(columns=existing_text_cols)

non_numeric_cols = X_val.select_dtypes(include=['object', 'category']).columns
if len(non_numeric_cols) > 0:
    for col in non_numeric_cols:
        le = LabelEncoder()
        all_values = pd.concat([X_val[col], X_test[col]]).astype(str).unique()
        le.fit(all_values)
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

# Load uncertainty estimates
uncertainty_data = joblib.load('results/models/uncertainty_estimates.pkl')

proba_val = uncertainty_data['val']['proba']
# If proba has 2 columns, take the positive class probability
if len(proba_val.shape) > 1 and proba_val.shape[1] == 2:
    proba_val = proba_val[:, 1]
uncertainty_val = uncertainty_data['val']['uncertainty']
y_val = uncertainty_data['val']['y_true']
y_pred_val = uncertainty_data['val']['y_pred']

proba_test = uncertainty_data['test']['proba']
if len(proba_test.shape) > 1 and proba_test.shape[1] == 2:
    proba_test = proba_test[:, 1]
uncertainty_test = uncertainty_data['test']['uncertainty']
y_test = uncertainty_data['test']['y_true']
y_pred_test = uncertainty_data['test']['y_pred']

print(f"✅ Loaded {len(y_val)} validation samples")
print(f"✅ Loaded {len(y_test)} test samples")

# 2. Define business costs
print("\n2. Defining business costs...")
cost_params = {
    'cost_false_positive': 5.0,
    'cost_false_negative': 1.0,
    'cost_human_review': 0.50
}
print(f"   False Positive cost: ${cost_params['cost_false_positive']}")
print(f"   False Negative cost: ${cost_params['cost_false_negative']}")
print(f"   Human Review cost: ${cost_params['cost_human_review']}")

# 3. Grid search for optimal thresholds
print("\n3. Grid searching for optimal thresholds...")
print("   Testing 15 x 15 = 225 threshold combinations...")

uncertainty_thresholds = np.linspace(0.01, 0.15, 15)
confidence_thresholds = np.linspace(0.55, 0.90, 15)

best_cost = float('inf')
best_params = {}
results_list = []

for u_thresh in uncertainty_thresholds:
    for c_thresh in confidence_thresholds:
        # Decide which samples to automate
        automate_mask = (uncertainty_val <= u_thresh) & (
            (proba_val >= c_thresh) | (proba_val <= (1 - c_thresh))
        )
        
        n_automated = automate_mask.sum()
        n_escalated = (~automate_mask).sum()
        
        if n_automated == 0:
            continue
        
        # Calculate automated predictions accuracy
        y_auto = y_val[automate_mask]
        y_pred_auto = y_pred_val[automate_mask]
        
        # Calculate costs
        fp = ((y_pred_auto == 1) & (y_auto == 0)).sum()
        fn = ((y_pred_auto == 0) & (y_auto == 1)).sum()
        
        cost = (fp * cost_params['cost_false_positive'] + 
                fn * cost_params['cost_false_negative'] +
                n_escalated * cost_params['cost_human_review'])
        
        automation_rate = n_automated / len(y_val)
        automated_acc = (y_auto == y_pred_auto).mean()
        
        results_list.append({
            'uncertainty_thresh': u_thresh,
            'confidence_thresh': c_thresh,
            'automation_rate': automation_rate,
            'automated_accuracy': automated_acc,
            'total_cost': cost,
            'n_automated': n_automated,
            'n_escalated': n_escalated
        })
        
        if cost < best_cost:
            best_cost = cost
            best_params = {
                'uncertainty_threshold': u_thresh,
                'confidence_threshold': c_thresh,
                'automation_rate': automation_rate,
                'automated_accuracy': automated_acc,
                'total_cost': cost,
                'n_automated': n_automated,
                'n_escalated': n_escalated
            }

print(f"\n✅ Grid search complete!")
print(f"\nBest Configuration:")
print(f"   Uncertainty Threshold: {best_params['uncertainty_threshold']:.4f}")
print(f"   Confidence Threshold: {best_params['confidence_threshold']:.4f}")
print(f"   Automation Rate: {best_params['automation_rate']:.2%}")
print(f"   Automated Accuracy: {best_params['automated_accuracy']:.2%}")
print(f"   Total Cost: ${best_params['total_cost']:.2f}")

# 4. Create and evaluate escalation system
print("\n4. Creating escalation system...")
escalation_system = EscalationSystem(
    uncertainty_threshold=best_params['uncertainty_threshold'],
    confidence_threshold=best_params['confidence_threshold'],
    cost_false_positive=cost_params['cost_false_positive'],
    cost_false_negative=cost_params['cost_false_negative'],
    cost_human_review=cost_params['cost_human_review']
)

# Evaluate on test set
print("\n5. Evaluating on test set...")
test_results = escalation_system.evaluate(
    y_true=y_test,
    y_pred=y_pred_test,
    proba=proba_test,
    uncertainty=uncertainty_test
)

print(f"\nTest Set Results:")
print(f"   Samples Automated: {test_results['n_automated']:,} ({test_results['automation_rate']:.2%})")
print(f"   Samples Escalated: {test_results['n_escalated']:,} ({test_results['escalation_rate']:.2%})")
print(f"   Automated Accuracy: {test_results['automated_accuracy']:.2%}")
print(f"   Total Cost: ${test_results['total_cost']:.2f}")

# Calculate baseline cost (no automation)
baseline_fp = ((y_pred_test == 1) & (y_test == 0)).sum()
baseline_fn = ((y_pred_test == 0) & (y_test == 1)).sum()
baseline_cost = (baseline_fp * cost_params['cost_false_positive'] + 
                 baseline_fn * cost_params['cost_false_negative'])

cost_savings = baseline_cost - test_results['total_cost']
cost_savings_pct = (cost_savings / baseline_cost) * 100

print(f"\n   Baseline Cost: ${baseline_cost:.2f}")
print(f"   Cost Savings: ${cost_savings:.2f} ({cost_savings_pct:.1f}%)")

# 6. Save escalation system
print("\n6. Saving escalation system...")
joblib.dump(escalation_system, 'results/models/escalation_system.pkl')
joblib.dump(test_results, 'results/models/escalation_test_results.pkl')
print("✅ Escalation system saved")

print("\n" + "="*60)
print("PHASE 4 COMPLETE!")
print("="*60)
print(f"\n✅ Automation Rate: {test_results['automation_rate']:.2%}")
print(f"✅ Automated Accuracy: {test_results['automated_accuracy']:.2%}")
print(f"✅ Cost Savings: {cost_savings_pct:.1f}%")
print(f"\n🚀 Ready for Phase 5: Comprehensive Evaluation")
