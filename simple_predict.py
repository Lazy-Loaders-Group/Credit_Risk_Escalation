"""
Simple Prediction Script for Pre-trained Credit Risk Model

This script works with the existing pre-processed data format.
Use this for quick predictions on data in the same format as the training data.

Usage:
    python simple_predict.py --input data/splits/X_test.csv --output predictions.csv
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def load_models():
    """Load trained models"""
    print("Loading models...")
    
    with open('results/models/preprocessor.pkl', 'rb') as f:
        preproc_info = pickle.load(f)
        feature_names = preproc_info.get('feature_names', [])
    
    with open('results/models/bootstrap_ensemble.pkl', 'rb') as f:
        ensemble = pickle.load(f)
    
    with open('results/models/escalation_system.pkl', 'rb') as f:
        escalation_system = pickle.load(f)
    
    print("✅ Models loaded\n")
    return ensemble, escalation_system, feature_names

def predict(X, ensemble, escalation_system):
    """Make predictions with escalation logic"""
    print(f"Processing {len(X)} samples...")
    
    # Get predictions with uncertainty
    proba_mean, uncertainty, _ = ensemble.predict_with_uncertainty(X)
    
    # Get escalation decisions
    escalate_mask, details = escalation_system.process_predictions(
        proba_mean, uncertainty, return_details=True
    )
    
    # Create results
    results = pd.DataFrame({
        'sample_id': range(len(X)),
        'prob_default': proba_mean[:, 1],
        'prob_paid': proba_mean[:, 0],
        'uncertainty': uncertainty,
        'confidence': np.max(proba_mean, axis=1),
        'should_escalate': escalate_mask,
        'automated_decision': ['REJECT' if p >= 0.5 else 'APPROVE' 
                               for p in proba_mean[:, 1]],
        'final_action': ['ESCALATE_TO_AGENT' if esc else f"AUTO_{dec}" 
                         for esc, dec in zip(escalate_mask, 
                                            ['REJECT' if p >= 0.5 else 'APPROVE' 
                                             for p in proba_mean[:, 1]])]
    })
    
    # Add escalation reasons
    results = results.merge(details[['sample_idx', 'reason']], 
                           left_on='sample_id', right_on='sample_idx', how='left')
    results.drop('sample_idx', axis=1, inplace=True)
    results.rename(columns={'reason': 'escalation_reason'}, inplace=True)
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict credit risk with escalation')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV file')
    parser.add_argument('--limit', type=int, help='Limit number of predictions')
    
    args = parser.parse_args()
    
    # Load models
    ensemble, escalation_system, feature_names = load_models()
    
    # Load data
    print(f"Loading data from {args.input}...")
    X = pd.read_csv(args.input)
    
    # Select only the features used in training
    if feature_names:
        print(f"Selecting {len(feature_names)} training features...")
        X = X[feature_names]
    else:
        # Drop ID columns if present and no feature list available
        id_cols = [col for col in X.columns if col.lower() in ['id', 'index', 'unnamed: 0']]
        if id_cols:
            print(f"Dropping ID columns: {id_cols}")
            X = X.drop(columns=id_cols)
    
    if args.limit:
        X = X.head(args.limit)
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features\n")
    
    # Make predictions
    results = predict(X.values, ensemble, escalation_system)
    
    # Summary
    n_total = len(results)
    n_escalated = results['should_escalate'].sum()
    n_automated = n_total - n_escalated
    
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total samples:       {n_total}")
    print(f"Automated decisions: {n_automated} ({n_automated/n_total*100:.1f}%)")
    print(f"Escalated to agents: {n_escalated} ({n_escalated/n_total*100:.1f}%)")
    print("="*60)
    
    if n_automated > 0:
        n_approve = (results['automated_decision'] == 'APPROVE').sum() - n_escalated
        n_reject = (results['automated_decision'] == 'REJECT').sum()
        print(f"\nAutomated Approvals:  {n_approve}")
        print(f"Automated Rejections: {n_reject}")
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")
    
    # Show first few results
    print(f"\nFirst 10 predictions:")
    print(results.head(10)[['sample_id', 'final_action', 'confidence', 'uncertainty']].to_string(index=False))

if __name__ == '__main__':
    main()
