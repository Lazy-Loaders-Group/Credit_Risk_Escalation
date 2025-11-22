"""
Training Pipeline for Credit Risk Escalation System

This script runs the complete training pipeline:
1. Load and preprocess data
2. Train baseline model
3. Train bootstrap ensemble for uncertainty quantification
4. Train and optimize escalation system
5. Save all models and metadata

Usage:
    python train_and_save.py
    python train_and_save.py --data-path data/raw/LC_loans_granting_model_dataset.csv

Author: Credit Risk ML Team
Date: November 22, 2025
"""

import numpy as np
import pandas as pd
import pickle
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_preprocessing import CreditDataPreprocessor
from src.uncertainty_quantification import BootstrapEnsemble
from src.escalation_system import EscalationSystem


def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        'results/models',
        'results/reports',
        'results/figures',
        'data/splits',
        'data/processed'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories ready\n")


def load_and_preprocess_data(data_path):
    """
    Load and preprocess the dataset
    
    Parameters:
    -----------
    data_path : str
        Path to raw data CSV file
        
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test : arrays
        Preprocessed and split data
    preprocessor : CreditDataPreprocessor
        Fitted preprocessor object
    """
    print("="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60 + "\n")
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples\n")
    
    # Initialize preprocessor
    preprocessor = CreditDataPreprocessor()
    
    # Preprocess data
    df_processed = preprocessor.fit_transform(
        df, 
        target_col='Default',
        encoding='label',
        scaling=True,
        create_features=False  # Keep it simple for production
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df_processed,
        target_col='Default',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Save splits
    print("\nSaving data splits...")
    X_train.to_csv('data/splits/X_train.csv', index=False)
    X_val.to_csv('data/splits/X_val.csv', index=False)
    X_test.to_csv('data/splits/X_test.csv', index=False)
    y_train.to_csv('data/splits/y_train.csv', index=False)
    y_val.to_csv('data/splits/y_val.csv', index=False)
    y_test.to_csv('data/splits/y_test.csv', index=False)
    print("✓ Splits saved to data/splits/\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


def train_baseline_model(X_train, y_train, X_val, y_val):
    """
    Train baseline XGBoost model
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data
    X_val, y_val : arrays
        Validation data
        
    Returns:
    --------
    model : trained model
    metrics : dict
        Performance metrics
    """
    print("="*60)
    print("STEP 2: TRAINING BASELINE MODEL")
    print("="*60 + "\n")
    
    print("Training XGBoost classifier...")
    
    # Train XGBoost with optimized parameters
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print("✓ Model trained\n")
    
    # Evaluate
    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_proba[:, 1])
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}\n")
    
    return model, metrics


def train_uncertainty_ensemble(base_model, X_train, y_train, X_val, y_val):
    """
    Train bootstrap ensemble for uncertainty quantification
    
    Parameters:
    -----------
    base_model : sklearn estimator
        Base model to use for ensemble
    X_train, y_train : arrays
        Training data
    X_val, y_val : arrays
        Validation data
        
    Returns:
    --------
    ensemble : BootstrapEnsemble
        Trained ensemble
    metrics : dict
        Performance metrics
    """
    print("="*60)
    print("STEP 3: TRAINING UNCERTAINTY ENSEMBLE")
    print("="*60 + "\n")
    
    # Create ensemble
    ensemble = BootstrapEnsemble(
        base_model=base_model,
        n_estimators=30,
        bootstrap_size=0.8,
        random_state=42
    )
    
    # Train
    ensemble.fit(X_train.values, y_train.values)
    
    # Evaluate with uncertainty
    print("\nEvaluating ensemble with uncertainty...")
    proba_mean, uncertainty, all_proba = ensemble.predict_with_uncertainty(X_val.values)
    y_pred_ensemble = (proba_mean[:, 1] >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred_ensemble),
        'precision': precision_score(y_val, y_pred_ensemble),
        'recall': recall_score(y_val, y_pred_ensemble),
        'f1': f1_score(y_val, y_pred_ensemble),
        'roc_auc': roc_auc_score(y_val, proba_mean[:, 1]),
        'mean_uncertainty': float(np.mean(uncertainty)),
        'std_uncertainty': float(np.std(uncertainty))
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nUncertainty Statistics:")
    print(f"  Mean: {metrics['mean_uncertainty']:.4f}")
    print(f"  Std:  {metrics['std_uncertainty']:.4f}\n")
    
    return ensemble, metrics


def train_escalation_system(ensemble, X_val, y_val):
    """
    Train and optimize escalation system
    
    Parameters:
    -----------
    ensemble : BootstrapEnsemble
        Trained ensemble
    X_val, y_val : arrays
        Validation data
        
    Returns:
    --------
    escalation_system : EscalationSystem
        Optimized escalation system
    metrics : dict
        Performance metrics
    """
    print("="*60)
    print("STEP 4: TRAINING ESCALATION SYSTEM")
    print("="*60 + "\n")
    
    # Get predictions with uncertainty
    proba_mean, uncertainty, _ = ensemble.predict_with_uncertainty(X_val.values)
    y_pred = (proba_mean[:, 1] >= 0.5).astype(int)
    
    # Initialize escalation system
    escalation_system = EscalationSystem(
        uncertainty_threshold=0.1,
        confidence_threshold=0.7,
        cost_false_positive=1.0,
        cost_false_negative=5.0,
        cost_human_review=0.5
    )
    
    # Optimize thresholds
    print("Optimizing escalation thresholds...")
    best_unc, best_conf, optimization_results = escalation_system.optimize_thresholds(
        y_val.values,
        proba_mean,
        y_pred,
        uncertainty,
        uncertainty_range=(0.05, 0.25),
        confidence_range=(0.5, 0.9),
        n_steps=10
    )
    
    # Evaluate optimized system
    print("\nEvaluating escalation system...")
    metrics = escalation_system.evaluate_system(
        y_val.values,
        y_pred,
        proba_mean,
        uncertainty
    )
    
    print(f"\nEscalation Performance:")
    print(f"  Automation Rate:     {metrics['automation_rate']:.2%}")
    print(f"  Automated Accuracy:  {metrics['accuracy_automated']:.4f}")
    print(f"  Escalation Rate:     {1-metrics['automation_rate']:.2%}")
    print(f"  Total Cost:          ${metrics['total_cost']:.2f}")
    print(f"  Cost Savings:        ${metrics['cost_savings']:.2f}\n")
    
    # Save optimization results
    optimization_results.to_csv('results/reports/threshold_optimization_results.csv', index=False)
    print("✓ Optimization results saved\n")
    
    return escalation_system, metrics


def save_models_and_metadata(preprocessor, base_model, ensemble, escalation_system, 
                            baseline_metrics, ensemble_metrics, escalation_metrics):
    """
    Save all models and metadata
    
    Parameters:
    -----------
    preprocessor : CreditDataPreprocessor
        Fitted preprocessor
    base_model : model
        Trained baseline model
    ensemble : BootstrapEnsemble
        Trained ensemble
    escalation_system : EscalationSystem
        Optimized escalation system
    baseline_metrics, ensemble_metrics, escalation_metrics : dict
        Performance metrics
    """
    print("="*60)
    print("STEP 5: SAVING MODELS AND METADATA")
    print("="*60 + "\n")
    
    models_dir = Path('results/models')
    
    # Save preprocessor
    print("Saving preprocessor...")
    with open(models_dir / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("✓ preprocessor.pkl saved")
    
    # Save baseline model
    print("Saving baseline model...")
    with open(models_dir / 'baseline_model.pkl', 'wb') as f:
        pickle.dump(base_model, f)
    print("✓ baseline_model.pkl saved")
    
    # Save ensemble
    print("Saving bootstrap ensemble...")
    with open(models_dir / 'bootstrap_ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("✓ bootstrap_ensemble.pkl saved")
    
    # Save escalation system
    print("Saving escalation system...")
    with open(models_dir / 'escalation_system.pkl', 'wb') as f:
        pickle.dump(escalation_system, f)
    print("✓ escalation_system.pkl saved")
    
    # Create and save metadata
    print("\nSaving metadata...")
    metadata = {
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'python_version': sys.version.split()[0],
        'models': {
            'baseline': {
                'type': 'XGBoostClassifier',
                'metrics': baseline_metrics
            },
            'ensemble': {
                'type': 'BootstrapEnsemble',
                'n_estimators': 30,
                'metrics': ensemble_metrics
            },
            'escalation': {
                'type': 'EscalationSystem',
                'uncertainty_threshold': escalation_system.uncertainty_threshold,
                'confidence_threshold': escalation_system.confidence_threshold,
                'metrics': escalation_metrics
            }
        },
        'decision_thresholds': {
            'approve_threshold': 0.2,
            'reject_threshold': 0.8,
            'uncertainty_accept': 0.15
        }
    }
    
    with open(models_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ model_metadata.json saved")
    
    print(f"\n{'='*60}")
    print("✅ ALL MODELS AND METADATA SAVED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\nModels saved in: {models_dir}")
    print("\nFiles created:")
    print("  • preprocessor.pkl")
    print("  • baseline_model.pkl")
    print("  • bootstrap_ensemble.pkl")
    print("  • escalation_system.pkl")
    print("  • model_metadata.json")
    print("\nYou can now use these models for prediction!")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description='Train Credit Risk Escalation models'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/LC_loans_granting_model_dataset.csv',
        help='Path to raw data CSV file'
    )
    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='Skip training if models already exist'
    )
    
    args = parser.parse_args()
    
    # Check if models already exist
    if args.skip_if_exists:
        models_dir = Path('results/models')
        if (models_dir / 'bootstrap_ensemble.pkl').exists():
            print("\n✓ Models already exist. Skipping training.")
            print("  Use 'python train_and_save.py' to retrain.\n")
            return
    
    print("\n" + "="*60)
    print("CREDIT RISK ESCALATION - TRAINING PIPELINE")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    
    # Ensure directories exist
    ensure_directories()
    
    # Step 1: Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_and_preprocess_data(args.data_path)
    
    # Step 2: Train baseline model
    base_model, baseline_metrics = train_baseline_model(X_train, y_train, X_val, y_val)
    
    # Step 3: Train uncertainty ensemble
    ensemble, ensemble_metrics = train_uncertainty_ensemble(base_model, X_train, y_train, X_val, y_val)
    
    # Step 4: Train escalation system
    escalation_system, escalation_metrics = train_escalation_system(ensemble, X_val, y_val)
    
    # Step 5: Save everything
    save_models_and_metadata(
        preprocessor, base_model, ensemble, escalation_system,
        baseline_metrics, ensemble_metrics, escalation_metrics
    )
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED IN {duration:.1f} SECONDS")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("  1. Run predictions: python predict_and_decide.py --input example_new_loans.csv")
    print("  2. Start web UI:    streamlit run app.py")
    print("  3. Run tests:       python -m pytest tests/\n")


if __name__ == "__main__":
    main()
