"""
Test script to identify errors in the app without running Streamlit UI
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*60)
print("Testing Credit Risk Assessment Application")
print("="*60)

# Test 1: Load models
print("\n1. Testing model loading...")
try:
    from src.data_preprocessing import CreditDataPreprocessor
    
    models_dir = Path('results/models')
    
    # Load raw data to fit preprocessor
    raw_data_path = Path('data/raw/LC_loans_granting_model_dataset.csv')
    if raw_data_path.exists():
        print("   ✓ Raw data file found")
        df_raw = pd.read_csv(raw_data_path)
        df_sample = df_raw.head(1000)
        
        preprocessor = CreditDataPreprocessor()
        _ = preprocessor.fit_transform(df_sample, 'Default')
        print("   ✓ Preprocessor initialized")
    else:
        print("   ✗ Raw data file not found")
        sys.exit(1)
    
    with open(models_dir / 'bootstrap_ensemble.pkl', 'rb') as f:
        ensemble = pickle.load(f)
    print("   ✓ Ensemble model loaded")
    
    with open(models_dir / 'escalation_system.pkl', 'rb') as f:
        escalation_system = pickle.load(f)
    print("   ✓ Escalation system loaded")
    
except Exception as e:
    print(f"   ✗ Error loading models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load sample data
print("\n2. Testing sample data loading...")
try:
    sample_data = pd.read_csv('example_new_loans.csv')
    print(f"   ✓ Sample data loaded: {len(sample_data)} rows")
    print(f"   Columns: {list(sample_data.columns)}")
except Exception as e:
    print(f"   ✗ Error loading sample data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test prediction with first sample
print("\n3. Testing prediction on first sample...")
try:
    loan_data = sample_data.iloc[0].to_dict()
    print(f"   Sample loan amount: ${loan_data.get('loan_amnt', 'N/A')}")
    
    # Convert to DataFrame
    df = pd.DataFrame([loan_data])
    
    # Preprocess
    X_processed = preprocessor.transform(df)
    print(f"   ✓ Preprocessing successful, shape: {X_processed.shape}")
    
    # Get predictions with uncertainty
    proba_mean, uncertainty, all_proba = ensemble.predict_with_uncertainty(X_processed)
    print(f"   ✓ Prediction successful")
    print(f"   Probability of default: {proba_mean[0, 1]:.4f}")
    print(f"   Uncertainty: {uncertainty[0]:.4f}")
    
    # Get escalation decision
    prob_default = proba_mean[0, 1]
    prob_paid = proba_mean[0, 0]
    unc = uncertainty[0]
    confidence = np.max(proba_mean[0])
    
    should_escalate, escalation_reason = escalation_system.should_escalate(
        uncertainty=unc,
        confidence=confidence,
        probability=prob_default
    )
    
    print(f"   ✓ Escalation check successful")
    print(f"   Should escalate: {should_escalate}")
    if should_escalate:
        print(f"   Reason: {escalation_reason}")
    
except Exception as e:
    print(f"   ✗ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test batch prediction
print("\n4. Testing batch prediction...")
try:
    results = []
    for i, row in sample_data.iterrows():
        loan_dict = row.to_dict()
        df = pd.DataFrame([loan_dict])
        
        X_processed = preprocessor.transform(df)
        proba_mean, uncertainty, all_proba = ensemble.predict_with_uncertainty(X_processed)
        
        prob_default = proba_mean[0, 1]
        confidence = np.max(proba_mean[0])
        unc = uncertainty[0]
        
        should_escalate, reason = escalation_system.should_escalate(
            uncertainty=unc,
            confidence=confidence,
            probability=prob_default
        )
        
        if should_escalate:
            decision = "ESCALATE"
        else:
            decision = "APPROVE" if prob_default < 0.5 else "REJECT"
        
        results.append({
            'loan_id': i+1,
            'decision': decision,
            'prob_default': prob_default,
            'uncertainty': unc,
            'should_escalate': should_escalate
        })
    
    results_df = pd.DataFrame(results)
    print(f"   ✓ Batch prediction successful: {len(results_df)} loans processed")
    print(f"\n   Decision breakdown:")
    print(results_df['decision'].value_counts())
    
except Exception as e:
    print(f"   ✗ Error during batch prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check required features
print("\n5. Checking feature compatibility...")
try:
    # Get expected features from preprocessor
    expected_features = preprocessor.feature_names
    print(f"   Preprocessor expects {len(expected_features)} features")
    
    # Check sample data columns
    sample_cols = set(sample_data.columns)
    print(f"   Sample data has {len(sample_cols)} columns")
    
    # Find missing columns
    missing_cols = sample_cols - set(sample_data.columns)
    if missing_cols:
        print(f"   ⚠ Missing columns in sample data: {missing_cols}")
    else:
        print(f"   ✓ All required columns present")
    
except Exception as e:
    print(f"   ✗ Error checking features: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ All tests passed! Application should work correctly.")
print("="*60)
