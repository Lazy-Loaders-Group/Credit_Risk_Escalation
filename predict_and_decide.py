"""
Prediction and Decision System for Credit Risk Assessment

This script loads saved models and makes predictions with decisions for loan applications.

**IMPORTANT:** This script works with PREPROCESSED data (same format as training data).
For raw loan data preprocessing and prediction, use the Streamlit app (app.py) or train new models.

The models were trained using the notebooks and expect specific preprocessed features.
Use this script with data in the same format as data/splits/X_test.csv

Usage:
    # Use with preprocessed data
    python predict_and_decide.py --input data/splits/X_test.csv
    
    # Specify output file
    python predict_and_decide.py --input data/splits/X_test.csv --output results/reports/predictions_demo.csv
    
    # Interactive mode (uses sample preprocessed data)
    python predict_and_decide.py --interactive

For raw data prediction, use:
    streamlit run app.py

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
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class PredictionSystem:
    """
    Complete prediction and decision system
    """
    
    def __init__(self, models_dir='results/models', config=None):
        """
        Initialize the prediction system
        
        Parameters:
        -----------
        models_dir : str
            Directory containing saved models
        config : dict, optional
            Decision configuration with thresholds
        """
        self.models_dir = Path(models_dir)
        
        # Default configuration
        self.config = {
            'approve_threshold': 0.2,
            'reject_threshold': 0.8,
            'uncertainty_accept': 0.15,
            'use_escalation': True
        }
        
        if config:
            self.config.update(config)
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        print("Loading models...")
        
        try:
            # Load preprocessor info (just metadata in current version)
            with open(self.models_dir / 'preprocessor.pkl', 'rb') as f:
                preproc_data = pickle.load(f)
                if isinstance(preproc_data, dict):
                    self.feature_names = preproc_data.get('feature_names', [])
                    self.preprocessor = None  # No actual preprocessor, we'll use raw features
                    print("  ✓ Preprocessor metadata loaded")
                else:
                    self.preprocessor = preproc_data
                    self.feature_names = []
                    print("  ✓ Preprocessor loaded")
            
            # Load ensemble
            with open(self.models_dir / 'bootstrap_ensemble.pkl', 'rb') as f:
                self.ensemble = pickle.load(f)
            print("  ✓ Bootstrap ensemble loaded")
            
            # Load escalation system
            with open(self.models_dir / 'escalation_system.pkl', 'rb') as f:
                self.escalation_system = pickle.load(f)
            print("  ✓ Escalation system loaded")
            
            # Load metadata
            try:
                with open(self.models_dir / 'model_metadata.json', 'r') as f:
                    self.metadata = json.load(f)
                print("  ✓ Metadata loaded")
                print(f"\nModel Version: {self.metadata.get('version', 'N/A')}")
                print(f"Training Date: {self.metadata.get('training_date', 'N/A')}")
            except FileNotFoundError:
                self.metadata = {}
                print("  ⚠ Metadata not found (optional)")
            
            print("\n✅ All models loaded successfully!\n")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: Model file not found: {e}")
            print("\nPlease train models first:")
            print("  python train_and_save.py")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error loading models: {e}")
            sys.exit(1)
    
    def predict_single(self, loan_data):
        """
        Predict for a single loan application
        
        Parameters:
        -----------
        loan_data : dict or pd.Series
            Loan application data
            
        Returns:
        --------
        result : dict
            Prediction with decision
        """
        # Convert to DataFrame
        if isinstance(loan_data, dict):
            df = pd.DataFrame([loan_data])
        elif isinstance(loan_data, pd.Series):
            df = pd.DataFrame([loan_data])
        else:
            df = loan_data.copy()
        
        # Preprocess
        try:
            if self.preprocessor is not None:
                X_processed = self.preprocessor.transform(df)
            else:
                # Work with preprocessed data directly
                X_processed = df.copy()
                id_cols = [col for col in X_processed.columns if col.lower() in ['id', 'index', 'unnamed: 0', 'unnamed']]
                if id_cols:
                    for col in id_cols:
                        if col in X_processed.columns:
                            X_processed = X_processed.drop(columns=[col])
                
                if self.feature_names:
                    available_features = [f for f in self.feature_names if f in X_processed.columns]
                    X_processed = X_processed[available_features]
                
                X_processed = X_processed.values
        except Exception as e:
            return {
                'id': loan_data.get('id', 'unknown') if isinstance(loan_data, dict) else 'unknown',
                'score': 0.0,
                'uncertainty': 1.0,
                'decision': 'ERROR',
                'model_version': self.metadata.get('version', 'N/A'),
                'error': str(e)
            }
        
        # Get predictions with uncertainty
        proba_mean, uncertainty, _ = self.ensemble.predict_with_uncertainty(X_processed)
        
        score = proba_mean[0, 1]  # Probability of default
        unc = uncertainty[0]
        confidence = np.max(proba_mean[0])
        
        # Make decision
        decision = self.make_decision(score, unc, confidence)
        
        return {
            'id': loan_data.get('id', 'unknown') if isinstance(loan_data, dict) else 'unknown',
            'score': round(float(score), 4),
            'uncertainty': round(float(unc), 4),
            'confidence': round(float(confidence), 4),
            'decision': decision,
            'model_version': self.metadata.get('version', 'N/A')
        }
    
    def make_decision(self, score, uncertainty, confidence):
        """
        Make decision based on score and uncertainty
        
        Parameters:
        -----------
        score : float
            Probability of default
        uncertainty : float
            Uncertainty score
        confidence : float
            Confidence score
            
        Returns:
        --------
        decision : str
            One of: APPROVE, REJECT, ESCALATE_TO_HUMAN
        """
        if self.config['use_escalation']:
            # Use escalation system
            should_escalate, _ = self.escalation_system.should_escalate(
                uncertainty=uncertainty,
                confidence=confidence,
                probability=score
            )
            
            if should_escalate:
                return 'ESCALATE_TO_HUMAN'
        
        # Simple threshold-based decision
        if score <= self.config['approve_threshold'] and uncertainty <= self.config['uncertainty_accept']:
            return 'APPROVE'
        elif score >= self.config['reject_threshold'] and uncertainty <= self.config['uncertainty_accept']:
            return 'REJECT'
        else:
            return 'ESCALATE_TO_HUMAN'
    
    def predict_batch(self, loans_df, show_progress=True):
        """
        Predict for multiple loan applications
        
        Parameters:
        -----------
        loans_df : pd.DataFrame
            Multiple loan applications
        show_progress : bool
            Whether to show progress
            
        Returns:
        --------
        results_df : pd.DataFrame
            Predictions for all applications
        """
        if show_progress:
            print(f"Processing {len(loans_df)} loan applications...\n")
        
        # Preprocess all data
        try:
            if self.preprocessor is not None:
                X_processed = self.preprocessor.transform(loans_df)
            else:
                # Work with preprocessed data directly (from training splits)
                # Drop ID columns if present
                X_processed = loans_df.copy()
                id_cols = [col for col in X_processed.columns if col.lower() in ['id', 'index', 'unnamed: 0', 'unnamed']]
                if id_cols:
                    for col in id_cols:
                        if col in X_processed.columns:
                            X_processed = X_processed.drop(columns=[col])
                
                # If we have feature names, select only those
                if self.feature_names:
                    # Check if feature names exist in data
                    available_features = [f for f in self.feature_names if f in X_processed.columns]
                    X_processed = X_processed[available_features]
                
                X_processed = X_processed.values
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            print(f"   Error details: {str(e)}")
            print(f"   Available columns: {loans_df.columns.tolist()}")
            return None
        
        # Get predictions with uncertainty
        proba_mean, uncertainty, _ = self.ensemble.predict_with_uncertainty(X_processed)
        
        # Calculate confidence
        confidence = np.max(proba_mean, axis=1)
        
        # Make decisions for all samples
        decisions = []
        for i in range(len(loans_df)):
            decision = self.make_decision(
                proba_mean[i, 1],
                uncertainty[i],
                confidence[i]
            )
            decisions.append(decision)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'id': range(1, len(loans_df) + 1) if 'id' not in loans_df.columns else loans_df['id'],
            'score': proba_mean[:, 1],
            'uncertainty': uncertainty,
            'decision': decisions,
            'model_version': self.metadata.get('version', 'N/A')
        })
        
        # Round numeric columns
        results['score'] = results['score'].round(4)
        results['uncertainty'] = results['uncertainty'].round(4)
        
        if show_progress:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print summary statistics"""
        n_total = len(results)
        n_approve = (results['decision'] == 'APPROVE').sum()
        n_reject = (results['decision'] == 'REJECT').sum()
        n_escalate = (results['decision'] == 'ESCALATE_TO_HUMAN').sum()
        
        print(f"{'='*60}")
        print(f"PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total applications:      {n_total}")
        print(f"  • Approved:            {n_approve} ({n_approve/n_total*100:.1f}%)")
        print(f"  • Rejected:            {n_reject} ({n_reject/n_total*100:.1f}%)")
        print(f"  • Escalated to human:  {n_escalate} ({n_escalate/n_total*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Automation rate:         {(n_approve+n_reject)/n_total*100:.1f}%")
        print(f"{'='*60}\n")


def create_sample_loan():
    """Create a sample loan application"""
    return {
        'loan_amnt': 10000,
        'term': '36 months',
        'int_rate': 10.5,
        'installment': 325.0,
        'grade': 'B',
        'sub_grade': 'B3',
        'emp_length': '5 years',
        'home_ownership': 'MORTGAGE',
        'annual_inc': 50000,
        'verification_status': 'Verified',
        'purpose': 'debt_consolidation',
        'dti': 15.5,
        'delinq_2yrs': 0,
        'inq_last_6mths': 1,
        'open_acc': 8,
        'pub_rec': 0,
        'revol_bal': 5000,
        'revol_util': 50.0,
        'total_acc': 15,
    }


def interactive_mode(system):
    """Interactive mode for single predictions"""
    print("\n" + "="*60)
    print("  CREDIT RISK PREDICTION - INTERACTIVE MODE")
    print("="*60 + "\n")
    
    use_sample = input("Use sample loan data? (y/n): ").lower().strip()
    
    if use_sample == 'y':
        loan_data = create_sample_loan()
        print("\n✓ Using sample loan data\n")
        
        # Show loan data
        print("Loan Application:")
        for key, value in loan_data.items():
            print(f"  {key:20s}: {value}")
        print()
    else:
        print("\n⚠️  Manual entry mode not implemented yet.")
        print("Please use CSV input or modify the sample data.\n")
        return
    
    # Make prediction
    result = system.predict_single(loan_data)
    
    # Display result
    print("="*60)
    print("PREDICTION RESULT")
    print("="*60)
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"ID:              {result['id']}")
        print(f"Score:           {result['score']} (probability of default)")
        print(f"Uncertainty:     {result['uncertainty']}")
        print(f"Decision:        {result['decision']}")
        print(f"Model Version:   {result['model_version']}")
        
        # Add explanation
        if result['decision'] == 'APPROVE':
            print("\n✅ Recommendation: APPROVE this loan")
            print("   Low risk of default with high confidence")
        elif result['decision'] == 'REJECT':
            print("\n🚫 Recommendation: REJECT this loan")
            print("   High risk of default with high confidence")
        else:
            print("\n🔴 Recommendation: ESCALATE to human review")
            print("   Uncertain prediction requires expert judgment")
    print("="*60 + "\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Predict and decide on loan applications'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to CSV file with loan applications'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save predictions CSV'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--use-saved-model',
        action='store_true',
        help='Use saved model (default behavior)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='results/models',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--approve-threshold',
        type=float,
        default=0.2,
        help='Threshold for auto-approval (default: 0.2)'
    )
    parser.add_argument(
        '--reject-threshold',
        type=float,
        default=0.8,
        help='Threshold for auto-rejection (default: 0.8)'
    )
    parser.add_argument(
        '--uncertainty-accept',
        type=float,
        default=0.15,
        help='Maximum acceptable uncertainty (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'approve_threshold': args.approve_threshold,
        'reject_threshold': args.reject_threshold,
        'uncertainty_accept': args.uncertainty_accept,
        'use_escalation': True
    }
    
    # Initialize system
    try:
        system = PredictionSystem(models_dir=args.models_dir, config=config)
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)
    
    # Run appropriate mode
    if args.interactive:
        interactive_mode(system)
    
    elif args.input:
        # Load input data
        try:
            loans_df = pd.read_csv(args.input)
            print(f"✓ Loaded {len(loans_df)} applications from {args.input}\n")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            sys.exit(1)
        
        # Make predictions
        results = system.predict_batch(loans_df)
        
        if results is not None:
            # Save or display results
            if args.output:
                # Ensure output directory exists
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                results.to_csv(output_path, index=False)
                print(f"✅ Predictions saved to: {output_path}\n")
            else:
                # Display first 10 results
                print("\nFirst 10 predictions:")
                print(results.head(10).to_string(index=False))
                print(f"\n... ({len(results)} total rows)\n")
    
    else:
        # No mode specified
        print("\n❌ Error: Please specify --input or --interactive\n")
        print("Usage examples:")
        print("  python predict_and_decide.py --input example_new_loans.csv")
        print("  python predict_and_decide.py --input loans.csv --output predictions.csv")
        print("  python predict_and_decide.py --interactive\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
