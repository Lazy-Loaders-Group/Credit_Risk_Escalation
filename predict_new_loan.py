"""
Credit Risk Prediction with Uncertainty-Based Escalation

This script predicts loan approval/rejection for new applications and
automatically escalates high-uncertainty cases to human agents.

Usage:
    python predict_new_loan.py --input new_loan.csv
    python predict_new_loan.py --interactive

Author: Credit Risk ML Team
Date: 2024
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class CreditRiskPredictor:
    """
    Main prediction system with uncertainty-based escalation
    """
    
    def __init__(self, models_dir='results/models'):
        """
        Initialize predictor by loading trained models
        
        Parameters:
        -----------
        models_dir : str
            Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.load_models()
        
    def load_models(self):
        """Load all required models and components"""
        print("Loading trained models...")
        
        try:
            # Load preprocessor (check if it's a dict or object)
            with open(self.models_dir / 'preprocessor.pkl', 'rb') as f:
                preprocessor_obj = pickle.load(f)
                # If it's a dict with a 'preprocessor' key, extract it
                if isinstance(preprocessor_obj, dict):
                    self.preprocessor = preprocessor_obj.get('preprocessor', preprocessor_obj)
                    self.feature_names = preprocessor_obj.get('feature_names', [])
                else:
                    self.preprocessor = preprocessor_obj
                    self.feature_names = []
            print("  ✓ Preprocessor loaded")
            
            # Load bootstrap ensemble (for uncertainty)
            with open(self.models_dir / 'bootstrap_ensemble.pkl', 'rb') as f:
                self.ensemble = pickle.load(f)
            print("  ✓ Bootstrap ensemble loaded")
            
            # Load escalation system
            with open(self.models_dir / 'escalation_system.pkl', 'rb') as f:
                self.escalation_system = pickle.load(f)
            print("  ✓ Escalation system loaded")
            
            print("\n✅ All models loaded successfully!\n")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: Could not find model file: {e}")
            print("\nPlease run the training notebooks first:")
            print("  1. notebooks/02_baseline_model.ipynb")
            print("  2. notebooks/03_uncertainty_quantification.ipynb")
            print("  3. notebooks/04_escalation_system.ipynb")
            sys.exit(1)
    
    def predict_single(self, loan_data):
        """
        Predict for a single loan application
        
        Parameters:
        -----------
        loan_data : dict or pd.DataFrame
            Loan application data
            
        Returns:
        --------
        result : dict
            Prediction results with decision and explanation
        """
        # Convert to DataFrame if dict
        if isinstance(loan_data, dict):
            df = pd.DataFrame([loan_data])
        else:
            df = loan_data.copy()
        
        # Preprocess
        try:
            X_processed = self.preprocessor.transform(df)
        except Exception as e:
            return {
                'error': f"Preprocessing failed: {str(e)}",
                'action': 'ESCALATE',
                'reason': 'Invalid input data',
                'color': '🔴'
            }
        
        # Get predictions with uncertainty
        proba_mean, uncertainty, all_proba = self.ensemble.predict_with_uncertainty(X_processed)
        
        # Extract values for single sample
        prob_default = proba_mean[0, 1]
        prob_paid = proba_mean[0, 0]
        unc = uncertainty[0]
        confidence = np.max(proba_mean[0])
        
        # Get escalation decision
        should_escalate, escalation_reason = self.escalation_system.should_escalate(
            uncertainty=unc,
            confidence=confidence,
            probability=prob_default
        )
        
        # Make decision
        if should_escalate:
            action = "ESCALATE TO AGENT"
            decision = "PENDING HUMAN REVIEW"
            color = "🔴"
        else:
            # Automated decision
            if prob_default >= 0.5:
                decision = "REJECT"
                color = "🚫"
            else:
                decision = "APPROVE"
                color = "✅"
            action = f"AUTOMATED {decision}"
        
        # Create result
        result = {
            'action': action,
            'decision': decision,
            'probability_default': round(prob_default, 4),
            'probability_paid': round(prob_paid, 4),
            'confidence': round(confidence, 4),
            'uncertainty': round(unc, 4),
            'should_escalate': should_escalate,
            'escalation_reason': escalation_reason,
            'color': color
        }
        
        return result
    
    def predict_batch(self, loans_df):
        """
        Predict for multiple loan applications
        
        Parameters:
        -----------
        loans_df : pd.DataFrame
            Multiple loan applications
            
        Returns:
        --------
        results_df : pd.DataFrame
            Predictions for all applications
        """
        print(f"Processing {len(loans_df)} loan applications...\n")
        
        # Preprocess
        try:
            X_processed = self.preprocessor.transform(loans_df)
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return None
        
        # Get predictions with uncertainty
        proba_mean, uncertainty, all_proba = self.ensemble.predict_with_uncertainty(X_processed)
        
        # Get escalation decisions
        escalate_mask, details = self.escalation_system.process_predictions(
            proba_mean, uncertainty, return_details=True
        )
        
        # Create results DataFrame
        results = pd.DataFrame({
            'application_id': range(1, len(loans_df) + 1),
            'probability_default': proba_mean[:, 1],
            'probability_paid': proba_mean[:, 0],
            'uncertainty': uncertainty,
            'confidence': np.max(proba_mean, axis=1),
            'should_escalate': escalate_mask,
            'automated_decision': ['REJECT' if p >= 0.5 else 'APPROVE' 
                                   for p in proba_mean[:, 1]],
        })
        
        # Add final action
        results['final_action'] = results.apply(
            lambda row: 'ESCALATE TO AGENT' if row['should_escalate'] 
            else f"AUTOMATED {row['automated_decision']}", 
            axis=1
        )
        
        # Add escalation reasons
        results = results.merge(
            details[['sample_idx', 'reason']], 
            left_index=True, 
            right_on='sample_idx', 
            how='left'
        ).drop('sample_idx', axis=1)
        results.rename(columns={'reason': 'escalation_reason'}, inplace=True)
        
        # Summary statistics
        n_total = len(results)
        n_escalated = escalate_mask.sum()
        n_automated = n_total - n_escalated
        automation_rate = (n_automated / n_total) * 100
        
        print(f"{'='*60}")
        print(f"BATCH PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total applications:      {n_total}")
        print(f"Automated decisions:     {n_automated} ({automation_rate:.1f}%)")
        print(f"Escalated to agents:     {n_escalated} ({100-automation_rate:.1f}%)")
        print(f"{'='*60}\n")
        
        if n_automated > 0:
            n_approve = (results['automated_decision'] == 'APPROVE').sum() - n_escalated
            n_reject = (results['automated_decision'] == 'REJECT').sum()
            print(f"Automated Approvals:     {n_approve}")
            print(f"Automated Rejections:    {n_reject}")
            print(f"{'='*60}\n")
        
        return results
    
    def print_prediction(self, result):
        """Pretty print prediction result"""
        print(f"\n{'='*60}")
        print(f"{result['color']} LOAN PREDICTION RESULT")
        print(f"{'='*60}\n")
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            print(f"   Action: {result['action']}")
            print(f"   Reason: {result['reason']}")
        else:
            print(f"FINAL ACTION:       {result['action']}")
            print(f"Decision:           {result['decision']}")
            print(f"\nPrediction Details:")
            print(f"  Probability of Default:  {result['probability_default']:.2%}")
            print(f"  Probability of Payment:  {result['probability_paid']:.2%}")
            print(f"  Confidence Level:        {result['confidence']:.2%}")
            print(f"  Uncertainty Score:       {result['uncertainty']:.4f}")
            
            if result['should_escalate']:
                print(f"\n⚠️  ESCALATION REQUIRED")
                print(f"  Reason: {result['escalation_reason']}")
                print(f"\n👤 ACTION: Please forward this application to a human agent")
                print(f"   for manual review and final decision.")
            else:
                print(f"\n✅ CONFIDENT AUTOMATED DECISION")
                print(f"   The model is confident in this prediction.")
                print(f"   No human review required.")
        
        print(f"\n{'='*60}\n")


def create_sample_loan():
    """Create a sample loan application for testing"""
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


def interactive_mode(predictor):
    """Interactive mode for single predictions"""
    print("\n" + "="*60)
    print("  CREDIT RISK PREDICTION - INTERACTIVE MODE")
    print("="*60)
    print("\nEnter loan application details (or press Enter for sample data):\n")
    
    # For simplicity, use sample data or load from user input
    use_sample = input("Use sample loan data? (y/n): ").lower().strip()
    
    if use_sample == 'y':
        loan_data = create_sample_loan()
        print("\n✓ Using sample loan data")
    else:
        print("\n⚠️  Manual data entry not implemented yet.")
        print("   Please use CSV file input or modify the script.")
        return
    
    # Make prediction
    result = predictor.predict_single(loan_data)
    
    # Display result
    predictor.print_prediction(result)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Predict credit risk with uncertainty-based escalation'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to CSV file with loan application(s)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save prediction results (CSV)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='results/models',
        help='Directory containing trained models'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = CreditRiskPredictor(models_dir=args.models_dir)
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        sys.exit(1)
    
    # Run appropriate mode
    if args.interactive:
        interactive_mode(predictor)
    
    elif args.input:
        # Load input data
        try:
            loans_df = pd.read_csv(args.input)
            print(f"✓ Loaded {len(loans_df)} loan applications from {args.input}\n")
        except Exception as e:
            print(f"❌ Error loading input file: {e}")
            sys.exit(1)
        
        # Make predictions
        results = predictor.predict_batch(loans_df)
        
        if results is not None:
            # Save results if output path provided
            if args.output:
                results.to_csv(args.output, index=False)
                print(f"✅ Results saved to: {args.output}")
            else:
                # Display first few results
                print("\nFirst 10 predictions:")
                print(results.head(10).to_string())
    
    else:
        print("❌ Error: Please specify --input or --interactive mode")
        print("\nUsage examples:")
        print("  python predict_new_loan.py --interactive")
        print("  python predict_new_loan.py --input new_loans.csv")
        print("  python predict_new_loan.py --input new_loans.csv --output predictions.csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
