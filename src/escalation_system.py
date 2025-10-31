"""
Human Escalation System for Credit Risk Assessment

This module implements an intelligent escalation system that:
- Identifies high-uncertainty predictions requiring human review
- Optimizes escalation thresholds based on cost-benefit analysis
- Tracks automation metrics and performance
- Provides explainability for escalated cases

Author: Credit Risk ML Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class EscalationSystem:
    """
    Intelligent Escalation System for High-Uncertainty Predictions
    
    This system automatically identifies predictions that should be
    escalated to human experts based on uncertainty estimates and
    configurable business rules.
    
    Parameters:
    -----------
    uncertainty_threshold : float, default=0.1
        Uncertainty above which predictions are escalated
    confidence_threshold : float, default=0.7
        Confidence below which predictions are escalated
    cost_false_positive : float, default=1.0
        Cost of incorrectly approving a loan (false positive)
    cost_false_negative : float, default=5.0
        Cost of incorrectly denying a loan (false negative)
    cost_human_review : float, default=0.5
        Cost of escalating to human review
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.1,
        confidence_threshold: float = 0.7,
        cost_false_positive: float = 1.0,
        cost_false_negative: float = 5.0,
        cost_human_review: float = 0.5
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        self.cost_false_positive = cost_false_positive
        self.cost_false_negative = cost_false_negative
        self.cost_human_review = cost_human_review
        
        # Tracking metrics
        self.escalation_history = []
        self.performance_history = []
        
    def should_escalate(
        self,
        uncertainty: float,
        confidence: float,
        probability: float
    ) -> Tuple[bool, str]:
        """
        Determine if a prediction should be escalated
        
        Parameters:
        -----------
        uncertainty : float
            Uncertainty score for the prediction
        confidence : float
            Confidence score (max probability)
        probability : float
            Predicted probability of default
            
        Returns:
        --------
        escalate : bool
            Whether to escalate this prediction
        reason : str
            Reason for escalation decision
        """
        reasons = []
        
        # High uncertainty
        if uncertainty > self.uncertainty_threshold:
            reasons.append(f"High uncertainty ({uncertainty:.4f} > {self.uncertainty_threshold})")
        
        # Low confidence
        if confidence < self.confidence_threshold:
            reasons.append(f"Low confidence ({confidence:.4f} < {self.confidence_threshold})")
        
        # Borderline probability (near decision boundary)
        if 0.4 <= probability <= 0.6:
            reasons.append(f"Borderline probability ({probability:.4f})")
        
        # Make decision
        escalate = len(reasons) > 0
        reason = "; ".join(reasons) if escalate else "Confident prediction"
        
        return escalate, reason
    
    def process_predictions(
        self,
        y_proba: np.ndarray,
        uncertainty: np.ndarray,
        return_details: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        """
        Process predictions and determine escalations
        
        Parameters:
        -----------
        y_proba : array-like, shape (n_samples, 2)
            Predicted probabilities
        uncertainty : array-like, shape (n_samples,)
            Uncertainty scores
        return_details : bool, default=False
            Whether to return detailed escalation information
            
        Returns:
        --------
        escalate_mask : array-like, shape (n_samples,)
            Boolean mask indicating which samples to escalate
        details : pd.DataFrame, optional
            Detailed escalation information (if return_details=True)
        """
        n_samples = len(y_proba)
        escalate_mask = np.zeros(n_samples, dtype=bool)
        
        if return_details:
            details = []
        
        for i in range(n_samples):
            prob_default = y_proba[i, 1]
            confidence = np.max(y_proba[i])
            unc = uncertainty[i]
            
            escalate, reason = self.should_escalate(unc, confidence, prob_default)
            escalate_mask[i] = escalate
            
            if return_details:
                details.append({
                    'sample_idx': i,
                    'probability': prob_default,
                    'confidence': confidence,
                    'uncertainty': unc,
                    'escalate': escalate,
                    'reason': reason
                })
        
        if return_details:
            return escalate_mask, pd.DataFrame(details)
        else:
            return escalate_mask
    
    def calculate_costs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        escalate_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate total costs including escalation
        
        Parameters:
        -----------
        y_true : array-like, shape (n_samples,)
            True labels
        y_pred : array-like, shape (n_samples,)
            Predicted labels (for non-escalated samples)
        escalate_mask : array-like, shape (n_samples,)
            Boolean mask indicating escalated samples
            
        Returns:
        --------
        costs : dict
            Dictionary with cost breakdown
        """
        # Non-escalated samples
        non_escalated = ~escalate_mask
        
        if np.sum(non_escalated) > 0:
            # Calculate prediction errors on non-escalated samples
            y_true_ne = y_true[non_escalated]
            y_pred_ne = y_pred[non_escalated]
            
            # False positives (predict default when actually paid)
            fp = np.sum((y_pred_ne == 1) & (y_true_ne == 0))
            
            # False negatives (predict paid when actually default)
            fn = np.sum((y_pred_ne == 0) & (y_true_ne == 1))
        else:
            fp = 0
            fn = 0
        
        # Number of escalations
        n_escalated = np.sum(escalate_mask)
        
        # Calculate costs
        cost_fp = fp * self.cost_false_positive
        cost_fn = fn * self.cost_false_negative
        cost_escalation = n_escalated * self.cost_human_review
        total_cost = cost_fp + cost_fn + cost_escalation
        
        # Baseline cost (no escalation)
        fp_baseline = np.sum((y_pred == 1) & (y_true == 0))
        fn_baseline = np.sum((y_pred == 0) & (y_true == 1))
        baseline_cost = (fp_baseline * self.cost_false_positive + 
                        fn_baseline * self.cost_false_negative)
        
        cost_savings = baseline_cost - total_cost
        
        return {
            'false_positive_cost': cost_fp,
            'false_negative_cost': cost_fn,
            'escalation_cost': cost_escalation,
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'cost_savings': cost_savings,
            'n_escalated': n_escalated,
            'n_false_positives': fp,
            'n_false_negatives': fn
        }
    
    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: np.ndarray,
        uncertainty_range: Tuple[float, float] = (0.05, 0.25),
        confidence_range: Tuple[float, float] = (0.5, 0.9),
        n_steps: int = 20
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Optimize escalation thresholds using grid search
        
        Parameters:
        -----------
        y_true : array-like, shape (n_samples,)
            True labels
        y_proba : array-like, shape (n_samples, 2)
            Predicted probabilities
        y_pred : array-like, shape (n_samples,)
            Predicted labels
        uncertainty : array-like, shape (n_samples,)
            Uncertainty scores
        uncertainty_range : tuple, default=(0.05, 0.25)
            Range of uncertainty thresholds to try
        confidence_range : tuple, default=(0.5, 0.9)
            Range of confidence thresholds to try
        n_steps : int, default=20
            Number of steps in grid search
            
        Returns:
        --------
        best_unc_threshold : float
            Optimal uncertainty threshold
        best_conf_threshold : float
            Optimal confidence threshold
        results : pd.DataFrame
            Grid search results
        """
        print("Optimizing escalation thresholds...")
        
        # Generate threshold candidates
        unc_thresholds = np.linspace(uncertainty_range[0], uncertainty_range[1], n_steps)
        conf_thresholds = np.linspace(confidence_range[0], confidence_range[1], n_steps)
        
        results = []
        best_cost = float('inf')
        best_unc_threshold = self.uncertainty_threshold
        best_conf_threshold = self.confidence_threshold
        
        for unc_thresh in unc_thresholds:
            for conf_thresh in conf_thresholds:
                # Temporarily set thresholds
                original_unc = self.uncertainty_threshold
                original_conf = self.confidence_threshold
                
                self.uncertainty_threshold = unc_thresh
                self.confidence_threshold = conf_thresh
                
                # Process predictions
                escalate_mask = self.process_predictions(y_proba, uncertainty)
                
                # Calculate costs
                costs = self.calculate_costs(y_true, y_pred, escalate_mask)
                
                # Calculate automation rate
                automation_rate = 1 - (np.sum(escalate_mask) / len(escalate_mask))
                
                # Calculate accuracy on non-escalated
                non_escalated = ~escalate_mask
                if np.sum(non_escalated) > 0:
                    accuracy_auto = accuracy_score(
                        y_true[non_escalated],
                        y_pred[non_escalated]
                    )
                else:
                    accuracy_auto = 0.0
                
                results.append({
                    'uncertainty_threshold': unc_thresh,
                    'confidence_threshold': conf_thresh,
                    'total_cost': costs['total_cost'],
                    'cost_savings': costs['cost_savings'],
                    'automation_rate': automation_rate,
                    'n_escalated': costs['n_escalated'],
                    'accuracy_automated': accuracy_auto
                })
                
                # Update best
                if costs['total_cost'] < best_cost:
                    best_cost = costs['total_cost']
                    best_unc_threshold = unc_thresh
                    best_conf_threshold = conf_thresh
                
                # Restore thresholds
                self.uncertainty_threshold = original_unc
                self.confidence_threshold = original_conf
        
        # Set optimal thresholds
        self.uncertainty_threshold = best_unc_threshold
        self.confidence_threshold = best_conf_threshold
        
        results_df = pd.DataFrame(results).sort_values('total_cost')
        
        print(f"\n✅ Optimization complete!")
        print(f"   Best uncertainty threshold: {best_unc_threshold:.4f}")
        print(f"   Best confidence threshold:  {best_conf_threshold:.4f}")
        print(f"   Total cost: {best_cost:.4f}")
        
        return best_unc_threshold, best_conf_threshold, results_df
    
    def evaluate_system(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        uncertainty: np.ndarray
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of escalation system
        
        Parameters:
        -----------
        y_true : array-like, shape (n_samples,)
            True labels
        y_pred : array-like, shape (n_samples,)
            Predicted labels
        y_proba : array-like, shape (n_samples, 2)
            Predicted probabilities
        uncertainty : array-like, shape (n_samples,)
            Uncertainty scores
            
        Returns:
        --------
        metrics : dict
            Comprehensive evaluation metrics
        """
        # Get escalation decisions
        escalate_mask, details = self.process_predictions(
            y_proba, uncertainty, return_details=True
        )
        
        # Calculate costs
        costs = self.calculate_costs(y_true, y_pred, escalate_mask)
        
        # Automation metrics
        n_total = len(y_true)
        n_automated = np.sum(~escalate_mask)
        automation_rate = n_automated / n_total
        
        # Performance on automated decisions
        if n_automated > 0:
            y_true_auto = y_true[~escalate_mask]
            y_pred_auto = y_pred[~escalate_mask]
            
            accuracy_auto = accuracy_score(y_true_auto, y_pred_auto)
            precision_auto = precision_score(y_true_auto, y_pred_auto, zero_division=0)
            recall_auto = recall_score(y_true_auto, y_pred_auto, zero_division=0)
            f1_auto = f1_score(y_true_auto, y_pred_auto, zero_division=0)
        else:
            accuracy_auto = precision_auto = recall_auto = f1_auto = 0.0
        
        # Performance on escalated cases
        n_escalated = np.sum(escalate_mask)
        if n_escalated > 0:
            y_true_esc = y_true[escalate_mask]
            y_pred_esc = y_pred[escalate_mask]
            
            # Accuracy if we had automated these
            accuracy_esc = accuracy_score(y_true_esc, y_pred_esc)
        else:
            accuracy_esc = 0.0
        
        # Overall metrics
        metrics = {
            'automation_rate': automation_rate,
            'n_automated': n_automated,
            'n_escalated': n_escalated,
            'accuracy_automated': accuracy_auto,
            'precision_automated': precision_auto,
            'recall_automated': recall_auto,
            'f1_automated': f1_auto,
            'accuracy_escalated_if_automated': accuracy_esc,
            **costs
        }
        
        # Store in history
        self.performance_history.append(metrics)
        
        return metrics
    
    def get_escalation_summary(
        self,
        y_proba: np.ndarray,
        uncertainty: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get detailed summary of escalated cases
        
        Parameters:
        -----------
        y_proba : array-like, shape (n_samples, 2)
            Predicted probabilities
        uncertainty : array-like, shape (n_samples,)
            Uncertainty scores
        feature_names : list, optional
            Feature names for additional context
        X : array-like, optional
            Feature values for escalated cases
            
        Returns:
        --------
        summary : pd.DataFrame
            Detailed summary of escalated cases
        """
        escalate_mask, details = self.process_predictions(
            y_proba, uncertainty, return_details=True
        )
        
        # Filter escalated cases
        escalated = details[details['escalate'] == True].copy()
        
        # Add feature information if provided
        if X is not None and feature_names is not None and len(escalated) > 0:
            escalated_indices = escalated['sample_idx'].values
            X_escalated = X[escalated_indices]
            
            for i, fname in enumerate(feature_names[:5]):  # Top 5 features
                escalated[f'feature_{fname}'] = X_escalated[:, i]
        
        return escalated.sort_values('uncertainty', ascending=False)


def simulate_human_review(
    y_true: np.ndarray,
    escalate_mask: np.ndarray,
    human_accuracy: float = 0.95
) -> np.ndarray:
    """
    Simulate human expert review for escalated cases
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True labels
    escalate_mask : array-like, shape (n_samples,)
        Boolean mask of escalated samples
    human_accuracy : float, default=0.95
        Assumed accuracy of human experts
        
    Returns:
    --------
    final_predictions : array-like, shape (n_samples,)
        Final predictions after human review
    """
    # Simulate human decisions (not perfect)
    n_escalated = np.sum(escalate_mask)
    if n_escalated > 0:
        human_predictions = y_true[escalate_mask].copy()
        
        # Add some human errors
        n_errors = int(n_escalated * (1 - human_accuracy))
        error_indices = np.random.choice(n_escalated, n_errors, replace=False)
        human_predictions[error_indices] = 1 - human_predictions[error_indices]
        
        return human_predictions
    else:
        return np.array([])


def analyze_escalation_patterns(
    details: pd.DataFrame,
    top_n: int = 10
) -> Dict[str, any]:
    """
    Analyze patterns in escalated cases
    
    Parameters:
    -----------
    details : pd.DataFrame
        Detailed escalation information
    top_n : int, default=10
        Number of top patterns to show
        
    Returns:
    --------
    patterns : dict
        Dictionary with pattern analysis
    """
    escalated = details[details['escalate'] == True]
    
    if len(escalated) == 0:
        return {'message': 'No escalations found'}
    
    patterns = {
        'total_escalated': len(escalated),
        'escalation_rate': len(escalated) / len(details),
        'avg_uncertainty': escalated['uncertainty'].mean(),
        'avg_confidence': escalated['confidence'].mean(),
        'avg_probability': escalated['probability'].mean(),
        'top_reasons': escalated['reason'].value_counts().head(top_n),
        'uncertainty_distribution': {
            'min': escalated['uncertainty'].min(),
            'q25': escalated['uncertainty'].quantile(0.25),
            'median': escalated['uncertainty'].median(),
            'q75': escalated['uncertainty'].quantile(0.75),
            'max': escalated['uncertainty'].max()
        }
    }
    
    return patterns


if __name__ == "__main__":
    print("Human Escalation System Module")
    print("This module provides:")
    print("  - EscalationSystem: Intelligent escalation decision-making")
    print("  - Cost-benefit optimization")
    print("  - Comprehensive evaluation metrics")
    print("  - Escalation pattern analysis")
