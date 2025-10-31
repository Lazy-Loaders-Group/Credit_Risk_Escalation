"""
Uncertainty Quantification Module for Credit Risk Assessment

This module implements various uncertainty quantification methods:
- Bootstrap Ensemble (Primary Method)
- MC Dropout (Optional Advanced Method)
- Temperature Scaling for Calibration
- Uncertainty Metrics and Analysis

Author: Credit Risk ML Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


class BootstrapEnsemble:
    """
    Bootstrap Ensemble for Uncertainty Quantification
    
    Creates multiple models trained on bootstrapped samples to estimate
    prediction uncertainty through ensemble variance.
    
    Parameters:
    -----------
    base_model : sklearn estimator
        Base model to use for ensemble
    n_estimators : int, default=30
        Number of bootstrap models
    bootstrap_size : float, default=0.8
        Fraction of data to use for each bootstrap sample
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all processors)
    """
    
    def __init__(
        self,
        base_model: BaseEstimator,
        n_estimators: int = 30,
        bootstrap_size: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.bootstrap_size = bootstrap_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = []
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BootstrapEnsemble':
        """
        Fit bootstrap ensemble
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : BootstrapEnsemble
        """
        self.models = []
        n_samples = int(len(X) * self.bootstrap_size)
        
        print(f"Training {self.n_estimators} bootstrap models...")
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = resample(
                X, y,
                n_samples=n_samples,
                random_state=self.random_state + i,
                stratify=y
            )
            
            # Clone and train model
            model = clone(self.base_model)
            model.fit(X_boot, y_boot)
            self.models.append(model)
            
            if (i + 1) % 10 == 0:
                print(f"  Trained {i + 1}/{self.n_estimators} models")
        
        self.is_fitted = True
        print(f"✅ Bootstrap ensemble trained with {self.n_estimators} models")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get mean predicted probabilities
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Prediction data
            
        Returns:
        --------
        proba_mean : array-like, shape (n_samples, n_classes)
            Mean predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from all models
        all_proba = np.array([
            model.predict_proba(X) for model in self.models
        ])
        
        # Return mean
        return np.mean(all_proba, axis=0)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Prediction data
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Prediction data
            
        Returns:
        --------
        proba_mean : array-like, shape (n_samples, n_classes)
            Mean predicted probabilities
        uncertainty : array-like, shape (n_samples,)
            Uncertainty scores (higher = more uncertain)
        all_proba : array-like, shape (n_estimators, n_samples, n_classes)
            All individual model predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from all models
        all_proba = np.array([
            model.predict_proba(X) for model in self.models
        ])
        
        # Calculate mean and uncertainty
        proba_mean = np.mean(all_proba, axis=0)
        
        # Uncertainty: standard deviation of positive class probability
        uncertainty = np.std(all_proba[:, :, 1], axis=0)
        
        return proba_mean, uncertainty, all_proba
    
    def get_prediction_intervals(
        self,
        X: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction intervals
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Prediction data
        confidence : float, default=0.95
            Confidence level for intervals
            
        Returns:
        --------
        proba_mean : array-like, shape (n_samples,)
            Mean predicted probability (positive class)
        lower_bound : array-like, shape (n_samples,)
            Lower confidence bound
        upper_bound : array-like, shape (n_samples,)
            Upper confidence bound
        """
        _, _, all_proba = self.predict_with_uncertainty(X)
        
        # Get positive class probabilities
        proba_pos = all_proba[:, :, 1]
        
        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100
        
        proba_mean = np.mean(proba_pos, axis=0)
        lower_bound = np.percentile(proba_pos, lower_percentile, axis=0)
        upper_bound = np.percentile(proba_pos, upper_percentile, axis=0)
        
        return proba_mean, lower_bound, upper_bound


class UncertaintyMetrics:
    """
    Calculate and analyze uncertainty metrics
    """
    
    @staticmethod
    def entropy(proba: np.ndarray) -> np.ndarray:
        """
        Calculate prediction entropy (measure of uncertainty)
        
        Parameters:
        -----------
        proba : array-like, shape (n_samples, n_classes)
            Predicted probabilities
            
        Returns:
        --------
        entropy : array-like, shape (n_samples,)
            Entropy values (higher = more uncertain)
        """
        # Avoid log(0)
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        return -np.sum(proba * np.log2(proba), axis=1)
    
    @staticmethod
    def confidence(proba: np.ndarray) -> np.ndarray:
        """
        Calculate prediction confidence (1 - uncertainty)
        
        Parameters:
        -----------
        proba : array-like, shape (n_samples, n_classes)
            Predicted probabilities
            
        Returns:
        --------
        confidence : array-like, shape (n_samples,)
            Confidence scores (higher = more confident)
        """
        return np.max(proba, axis=1)
    
    @staticmethod
    def margin(proba: np.ndarray) -> np.ndarray:
        """
        Calculate margin between top two probabilities
        
        Parameters:
        -----------
        proba : array-like, shape (n_samples, n_classes)
            Predicted probabilities
            
        Returns:
        --------
        margin : array-like, shape (n_samples,)
            Margin scores (lower = more uncertain)
        """
        sorted_proba = np.sort(proba, axis=1)
        return sorted_proba[:, -1] - sorted_proba[:, -2]
    
    @staticmethod
    def calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate Expected Calibration Error (ECE)
        
        Parameters:
        -----------
        y_true : array-like, shape (n_samples,)
            True labels
        y_proba : array-like, shape (n_samples,)
            Predicted probabilities
        n_bins : int, default=10
            Number of bins for calibration
            
        Returns:
        --------
        ece : float
            Expected Calibration Error
        bin_accuracy : array-like, shape (n_bins,)
            Accuracy per bin
        bin_confidence : array-like, shape (n_bins,)
            Average confidence per bin
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        bin_accuracy = np.zeros(n_bins)
        bin_confidence = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_proba[mask])
                bin_accuracy[i] = bin_acc
                bin_confidence[i] = bin_conf
                
                # Weighted contribution to ECE
                ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(y_true)
        
        return ece, bin_accuracy, bin_confidence
    
    @staticmethod
    def uncertainty_vs_error_correlation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze correlation between uncertainty and prediction errors
        
        Parameters:
        -----------
        y_true : array-like, shape (n_samples,)
            True labels
        y_pred : array-like, shape (n_samples,)
            Predicted labels
        uncertainty : array-like, shape (n_samples,)
            Uncertainty scores
            
        Returns:
        --------
        metrics : dict
            Dictionary with correlation metrics
        """
        errors = (y_true != y_pred).astype(int)
        
        # Pearson correlation
        correlation = np.corrcoef(uncertainty, errors)[0, 1]
        
        # Average uncertainty for correct vs incorrect predictions
        avg_unc_correct = np.mean(uncertainty[errors == 0])
        avg_unc_incorrect = np.mean(uncertainty[errors == 1])
        
        # Uncertainty ratio
        unc_ratio = avg_unc_incorrect / (avg_unc_correct + 1e-10)
        
        return {
            'correlation': correlation,
            'avg_uncertainty_correct': avg_unc_correct,
            'avg_uncertainty_incorrect': avg_unc_incorrect,
            'uncertainty_ratio': unc_ratio
        }


class TemperatureScaling:
    """
    Temperature Scaling for model calibration
    
    Simple post-hoc calibration method that learns a single temperature
    parameter to calibrate predicted probabilities.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to calibrate
    """
    
    def __init__(self, model: BaseEstimator):
        self.model = model
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TemperatureScaling':
        """
        Learn optimal temperature on validation set
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Validation data
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        self : TemperatureScaling
        """
        from scipy.optimize import minimize
        
        # Get logits (pre-softmax scores)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            # Convert probabilities back to logits
            logits = np.log(proba + 1e-10)
        else:
            raise ValueError("Model must have predict_proba method")
        
        def nll_loss(temp):
            """Negative log likelihood loss"""
            scaled_proba = self._apply_temperature(logits, temp)
            # Cross-entropy loss
            loss = -np.mean(
                y * np.log(scaled_proba[:, 1] + 1e-10) +
                (1 - y) * np.log(scaled_proba[:, 0] + 1e-10)
            )
            return loss
        
        # Optimize temperature
        result = minimize(nll_loss, x0=[1.0], bounds=[(0.1, 10.0)])
        self.temperature = result.x[0]
        self.is_fitted = True
        
        print(f"✅ Optimal temperature: {self.temperature:.4f}")
        return self
    
    def _apply_temperature(
        self,
        logits: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling to logits"""
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Prediction data
            
        Returns:
        --------
        proba_calibrated : array-like, shape (n_samples, n_classes)
            Calibrated predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Temperature scaling must be fitted first")
        
        proba = self.model.predict_proba(X)
        logits = np.log(proba + 1e-10)
        return self._apply_temperature(logits, self.temperature)


def analyze_uncertainty_quality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    uncertainty: np.ndarray,
    title: str = "Uncertainty Analysis"
) -> pd.DataFrame:
    """
    Comprehensive uncertainty quality analysis
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True labels
    y_pred : array-like, shape (n_samples,)
        Predicted labels
    y_proba : array-like, shape (n_samples,)
        Predicted probabilities (positive class)
    uncertainty : array-like, shape (n_samples,)
        Uncertainty scores
    title : str, default="Uncertainty Analysis"
        Title for analysis
        
    Returns:
    --------
    results : pd.DataFrame
        Comprehensive analysis results
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")
    
    # Basic statistics
    print("Uncertainty Statistics:")
    print(f"  Mean: {np.mean(uncertainty):.4f}")
    print(f"  Std:  {np.std(uncertainty):.4f}")
    print(f"  Min:  {np.min(uncertainty):.4f}")
    print(f"  Max:  {np.max(uncertainty):.4f}")
    
    # Correlation with errors
    metrics = UncertaintyMetrics()
    corr_results = metrics.uncertainty_vs_error_correlation(y_true, y_pred, uncertainty)
    
    print(f"\nUncertainty vs Error Correlation:")
    print(f"  Correlation: {corr_results['correlation']:.4f}")
    print(f"  Avg Uncertainty (Correct):   {corr_results['avg_uncertainty_correct']:.4f}")
    print(f"  Avg Uncertainty (Incorrect): {corr_results['avg_uncertainty_incorrect']:.4f}")
    print(f"  Ratio (Incorrect/Correct):   {corr_results['uncertainty_ratio']:.4f}")
    
    # Calibration error
    ece, _, _ = metrics.calibration_error(y_true, y_proba, n_bins=10)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    
    # Quality assessment
    print(f"\n{'='*60}")
    if corr_results['correlation'] > 0.3 and corr_results['uncertainty_ratio'] > 1.5:
        print("✅ EXCELLENT: Uncertainty strongly correlates with errors")
    elif corr_results['correlation'] > 0.15:
        print("✅ GOOD: Uncertainty moderately correlates with errors")
    else:
        print("⚠️  WARNING: Weak uncertainty-error correlation")
    print(f"{'='*60}\n")
    
    # Create summary dataframe
    results = pd.DataFrame({
        'Metric': [
            'Mean Uncertainty',
            'Std Uncertainty',
            'Uncertainty-Error Correlation',
            'Avg Unc (Correct)',
            'Avg Unc (Incorrect)',
            'Uncertainty Ratio',
            'Expected Calibration Error'
        ],
        'Value': [
            np.mean(uncertainty),
            np.std(uncertainty),
            corr_results['correlation'],
            corr_results['avg_uncertainty_correct'],
            corr_results['avg_uncertainty_incorrect'],
            corr_results['uncertainty_ratio'],
            ece
        ]
    })
    
    return results


if __name__ == "__main__":
    print("Uncertainty Quantification Module")
    print("This module provides:")
    print("  - BootstrapEnsemble: Main uncertainty quantification method")
    print("  - UncertaintyMetrics: Metrics for analyzing uncertainty")
    print("  - TemperatureScaling: Post-hoc calibration")
    print("  - analyze_uncertainty_quality: Comprehensive analysis tool")
