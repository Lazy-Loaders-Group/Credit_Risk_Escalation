"""
Unit tests for decision logic in Credit Risk Escalation System

Tests the core decision-making logic including:
- Approve decisions
- Reject decisions  
- Escalation decisions
- Edge cases

Run with: python -m pytest tests/test_decision_logic.py -v

Author: Credit Risk ML Team
Date: November 22, 2025
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.escalation_system import EscalationSystem


class TestDecisionLogic:
    """Test decision logic for loan approval/rejection/escalation"""
    
    @pytest.fixture
    def escalation_system(self):
        """Create an escalation system for testing"""
        return EscalationSystem(
            uncertainty_threshold=0.1,
            confidence_threshold=0.7,
            cost_false_positive=1.0,
            cost_false_negative=5.0,
            cost_human_review=0.5
        )
    
    def test_high_confidence_low_risk_approve(self, escalation_system):
        """Test: Low risk + high confidence = APPROVE"""
        uncertainty = 0.05  # Low uncertainty
        confidence = 0.85   # High confidence
        probability = 0.15  # Low default probability
        
        should_escalate, reason = escalation_system.should_escalate(
            uncertainty, confidence, probability
        )
        
        assert not should_escalate, "Should NOT escalate low-risk, high-confidence case"
        assert reason == "Confident prediction"
    
    def test_high_confidence_high_risk_reject(self, escalation_system):
        """Test: High risk + high confidence = REJECT"""
        uncertainty = 0.05  # Low uncertainty
        confidence = 0.85   # High confidence
        probability = 0.85  # High default probability
        
        should_escalate, reason = escalation_system.should_escalate(
            uncertainty, confidence, probability
        )
        
        assert not should_escalate, "Should NOT escalate high-risk, high-confidence case"
        assert reason == "Confident prediction"
    
    def test_high_uncertainty_escalate(self, escalation_system):
        """Test: High uncertainty = ESCALATE"""
        uncertainty = 0.20  # High uncertainty (> 0.1)
        confidence = 0.85   # High confidence (doesn't matter)
        probability = 0.50  # Borderline
        
        should_escalate, reason = escalation_system.should_escalate(
            uncertainty, confidence, probability
        )
        
        assert should_escalate, "Should escalate high-uncertainty case"
        assert "High uncertainty" in reason
    
    def test_low_confidence_escalate(self, escalation_system):
        """Test: Low confidence = ESCALATE"""
        uncertainty = 0.05  # Low uncertainty (doesn't matter)
        confidence = 0.60   # Low confidence (< 0.7)
        probability = 0.50  # Borderline
        
        should_escalate, reason = escalation_system.should_escalate(
            uncertainty, confidence, probability
        )
        
        assert should_escalate, "Should escalate low-confidence case"
        assert "Low confidence" in reason
    
    def test_borderline_probability_escalate(self, escalation_system):
        """Test: Borderline probability (0.4-0.6) = ESCALATE"""
        uncertainty = 0.05  # Low uncertainty
        confidence = 0.85   # High confidence
        probability = 0.50  # Borderline (0.4 <= p <= 0.6)
        
        should_escalate, reason = escalation_system.should_escalate(
            uncertainty, confidence, probability
        )
        
        assert should_escalate, "Should escalate borderline probability"
        assert "Borderline probability" in reason
    
    def test_multiple_escalation_reasons(self, escalation_system):
        """Test: Multiple reasons for escalation"""
        uncertainty = 0.20  # High uncertainty
        confidence = 0.60   # Low confidence
        probability = 0.50  # Borderline
        
        should_escalate, reason = escalation_system.should_escalate(
            uncertainty, confidence, probability
        )
        
        assert should_escalate, "Should escalate with multiple reasons"
        # Check that multiple reasons are present
        assert "High uncertainty" in reason
        assert "Low confidence" in reason
        assert "Borderline probability" in reason
    
    def test_edge_case_exactly_at_threshold(self, escalation_system):
        """Test: Values exactly at threshold"""
        # Exactly at uncertainty threshold
        uncertainty = 0.10
        confidence = 0.85
        probability = 0.30
        
        should_escalate, reason = escalation_system.should_escalate(
            uncertainty, confidence, probability
        )
        
        # At threshold should NOT escalate (> not >=)
        assert not should_escalate, "Exactly at threshold should not escalate"
    
    def test_batch_processing(self, escalation_system):
        """Test: Batch processing of predictions"""
        # Create batch data
        n_samples = 100
        y_proba = np.random.rand(n_samples, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize
        uncertainty = np.random.rand(n_samples) * 0.3  # 0 to 0.3
        
        # Process batch
        escalate_mask = escalation_system.process_predictions(y_proba, uncertainty)
        
        # Check output shape
        assert len(escalate_mask) == n_samples
        assert escalate_mask.dtype == bool
        
        # Check that some are escalated and some are not (with high probability)
        assert np.any(escalate_mask), "Should have some escalations"
        assert not np.all(escalate_mask), "Should have some non-escalations"
    
    def test_cost_calculation(self, escalation_system):
        """Test: Cost calculation for escalations"""
        # Create test data
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        escalate_mask = np.random.rand(n_samples) > 0.7  # 30% escalation
        
        # Calculate costs
        costs = escalation_system.calculate_costs(y_true, y_pred, escalate_mask)
        
        # Check that all expected keys are present
        expected_keys = [
            'false_positive_cost', 'false_negative_cost', 'escalation_cost',
            'total_cost', 'baseline_cost', 'cost_savings', 'n_escalated',
            'n_false_positives', 'n_false_negatives'
        ]
        
        for key in expected_keys:
            assert key in costs, f"Missing key: {key}"
        
        # Check that costs are non-negative
        assert costs['total_cost'] >= 0
        assert costs['escalation_cost'] >= 0
        assert costs['n_escalated'] == np.sum(escalate_mask)
    
    def test_zero_escalations(self, escalation_system):
        """Test: Handle case with zero escalations"""
        # Perfect predictions, low uncertainty
        y_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
        uncertainty = np.array([0.01, 0.01, 0.01])
        
        escalate_mask = escalation_system.process_predictions(y_proba, uncertainty)
        
        # Should have no escalations
        assert np.sum(escalate_mask) == 0, "Should have zero escalations"
    
    def test_all_escalations(self, escalation_system):
        """Test: Handle case with all escalations"""
        # High uncertainty for all
        y_proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        uncertainty = np.array([0.25, 0.25, 0.25])
        
        escalate_mask = escalation_system.process_predictions(y_proba, uncertainty)
        
        # Should escalate all
        assert np.sum(escalate_mask) == len(y_proba), "Should escalate all cases"


class TestDecisionThresholds:
    """Test configurable decision thresholds"""
    
    def test_custom_uncertainty_threshold(self):
        """Test: Custom uncertainty threshold"""
        system = EscalationSystem(uncertainty_threshold=0.2)
        
        # Should not escalate below threshold
        should_escalate, _ = system.should_escalate(
            uncertainty=0.15, confidence=0.8, probability=0.3
        )
        assert not should_escalate
        
        # Should escalate above threshold
        should_escalate, _ = system.should_escalate(
            uncertainty=0.25, confidence=0.8, probability=0.3
        )
        assert should_escalate
    
    def test_custom_confidence_threshold(self):
        """Test: Custom confidence threshold"""
        system = EscalationSystem(confidence_threshold=0.8)
        
        # Should escalate below threshold
        should_escalate, _ = system.should_escalate(
            uncertainty=0.05, confidence=0.75, probability=0.3
        )
        assert should_escalate
        
        # Should not escalate above threshold
        should_escalate, _ = system.should_escalate(
            uncertainty=0.05, confidence=0.85, probability=0.3
        )
        assert not should_escalate
    
    def test_cost_parameters(self):
        """Test: Different cost parameters"""
        system = EscalationSystem(
            cost_false_positive=2.0,
            cost_false_negative=10.0,
            cost_human_review=1.0
        )
        
        # Check that costs are set correctly
        assert system.cost_false_positive == 2.0
        assert system.cost_false_negative == 10.0
        assert system.cost_human_review == 1.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
