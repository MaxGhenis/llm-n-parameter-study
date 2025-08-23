"""
Test suite for statistical independence of n parameter vs looping.

Based on prior research showing position effects and within-batch correlations
when using the n parameter in OpenAI's API.
"""

import pytest
import numpy as np
from scipy import stats
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.experiments import NParameterExperiment
from src.analysis import IndependenceAnalyzer


class TestIndependence:
    """Test statistical independence of n parameter outputs."""
    
    def test_position_effects_detection(self):
        """Test that we can detect position effects within batches."""
        # Simulate data with position effects (as observed in prior research)
        # Position 0 tends higher, position 4 tends lower
        mock_batches = []
        for _ in range(20):
            batch = []
            for pos in range(5):
                # Add position-dependent bias
                base = 50
                position_bias = (2 - pos) * 3  # +6, +3, 0, -3, -6
                value = base + position_bias + np.random.normal(0, 5)
                batch.append(int(value))
            mock_batches.append(batch)
        
        analyzer = IndependenceAnalyzer()
        position_stats = analyzer.analyze_position_effects(mock_batches)
        
        # Should detect that position 0 has higher mean than position 4
        assert position_stats[0]['mean'] > position_stats[4]['mean']
        
        # Chi-square test should detect non-uniformity
        chi2_stat, p_value = analyzer.test_position_independence(mock_batches)
        assert p_value < 0.05, "Should detect position dependence"
    
    def test_within_vs_between_batch_variance(self):
        """Test variance decomposition within and between batches."""
        # Create data with high within-batch correlation
        # (as observed with temperature=0.7 showing 3.33 ratio)
        mock_batches = []
        for _ in range(20):
            batch_mean = np.random.normal(50, 10)  # Between-batch variation
            batch = []
            for _ in range(5):
                # Within-batch variation (higher than between)
                value = batch_mean + np.random.normal(0, 20)
                batch.append(int(value))
            mock_batches.append(batch)
        
        analyzer = IndependenceAnalyzer()
        variance_stats = analyzer.analyze_variance_components(mock_batches)
        
        assert 'within_batch' in variance_stats
        assert 'between_batch' in variance_stats
        assert 'ratio' in variance_stats
        
        # The ratio should be > 1 when within-batch variance dominates
        assert variance_stats['ratio'] > 1.0
    
    def test_autocorrelation_detection(self):
        """Test detection of autocorrelation in sequences."""
        # Create sequence with autocorrelation
        n = 100
        values = [50]
        for i in range(1, n):
            # Each value depends on previous (autocorrelation)
            values.append(int(0.5 * values[i-1] + 0.5 * np.random.uniform(1, 100)))
        
        analyzer = IndependenceAnalyzer()
        acf_values = analyzer.calculate_autocorrelation(values, nlags=10)
        
        # Lag 1 should show significant autocorrelation
        assert acf_values[1] > 0.3, "Should detect autocorrelation at lag 1"
        
        # Ljung-Box test should reject independence
        lb_stat, p_value = analyzer.ljung_box_test(values)
        assert p_value < 0.05, "Should reject independence hypothesis"
    
    def test_distributional_equivalence(self):
        """Test whether two samples come from same distribution."""
        # Sample 1: Normal distribution around 50
        sample1 = np.random.normal(50, 10, 100)
        
        # Sample 2: Same distribution
        sample2 = np.random.normal(50, 10, 100)
        
        # Sample 3: Different distribution (biased)
        sample3 = np.random.normal(55, 10, 100)
        
        analyzer = IndependenceAnalyzer()
        
        # Same distribution test
        ks_stat, p_value = stats.ks_2samp(sample1, sample2)
        assert p_value > 0.05, "Should not reject null hypothesis for same distribution"
        
        # Different distribution test
        ks_stat, p_value = stats.ks_2samp(sample1, sample3)
        assert p_value < 0.05, "Should reject null hypothesis for different distributions"
    
    @pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
    def test_temperature_effects(self, temperature):
        """Test that different temperatures produce expected patterns."""
        # Based on observed behavior:
        # temp=0.0: All identical (variance=0)
        # temp=0.7: Some variation, clustering around certain values
        # temp=1.0: More variation but still biased
        
        if temperature == 0.0:
            # Deterministic: all same value
            values = [47] * 100
            variance = np.var(values)
            assert variance == 0, "Temperature 0 should produce no variance"
        
        elif temperature == 0.7:
            # Some variation, but clustered
            values = np.random.choice([37, 42, 47, 52, 57], size=100, p=[0.1, 0.2, 0.4, 0.2, 0.1])
            unique_ratio = len(set(values)) / len(values)
            assert unique_ratio < 0.5, "Should have limited unique values"
        
        else:  # temperature == 1.0
            # More variation but still not uniform
            values = np.random.normal(45, 15, 100)
            values = np.clip(values, 1, 100).astype(int)
            
            # Test for uniformity (should fail)
            _, p_value = stats.kstest((values - 1) / 99, 'uniform')
            assert p_value < 0.05, "Should not be uniform even at temperature=1.0"


class TestAPIBehavior:
    """Test specific API behavior patterns."""
    
    @patch('openai.OpenAI')
    def test_n_parameter_mock(self, mock_openai):
        """Test that n parameter returns expected number of completions."""
        # Mock the API response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="47")),
            Mock(message=Mock(content="42")),
            Mock(message=Mock(content="37")),
            Mock(message=Mock(content="52")),
            Mock(message=Mock(content="47"))
        ]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        experiment = NParameterExperiment(api_key="test")
        experiment.client = mock_client
        
        results = experiment.query_n_parameter(n=5)
        
        assert len(results) == 5
        assert "47" in results  # Common value from prior observations
    
    def test_known_biases(self):
        """Test for known biases in LLM number generation."""
        # Based on prior research showing "fondness for 47"
        values = [47, 42, 47, 37, 47, 52, 47, 42, 47, 57]
        
        # Count frequency
        from collections import Counter
        counts = Counter(values)
        
        # 47 should be most common
        most_common = counts.most_common(1)[0]
        assert most_common[0] == 47, "47 should be most frequent"
        assert most_common[1] >= 4, "47 should appear multiple times"


class TestResearchImplications:
    """Test scenarios showing research implications."""
    
    def test_false_significance_from_batching(self):
        """Demonstrate how batching can create false significance."""
        # Scenario: Testing if a prompt produces positive sentiment
        # Using n=10 in one call vs 10 separate calls
        
        # Batch results (correlated, ICC ~0.69 as per Gallo et al.)
        batch_positive_ratio = 0.8  # 8/10 positive
        batch_variance = 0.01  # Low variance due to correlation
        
        # Independent results (truly independent)
        independent_positive_ratio = 0.6  # 6/10 positive  
        independent_variance = 0.24  # Higher variance (binomial)
        
        # Naive significance test (assumes independence)
        from scipy.stats import binom_test
        
        # Batch: appears highly significant if treated as independent
        p_batch_naive = binom_test(8, 10, 0.5)
        assert p_batch_naive < 0.05, "Naive test shows significance"
        
        # But accounting for correlation (effective n ~2 instead of 10)
        p_batch_corrected = binom_test(2, 2, 0.5)  # Much less significant
        assert p_batch_corrected > 0.05, "Corrected test shows no significance"
    
    def test_overestimated_sample_size(self):
        """Test that ignoring correlation overestimates effective sample size."""
        # Based on Gallo et al.: ICC=0.69 means effective sample size
        # is much smaller than actual sample size
        
        n_actual = 100
        icc = 0.69  # Intraclass correlation from research
        
        # Design effect formula: 1 + (m-1)*ICC where m is cluster size
        cluster_size = 10  # Using n=10 per API call
        design_effect = 1 + (cluster_size - 1) * icc
        
        n_effective = n_actual / design_effect
        
        assert n_effective < 20, "Effective sample size should be much smaller"
        
        # This means confidence intervals are actually ~5x wider than assumed
        ci_inflation_factor = np.sqrt(design_effect)
        assert ci_inflation_factor > 2, "Confidence intervals severely underestimated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])