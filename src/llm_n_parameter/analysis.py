"""
Statistical analysis for n parameter experiments.
Split into specialized analyzers for different aspects.
"""

import numpy as np
from scipy import stats
from scipy.stats import kstest, ks_2samp, mannwhitneyu, chi2_contingency
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


class DistributionAnalyzer:
    """Analyze and compare distributions."""
    
    def compare_distributions(
        self,
        sample1: List[float],
        sample2: List[float]
    ) -> Dict[str, Any]:
        """
        Compare two samples using multiple statistical tests.
        
        Args:
            sample1: First sample (e.g., n parameter results)
            sample2: Second sample (e.g., separate calls results)
            
        Returns:
            Dictionary with test results and conclusions
        """
        # Remove None values
        s1 = [v for v in sample1 if v is not None]
        s2 = [v for v in sample2 if v is not None]
        
        if not s1 or not s2:
            return {
                'error': 'Insufficient data',
                'sample1_size': len(s1),
                'sample2_size': len(s2)
            }
        
        results = {
            'sample_sizes': {'sample1': len(s1), 'sample2': len(s2)},
            'descriptive_stats': {
                'sample1': {
                    'mean': np.mean(s1),
                    'median': np.median(s1),
                    'std': np.std(s1),
                    'min': np.min(s1),
                    'max': np.max(s1)
                },
                'sample2': {
                    'mean': np.mean(s2),
                    'median': np.median(s2),
                    'std': np.std(s2),
                    'min': np.min(s2),
                    'max': np.max(s2)
                }
            }
        }
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(s1, s2)
        results['ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'significant': ks_pvalue < 0.05,
            'interpretation': 'Different distributions' if ks_pvalue < 0.05 
                            else 'Same distribution'
        }
        
        # Mann-Whitney U test
        mw_stat, mw_pvalue = mannwhitneyu(s1, s2, alternative='two-sided')
        results['mann_whitney'] = {
            'statistic': mw_stat,
            'p_value': mw_pvalue,
            'significant': mw_pvalue < 0.05,
            'interpretation': 'Different medians' if mw_pvalue < 0.05 
                            else 'Same median'
        }
        
        # Anderson-Darling test (if samples are large enough)
        if len(s1) >= 25 and len(s2) >= 25:
            try:
                from scipy.stats import anderson_ksamp
                ad_stat, _, ad_pvalue = anderson_ksamp([s1, s2])
                results['anderson_darling'] = {
                    'statistic': ad_stat,
                    'p_value': ad_pvalue,
                    'significant': ad_pvalue < 0.05
                }
            except:
                pass
        
        # Overall conclusion
        if results['ks_test']['significant'] and results['mann_whitney']['significant']:
            conclusion = "Strong evidence of different distributions"
        elif results['ks_test']['significant'] or results['mann_whitney']['significant']:
            conclusion = "Mixed evidence - possible difference"
        else:
            conclusion = "Distributions are statistically equivalent"
        
        results['conclusion'] = conclusion
        
        return results
    
    def test_uniformity(self, sample: List[float], range_min: float = 1, 
                       range_max: float = 100) -> Dict[str, Any]:
        """
        Test if a sample follows a uniform distribution.
        
        Args:
            sample: Data to test
            range_min: Minimum of expected range
            range_max: Maximum of expected range
            
        Returns:
            Test results
        """
        clean_sample = [v for v in sample if v is not None]
        
        if not clean_sample:
            return {'error': 'No valid data'}
        
        # Normalize to [0, 1]
        normalized = [(v - range_min) / (range_max - range_min) 
                     for v in clean_sample]
        
        # KS test against uniform
        ks_stat, ks_pvalue = kstest(normalized, 'uniform')
        
        # Chi-square test
        observed, bins = np.histogram(clean_sample, bins=10, 
                                     range=(range_min, range_max))
        expected = len(clean_sample) / 10
        chi2_stat, chi2_pvalue = stats.chisquare(observed, 
                                                [expected] * 10)
        
        return {
            'ks_test': {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'is_uniform': ks_pvalue > 0.05
            },
            'chi_square': {
                'statistic': chi2_stat,
                'p_value': chi2_pvalue,
                'is_uniform': chi2_pvalue > 0.05
            },
            'conclusion': 'Appears uniform' if ks_pvalue > 0.05 and chi2_pvalue > 0.05
                        else 'Not uniform'
        }


class IndependenceAnalyzer:
    """Analyze independence and correlation in outputs."""
    
    def analyze_position_effects(
        self, 
        batches: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Analyze if position within batch affects output.
        
        Args:
            batches: List of batches, each containing values
            
        Returns:
            Position statistics and test results
        """
        if not batches or not batches[0]:
            return {'error': 'No data provided'}
        
        batch_size = len(batches[0])
        position_values = [[] for _ in range(batch_size)]
        
        for batch in batches:
            for i, value in enumerate(batch):
                if value is not None and i < batch_size:
                    position_values[i].append(value)
        
        # Calculate statistics per position
        position_stats = []
        for i, values in enumerate(position_values):
            if values:
                position_stats.append({
                    'position': i,
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                })
        
        # Test for position independence using ANOVA
        if all(len(vals) > 0 for vals in position_values):
            f_stat, p_value = stats.f_oneway(*position_values)
            independence_test = {
                'method': 'One-way ANOVA',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': 'Position affects values' if p_value < 0.05 
                                else 'Position independent'
            }
        else:
            independence_test = {'error': 'Insufficient data for ANOVA'}
        
        return {
            'position_statistics': position_stats,
            'independence_test': independence_test,
            'batch_size': batch_size,
            'num_batches': len(batches)
        }
    
    def test_position_independence(
        self, 
        batches: List[List[float]]
    ) -> Tuple[float, float]:
        """
        Chi-square test for position independence.
        
        Returns:
            (chi2_statistic, p_value)
        """
        if not batches or not batches[0]:
            return 0.0, 1.0
        
        # Create contingency table
        all_values = []
        positions = []
        
        for batch in batches:
            for i, value in enumerate(batch):
                if value is not None:
                    all_values.append(value)
                    positions.append(i)
        
        if not all_values:
            return 0.0, 1.0
        
        # Bin the values
        bins = np.percentile(all_values, [0, 25, 50, 75, 100])
        binned_values = np.digitize(all_values, bins[1:-1])
        
        # Create contingency table
        contingency = pd.crosstab(positions, binned_values)
        
        # Chi-square test
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        return chi2, p_value
    
    def calculate_autocorrelation(
        self, 
        values: List[float], 
        nlags: int = 10
    ) -> np.ndarray:
        """Calculate autocorrelation function."""
        clean_values = [v for v in values if v is not None]
        
        if len(clean_values) < nlags + 1:
            return np.zeros(nlags + 1)
        
        from statsmodels.tsa.stattools import acf
        return acf(clean_values, nlags=nlags)
    
    def ljung_box_test(
        self, 
        values: List[float], 
        lags: int = 10
    ) -> Dict[str, Any]:
        """
        Ljung-Box test for independence in sequence.
        
        Returns:
            Test results with interpretation
        """
        clean_values = [v for v in values if v is not None]
        
        if len(clean_values) < lags + 1:
            return {'error': 'Insufficient data for test'}
        
        result = acorr_ljungbox(clean_values, lags=lags, return_df=True)
        
        # Check if any lag shows significant autocorrelation
        significant_lags = result[result['lb_pvalue'] < 0.05]
        
        return {
            'test_results': result.to_dict(),
            'num_significant_lags': len(significant_lags),
            'min_pvalue': result['lb_pvalue'].min(),
            'max_statistic': result['lb_stat'].max(),
            'conclusion': 'Evidence of autocorrelation' if len(significant_lags) > 0
                        else 'No significant autocorrelation detected'
        }
    
    def calculate_icc(self, batches: List[List[float]]) -> Dict[str, float]:
        """
        Calculate Intraclass Correlation Coefficient (ICC).
        
        Args:
            batches: List of batches
            
        Returns:
            ICC value and interpretation
        """
        # Prepare data for ICC calculation
        data = []
        groups = []
        
        for i, batch in enumerate(batches):
            for value in batch:
                if value is not None:
                    data.append(value)
                    groups.append(i)
        
        if not data:
            return {'error': 'No valid data'}
        
        df = pd.DataFrame({'value': data, 'group': groups})
        
        # Calculate variance components
        group_means = df.groupby('group')['value'].mean()
        grand_mean = df['value'].mean()
        
        # Between-group variance
        n_per_group = df.groupby('group').size().mean()
        between_var = np.var(group_means) * n_per_group
        
        # Within-group variance
        within_var = df.groupby('group')['value'].var().mean()
        
        # Calculate ICC
        total_var = between_var + within_var
        if total_var == 0:
            icc = 0.0
        else:
            icc = between_var / total_var
        
        # Calculate design effect
        avg_cluster_size = df.groupby('group').size().mean()
        design_effect = 1 + (avg_cluster_size - 1) * icc
        
        return {
            'icc': icc,
            'between_variance': between_var,
            'within_variance': within_var,
            'design_effect': design_effect,
            'effective_sample_size_ratio': 1 / design_effect if design_effect > 0 else 1,
            'interpretation': self._interpret_icc(icc)
        }
    
    def _interpret_icc(self, icc: float) -> str:
        """Interpret ICC value."""
        if icc < 0.1:
            return "Very low correlation within batches"
        elif icc < 0.3:
            return "Low correlation within batches"
        elif icc < 0.5:
            return "Moderate correlation within batches"
        elif icc < 0.7:
            return "High correlation within batches (like Gallo et al. 2025)"
        else:
            return "Very high correlation within batches"


class VarianceAnalyzer:
    """Analyze variance components and decomposition."""
    
    def analyze_variance_components(
        self, 
        batches: List[List[float]]
    ) -> Dict[str, float]:
        """
        Decompose variance into within-batch and between-batch components.
        
        Args:
            batches: List of batches
            
        Returns:
            Variance components and ratio
        """
        batch_means = []
        within_batch_vars = []
        all_values = []
        
        for batch in batches:
            valid_values = [v for v in batch if v is not None]
            if valid_values:
                batch_means.append(np.mean(valid_values))
                within_batch_vars.append(np.var(valid_values))
                all_values.extend(valid_values)
        
        if not all_values or not batch_means:
            return {
                'overall': 0.0,
                'within_batch': 0.0,
                'between_batch': 0.0,
                'ratio': 0.0,
                'interpretation': 'Insufficient data'
            }
        
        overall_variance = np.var(all_values)
        mean_within_batch_var = np.mean(within_batch_vars) if within_batch_vars else 0
        between_batch_var = np.var(batch_means) if batch_means else 0
        
        ratio = mean_within_batch_var / between_batch_var if between_batch_var > 0 else np.inf
        
        # Interpret the ratio
        if ratio == np.inf:
            interpretation = "No between-batch variance"
        elif ratio < 0.5:
            interpretation = "More variation between batches (suggests batch effects)"
        elif ratio < 1.5:
            interpretation = "Balanced variation (suggests independence)"
        else:
            interpretation = "More variation within batches (suggests position effects)"
        
        return {
            'overall': overall_variance,
            'within_batch': mean_within_batch_var,
            'between_batch': between_batch_var,
            'ratio': ratio,
            'interpretation': interpretation,
            'num_batches': len(batch_means),
            'avg_batch_size': len(all_values) / len(batch_means) if batch_means else 0
        }
    
    def calculate_effective_sample_size(
        self,
        actual_n: int,
        icc: float,
        cluster_size: int
    ) -> Dict[str, Any]:
        """
        Calculate effective sample size accounting for clustering.
        
        Args:
            actual_n: Actual number of samples
            icc: Intraclass correlation coefficient
            cluster_size: Size of each cluster/batch
            
        Returns:
            Effective sample size and related metrics
        """
        design_effect = 1 + (cluster_size - 1) * icc
        effective_n = actual_n / design_effect
        
        # Calculate impact on confidence intervals
        ci_inflation = np.sqrt(design_effect)
        
        # Calculate impact on required sample size
        required_n_adjusted = actual_n * design_effect
        
        return {
            'actual_n': actual_n,
            'effective_n': effective_n,
            'design_effect': design_effect,
            'sample_size_reduction': (1 - effective_n/actual_n) * 100,
            'ci_inflation_factor': ci_inflation,
            'power_loss_percent': (1 - effective_n/actual_n) * 100,
            'required_n_for_same_power': required_n_adjusted,
            'interpretation': self._interpret_design_effect(design_effect)
        }
    
    def _interpret_design_effect(self, deff: float) -> str:
        """Interpret design effect value."""
        if deff < 1.5:
            return "Minimal impact on statistical power"
        elif deff < 3:
            return "Moderate impact - consider adjusting sample size"
        elif deff < 5:
            return "Substantial impact - significant power loss"
        else:
            return "Severe impact - dramatic reduction in effective sample size"