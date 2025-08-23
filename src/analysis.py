"""
Statistical analysis for n parameter experiments.
"""

import numpy as np
from scipy import stats
from scipy.stats import kstest, ks_2samp, mannwhitneyu, chi2_contingency
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class IndependenceAnalyzer:
    """Analyze independence and distribution of LLM outputs."""
    
    def analyze_position_effects(
        self, 
        batches: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """Analyze if position within batch affects output."""
        batch_size = len(batches[0]) if batches else 0
        position_values = [[] for _ in range(batch_size)]
        
        for batch in batches:
            for i, value in enumerate(batch):
                if value is not None and i < batch_size:
                    position_values[i].append(value)
        
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
        
        return position_stats
    
    def test_position_independence(
        self, 
        batches: List[List[int]]
    ) -> Tuple[float, float]:
        """Test if position affects distribution using chi-square test."""
        # Create contingency table: positions vs value bins
        batch_size = len(batches[0]) if batches else 0
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
    
    def analyze_variance_components(
        self, 
        batches: List[List[int]]
    ) -> Dict[str, float]:
        """Decompose variance into within-batch and between-batch components."""
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
                'ratio': 0.0
            }
        
        overall_variance = np.var(all_values)
        mean_within_batch_var = np.mean(within_batch_vars) if within_batch_vars else 0
        between_batch_var = np.var(batch_means) if batch_means else 0
        
        ratio = mean_within_batch_var / between_batch_var if between_batch_var > 0 else np.inf
        
        return {
            'overall': overall_variance,
            'within_batch': mean_within_batch_var,
            'between_batch': between_batch_var,
            'ratio': ratio
        }
    
    def calculate_autocorrelation(
        self, 
        values: List[int], 
        nlags: int = 10
    ) -> np.ndarray:
        """Calculate autocorrelation function."""
        if len(values) < nlags + 1:
            return np.zeros(nlags + 1)
        
        from statsmodels.tsa.stattools import acf
        return acf(values, nlags=nlags)
    
    def ljung_box_test(
        self, 
        values: List[int], 
        lags: int = 10
    ) -> Tuple[float, float]:
        """Ljung-Box test for independence."""
        if len(values) < lags + 1:
            return 0.0, 1.0
        
        result = acorr_ljungbox(values, lags=lags, return_df=False)
        # Return the test statistic and p-value for the last lag
        return result[0][-1], result[1][-1]
    
    def compare_distributions(
        self,
        sample1: List[int],
        sample2: List[int]
    ) -> Dict[str, Any]:
        """Compare two samples using multiple statistical tests."""
        results = {}
        
        # Remove None values
        s1 = [v for v in sample1 if v is not None]
        s2 = [v for v in sample2 if v is not None]
        
        if not s1 or not s2:
            return {
                'ks_statistic': 0.0,
                'ks_pvalue': 1.0,
                'mw_statistic': 0.0,
                'mw_pvalue': 1.0,
                'mean_diff': 0.0,
                'conclusion': 'Insufficient data'
            }
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(s1, s2)
        
        # Mann-Whitney U test
        mw_stat, mw_pvalue = mannwhitneyu(s1, s2, alternative='two-sided')
        
        # Mean difference
        mean_diff = np.mean(s1) - np.mean(s2)
        
        # Conclusion
        if ks_pvalue < 0.05 and mw_pvalue < 0.05:
            conclusion = "Distributions are significantly different"
        elif ks_pvalue < 0.05 or mw_pvalue < 0.05:
            conclusion = "Mixed evidence for difference"
        else:
            conclusion = "Distributions are statistically equivalent"
        
        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mw_statistic': mw_stat,
            'mw_pvalue': mw_pvalue,
            'mean_diff': mean_diff,
            'std_diff': np.std(s1) - np.std(s2),
            'conclusion': conclusion
        }
    
    def calculate_icc(
        self,
        batches: List[List[int]]
    ) -> float:
        """Calculate Intraclass Correlation Coefficient (ICC)."""
        # Flatten data for ICC calculation
        data = []
        groups = []
        
        for i, batch in enumerate(batches):
            for value in batch:
                if value is not None:
                    data.append(value)
                    groups.append(i)
        
        if not data:
            return 0.0
        
        # Simple ICC calculation using variance components
        df = pd.DataFrame({'value': data, 'group': groups})
        
        # Between-group variance
        group_means = df.groupby('group')['value'].mean()
        grand_mean = df['value'].mean()
        between_var = np.var(group_means) * len(batches[0])
        
        # Within-group variance
        within_var = df.groupby('group')['value'].var().mean()
        
        # ICC
        if between_var + within_var == 0:
            return 0.0
        
        icc = between_var / (between_var + within_var)
        return icc
    
    def create_visualization(
        self,
        n_param_data: Dict[str, Any],
        separate_data: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Create comprehensive visualization of results."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        n_numbers = n_param_data.get('numbers', [])
        s_numbers = separate_data.get('numbers', [])
        
        # 1. Distribution comparison
        ax = axes[0, 0]
        if n_numbers and s_numbers:
            ax.hist(n_numbers, bins=20, alpha=0.5, label='n parameter', color='blue')
            ax.hist(s_numbers, bins=20, alpha=0.5, label='separate calls', color='red')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution Comparison')
            ax.legend()
        
        # 2. Position effects (n parameter)
        ax = axes[0, 1]
        n_batch_numbers = n_param_data.get('batch_numbers', [])
        if n_batch_numbers:
            position_data = [[] for _ in range(len(n_batch_numbers[0]))]
            for batch in n_batch_numbers:
                for i, val in enumerate(batch):
                    if val is not None:
                        position_data[i].append(val)
            
            positions = []
            values = []
            for i, pos_vals in enumerate(position_data):
                positions.extend([i] * len(pos_vals))
                values.extend(pos_vals)
            
            ax.boxplot([pos_vals for pos_vals in position_data if pos_vals])
            ax.set_xlabel('Position in Batch')
            ax.set_ylabel('Value')
            ax.set_title('Position Effects (n parameter)')
        
        # 3. Position effects (separate calls)
        ax = axes[0, 2]
        s_batch_numbers = separate_data.get('batch_numbers', [])
        if s_batch_numbers:
            position_data = [[] for _ in range(len(s_batch_numbers[0]))]
            for batch in s_batch_numbers:
                for i, val in enumerate(batch):
                    if val is not None:
                        position_data[i].append(val)
            
            ax.boxplot([pos_vals for pos_vals in position_data if pos_vals])
            ax.set_xlabel('Position in Batch')
            ax.set_ylabel('Value')
            ax.set_title('Position Effects (separate calls)')
        
        # 4. Q-Q plot (n parameter)
        ax = axes[1, 0]
        if n_numbers:
            stats.probplot(n_numbers, dist="uniform", plot=ax)
            ax.set_title('Q-Q Plot: n parameter vs Uniform')
        
        # 5. Q-Q plot (separate calls)
        ax = axes[1, 1]
        if s_numbers:
            stats.probplot(s_numbers, dist="uniform", plot=ax)
            ax.set_title('Q-Q Plot: separate calls vs Uniform')
        
        # 6. Autocorrelation (n parameter)
        ax = axes[1, 2]
        if n_numbers and len(n_numbers) > 10:
            acf_values = self.calculate_autocorrelation(n_numbers, nlags=min(20, len(n_numbers)//4))
            ax.stem(range(len(acf_values)), acf_values)
            ax.axhline(y=0, linestyle='--', color='gray')
            ax.axhline(y=1.96/np.sqrt(len(n_numbers)), linestyle='--', color='r', alpha=0.5)
            ax.axhline(y=-1.96/np.sqrt(len(n_numbers)), linestyle='--', color='r', alpha=0.5)
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            ax.set_title('Autocorrelation (n parameter)')
        
        # 7. Batch means over time (n parameter)
        ax = axes[2, 0]
        n_batch_means = []
        for batch in n_batch_numbers:
            valid = [v for v in batch if v is not None]
            if valid:
                n_batch_means.append(np.mean(valid))
        
        if n_batch_means:
            ax.plot(n_batch_means, 'b-', marker='o', label='n parameter')
            ax.set_xlabel('Batch Number')
            ax.set_ylabel('Mean Value')
            ax.set_title('Batch Means Over Time')
            ax.legend()
        
        # 8. Batch means over time (separate calls)
        ax = axes[2, 1]
        s_batch_means = []
        for batch in s_batch_numbers:
            valid = [v for v in batch if v is not None]
            if valid:
                s_batch_means.append(np.mean(valid))
        
        if s_batch_means:
            ax.plot(s_batch_means, 'r-', marker='s', label='separate calls')
            ax.set_xlabel('Batch Number')
            ax.set_ylabel('Mean Value')
            ax.set_title('Batch Means Over Time')
            ax.legend()
        
        # 9. Variance comparison
        ax = axes[2, 2]
        if n_batch_numbers and s_batch_numbers:
            n_vars = self.analyze_variance_components(n_batch_numbers)
            s_vars = self.analyze_variance_components(s_batch_numbers)
            
            categories = ['Overall', 'Within-Batch', 'Between-Batch']
            n_values = [n_vars['overall'], n_vars['within_batch'], n_vars['between_batch']]
            s_values = [s_vars['overall'], s_vars['within_batch'], s_vars['between_batch']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, n_values, width, label='n parameter', color='blue')
            ax.bar(x + width/2, s_values, width, label='separate calls', color='red')
            ax.set_xlabel('Variance Type')
            ax.set_ylabel('Variance')
            ax.set_title('Variance Decomposition')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig