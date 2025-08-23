"""
Visualization tools for n-parameter experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ExperimentVisualizer:
    """Create publication-quality visualizations for experiments."""
    
    def __init__(self, style: str = "scientific"):
        """Initialize visualizer with style settings."""
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Set up matplotlib and seaborn styles."""
        if self.style == "scientific":
            plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid'])
            sns.set_context("paper", font_scale=1.2)
            sns.set_palette("husl")
        elif self.style == "presentation":
            plt.style.use(['seaborn-v0_8-talk', 'seaborn-v0_8-darkgrid'])
            sns.set_context("talk", font_scale=1.4)
            sns.set_palette("bright")
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
        })
    
    def plot_distribution_comparison(
        self,
        n_param_values: List[float],
        separate_values: List[float],
        title: str = "Distribution Comparison: n Parameter vs Separate Calls",
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create distribution comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Overlapping histograms
        ax = axes[0, 0]
        ax.hist(n_param_values, bins=30, alpha=0.5, label='n parameter', 
                color='blue', density=True, edgecolor='black')
        ax.hist(separate_values, bins=30, alpha=0.5, label='separate calls', 
                color='red', density=True, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Probability Density Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. KDE plots
        ax = axes[0, 1]
        if len(n_param_values) > 1 and len(separate_values) > 1:
            pd.Series(n_param_values).plot(kind='kde', ax=ax, label='n parameter', 
                                          color='blue', linewidth=2)
            pd.Series(separate_values).plot(kind='kde', ax=ax, label='separate calls', 
                                           color='red', linewidth=2)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title('Kernel Density Estimation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Box plots
        ax = axes[1, 0]
        box_data = [n_param_values, separate_values]
        bp = ax.boxplot(box_data, labels=['n parameter', 'separate calls'],
                        patch_artist=True, notch=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel('Value')
        ax.set_title('Box Plot Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Q-Q plot
        ax = axes[1, 1]
        if len(n_param_values) > 0 and len(separate_values) > 0:
            # Standardize for comparison
            n_sorted = np.sort(n_param_values)
            s_sorted = np.sort(separate_values)
            
            # Interpolate to same length
            min_len = min(len(n_sorted), len(s_sorted))
            n_quantiles = np.percentile(n_sorted, np.linspace(0, 100, min_len))
            s_quantiles = np.percentile(s_sorted, np.linspace(0, 100, min_len))
            
            ax.scatter(s_quantiles, n_quantiles, alpha=0.6)
            
            # Add diagonal reference line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            
            ax.set_xlabel('Separate Calls Quantiles')
            ax.set_ylabel('n Parameter Quantiles')
            ax.set_title('Q-Q Plot')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig, axes
    
    def plot_position_effects(
        self,
        batches: List[List[float]],
        method_name: str = "n parameter",
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize position effects within batches."""
        if not batches or not batches[0]:
            return None, None
        
        batch_size = len(batches[0])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Organize data by position
        position_data = [[] for _ in range(batch_size)]
        for batch in batches:
            for i, value in enumerate(batch):
                if value is not None and i < batch_size:
                    position_data[i].append(value)
        
        # 1. Box plot by position
        ax = axes[0]
        bp = ax.boxplot(position_data, positions=range(batch_size),
                        patch_artist=True, notch=True)
        
        # Color gradient for positions
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, batch_size))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Position in Batch')
        ax.set_ylabel('Value')
        ax.set_title(f'Distribution by Position ({method_name})')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Mean and std by position
        ax = axes[1]
        means = [np.mean(pos_data) if pos_data else 0 for pos_data in position_data]
        stds = [np.std(pos_data) if pos_data else 0 for pos_data in position_data]
        
        x = range(batch_size)
        ax.errorbar(x, means, yerr=stds, fmt='o-', linewidth=2, 
                   markersize=8, capsize=5, capthick=2)
        ax.set_xlabel('Position in Batch')
        ax.set_ylabel('Mean ± Std Dev')
        ax.set_title('Mean Value by Position')
        ax.grid(True, alpha=0.3)
        
        # 3. Heatmap of values
        ax = axes[2]
        
        # Create matrix for heatmap (batches x positions)
        max_batches = min(20, len(batches))  # Limit for visibility
        matrix = []
        for batch in batches[:max_batches]:
            row = []
            for value in batch:
                row.append(value if value is not None else np.nan)
            matrix.append(row)
        
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('Position in Batch')
        ax.set_ylabel('Batch Number')
        ax.set_title('Value Heatmap')
        plt.colorbar(im, ax=ax, label='Value')
        
        fig.suptitle(f'Position Effects Analysis: {method_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig, axes
    
    def plot_variance_decomposition(
        self,
        n_param_variance: Dict[str, float],
        separate_variance: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot variance decomposition comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        categories = ['Overall', 'Within-Batch', 'Between-Batch']
        
        # 1. Side-by-side bars
        ax = axes[0]
        x = np.arange(len(categories))
        width = 0.35
        
        n_values = [n_param_variance.get(k.lower().replace('-', '_'), 0) 
                   for k in categories]
        s_values = [separate_variance.get(k.lower().replace('-', '_'), 0) 
                   for k in categories]
        
        bars1 = ax.bar(x - width/2, n_values, width, label='n parameter', 
                       color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, s_values, width, label='separate calls', 
                       color='coral', edgecolor='black')
        
        ax.set_xlabel('Variance Component')
        ax.set_ylabel('Variance')
        ax.set_title('Variance Decomposition Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
        
        # 2. Ratio comparison
        ax = axes[1]
        n_ratio = n_param_variance.get('ratio', 0)
        s_ratio = separate_variance.get('ratio', 0)
        
        bars = ax.bar(['n parameter', 'separate calls'], [n_ratio, s_ratio],
                     color=['steelblue', 'coral'], edgecolor='black')
        
        ax.set_ylabel('Within/Between Variance Ratio')
        ax.set_title('Variance Ratio Comparison')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, 
                  label='Equal variance')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
        
        plt.suptitle('Variance Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def create_interactive_comparison(
        self,
        results: Dict[str, Any],
        output_file: str = "interactive_results.html"
    ) -> go.Figure:
        """Create interactive Plotly visualization."""
        n_numbers = results['n_parameter']['numbers']
        s_numbers = results['separate']['numbers']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution Comparison', 'Box Plots',
                          'Cumulative Distribution', 'Scatter Plot'),
            specs=[[{'type': 'histogram'}, {'type': 'box'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 1. Histograms
        fig.add_trace(
            go.Histogram(x=n_numbers, name='n parameter', 
                        opacity=0.7, marker_color='blue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=s_numbers, name='separate calls', 
                        opacity=0.7, marker_color='red'),
            row=1, col=1
        )
        
        # 2. Box plots
        fig.add_trace(
            go.Box(y=n_numbers, name='n parameter', marker_color='blue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=s_numbers, name='separate calls', marker_color='red'),
            row=1, col=2
        )
        
        # 3. ECDF
        n_sorted = np.sort(n_numbers)
        s_sorted = np.sort(s_numbers)
        n_ecdf = np.arange(1, len(n_sorted) + 1) / len(n_sorted)
        s_ecdf = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
        
        fig.add_trace(
            go.Scatter(x=n_sorted, y=n_ecdf, mode='lines', 
                      name='n parameter ECDF', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=s_sorted, y=s_ecdf, mode='lines', 
                      name='separate calls ECDF', line=dict(color='red')),
            row=2, col=1
        )
        
        # 4. Scatter plot (sample correlation)
        min_len = min(len(n_numbers), len(s_numbers))
        fig.add_trace(
            go.Scatter(x=s_numbers[:min_len], y=n_numbers[:min_len],
                      mode='markers', name='Value Comparison',
                      marker=dict(color='purple', size=8, opacity=0.6)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive n Parameter Analysis",
            showlegend=True,
            height=800,
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Method", row=1, col=2)
        fig.update_xaxes(title_text="Value", row=2, col=1)
        fig.update_xaxes(title_text="Separate Calls Value", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
        fig.update_yaxes(title_text="n Parameter Value", row=2, col=2)
        
        # Save to HTML
        fig.write_html(output_file)
        
        return fig