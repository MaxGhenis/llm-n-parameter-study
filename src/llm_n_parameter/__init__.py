"""
llm_n_parameter: Statistical analysis of the n parameter in LLM APIs.

This package provides tools to analyze whether using the n parameter in LLM APIs
produces statistically equivalent outputs compared to making separate API calls.
"""

__version__ = "0.1.0"
__author__ = "Max Ghenis"

from .experiments import NParameterExperiment, ExperimentConfig
from .analysis import IndependenceAnalyzer, VarianceAnalyzer, DistributionAnalyzer
from .visualization import ExperimentVisualizer

__all__ = [
    "NParameterExperiment",
    "ExperimentConfig",
    "IndependenceAnalyzer",
    "VarianceAnalyzer",
    "DistributionAnalyzer",
    "ExperimentVisualizer",
]