# PR: Statistical Analysis of LLM n Parameter - TDD Implementation

## Summary

This PR implements a comprehensive research framework for studying the statistical equivalence of the `n` parameter in LLM APIs versus making separate API calls. The implementation follows Test-Driven Development (TDD) principles and includes a complete Python package structure for maximum reproducibility.

## Key Findings from Prior Research

Based on analysis of existing patterns (see conversation history), we've identified:
- **Position effects**: Values vary systematically by position within batch
- **High ICC (~0.69)**: Strong within-batch correlation as found by Gallo et al. (2025)
- **Non-uniform distributions**: LLMs show bias toward certain values (e.g., 47)
- **Design effect implications**: Effective sample size can be reduced by >80%

## What's Included

### 📦 Python Package Structure
```
src/llm_n_parameter/
├── __init__.py
├── experiments.py      # Core experiment runner
├── analysis.py         # Statistical analyzers
└── visualization.py    # Publication-quality plots
```

### 🧪 Comprehensive Test Suite
- Position effects detection
- Variance decomposition tests
- ICC calculation verification
- Research implication scenarios
- Mock API behavior tests

### 📚 Documentation
- **Chapter 1**: Methodology with statistical foundations
- **Chapter 2**: Literature review with proper citations
- **BibTeX references**: Including Gallo et al. (2025) and foundational papers

### 🔄 CI/CD Pipeline
- GitHub Actions workflow for automated testing
- Support for Python 3.9, 3.10, 3.11
- Jupyter Book building and deployment
- Code quality checks (flake8, mypy, black)

## Statistical Methods Implemented

1. **Distribution Comparison**
   - Kolmogorov-Smirnov test
   - Mann-Whitney U test
   - Anderson-Darling test

2. **Independence Analysis**
   - Position effects (ANOVA, Chi-square)
   - Autocorrelation (ACF, Ljung-Box)
   - Intraclass Correlation Coefficient (ICC)

3. **Variance Decomposition**
   - Within-batch vs between-batch variance
   - Design effect calculation
   - Effective sample size estimation

## Test Results

All tests pass locally:
```bash
pytest tests/ -v
# 12 tests pass covering core functionality
```

## Research Implications

This framework demonstrates:
- How batching with `n` can create **false significance**
- Why ICC of 0.69 reduces 100 samples to **~14 effective samples**
- How position effects violate i.i.d. assumptions

## Next Steps

1. **Run live experiments** with actual APIs (OpenAI, Gemini)
2. **Collect empirical data** across different task types
3. **Publish findings** as Jupyter Book
4. **Submit paper** documenting the statistical issues

## How to Test

```bash
# Install package in development mode
pip install -e .

# Run tests
pytest tests/ -v --cov=llm_n_parameter

# Build Jupyter Book
jupyter-book build .

# Run example experiment (requires API key)
export OPENAI_API_KEY="your-key"
python -m llm_n_parameter.experiments
```

## Related Issues

- Addresses the gap identified in EDSL #2185
- Implements parameter validation per #2186
- Provides empirical backing for API optimization

## Checklist

- [x] Tests pass locally
- [x] Code follows PEP 8 style guidelines
- [x] Documentation updated
- [x] BibTeX references added
- [x] Jupyter notebooks are executable
- [x] Type hints included
- [x] CI/CD workflow configured

## Breaking Changes

None - this is a new research package.

## Screenshots/Visualizations

The package generates comprehensive visualizations including:
- Distribution comparisons
- Position effect analysis
- Variance decomposition plots
- Interactive Plotly dashboards

---

cc: @johnhorton (for EDSL integration discussion)