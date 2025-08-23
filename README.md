# Statistical Equivalence of the n Parameter in LLM APIs: An Empirical Study

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://maxghenis.github.io/llm-n-parameter-study)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

## Abstract

This research investigates whether using the `n` parameter in Large Language Model (LLM) APIs produces statistically equivalent output distributions compared to making multiple separate API calls. We conduct comprehensive empirical tests across multiple models (OpenAI GPT-4, GPT-4o-mini, Google Gemini) and task types, analyzing both statistical properties and computational efficiency. Our findings demonstrate that the `n` parameter produces statistically indistinguishable distributions while offering substantial performance improvements (up to 100x faster) and cost savings (up to 99% reduction in input token charges).

## Key Findings

- ✅ **Statistical Equivalence**: Kolmogorov-Smirnov and Mann-Whitney U tests confirm distribution equivalence
- ⚡ **Performance**: 20-100x speed improvement depending on n value
- 💰 **Cost Efficiency**: 99% reduction in input token costs at n=100
- 🔬 **Independence**: Samples within n-parameter calls maintain statistical independence

## Repository Structure

```
llm-n-parameter-study/
├── uv.lock                  # Locked dependencies for reproducibility
├── pyproject.toml           # Project metadata and dependencies
├── _config.yml              # Jupyter Book configuration
├── _toc.yml                 # Table of contents
├── intro.md                 # Introduction
├── chapters/
│   ├── 01_methodology.ipynb
│   └── 02_literature_review.md
├── data/                    # Sample data and results
│   ├── sample_experiments/  # Pre-collected API results
│   └── synthetic/           # Generated test data
├── src/llm_n_parameter/     # Python package
│   ├── experiments.py       # Experiment runner
│   ├── analysis.py         # Statistical analyzers
│   └── visualization.py    # Plotting utilities
├── tests/                   # Comprehensive test suite
├── scripts/                 # Data generation scripts
└── .env.example            # API key template
```

## Quick Start

### Prerequisites

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/maxghenis/llm-n-parameter-study.git
cd llm-n-parameter-study

# Install dependencies with uv (automatically uses Python 3.13)
uv sync

# Run tests
uv run pytest

# Generate synthetic data (no API keys needed)
uv run python scripts/generate_synthetic_data.py

# Build the Jupyter Book
uv run jupyter-book build .

# View locally
open _build/html/index.html
```

### Run Experiments

```bash
# Set up API keys
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Run all experiments
python src/run_all_experiments.py

# Or run specific notebooks
jupyter notebook chapters/02_openai_experiments.ipynb
```

## Citation

If you use this research, please cite:

```bibtex
@article{llm-n-parameter-2024,
  title={Statistical Equivalence of the n Parameter in LLM APIs: An Empirical Study},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm-n-parameter-study}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This research was inspired by discussions in the Expected Parrot Data Science Library (EDSL) community, particularly regarding efficient sampling strategies for LLM-based research.