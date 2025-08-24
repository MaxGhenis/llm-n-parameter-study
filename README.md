# Statistical Equivalence of the n Parameter in LLM APIs: A Research Framework

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://maxghenis.github.io/llm-n-parameter-study)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

## ⚠️ PROJECT STATUS: FRAMEWORK ONLY - NO REAL DATA YET

**This repository contains a research framework and methodology for testing the n parameter hypothesis. No actual experiments have been conducted yet. All data in `data/sample_experiments/` is SYNTHETIC and for demonstration purposes only.**

## Proposed Research Question

This framework will investigate whether using the `n` parameter in Large Language Model (LLM) APIs produces statistically equivalent output distributions compared to making multiple separate API calls. 

## Planned Experiments

Once API keys are configured and experiments are run, we will test:
- OpenAI's `n` parameter (GPT-4o-mini, GPT-4)
- Google's `candidateCount` parameter (Gemini 1.5)
- Statistical independence within batches
- Performance and cost implications

## Current Status

- ✅ **Framework**: Complete statistical analysis pipeline
- ✅ **Verification System**: Cryptographic proof of real API calls
- ❌ **Real Data**: Not collected yet
- ❌ **Findings**: No empirical results to report

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

## Running Real Experiments

### ⚠️ WARNING: Real API Calls Cost Money!

To run actual experiments with real data:

```bash
# Set up API keys
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"

# Run small test (costs ~$0.05)
uv run python scripts/run_real_experiments.py \
  --provider openai \
  --n 5 \
  --batches 10 \
  --separate-calls 50

# Run full experiment (costs ~$1-2)
uv run python scripts/run_real_experiments.py \
  --provider both \
  --n 5 \
  --batches 100 \
  --separate-calls 500
```

### Verification System

All real API calls are cryptographically verified:
- Request hashes prove when calls were made
- Response metadata confirms authenticity
- Latency measurements detect fake data
- API headers validate genuine responses

## Quick Start (Framework Testing Only)

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