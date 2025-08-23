# Statistical Equivalence of the n Parameter in LLM APIs: An Empirical Study

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://yourusername.github.io/llm-n-parameter-study)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
├── _config.yml              # Jupyter Book configuration
├── _toc.yml                 # Table of contents
├── intro.md                 # Introduction
├── chapters/
│   ├── 01_methodology.ipynb
│   ├── 02_openai_experiments.ipynb
│   ├── 03_gemini_experiments.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_cost_analysis.ipynb
│   └── 06_conclusions.md
├── data/                    # Experimental data
├── src/                     # Source code for experiments
└── requirements.txt         # Python dependencies
```

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-n-parameter-study.git
cd llm-n-parameter-study

# Install dependencies
pip install -r requirements.txt

# Build the book
jupyter-book build .

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