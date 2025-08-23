# Data Directory Structure

This directory contains sample data and results from the n-parameter experiments. All data is synthetic or anonymized to enable reproduction without API costs.

## Directory Structure

```
data/
├── README.md                    # This file
├── sample_experiments/          # Sample experimental results
│   ├── openai_n_param.json     # Results using n parameter
│   ├── openai_separate.json    # Results using separate calls
│   ├── gemini_n_param.json     # Gemini candidateCount results
│   └── gemini_separate.json    # Gemini separate call results
├── synthetic/                   # Synthetic data for testing
│   ├── random_numbers.csv      # Random number generation results
│   ├── sentiment_analysis.csv  # Sentiment classification results
│   └── creative_text.json      # Creative generation samples
├── statistical_tests/           # Pre-computed statistical test results
│   ├── ks_test_results.json    # Kolmogorov-Smirnov test outputs
│   ├── position_effects.json   # Position effect analysis
│   └── icc_calculations.json   # ICC computation results
└── figures/                     # Generated visualizations
    ├── distribution_comparison.png
    ├── position_effects.png
    └── variance_decomposition.png
```

## Data Formats

### Experimental Results Format
```json
{
  "metadata": {
    "model": "gpt-4o-mini",
    "temperature": 1.0,
    "method": "n_parameter",
    "n_value": 5,
    "num_batches": 100,
    "timestamp": "2024-08-23T12:00:00Z"
  },
  "prompts": [...],
  "responses": [...],
  "timing": {...}
}
```

### Statistical Test Results Format
```json
{
  "test_name": "kolmogorov_smirnov",
  "statistic": 0.123,
  "p_value": 0.045,
  "reject_null": true,
  "parameters": {...}
}
```

## Usage

### Loading Sample Data
```python
import json
import pandas as pd

# Load experimental results
with open('data/sample_experiments/openai_n_param.json') as f:
    n_param_data = json.load(f)

# Load synthetic data
synthetic_df = pd.read_csv('data/synthetic/random_numbers.csv')
```

### Reproducing Results
```python
from llm_n_parameter import NParameterExperiment

# Use sample data instead of API calls
experiment = NParameterExperiment.from_cached_data('data/sample_experiments/')
```

## Data Generation

To regenerate synthetic data:
```bash
python scripts/generate_synthetic_data.py --output-dir data/synthetic/
```

To collect new experimental data (requires API keys):
```bash
python scripts/run_experiments.py --cache-dir data/sample_experiments/
```

## Notes

- All timestamp data uses ISO 8601 format in UTC
- Random seeds are documented in each file's metadata
- Synthetic data uses seed=42 for reproducibility
- Real API responses have been anonymized where necessary