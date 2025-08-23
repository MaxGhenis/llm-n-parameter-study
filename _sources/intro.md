# Statistical Equivalence of the n Parameter in LLM APIs: An Empirical Study

```{admonition} Key Finding
:class: tip
The `n` parameter in LLM APIs produces statistically equivalent distributions to making separate API calls, while offering 20-100x performance improvements and up to 99% cost savings on input tokens.
```

## Executive Summary

Large Language Model (LLM) APIs like OpenAI's GPT and Google's Gemini offer an `n` parameter (or `candidateCount` in Gemini's case) that generates multiple completions in a single API call. Despite its significant implications for research efficiency and cost, **no formal study has evaluated whether this parameter produces statistically equivalent outputs to making multiple separate API calls**.

This research fills that gap through comprehensive empirical testing across multiple models and task types. Our findings have immediate practical implications for:

- **Researchers** conducting LLM-based studies requiring multiple samples
- **Developers** optimizing API usage and costs
- **Libraries** like EDSL implementing efficient sampling strategies

## The Problem

When researchers need multiple LLM responses for statistical analysis, they face a choice:

### Option 1: Loop Through API Calls
```python
responses = []
for i in range(100):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    responses.append(response)
```
- ❌ 100 separate API calls
- ❌ 100x input token charges
- ❌ Slower due to network overhead
- ✅ Guaranteed independence between samples

### Option 2: Use the n Parameter
```python
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    n=100
)
```
- ✅ Single API call
- ✅ Input tokens charged once
- ✅ ~100x faster
- ❓ **Unknown: Are outputs statistically equivalent?**

## Research Questions

This study addresses three critical questions:

1. **Statistical Equivalence**: Do the n parameter and looping produce statistically indistinguishable output distributions?

2. **Independence**: Are samples within an n-parameter call statistically independent?

3. **Practical Impact**: What are the performance and cost implications across different models and use cases?

## Key Contributions

1. **First formal evaluation** of the statistical properties of the n parameter across major LLM APIs

2. **Comprehensive testing** across multiple:
   - Models (GPT-4, GPT-4o-mini, Gemini-Pro)
   - Task types (random generation, classification, creative writing)
   - Sample sizes (n=10 to n=128)

3. **Practical implementation guide** for researchers and developers

4. **Open-source toolkit** for reproducing and extending our experiments

## Findings Preview

Our experiments reveal:

- **Statistical Equivalence**: Kolmogorov-Smirnov tests show p-values > 0.05 across all tested scenarios
- **Performance**: 23-95x speed improvements depending on n value
- **Cost Savings**: Up to 99% reduction in input token costs
- **Independence**: Ljung-Box tests confirm sample independence within n-parameter calls

## Impact

This research has immediate implications for:

### Cost Optimization
At current OpenAI pricing, generating 100 samples for a 1000-token prompt:
- Looping: $0.15 (100,000 input tokens)
- n parameter: $0.0015 (1,000 input tokens)
- **Savings: $0.1485 (99%)**

### Research Efficiency
For a study requiring 10,000 samples:
- Looping: ~2.8 hours
- n parameter: ~2 minutes
- **Time saved: 2.7 hours (98.8%)**

### Library Design
Libraries like EDSL can transparently optimize sampling:
```python
def run(self, n=100):
    if self.model.supports_n_parameter:
        return self._run_with_n(n=min(n, self.model.max_n))
    else:
        return [self._run_single() for _ in range(n)]
```

## Navigation Guide

This book is organized into four main parts:

1. **Background** (Chapters 1-2): Methodology and literature review
2. **Experiments** (Chapters 3-5): Detailed experiments across models
3. **Analysis** (Chapters 6-8): Statistical, performance, and independence analysis
4. **Applications** (Chapters 9-10): Practical implications and implementation

```{tableofcontents}
```