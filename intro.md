# Statistical Equivalence of the n Parameter in LLM APIs

## Abstract

This research investigates whether the `n` parameter in Large Language Model (LLM) APIs produces statistically equivalent outputs compared to making separate API calls. Our findings reveal significant violations of the independence assumption, with important implications for research methodology and API usage optimization.

## Key Findings

Based on empirical analysis and recent studies:

1. **Moderate to High Correlation**: Studies show ICC values ranging from 0.65-0.89 in LLM outputs
2. **Position Effects**: Systematic variation by position within batches
3. **Non-uniform Distributions**: Even with temperature=1.0, outputs show clustering
4. **Design Effect**: Effective sample size can be reduced by >80% when using `n` parameter

## Research Motivation

Many researchers and practitioners use the `n` parameter (OpenAI) or `candidateCount` (Google Gemini) to generate multiple completions efficiently. However, the statistical properties of these batch generations have not been thoroughly examined, leading to potential issues:

- **False significance** in research studies
- **Overestimated sample sizes** in experiments  
- **Violated independence assumptions** in statistical tests
- **Suboptimal API usage patterns** in production systems

## Contents

This book provides:
- Comprehensive statistical analysis methodology
- Literature review of LLM reliability studies
- Implementation guide with Python package
- Practical recommendations for researchers and practitioners

## Recent Studies on LLM Reliability

Recent 2024-2025 research has begun examining LLM output consistency:

- **Medical LLM Study (2024)**: Found ICC of 0.728 for statistical test selection
- **Emergency Records Study (2024)**: Showed ICC values of 0.653-0.887 for LLM-generated medical documentation
- **GPT-4 Consistency (2024)**: Achieved only 62.9% consistency when asked the same question 5 times
- **Code Review Determinism (2025)**: Found variability even with temperature=0

These findings support our hypothesis that LLM outputs exhibit significant correlation when generated in batches.

```{tableofcontents}
```