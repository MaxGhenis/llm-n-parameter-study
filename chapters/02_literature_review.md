# Chapter 2: Literature Review

## Background

Large Language Models (LLMs) have become fundamental tools in natural language processing research {cite}`brown2020language,openai2023gpt4`. As these models are increasingly used for scientific studies, understanding the statistical properties of their outputs becomes critical for research validity.

## The n Parameter in LLM APIs

The `n` parameter in OpenAI's API {cite}`openai2024api` and the `candidateCount` parameter in Google's Gemini API {cite}`gemini2024api` allow researchers to generate multiple completions in a single API call. While this feature offers significant computational and cost benefits, its statistical implications have not been formally studied.

### Current Understanding

OpenAI's documentation states that the `n` parameter generates "n chat completion choices for each input message" {cite}`openai2024api`. Similarly, Google's Gemini API documentation indicates that `candidateCount` returns multiple response variations while charging for input tokens only once {cite}`gemini2024api`.

## Evidence of Non-Independence

Recent empirical work has begun to reveal potential issues with assuming independence in LLM outputs:

### Research Gap: No Prior Studies on n Parameter Correlation

Our extensive literature search found **no published studies** specifically examining the statistical properties of the `n` parameter in LLM APIs. This represents a significant research gap, as:

- **Thousands of researchers** use the `n` parameter assuming independence
- **No empirical validation** exists for this assumption
- **Potential for widespread statistical errors** in published LLM research

```{warning}
The absence of prior work on this topic means that many published studies using the `n` parameter may have inflated significance levels or incorrect confidence intervals.
```

### Community Observations

Several community reports have documented anomalies in batch generation:

1. **HuggingFace Users** {cite}`huggingface2024generation` reported that using `num_return_sequences > 1` often produces nearly identical outputs, while separate calls yield diverse results.

2. **vLLM Implementation** {cite}`vllm2024nbias` found that high `n` values (e.g., n=40) produced aberrant output length distributions compared to independent calls.

## Statistical Methods for Correlated Data

### Intraclass Correlation Coefficient (ICC)

The ICC, formalized by {cite:t}`shrout1979intraclass`, measures the proportion of variance attributable to between-group differences:

$$ICC = \frac{\sigma^2_{between}}{\sigma^2_{between} + \sigma^2_{within}}$$

### Design Effect

{cite:t}`kish1965survey` introduced the design effect concept for survey sampling, which applies directly to our context:

$$D_{eff} = 1 + (m-1) \times ICC$$

where $m$ is the cluster size (batch size in our case).

### Multilevel Modeling

{cite:t}`snijders2011multilevel` provide comprehensive methods for analyzing hierarchical data structures, which are applicable when LLM outputs are nested within API calls.

## Sampling and Generation in LLMs

### Temperature and Diversity

{cite:t}`holtzman2019curious` demonstrated that standard sampling methods in language models can lead to degenerate text. Their work on nucleus sampling (top-p) provides context for understanding how sampling parameters affect output diversity.

### Self-Consistency and Multiple Samples

{cite:t}`wang2023self` leveraged multiple samples through self-consistency to improve reasoning performance. However, their work assumes independence among samples, an assumption our research questions.

### Instruction Following and RLHF

{cite:t}`ouyang2022training` describe how Reinforcement Learning from Human Feedback (RLHF) shapes model outputs. This training process may contribute to the systematic biases we observe in batch generation.

## Statistical Testing Methods

### Distribution Comparison

The Kolmogorov-Smirnov test {cite}`kolmogorov1933sulla,smirnov1948table` provides a non-parametric method to compare distributions:

$$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$

### Rank-Based Tests

The Mann-Whitney U test {cite}`mann1947test` offers a robust alternative for comparing central tendencies without assuming specific distributions.

### Time Series Independence

The Ljung-Box test {cite}`ljungbox1978measure` can detect autocorrelation in sequences, applicable to testing independence in sequential LLM outputs.

## Research Gap

Despite extensive work on LLM capabilities and applications, the literature reveals a critical gap:

1. **No formal comparison** of `n` parameter vs. separate API calls
2. **Limited awareness** of correlation in batch-generated outputs
3. **Absence of guidelines** for researchers using multiple LLM completions
4. **No quantification** of the impact on statistical analyses

## Implications for Current Research

Many studies implicitly assume i.i.d. outputs when using LLMs:

- **Prompt Engineering Studies**: Often compare prompts using multiple samples
- **Bias Detection**: Aggregate multiple outputs to measure model biases
- **Chain-of-Thought Reasoning** {cite}`wei2022chain`: Sample multiple reasoning paths

If the independence assumption is violated, these studies may report inflated significance or incorrect effect sizes.

## Summary

The literature suggests that:

1. **No prior studies** have examined the statistical properties of the `n` parameter
2. **Implementation matters** for output diversity {cite}`huggingface2024generation,vllm2024nbias`
3. **Statistical methods exist** to handle correlated data {cite}`shrout1979intraclass,kish1965survey,snijders2011multilevel`
4. **The specific impact of the `n` parameter remains unstudied**

This gap motivates our systematic investigation of whether the `n` parameter produces statistically equivalent outputs to separate API calls.