# Journal Reviewer Agents for LLM Research Papers

## 1. Statistical Rigor Reviewer

### Purpose
Evaluates statistical methodology, power analysis, and validity of conclusions.

### Key Checks
- **Sample Size Justification**: Is n sufficient for claimed effects?
- **Multiple Comparisons**: Bonferroni/FDR corrections applied?
- **Effect Sizes**: Reported alongside p-values?
- **Assumptions**: Independence, normality, homoscedasticity verified?
- **Power Analysis**: Post-hoc power calculated?
- **Confidence Intervals**: Provided for all estimates?

### Specific to This Study
- ICC calculation methodology correct?
- Design effect formula properly applied?
- KS test appropriate for distribution comparison?
- Position effects tested with appropriate methods?

## 2. Computational Reproducibility Reviewer

### Purpose
Ensures computational experiments can be reproduced by reviewers.

### Key Checks
- **Compute Environment**: Hardware specs documented?
- **Runtime**: Execution time and resource usage reported?
- **Determinism**: Random seeds set and documented?
- **Versioning**: Exact versions of all libraries?
- **Data Availability**: Can reviewers access the data?
- **Code Quality**: Clean, documented, modular?

### Journal-Specific Requirements
- **JMLR**: Requires code release with paper
- **Nature**: Requires Code Availability Statement
- **Science**: Requires materials availability form

## 3. Ethics and Responsible AI Reviewer

### Purpose
Evaluates ethical considerations and responsible AI practices.

### Key Checks
- **Bias Analysis**: Systematic biases identified and discussed?
- **Environmental Impact**: CO₂ emissions from experiments?
- **Data Privacy**: Any PII in prompts/responses?
- **Dual Use**: Could findings be misused?
- **Limitations**: Clearly stated and not overreaching?
- **Broader Impacts**: Societal implications discussed?

### For LLM Studies
- API costs documented (accessibility concern)?
- Model biases acknowledged?
- Implications for research inequality?

## 4. Literature Completeness Reviewer

### Purpose
Ensures comprehensive coverage of related work.

### Key Checks
- **Recent Papers**: Last 2 years adequately covered?
- **Seminal Works**: Foundational papers cited?
- **Cross-Disciplinary**: Stats + ML + NLP papers?
- **Competing Methods**: Alternative approaches discussed?
- **Research Gap**: Clearly positioned in literature?

### Search Strategy
```python
search_terms = [
    "LLM reproducibility",
    "n parameter API",
    "batch generation correlation",
    "statistical equivalence testing",
    "design effect NLP",
    "ICC language models"
]
venues = ["ACL", "NeurIPS", "ICML", "EMNLP", "JMLR"]
years = [2023, 2024, 2025]
```

## 5. Methodological Soundness Reviewer

### Purpose
Evaluates experimental design and methodology.

### Key Checks
- **Control Variables**: Properly controlled?
- **Confounding Factors**: Identified and addressed?
- **Experimental Design**: Randomization used?
- **Validation**: Cross-validation or holdout?
- **Robustness**: Sensitivity analysis performed?

### Critical Questions
- Why these specific prompts?
- Why these models (GPT-4, Gemini)?
- Why these sample sizes?
- Alternative explanations considered?

## 6. Writing Quality Reviewer

### Purpose
Ensures clarity, concision, and journal style compliance.

### Key Checks
- **Abstract**: 150-250 words, structured?
- **Introduction**: Research question clear by paragraph 2?
- **Methods**: Reproducible from description alone?
- **Results**: Figures before first reference?
- **Discussion**: Balanced, not overstating?
- **References**: Journal format, all DOIs?

### Journal-Specific
- **Nature/Science**: 3000 word limit
- **JMLR**: LaTeX required
- **ACL**: 8 pages + unlimited references

## 7. Impact and Novelty Reviewer

### Purpose
Assesses contribution and significance.

### Key Checks
- **Novelty**: What's genuinely new?
- **Impact**: Who cares and why?
- **Practical Applications**: Real-world relevance?
- **Theoretical Contribution**: Advances understanding?
- **Future Work**: Opens new directions?

### Impact Metrics
- How many researchers use n parameter?
- Cost savings quantified?
- Statistical errors prevented?
- Reproducibility improved by how much?

## 8. Data and Code Availability Reviewer

### Purpose
Ensures FAIR (Findable, Accessible, Interoperable, Reusable) principles.

### Key Checks
- **DOI**: Zenodo/Figshare/Dryad deposit?
- **License**: Appropriate for reuse?
- **Documentation**: README, requirements, instructions?
- **Archive**: Long-term preservation plan?
- **Metadata**: Properly tagged and searchable?

### Tier 1 Journal Requirements
- Zenodo DOI required
- Code review during peer review
- Data availability statement
- Software heritage archive

## 9. Statistical Software Reviewer

### Purpose
Validates statistical implementation correctness.

### Key Checks
- **Function Calls**: Correct parameters?
- **Implementations**: Hand-rolled stats correct?
- **Numerical Stability**: Edge cases handled?
- **Precision**: Float64 where needed?
- **Vectorization**: Efficient operations?

### Red Flags
```python
# Bad: Hand-rolled statistics
def variance(x):
    return sum((i - mean(x))**2 for i in x) / len(x)

# Good: Use proven libraries
from scipy import stats
variance = np.var(x, ddof=1)  # Sample variance
```

## 10. Experimental Cost Reviewer

### Purpose
Documents and justifies experimental costs.

### Key Checks
- **Total Cost**: API calls, compute, storage
- **Cost per Claim**: $ per statistical test
- **Alternatives**: Cheaper methods considered?
- **Necessity**: Each experiment justified?
- **Replication Cost**: For other researchers?

### For This Study
- OpenAI API costs: ~$X
- Gemini API costs: ~$Y  
- Synthetic data alternative: $0
- Cost barrier for replication?

## Automated Review Pipeline

```python
def run_journal_review(paper_path, target_journal="JMLR"):
    """Run all relevant reviewer agents for journal submission."""
    
    reviewers = [
        StatisticalRigorReviewer(),
        ComputationalReproducibilityReviewer(),
        EthicsReviewer(),
        LiteratureReviewer(),
        MethodologyReviewer(),
        WritingQualityReviewer(journal=target_journal),
        ImpactReviewer(),
        DataAvailabilityReviewer(),
        StatisticalSoftwareReviewer(),
        CostReviewer()
    ]
    
    results = {}
    for reviewer in reviewers:
        results[reviewer.name] = reviewer.review(paper_path)
    
    return JournalReadinessReport(results)
```

## Journal-Specific Configurations

### JMLR (Journal of Machine Learning Research)
```yaml
focus_areas:
  - statistical_rigor: critical
  - reproducibility: critical  
  - code_quality: high
  - novelty: moderate
  - writing: moderate
special_requirements:
  - LaTeX submission
  - Code on GitHub
  - Reproducibility checklist
```

### Nature Machine Intelligence
```yaml
focus_areas:
  - novelty: critical
  - impact: critical
  - writing: critical
  - statistics: high
  - ethics: high
special_requirements:
  - 3000 word limit
  - Extended data
  - Code availability
  - Reporting checklist
```

### ACL/EMNLP
```yaml
focus_areas:
  - nlp_relevance: critical
  - experimental_design: high
  - reproducibility: high
  - related_work: high
  - clarity: moderate
special_requirements:
  - 8 page limit
  - Anonymized submission
  - Reproducibility checklist
  - Ethics statement
```

## Pre-Submission Checklist Generator

```python
def generate_submission_checklist(journal):
    """Generate journal-specific submission checklist."""
    
    checklist = {
        "Required Documents": [],
        "Statistical Requirements": [],
        "Reproducibility Items": [],
        "Ethical Considerations": [],
        "Format Requirements": []
    }
    
    if journal in ["Nature", "Science"]:
        checklist["Required Documents"].extend([
            "[ ] Cover letter",
            "[ ] Reporting summary",
            "[ ] Code availability statement",
            "[ ] Data availability statement"
        ])
    
    return checklist
```