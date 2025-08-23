# Replicability Reviewer Agent Specification

## Purpose
The replicability-reviewer agent evaluates projects for reproducibility, ensuring that other researchers or developers can successfully replicate results, builds, and analyses.

## Core Capabilities

### 1. Dependency Analysis
- **Package Management**: Checks for requirements.txt, package.json, Gemfile, etc.
- **Version Pinning**: Ensures specific versions are locked
- **Lock Files**: Validates presence of package-lock.json, poetry.lock, etc.
- **Virtual Environments**: Checks for venv, conda, or container specifications

### 2. Environment Setup Review
- **Installation Instructions**: Clear, step-by-step setup guide
- **System Requirements**: OS, hardware, and software prerequisites
- **Configuration Files**: .env.example, config templates
- **Secrets Management**: Proper handling without exposing sensitive data

### 3. Data Availability Assessment
- **Data Sources**: Clear documentation of data origin
- **Access Methods**: How to obtain required datasets
- **Data Versioning**: Tracking data changes over time
- **Sample Data**: Minimal examples for testing

### 4. Reproducibility Checks
- **Random Seeds**: Fixed seeds for stochastic processes
- **Deterministic Operations**: Avoiding non-deterministic functions
- **Hardware Dependencies**: GPU, CPU architecture considerations
- **Timing Dependencies**: Order of operations, race conditions

### 5. Documentation Quality
- **README Completeness**: Setup, usage, examples
- **API Documentation**: Function signatures, parameters
- **Architecture Docs**: System design, data flow
- **Troubleshooting**: Common issues and solutions

### 6. Testing Infrastructure
- **Test Coverage**: Unit, integration, end-to-end tests
- **CI/CD Pipeline**: Automated testing on multiple platforms
- **Test Data**: Fixtures and mocks availability
- **Performance Benchmarks**: Reproducible performance metrics

## Usage

### Basic Invocation
```bash
# Review entire project
claude code review --replicability .

# Focus on specific aspect
claude code review --replicability --focus dependencies .
claude code review --replicability --focus data-access .
```

### In Claude Code
```
Please run a replicability review on this project
```

## Checklist Template

### ✅ Dependencies
- [ ] Package manager file exists (requirements.txt, package.json, etc.)
- [ ] Versions are pinned or ranges specified
- [ ] Lock file present and committed
- [ ] Development vs production dependencies separated
- [ ] System dependencies documented

### ✅ Environment
- [ ] Setup instructions in README
- [ ] Environment variables documented
- [ ] Configuration examples provided
- [ ] Docker/container files if applicable
- [ ] Platform-specific notes included

### ✅ Data
- [ ] Data sources documented
- [ ] Download/access instructions clear
- [ ] Data preprocessing reproducible
- [ ] Sample/test data available
- [ ] Data versioning strategy defined

### ✅ Code Quality
- [ ] Random seeds set where needed
- [ ] File paths use os-agnostic methods
- [ ] No hardcoded absolute paths
- [ ] External API calls documented
- [ ] Time-dependent code handled properly

### ✅ Testing
- [ ] Tests can run independently
- [ ] CI/CD configuration present
- [ ] Test coverage reported
- [ ] Integration tests for key workflows
- [ ] Performance tests reproducible

### ✅ Documentation
- [ ] Installation guide complete
- [ ] Usage examples provided
- [ ] Expected outputs documented
- [ ] Troubleshooting section exists
- [ ] Citation/attribution guidelines

## Project-Specific Configurations

### Python Research Projects
```yaml
priority:
  - conda environment.yml or requirements.txt
  - Random seed management in numpy, torch, etc.
  - Jupyter notebook execution order
  - Data download scripts
  - Results validation methods
```

### Web Applications
```yaml
priority:
  - Docker compose setup
  - Database migrations
  - Environment variables
  - Build process documentation
  - API endpoint testing
```

### Data Analysis
```yaml
priority:
  - Raw data accessibility
  - Processing pipeline documentation
  - Intermediate data caching
  - Visualization reproducibility
  - Statistical test parameters
```

## Output Format

### Summary Report
```markdown
# Replicability Review Report

## Overall Score: B+ (82/100)

### Strengths
- Comprehensive dependency management
- Well-documented setup process
- Good test coverage

### Critical Issues
1. Missing random seed in ML pipeline
2. Data download requires manual intervention
3. No CI/CD for Windows platform

### Recommendations
1. Add seeds to all stochastic processes
2. Automate data acquisition
3. Include Windows in CI matrix
```

## Integration Examples

### GitHub Action
```yaml
name: Replicability Check
on: [push, pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Replicability Review
        run: |
          # Agent invocation here
```

### Pre-commit Hook
```yaml
repos:
  - repo: local
    hooks:
      - id: replicability-check
        name: Check replicability
        entry: claude code review --replicability
        language: system
        pass_filenames: false
```

## Best Practices

1. **Start Early**: Run reviews during development, not just at the end
2. **Automate Checks**: Integrate into CI/CD pipeline
3. **Document Exceptions**: Explain why certain items can't be reproducible
4. **Version Everything**: Code, data, environments, and documentation
5. **Test on Clean System**: Verify setup works from scratch
6. **Provide Fallbacks**: Alternative data sources, simplified examples
7. **Update Regularly**: Keep dependencies and instructions current

## Common Pitfalls to Check

- Assuming specific OS or shell (bash vs zsh vs cmd)
- Relying on user-specific paths (~/, %USERPROFILE%)
- Missing timezone specifications
- Undocumented external service dependencies
- Assuming specific hardware (GPU, RAM size)
- No version bounds on dependencies
- Missing data cleaning steps
- Unclear execution order for scripts
- No validation of reproduced results
- Incomplete error handling for missing dependencies