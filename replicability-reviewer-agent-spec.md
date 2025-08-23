# Replicability Reviewer Agent Specification

## Agent Purpose and Overview

The **Replicability Reviewer Agent** is an AI assistant designed to comprehensively evaluate projects for reproducibility and replicability. It systematically reviews codebases, research materials, and documentation to identify gaps that could prevent others from successfully reproducing results, running code, or replicating experiments.

### Core Capabilities

- **Multi-domain Analysis**: Supports Python packages, R projects, JavaScript/Node.js applications, research papers, data analysis pipelines, web applications, and machine learning projects
- **Dependency Assessment**: Evaluates dependency management practices across different package managers and ecosystems
- **Environment Validation**: Reviews containerization, virtual environment setup, and platform compatibility
- **Documentation Review**: Assesses completeness and clarity of setup instructions, usage examples, and troubleshooting guides
- **Configuration Analysis**: Examines configuration management, secrets handling, and environment variables
- **Reproducibility Validation**: Checks for deterministic behavior, seed management, and version control practices

## What the Agent Checks For

### 1. Dependency Management
- **Package Files**: `requirements.txt`, `package.json`, `environment.yml`, `Pipfile`, `pyproject.toml`, `renv.lock`, `Cargo.toml`, etc.
- **Version Pinning**: Exact version specifications vs. range specifications
- **Lock Files**: Presence of lock files (`package-lock.json`, `Pipfile.lock`, `yarn.lock`)
- **Dependency Tree**: Analysis of transitive dependencies and potential conflicts
- **Security**: Outdated packages and known vulnerabilities

### 2. Environment Setup
- **Virtual Environment**: Instructions for creating isolated environments
- **Containerization**: Docker files, docker-compose configurations
- **Platform Requirements**: Operating system compatibility, architecture requirements
- **System Dependencies**: Non-package dependencies (databases, external tools, system libraries)
- **Installation Scripts**: Automated setup scripts and their robustness

### 3. Data Management
- **Data Availability**: Accessibility of datasets, APIs, and external resources
- **Data Documentation**: Schema documentation, data dictionaries, format specifications
- **Sample Data**: Availability of test datasets or data generators
- **Data Provenance**: Clear sourcing and licensing information
- **Data Processing**: Reproducible data preprocessing pipelines

### 4. Configuration and Secrets
- **Configuration Files**: Template files, example configurations
- **Environment Variables**: Documentation of required variables
- **Secrets Management**: Secure handling of API keys, credentials
- **Feature Flags**: Configurable behaviors and their documentation
- **Default Values**: Sensible defaults for optional configurations

### 5. Randomness and Determinism
- **Random Seeds**: Proper seed setting in all random operations
- **Stochastic Processes**: Documentation of non-deterministic elements
- **Hardware Dependencies**: GPU-specific behaviors, parallel processing effects
- **External API Calls**: Handling of variable external responses

### 6. Documentation Quality
- **README Completeness**: Installation, usage, examples, troubleshooting
- **API Documentation**: Function/class documentation, parameter descriptions
- **Examples**: Working code examples and tutorials
- **Changelog**: Version history and breaking changes
- **Contributing Guidelines**: Instructions for developers
- **License Information**: Clear licensing terms

### 7. Testing and Validation
- **Test Coverage**: Unit tests, integration tests, end-to-end tests
- **Test Data**: Availability of test fixtures and mock data
- **Continuous Integration**: Automated testing pipelines
- **Benchmark Tests**: Performance and accuracy validation
- **Cross-platform Testing**: Multi-OS and multi-environment validation

### 8. Version Control and Release Management
- **Git Practices**: Proper branching, tagging, and commit practices
- **Release Process**: Automated releases, version numbering
- **Binary Artifacts**: Reproducible build processes
- **Dependency Updates**: Procedures for updating dependencies safely

## How to Invoke the Agent

### Basic Usage
```
@replicability-reviewer Please review this project for reproducibility and replicability issues.
```

### Focused Reviews
```
@replicability-reviewer --focus=dependencies,environment
Review only the dependency management and environment setup aspects.
```

### Project-Type Specific Reviews
```
@replicability-reviewer --type=research-paper
Review this computational research project with focus on experimental reproducibility.
```

### Custom Checklist
```
@replicability-reviewer --checklist=custom --include=security,performance
Use a custom checklist that includes security and performance considerations.
```

## Standard Review Template

The agent follows this comprehensive checklist during reviews:

### Phase 1: Project Discovery and Classification

- [ ] **Project Type Identification**
  - [ ] Programming language(s) and frameworks
  - [ ] Project category (library, application, research, analysis)
  - [ ] Target platforms and environments
  - [ ] External dependencies and services

- [ ] **Repository Structure Analysis**
  - [ ] Standard directory structure for project type
  - [ ] Presence of configuration files
  - [ ] Documentation organization
  - [ ] Test directory structure

### Phase 2: Dependency and Environment Assessment

- [ ] **Package Management**
  - [ ] Primary package file exists (requirements.txt, package.json, etc.)
  - [ ] Version constraints are appropriate (not too loose, not unnecessarily strict)
  - [ ] Lock files present for reproducible installs
  - [ ] Development vs. production dependencies clearly separated
  - [ ] Optional dependencies documented

- [ ] **Environment Setup**
  - [ ] Clear installation instructions provided
  - [ ] Virtual environment setup documented
  - [ ] System requirements specified (Python version, Node version, etc.)
  - [ ] Platform-specific instructions where needed
  - [ ] Containerization available (Docker, etc.)

- [ ] **External Dependencies**
  - [ ] Database requirements and setup instructions
  - [ ] Third-party services and API requirements
  - [ ] System-level dependencies documented
  - [ ] Network requirements (ports, protocols)

### Phase 3: Configuration and Data Review

- [ ] **Configuration Management**
  - [ ] Environment variables documented
  - [ ] Configuration file templates provided
  - [ ] Default values specified
  - [ ] Sensitive information handling documented
  - [ ] Feature flags and toggles documented

- [ ] **Data Requirements**
  - [ ] Data sources clearly identified
  - [ ] Data access instructions provided
  - [ ] Sample or test data available
  - [ ] Data format and schema documented
  - [ ] Data preprocessing steps reproducible

### Phase 4: Code Quality and Determinism

- [ ] **Randomness Control**
  - [ ] Random seeds set appropriately
  - [ ] Non-deterministic operations documented
  - [ ] Parallel processing considerations addressed
  - [ ] External API variability handled

- [ ] **Code Organization**
  - [ ] Clear module structure
  - [ ] Function and class documentation
  - [ ] Error handling implemented
  - [ ] Logging configuration appropriate

### Phase 5: Testing and Validation

- [ ] **Test Suite**
  - [ ] Unit tests present and comprehensive
  - [ ] Integration tests for key workflows
  - [ ] Test data and fixtures available
  - [ ] Tests runnable in isolation
  - [ ] Performance/benchmark tests where appropriate

- [ ] **Continuous Integration**
  - [ ] CI pipeline configured
  - [ ] Multi-environment testing
  - [ ] Automated dependency checking
  - [ ] Code quality checks integrated

### Phase 6: Documentation Review

- [ ] **Primary Documentation**
  - [ ] README with clear purpose and usage
  - [ ] Installation instructions complete and tested
  - [ ] Usage examples provided
  - [ ] Troubleshooting section available
  - [ ] Known limitations documented

- [ ] **Developer Documentation**
  - [ ] API/code documentation complete
  - [ ] Architecture overview provided
  - [ ] Contributing guidelines available
  - [ ] Development setup instructions

- [ ] **Project Metadata**
  - [ ] License clearly specified
  - [ ] Version information available
  - [ ] Contact/support information provided
  - [ ] Citation information (for research projects)

## Output Format

The agent provides a structured review report containing:

### Executive Summary
- **Overall Reproducibility Score** (1-10)
- **Critical Issues Count**
- **Recommended Priority Actions**
- **Project Classification and Context**

### Detailed Findings

Each section includes:
- **Status**: ✅ Pass, ⚠️ Warning, ❌ Fail
- **Issue Description**: Clear explanation of problems found
- **Impact Assessment**: How this affects reproducibility
- **Recommendations**: Specific actions to address issues
- **Examples**: Code snippets or configuration examples where helpful

### Priority Matrix

Issues categorized by:
- **Critical**: Blocks basic reproduction
- **High**: Significantly impacts reproducibility
- **Medium**: Creates friction or uncertainty
- **Low**: Best practice improvements

### Project-Specific Recommendations

Tailored advice based on:
- Project type and domain
- Technology stack
- Intended audience
- Complexity level

## Customization Options

### Project Type Profiles

The agent can use specialized checklists for:
- **Research Papers**: Focus on experimental reproducibility, data availability
- **Python Packages**: PyPI publishing, testing across Python versions
- **Web Applications**: Deployment configuration, database migrations
- **Data Analysis**: Notebook reproducibility, data pipeline validation
- **Machine Learning**: Model artifacts, training reproducibility

### Severity Levels

Configure issue classification:
- **Strict Mode**: Academic/research standards
- **Standard Mode**: Industry best practices
- **Permissive Mode**: Minimum viable reproducibility

### Custom Rules

Add organization-specific requirements:
- Internal tool requirements
- Security compliance checks
- Performance benchmarks
- Documentation standards

## Integration Examples

### GitHub Actions Integration
```yaml
- name: Replicability Review
  uses: replicability-reviewer-action@v1
  with:
    project-type: python-package
    focus: dependencies,testing
    output-format: markdown
```

### Pre-commit Hook
```yaml
repos:
  - repo: local
    hooks:
      - id: replicability-check
        name: Replicability Review
        entry: replicability-reviewer --quick-check
        language: system
```

### CLI Usage
```bash
# Full review
replicability-reviewer review ./project

# Quick dependency check  
replicability-reviewer check-deps --warn-only

# Generate report
replicability-reviewer report --format=pdf --output=review.pdf
```

## Best Practices for Agent Usage

### When to Use
- Before publishing or releasing projects
- During code reviews for reproducibility
- When onboarding new team members
- Before submitting research papers
- When creating project templates

### How to Maximize Value
1. **Run Early**: Use during development, not just before release
2. **Iterate**: Re-run after addressing issues to validate fixes
3. **Customize**: Adapt checklists to your domain and requirements
4. **Document**: Keep review results as part of project documentation
5. **Automate**: Integrate into CI/CD pipelines for continuous monitoring

### Common Pitfalls to Avoid
- Don't ignore "low priority" issues—they accumulate
- Don't assume one-time review is sufficient
- Don't skip testing recommended fixes
- Don't apply generic solutions without context consideration

This specification provides a comprehensive framework for implementing a replicability reviewer agent that can adapt to various project types while maintaining consistent, thorough evaluation standards.