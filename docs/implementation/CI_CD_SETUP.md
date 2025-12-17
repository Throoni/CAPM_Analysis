# CI/CD Setup Guide

##  **COMPLETE: Continuous Integration & Deployment**

This document describes the CI/CD setup for automated testing, auditing, and quality checks.

---

##  What's Included

### 1. **GitHub Actions Workflows**

#### **CI Workflow** (`.github/workflows/ci.yml`)
- **Triggers:** Push, PR, Manual
- **Jobs:**
  - **Test:** Runs unit and integration tests across Python 3.10, 3.11, 3.12
  - **Lint:** Code quality checks (flake8, pylint)
  - **Security:** Security scanning (bandit, safety)

#### **Audit Workflow** (`.github/workflows/audit.yml`)
- **Triggers:** Daily schedule (2 AM UTC), Push to main, Manual
- **Jobs:**
  - **Full Audit:** Complete system audit with report
  - **Quick Audit:** Security-focused quick check

### 2. **Pre-commit Hooks** (`.pre-commit-config.yaml`)
- **Auto-formatting:** Black, isort
- **Quality checks:** flake8, trailing whitespace, file checks
- **Security:** Private key detection
- **Testing:** Runs unit tests before commit

---

##  Setup Instructions

### **1. GitHub Actions (Automatic)**

GitHub Actions run automatically when you:
- Push to `main`, `master`, or `develop` branches
- Open a pull request
- Manually trigger via GitHub UI

**No setup required** - workflows are in `.github/workflows/`

### **2. Pre-commit Hooks (Local)**

Install pre-commit hooks for local quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

**What it does:**
- Auto-formats code (Black, isort)
- Runs quality checks (flake8)
- Detects security issues
- Runs unit tests

**Bypass (if needed):**
```bash
git commit --no-verify -m "message"
```

---

##  Workflow Details

### **CI Workflow**

**Test Job:**
- Runs on Python 3.10, 3.11, 3.12
- Executes unit tests with coverage
- Executes integration tests
- Uploads coverage reports

**Lint Job:**
- Checks code style (flake8)
- Validates import structure
- Reports complexity issues

**Security Job:**
- Scans for security vulnerabilities (bandit)
- Checks dependencies (safety)
- Reports potential issues

### **Audit Workflow**

**Full Audit:**
- Runs complete system audit (24 modules)
- Generates audit report
- Uploads artifacts (30-day retention)
- Comments on PRs with summary

**Quick Audit:**
- Fast security-focused check
- Validates code quality
- Reports critical issues

---

##  Status Badges

Add to README.md (update YOUR_USERNAME):

```markdown
[![CI](https://github.com/YOUR_USERNAME/CAPM_Analysis/workflows/CI/badge.svg)](https://github.com/YOUR_USERNAME/CAPM_Analysis/actions)
[![Audit](https://github.com/YOUR_USERNAME/CAPM_Analysis/workflows/Audit/badge.svg)](https://github.com/YOUR_USERNAME/CAPM_Analysis/actions)
```

---

##  Benefits

### **Automated Quality Assurance**
-  Tests run on every push/PR
-  Code quality checked automatically
-  Security vulnerabilities detected
-  Audit reports generated daily

### **Early Problem Detection**
-  Catch issues before merge
-  Consistent code style
-  Prevent security vulnerabilities
-  Maintain test coverage

### **Time Savings**
-  No manual testing needed
-  Automated formatting
-  Pre-commit hooks catch issues early
-  Daily audit reports

---

##  Customization

### **Modify Test Matrix**

Edit `.github/workflows/ci.yml`:
```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']  # Add/remove versions
```

### **Change Audit Schedule**

Edit `.github/workflows/audit.yml`:
```yaml
schedule:
  - cron: '0 2 * * *'  # Daily at 2 AM UTC (adjust as needed)
```

### **Add More Hooks**

Edit `.pre-commit-config.yaml`:
```yaml
- repo: https://github.com/your-repo/your-hook
  rev: v1.0.0
  hooks:
    - id: your-hook-id
```

---

##  Usage Examples

### **View CI Results**

1. Go to GitHub repository
2. Click "Actions" tab
3. Select workflow run
4. View job results

### **View Audit Reports**

1. Go to GitHub Actions
2. Select "Automated Audit" workflow
3. Download artifacts
4. View `audit_report.md`

### **Run Pre-commit Manually**

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run flake8
```

---

##  Troubleshooting

### **CI Fails on Tests**

1. Check test output in Actions tab
2. Run tests locally: `pytest tests/ -v`
3. Fix failing tests
4. Push fix

### **Pre-commit Too Slow**

1. Skip hooks: `git commit --no-verify`
2. Or disable slow hooks in `.pre-commit-config.yaml`

### **Audit Fails**

1. Check audit logs in Actions artifacts
2. Run locally: `python audit/run_full_audit.py`
3. Fix issues reported
4. Re-run workflow

---

##  Related Files

- `.github/workflows/ci.yml` - CI workflow
- `.github/workflows/audit.yml` - Audit workflow
- `.pre-commit-config.yaml` - Pre-commit hooks
- `requirements.txt` - Dependencies

---

##  Verification

### **Test CI Locally**

```bash
# Install act (GitHub Actions locally)
brew install act  # macOS
# or: https://github.com/nektos/act

# Run CI workflow
act push
```

### **Test Pre-commit**

```bash
# Install and test
pre-commit install
pre-commit run --all-files
```

---

**Status:**  **COMPLETE - CI/CD Fully Configured**

**Last Updated:** December 8, 2025

