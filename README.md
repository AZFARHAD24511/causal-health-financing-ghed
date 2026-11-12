# Causal Effects of Public Health Financing on Out-of-Pocket Expenditure

**Target Trial Emulation + Double Machine Learning & Causal Forest**  
*Using the WHO Global Health Expenditure Database (GHED)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Motivation

Financial protection is a core goal of health systems and a key determinant of equitable access to care and prevention.  
This project asks a policy-relevant causal question:

> **Does increasing the share of domestic general government health expenditure reduce the burden of out-of-pocket payments on households?**

We address this question using modern causal inference methods (Double Machine Learning and Causal Forests) within an explicit **Target Trial Emulation** framework.

---

## Target Trial Emulation Protocol

We map the observational analysis to a hypothetical target trial:

| Component              | Specification                                                                 |
|------------------------|-------------------------------------------------------------------------------|
| **Eligibility**        | Country-years with complete data on financing and expenditure (2000–2023)    |
| **Treatment strategies**| Increase in GGHE-D as % of Current Health Expenditure (CHE)                  |
| **Causal contrast**    | Average Treatment Effect (ATE) of higher public financing share              |
| **Outcome**            | Out-of-pocket expenditure as % of CHE (`oops_che`)                           |
| **Causal estimand**    | ATE under the partially linear model, with heterogeneous effects by income   |
| **Analysis method**    | Double Machine Learning (double-robust) + Causal Forest for HTE              |

This framing makes the assumptions transparent and aligns the analysis with current best practices in epidemiology and health services research.

---

## Key Findings

### Main Result (DoubleML + XGBoost)

| Metric                         | Value                |
|--------------------------------|----------------------|
| Average Treatment Effect (ATE) | **−0.689**           |
| Standard Error                 | 0.050                |
| 95% Confidence Interval        | [−0.787, −0.590]     |
| p-value                        | < 0.001              |

**Interpretation:**  
A 1 percentage point increase in the government share of health spending is associated with an average **0.69 percentage point reduction** in the out-of-pocket share. The effect is highly significant and robust.

### Robustness Across Learners

| Learner       | ATE    | 95% CI             |
|---------------|--------|--------------------|
| XGBoost       | −0.674 | [−0.776, −0.572]   |
| Random Forest | −0.557 | [−0.636, −0.478]   |

![ATE by Learner](results/ate_by_learner.png)

### Heterogeneous Treatment Effects by Income Group

The protective effect of public financing is **substantially stronger in lower-income countries**:

| Income Group  | ATE    | 95% CI             | N     |
|---------------|--------|--------------------|-------|
| Low           | −0.866 | [−1.062, −0.669]   | 565   |
| Lower-middle  | −0.848 | [−0.979, −0.717]   | 1,207 |
| Upper-middle  | −0.748 | [−0.931, −0.565]   | 1,159 |
| High          | −0.525 | [−0.615, −0.435]   | 1,093 |

![ATE by Income Group](results/ate_by_income.png)

---

## Methods

### 1. Double Machine Learning (Partial Linear Model)

We estimate the ATE using the partially linear regression model within the DoubleML framework (Chernozhukov et al., 2018):

$$
Y = \theta D + g(X) + \varepsilon, \quad D = m(X) + \nu
$$

- **Treatment (D):** `gghed_che`
- **Outcome (Y):** `oops_che`
- **Confounders (X):** External financing share, CHE as % of GDP, log(CHE per capita), GGHE-D as % of GDP
- Nuisance functions estimated with XGBoost / Random Forest + 5-fold cross-fitting

This estimator is **double-robust**.

### 2. Causal Forest for Heterogeneous Effects

In addition to stratified estimates, we implement approaches related to Causal Forests to examine conditional average treatment effects (CATE) more flexibly. The current version reports stratified HTEs by income group; extension to full Causal Forest (e.g. via `econml`) is straightforward within the existing code structure.

---

## Data

- **Source:** WHO Global Health Expenditure Database (GHED), December 2025 update  
- **Sample:** 183 countries, 2000–2023 → **4,024 country-year observations** after cleaning  
- Raw data are not stored in the repository (see `data/README.md`)

---

## Repository Structure

```
causal-health-financing-ghed/
├── data/                   # Place GHED_data.XLSX here
├── src/
│   ├── data_prep.py        # Data loading & cleaning
│   ├── causal_model.py     # DoubleML + helpers
│   └── utils.py            # Plotting & reporting
├── scripts/
│   └── run_doubleml.py     # Main analysis pipeline
├── results/                # Tables and figures
└── README.md
```

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place GHED_data.XLSX in data/, then:
python scripts/run_doubleml.py
```

---

## Relevance for Methodological Research in Epidemiology

This project demonstrates:

- Explicit use of the **Target Trial Emulation** framework
- Application of **double-robust machine learning** methods (DoubleML)
- Estimation of **heterogeneous treatment effects** relevant to equity and prevention
- Clean, reproducible, modular code suitable for extension into research software

These elements align directly with current priorities in causal inference for large-scale health and epidemiological data.

---

## References

- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*.
- Hernán & Robins. *Causal Inference: What If* (Target Trial framework)
- WHO Global Health Expenditure Database: https://apps.who.int/nha/database
- System of Health Accounts 2011 (SHA 2011)

---

## License

MIT License
