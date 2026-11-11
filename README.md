# Causal Effects of Public Health Financing on Out-of-Pocket Expenditure

**Double Machine Learning analysis using the WHO Global Health Expenditure Database (GHED)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Research Question

**What is the causal effect of increasing the share of Domestic General Government Health Expenditure (GGHE-D) on household Out-of-Pocket (OOP) health payments?**

We estimate this effect across 195 countries over the period 2000–2023 using **Double Machine Learning (DML)**, a double-robust causal inference method that combines flexible machine learning with orthogonalization to reduce regularization and overfitting bias.

This question is central to health systems research and financial protection policy:  
Does higher public financing of health care reduce the financial burden on households?

---

## Why Double Machine Learning?

Traditional regression approaches struggle with:
- High-dimensional confounding
- Non-linear relationships
- Regularization bias when using flexible learners

Double Machine Learning (Chernozhukov et al., 2018) addresses these issues by:
1. Using flexible ML models (e.g. Gradient Boosting) to estimate nuisance functions
2. Orthogonalizing the treatment and outcome
3. Providing valid inference under weaker assumptions than parametric models

This makes DML particularly suitable for multi-country panel data with complex confounding structures.

---

## Data

We use the **WHO Global Health Expenditure Database (GHED)** (December 2025 update):

- ~195 countries
- Years: 2000–2023 (partial 2024)
- Key indicators following the System of Health Accounts (SHA) 2011

**Main variables used:**

| Role              | Variable              | Description                                      |
|-------------------|-----------------------|--------------------------------------------------|
| Treatment (D)     | `gghed_che`           | GGHE-D as % of Current Health Expenditure (CHE) |
| Outcome (Y)       | `oops_che`            | Out-of-pocket expenditure as % of CHE            |
| Confounders (X)   | GDP per capita, income group, region, year, population structure proxies, external financing share, etc. | |

> **Note:** The raw GHED file is not included in this repository due to size. Place `GHED_data.XLSX` in the `data/` folder.

---

## Method Overview

We implement the **partially linear regression model** within the DoubleML framework:

\[
Y = \theta D + g(X) + \varepsilon, \quad \mathbb{E}[\varepsilon|D,X] = 0
\]
\[
D = m(X) + \nu, \quad \mathbb{E}[\nu|X] = 0
\]

Where:
- \( \theta \) is the average treatment effect of interest
- \( g(X) \) and \( m(X) \) are estimated with machine learning (XGBoost / LightGBM / Random Forest)
- Cross-fitting is used to avoid overfitting bias

We also explore **heterogeneous treatment effects** by income group and WHO region.

---

## Repository Structure

```
causal-health-financing-ghed/
├── data/                       # Place GHED_data.XLSX here
├── src/
│   ├── data_prep.py            # Loading & cleaning GHED
│   ├── causal_model.py         # DoubleML estimation
│   └── utils.py                # Helpers & plotting
├── scripts/
│   └── run_doubleml.py         # Main analysis script
├── results/                    # Output tables & figures
├── notebooks/                  # Exploratory analysis (optional)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/causal-health-financing-ghed.git
cd causal-health-financing-ghed

# 2. Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Place the GHED file
# Copy GHED_data.XLSX into the data/ folder

# 4. Run the analysis
python scripts/run_doubleml.py
```

---

## Key Results (Preliminary)

*(Results will be updated after running the full pipeline)*

- Point estimate of the average treatment effect (ATE)
- Confidence intervals
- Heterogeneity by World Bank income group
- Robustness checks (different learners, different sets of confounders)

---

## Relevance to Epidemiological & Prevention Research

This project demonstrates:
- Application of **modern causal inference methods** (DoubleML) to real-world health systems data
- Ability to handle high-dimensional, multi-country panel data
- Focus on **financial protection** — a key determinant of equitable access to care and prevention
- Clean, reproducible, and modular code suitable for collaborative research

These skills align directly with methodological needs in large cohort studies and secondary data analysis in epidemiology.

---

## References

- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*.
- WHO Global Health Expenditure Database: https://apps.who.int/nha/database
- System of Health Accounts 2011 (SHA 2011)

---

## Author

Prepared as part of methodological research in causal inference for health systems and prevention research.

---

## License

MIT License
