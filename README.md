# Causal Effects of Public Health Financing on Out-of-Pocket Expenditure

**Double Machine Learning analysis using the WHO Global Health Expenditure Database (GHED)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Research Question

**What is the causal effect of increasing the share of Domestic General Government Health Expenditure (GGHE-D) on household Out-of-Pocket (OOP) health payments?**

We estimate this effect across 183 countries over the period 2000–2023 (4,024 country-year observations) using **Double Machine Learning (DML)**, a double-robust causal inference method.

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
A 1 percentage point increase in the share of government health spending (GGHE-D as % of CHE) is associated with an average **0.69 percentage point reduction** in the share of out-of-pocket expenditure. The effect is highly statistically significant and robust across different machine learning learners.

### Robustness Across Learners

| Learner       | ATE    | 95% CI             |
|---------------|--------|--------------------|
| XGBoost       | −0.674 | [−0.776, −0.572]   |
| Random Forest | −0.557 | [−0.636, −0.478]   |

![ATE by Learner](results/ate_by_learner.png)

### Heterogeneous Effects by Income Group

The protective effect of public financing is **stronger in lower-income countries**:

| Income Group  | ATE    | 95% CI             | N     |
|---------------|--------|--------------------|-------|
| Low           | −0.866 | [−1.062, −0.669]   | 565   |
| Lower-middle  | −0.848 | [−0.979, −0.717]   | 1,207 |
| Upper-middle  | −0.748 | [−0.931, −0.565]   | 1,159 |
| High          | −0.525 | [−0.615, −0.435]   | 1,093 |

![ATE by Income Group](results/ate_by_income.png)

---

## Method

We implement the **partially linear regression** model within the DoubleML framework (Chernozhukov et al., 2018):

\[
Y = \theta D + g(X) + \varepsilon, \quad D = m(X) + \nu
\]

- **Treatment (D):** `gghed_che` — Domestic General Government Health Expenditure as % of Current Health Expenditure  
- **Outcome (Y):** `oops_che` — Out-of-pocket expenditure as % of Current Health Expenditure  
- **Confounders (X):** External financing share, CHE as % of GDP, CHE per capita (log), GGHE-D as % of GDP  
- **Nuisance functions:** Estimated with XGBoost / Random Forest using 5-fold cross-fitting  

This approach is double-robust and reduces regularization bias compared to naive machine learning or simple OLS.

---

## Data

- **Source:** WHO Global Health Expenditure Database (GHED), December 2025 update  
- **Coverage:** 183 countries, 2000–2023  
- **Sample size after cleaning:** 4,024 country-year observations  

Raw data file is not included in the repository (see `data/README.md`).

---

## Repository Structure

```
causal-health-financing-ghed/
├── data/                   # Place GHED_data.XLSX here
├── src/
│   ├── data_prep.py        # Data loading & cleaning
│   ├── causal_model.py     # DoubleML estimation
│   └── utils.py            # Plotting helpers
├── scripts/
│   └── run_doubleml.py     # Main analysis script
├── results/                # Tables and figures
└── README.md
```

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place GHED_data.XLSX in the data/ folder, then run:
python scripts/run_doubleml.py
```

---

## Relevance

This project demonstrates modern **causal inference** methods (Double Machine Learning) applied to real multi-country health systems data. The findings are directly relevant to questions of financial protection and the role of public financing in reducing household health expenditure burden — core topics in health economics, epidemiology, and prevention research.

---

## References

- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*.
- WHO Global Health Expenditure Database: https://apps.who.int/nha/database
- System of Health Accounts 2011 (SHA 2011)

---

## License

MIT License
