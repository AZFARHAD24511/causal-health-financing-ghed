#!/usr/bin/env python
"""
Main script: Double Machine Learning analysis of public health financing
effects on out-of-pocket expenditure using WHO GHED data.

Usage
-----
python scripts/run_doubleml.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import (
    TREATMENT,
    OUTCOME,
    load_ghed,
    prepare_analysis_sample,
)
from src.causal_model import (
    run_doubleml_plr,
    run_multiple_learners,
    heterogeneous_effects_by_group,
)
from src.utils import save_summary_table, plot_ate_comparison, plot_heterogeneity


def main():
    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    data_file = PROJECT_ROOT / "data" / "GHED_data.XLSX"

    # Only load the columns we need (much faster for the large file)
    usecols = [
        "location", "code", "region", "income", "year",
        "gghed_che", "oops_che", "ext_che", "che_gdp",
        "che_pc_usd", "gghed_gdp",
    ]

    raw = load_ghed(data_file, usecols=usecols)
    sample, feature_cols = prepare_analysis_sample(raw, min_year=2000, max_year=2023)

    print("\nConfounders used:", feature_cols)
    print(sample[[TREATMENT, OUTCOME] + feature_cols].describe().round(2))

    # ------------------------------------------------------------------
    # 2. Main DoubleML estimation (XGBoost)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MAIN ANALYSIS: DoubleML with XGBoost")
    print("=" * 60)

    dml_model, main_summary = run_doubleml_plr(
        data=sample,
        treatment=TREATMENT,
        outcome=OUTCOME,
        confounders=feature_cols,
        learner_name="xgboost",
        n_folds=5,
        n_rep=1,
    )

    # ------------------------------------------------------------------
    # 3. Robustness: different learners
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROBUSTNESS: Multiple learners")
    print("=" * 60)

    robustness = run_multiple_learners(
        data=sample,
        treatment=TREATMENT,
        outcome=OUTCOME,
        confounders=feature_cols,
        learners=["xgboost", "lightgbm", "rf"],
    )

    # ------------------------------------------------------------------
    # 4. Heterogeneity by income group
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HETEROGENEITY: By World Bank income group")
    print("=" * 60)

    het_income = heterogeneous_effects_by_group(
        data=sample,
        treatment=TREATMENT,
        outcome=OUTCOME,
        confounders=feature_cols,
        group_var="income",
        learner_name="xgboost",
    )

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Main result
    pd_main = pd.DataFrame([main_summary])
    save_summary_table(pd_main, results_dir / "main_ate.csv")

    # Robustness
    save_summary_table(robustness, results_dir / "robustness_learners.csv")
    plot_ate_comparison(
        robustness,
        title="ATE of GGHE-D share on OOP share – by learner",
        save_path=results_dir / "ate_by_learner.png",
    )

    # Heterogeneity
    if not het_income.empty:
        save_summary_table(het_income, results_dir / "heterogeneity_income.csv")
        plot_heterogeneity(
            het_income,
            title="Heterogeneous ATE by Income Group",
            save_path=results_dir / "ate_by_income.png",
        )

    print("\n" + "=" * 60)
    print("Analysis finished. Results saved in /results")
    print("=" * 60)


if __name__ == "__main__":
    import pandas as pd  # needed for the main block
    main()
