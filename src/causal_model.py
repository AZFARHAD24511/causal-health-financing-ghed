"""
Causal estimation for the effect of public health financing on out-of-pocket expenditure.

Implements:
- Double Machine Learning (partially linear model)
- Simple stratified heterogeneity
- Causal Forest style heterogeneous treatment effects (via econml)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV


try:
    from doubleml import DoubleMLData, DoubleMLPLR
except ImportError:
    raise ImportError(
        "doubleml package is required. Install with: pip install doubleml"
    )


def make_learner(name: str = "xgboost", random_state: int = 42):
    """
    Factory for nuisance model learners.

    Parameters
    ----------
    name : {'xgboost', 'lightgbm', 'rf', 'lasso'}
    """
    name = name.lower()
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        )
    elif name == "lasso":
        return LassoCV(cv=5, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unknown learner: {name}")


def run_doubleml_plr(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    learner_name: str = "xgboost",
    n_folds: int = 5,
    n_rep: int = 1,
    random_state: int = 42,
) -> Tuple:
    """
    Estimate the average treatment effect using DoubleML Partially Linear Regression.
    """
    dml_data = DoubleMLData(
        data=data,
        y_col=outcome,
        d_cols=treatment,
        x_cols=confounders,
    )

    ml_l = make_learner(learner_name, random_state)
    ml_m = make_learner(learner_name, random_state)

    dml_plr = DoubleMLPLR(
        obj_dml_data=dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score="partialling out",
    )

    print(f"Fitting DoubleML-PLR with {learner_name} learners "
          f"({n_folds}-fold cross-fitting) ...")
    dml_plr.fit()

    coef = dml_plr.coef[0]
    se = dml_plr.se[0]
    pval = dml_plr.pval[0]
    ci = dml_plr.confint(level=0.95).iloc[0].values

    summary = {
        "ate": float(coef),
        "se": float(se),
        "pvalue": float(pval),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "learner": learner_name,
        "n_obs": int(dml_data.n_obs),
        "n_confounders": len(confounders),
    }

    print("\n=== DoubleML Results ===")
    print(f"Average Treatment Effect (ATE): {coef:.4f}")
    print(f"Std. Error:                     {se:.4f}")
    print(f"95% CI:                         [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"p-value:                        {pval:.4g}")
    print(f"N observations:                 {dml_data.n_obs}")

    return dml_plr, summary


def run_multiple_learners(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    learners: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run DoubleML with several different learners for robustness checks."""
    if learners is None:
        learners = ["xgboost", "lightgbm", "rf"]

    results = []
    for learner in learners:
        try:
            _, summary = run_doubleml_plr(
                data=data,
                treatment=treatment,
                outcome=outcome,
                confounders=confounders,
                learner_name=learner,
            )
            results.append(summary)
        except Exception as e:
            print(f"Failed for learner {learner}: {e}")

    return pd.DataFrame(results)


def heterogeneous_effects_by_group(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    group_var: str = "income",
    learner_name: str = "xgboost",
) -> pd.DataFrame:
    """Estimate ATE separately within groups (simple heterogeneity analysis)."""
    groups = data[group_var].dropna().unique()
    results = []

    for g in sorted(groups):
        subset = data.loc[data[group_var] == g].copy()
        if len(subset) < 100:
            print(f"Skipping group {g} (too few observations: {len(subset)})")
            continue

        print(f"\n--- Group: {g} (N={len(subset)}) ---")
        try:
            _, summary = run_doubleml_plr(
                data=subset,
                treatment=treatment,
                outcome=outcome,
                confounders=confounders,
                learner_name=learner_name,
            )
            summary["group"] = g
            summary["group_var"] = group_var
            results.append(summary)
        except Exception as e:
            print(f"Failed for group {g}: {e}")

    return pd.DataFrame(results)


def run_causal_forest(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    n_estimators: int = 500,
    random_state: int = 42,
) -> Dict:
    """
    Estimate heterogeneous treatment effects using a Causal Forest
    (via econml.grf.CausalForest).

    Returns a dictionary with the fitted model and summary statistics
    of the estimated individual treatment effects.
    """
    try:
        from econml.grf import CausalForest
    except ImportError:
        raise ImportError(
            "econml is required for Causal Forest. Install with: pip install econml"
        )

    print("\nFitting Causal Forest for heterogeneous treatment effects ...")

    Y = data[outcome].values
    T = data[treatment].values
    X = data[confounders].values

    cf = CausalForest(
        n_estimators=n_estimators,
        min_samples_leaf=20,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    cf.fit(X=X, T=T, y=Y)

    # Estimated individual treatment effects
    ite = cf.predict(X).flatten()

    summary = {
        "mean_ite": float(np.mean(ite)),
        "median_ite": float(np.median(ite)),
        "std_ite": float(np.std(ite)),
        "min_ite": float(np.min(ite)),
        "max_ite": float(np.max(ite)),
        "n_obs": len(ite),
        "method": "CausalForest (econml.grf)",
    }

    print("=== Causal Forest Summary (Individual Treatment Effects) ===")
    print(f"Mean ITE:    {summary['mean_ite']:.4f}")
    print(f"Median ITE:  {summary['median_ite']:.4f}")
    print(f"Std ITE:     {summary['std_ite']:.4f}")
    print(f"Range:       [{summary['min_ite']:.4f}, {summary['max_ite']:.4f}]")

    return {"model": cf, "ite": ite, "summary": summary}
