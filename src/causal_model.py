"""
Double Machine Learning estimation for the effect of public health financing
on out-of-pocket expenditure.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

try:
    from doubleml import DoubleMLData, DoubleMLPLR
    from doubleml.utils import DMLBackend
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
) -> Tuple[DoubleMLPLR, Dict]:
    """
    Estimate the average treatment effect using DoubleML Partially Linear Regression.

    Parameters
    ----------
    data : analysis sample
    treatment : name of treatment column
    outcome : name of outcome column
    confounders : list of confounder column names
    learner_name : which ML model to use for nuisance functions
    n_folds : number of cross-fitting folds
    n_rep : number of repetitions (for stability)
    random_state : seed

    Returns
    -------
    dml_model : fitted DoubleMLPLR object
    summary : dictionary with key results
    """
    # Prepare DoubleML data object
    dml_data = DoubleMLData(
        data=data,
        y_col=outcome,
        d_cols=treatment,
        x_cols=confounders,
    )

    # Learners for outcome and treatment models
    ml_l = make_learner(learner_name, random_state)   # E[Y|X]
    ml_m = make_learner(learner_name, random_state)   # E[D|X]

    # Partially Linear Regression model
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

    # Extract summary
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
    """
    Run DoubleML with several different learners for robustness checks.
    """
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
    """
    Estimate ATE separately within groups (e.g. income groups or regions)
    as a simple form of heterogeneity analysis.
    """
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
