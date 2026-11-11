"""
Data preparation for GHED-based causal analysis.

Loads the WHO Global Health Expenditure Database and constructs
the analysis sample for Double Machine Learning.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Core variables for the causal analysis
TREATMENT = "gghed_che"          # GGHE-D as % of CHE
OUTCOME = "oops_che"             # OOP as % of CHE

# Recommended confounders (available in standard GHED extracts)
BASE_CONFOUNDERS = [
    "ext_che",                   # External expenditure as % of CHE
    "che_gdp",                   # CHE as % of GDP
    "che_pc_usd",                # CHE per capita (USD)
    "gghed_gdp",                 # GGHE-D as % of GDP
]

# Categorical / grouping variables
GROUP_VARS = ["region", "income", "year"]


def load_ghed(
    data_path: str | Path,
    sheet_name: str = "Data",
    usecols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load GHED data from Excel file.

    Parameters
    ----------
    data_path : path to GHED_data.XLSX
    sheet_name : sheet to read (default: 'Data')
    usecols : optional list of columns to load (recommended for large file)

    Returns
    -------
    pd.DataFrame
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"GHED file not found at {data_path}. "
            "Please place GHED_data.XLSX in the data/ folder."
        )

    print(f"Loading GHED data from {data_path} ...")
    df = pd.read_excel(data_path, sheet_name=sheet_name, usecols=usecols)
    print(f"Loaded shape: {df.shape}")
    return df


def prepare_analysis_sample(
    df: pd.DataFrame,
    min_year: int = 2000,
    max_year: int = 2023,
    drop_missing: bool = True,
) -> Tuple[pd.DataFrame, list]:
    """
    Construct the analysis sample for DoubleML.

    - Restricts years
    - Selects relevant columns
    - Creates basic derived variables
    - Handles missing values

    Returns
    -------
    sample : cleaned DataFrame
    feature_cols : list of confounder column names
    """
    # Basic filters
    sample = df.copy()
    sample = sample.loc[sample["year"].between(min_year, max_year)].copy()

    # Ensure key columns exist
    required = ["code", "year", TREATMENT, OUTCOME] + GROUP_VARS
    missing = [c for c in required if c not in sample.columns]
    if missing:
        raise ValueError(f"Missing required columns in GHED extract: {missing}")

    # Select confounders that are actually present
    feature_cols = [c for c in BASE_CONFOUNDERS if c in sample.columns]

    # Simple log transformations for skewed continuous variables
    if "che_pc_usd" in sample.columns:
        sample["log_che_pc"] = np.log1p(sample["che_pc_usd"])
        if "log_che_pc" not in feature_cols:
            feature_cols.append("log_che_pc")

    # Keep only necessary columns
    keep_cols = (
        ["location", "code", "region", "income", "year"]
        + [TREATMENT, OUTCOME]
        + feature_cols
    )
    keep_cols = list(dict.fromkeys(keep_cols))  # preserve order, drop dups
    sample = sample[[c for c in keep_cols if c in sample.columns]].copy()

    # Drop rows with missing treatment or outcome
    if drop_missing:
        before = len(sample)
        sample = sample.dropna(subset=[TREATMENT, OUTCOME])
        print(f"Dropped {before - len(sample)} rows with missing treatment/outcome.")

    # Optional: drop rows with too many missing confounders
    sample = sample.dropna(subset=feature_cols, how="any")

    print(f"Final analysis sample: {sample.shape[0]} observations "
          f"from {sample['code'].nunique()} countries "
          f"({sample['year'].min()}–{sample['year'].max()})")

    return sample, feature_cols


def create_dummy_features(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Create dummy variables for region and income group.
    Useful when using linear models inside DoubleML.
    """
    sample = sample.copy()
    if "region" in sample.columns:
        region_dummies = pd.get_dummies(sample["region"], prefix="region", drop_first=True)
        sample = pd.concat([sample, region_dummies], axis=1)
    if "income" in sample.columns:
        income_dummies = pd.get_dummies(sample["income"], prefix="income", drop_first=True)
        sample = pd.concat([sample, income_dummies], axis=1)
    return sample


if __name__ == "__main__":
    # Quick test (requires data file)
    path = Path(__file__).resolve().parents[1] / "data" / "GHED_data.XLSX"
    if path.exists():
        raw = load_ghed(path, usecols=["location", "code", "region", "income", "year",
                                       "gghed_che", "oops_che", "ext_che", "che_gdp",
                                       "che_pc_usd", "gghed_gdp"])
        sample, features = prepare_analysis_sample(raw)
        print(sample.head())
        print("Features:", features)
    else:
        print("Data file not found – skipping live test.")
