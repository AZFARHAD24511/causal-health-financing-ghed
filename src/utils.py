"""
Utility functions for plotting and reporting.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_summary_table(summary_df: pd.DataFrame, path: str | Path) -> None:
    """Save results table to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(path, index=False)
    print(f"Saved summary table → {path}")


def plot_ate_comparison(
    results_df: pd.DataFrame,
    title: str = "Average Treatment Effect by Learner",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Simple bar plot of ATEs with confidence intervals.
    Expects columns: learner, ate, ci_lower, ci_upper
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(results_df))
    ax.bar(x, results_df["ate"], color="steelblue", alpha=0.8)
    ax.errorbar(
        x,
        results_df["ate"],
        yerr=[
            results_df["ate"] - results_df["ci_lower"],
            results_df["ci_upper"] - results_df["ate"],
        ],
        fmt="none",
        ecolor="black",
        capsize=5,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df["learner"], rotation=0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Estimated ATE (percentage points)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {save_path}")
    plt.close()


def plot_heterogeneity(
    het_df: pd.DataFrame,
    title: str = "Heterogeneous Treatment Effects",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot ATEs across groups."""
    fig, ax = plt.subplots(figsize=(9, 5))

    order = het_df.sort_values("ate")["group"]
    sns.pointplot(
        data=het_df,
        x="group",
        y="ate",
        order=order,
        join=False,
        ax=ax,
        color="darkorange",
    )

    # Add approximate error bars if CI available
    if "ci_lower" in het_df.columns and "ci_upper" in het_df.columns:
        for i, row in het_df.iterrows():
            ax.plot(
                [row["group"], row["group"]],
                [row["ci_lower"], row["ci_upper"]],
                color="gray",
                linewidth=2,
            )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Estimated ATE (percentage points)")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {save_path}")
    plt.close()
