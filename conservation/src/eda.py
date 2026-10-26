"""
eda.py
Exploratory data analysis for the niche shift analysis.

Generates summary statistics and visualizations comparing environmental
distributions between native and invasive ranges for selected species.

Usage:
    python src/eda.py --species "Procambarus clarkii" \
                      --data data/processed/procambarus_clarkii_combined.csv \
                      --vars data/processed/procambarus_clarkii_env_vars.txt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def load_species_data(
    data_path: str, vars_path: str
) -> tuple[pd.DataFrame, list]:
    """Load a prepared species dataset and its variable list."""
    df = pd.read_csv(data_path, low_memory=False)
    env_vars = pd.read_csv(vars_path, header=None)[0].tolist()
    return df, env_vars


def summary_statistics(
    df: pd.DataFrame, env_vars: list, species_name: str
) -> pd.DataFrame:
    """
    Compare each environmental variable between native and invasive ranges.

    Uses Mann-Whitney U test with Benjamini-Hochberg FDR correction.

    Returns DataFrame with one row per variable, sorted by p-value.
    """
    native = df[df["range_label"] == "native"]
    invasive = df[df["range_label"] == "invasive"]

    rows = []
    for var in env_vars:
        nat_vals = native[var].dropna()
        inv_vals = invasive[var].dropna()

        if len(nat_vals) < 3 or len(inv_vals) < 3:
            continue

        try:
            u_stat, p_val = stats.mannwhitneyu(
                nat_vals, inv_vals, alternative="two-sided"
            )
        except ValueError:
            u_stat, p_val = np.nan, np.nan

        # Effect size: rank-biserial correlation
        n1, n2 = len(nat_vals), len(inv_vals)
        r_effect = 1 - (2 * u_stat) / (n1 * n2) if not np.isnan(u_stat) else np.nan

        rows.append({
            "variable": var,
            "native_mean": nat_vals.mean(),
            "native_std": nat_vals.std(),
            "native_median": nat_vals.median(),
            "invasive_mean": inv_vals.mean(),
            "invasive_std": inv_vals.std(),
            "invasive_median": inv_vals.median(),
            "mean_diff": inv_vals.mean() - nat_vals.mean(),
            "mean_diff_pct": (
                (inv_vals.mean() - nat_vals.mean()) / abs(nat_vals.mean()) * 100
                if abs(nat_vals.mean()) > 1e-10 else np.nan
            ),
            "mann_whitney_U": u_stat,
            "p_value": p_val,
            "effect_size_r": r_effect,
            "significant_005": p_val < 0.05 if not np.isnan(p_val) else False,
        })

    result = pd.DataFrame(rows)

    # Multiple testing correction (Benjamini-Hochberg)
    if not result.empty and result["p_value"].notna().any():
        try:
            from statsmodels.stats.multitest import multipletests
            valid_mask = result["p_value"].notna()
            reject, p_adj, _, _ = multipletests(
                result.loc[valid_mask, "p_value"], method="fdr_bh"
            )
            result.loc[valid_mask, "p_adjusted"] = p_adj
            result.loc[valid_mask, "significant_adjusted"] = reject
        except ImportError:
            print("  Note: install statsmodels for FDR correction")
            result["p_adjusted"] = np.nan
            result["significant_adjusted"] = False

    return result.sort_values("p_value").reset_index(drop=True)


def plot_top_variables(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    species_name: str,
    n_top: int = 12,
    output_dir: str = "results/figures",
) -> None:
    """Plot distributions of the most different variables between ranges."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    top_vars = (
        stats_df[stats_df["significant_005"]]
        .sort_values("effect_size_r", key=abs, ascending=False)
        .head(n_top)
    )

    if top_vars.empty:
        print("  No significantly different variables to plot.")
        return

    n_vars = len(top_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (_, row) in enumerate(top_vars.iterrows()):
        ax = axes[i]
        var = row["variable"]

        native_data = df[df["range_label"] == "native"][var].dropna()
        invasive_data = df[df["range_label"] == "invasive"][var].dropna()

        ax.hist(native_data, bins=30, alpha=0.6, label="Native",
                color="#2196F3", density=True)
        ax.hist(invasive_data, bins=30, alpha=0.6, label="Invasive",
                color="#F44336", density=True)
        ax.set_title(f"{var}\np={row['p_value']:.2e}, r={row['effect_size_r']:.2f}",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{species_name}: Top environmental differences\n"
                 f"(native vs. invasive range)", fontsize=13, y=1.02)
    plt.tight_layout()

    safe_name = species_name.replace(" ", "_").lower()
    fig_path = out / f"{safe_name}_top_variables.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def plot_record_counts(
    species_data: dict,
    output_dir: str = "results/figures",
) -> None:
    """Bar chart of record counts (native vs invasive) for all species."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    species = list(species_data.keys())
    native_counts = [species_data[sp]["n_native"] for sp in species]
    invasive_counts = [species_data[sp]["n_invasive"] for sp in species]

    x = np.arange(len(species))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(species) * 1.5), 5))
    ax.bar(x - width / 2, native_counts, width, label="Native", color="#2196F3")
    ax.bar(x + width / 2, invasive_counts, width, label="Invasive", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace(" ", "\n") for s in species],
        fontsize=9, style="italic",
    )
    ax.set_ylabel("Number of records")
    ax.set_title("Record counts per species and range")
    ax.legend()
    plt.tight_layout()

    fig_path = out / "species_record_counts.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Exploratory data analysis")
    parser.add_argument("--species", "-s", required=True, help="Species name")
    parser.add_argument("--data", "-d", required=True,
                        help="Path to combined species CSV")
    parser.add_argument("--vars", "-v", required=True,
                        help="Path to env vars list file")
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory")
    args = parser.parse_args()

    print(f"EDA for: {args.species}")
    df, env_vars = load_species_data(args.data, args.vars)
    print(f"  Records: {len(df)}, Variables: {len(env_vars)}")

    # Summary statistics
    stats_df = summary_statistics(df, env_vars, args.species)

    n_sig = stats_df["significant_005"].sum() if not stats_df.empty else 0
    print(f"  Significant variables (p<0.05): {n_sig} / {len(stats_df)}")

    if "significant_adjusted" in stats_df.columns:
        n_adj = stats_df["significant_adjusted"].sum()
        print(f"  Significant after FDR correction: {n_adj}")

    # Save statistics
    tables_dir = Path(args.output) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.species.replace(" ", "_").lower()
    stats_path = tables_dir / f"{safe_name}_env_comparison.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Statistics saved to {stats_path}")

    # Plot
    plot_top_variables(
        df, stats_df, args.species,
        output_dir=str(Path(args.output) / "figures"),
    )

    # Print top 10
    print(f"\n  Top 10 most different variables:")
    top10 = stats_df.head(10)[
        ["variable", "native_mean", "invasive_mean", "mean_diff_pct",
         "p_value", "effect_size_r"]
    ]
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
