"""
pseudoabsence_sensitivity.py

Sensitivity analysis for separate niche models under different
pseudo-absence/background strategies.

IMPORTANT:
This script uses only the currently available occurrence dataset.
So the "random background" here is NOT a true full-network background.
It is an alternative random background drawn from non-focal occurrence records.

Strategies compared:
A) other_species_same_status   (current/default)
B) random_other_species_all    (alternative)

For each species and for each range (native / invasive), we compare:
- CV accuracy and ROC AUC
- top split variables
- top variable-type composition
- overlap in top variables between strategies

Usage:
    python src/pseudoabsence_sensitivity.py --input data/raw/combined_data_true_master.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_loader import (
    load_geotraits, load_config, get_env_variables,
    COL_SPECIES, COL_STATUS,
)
from species_selector import (
    apply_quality_filters, STUDY_SPECIES, STATUS_NATIVE, STATUS_ALIEN,
)
from data_preparation import (
    handle_missing_values, remove_constant_variables,
    remove_highly_correlated,
)
from variable_glossary import load_glossary, translate


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def classify_variable(var_name: str) -> str:
    if "CLI" in var_name:
        return "Climate"
    elif "TOP" in var_name:
        return "Topography"
    elif "SOL" in var_name:
        return "Soil"
    elif "LAC" in var_name:
        return "Land Cover"
    return "Other"


def sample_background(
    df_all: pd.DataFrame,
    target_species: str,
    status: str,
    strategy: str,
    n_presence: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Background strategies:

    A) other_species_same_status:
       all non-focal species with the same status as focal records

    B) random_other_species_all:
       all non-focal species regardless of status, random sample
    """
    rng = np.random.default_rng(random_state)

    non_focal = df_all[df_all[COL_SPECIES] != target_species].copy()

    if strategy == "other_species_same_status":
        pool = non_focal[non_focal[COL_STATUS] == status].copy()

    elif strategy == "random_other_species_all":
        pool = non_focal.copy()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if len(pool) == 0:
        return pd.DataFrame()

    n_sample = min(n_presence, len(pool))
    idx = rng.choice(pool.index.values, size=n_sample, replace=False)
    return pool.loc[idx].copy()


def prepare_presence_background(
    presence: pd.DataFrame,
    background: pd.DataFrame,
    env_vars: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Combine presence/background and clean predictors.
    """
    presence = presence.copy()
    background = background.copy()

    presence["_target"] = 1
    background["_target"] = 0

    combined = pd.concat([presence, background], ignore_index=True)

    combined, clean_vars = handle_missing_values(
        combined,
        env_vars,
        max_missing_pct=0.30,
    )
    clean_vars = remove_constant_variables(combined, clean_vars)
    clean_vars = remove_highly_correlated(combined, clean_vars, threshold=0.98)

    return combined, clean_vars


def train_tree_model(
    combined: pd.DataFrame,
    clean_vars: list[str],
    max_depth: int = 4,
    random_state: int = 42,
) -> dict:
    """
    Train tree model and return metrics/importances.
    """
    X = combined[clean_vars].values
    y = combined["_target"].values.astype(int)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "roc_auc"],
        n_jobs=1,
    )

    clf.fit(X, y)

    imp_df = pd.DataFrame({
        "variable": clean_vars,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df["type"] = imp_df["variable"].apply(classify_variable)

    type_df = (
        imp_df.groupby("type")["importance"]
        .sum()
        .reset_index()
    )
    total = type_df["importance"].sum()
    if total > 0:
        type_df["pct"] = type_df["importance"] / total * 100
    else:
        type_df["pct"] = 0.0

    return {
        "accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "accuracy_sd": float(cv_results["test_accuracy"].std()),
        "roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
        "roc_auc_sd": float(cv_results["test_roc_auc"].std()),
        "n_features": int(len(clean_vars)),
        "importance_df": imp_df,
        "type_df": type_df,
    }


def compare_top_variables(
    imp_a: pd.DataFrame,
    imp_b: pd.DataFrame,
    k: int = 10,
) -> dict:
    top_a = imp_a["variable"].head(k).tolist()
    top_b = imp_b["variable"].head(k).tolist()

    overlap = sorted(set(top_a) & set(top_b))
    return {
        "top_a": top_a,
        "top_b": top_b,
        "overlap": overlap,
        "n_overlap": len(overlap),
    }


def type_pct_dict(type_df: pd.DataFrame) -> dict:
    return {row["type"]: row["pct"] for _, row in type_df.iterrows()}


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def safe_name(species_name: str) -> str:
    return species_name.replace(" ", "_").lower()


def plot_strategy_comparison(
    species_name: str,
    range_label: str,
    type_a: pd.DataFrame,
    type_b: pd.DataFrame,
    output_dir: str = "results/figures/pseudoabsence_sensitivity",
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    order = ["Climate", "Topography", "Soil", "Land Cover"]

    def get_vals(df):
        d = {row["type"]: row["pct"] for _, row in df.iterrows()}
        return [d.get(t, 0.0) for t in order]

    vals_a = get_vals(type_a)
    vals_b = get_vals(type_b)

    x = np.arange(len(order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, vals_a, width, label="Same-status other species")
    ax.bar(x + width / 2, vals_b, width, label="Random other species (all)")

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Importance (%)")
    ax.set_title(f"{species_name} | {range_label}\nPseudo-absence sensitivity")
    ax.legend()
    plt.tight_layout()

    path = out / f"{safe_name(species_name)}_{range_label.lower()}_strategy_compare.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------

def analyze_species_range(
    df_filtered: pd.DataFrame,
    species_name: str,
    status: str,
    env_vars: list[str],
    glossary: dict,
    max_depth: int = 4,
) -> dict:
    """
    Compare the 2 background strategies for one species and one range.
    """
    presence = df_filtered[
        (df_filtered[COL_SPECIES] == species_name) &
        (df_filtered[COL_STATUS] == status)
    ].copy()

    range_label = "native" if status == STATUS_NATIVE else "invasive"

    if len(presence) < 10:
        raise ValueError(
            f"{species_name} | {range_label}: too few presence records ({len(presence)})"
        )

    bg_a = sample_background(
        df_filtered, species_name, status,
        strategy="other_species_same_status",
        n_presence=len(presence),
        random_state=42,
    )
    bg_b = sample_background(
        df_filtered, species_name, status,
        strategy="random_other_species_all",
        n_presence=len(presence),
        random_state=42,
    )

    combined_a, vars_a = prepare_presence_background(presence, bg_a, env_vars)
    combined_b, vars_b = prepare_presence_background(presence, bg_b, env_vars)

    res_a = train_tree_model(combined_a, vars_a, max_depth=max_depth, random_state=42)
    res_b = train_tree_model(combined_b, vars_b, max_depth=max_depth, random_state=42)

    top_compare = compare_top_variables(
        res_a["importance_df"],
        res_b["importance_df"],
        k=10,
    )

    fig_path = plot_strategy_comparison(
        species_name,
        range_label,
        res_a["type_df"],
        res_b["type_df"],
    )

    # translate overlapping variables for readability
    overlap_translated = [
        f"{v}: {translate(v, glossary)}"
        for v in top_compare["overlap"]
    ]

    return {
        "species": species_name,
        "range": range_label,
        "n_presence": int(len(presence)),
        "strategy_a_accuracy": res_a["accuracy_mean"],
        "strategy_a_auc": res_a["roc_auc_mean"],
        "strategy_b_accuracy": res_b["accuracy_mean"],
        "strategy_b_auc": res_b["roc_auc_mean"],
        "strategy_a_features": res_a["n_features"],
        "strategy_b_features": res_b["n_features"],
        "strategy_a_type_pct": type_pct_dict(res_a["type_df"]),
        "strategy_b_type_pct": type_pct_dict(res_b["type_df"]),
        "top10_overlap_n": top_compare["n_overlap"],
        "top10_overlap_vars": top_compare["overlap"],
        "top10_overlap_vars_readable": overlap_translated,
        "strategy_a_top10": res_a["importance_df"]["variable"].head(10).tolist(),
        "strategy_b_top10": res_b["importance_df"]["variable"].head(10).tolist(),
        "figure": fig_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pseudo-absence sensitivity analysis"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to combined_data_true_master.csv"
    )
    parser.add_argument(
        "--config", "-c", default="config/species_config.yaml",
        help="Path to species config YAML"
    )
    parser.add_argument(
        "--max-depth", type=int, default=4,
        help="Decision tree max depth (default: 4)"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)
    env_vars = get_env_variables(df_filtered)["all_env"]
    glossary = load_glossary("data/raw/S2.xlsx")

    print("\n" + "=" * 80)
    print("PSEUDO-ABSENCE SENSITIVITY")
    print("=" * 80)
    print("Strategy A = other species, same status (current)")
    print("Strategy B = random sample from all non-focal occurrence records")
    print("NOTE: this is a useful approximation with the current file,")
    print("      but it is not yet a true full-network random background.")
    print("=" * 80)

    all_rows = []

    for species in STUDY_SPECIES:
        for status in [STATUS_NATIVE, STATUS_ALIEN]:
            res = analyze_species_range(
                df_filtered=df_filtered,
                species_name=species,
                status=status,
                env_vars=env_vars,
                glossary=glossary,
                max_depth=args.max_depth,
            )
            all_rows.append(res)

            print(f"\n{species} | {res['range'].upper()}")
            print(f"  Presence n: {res['n_presence']}")
            print(
                f"  Strategy A: acc={res['strategy_a_accuracy']:.3f}, "
                f"auc={res['strategy_a_auc']:.3f}, "
                f"features={res['strategy_a_features']}"
            )
            print(
                f"  Strategy B: acc={res['strategy_b_accuracy']:.3f}, "
                f"auc={res['strategy_b_auc']:.3f}, "
                f"features={res['strategy_b_features']}"
            )

            a = res["strategy_a_type_pct"]
            b = res["strategy_b_type_pct"]

            print("  Type-level importance (A vs B):")
            for t in ["Climate", "Topography", "Soil", "Land Cover"]:
                print(
                    f"    {t:<12s} "
                    f"A={a.get(t, 0.0):5.1f}%   "
                    f"B={b.get(t, 0.0):5.1f}%"
                )

            print(f"  Top-10 overlap between strategies: {res['top10_overlap_n']}/10")
            if res["top10_overlap_vars"]:
                for v in res["top10_overlap_vars_readable"]:
                    print(f"    - {v}")

    out_dir = Path("results/pseudoabsence_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    # save json
    with open(out_dir / "pseudoabsence_sensitivity_summary.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    # save flat csv
    flat_rows = []
    for r in all_rows:
        flat_rows.append({
            "species": r["species"],
            "range": r["range"],
            "n_presence": r["n_presence"],
            "strategy_a_accuracy": r["strategy_a_accuracy"],
            "strategy_a_auc": r["strategy_a_auc"],
            "strategy_b_accuracy": r["strategy_b_accuracy"],
            "strategy_b_auc": r["strategy_b_auc"],
            "strategy_a_features": r["strategy_a_features"],
            "strategy_b_features": r["strategy_b_features"],
            "strategy_a_climate": r["strategy_a_type_pct"].get("Climate", 0.0),
            "strategy_a_topography": r["strategy_a_type_pct"].get("Topography", 0.0),
            "strategy_a_soil": r["strategy_a_type_pct"].get("Soil", 0.0),
            "strategy_a_landcover": r["strategy_a_type_pct"].get("Land Cover", 0.0),
            "strategy_b_climate": r["strategy_b_type_pct"].get("Climate", 0.0),
            "strategy_b_topography": r["strategy_b_type_pct"].get("Topography", 0.0),
            "strategy_b_soil": r["strategy_b_type_pct"].get("Soil", 0.0),
            "strategy_b_landcover": r["strategy_b_type_pct"].get("Land Cover", 0.0),
            "top10_overlap_n": r["top10_overlap_n"],
        })

    pd.DataFrame(flat_rows).to_csv(
        out_dir / "pseudoabsence_sensitivity_summary.csv",
        index=False,
    )

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()