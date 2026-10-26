"""
sample_size_sensitivity.py

Sample-size sensitivity analysis.

Goal:
Test whether the main climatic signal remains visible when a well-sampled
species (Procambarus clarkii) is artificially downsampled to the same
native sample size as Pacifastacus leniusculus (117 native records).

Design:
- focal benchmark species: Procambarus clarkii
- repeatedly subsample native records down to n = 117
- keep invasive records fixed
- fit the same RF classifier each repetition
- summarize:
    1. performance
    2. type-level importances
    3. top variables
    4. overlap with full-data top variables

Usage:
    python src/sample_size_sensitivity.py --input data/raw/combined_data_true_master.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_loader import (
    load_geotraits, load_config, get_env_variables,
    COL_SPECIES, COL_STATUS,
)
from species_selector import (
    apply_quality_filters, STATUS_NATIVE, STATUS_ALIEN,
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


def prepare_species_data(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare cleaned data for one species.
    """
    sp = df_filtered[df_filtered[COL_SPECIES] == species_name].copy()

    combined, clean_vars = handle_missing_values(
        sp, env_vars, max_missing_pct=0.30
    )
    clean_vars = remove_constant_variables(combined, clean_vars)
    clean_vars = remove_highly_correlated(combined, clean_vars, threshold=0.98)

    combined["range_label"] = combined[COL_STATUS].map(
        {STATUS_NATIVE: 0, STATUS_ALIEN: 1}
    )

    return combined, clean_vars


def fit_rf_summary(
    df_species: pd.DataFrame,
    feature_names: list[str],
    n_estimators: int = 500,
    random_state: int = 42,
) -> dict:
    """
    Fit RF and return CV scores + feature importance summaries.
    """
    X = df_species[feature_names].values
    y = df_species["range_label"].values.astype(int)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=1,
    )

    clf.fit(X, y)

    imp_df = pd.DataFrame({
        "variable": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df["type"] = imp_df["variable"].apply(classify_variable)

    by_type = (
        imp_df.groupby("type")["importance"]
        .sum()
        .reset_index()
    )

    type_dict = {row["type"]: row["importance"] for _, row in by_type.iterrows()}
    total = sum(type_dict.values())

    type_pct = {
        k: (v / total * 100 if total > 0 else 0.0)
        for k, v in type_dict.items()
    }

    return {
        "accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "accuracy_sd": float(cv_results["test_accuracy"].std()),
        "f1_mean": float(cv_results["test_f1"].mean()),
        "f1_sd": float(cv_results["test_f1"].std()),
        "roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
        "roc_auc_sd": float(cv_results["test_roc_auc"].std()),
        "feature_importance_df": imp_df,
        "type_pct": type_pct,
    }


def top_k_variables(imp_df: pd.DataFrame, k: int = 10) -> list[str]:
    return imp_df["variable"].head(k).tolist()


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_type_distribution(
    type_summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    order = ["Climate", "Topography", "Soil", "Land Cover"]
    df = type_summary_df.copy()
    df["type"] = pd.Categorical(df["type"], categories=order, ordered=True)
    df = df.sort_values("type")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        df["type"],
        df["mean_pct"],
        yerr=df["sd_pct"],
        capsize=5,
    )
    ax.set_ylabel("Importance (%)")
    ax.set_title("P. clarkii downsampled native records (n = 117)\nType-level RF importance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_variable_frequency(
    freq_df: pd.DataFrame,
    glossary: dict,
    output_path: str,
    top_n: int = 12,
) -> None:
    df = freq_df.head(top_n).iloc[::-1].copy()
    labels = [
        f"{v}: {translate(v, glossary)[:45]}"
        for v in df["variable"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(df))))
    ax.barh(range(len(df)), df["count_in_top10"])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of repetitions appearing in top 10")
    ax.set_title("Top-variable recurrence across downsampling repetitions")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample-size sensitivity analysis"
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
        "--reps", type=int, default=100,
        help="Number of downsampling repetitions (default: 100)"
    )
    parser.add_argument(
        "--target-native-n", type=int, default=117,
        help="Target native sample size (default: 117)"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=500,
        help="Number of RF trees (default: 500)"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)
    env_vars = get_env_variables(df_filtered)["all_env"]
    glossary = load_glossary("data/raw/S2.xlsx")

    species_name = "Procambarus clarkii"
    combined, clean_vars = prepare_species_data(df_filtered, species_name, env_vars)

    native_full = combined[combined[COL_STATUS] == STATUS_NATIVE].copy()
    invasive_full = combined[combined[COL_STATUS] == STATUS_ALIEN].copy()

    print("\n" + "=" * 80)
    print("SAMPLE-SIZE SENSITIVITY")
    print("=" * 80)
    print(f"Benchmark species: {species_name}")
    print(f"Full native n: {len(native_full)}")
    print(f"Full invasive n: {len(invasive_full)}")
    print(f"Target native n: {args.target_native_n}")
    print(f"Repetitions: {args.reps}")
    print(f"Features: {len(clean_vars)}")

    if len(native_full) < args.target_native_n:
        raise ValueError(
            f"Not enough native records: {len(native_full)} < {args.target_native_n}"
        )

    # --- full-data benchmark ---
    full_res = fit_rf_summary(
        combined,
        clean_vars,
        n_estimators=args.n_estimators,
        random_state=42,
    )
    full_top10 = top_k_variables(full_res["feature_importance_df"], 10)

    print("\nFULL-DATA BENCHMARK")
    print(
        f"  Accuracy: {full_res['accuracy_mean']:.3f} ± {full_res['accuracy_sd']:.3f}"
    )
    print(
        f"  ROC AUC:  {full_res['roc_auc_mean']:.3f} ± {full_res['roc_auc_sd']:.3f}"
    )
    print("  Type-level importance:")
    for t in ["Climate", "Topography", "Soil", "Land Cover"]:
        print(f"    {t:<12s} {full_res['type_pct'].get(t, 0.0):.1f}%")
    print("  Full-data top 10 variables:")
    for i, var in enumerate(full_top10, start=1):
        print(f"    {i:2d}. {var:<15s} {translate(var, glossary)}")

    # --- repeated downsampling ---
    rep_rows = []
    feature_freq = {}
    rank_rows = []

    for rep in range(1, args.reps + 1):
        native_sub = native_full.sample(
            n=args.target_native_n,
            replace=False,
            random_state=1000 + rep,
        )

        rep_df = pd.concat([native_sub, invasive_full], ignore_index=True)

        rep_res = fit_rf_summary(
            rep_df,
            clean_vars,
            n_estimators=args.n_estimators,
            random_state=2000 + rep,
        )

        top10 = top_k_variables(rep_res["feature_importance_df"], 10)

        # store repetition summary
        row = {
            "rep": rep,
            "accuracy_mean": rep_res["accuracy_mean"],
            "roc_auc_mean": rep_res["roc_auc_mean"],
            "climate_pct": rep_res["type_pct"].get("Climate", 0.0),
            "topography_pct": rep_res["type_pct"].get("Topography", 0.0),
            "soil_pct": rep_res["type_pct"].get("Soil", 0.0),
            "land_cover_pct": rep_res["type_pct"].get("Land Cover", 0.0),
            "top10_overlap_with_full": len(set(top10) & set(full_top10)),
        }
        rep_rows.append(row)

        # feature frequency in top10
        for rank, var in enumerate(top10, start=1):
            feature_freq[var] = feature_freq.get(var, 0) + 1
            rank_rows.append({
                "rep": rep,
                "rank": rank,
                "variable": var,
                "description": translate(var, glossary),
            })

    rep_df = pd.DataFrame(rep_rows)
    rank_df = pd.DataFrame(rank_rows)

    type_summary_df = pd.DataFrame({
        "type": ["Climate", "Topography", "Soil", "Land Cover"],
        "mean_pct": [
            rep_df["climate_pct"].mean(),
            rep_df["topography_pct"].mean(),
            rep_df["soil_pct"].mean(),
            rep_df["land_cover_pct"].mean(),
        ],
        "sd_pct": [
            rep_df["climate_pct"].std(ddof=1),
            rep_df["topography_pct"].std(ddof=1),
            rep_df["soil_pct"].std(ddof=1),
            rep_df["land_cover_pct"].std(ddof=1),
        ],
    })

    freq_df = pd.DataFrame(
        [{"variable": k, "count_in_top10": v} for k, v in feature_freq.items()]
    ).sort_values(["count_in_top10", "variable"], ascending=[False, True]).reset_index(drop=True)

    # -----------------------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DOWNSAMPLED RESULTS ACROSS REPETITIONS")
    print("=" * 80)
    print(
        f"Accuracy: {rep_df['accuracy_mean'].mean():.3f} ± {rep_df['accuracy_mean'].std(ddof=1):.3f}"
    )
    print(
        f"ROC AUC:  {rep_df['roc_auc_mean'].mean():.3f} ± {rep_df['roc_auc_mean'].std(ddof=1):.3f}"
    )

    print("\nType-level importance across repetitions:")
    for _, row in type_summary_df.iterrows():
        print(f"  {row['type']:<12s} {row['mean_pct']:.1f}% ± {row['sd_pct']:.1f}%")

    print(
        f"\nMean overlap with full-data top 10: "
        f"{rep_df['top10_overlap_with_full'].mean():.2f} / 10"
    )

    print("\nMost recurrent top-10 variables across repetitions:")
    for i, (_, row) in enumerate(freq_df.head(12).iterrows(), start=1):
        print(
            f"  {i:2d}. {row['variable']:<15s} "
            f"appeared in top 10 = {int(row['count_in_top10'])}/{args.reps}"
        )

    # -----------------------------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------------------------
    out_dir = Path("results/sample_size_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_df.to_csv(out_dir / "downsampled_repetition_summary.csv", index=False)
    rank_df.to_csv(out_dir / "downsampled_top10_by_rep.csv", index=False)
    type_summary_df.to_csv(out_dir / "downsampled_type_summary.csv", index=False)
    freq_df.to_csv(out_dir / "downsampled_top_variable_frequency.csv", index=False)

    with open(out_dir / "sample_size_sensitivity_summary.json", "w") as f:
        json.dump({
            "benchmark_species": species_name,
            "target_native_n": args.target_native_n,
            "repetitions": args.reps,
            "full_data_type_pct": full_res["type_pct"],
            "full_data_top10": full_top10,
            "downsampled_accuracy_mean": float(rep_df["accuracy_mean"].mean()),
            "downsampled_accuracy_sd": float(rep_df["accuracy_mean"].std(ddof=1)),
            "downsampled_auc_mean": float(rep_df["roc_auc_mean"].mean()),
            "downsampled_auc_sd": float(rep_df["roc_auc_mean"].std(ddof=1)),
            "mean_top10_overlap_with_full": float(rep_df["top10_overlap_with_full"].mean()),
        }, f, indent=2)

    plot_type_distribution(
        type_summary_df,
        str(out_dir / "downsampled_type_stability.png"),
    )
    plot_top_variable_frequency(
        freq_df,
        glossary,
        str(out_dir / "downsampled_top_variable_frequency.png"),
    )

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()