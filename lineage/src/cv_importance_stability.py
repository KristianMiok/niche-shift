"""
cv_importance_stability.py

Variable-importance stability across CV folds.

For each focal species:
- use the same native-vs-invasive random forest setup
- run 5-fold stratified CV
- store feature importances from each fold
- summarize:
    1. importance by variable type (Climate / Topography / Soil / Land Cover)
    2. top individual variables by mean importance across folds
- save summary tables and stability figures

Usage:
    python src/cv_importance_stability.py --input data/raw/combined_data_true_master.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from data_loader import (
    load_geotraits, load_config, get_env_variables,
    COL_SPECIES, COL_STATUS,
)
from species_selector import (
    apply_quality_filters, STATUS_NATIVE, STATUS_ALIEN, STUDY_SPECIES,
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


def prepare_species_xy(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Build cleaned X/y for one species.
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

    X = combined[clean_vars].values
    y = combined["range_label"].values.astype(int)

    return X, y, clean_vars, combined


def summarize_type_importance(
    feature_importance_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert feature-level importances to type-level importances.
    """
    df = feature_importance_df.copy()
    df["type"] = df["variable"].apply(classify_variable)

    out = (
        df.groupby(["fold", "type"])["importance"]
        .sum()
        .reset_index()
    )
    return out


def top_variable_summary(
    feature_importance_df: pd.DataFrame,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Mean/stability summary for individual variables across folds.
    """
    grouped = (
        feature_importance_df.groupby("variable")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    grouped["cv"] = grouped["std"] / grouped["mean"].replace(0, np.nan)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped.head(n_top)


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def safe_name(species_name: str) -> str:
    return species_name.replace(" ", "_").lower()


def plot_type_stability(
    type_summary: pd.DataFrame,
    species_name: str,
    output_dir: str = "results/figures/cv_stability",
) -> str:
    """
    Bar chart: mean importance by variable type with SD error bars.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    order = ["Climate", "Topography", "Soil", "Land Cover"]
    df = type_summary.copy()
    df["type"] = pd.Categorical(df["type"], categories=order, ordered=True)
    df = df.sort_values("type")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        df["type"],
        df["mean_importance"],
        yerr=df["sd_importance"],
        capsize=5,
    )
    ax.set_ylabel("Importance across folds")
    ax.set_title(f"{species_name}\nVariable-type importance stability (5-fold CV)")
    plt.tight_layout()

    path = out / f"{safe_name(species_name)}_type_stability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_top_variable_stability(
    top_vars: pd.DataFrame,
    species_name: str,
    glossary: dict,
    output_dir: str = "results/figures/cv_stability",
) -> str:
    """
    Horizontal bar chart of top variables with SD error bars.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = top_vars.copy().iloc[::-1]
    labels = [
        f"{v}: {translate(v, glossary)[:45]}"
        for v in df["variable"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(df))))
    ax.barh(
        range(len(df)),
        df["mean"],
        xerr=df["std"],
        capsize=4,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean feature importance across folds")
    ax.set_title(f"{species_name}\nTop variable stability (5-fold CV)")
    plt.tight_layout()

    path = out / f"{safe_name(species_name)}_top_variable_stability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_cross_species_type_stability(
    cross_species_df: pd.DataFrame,
    output_dir: str = "results/figures/cv_stability",
) -> str:
    """
    Cross-species summary: mean climate/topography importance with SD.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    species_order = cross_species_df["species"].unique().tolist()
    short_labels = [
        s.split(" ")[0][0] + ". " + s.split(" ")[1]
        for s in species_order
    ]

    climate = cross_species_df[cross_species_df["type"] == "Climate"].copy()
    topo = cross_species_df[cross_species_df["type"] == "Topography"].copy()

    x = np.arange(len(species_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        climate["mean_importance"],
        width,
        yerr=climate["sd_importance"],
        capsize=4,
        label="Climate",
    )
    ax.bar(
        x + width / 2,
        topo["mean_importance"],
        width,
        yerr=topo["sd_importance"],
        capsize=4,
        label="Topography",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, style="italic")
    ax.set_ylabel("Mean importance across folds")
    ax.set_title("Cross-species CV stability: Climate vs Topography")
    ax.legend()
    plt.tight_layout()

    path = out / "cross_species_climate_topography_stability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------

def analyze_species_cv_stability(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list[str],
    glossary: dict,
    n_estimators: int = 500,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run CV and store fold-wise RF feature importances.
    """
    X, y, feature_names, combined = prepare_species_xy(
        df_filtered, species_name, env_vars
    )

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_feature_rows = []
    fold_perf_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)

        fold_perf_rows.append({
            "species": species_name,
            "fold": fold_idx,
            "accuracy": float(acc),
        })

        for var, imp in zip(feature_names, clf.feature_importances_):
            fold_feature_rows.append({
                "species": species_name,
                "fold": fold_idx,
                "variable": var,
                "importance": float(imp),
                "type": classify_variable(var),
            })

    feature_df = pd.DataFrame(fold_feature_rows)
    perf_df = pd.DataFrame(fold_perf_rows)

    # --- type-level summary ---
    type_fold_df = summarize_type_importance(feature_df)

    type_summary = (
        type_fold_df.groupby("type")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "mean_importance",
            "std": "sd_importance",
        })
    )

    # percentages can help interpretation
    total_mean = type_summary["mean_importance"].sum()
    if total_mean > 0:
        type_summary["mean_pct"] = type_summary["mean_importance"] / total_mean * 100
        type_summary["sd_pct"] = type_summary["sd_importance"] / total_mean * 100
    else:
        type_summary["mean_pct"] = 0.0
        type_summary["sd_pct"] = 0.0

    # --- top variable summary ---
    top_vars = top_variable_summary(feature_df, n_top=10)

    # --- performance summary ---
    perf_summary = {
        "accuracy_mean": float(perf_df["accuracy"].mean()),
        "accuracy_sd": float(perf_df["accuracy"].std(ddof=1)),
    }

    # plots
    type_plot = plot_type_stability(type_summary, species_name)
    top_plot = plot_top_variable_stability(top_vars, species_name, glossary)

    return {
        "species": species_name,
        "n_samples": int(len(y)),
        "n_native": int((y == 0).sum()),
        "n_invasive": int((y == 1).sum()),
        "n_features": int(len(feature_names)),
        "performance_by_fold": perf_df,
        "feature_importance_by_fold": feature_df,
        "type_summary": type_summary,
        "top_variables": top_vars,
        "performance_summary": perf_summary,
        "type_plot": type_plot,
        "top_plot": top_plot,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Variable-importance stability across CV folds"
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
        "--n-estimators", type=int, default=500,
        help="Number of RF trees (default: 500)"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)

    env_info = get_env_variables(df_filtered)
    env_vars = env_info["all_env"]
    glossary = load_glossary("data/raw/S2.xlsx")

    output_tables = Path("results/tables/cv_stability")
    output_tables.mkdir(parents=True, exist_ok=True)

    all_results = []
    cross_species_rows = []

    print("\n" + "=" * 80)
    print("CV IMPORTANCE STABILITY")
    print("=" * 80)

    for species in STUDY_SPECIES:
        print(f"\nAnalyzing {species}...")
        res = analyze_species_cv_stability(
            df_filtered=df_filtered,
            species_name=species,
            env_vars=env_vars,
            glossary=glossary,
            n_estimators=args.n_estimators,
        )

        print(
            f"  Samples: {res['n_samples']} "
            f"({res['n_native']} native, {res['n_invasive']} invasive)"
        )
        print(f"  Features: {res['n_features']}")
        print(
            f"  CV accuracy: {res['performance_summary']['accuracy_mean']:.3f} "
            f"± {res['performance_summary']['accuracy_sd']:.3f}"
        )

        print("  Type-level stability:")
        tmp = res["type_summary"].copy()
        order = ["Climate", "Topography", "Soil", "Land Cover"]
        tmp["type"] = pd.Categorical(tmp["type"], categories=order, ordered=True)
        tmp = tmp.sort_values("type")
        for _, row in tmp.iterrows():
            print(
                f"    {row['type']:<12s} "
                f"{row['mean_pct']:.1f}% ± {row['sd_pct']:.1f}%"
            )
            cross_species_rows.append({
                "species": species,
                "type": row["type"],
                "mean_importance": row["mean_pct"],
                "sd_importance": row["sd_pct"],
            })

        print("  Top variables across folds:")
        for _, row in res["top_variables"].iterrows():
            print(
                f"    {int(row['rank']):2d}. {row['variable']:<15s} "
                f"mean={row['mean']:.4f}, sd={row['std']:.4f}, cv={row['cv']:.3f}"
            )

        # save species tables
        sname = safe_name(species)
        res["performance_by_fold"].to_csv(
            output_tables / f"{sname}_performance_by_fold.csv", index=False
        )
        res["feature_importance_by_fold"].to_csv(
            output_tables / f"{sname}_feature_importance_by_fold.csv", index=False
        )
        res["type_summary"].to_csv(
            output_tables / f"{sname}_type_summary.csv", index=False
        )
        res["top_variables"].to_csv(
            output_tables / f"{sname}_top_variable_summary.csv", index=False
        )

        all_results.append({
            "species": species,
            "n_samples": res["n_samples"],
            "n_native": res["n_native"],
            "n_invasive": res["n_invasive"],
            "n_features": res["n_features"],
            "accuracy_mean": res["performance_summary"]["accuracy_mean"],
            "accuracy_sd": res["performance_summary"]["accuracy_sd"],
            "type_summary": res["type_summary"].to_dict("records"),
            "top_variables": res["top_variables"].to_dict("records"),
        })

    cross_species_df = pd.DataFrame(cross_species_rows)
    cross_species_fig = plot_cross_species_type_stability(cross_species_df)

    cross_species_df.to_csv(
        output_tables / "cross_species_type_stability.csv", index=False
    )

    with open(output_tables / "cv_stability_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("CROSS-SPECIES CLIMATE VS TOPOGRAPHY STABILITY")
    print("=" * 80)
    display_df = cross_species_df[
        cross_species_df["type"].isin(["Climate", "Topography"])
    ].copy()
    print(display_df.to_string(index=False))

    print(f"\nSaved cross-species figure: {cross_species_fig}")
    print(f"Saved summary JSON: results/tables/cv_stability/cv_stability_summary.json")


if __name__ == "__main__":
    main()