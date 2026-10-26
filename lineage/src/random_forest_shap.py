"""
random_forest_shap.py
Random Forest classification with SHAP values as robustness check
for the decision tree analysis.

For each species, trains a Random Forest to distinguish native from
invasive occurrences, extracts feature importances and SHAP values,
and confirms whether the climate vs. topography pattern holds.

Usage:
    python src/random_forest_shap.py --input data/raw/combined_data_true_master.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

from data_loader import (
    load_geotraits, load_config, get_env_variables,
    COL_SPECIES, COL_STATUS,
)
from species_selector import (
    apply_quality_filters, STUDY_SPECIES,
    STATUS_NATIVE, STATUS_ALIEN,
)
from data_preparation import (
    handle_missing_values, remove_constant_variables,
    remove_highly_correlated,
)
from variable_glossary import load_glossary, translate, make_label
from cross_species_comparison import classify_variable


def prepare_species_data(df_filtered, species_name, env_vars):
    """Prepare X, y, feature_names for a species."""
    sp_data = df_filtered[df_filtered[COL_SPECIES] == species_name].copy()
    native = sp_data[sp_data[COL_STATUS] == STATUS_NATIVE]
    invasive = sp_data[sp_data[COL_STATUS] == STATUS_ALIEN]
    combined = pd.concat([native, invasive], ignore_index=True)

    combined, clean_vars = handle_missing_values(combined, env_vars)
    clean_vars = remove_constant_variables(combined, clean_vars)
    clean_vars = remove_highly_correlated(combined, clean_vars, threshold=0.98)

    combined["range_label"] = combined[COL_STATUS].map(
        {STATUS_NATIVE: "native", STATUS_ALIEN: "invasive"}
    )

    X = combined[clean_vars].values
    y = (combined["range_label"] == "invasive").astype(int).values
    return X, y, clean_vars, combined


def train_and_evaluate_rf(
    X, y, feature_names,
    n_estimators=500,
    max_depth=None,
    n_folds=5,
    random_state=42,
):
    """Train RF with cross-validation, return clf and scores."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
    )

    cv_scores = {
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
        "f1_mean": cv_results["test_f1"].mean(),
        "f1_std": cv_results["test_f1"].std(),
        "roc_auc_mean": cv_results["test_roc_auc"].mean(),
        "roc_auc_std": cv_results["test_roc_auc"].std(),
    }

    # Train final model on all data
    clf.fit(X, y)

    return clf, cv_scores


def compute_shap_values(clf, X, feature_names, max_samples=2000, random_state=42):
    """Compute SHAP values using TreeExplainer."""
    import shap

    # Subsample if dataset is large
    if X.shape[0] > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        # Shape (n_samples, n_features, n_classes) — take class 1
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(sv).mean(axis=0).flatten()
    shap_importance = pd.DataFrame({
        "variable": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return shap_importance, sv, X_sample

def get_gini_importances(clf, feature_names):
    """Extract Gini importances from RF."""
    importances = pd.DataFrame({
        "variable": feature_names,
        "gini_importance": clf.feature_importances_,
    }).sort_values("gini_importance", ascending=False).reset_index(drop=True)
    return importances


def compute_importance_by_type(importances_df, importance_col):
    """Aggregate importances by variable type."""
    df = importances_df.copy()
    df["type"] = df["variable"].apply(classify_variable)
    by_type = df.groupby("type")[importance_col].sum()
    total = by_type.sum()
    by_type_pct = (by_type / total * 100).to_dict()
    return by_type_pct


def plot_shap_summary(
    shap_values, X_sample, feature_names,
    species_name, n_top=15,
    output_dir="results/figures",
):
    """SHAP beeswarm plot for top features."""
    import shap

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(5, n_top * 0.4)))
    shap.summary_plot(
        shap_values,
        features=X_sample,
        feature_names=feature_names,
        max_display=n_top,
        show=False,
        plot_size=None,
    )
    plt.title(f"{species_name}: SHAP Values (top {n_top} features)", fontsize=12)
    plt.tight_layout()

    safe_name = species_name.replace(" ", "_").lower()
    fig_path = out / f"{safe_name}_shap_summary.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    return str(fig_path)


def plot_rf_vs_dt_comparison(
    all_rf_by_type, all_dt_by_type,
    output_dir="results/figures",
):
    """
    Side-by-side comparison: DT vs RF importance by variable type for each species.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    types = ["Climate", "Topography", "Soil", "Land Cover"]
    colors = {"Climate": "#E53935", "Topography": "#1E88E5",
              "Soil": "#8D6E63", "Land Cover": "#43A047"}

    species_list = list(all_rf_by_type.keys())
    n = len(species_list)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(len(types), 1, figsize=(10, 3 * len(types)), sharex=True)

    for i, t in enumerate(types):
        ax = axes[i]
        dt_vals = [all_dt_by_type[sp].get(t, 0) for sp in species_list]
        rf_vals = [all_rf_by_type[sp].get(t, 0) for sp in species_list]

        ax.bar(x - width / 2, dt_vals, width, label="Decision Tree", color=colors[t], alpha=0.7)
        ax.bar(x + width / 2, rf_vals, width, label="Random Forest", color=colors[t], alpha=1.0,
               edgecolor="black", linewidth=0.5)

        ax.set_ylabel(f"{t} (%)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)

    short_labels = [sp.split(" ")[0][0] + ". " + sp.split(" ")[1] for sp in species_list]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(short_labels, fontsize=10, style="italic")
    fig.suptitle("Decision Tree vs. Random Forest: Variable Type Importance",
                 fontsize=13)
    plt.tight_layout()

    fig_path = out / "rf_vs_dt_variable_type_comparison.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def plot_rf_importance_by_type_stacked(
    all_rf_by_type,
    output_dir="results/figures",
):
    """Stacked bar chart of RF importance by type (same style as DT figure)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    types = ["Climate", "Topography", "Soil", "Land Cover"]
    colors = {"Climate": "#E53935", "Topography": "#1E88E5",
              "Soil": "#8D6E63", "Land Cover": "#43A047"}

    species_list = list(all_rf_by_type.keys())
    species_labels = [sp.split(" ")[0][0] + ". " + sp.split(" ")[1] for sp in species_list]

    x = np.arange(len(species_labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(species_labels))

    for t in types:
        vals = np.array([all_rf_by_type[sp].get(t, 0) for sp in species_list])
        ax.bar(x, vals, width, bottom=bottom, label=t, color=colors[t])
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(x[i], bottom[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(species_labels, fontsize=10, style="italic")
    ax.set_ylabel("Relative Feature Importance (%)")
    ax.set_title("Random Forest: Environmental Drivers of Niche Shift by Species\n"
                 "(Gini importance, 500 trees)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(1, 102, "Intercontinental invaders",
            ha="center", fontsize=9, color="gray")
    ax.text(3.5, 102, "Within-continent",
            ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    fig_path = out / "rf_importance_by_type_stacked.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def analyze_species_rf(
    df_filtered, species_name, env_vars,
    glossary=None, output_dir="results",
):
    """Full RF + SHAP analysis for one species."""
    safe_name = species_name.replace(" ", "_").lower()

    X, y, feature_names, combined = prepare_species_data(
        df_filtered, species_name, env_vars
    )

    print(f"\n{'='*60}")
    print(f"RANDOM FOREST ANALYSIS: {species_name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(y)} ({(y==0).sum()} native, {(y==1).sum()} invasive)")
    print(f"  Features: {len(feature_names)}")

    # Train RF
    clf, cv_scores = train_and_evaluate_rf(X, y, feature_names)
    print(f"\n  Cross-validation (5-fold):")
    print(f"    Accuracy: {cv_scores['accuracy_mean']:.3f} ± {cv_scores['accuracy_std']:.3f}")
    print(f"    F1 Score: {cv_scores['f1_mean']:.3f} ± {cv_scores['f1_std']:.3f}")
    print(f"    ROC AUC:  {cv_scores['roc_auc_mean']:.3f} ± {cv_scores['roc_auc_std']:.3f}")

    # Gini importances
    gini_imp = get_gini_importances(clf, feature_names)
    print(f"\n  Top 10 features (Gini importance):")
    for i, row in gini_imp.head(10).iterrows():
        desc = translate(row["variable"], glossary) if glossary else row["variable"]
        print(f"    {i+1:2d}. {row['variable']:<15s} {row['gini_importance']:.4f}  {desc}")

    # Importance by type
    gini_by_type = compute_importance_by_type(gini_imp, "gini_importance")
    print(f"\n  Importance by type (Gini):")
    for t in ["Climate", "Topography", "Soil", "Land Cover"]:
        print(f"    {t:<15s} {gini_by_type.get(t, 0):.1f}%")

    # SHAP values
    print(f"\n  Computing SHAP values...")
    shap_imp, shap_vals, X_sample = compute_shap_values(
        clf, X, feature_names
    )
    print(f"  Top 10 features (SHAP):")
    for i, row in shap_imp.head(10).iterrows():
        desc = translate(row["variable"], glossary) if glossary else row["variable"]
        print(f"    {i+1:2d}. {row['variable']:<15s} {row['mean_abs_shap']:.4f}  {desc}")

    shap_by_type = compute_importance_by_type(shap_imp, "mean_abs_shap")
    print(f"\n  Importance by type (SHAP):")
    for t in ["Climate", "Topography", "Soil", "Land Cover"]:
        print(f"    {t:<15s} {shap_by_type.get(t, 0):.1f}%")

    # Save
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    gini_imp.to_csv(tables_dir / f"{safe_name}_rf_gini_importances.csv", index=False)
    shap_imp.to_csv(tables_dir / f"{safe_name}_rf_shap_importances.csv", index=False)

    # SHAP summary plot
    figures_dir = str(Path(output_dir) / "figures")
    shap_fig = plot_shap_summary(
        shap_vals, X_sample, feature_names,
        species_name, output_dir=figures_dir,
    )
    print(f"  SHAP plot saved to {shap_fig}")

    return {
        "species": species_name,
        "n_native": int((y == 0).sum()),
        "n_invasive": int((y == 1).sum()),
        "n_features": len(feature_names),
        "cv_scores": cv_scores,
        "gini_by_type": gini_by_type,
        "shap_by_type": shap_by_type,
        "top_gini": gini_imp.head(10).to_dict("records"),
        "top_shap": shap_imp.head(10).to_dict("records"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Random Forest + SHAP robustness analysis"
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--config", "-c", default="config/species_config.yaml")
    parser.add_argument("--output", "-o", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)

    env_info = get_env_variables(df_filtered)
    env_vars = env_info["all_env"]
    glossary = load_glossary("data/raw/S2.xlsx")

    all_results = {}
    all_rf_by_type = {}

    for species in STUDY_SPECIES:
        result = analyze_species_rf(
            df_filtered, species, env_vars,
            glossary=glossary, output_dir=args.output,
        )
        all_results[species] = result
        all_rf_by_type[species] = result["gini_by_type"]

    # Load DT results for comparison
    dt_summary_path = Path(args.output) / "tables" / "dt_summary.json"
    all_dt_by_type = {}
    if dt_summary_path.exists():
        with open(dt_summary_path) as f:
            dt_data = json.load(f)
        for species in STUDY_SPECIES:
            if species in dt_data:
                top_feats = dt_data[species].get("top_features", [])
                if top_feats:
                    dt_imp = pd.DataFrame(top_feats)
                    # Recompute from full DT importances if available
                    dt_imp_path = Path(args.output) / "tables" / f"{species.replace(' ', '_').lower()}_feature_importances.csv"
                    if dt_imp_path.exists():
                        dt_imp = pd.read_csv(dt_imp_path)
                    all_dt_by_type[species] = compute_importance_by_type(dt_imp, "importance")

    # Comparison figures
    figures_dir = str(Path(args.output) / "figures")

    stacked_fig = plot_rf_importance_by_type_stacked(all_rf_by_type, output_dir=figures_dir)
    print(f"\nRF stacked figure saved to {stacked_fig}")

    if all_dt_by_type:
        comp_fig = plot_rf_vs_dt_comparison(all_rf_by_type, all_dt_by_type, output_dir=figures_dir)
        print(f"DT vs RF comparison saved to {comp_fig}")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY: RANDOM FOREST vs DECISION TREE")
    print(f"{'='*80}")
    print(f"{'Species':<28s} {'RF Acc':>7s} {'RF AUC':>7s} | "
          f"{'RF Clim%':>8s} {'RF Topo%':>8s} | "
          f"{'DT Clim%':>8s} {'DT Topo%':>8s}")
    print("-" * 80)
    for sp in STUDY_SPECIES:
        rf = all_results[sp]
        cv = rf["cv_scores"]
        rf_t = rf["gini_by_type"]
        dt_t = all_dt_by_type.get(sp, {})
        short = sp.split(" ")[0][0] + ". " + sp.split(" ")[1]
        print(f"{short:<28s} "
              f"{cv['accuracy_mean']:>7.3f} {cv['roc_auc_mean']:>7.3f} | "
              f"{rf_t.get('Climate', 0):>7.1f}% {rf_t.get('Topography', 0):>7.1f}% | "
              f"{dt_t.get('Climate', 0):>7.1f}% {dt_t.get('Topography', 0):>7.1f}%")

    # Save
    summary_path = Path(args.output) / "tables" / "rf_shap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()