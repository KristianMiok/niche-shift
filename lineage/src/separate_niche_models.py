"""
separate_niche_models.py
Separate niche models: compare ecological thresholds between native
and invasive ranges.

For each species, trains two decision trees:
  1. Native range: presence vs background (using other species in native region)
  2. Invasive range: presence vs background (using other species in invasive region)

Then compares the learned thresholds to quantify niche shifts.

Usage:
    python src/separate_niche_models.py --input data/raw/combined_data_true_master.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

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
from decision_tree import extract_feature_importances, extract_decision_rules
from variable_glossary import load_glossary, translate, make_label


def sample_background(
    df_all: pd.DataFrame,
    target_species: str,
    status: str,
    env_vars: list,
    n_ratio: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample pseudo-absences from other species' records in the same status
    (Native or Alien) as environmental background.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full filtered dataset (all species).
    target_species : str
        The focal species.
    status : str
        STATUS_NATIVE or STATUS_ALIEN — which range to build background for.
    env_vars : list
        Environmental variable columns.
    n_ratio : float
        Ratio of background samples to presence samples. 1.0 = equal.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Background samples with env vars and label column.
    """
    # Presence records for focal species in this range
    presence = df_all[
        (df_all[COL_SPECIES] == target_species) &
        (df_all[COL_STATUS] == status)
    ]
    n_presence = len(presence)

    if n_presence == 0:
        return pd.DataFrame()

    # Background: all other species' records with same status
    background_pool = df_all[
        (df_all[COL_SPECIES] != target_species) &
        (df_all[COL_STATUS] == status)
    ]

    # Sample up to n_ratio * n_presence records
    n_sample = min(int(n_presence * n_ratio), len(background_pool))
    if n_sample < 10:
        # If not enough same-status records, use all records from other species
        background_pool = df_all[df_all[COL_SPECIES] != target_species]
        n_sample = min(int(n_presence * n_ratio), len(background_pool))

    background = background_pool.sample(
        n=n_sample, random_state=random_state, replace=False
    )

    return background


def train_niche_tree(
    presence: pd.DataFrame,
    background: pd.DataFrame,
    env_vars: list,
    max_depth: int = 4,
    random_state: int = 42,
) -> tuple:
    """
    Train a decision tree: presence (1) vs background (0).

    Returns (clf, X, y, clean_vars, cv_scores).
    """
    # Combine and label
    presence = presence.copy()
    background = background.copy()
    presence["_target"] = 1
    background["_target"] = 0
    combined = pd.concat([presence, background], ignore_index=True)

    # Clean variables
    combined, clean_vars = handle_missing_values(combined, env_vars, max_missing_pct=0.3)
    clean_vars = remove_constant_variables(combined, clean_vars)
    clean_vars = remove_highly_correlated(combined, clean_vars, threshold=0.98)

    X = combined[clean_vars].values
    y = combined["_target"].values

    # Cross-validate
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        clf, X, y, cv=cv, scoring=["accuracy", "roc_auc"]
    )
    cv_scores = {
        "accuracy": cv_results["test_accuracy"].mean(),
        "roc_auc": cv_results["test_roc_auc"].mean(),
    }

    # Train final
    clf.fit(X, y)

    return clf, X, y, clean_vars, cv_scores


def extract_top_splits(
    clf: DecisionTreeClassifier, feature_names: list, n_top: int = 5
) -> list[dict]:
    """
    Extract the top split variables and thresholds from the tree.

    Returns list of dicts with: variable, threshold, depth, samples_left,
    samples_right, improvement.
    """
    tree = clf.tree_
    splits = []

    for node in range(tree.node_count):
        if tree.children_left[node] == tree.children_right[node]:
            continue  # Skip leaf nodes

        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        depth = _node_depth(tree, node)
        n_left = tree.n_node_samples[tree.children_left[node]]
        n_right = tree.n_node_samples[tree.children_right[node]]
        improvement = tree.impurity[node] - (
            n_left / (n_left + n_right) * tree.impurity[tree.children_left[node]] +
            n_right / (n_left + n_right) * tree.impurity[tree.children_right[node]]
        )

        splits.append({
            "variable": feature,
            "threshold": round(threshold, 4),
            "depth": depth,
            "n_samples": tree.n_node_samples[node],
            "improvement": round(improvement, 6),
        })

    # Sort by improvement (information gain)
    splits.sort(key=lambda x: -x["improvement"])
    return splits[:n_top]


def _node_depth(tree, node_id):
    """Compute depth of a node in the tree."""
    depth = 0
    current = node_id
    # Walk up to root by checking all nodes
    # Simple approach: BFS from root
    depths = np.zeros(tree.node_count, dtype=int)
    stack = [0]
    while stack:
        node = stack.pop()
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left != right:  # Not a leaf
            depths[left] = depths[node] + 1
            depths[right] = depths[node] + 1
            stack.append(left)
            stack.append(right)
    return depths[node_id]


def compare_thresholds(
    native_splits: list[dict],
    invasive_splits: list[dict],
    glossary: dict = None,
) -> list[dict]:
    """
    Compare splits between native and invasive niche models.

    For variables that appear in both models, compute the threshold shift.
    """
    native_dict = {}
    for s in native_splits:
        var = s["variable"]
        if var not in native_dict or s["improvement"] > native_dict[var]["improvement"]:
            native_dict[var] = s

    invasive_dict = {}
    for s in invasive_splits:
        var = s["variable"]
        if var not in invasive_dict or s["improvement"] > invasive_dict[var]["improvement"]:
            invasive_dict[var] = s

    # All variables from both models
    all_vars = set(native_dict.keys()) | set(invasive_dict.keys())

    comparisons = []
    for var in all_vars:
        nat = native_dict.get(var)
        inv = invasive_dict.get(var)
        desc = translate(var, glossary) if glossary else var

        comp = {
            "variable": var,
            "description": desc,
            "native_threshold": nat["threshold"] if nat else None,
            "invasive_threshold": inv["threshold"] if inv else None,
            "native_depth": nat["depth"] if nat else None,
            "invasive_depth": inv["depth"] if inv else None,
            "native_importance": nat["improvement"] if nat else 0,
            "invasive_importance": inv["improvement"] if inv else 0,
            "in_both": nat is not None and inv is not None,
        }

        if comp["in_both"]:
            comp["threshold_shift"] = round(
                comp["invasive_threshold"] - comp["native_threshold"], 4
            )
            # Relative shift
            if abs(comp["native_threshold"]) > 1e-10:
                comp["shift_pct"] = round(
                    comp["threshold_shift"] / abs(comp["native_threshold"]) * 100, 1
                )
            else:
                comp["shift_pct"] = None
        else:
            comp["threshold_shift"] = None
            comp["shift_pct"] = None

        comparisons.append(comp)

    # Sort: shared variables first, then by combined importance
    comparisons.sort(key=lambda c: (
        -int(c["in_both"]),
        -(c["native_importance"] + c["invasive_importance"]),
    ))
    return comparisons


def plot_threshold_comparison(
    comparisons: list[dict],
    species_name: str,
    output_dir: str = "results/figures",
) -> str:
    """
    Plot comparing native vs invasive thresholds for shared variables.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Only variables present in both models
    shared = [c for c in comparisons if c["in_both"]]

    if not shared:
        print(f"  No shared variables to plot for {species_name}")
        return ""

    n = len(shared)
    fig, axes = plt.subplots(1, min(n, 6), figsize=(min(n, 6) * 3.5, 4))
    if min(n, 6) == 1:
        axes = [axes]

    for i, comp in enumerate(shared[:6]):
        ax = axes[i]
        var = comp["variable"]
        nat_t = comp["native_threshold"]
        inv_t = comp["invasive_threshold"]

        ax.barh(["Native", "Invasive"], [nat_t, inv_t],
                color=["#2196F3", "#F44336"], height=0.5)
        ax.set_title(f"{var}\n{comp['description'][:35]}", fontsize=8)
        ax.tick_params(labelsize=8)

        # Show shift
        if comp["shift_pct"] is not None:
            shift_str = f"Shift: {comp['threshold_shift']:+.1f}"
            if comp["shift_pct"] is not None:
                shift_str += f" ({comp['shift_pct']:+.0f}%)"
            ax.set_xlabel(shift_str, fontsize=7)

    fig.suptitle(f"{species_name}: Niche Threshold Comparison\n"
                 f"(native vs. invasive range)", fontsize=12)
    plt.tight_layout()

    safe_name = species_name.replace(" ", "_").lower()
    fig_path = out / f"{safe_name}_threshold_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def analyze_species_niche(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list,
    max_depth: int = 4,
    glossary: dict = None,
    output_dir: str = "results",
) -> dict:
    """Full separate niche model analysis for one species."""
    safe_name = species_name.replace(" ", "_").lower()

    # Get presence data
    sp_data = df_filtered[df_filtered[COL_SPECIES] == species_name]
    native_presence = sp_data[sp_data[COL_STATUS] == STATUS_NATIVE]
    invasive_presence = sp_data[sp_data[COL_STATUS] == STATUS_ALIEN]

    print(f"\n{'='*60}")
    print(f"SEPARATE NICHE MODELS: {species_name}")
    print(f"{'='*60}")
    print(f"  Native presence:   {len(native_presence)}")
    print(f"  Invasive presence: {len(invasive_presence)}")

    # Sample backgrounds
    native_bg = sample_background(
        df_filtered, species_name, STATUS_NATIVE, env_vars
    )
    invasive_bg = sample_background(
        df_filtered, species_name, STATUS_ALIEN, env_vars
    )
    print(f"  Native background:   {len(native_bg)}")
    print(f"  Invasive background: {len(invasive_bg)}")

    # Train native niche model
    print(f"\n  --- Native Niche Model ---")
    nat_clf, nat_X, nat_y, nat_vars, nat_cv = train_niche_tree(
        native_presence, native_bg, env_vars, max_depth=max_depth
    )
    print(f"    CV Accuracy: {nat_cv['accuracy']:.3f}, "
          f"ROC AUC: {nat_cv['roc_auc']:.3f}")
    print(f"    Tree: depth={nat_clf.get_depth()}, "
          f"leaves={nat_clf.get_n_leaves()}, "
          f"features={len(nat_vars)}")

    nat_importances = extract_feature_importances(nat_clf, nat_vars)
    nat_splits = extract_top_splits(nat_clf, nat_vars, n_top=10)

    print(f"    Top splits:")
    for s in nat_splits[:5]:
        desc = translate(s['variable'], glossary) if glossary else s['variable']
        print(f"      {s['variable']:<15s} <= {s['threshold']:<10} "
              f"(depth {s['depth']}, gain={s['improvement']:.4f})  {desc}")

    # Train invasive niche model
    print(f"\n  --- Invasive Niche Model ---")
    inv_clf, inv_X, inv_y, inv_vars, inv_cv = train_niche_tree(
        invasive_presence, invasive_bg, env_vars, max_depth=max_depth
    )
    print(f"    CV Accuracy: {inv_cv['accuracy']:.3f}, "
          f"ROC AUC: {inv_cv['roc_auc']:.3f}")
    print(f"    Tree: depth={inv_clf.get_depth()}, "
          f"leaves={inv_clf.get_n_leaves()}, "
          f"features={len(inv_vars)}")

    inv_importances = extract_feature_importances(inv_clf, inv_vars)
    inv_splits = extract_top_splits(inv_clf, inv_vars, n_top=10)

    print(f"    Top splits:")
    for s in inv_splits[:5]:
        desc = translate(s['variable'], glossary) if glossary else s['variable']
        print(f"      {s['variable']:<15s} <= {s['threshold']:<10} "
              f"(depth {s['depth']}, gain={s['improvement']:.4f})  {desc}")

    # Compare thresholds
    comparisons = compare_thresholds(nat_splits, inv_splits, glossary=glossary)

    shared = [c for c in comparisons if c["in_both"]]
    print(f"\n  --- Threshold Comparison ---")
    print(f"  Variables in both models: {len(shared)}")
    for c in shared:
        print(f"    {c['variable']:<15s} Native: {c['native_threshold']:<10} "
              f"Invasive: {c['invasive_threshold']:<10} "
              f"Shift: {c['threshold_shift']:+.2f}", end="")
        if c['shift_pct'] is not None:
            print(f" ({c['shift_pct']:+.1f}%)", end="")
        print(f"  | {c['description']}")

    # Save results
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(tables_dir / f"{safe_name}_threshold_comparison.csv", index=False)

    # Plot
    figures_dir = str(Path(output_dir) / "figures")
    fig_path = plot_threshold_comparison(
        comparisons, species_name, output_dir=figures_dir
    )
    if fig_path:
        print(f"\n  Plot saved to {fig_path}")

    return {
        "species": species_name,
        "native_cv": nat_cv,
        "invasive_cv": inv_cv,
        "native_n_presence": len(native_presence),
        "invasive_n_presence": len(invasive_presence),
        "native_top_splits": nat_splits[:5],
        "invasive_top_splits": inv_splits[:5],
        "shared_variables": shared,
        "n_shared": len(shared),
    }


def plot_summary_threshold_shifts(
    all_results: dict,
    glossary: dict = None,
    output_dir: str = "results/figures",
) -> str:
    """
    Summary figure: threshold shifts across all species for shared variables.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect all shared comparisons
    rows = []
    for species, result in all_results.items():
        short = species.split(" ")[0][0] + ". " + species.split(" ")[1]
        for comp in result["shared_variables"]:
            rows.append({
                "species": short,
                "variable": comp["variable"],
                "description": comp["description"][:40],
                "native_threshold": comp["native_threshold"],
                "invasive_threshold": comp["invasive_threshold"],
                "shift": comp["threshold_shift"],
                "shift_pct": comp["shift_pct"],
            })

    if not rows:
        print("  No shared variables across species to plot.")
        return ""

    df = pd.DataFrame(rows)

    # Plot: grouped by species, showing native vs invasive thresholds
    species_list = df["species"].unique()
    n_species = len(species_list)

    fig, axes = plt.subplots(n_species, 1, figsize=(12, n_species * 3),
                             squeeze=False)

    for i, species in enumerate(species_list):
        ax = axes[i, 0]
        sp_data = df[df["species"] == species]

        y_pos = range(len(sp_data))
        labels = [f"{row['variable']}: {row['description']}"
                  for _, row in sp_data.iterrows()]

        ax.barh([p - 0.15 for p in y_pos], sp_data["native_threshold"],
                height=0.3, color="#2196F3", label="Native", alpha=0.8)
        ax.barh([p + 0.15 for p in y_pos], sp_data["invasive_threshold"],
                height=0.3, color="#F44336", label="Invasive", alpha=0.8)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{species}", fontsize=11, style="italic")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle("Niche Threshold Comparison: Native vs. Invasive Range",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    fig_path = out / "summary_threshold_shifts.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def main():
    parser = argparse.ArgumentParser(
        description="Separate niche models: native vs invasive thresholds"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to combined_data_true_master.csv")
    parser.add_argument("--config", "-c", default="config/species_config.yaml")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Max tree depth for niche models (default: 4)")
    parser.add_argument("--output", "-o", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)

    env_info = get_env_variables(df_filtered)
    env_vars = env_info["all_env"]

    glossary = load_glossary("data/raw/S2.xlsx")
    print(f"Loaded glossary: {len(glossary)} definitions")

    all_results = {}
    for species in STUDY_SPECIES:
        result = analyze_species_niche(
            df_filtered, species, env_vars,
            max_depth=args.max_depth,
            glossary=glossary,
            output_dir=args.output,
        )
        all_results[species] = result

    # Summary figure
    figures_dir = str(Path(args.output) / "figures")
    summary_fig = plot_summary_threshold_shifts(
        all_results, glossary=glossary, output_dir=figures_dir
    )
    if summary_fig:
        print(f"\nSummary figure saved to {summary_fig}")

    # Save all results
    tables_dir = Path(args.output) / "tables"
    summary_path = tables_dir / "separate_niche_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Full results saved to {summary_path}")

    # Print summary
    print(f"\n\n{'='*75}")
    print("SUMMARY: SEPARATE NICHE MODELS")
    print(f"{'='*75}")
    print(f"{'Species':<28s} {'Nat CV':>7s} {'Inv CV':>7s} {'Shared':>7s}")
    print("-" * 55)
    for sp, res in all_results.items():
        short = sp.split(" ")[0][0] + ". " + sp.split(" ")[1]
        print(f"{short:<28s} "
              f"{res['native_cv']['roc_auc']:>7.3f} "
              f"{res['invasive_cv']['roc_auc']:>7.3f} "
              f"{res['n_shared']:>7d}")


if __name__ == "__main__":
    main()