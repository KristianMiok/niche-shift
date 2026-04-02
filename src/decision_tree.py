"""
decision_tree.py
Decision tree classification: native vs invasive range.

For each species, trains a decision tree to distinguish native from
invasive occurrences based on environmental variables. Extracts
interpretable ecological thresholds and feature importances.

Usage:
    python src/decision_tree.py --input data/raw/combined_data_true_master.csv
    python src/decision_tree.py --input data/raw/combined_data_true_master.csv --max-depth 4
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from data_loader import (
    load_geotraits, load_config, get_env_variables,
    COL_SPECIES, COL_STATUS,
)
from species_selector import (
    apply_quality_filters, select_species,
    STATUS_NATIVE, STATUS_ALIEN, STUDY_SPECIES,
)
from data_preparation import (
    handle_missing_values, remove_constant_variables,
    remove_highly_correlated,
)
from variable_glossary import load_glossary, translate, make_label


def prepare_xy(
    df: pd.DataFrame, env_vars: list
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare feature matrix X and target vector y.

    Target: 0 = native, 1 = invasive.

    Returns (X, y, feature_names).
    """
    X = df[env_vars].values
    y = (df["range_label"] == "invasive").astype(int).values
    return X, y, env_vars


def cross_validate_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Stratified k-fold cross-validation for the decision tree.

    Returns dict with mean and std of accuracy, f1, and roc_auc.
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scoring = ["accuracy", "f1", "roc_auc"]
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring)

    return {
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
        "f1_mean": cv_results["test_f1"].mean(),
        "f1_std": cv_results["test_f1"].std(),
        "roc_auc_mean": cv_results["test_roc_auc"].mean(),
        "roc_auc_std": cv_results["test_roc_auc"].std(),
    }


def train_final_tree(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    max_depth: int = 5,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train the final decision tree on all data."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def extract_feature_importances(
    clf: DecisionTreeClassifier, feature_names: list
) -> pd.DataFrame:
    """Extract and rank feature importances."""
    importances = pd.DataFrame({
        "variable": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importances["cumulative"] = importances["importance"].cumsum()
    importances["rank"] = range(1, len(importances) + 1)
    return importances


def extract_decision_rules(
    clf: DecisionTreeClassifier, feature_names: list
) -> list[dict]:
    """
    Extract decision rules (paths from root to each leaf).

    Returns a list of dicts, each representing one leaf with:
      - path: list of (feature, threshold, direction) tuples
      - class: predicted class (native/invasive)
      - n_samples: number of training samples in this leaf
      - native_pct: proportion classified as native
    """
    tree = clf.tree_
    class_names = ["native", "invasive"]

    rules = []

    def recurse(node, path):
        if tree.children_left[node] == tree.children_right[node]:
            # Leaf node
            values = tree.value[node][0]
            total = values.sum()
            predicted = class_names[np.argmax(values)]
            rules.append({
                "path": list(path),
                "class": predicted,
                "n_samples": int(total),
                "native_count": int(values[0]),
                "invasive_count": int(values[1]),
                "native_pct": round(values[0] / total * 100, 1),
                "invasive_pct": round(values[1] / total * 100, 1),
            })
            return

        feature = feature_names[tree.feature[node]]
        threshold = round(tree.threshold[node], 4)

        # Left child: feature <= threshold
        recurse(
            tree.children_left[node],
            path + [(feature, "<=", threshold)],
        )
        # Right child: feature > threshold
        recurse(
            tree.children_right[node],
            path + [(feature, ">", threshold)],
        )

    recurse(0, [])
    return rules


def format_rules_text(rules: list[dict], species_name: str, glossary: dict = None) -> str:
    """Format decision rules as human-readable text."""
    lines = [
        f"Decision Rules for {species_name}",
        f"{'='*60}",
        f"Total leaves: {len(rules)}",
        "",
    ]

    for i, rule in enumerate(
        sorted(rules, key=lambda r: (-r["n_samples"],)), 1
    ):
        if glossary:
            path_str = " AND ".join(
                f"{translate(feat, glossary)} {op} {thresh}"
                for feat, op, thresh in rule["path"]
            )
        else:
            path_str = " AND ".join(
                f"{feat} {op} {thresh}" for feat, op, thresh in rule["path"]
            )
        lines.append(f"Rule {i}: {rule['class'].upper()}")
        lines.append(f"  IF {path_str}")
        lines.append(f"  Samples: {rule['n_samples']} "
                      f"(native: {rule['native_pct']}%, "
                      f"invasive: {rule['invasive_pct']}%)")
        lines.append("")

    return "\n".join(lines)


def plot_decision_tree(
    clf: DecisionTreeClassifier,
    feature_names: list,
    species_name: str,
    output_dir: str = "results/figures",
) -> str:
    """Plot and save the decision tree visualization."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Calculate figure size based on tree depth
    depth = clf.get_depth()
    n_leaves = clf.get_n_leaves()
    fig_width = max(20, n_leaves * 2.5)
    fig_height = max(10, depth * 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["native", "invasive"],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=8,
        impurity=False,
        proportion=True,
    )
    ax.set_title(f"{species_name}: Decision Tree (native vs. invasive)",
                 fontsize=14)

    safe_name = species_name.replace(" ", "_").lower()
    fig_path = out / f"{safe_name}_decision_tree.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def plot_feature_importances(
    importances: pd.DataFrame,
    species_name: str,
    n_top: int = 20,
    output_dir: str = "results/figures",
    glossary: dict = None,
) -> str:
    """Plot top feature importances as horizontal bar chart."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    top = importances.head(n_top).iloc[::-1]

    if glossary:
        labels = [make_label(v, glossary, max_len=55) for v in top["variable"]]
    else:
        labels = top["variable"].tolist()

    fig, ax = plt.subplots(figsize=(10, max(5, n_top * 0.35)))
    ax.barh(range(len(labels)), top["importance"], color="#2196F3")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{species_name}: Top {n_top} Environmental Variables\n"
                 f"(distinguishing native from invasive range)")
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()

    safe_name = species_name.replace(" ", "_").lower()
    fig_path = out / f"{safe_name}_feature_importance.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def analyze_species(
    df: pd.DataFrame,
    species_name: str,
    env_vars: list,
    max_depth: int = 5,
    output_dir: str = "results",
    glossary: dict = None,
) -> dict:
    """
    Full decision tree analysis for one species.

    Returns dict with all results.
    """
    safe_name = species_name.replace(" ", "_").lower()

    # Prepare data
    sp_data = df[df[COL_SPECIES] == species_name].copy()
    native = sp_data[sp_data[COL_STATUS] == STATUS_NATIVE]
    invasive = sp_data[sp_data[COL_STATUS] == STATUS_ALIEN]

    combined = pd.concat([native, invasive], ignore_index=True)

    # Clean variables for this species
    combined, clean_vars = handle_missing_values(combined, env_vars)
    clean_vars = remove_constant_variables(combined, clean_vars)
    clean_vars = remove_highly_correlated(combined, clean_vars, threshold=0.98)

    # Add range label
    combined["range_label"] = combined[COL_STATUS].map(
        {STATUS_NATIVE: "native", STATUS_ALIEN: "invasive"}
    )

    X, y, feature_names = prepare_xy(combined, clean_vars)

    print(f"\n{'='*60}")
    print(f"DECISION TREE ANALYSIS: {species_name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(y)} ({(y==0).sum()} native, {(y==1).sum()} invasive)")
    print(f"  Features: {len(feature_names)}")
    print(f"  Max depth: {max_depth}")

    # Cross-validation
    print(f"\n  Cross-validation (5-fold stratified):")
    cv_scores = cross_validate_tree(X, y, max_depth=max_depth)
    print(f"    Accuracy: {cv_scores['accuracy_mean']:.3f} "
          f"± {cv_scores['accuracy_std']:.3f}")
    print(f"    F1 Score: {cv_scores['f1_mean']:.3f} "
          f"± {cv_scores['f1_std']:.3f}")
    print(f"    ROC AUC:  {cv_scores['roc_auc_mean']:.3f} "
          f"± {cv_scores['roc_auc_std']:.3f}")

    # Train final tree
    clf = train_final_tree(X, y, feature_names, max_depth=max_depth)
    print(f"\n  Final tree: depth={clf.get_depth()}, "
          f"leaves={clf.get_n_leaves()}")

    # Feature importances
    importances = extract_feature_importances(clf, feature_names)
    print(f"\n  Top 10 features:")
    for _, row in importances.head(10).iterrows():
        var = row['variable']
        desc = translate(var, glossary) if glossary else var
        print(f"    {row['rank']:2d}. {var:<15s} {row['importance']:.4f}  {desc}")

    # Decision rules
    rules = extract_decision_rules(clf, feature_names)
    rules_text = format_rules_text(rules, species_name, glossary=glossary)

    # Save results
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    importances.to_csv(
        tables_dir / f"{safe_name}_feature_importances.csv", index=False
    )

    rules_path = tables_dir / f"{safe_name}_decision_rules.txt"
    with open(rules_path, "w") as f:
        f.write(rules_text)
    print(f"\n  Rules saved to {rules_path}")

    # Save rules as JSON for programmatic access
    rules_json_path = tables_dir / f"{safe_name}_decision_rules.json"
    with open(rules_json_path, "w") as f:
        json.dump(rules, f, indent=2)

    # Plots
    figures_dir = str(Path(output_dir) / "figures")
    tree_fig = plot_decision_tree(clf, feature_names, species_name, figures_dir)
    print(f"  Tree plot saved to {tree_fig}")

    imp_fig = plot_feature_importances(importances, species_name,
                                       output_dir=figures_dir, glossary=glossary)
    print(f"  Importance plot saved to {imp_fig}")

    # Full classification report
    y_pred = clf.predict(X)
    report = classification_report(
        y, y_pred, target_names=["native", "invasive"], output_dict=True
    )

    return {
        "species": species_name,
        "n_native": int((y == 0).sum()),
        "n_invasive": int((y == 1).sum()),
        "n_features": len(feature_names),
        "max_depth": max_depth,
        "actual_depth": clf.get_depth(),
        "n_leaves": clf.get_n_leaves(),
        "cv_scores": cv_scores,
        "classification_report": report,
        "top_features": importances.head(20).to_dict("records"),
        "n_rules": len(rules),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Decision tree analysis: native vs invasive"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to combined_data_true_master.csv")
    parser.add_argument("--config", "-c", default="config/species_config.yaml",
                        help="Path to species config YAML")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum tree depth (default: 5)")
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)

    env_info = get_env_variables(df_filtered)
    env_vars = env_info["all_env"]

    # Load variable glossary
    glossary_path = "data/raw/S2.xlsx"
    glossary = load_glossary(glossary_path)
    print(f"Loaded glossary: {len(glossary)} variable definitions")

    # Analyze each study species
    all_results = {}
    for species in STUDY_SPECIES:
        result = analyze_species(
            df_filtered, species, env_vars,
            max_depth=args.max_depth,
            output_dir=args.output,
            glossary=glossary,
        )
        all_results[species] = result

    # Summary table
    print(f"\n\n{'='*75}")
    print(f"SUMMARY ACROSS ALL SPECIES")
    print(f"{'='*75}")
    print(f"{'Species':<30s} {'Native':>7s} {'Invasive':>9s} "
          f"{'CV Acc':>7s} {'CV F1':>7s} {'CV AUC':>7s} {'Depth':>6s} {'Leaves':>7s}")
    print("-" * 75)
    for sp, res in all_results.items():
        cv = res["cv_scores"]
        print(f"{sp:<30s} {res['n_native']:>7d} {res['n_invasive']:>9d} "
              f"{cv['accuracy_mean']:>7.3f} {cv['f1_mean']:>7.3f} "
              f"{cv['roc_auc_mean']:>7.3f} {res['actual_depth']:>6d} "
              f"{res['n_leaves']:>7d}")

    # Save summary
    summary_path = Path(args.output) / "tables" / "dt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()