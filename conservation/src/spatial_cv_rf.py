from __future__ import annotations

"""
spatial_cv_rf.py

Random Forest classification with comparison between:
1) standard non-spatial cross-validation
2) spatially blocked cross-validation using basin_id as groups

Target setting for the conservation manuscript:
Native vs Introduced for three focal species.

Outputs:
- results/tables/spatial_cv_rf_summary.csv
- results/tables/spatial_cv_rf_summary.json
- results/tables/spatial_cv_rf_fold_details.csv

Usage:
    python src/spatial_cv_rf.py --input data/raw/combined_data_true_master.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from data_loader import (
    load_geotraits,
    load_config,
    get_env_variables,
    COL_SPECIES,
    COL_STATUS,
)
from species_selector import (
    apply_quality_filters,
    STATUS_NATIVE,
    STATUS_ALIEN,   # in conservation branch this should already be "Introduced"
    STUDY_SPECIES,
)
from data_preparation import (
    handle_missing_values,
    remove_constant_variables,
    remove_highly_correlated,
)


GROUP_COL = "basin_id"


def classify_variable(var_name: str) -> str:
    if "CLI" in var_name:
        return "Climate"
    if "TOP" in var_name:
        return "Topography"
    if "SOL" in var_name:
        return "Soil"
    if "LAC" in var_name:
        return "Land Cover"
    return "Other"


def summarise_importance_by_type(
    importances: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    df = pd.DataFrame({
        "variable": feature_names,
        "importance": importances,
    })
    df["type"] = df["variable"].apply(classify_variable)

    grouped = df.groupby("type", as_index=False)["importance"].sum()
    total = grouped["importance"].sum()

    out = {
        "Climate": 0.0,
        "Topography": 0.0,
        "Soil": 0.0,
        "Land Cover": 0.0,
        "Other": 0.0,
    }

    if total <= 0:
        return out

    for _, row in grouped.iterrows():
        out[row["type"]] = float(row["importance"] / total * 100.0)

    return out


def prepare_species_data(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    sp = df_filtered[df_filtered[COL_SPECIES] == species_name].copy()

    keep_cols = [COL_SPECIES, COL_STATUS, GROUP_COL] + env_vars
    sp = sp[keep_cols].copy()

    sp, clean_vars = handle_missing_values(sp, env_vars)
    clean_vars = remove_constant_variables(sp, clean_vars)
    clean_vars = remove_highly_correlated(sp, clean_vars, threshold=0.98)

    sp["range_label"] = sp[COL_STATUS].map({
        STATUS_NATIVE: 0,
        STATUS_ALIEN: 1,
    })

    sp = sp.dropna(subset=["range_label", GROUP_COL]).copy()
    sp["range_label"] = sp["range_label"].astype(int)

    X = sp[clean_vars].to_numpy()
    y = sp["range_label"].to_numpy()
    groups = sp[GROUP_COL].to_numpy()

    return X, y, groups, clean_vars, sp


def build_rf(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    splitter: Any,
    groups: np.ndarray | None = None,
    label: str = "non_spatial",
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    fold_rows: list[dict[str, Any]] = []
    type_importances: list[dict[str, float]] = []

    if groups is None:
        split_iter = splitter.split(X, y)
    else:
        split_iter = splitter.split(X, y, groups)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        clf = build_rf(random_state=random_state + fold_idx)
        clf.fit(X[train_idx], y[train_idx])

        pred = clf.predict(X[test_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]

        acc = accuracy_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], prob)

        type_pct = summarise_importance_by_type(
            clf.feature_importances_,
            feature_names,
        )
        type_importances.append(type_pct)

        fold_rows.append({
            "cv_type": label,
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": float(acc),
            "auc": float(auc),
            "Climate_pct": float(type_pct["Climate"]),
            "Topography_pct": float(type_pct["Topography"]),
            "Soil_pct": float(type_pct["Soil"]),
            "LandCover_pct": float(type_pct["Land Cover"]),
        })

    type_df = pd.DataFrame(type_importances)
    type_summary = {
        "Climate_mean_pct": float(type_df["Climate"].mean()),
        "Climate_sd_pct": float(type_df["Climate"].std(ddof=0)),
        "Topography_mean_pct": float(type_df["Topography"].mean()),
        "Topography_sd_pct": float(type_df["Topography"].std(ddof=0)),
        "Soil_mean_pct": float(type_df["Soil"].mean()),
        "Soil_sd_pct": float(type_df["Soil"].std(ddof=0)),
        "LandCover_mean_pct": float(type_df["Land Cover"].mean()),
        "LandCover_sd_pct": float(type_df["Land Cover"].std(ddof=0)),
    }

    return fold_rows, type_summary


def aggregate_cv_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    return {
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_sd": float(df["accuracy"].std(ddof=0)),
        "auc_mean": float(df["auc"].mean()),
        "auc_sd": float(df["auc"].std(ddof=0)),
    }


def analyse_species(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X, y, groups, feature_names, sp_df = prepare_species_data(
        df_filtered=df_filtered,
        species_name=species_name,
        env_vars=env_vars,
    )

    n_native = int((y == 0).sum())
    n_non_native = int((y == 1).sum())
    n_groups = int(pd.Series(groups).nunique())

    non_spatial_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    spatial_cv = GroupKFold(n_splits=n_splits)

    non_rows, non_type = run_cv(
        X=X,
        y=y,
        feature_names=feature_names,
        splitter=non_spatial_cv,
        groups=None,
        label="non_spatial",
        random_state=random_state,
    )
    sp_rows, sp_type = run_cv(
        X=X,
        y=y,
        feature_names=feature_names,
        splitter=spatial_cv,
        groups=groups,
        label="spatial_basin",
        random_state=random_state,
    )

    non_metrics = aggregate_cv_rows(non_rows)
    sp_metrics = aggregate_cv_rows(sp_rows)

    summary = {
        "species": species_name,
        "grouping_variable": GROUP_COL,
        "n_records": int(len(sp_df)),
        "n_native": n_native,
        "n_non_native": n_non_native,
        "n_unique_groups": n_groups,

        "non_spatial_accuracy_mean": non_metrics["accuracy_mean"],
        "non_spatial_accuracy_sd": non_metrics["accuracy_sd"],
        "non_spatial_auc_mean": non_metrics["auc_mean"],
        "non_spatial_auc_sd": non_metrics["auc_sd"],

        "spatial_accuracy_mean": sp_metrics["accuracy_mean"],
        "spatial_accuracy_sd": sp_metrics["accuracy_sd"],
        "spatial_auc_mean": sp_metrics["auc_mean"],
        "spatial_auc_sd": sp_metrics["auc_sd"],

        "accuracy_drop_spatial_minus_nonspatial": (
            sp_metrics["accuracy_mean"] - non_metrics["accuracy_mean"]
        ),
        "auc_drop_spatial_minus_nonspatial": (
            sp_metrics["auc_mean"] - non_metrics["auc_mean"]
        ),

        "non_spatial_importance_by_type": non_type,
        "spatial_importance_by_type": sp_type,
    }

    fold_df = pd.DataFrame(non_rows + sp_rows)
    fold_df.insert(0, "species", species_name)

    return summary, fold_df


def print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 120)
    print("SPATIAL vs NON-SPATIAL RANDOM FOREST CV")
    print("=" * 120)
    print(
        f"{'Species':<32}"
        f"{'Non-sp Acc':>12}"
        f"{'Sp Acc':>10}"
        f"{'ΔAcc':>9}"
        f"{'Non-sp AUC':>12}"
        f"{'Sp AUC':>10}"
        f"{'ΔAUC':>9}"
        f"{'Sp Climate':>12}"
        f"{'Sp Topo':>10}"
        f"{'Sp Soil':>10}"
        f"{'Sp LC':>9}"
    )
    print("-" * 120)

    for row in summary_rows:
        sp_type = row["spatial_importance_by_type"]
        print(
            f"{row['species']:<32}"
            f"{row['non_spatial_accuracy_mean']:>12.3f}"
            f"{row['spatial_accuracy_mean']:>10.3f}"
            f"{row['accuracy_drop_spatial_minus_nonspatial']:>9.3f}"
            f"{row['non_spatial_auc_mean']:>12.3f}"
            f"{row['spatial_auc_mean']:>10.3f}"
            f"{row['auc_drop_spatial_minus_nonspatial']:>9.3f}"
            f"{sp_type['Climate_mean_pct']:>12.1f}"
            f"{sp_type['Topography_mean_pct']:>10.1f}"
            f"{sp_type['Soil_mean_pct']:>10.1f}"
            f"{sp_type['LandCover_mean_pct']:>9.1f}"
        )


def save_outputs(
    summary_rows: list[dict[str, Any]],
    fold_details: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df_rows = []
    for row in summary_rows:
        sp_non = row["non_spatial_importance_by_type"]
        sp_spa = row["spatial_importance_by_type"]

        summary_df_rows.append({
            "species": row["species"],
            "grouping_variable": row["grouping_variable"],
            "n_records": row["n_records"],
            "n_native": row["n_native"],
            "n_non_native": row["n_non_native"],
            "n_unique_groups": row["n_unique_groups"],

            "non_spatial_accuracy_mean": row["non_spatial_accuracy_mean"],
            "non_spatial_accuracy_sd": row["non_spatial_accuracy_sd"],
            "spatial_accuracy_mean": row["spatial_accuracy_mean"],
            "spatial_accuracy_sd": row["spatial_accuracy_sd"],
            "accuracy_drop_spatial_minus_nonspatial": row["accuracy_drop_spatial_minus_nonspatial"],

            "non_spatial_auc_mean": row["non_spatial_auc_mean"],
            "non_spatial_auc_sd": row["non_spatial_auc_sd"],
            "spatial_auc_mean": row["spatial_auc_mean"],
            "spatial_auc_sd": row["spatial_auc_sd"],
            "auc_drop_spatial_minus_nonspatial": row["auc_drop_spatial_minus_nonspatial"],

            "non_spatial_climate_mean_pct": sp_non["Climate_mean_pct"],
            "non_spatial_topography_mean_pct": sp_non["Topography_mean_pct"],
            "non_spatial_soil_mean_pct": sp_non["Soil_mean_pct"],
            "non_spatial_landcover_mean_pct": sp_non["LandCover_mean_pct"],

            "spatial_climate_mean_pct": sp_spa["Climate_mean_pct"],
            "spatial_topography_mean_pct": sp_spa["Topography_mean_pct"],
            "spatial_soil_mean_pct": sp_spa["Soil_mean_pct"],
            "spatial_landcover_mean_pct": sp_spa["LandCover_mean_pct"],
        })

    summary_df = pd.DataFrame(summary_df_rows)
    summary_csv = out_dir / "spatial_cv_rf_summary.csv"
    summary_json = out_dir / "spatial_cv_rf_summary.json"
    fold_csv = out_dir / "spatial_cv_rf_fold_details.csv"

    summary_df.to_csv(summary_csv, index=False)
    fold_details.to_csv(fold_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {fold_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare non-spatial vs spatially blocked RF CV"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to combined_data_true_master.csv",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/species_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results/tables",
        help="Output directory for summary tables",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)

    env_info = get_env_variables(df_filtered)
    env_vars = env_info["all_env"]

    summary_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []

    print(f"\nUsing spatial grouping variable: {GROUP_COL}")
    print(f"Study species: {', '.join(STUDY_SPECIES)}")

    for species in STUDY_SPECIES:
        print(f"\nAnalyzing {species} ...")
        summary, fold_df = analyse_species(
            df_filtered=df_filtered,
            species_name=species,
            env_vars=env_vars,
            n_splits=args.n_splits,
        )
        summary_rows.append(summary)
        fold_frames.append(fold_df)

    all_folds = pd.concat(fold_frames, ignore_index=True)
    print_summary_table(summary_rows)
    save_outputs(summary_rows, all_folds, Path(args.output_dir))


if __name__ == "__main__":
    main()