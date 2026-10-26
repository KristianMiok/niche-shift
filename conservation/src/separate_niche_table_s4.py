from __future__ import annotations

"""
separate_niche_table_s4.py

Build an expanded supplementary Table S4 for the conservation manuscript.

The table includes, for each focal species and each range-specific niche model:
- presence sample size
- pseudo-absence sample size
- whether the background was biogeographically constrained
- top 5 variables + importances for the native model
- top 5 variables + importances for the non-native model

This script reconstructs the separate niche models directly from the current
pipeline settings, so the output is transparent and reproducible.

Usage:
    python src/separate_niche_table_s4.py --input data/raw/combined_data_true_master.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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
    STATUS_ALIEN,   # in this branch = "Introduced"
    STUDY_SPECIES,
)
from data_preparation import (
    handle_missing_values,
    remove_constant_variables,
    remove_highly_correlated,
)
from decision_tree import extract_feature_importances
from variable_glossary import load_glossary, translate


def sample_background(
    df_all: pd.DataFrame,
    target_species: str,
    status: str,
    n_presence: int,
    n_ratio: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    """
    Sample pseudo-absence/background points for a species and status.

    Background is constrained to records from other species with the same status.
    If that yields too few samples, it falls back to all other species.

    Returns:
        background_df, background_description
    """
    background_pool = df_all[
        (df_all[COL_SPECIES] != target_species) &
        (df_all[COL_STATUS] == status)
    ].copy()

    n_sample = min(int(n_presence * n_ratio), len(background_pool))

    if n_sample < 10:
        background_pool = df_all[df_all[COL_SPECIES] != target_species].copy()
        n_sample = min(int(n_presence * n_ratio), len(background_pool))
        bg_desc = (
            "Fallback: not enough same-status records; background drawn from all other species"
        )
    else:
        bg_desc = (
            f"Yes: background constrained to other species with the same status ({status})"
        )

    background = background_pool.sample(
        n=n_sample,
        replace=False,
        random_state=random_state,
    ).copy()

    return background, bg_desc


def train_niche_model(
    presence: pd.DataFrame,
    background: pd.DataFrame,
    env_vars: list[str],
    max_depth: int = 4,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, list[str], pd.DataFrame]:
    """
    Train a presence-vs-background decision tree niche model.

    Returns:
        fitted_clf, clean_feature_names, combined_dataframe
    """
    presence = presence.copy()
    background = background.copy()

    presence["_target"] = 1
    background["_target"] = 0

    combined = pd.concat([presence, background], ignore_index=True)

    combined, clean_vars = handle_missing_values(
        combined,
        env_vars,
        max_missing_pct=0.3,
    )
    clean_vars = remove_constant_variables(combined, clean_vars)
    clean_vars = remove_highly_correlated(combined, clean_vars, threshold=0.98)

    X = combined[clean_vars].to_numpy()
    y = combined["_target"].to_numpy()

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X, y)

    return clf, clean_vars, combined


def top5_with_glossary(
    importances_df: pd.DataFrame,
    glossary: dict[str, str],
) -> list[dict[str, Any]]:
    top5 = []
    for _, row in importances_df.head(5).iterrows():
        var = row["variable"]
        imp = float(row["importance"])
        top5.append({
            "variable": var,
            "description": translate(var, glossary),
            "importance": imp,
        })
    return top5


def fmt_top5_text(top5: list[dict[str, Any]]) -> str:
    parts = []
    for item in top5:
        parts.append(
            f"{item['variable']} ({item['importance']:.3f}; {item['description']})"
        )
    return " | ".join(parts)


def analyse_species(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list[str],
    glossary: dict[str, str],
    max_depth: int = 4,
) -> dict[str, Any]:
    sp_df = df_filtered[df_filtered[COL_SPECIES] == species_name].copy()

    native_presence = sp_df[sp_df[COL_STATUS] == STATUS_NATIVE].copy()
    nonnative_presence = sp_df[sp_df[COL_STATUS] == STATUS_ALIEN].copy()

    native_bg, native_bg_desc = sample_background(
        df_all=df_filtered,
        target_species=species_name,
        status=STATUS_NATIVE,
        n_presence=len(native_presence),
        n_ratio=1.0,
        random_state=42,
    )
    nonnative_bg, nonnative_bg_desc = sample_background(
        df_all=df_filtered,
        target_species=species_name,
        status=STATUS_ALIEN,
        n_presence=len(nonnative_presence),
        n_ratio=1.0,
        random_state=42,
    )

    native_clf, native_vars, _ = train_niche_model(
        presence=native_presence,
        background=native_bg,
        env_vars=env_vars,
        max_depth=max_depth,
        random_state=42,
    )
    nonnative_clf, nonnative_vars, _ = train_niche_model(
        presence=nonnative_presence,
        background=nonnative_bg,
        env_vars=env_vars,
        max_depth=max_depth,
        random_state=43,
    )

    native_imp = extract_feature_importances(native_clf, native_vars)
    nonnative_imp = extract_feature_importances(nonnative_clf, nonnative_vars)

    native_top5 = top5_with_glossary(native_imp, glossary)
    nonnative_top5 = top5_with_glossary(nonnative_imp, glossary)

    return {
        "species": species_name,
        "native_presence_n": int(len(native_presence)),
        "native_pseudoabsence_n": int(len(native_bg)),
        "native_background_constrained": native_bg_desc,
        "native_top5": native_top5,

        "non_native_label": STATUS_ALIEN,
        "non_native_presence_n": int(len(nonnative_presence)),
        "non_native_pseudoabsence_n": int(len(nonnative_bg)),
        "non_native_background_constrained": nonnative_bg_desc,
        "non_native_top5": nonnative_top5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build expanded supplementary Table S4"
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
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results/tables",
        help="Output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)
    env_vars = get_env_variables(df_filtered)["all_env"]
    glossary = load_glossary("data/raw/S2.xlsx")

    all_results: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for species in STUDY_SPECIES:
        res = analyse_species(
            df_filtered=df_filtered,
            species_name=species,
            env_vars=env_vars,
            glossary=glossary,
            max_depth=4,
        )
        all_results.append(res)

        compact_rows.append({
            "species": res["species"],
            "native_presence_n": res["native_presence_n"],
            "native_pseudoabsence_n": res["native_pseudoabsence_n"],
            "native_background_constrained": res["native_background_constrained"],
            "native_top5_variables": fmt_top5_text(res["native_top5"]),
            "non_native_label": res["non_native_label"],
            "non_native_presence_n": res["non_native_presence_n"],
            "non_native_pseudoabsence_n": res["non_native_pseudoabsence_n"],
            "non_native_background_constrained": res["non_native_background_constrained"],
            "non_native_top5_variables": fmt_top5_text(res["non_native_top5"]),
        })

        for rank, item in enumerate(res["native_top5"], start=1):
            long_rows.append({
                "species": res["species"],
                "model": "native",
                "rank": rank,
                "variable": item["variable"],
                "description": item["description"],
                "importance": item["importance"],
                "presence_n": res["native_presence_n"],
                "pseudoabsence_n": res["native_pseudoabsence_n"],
                "background_constrained": res["native_background_constrained"],
            })

        for rank, item in enumerate(res["non_native_top5"], start=1):
            long_rows.append({
                "species": res["species"],
                "model": "non_native",
                "rank": rank,
                "variable": item["variable"],
                "description": item["description"],
                "importance": item["importance"],
                "presence_n": res["non_native_presence_n"],
                "pseudoabsence_n": res["non_native_pseudoabsence_n"],
                "background_constrained": res["non_native_background_constrained"],
            })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compact_csv = out_dir / "table_s4_expanded.csv"
    long_csv = out_dir / "table_s4_expanded_long.csv"
    json_path = out_dir / "table_s4_expanded.json"

    pd.DataFrame(compact_rows).to_csv(compact_csv, index=False)
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nExpanded Table S4 saved:")
    print(f"  {compact_csv}")
    print(f"  {long_csv}")
    print(f"  {json_path}")

    print("\nCompact preview:")
    print(pd.DataFrame(compact_rows).to_string(index=False))


if __name__ == "__main__":
    main()