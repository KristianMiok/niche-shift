"""
data_preparation.py
Prepare analysis-ready datasets for each selected species.

For each qualifying species, produces:
  - Native-range dataset with environmental variables
  - Invasive-range dataset with environmental variables
  - Combined dataset with a range label (native/invasive)

Usage:
    python src/data_preparation.py --input data/raw/combined_data_true_master.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    load_geotraits, load_config, get_env_variables,
    COL_SPECIES, COL_STATUS, COL_ACCURACY, COL_DISTANCE, COL_SEGMENT,
    METADATA_COLS,
)
from species_selector import (
    apply_quality_filters, select_species,
    STATUS_NATIVE, STATUS_ALIEN,
)


def handle_missing_values(
    df: pd.DataFrame,
    env_vars: list,
    max_missing_pct: float = 0.3,
) -> tuple[pd.DataFrame, list]:
    """
    Handle missing values in environmental variables.

    1. Drop variables with >max_missing_pct missing
    2. Drop records with >50% missing env vars
    3. Impute remaining with column median

    Returns (cleaned_df, retained_vars).
    """
    n_start = len(df)
    vars_start = len(env_vars)

    # Step 1: Drop high-missing variables
    missing_pct = df[env_vars].isnull().mean()
    retained_vars = missing_pct[missing_pct <= max_missing_pct].index.tolist()
    n_dropped_vars = vars_start - len(retained_vars)
    if n_dropped_vars > 0:
        print(f"  Dropped {n_dropped_vars} variables with "
              f">{max_missing_pct*100:.0f}% missing")

    # Step 2: Drop records with too many missing
    record_missing = df[retained_vars].isnull().mean(axis=1)
    df = df[record_missing <= 0.5].copy()
    n_dropped_records = n_start - len(df)
    if n_dropped_records > 0:
        print(f"  Dropped {n_dropped_records} records with >50% missing env vars")

    # Step 3: Median imputation
    n_imputed = df[retained_vars].isnull().sum().sum()
    if n_imputed > 0:
        for var in retained_vars:
            if df[var].isnull().any():
                df[var] = df[var].fillna(df[var].median())
        print(f"  Imputed {n_imputed} remaining missing values (median)")

    print(f"  Result: {len(df)} records, {len(retained_vars)} variables")
    return df, retained_vars


def remove_constant_variables(df: pd.DataFrame, env_vars: list) -> list:
    """Remove variables with zero or near-zero variance."""
    stds = df[env_vars].std()
    constant = stds[stds < 1e-10].index.tolist()
    if constant:
        print(f"  Removed {len(constant)} constant variables")
    return [v for v in env_vars if v not in constant]


def remove_highly_correlated(
    df: pd.DataFrame, env_vars: list, threshold: float = 0.98
) -> list:
    """
    Remove one of each pair of highly correlated variables.

    Drops whichever has higher mean absolute correlation with all others.
    """
    if len(env_vars) < 2:
        return env_vars

    corr_matrix = df[env_vars].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        if high_corr:
            mean_corr_col = corr_matrix[col].mean()
            for hc in high_corr:
                mean_corr_hc = corr_matrix[hc].mean()
                to_drop.add(hc if mean_corr_hc > mean_corr_col else col)

    if to_drop:
        print(f"  Removed {len(to_drop)} highly correlated variables (r>{threshold})")
    return [v for v in env_vars if v not in to_drop]


def prepare_species_dataset(
    df: pd.DataFrame,
    species_name: str,
    env_vars: list,
    remove_correlated: bool = True,
    corr_threshold: float = 0.98,
) -> dict | None:
    """
    Prepare analysis-ready dataset for a single species.

    Returns dict with keys: 'species', 'native', 'invasive', 'combined',
    'env_vars', 'n_native', 'n_invasive'. None if insufficient data.
    """
    sp_data = df[df[COL_SPECIES] == species_name].copy()
    native = sp_data[sp_data[COL_STATUS] == STATUS_NATIVE]
    invasive = sp_data[sp_data[COL_STATUS] == STATUS_ALIEN]

    print(f"\n{'─'*50}")
    print(f"Preparing: {species_name}")
    print(f"  Native records:   {len(native)}")
    print(f"  Invasive records: {len(invasive)}")

    if len(native) < 10 or len(invasive) < 10:
        print(f"  SKIPPED: insufficient records")
        return None

    # Combine and clean
    combined = pd.concat([native, invasive], ignore_index=True)
    combined, clean_vars = handle_missing_values(combined, env_vars)

    # Remove constant variables within this species
    clean_vars = remove_constant_variables(combined, clean_vars)

    # Optionally remove highly correlated
    if remove_correlated:
        clean_vars = remove_highly_correlated(
            combined, clean_vars, threshold=corr_threshold
        )

    # Add range label for convenience
    combined["range_label"] = combined[COL_STATUS].map(
        {STATUS_NATIVE: "native", STATUS_ALIEN: "invasive"}
    )

    native_clean = combined[combined["range_label"] == "native"].copy()
    invasive_clean = combined[combined["range_label"] == "invasive"].copy()

    print(f"  Final: {len(native_clean)} native, {len(invasive_clean)} invasive, "
          f"{len(clean_vars)} env variables")

    return {
        "species": species_name,
        "native": native_clean,
        "invasive": invasive_clean,
        "combined": combined,
        "env_vars": clean_vars,
        "n_native": len(native_clean),
        "n_invasive": len(invasive_clean),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare analysis-ready species datasets"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to combined_data_true_master.csv")
    parser.add_argument("--config", "-c", default="config/species_config.yaml",
                        help="Path to species config YAML")
    parser.add_argument("--corr-threshold", type=float, default=0.98,
                        help="Correlation threshold for variable removal")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_geotraits(args.input)

    # Quality filter
    df_filtered = apply_quality_filters(df, config)

    # Identify env variables
    env_info = get_env_variables(df_filtered)
    env_vars = env_info["all_env"]
    print(f"\nTotal environmental variables available: {len(env_vars)}")

    if not env_vars:
        print("ERROR: No environmental variables found.")
        return

    # Select qualifying species
    summary = select_species(df_filtered, config)
    qualifying = summary[summary["qualifies"]]

    if qualifying.empty:
        print("No qualifying species found.")
        return

    # Prepare dataset for each qualifying species
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for _, row in qualifying.iterrows():
        sp_name = row[COL_SPECIES]
        result = prepare_species_dataset(
            df_filtered, sp_name, env_vars,
            remove_correlated=True,
            corr_threshold=args.corr_threshold,
        )
        if result is not None:
            prepared[sp_name] = result

            # Save per-species files
            safe_name = sp_name.replace(" ", "_").lower()
            result["combined"].to_csv(
                output_dir / f"{safe_name}_combined.csv", index=False
            )
            result["native"].to_csv(
                output_dir / f"{safe_name}_native.csv", index=False
            )
            result["invasive"].to_csv(
                output_dir / f"{safe_name}_invasive.csv", index=False
            )
            # Save variable list
            pd.Series(result["env_vars"]).to_csv(
                output_dir / f"{safe_name}_env_vars.txt",
                index=False, header=False,
            )

    # Summary
    print(f"\n{'='*70}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"Species prepared: {len(prepared)}")
    for sp_name, result in prepared.items():
        print(f"  {sp_name}: {result['n_native']} native, "
              f"{result['n_invasive']} invasive, "
              f"{len(result['env_vars'])} variables")
    print(f"\nOutput saved to {output_dir}/")


if __name__ == "__main__":
    main()
