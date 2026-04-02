"""
species_selector.py
Select species meeting inclusion criteria for niche shift analysis.

A species is included if it has sufficient occurrence records in BOTH
its native and invasive range after quality filtering.

Usage:
    python src/species_selector.py --input data/raw/combined_data_true_master.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from data_loader import (
    load_geotraits, load_config,
    COL_SPECIES, COL_STATUS, COL_ACCURACY, COL_DISTANCE, COL_SEGMENT,
)

# Expected status values (adjust if data uses different labels)
STATUS_NATIVE = "Native"
STATUS_ALIEN = "Alien"


def apply_quality_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply the quality filtering pipeline.

    Steps:
        1. Keep only High accuracy records
        2. Remove records beyond max snapping distance (1 km)
        3. Deduplicate by segment per species per status

    Parameters
    ----------
    df : pd.DataFrame
        Raw data.
    config : dict
        Selection criteria from species_config.yaml.

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    criteria = config["selection_criteria"]
    n_start = len(df)
    print(f"\nApplying quality filters (starting with {n_start:,} records)...")

    # Step 1: Accuracy filter
    acc_val = criteria.get("accuracy_filter", "High")
    df = df[df[COL_ACCURACY] == acc_val].copy()
    print(f"  After accuracy filter ('{acc_val}'): {len(df):,} records")

    # Step 2: Distance filter
    max_dist = criteria.get("max_snapping_distance", 1000)
    df = df[df[COL_DISTANCE] <= max_dist].copy()
    print(f"  After distance filter (<={max_dist}m): {len(df):,} records")

    # Step 3: Deduplication by segment per species per status
    dedup_cols = [COL_SPECIES, COL_STATUS, COL_SEGMENT]
    n_before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()
    print(f"  After deduplication: {len(df):,} records "
          f"(removed {n_before - len(df):,})")

    print(f"  Total removed: {n_start - len(df):,} records")
    return df


def select_species(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Identify species with sufficient records in both native and invasive ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Quality-filtered data.
    config : dict
        Configuration with selection criteria and candidate species.

    Returns
    -------
    pd.DataFrame
        Summary table with record counts and qualification flag.
    """
    criteria = config["selection_criteria"]
    min_per_range = criteria.get("min_records_per_range", 40)

    # Count records per species per status
    counts = (
        df.groupby([COL_SPECIES, COL_STATUS])
        .size()
        .reset_index(name="n_records")
    )

    # Report actual status values
    status_values = df[COL_STATUS].unique()
    print(f"\nStatus values in filtered data: {status_values.tolist()}")

    # Pivot to native/alien counts
    native_counts = (
        counts[counts[COL_STATUS] == STATUS_NATIVE]
        .rename(columns={"n_records": "n_native"})
        .drop(columns=[COL_STATUS])
    )
    alien_counts = (
        counts[counts[COL_STATUS] == STATUS_ALIEN]
        .rename(columns={"n_records": "n_alien"})
        .drop(columns=[COL_STATUS])
    )

    summary = pd.merge(
        native_counts, alien_counts, on=COL_SPECIES, how="outer"
    ).fillna(0)
    summary["n_native"] = summary["n_native"].astype(int)
    summary["n_alien"] = summary["n_alien"].astype(int)
    summary["n_total"] = summary["n_native"] + summary["n_alien"]
    summary["min_range_n"] = summary[["n_native", "n_alien"]].min(axis=1)

    # Qualification flag
    summary["qualifies"] = summary["min_range_n"] >= min_per_range

    # Mark candidate species
    candidates = config.get("candidate_species", [])
    candidate_names = [c["name"] for c in candidates]
    summary["is_candidate"] = summary[COL_SPECIES].isin(candidate_names)

    summary = summary.sort_values(
        ["qualifies", "is_candidate", "n_total"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Select species for niche shift analysis"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to combined_data_true_master.csv")
    parser.add_argument("--config", "-c", default="config/species_config.yaml",
                        help="Path to species config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_geotraits(args.input)

    # Quality filter
    df_filtered = apply_quality_filters(df, config)

    # Select species
    summary = select_species(df_filtered, config)

    if summary.empty:
        print("\nNo species summary could be generated.")
        return

    min_n = config["selection_criteria"]["min_records_per_range"]
    qualifying = summary[summary["qualifies"]]
    has_alien = summary[summary["n_alien"] > 0]

    print(f"\n{'='*75}")
    print(f"SPECIES SELECTION RESULTS (min {min_n} records per range)")
    print(f"{'='*75}")

    print(f"\nQualifying species ({len(qualifying)}):")
    if not qualifying.empty:
        print(qualifying.to_string(index=False))
    else:
        print("  None found. Consider lowering min_records_per_range.")

    print(f"\nAll species with any Alien records ({len(has_alien)}):")
    if not has_alien.empty:
        print(has_alien.to_string(index=False))

    # Save
    out_path = Path("data/interim/species_selection_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nFull summary saved to {out_path}")


if __name__ == "__main__":
    main()
