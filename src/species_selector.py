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

# Status values: only Native vs Alien (ignoring Introduced and Type locality)
STATUS_NATIVE = "Native"
STATUS_ALIEN = "Alien"

# Core study species selected based on data availability
STUDY_SPECIES = [
    "Procambarus clarkii",       # Native: 2208, Alien: 8527
    "Faxonius limosus",          # Native: 382,  Alien: 4089
    "Pacifastacus leniusculus",  # Native: 117,  Alien: 4459
    "Faxonius virilis",          # Native: 1856, Alien: 500
    "Faxonius rusticus",         # Native: 1223, Alien: 670
 #   "Procambarus acutus",       # Native: 122,  Alien: 1200
]

def apply_quality_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply the quality filtering pipeline.

    Steps:
        1. Keep only Native and Alien status (drop Introduced, Type locality)
        2. Keep only High accuracy records
        3. Remove records beyond max snapping distance (1 km)
        4. Deduplicate by segment per species per status
    """
    criteria = config["selection_criteria"]
    n_start = len(df)
    print(f"\nApplying quality filters (starting with {n_start:,} records)...")

    # Step 0: Keep only Native and Alien (drop Introduced, Type locality)
    df = df[df[COL_STATUS].isin([STATUS_NATIVE, STATUS_ALIEN])].copy()
    print(f"  After status filter (Native/Alien only): {len(df):,} records")

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

    Filters to STUDY_SPECIES list and checks minimum record counts.
    """
    criteria = config["selection_criteria"]
    min_per_range = criteria.get("min_records_per_range", 40)

    # Filter to study species only
    df_study = df[df[COL_SPECIES].isin(STUDY_SPECIES)]
    print(f"\nRecords for study species: {len(df_study):,}")

    # Count records per species per status
    counts = (
        df_study.groupby([COL_SPECIES, COL_STATUS])
        .size()
        .reset_index(name="n_records")
    )

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
    summary["qualifies"] = summary["min_range_n"] >= min_per_range

    summary = summary.sort_values("n_total", ascending=False).reset_index(drop=True)
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

    print(f"\n{'='*75}")
    print(f"SPECIES SELECTION RESULTS (min {min_n} records per range)")
    print(f"{'='*75}")
    print(f"\nStudy species ({len(summary)}):")
    print(summary.to_string(index=False))

    not_qual = summary[~summary["qualifies"]]
    if not not_qual.empty:
        print(f"\nWARNING: {len(not_qual)} species below threshold:")
        print(not_qual[[COL_SPECIES, "n_native", "n_alien"]].to_string(index=False))
        print("  Consider lowering min_records_per_range or removing these species.")

    # Save
    out_path = Path("data/interim/species_selection_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()