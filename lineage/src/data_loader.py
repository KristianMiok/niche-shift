"""
data_loader.py
Load and validate the raw GeoTraits Full Integrated Dataset.

Expected input: combined_data_true_master.csv from the Global Crayfish
Database of Geospatial Traits.

Usage:
    python src/data_loader.py --input data/raw/combined_data_true_master.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# === Exact column names from the GeoTraits dataset ===
COL_SPECIES = "Crayfish_scientific_name"
COL_STATUS = "Status"
COL_ACCURACY = "Accuracy"
COL_DISTANCE = "distance_m"
COL_SEGMENT = "subc_id"
COL_WOCID = "WoCID"
COL_BASIN = "basin_id"
COL_STRAHLER = "strahler"
COL_LAT = "lat_or"
COL_LON = "long_or"
COL_LAT_SNAP = "lat_snap"
COL_LON_SNAP = "long_snap"
COL_AB200 = "ab_200m"
COL_AB500 = "ab_500m"
COL_AB1000 = "ab_1000m"

# All metadata columns (non-environmental)
METADATA_COLS = [
    COL_WOCID, COL_LAT, COL_LON, COL_ACCURACY, COL_SPECIES, COL_STATUS,
    "Year_of_record", COL_BASIN, COL_SEGMENT, "reg_id", COL_STRAHLER,
    "area_sqm", "sum_area_sqm", COL_LAT_SNAP, COL_LON_SNAP,
    "hylak_id", "is_coastal", COL_DISTANCE, COL_AB200, COL_AB500, COL_AB1000,
]


def load_config(config_path: str = "config/species_config.yaml") -> dict:
    """Load the species configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_geotraits(filepath: str) -> pd.DataFrame:
    """
    Load the GeoTraits Full Integrated Dataset.

    Parameters
    ----------
    filepath : str
        Path to combined_data_true_master.csv (or .xlsx).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading data from {filepath}...")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(filepath, low_memory=False)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def get_env_variables(df: pd.DataFrame) -> dict:
    """
    Identify environmental variable columns by prefix.

    Returns dict with keys: 'local_cli', 'local_top', 'local_sol', 'local_lac',
    'upstream_cli', 'upstream_top', 'upstream_sol', 'upstream_lac',
    'all_local', 'all_upstream', 'all_env'.
    """
    cols = df.columns.tolist()

    result = {
        "local_cli": sorted([c for c in cols if c.startswith("l_CLI")]),
        "local_top": sorted([c for c in cols if c.startswith("l_TOP")]),
        "local_sol": sorted([c for c in cols if c.startswith("l_SOL")]),
        "local_lac": sorted([c for c in cols if c.startswith("l_LAC")]),
        "upstream_cli": sorted([c for c in cols if c.startswith("u_CLI")]),
        "upstream_top": sorted([c for c in cols if c.startswith("u_TOP")]),
        "upstream_sol": sorted([c for c in cols if c.startswith("u_SOL")]),
        "upstream_lac": sorted([c for c in cols if c.startswith("u_LAC")]),
    }

    result["all_local"] = sorted(
        result["local_cli"] + result["local_top"] +
        result["local_sol"] + result["local_lac"]
    )
    result["all_upstream"] = sorted(
        result["upstream_cli"] + result["upstream_top"] +
        result["upstream_sol"] + result["upstream_lac"]
    )
    result["all_env"] = sorted(result["all_local"] + result["all_upstream"])

    return result


def validate_data(df: pd.DataFrame) -> dict:
    """Validate the loaded dataset against expected structure."""
    report = {
        "status": "OK",
        "n_records": len(df),
        "n_columns": len(df.columns),
        "warnings": [],
    }

    # Check required columns
    required = [COL_SPECIES, COL_STATUS, COL_ACCURACY, COL_DISTANCE, COL_SEGMENT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        report["warnings"].append(f"Missing required columns: {missing}")
        report["status"] = "ERROR"

    # Environmental variables
    env = get_env_variables(df)
    report["env_counts"] = {
        "local_cli": len(env["local_cli"]),
        "local_top": len(env["local_top"]),
        "local_sol": len(env["local_sol"]),
        "local_lac": len(env["local_lac"]),
        "upstream_cli": len(env["upstream_cli"]),
        "upstream_top": len(env["upstream_top"]),
        "upstream_sol": len(env["upstream_sol"]),
        "upstream_lac": len(env["upstream_lac"]),
        "total_local": len(env["all_local"]),
        "total_upstream": len(env["all_upstream"]),
        "total_env": len(env["all_env"]),
    }

    # Status values
    if COL_STATUS in df.columns:
        report["status_values"] = df[COL_STATUS].value_counts().to_dict()

    # Accuracy values
    if COL_ACCURACY in df.columns:
        report["accuracy_values"] = df[COL_ACCURACY].value_counts().to_dict()

    # Missing data
    if env["all_env"]:
        null_pct = df[env["all_env"]].isnull().mean()
        high_null = null_pct[null_pct > 0.3]
        if len(high_null) > 0:
            report["warnings"].append(
                f"{len(high_null)} env variables have >30% missing values"
            )

    if report["warnings"]:
        report["status"] = "WARNINGS" if report["status"] == "OK" else report["status"]

    return report


def print_report(report: dict) -> None:
    """Pretty-print the validation report."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Status:       {report['status']}")
    print(f"Records:      {report['n_records']:,}")
    print(f"Columns:      {report['n_columns']}")

    if "status_values" in report:
        print(f"\nStatus values:")
        for val, count in report["status_values"].items():
            print(f"  {val}: {count:,}")

    if "accuracy_values" in report:
        print(f"\nAccuracy values:")
        for val, count in report["accuracy_values"].items():
            print(f"  {val}: {count:,}")

    env = report.get("env_counts", {})
    if env:
        print(f"\nEnvironmental variables:")
        print(f"  Local  - CLI: {env['local_cli']}, TOP: {env['local_top']}, "
              f"SOL: {env['local_sol']}, LAC: {env['local_lac']}")
        print(f"  Upstream - CLI: {env['upstream_cli']}, TOP: {env['upstream_top']}, "
              f"SOL: {env['upstream_sol']}, LAC: {env['upstream_lac']}")
        print(f"  Total: {env['total_env']} "
              f"({env['total_local']} local + {env['total_upstream']} upstream)")

    if report["warnings"]:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  - {w}")

    print("=" * 60)


def summarize_species(df: pd.DataFrame) -> pd.DataFrame:
    """Record counts per species and status."""
    return (
        df.groupby([COL_SPECIES, COL_STATUS])
        .size()
        .reset_index(name="n_records")
        .sort_values("n_records", ascending=False)
    )


def main():
    parser = argparse.ArgumentParser(description="Load and validate GeoTraits data")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to combined_data_true_master.csv")
    args = parser.parse_args()

    df = load_geotraits(args.input)
    report = validate_data(df)
    print_report(report)

    # Species summary
    print(f"\nSpecies x Status record counts (top 40):")
    summary = summarize_species(df)
    print(summary.head(40).to_string(index=False))

    # Save
    out_path = Path("data/interim/species_record_counts.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nFull summary saved to {out_path}")

    # Species with both Native and Alien records
    pivot = summary.pivot_table(
        index=COL_SPECIES, columns=COL_STATUS,
        values="n_records", fill_value=0
    ).reset_index()

    status_cols = [c for c in pivot.columns if c != COL_SPECIES]
    print(f"\nStatus values found: {status_cols}")

    multi = pivot[pivot[status_cols].gt(0).sum(axis=1) > 1]
    if not multi.empty:
        print(f"\nSpecies with records in multiple status categories:")
        print(multi.to_string(index=False))
        print(f"\n  Total: {len(multi)} species")
    else:
        print("\nNo species found with records in multiple status categories.")


if __name__ == "__main__":
    main()
