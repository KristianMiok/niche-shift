"""
niche_overlap_metrics.py

Compute classical niche overlap metrics between native and invasive
environmental space for each focal species.

Approach:
- Use the same filtered, cleaned environmental predictors as in the ML pipeline
- For each species:
    1. combine native + invasive records
    2. clean variables (missingness, constants, correlation)
    3. standardize predictors
    4. project to first 2 PCA axes
    5. estimate occupancy density in shared 2D environmental space
    6. compute Schoener's D and Warren's I

Outputs:
- species-level table with D and I
- per-species PCA niche plots
- summary figure

Usage:
    python src/niche_overlap_metrics.py --input data/raw/combined_data_true_master.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


# ---------------------------------------------------------------------
# OVERLAP METRICS
# ---------------------------------------------------------------------

def schoeners_d(p: np.ndarray, q: np.ndarray) -> float:
    """
    Schoener's D:
        D = 1 - 0.5 * sum(|p_i - q_i|)
    where p and q are normalized probability distributions.
    """
    p = p.ravel()
    q = q.ravel()
    return float(1.0 - 0.5 * np.abs(p - q).sum())


def warrens_i(p: np.ndarray, q: np.ndarray) -> float:
    """
    Warren's I:
        I = sum(sqrt(p_i * q_i))
    Equivalent to 1 - 0.5 * sum((sqrt(p_i)-sqrt(q_i))^2)
    """
    p = p.ravel()
    q = q.ravel()
    return float(np.sqrt(p * q).sum())


def estimate_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Estimate smoothed 2D occupancy density on a shared grid.
    """
    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    if sigma is not None and sigma > 0:
        hist = gaussian_filter(hist, sigma=sigma)

    total = hist.sum()
    if total == 0:
        return hist

    return hist / total


# ---------------------------------------------------------------------
# SPECIES ANALYSIS
# ---------------------------------------------------------------------

def prepare_species_data(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Get cleaned species data and cleaned variable list.
    """
    sp = df_filtered[df_filtered[COL_SPECIES] == species_name].copy()

    combined, clean_vars = handle_missing_values(
        sp, env_vars, max_missing_pct=0.30
    )
    clean_vars = remove_constant_variables(combined, clean_vars)
    clean_vars = remove_highly_correlated(combined, clean_vars, threshold=0.98)

    return combined, clean_vars


def compute_species_overlap(
    df_filtered: pd.DataFrame,
    species_name: str,
    env_vars: list[str],
    n_bins: int = 100,
    sigma: float = 1.0,
) -> dict:
    """
    Compute PCA-based overlap metrics for one species.
    """
    combined, clean_vars = prepare_species_data(df_filtered, species_name, env_vars)

    native = combined[combined[COL_STATUS] == STATUS_NATIVE].copy()
    invasive = combined[combined[COL_STATUS] == STATUS_ALIEN].copy()

    if len(native) < 10 or len(invasive) < 10:
        raise ValueError(
            f"{species_name}: too few records after filtering "
            f"(native={len(native)}, invasive={len(invasive)})"
        )

    X = combined[clean_vars].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    combined["PC1"] = X_pca[:, 0]
    combined["PC2"] = X_pca[:, 1]

    native = combined[combined[COL_STATUS] == STATUS_NATIVE].copy()
    invasive = combined[combined[COL_STATUS] == STATUS_ALIEN].copy()

    # Shared environmental-space bounds
    x_min = combined["PC1"].min()
    x_max = combined["PC1"].max()
    y_min = combined["PC2"].min()
    y_max = combined["PC2"].max()

    # small padding
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, n_bins + 1)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, n_bins + 1)

    native_grid = estimate_density_grid(
        native["PC1"].values, native["PC2"].values,
        x_edges, y_edges, sigma=sigma
    )
    invasive_grid = estimate_density_grid(
        invasive["PC1"].values, invasive["PC2"].values,
        x_edges, y_edges, sigma=sigma
    )

    D = schoeners_d(native_grid, invasive_grid)
    I = warrens_i(native_grid, invasive_grid)

    return {
        "species": species_name,
        "n_native": int(len(native)),
        "n_invasive": int(len(invasive)),
        "n_features": int(len(clean_vars)),
        "pc1_variance_explained": float(pca.explained_variance_ratio_[0]),
        "pc2_variance_explained": float(pca.explained_variance_ratio_[1]),
        "schoeners_D": float(D),
        "warrens_I": float(I),
        "combined": combined,
        "native_grid": native_grid,
        "invasive_grid": invasive_grid,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def safe_name(species_name: str) -> str:
    return species_name.replace(" ", "_").lower()


def plot_species_pca(
    result: dict,
    output_dir: str = "results/figures/niche_overlap",
) -> str:
    """
    Scatter plot in PCA space.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = result["combined"]
    native = df[df[COL_STATUS] == STATUS_NATIVE]
    invasive = df[df[COL_STATUS] == STATUS_ALIEN]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        native["PC1"], native["PC2"],
        s=10, alpha=0.45, label="Native"
    )
    ax.scatter(
        invasive["PC1"], invasive["PC2"],
        s=10, alpha=0.45, label="Invasive"
    )

    ax.set_xlabel(
        f"PC1 ({result['pc1_variance_explained'] * 100:.1f}% var.)"
    )
    ax.set_ylabel(
        f"PC2 ({result['pc2_variance_explained'] * 100:.1f}% var.)"
    )
    ax.set_title(
        f"{result['species']}\n"
        f"Schoener's D = {result['schoeners_D']:.3f}, "
        f"Warren's I = {result['warrens_I']:.3f}"
    )
    ax.legend()
    plt.tight_layout()

    path = out / f"{safe_name(result['species'])}_pca_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_species_density(
    result: dict,
    output_dir: str = "results/figures/niche_overlap",
) -> str:
    """
    Side-by-side density maps in PCA space.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    native_grid = result["native_grid"].T
    invasive_grid = result["invasive_grid"].T
    x_edges = result["x_edges"]
    y_edges = result["y_edges"]

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    im1 = axes[0].imshow(
        native_grid, origin="lower", extent=extent, aspect="auto"
    )
    axes[0].set_title("Native density")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(
        invasive_grid, origin="lower", extent=extent, aspect="auto"
    )
    axes[1].set_title("Invasive density")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    fig.suptitle(
        f"{result['species']} | D = {result['schoeners_D']:.3f}, "
        f"I = {result['warrens_I']:.3f}"
    )
    plt.tight_layout()

    path = out / f"{safe_name(result['species'])}_density_maps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_summary_overlap(
    summary_df: pd.DataFrame,
    output_dir: str = "results/figures/niche_overlap",
) -> str:
    """
    Summary bar chart for D and I across species.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = summary_df.copy()
    df["short"] = df["species"].apply(
        lambda s: s.split(" ")[0][0] + ". " + s.split(" ")[1]
    )

    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, df["schoeners_D"], width, label="Schoener's D")
    ax.bar(x + width / 2, df["warrens_I"], width, label="Warren's I")

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=0, fontsize=10, style="italic")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Overlap")
    ax.set_title("Classical niche overlap metrics: native vs. invasive")
    ax.legend()

    plt.tight_layout()
    path = out / "niche_overlap_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute Schoener's D and Warren's I"
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
        "--bins", type=int, default=100,
        help="Number of 2D grid bins per axis (default: 100)"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="Gaussian smoothing sigma for density grids (default: 1.0)"
    )
    args = parser.parse_args()

    output_tables = Path("results/tables")
    output_tables.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input}...")
    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)

    env_info = get_env_variables(df_filtered)
    env_vars = env_info["all_env"]

    print("\n" + "=" * 80)
    print("NICHE OVERLAP METRICS")
    print("=" * 80)

    all_results = []
    summary_rows = []

    for species in STUDY_SPECIES:
        print(f"\nAnalyzing {species}...")
        result = compute_species_overlap(
            df_filtered,
            species,
            env_vars,
            n_bins=args.bins,
            sigma=args.sigma,
        )

        print(
            f"  Native n = {result['n_native']}, "
            f"Invasive n = {result['n_invasive']}, "
            f"Features = {result['n_features']}"
        )
        print(
            f"  PCA variance explained: "
            f"PC1 = {result['pc1_variance_explained']:.3f}, "
            f"PC2 = {result['pc2_variance_explained']:.3f}"
        )
        print(f"  Schoener's D = {result['schoeners_D']:.4f}")
        print(f"  Warren's I   = {result['warrens_I']:.4f}")

        scatter_path = plot_species_pca(result)
        density_path = plot_species_density(result)

        print(f"  Saved scatter plot: {scatter_path}")
        print(f"  Saved density plot: {density_path}")

        summary_rows.append({
            "species": result["species"],
            "n_native": result["n_native"],
            "n_invasive": result["n_invasive"],
            "n_features": result["n_features"],
            "pc1_variance_explained": result["pc1_variance_explained"],
            "pc2_variance_explained": result["pc2_variance_explained"],
            "schoeners_D": result["schoeners_D"],
            "warrens_I": result["warrens_I"],
        })

        all_results.append({
            "species": result["species"],
            "n_native": result["n_native"],
            "n_invasive": result["n_invasive"],
            "n_features": result["n_features"],
            "pc1_variance_explained": result["pc1_variance_explained"],
            "pc2_variance_explained": result["pc2_variance_explained"],
            "schoeners_D": result["schoeners_D"],
            "warrens_I": result["warrens_I"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("schoeners_D", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    summary_path = output_tables / "niche_overlap_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    json_path = output_tables / "niche_overlap_summary.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_fig = plot_summary_overlap(summary_df)
    print(f"\nSaved summary table: {summary_path}")
    print(f"Saved summary JSON:  {json_path}")
    print(f"Saved summary plot:  {summary_fig}")


if __name__ == "__main__":
    main()