from __future__ import annotations

"""
niche_overlap_conservation_figure.py

Regenerate conservation-paper niche overlap figures with species-specific labels
and assemble a combined Figure 3 panel.

Outputs:
- results/figures/niche_overlap/<species>_pca_scatter.png
- results/figures/niche_overlap/<species>_density_maps.png
- results/figures/niche_overlap/conservation_figure3_niche_overlap_panels.png

Usage:
    python src/niche_overlap_conservation_figure.py --input data/raw/combined_data_true_master.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    STATUS_ALIEN,
    STUDY_SPECIES,
)
from data_preparation import (
    handle_missing_values,
    remove_constant_variables,
    remove_highly_correlated,
)


OUT_DIR = Path("results/figures/niche_overlap")


def short_name(species: str) -> str:
    parts = species.split()
    return f"{parts[0][0]}. {parts[1]}"


def non_native_label(species: str) -> str:
    mapping = {
        "Pontastacus leptodactylus": "Cryptogenic",
        "Austropotamobius fulcisianus": "Introduced",
        "Austropotamobius pallipes": "Introduced",
    }
    return mapping.get(species, "Non-native")


def safe_name(species: str) -> str:
    return species.replace(" ", "_").lower()


def prepare_species_matrix(
    df_filtered: pd.DataFrame,
    species: str,
    env_vars: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    sp = df_filtered[df_filtered[COL_SPECIES] == species].copy()

    sp, clean_vars = handle_missing_values(sp, env_vars)
    clean_vars = remove_constant_variables(sp, clean_vars)
    clean_vars = remove_highly_correlated(sp, clean_vars, threshold=0.98)

    sp = sp[[COL_SPECIES, COL_STATUS] + clean_vars].copy()
    sp["range_label"] = sp[COL_STATUS].map({
        STATUS_NATIVE: "Native",
        STATUS_ALIEN: non_native_label(species),
    })

    return sp, clean_vars


def make_pca(sp_df: pd.DataFrame, clean_vars: list[str]) -> tuple[pd.DataFrame, PCA]:
    X = sp_df[clean_vars].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    out = sp_df[[COL_STATUS, "range_label"]].copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    return out, pca


def compute_density_grid(x: np.ndarray, y: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    values = np.vstack([x, y])
    kde = gaussian_kde(values)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid_coords).reshape(xx.shape)
    return zz


def plot_species_scatter(
    species: str,
    coords_df: pd.DataFrame,
    pca: PCA,
) -> Path:
    native_lab = "Native"
    nonnat_lab = non_native_label(species)

    fig, ax = plt.subplots(figsize=(6, 5))

    native = coords_df[coords_df["range_label"] == native_lab]
    nonnat = coords_df[coords_df["range_label"] == nonnat_lab]

    ax.scatter(
        native["PC1"], native["PC2"],
        s=18, alpha=0.55, label=native_lab
    )
    ax.scatter(
        nonnat["PC1"], nonnat["PC2"],
        s=18, alpha=0.55, label=nonnat_lab
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"{short_name(species)}: PCA overlap")
    ax.legend(frameon=True)

    out = OUT_DIR / f"{safe_name(species)}_pca_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_species_density(
    species: str,
    coords_df: pd.DataFrame,
    pca: PCA,
) -> Path:
    native_lab = "Native"
    nonnat_lab = non_native_label(species)

    native = coords_df[coords_df["range_label"] == native_lab]
    nonnat = coords_df[coords_df["range_label"] == nonnat_lab]

    x_all = coords_df["PC1"].to_numpy()
    y_all = coords_df["PC2"].to_numpy()

    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()

    pad_x = 0.08 * (x_max - x_min)
    pad_y = 0.08 * (y_max - y_min)

    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, 200),
        np.linspace(y_min - pad_y, y_max + pad_y, 200),
    )

    z_native = compute_density_grid(
        native["PC1"].to_numpy(), native["PC2"].to_numpy(), xx, yy
    )
    z_nonnat = compute_density_grid(
        nonnat["PC1"].to_numpy(), nonnat["PC2"].to_numpy(), xx, yy
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)

    im0 = axes[0].contourf(xx, yy, z_native, levels=12)
    axes[0].set_title("Native")

    im1 = axes[1].contourf(xx, yy, z_nonnat, levels=12)
    axes[1].set_title(nonnat_lab)

    overlap = np.minimum(z_native, z_nonnat)
    im2 = axes[2].contourf(xx, yy, overlap, levels=12)
    axes[2].set_title("Overlap")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")

    fig.suptitle(f"{short_name(species)}: PCA density maps", y=1.02)

    out = OUT_DIR / f"{safe_name(species)}_density_maps.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined_figure3(results: list[dict]) -> Path:
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(12, 15),
        gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    panel_labels = ["A", "B", "C"]

    for i, item in enumerate(results):
        species = item["species"]
        coords_df = item["coords_df"]
        pca = item["pca"]

        native_lab = "Native"
        nonnat_lab = non_native_label(species)

        # Left: scatter
        ax = axes[i, 0]
        native = coords_df[coords_df["range_label"] == native_lab]
        nonnat = coords_df[coords_df["range_label"] == nonnat_lab]

        ax.scatter(native["PC1"], native["PC2"], s=14, alpha=0.5, label=native_lab)
        ax.scatter(nonnat["PC1"], nonnat["PC2"], s=14, alpha=0.5, label=nonnat_lab)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title(f"{panel_labels[i]}. {short_name(species)}", loc="left", fontweight="bold")
        ax.legend(frameon=True, fontsize=9)

        # Right: overlap map
        ax2 = axes[i, 1]
        x_all = coords_df["PC1"].to_numpy()
        y_all = coords_df["PC2"].to_numpy()
        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()
        pad_x = 0.08 * (x_max - x_min)
        pad_y = 0.08 * (y_max - y_min)

        xx, yy = np.meshgrid(
            np.linspace(x_min - pad_x, x_max + pad_x, 200),
            np.linspace(y_min - pad_y, y_max + pad_y, 200),
        )

        z_native = compute_density_grid(
            native["PC1"].to_numpy(), native["PC2"].to_numpy(), xx, yy
        )
        z_nonnat = compute_density_grid(
            nonnat["PC1"].to_numpy(), nonnat["PC2"].to_numpy(), xx, yy
        )
        overlap = np.minimum(z_native, z_nonnat)

        ax2.contourf(xx, yy, overlap, levels=12)
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax2.set_title(f"Overlap density: Native vs. {nonnat_lab}", fontsize=10)

    fig.suptitle("PCA niche overlap of native and non-native populations", fontsize=15, y=0.995)

    out = OUT_DIR / "conservation_figure3_niche_overlap_panels.png"
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate conservation niche overlap figures"
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
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)
    env_vars = get_env_variables(df_filtered)["all_env"]

    results = []

    for species in STUDY_SPECIES:
        print(f"\nProcessing {species} ...")
        sp_df, clean_vars = prepare_species_matrix(df_filtered, species, env_vars)
        coords_df, pca = make_pca(sp_df, clean_vars)

        scatter_path = plot_species_scatter(species, coords_df, pca)
        density_path = plot_species_density(species, coords_df, pca)

        print(f"  Saved scatter: {scatter_path}")
        print(f"  Saved density: {density_path}")

        results.append({
            "species": species,
            "coords_df": coords_df,
            "pca": pca,
        })

    combined_path = plot_combined_figure3(results)
    print(f"\nSaved combined Figure 3: {combined_path}")


if __name__ == "__main__":
    main()