"""
cross_species_comparison.py
Cross-species comparison of decision tree results for the conservation paper.

Produces summary figures showing which environmental variable types
(climate, topography, soil, land cover) dominate for each species,
using conservation-paper wording and species-specific non-native labels.

Usage:
    python src/cross_species_comparison.py --input data/raw/combined_data_true_master.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

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
from decision_tree import train_final_tree, extract_feature_importances, prepare_xy
from variable_glossary import load_glossary, translate


def classify_variable(var_name: str) -> str:
    if "CLI" in var_name:
        return "Climate"
    elif "TOP" in var_name:
        return "Topography"
    elif "SOL" in var_name:
        return "Soil"
    elif "LAC" in var_name:
        return "Land Cover"
    return "Other"


def classify_scale(var_name: str) -> str:
    if var_name.startswith("l_"):
        return "Local"
    elif var_name.startswith("u_"):
        return "Upstream"
    return "Other"


def get_range_label(species: str) -> str:
    """Species-specific non-native label requested by Lucian."""
    mapping = {
        "Pontastacus leptodactylus": "Cryptogenic",
        "Austropotamobius fulcisianus": "Introduced",
        "Austropotamobius pallipes": "Introduced",
    }
    return mapping.get(species, "Non-native")


def compute_importance_by_type(importances: pd.DataFrame) -> pd.DataFrame:
    importances = importances.copy()
    importances["type"] = importances["variable"].apply(classify_variable)
    importances["scale"] = importances["variable"].apply(classify_scale)

    by_type = (
        importances.groupby("type")["importance"]
        .sum()
        .reset_index()
        .sort_values("importance", ascending=False)
    )
    return by_type


def run_all_species(df_filtered, env_vars, max_depth=5):
    all_importances = {}

    for species in STUDY_SPECIES:
        sp_data = df_filtered[df_filtered[COL_SPECIES] == species].copy()
        native = sp_data[sp_data[COL_STATUS] == STATUS_NATIVE]
        non_native = sp_data[sp_data[COL_STATUS] == STATUS_ALIEN]
        combined = pd.concat([native, non_native], ignore_index=True)

        combined, clean_vars = handle_missing_values(combined, env_vars)
        clean_vars = remove_constant_variables(combined, clean_vars)
        clean_vars = remove_highly_correlated(combined, clean_vars, threshold=0.98)

        combined["range_label"] = combined[COL_STATUS].map(
            {STATUS_NATIVE: "native", STATUS_ALIEN: "invasive"}
        )

        X, y, feature_names = prepare_xy(combined, clean_vars)
        clf = train_final_tree(X, y, feature_names, max_depth=max_depth)
        importances = extract_feature_importances(clf, feature_names)
        all_importances[species] = importances

        print(
            f"  {species}: {len(feature_names)} features, "
            f"depth={clf.get_depth()}, leaves={clf.get_n_leaves()}"
        )

    return all_importances


def plot_importance_by_type(
    all_importances: dict,
    output_dir: str = "results/figures",
) -> str:
    """
    Main conservation-paper Figure 2.
    Stacked bar chart: proportion of total importance by variable type for each species.
    Includes species-specific range-pair labels.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    types = ["Climate", "Topography", "Soil", "Land Cover"]
    colors = {
        "Climate": "#E53935",
        "Topography": "#1E88E5",
        "Soil": "#8D6E63",
        "Land Cover": "#43A047",
    }

    species_labels = []
    range_pair_labels = []
    type_data = {t: [] for t in types}

    for species, importances in all_importances.items():
        by_type = compute_importance_by_type(importances)
        type_dict = dict(zip(by_type["type"], by_type["importance"]))
        total = sum(type_dict.values())

        short_name = species.split(" ")[0][0] + ". " + species.split(" ")[1]
        species_labels.append(short_name)
        range_pair_labels.append(f"Native vs. {get_range_label(species)}")

        for t in types:
            type_data[t].append(type_dict.get(t, 0) / total * 100)

    x = np.arange(len(species_labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6.8))
    bottom = np.zeros(len(species_labels))

    for t in types:
        vals = np.array(type_data[t])
        ax.bar(x, vals, width, bottom=bottom, label=t, color=colors[t])
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(
                    x[i], bottom[i] + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white"
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(species_labels, fontsize=10, style="italic")
    ax.set_ylabel("Relative feature importance (%)")
    ax.set_title(
        "Environmental drivers distinguishing native and non-native populations",
        fontsize=12
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, 114)

    for i, lbl in enumerate(range_pair_labels):
        ax.text(
            x[i], 102.5, lbl,
            ha="center", va="bottom",
            fontsize=8, color="dimgray"
        )

    plt.tight_layout()
    fig_path = out / "cross_species_importance_by_type.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def plot_top_features_heatmap(
    all_importances: dict,
    glossary: dict = None,
    n_top: int = 15,
    output_dir: str = "results/figures",
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_top_vars = set()
    for species, imp in all_importances.items():
        all_top_vars.update(imp.head(n_top)["variable"].tolist())

    all_top_vars = sorted(all_top_vars)
    species_list = list(all_importances.keys())
    matrix = pd.DataFrame(0.0, index=all_top_vars, columns=species_list)

    for species, imp in all_importances.items():
        imp_dict = dict(zip(imp["variable"], imp["importance"]))
        for var in all_top_vars:
            matrix.loc[var, species] = imp_dict.get(var, 0.0)

    matrix["max_imp"] = matrix[species_list].max(axis=1)
    matrix = matrix.sort_values("max_imp", ascending=True).drop(columns=["max_imp"])

    if glossary:
        row_labels = [f"{v}: {translate(v, glossary)[:45]}" for v in matrix.index]
    else:
        row_labels = matrix.index.tolist()

    short_cols = [s.split(" ")[0][0] + ". " + s.split(" ")[1] for s in species_list]

    type_colors = {
        "CLI": "#E53935",
        "TOP": "#1E88E5",
        "SOL": "#8D6E63",
        "LAC": "#43A047",
    }

    fig, ax = plt.subplots(figsize=(10, max(8, len(all_top_vars) * 0.35)))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(short_cols)))
    ax.set_xticklabels(short_cols, fontsize=10, style="italic")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    for i, var in enumerate(matrix.index):
        for prefix, color in type_colors.items():
            if prefix in var:
                ax.get_yticklabels()[i].set_color(color)
                break

    for i in range(len(matrix.index)):
        for j in range(len(species_list)):
            val = matrix.values[i, j]
            if val > 0.01:
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7, color="white" if val > 0.3 else "black"
                )

    plt.colorbar(im, ax=ax, label="Feature importance", shrink=0.8)
    ax.set_title("Feature importance across species")

    patches = [mpatches.Patch(color=c, label=l)
               for l, c in [("Climate", "#E53935"), ("Topography", "#1E88E5"),
                             ("Soil", "#8D6E63"), ("Land Cover", "#43A047")]]

    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        fontsize=8,
        title="Variable Type",
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    fig_path = out / "cross_species_feature_heatmap.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def plot_importance_by_scale(
    all_importances: dict,
    output_dir: str = "results/figures",
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    species_labels = []
    local_vals = []
    upstream_vals = []

    for species, imp in all_importances.items():
        imp = imp.copy()
        imp["scale"] = imp["variable"].apply(classify_scale)
        by_scale = imp.groupby("scale")["importance"].sum()

        short_name = species.split(" ")[0][0] + ". " + species.split(" ")[1]
        species_labels.append(short_name)
        total = by_scale.sum()
        local_vals.append(by_scale.get("Local", 0) / total * 100)
        upstream_vals.append(by_scale.get("Upstream", 0) / total * 100)

    x = np.arange(len(species_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2, local_vals, width,
        label="Local (segment scale)",
        color="#1565C0"
    )
    ax.bar(
        x + width / 2, upstream_vals, width,
        label="Upstream (catchment scale)",
        color="#42A5F5"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(species_labels, fontsize=10, style="italic")
    ax.set_ylabel("Relative feature importance (%)")
    ax.set_title("Local vs upstream scale importance")
    ax.legend()
    plt.tight_layout()

    fig_path = out / "cross_species_local_vs_upstream.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-species comparison of native vs non-native drivers"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to combined_data_true_master.csv"
    )
    parser.add_argument(
        "--config", "-c", default="config/species_config.yaml"
    )
    parser.add_argument(
        "--max-depth", type=int, default=5
    )
    parser.add_argument(
        "--output", "-o", default="results/figures"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_geotraits(args.input)
    df_filtered = apply_quality_filters(df, config)

    env_info = get_env_variables(df_filtered)
    env_vars = env_info["all_env"]
    glossary = load_glossary("data/raw/S2.xlsx")

    print("\nRunning decision trees for all species...")
    all_importances = run_all_species(df_filtered, env_vars, max_depth=args.max_depth)

    fig1 = plot_importance_by_type(all_importances, output_dir=args.output)
    print(f"\nSaved: {fig1}")

    fig2 = plot_top_features_heatmap(
        all_importances,
        glossary=glossary,
        output_dir=args.output
    )
    print(f"Saved: {fig2}")

    fig3 = plot_importance_by_scale(all_importances, output_dir=args.output)
    print(f"Saved: {fig3}")

    print(f"\n{'='*72}")
    print("IMPORTANCE BY VARIABLE TYPE (%)")
    print(f"{'='*72}")
    print(f"{'Species':<28s} {'Climate':>8s} {'Topogr.':>8s} {'Soil':>8s} {'LandCov':>8s}")
    print("-" * 72)
    for species, imp in all_importances.items():
        by_type = compute_importance_by_type(imp)
        type_dict = dict(zip(by_type["type"], by_type["importance"]))
        total = sum(type_dict.values())
        short = species.split(" ")[0][0] + ". " + species.split(" ")[1]
        print(
            f"{short:<28s} "
            f"{type_dict.get('Climate', 0)/total*100:>7.1f}% "
            f"{type_dict.get('Topography', 0)/total*100:>7.1f}% "
            f"{type_dict.get('Soil', 0)/total*100:>7.1f}% "
            f"{type_dict.get('Land Cover', 0)/total*100:>7.1f}%"
        )


if __name__ == "__main__":
    main()