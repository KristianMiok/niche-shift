"""
cross_paper_comparison.py
Generate the 8-species comparison figure overlaying invasive species
(Paper 1 / GEB) with historically translocated species (Paper 2 / Conservation).

Creates a gradient plot from clearly invasive to ecologically integrated.

Usage:
    python src/cross_paper_comparison.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_results():
    """Load results from both papers."""

    # Paper 1 (invasive species - from main branch)
    with open("results/paper1_reference_dt.json") as f:
        p1_dt = json.load(f)
    with open("results/paper1_reference_rf.json") as f:
        p1_rf = json.load(f)
    with open("results/paper1_reference_niche.json") as f:
        p1_niche = json.load(f)

    # Paper 2 (translocated species - current branch)
    with open("results/tables/dt_summary.json") as f:
        p2_dt = json.load(f)
    with open("results/tables/rf_shap_summary.json") as f:
        p2_rf = json.load(f)
    with open("results/tables/niche_overlap_summary.json") as f:
        p2_niche = json.load(f)

    # Niche overlap is a list in paper 2
    if isinstance(p2_niche, list):
        p2_niche = {item["species"]: item for item in p2_niche}
    if isinstance(p1_niche, list):
        p1_niche = {item["species"]: item for item in p1_niche}

    return p1_dt, p1_rf, p1_niche, p2_dt, p2_rf, p2_niche


def build_species_table(p1_dt, p1_rf, p1_niche, p2_dt, p2_rf, p2_niche):
    """Build a unified table of all 8 species with key metrics."""

    species_order = [
        # Intercontinental invasive
        ("Procambarus clarkii", "Intercontinental invasive", "#D32F2F"),
        ("Faxonius limosus", "Intercontinental invasive", "#D32F2F"),
        ("Pacifastacus leniusculus", "Intercontinental invasive", "#D32F2F"),
        # Within-continent invasive
        ("Faxonius virilis", "Within-continent invasive", "#F57C00"),
        ("Faxonius rusticus", "Within-continent invasive", "#F57C00"),
        # Historically translocated
        ("Austropotamobius fulcisianus", "Historical translocation", "#1976D2"),
        ("Austropotamobius pallipes", "Multiple translocations", "#1976D2"),
        # Cryptogenic / natural expansion
        ("Pontastacus leptodactylus", "Cryptogenic / expansion", "#388E3C"),
    ]

    rows = []
    for sp, category, color in species_order:
        # Find in paper 1 or paper 2
        if sp in p1_dt:
            dt = p1_dt[sp]
            rf = p1_rf[sp]
            niche = p1_niche.get(sp, {})
        elif sp in p2_dt:
            dt = p2_dt[sp]
            rf = p2_rf[sp]
            niche = p2_niche.get(sp, {})
        else:
            continue

        rows.append({
            "species": sp,
            "short_name": sp.split(" ")[0][0] + ". " + sp.split(" ")[1],
            "category": category,
            "color": color,
            "dt_accuracy": dt["cv_scores"]["accuracy_mean"],
            "dt_auc": dt["cv_scores"]["roc_auc_mean"],
            "rf_accuracy": rf["cv_scores"]["accuracy_mean"],
            "rf_auc": rf["cv_scores"]["roc_auc_mean"],
            "rf_climate": rf["gini_by_type"].get("Climate", 0),
            "rf_topography": rf["gini_by_type"].get("Topography", 0),
            "rf_soil": rf["gini_by_type"].get("Soil", 0),
            "schoeners_d": niche.get("schoeners_D", niche.get("schoeners_d", None)),
            "warrens_i": niche.get("warrens_I", niche.get("warrens_i", None)),
        })

    return rows


def plot_gradient_accuracy(rows, output_dir="results/figures"):
    """
    Main figure: accuracy gradient from invasive to integrated.
    X = species ordered by invasion intensity, Y = classification accuracy.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12),
                                         gridspec_kw={"height_ratios": [1, 1, 1]})

    x = np.arange(len(rows))
    names = [r["short_name"] for r in rows]
    colors = [r["color"] for r in rows]

    # --- Panel A: Classification accuracy ---
    dt_acc = [r["dt_accuracy"] for r in rows]
    rf_acc = [r["rf_accuracy"] for r in rows]
    width = 0.35

    bars1 = ax1.bar(x - width/2, dt_acc, width, color=colors, alpha=0.6,
                     edgecolor="black", linewidth=0.5, label="Decision Tree")
    bars2 = ax1.bar(x + width/2, rf_acc, width, color=colors, alpha=1.0,
                     edgecolor="black", linewidth=0.5, label="Random Forest")

    ax1.set_ylabel("Cross-validated Accuracy")
    ax1.set_title("A) Classification Accuracy: Native vs. Non-native", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, style="italic", rotation=15, ha="right")
    ax1.set_ylim(0.75, 1.02)
    ax1.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9)

    # Add category separators
    ax1.axvline(x=2.5, color="gray", linestyle="--", alpha=0.4)
    ax1.axvline(x=4.5, color="gray", linestyle="--", alpha=0.4)
    ax1.axvline(x=6.5, color="gray", linestyle="--", alpha=0.4)

    # Category labels
    ax1.text(1, 1.015, "Intercontinental\ninvasive", ha="center", fontsize=7.5, color="gray")
    ax1.text(3.5, 1.015, "Within-cont.\ninvasive", ha="center", fontsize=7.5, color="gray")
    ax1.text(5.5, 1.015, "Historical\ntranslocation", ha="center", fontsize=7.5, color="gray")
    ax1.text(7, 1.015, "Cryptogenic", ha="center", fontsize=7.5, color="gray")

    # --- Panel B: Climate vs Topography importance ---
    climate = [r["rf_climate"] for r in rows]
    topo = [r["rf_topography"] for r in rows]
    soil = [r["rf_soil"] for r in rows]

    ax2.bar(x, climate, 0.6, label="Climate", color="#E53935", alpha=0.8)
    ax2.bar(x, topo, 0.6, bottom=climate, label="Topography", color="#1E88E5", alpha=0.8)
    ax2.bar(x, soil, 0.6, bottom=[c+t for c,t in zip(climate, topo)],
            label="Soil", color="#8D6E63", alpha=0.8)

    ax2.set_ylabel("RF Gini Importance (%)")
    ax2.set_title("B) Environmental Driver Composition", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9, style="italic", rotation=15, ha="right")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.axvline(x=2.5, color="gray", linestyle="--", alpha=0.4)
    ax2.axvline(x=4.5, color="gray", linestyle="--", alpha=0.4)
    ax2.axvline(x=6.5, color="gray", linestyle="--", alpha=0.4)

    # --- Panel C: Niche overlap ---
    d_vals = [r["schoeners_d"] if r["schoeners_d"] is not None else 0 for r in rows]
    i_vals = [r["warrens_i"] if r["warrens_i"] is not None else 0 for r in rows]

    ax3.bar(x - width/2, d_vals, width, color=colors, alpha=0.6,
            edgecolor="black", linewidth=0.5, label="Schoener's D")
    ax3.bar(x + width/2, i_vals, width, color=colors, alpha=1.0,
            edgecolor="black", linewidth=0.5, label="Warren's I")

    ax3.set_ylabel("Niche Overlap")
    ax3.set_title("C) Classical Niche Overlap Metrics", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=9, style="italic", rotation=15, ha="right")
    ax3.set_ylim(0, 1.0)
    ax3.legend(fontsize=9)
    ax3.axvline(x=2.5, color="gray", linestyle="--", alpha=0.4)
    ax3.axvline(x=4.5, color="gray", linestyle="--", alpha=0.4)
    ax3.axvline(x=6.5, color="gray", linestyle="--", alpha=0.4)

    # Legend for categories
    legend_patches = [
        mpatches.Patch(color="#D32F2F", label="Intercontinental invasive"),
        mpatches.Patch(color="#F57C00", label="Within-continent invasive"),
        mpatches.Patch(color="#1976D2", label="Historical translocation"),
        mpatches.Patch(color="#388E3C", label="Cryptogenic / expansion"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig_path = out / "cross_paper_8species_gradient.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")
    return str(fig_path)


def plot_accuracy_vs_overlap(rows, output_dir="results/figures"):
    """
    Scatter plot: classification accuracy vs niche overlap (Schoener's D).
    Each point is a species, colored by category.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in rows:
        if r["schoeners_d"] is None:
            continue
        ax.scatter(r["schoeners_d"], r["rf_accuracy"],
                   c=r["color"], s=120, edgecolors="black", linewidth=0.8, zorder=5)
        ax.annotate(r["short_name"],
                    (r["schoeners_d"], r["rf_accuracy"]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, style="italic")

    ax.set_xlabel("Schoener's D (niche overlap)", fontsize=11)
    ax.set_ylabel("RF Classification Accuracy", fontsize=11)
    ax.set_title("Niche Overlap vs. Environmental Separability\n"
                 "across invasive and translocated crayfish", fontsize=12)

    legend_patches = [
        mpatches.Patch(color="#D32F2F", label="Intercontinental invasive"),
        mpatches.Patch(color="#F57C00", label="Within-continent invasive"),
        mpatches.Patch(color="#1976D2", label="Historical translocation"),
        mpatches.Patch(color="#388E3C", label="Cryptogenic / expansion"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower left")

    plt.tight_layout()
    fig_path = out / "cross_paper_accuracy_vs_overlap.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")
    return str(fig_path)


def main():
    p1_dt, p1_rf, p1_niche, p2_dt, p2_rf, p2_niche = load_results()
    rows = build_species_table(p1_dt, p1_rf, p1_niche, p2_dt, p2_rf, p2_niche)

    print("8-Species Comparison Table:")
    print(f"{'Species':<30s} {'Category':<28s} {'DT Acc':>7s} {'RF Acc':>7s} "
          f"{'Climate%':>8s} {'Topo%':>7s} {'D':>6s}")
    print("-" * 95)
    for r in rows:
        d = f"{r['schoeners_d']:.3f}" if r["schoeners_d"] else "N/A"
        print(f"{r['short_name']:<30s} {r['category']:<28s} "
              f"{r['dt_accuracy']:>7.3f} {r['rf_accuracy']:>7.3f} "
              f"{r['rf_climate']:>7.1f}% {r['rf_topography']:>6.1f}% {d:>6s}")

    plot_gradient_accuracy(rows)
    plot_accuracy_vs_overlap(rows)


if __name__ == "__main__":
    main()