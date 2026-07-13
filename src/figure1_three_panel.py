"""
figure1_three_panel.py — Task (ix)/Decision 1: rebuild Figure 1 with a grouped-permutation
panel so the retired "64% topography" is not left drawn in the main figure.

Three stacked panels, five species (intercontinental | within-continent), four domains:
  (A) Decision-tree Gini importance  — the single-tree view (overstates topography)
  (B) Random-forest Gini importance  — diffuses across correlated domains
  (C) Grouped-permutation dAUC        — collinearity-immune, carries the domain claim

Panels A/B are percentages (share of total importance); panel C is dAUC (drop in AUC when
a domain is block-permuted), a different scale, so it has its own axis and is annotated as
such. Together they show all three methods agree on the dichotomy while making explicit
that the DT magnitude is not the one to quote.

Reads DT/RF from the post-dedup fits (recomputed here) and grouped permutation from
grouped_permutation_summary.json. All post-dedup, mutually consistent with Table S2/S11.

Run:  python src/figure1_three_panel.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame
from decision_tree import train_final_tree, extract_feature_importances
from random_forest_shap import get_gini_importances, compute_importance_by_type
from cross_species_comparison import classify_variable
from sklearn.ensemble import RandomForestClassifier

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, TREES = 42, 500
FIGDIR = Path("results/figures"); FIGDIR.mkdir(parents=True, exist_ok=True)
TABLES = Path("results/tables")
DOMAINS = ["Climate", "Topography", "Soil", "Land Cover"]
COLORS = {"Climate": "#E53935", "Topography": "#1E88E5", "Soil": "#8D6E63", "Land Cover": "#43A047"}
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT


def dt_domain_pct(X, y, clean):
    clf = train_final_tree(X, y, clean, max_depth=5, random_state=SEED)
    imp = extract_feature_importances(clf, clean)
    imp["type"] = imp["variable"].apply(classify_variable)
    by = imp.groupby("type")["importance"].sum()
    tot = sum(by.get(d, 0) for d in DOMAINS)
    return {d: 100 * by.get(d, 0) / tot for d in DOMAINS}


def rf_domain_pct(X, y, clean):
    clf = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1).fit(X, y)
    bt = compute_importance_by_type(get_gini_importances(clf, clean), "gini_importance")
    tot = sum(bt.get(d, 0) for d in DOMAINS)
    return {d: 100 * bt.get(d, 0) / tot for d in DOMAINS}


def stacked_panel(ax, data, species, ylabel, as_pct, grouped=False):
    """Stacked bars for compositional panels (Gini %, sums to 100).
    GROUPED bars when grouped=True: permuting each domain independently does NOT
    partition AUC, so a stacked bar would imply a decomposition that does not exist."""
    labels = [s.split(" ")[0][0] + ". " + s.split(" ")[1] for s in species]
    x = np.arange(len(species))
    if grouped:
        w = 0.2
        for k, d in enumerate(DOMAINS):
            vals = np.array([data[s][d] for s in species])
            pos = x + (k - 1.5) * w
            ax.bar(pos, vals, w, label=d, color=COLORS[d])
            for i, v in enumerate(vals):
                if v > 0.005:
                    ax.text(pos[i], v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=6, rotation=90)
        ax.set_xticks(x)
    else:
        bottom = np.zeros(len(species))
        for d in DOMAINS:
            vals = np.array([data[s][d] for s in species])
            ax.bar(x, vals, 0.62, bottom=bottom, label=d, color=COLORS[d])
            for i, v in enumerate(vals):
                if v > 6:
                    ax.text(x[i], bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bottom += vals
        ax.set_xticks(x)
    ax.set_xticklabels(labels, style="italic", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axvline(2.5, color="gray", ls="--", alpha=0.5)


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    gp = json.loads((TABLES / "grouped_permutation_summary.json").read_text())
    s2raw = json.loads((TABLES / "tables_s2_s5_summary.json").read_text())
    S2 = {r["species"]: {d: r[d + "_mean"] for d in DOMAINS} for r in s2raw["table_s2"]}

    dt_d, rf_d, gp_d = {}, {}, {}
    print("Building three-panel Figure 1 (post-dedup):")
    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        X = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values
        dt_d[sp] = dt_domain_pct(X, y, clean)
        rf_d[sp] = S2[sp]                      # CANONICAL Table S2 (single pipeline run)
        gp_d[sp] = {d: gp[sp]["RF"][d]["drop_auc"] for d in DOMAINS}
        print(f"  {sp:<26} DT topo={dt_d[sp]['Topography']:.0f}%  RF topo={rf_d[sp]['Topography']:.0f}%  "
              f"GP topo dAUC={gp_d[sp]['Topography']:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    stacked_panel(axes[0], dt_d, SPECIES, "Importance (%)", True)
    stacked_panel(axes[1], rf_d, SPECIES, "Importance (%)", True)
    stacked_panel(axes[2], gp_d, SPECIES, "Drop in AUC when\ndomain permuted", False, grouped=True)

    axes[0].set_title("(A) Decision tree (Gini) — single-tree splits overstate topography",
                      fontsize=10, loc="left")
    axes[1].set_title("(B) Random forest (Gini) — diffuses across correlated domains",
                      fontsize=10, loc="left")
    axes[2].set_title("(C) Grouped permutation (ΔAUC) — collinearity-immune; domains permuted independently (not a decomposition)",
                      fontsize=10, loc="left")
    for ax in axes[:2]:
        ax.set_ylim(0, 105)
        ax.text(1, 101, "Intercontinental", ha="center", fontsize=8, color="gray")
        ax.text(3.5, 101, "Within-continent", ha="center", fontsize=8, color="gray")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="Domain")
    fig.suptitle("Figure 1. Environmental drivers of native-vs-invaded differentiation by species\n"
                 "Climate dominates in all five species; the pathway groups differ in whether "
                 "topography contributes (panel C)", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 0.86, 0.97])
    out = FIGDIR / "figure1_three_panel.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
