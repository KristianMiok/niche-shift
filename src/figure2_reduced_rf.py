"""
figure2_reduced.py — Task (h): reduced Figure 2 + full Figure S12 (Reviewer 1.2).

Reviewer 1 asked for a readable main-text heatmap: drop rows that are ~0 in EVERY
species, keep the four-domain colour coding, enlarge the font. The full heatmap moves
to the supplement as Figure S12.

Row-keeping rule (per Lucian): keep a variable if EITHER
  (i) its DT Gini importance exceeds THRESHOLD in at least one species, OR
  (ii) it is a top-5 driver for at least one species (protects a variable that is
       strong in one species even if below threshold elsewhere).
Everything else (near-zero in all five species) is dropped to the supplement.

Reports THRESHOLD, N_OLD, N_NEW for the rebuttal placeholders.

Run:  python src/figure2_reduced.py [--threshold 0.01]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from cross_species_comparison import run_all_species, classify_variable
from sklearn.ensemble import RandomForestClassifier
from data_loader import COL_STATUS
from grouped_permutation import build_species_frame
from variable_glossary import load_glossary, translate

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
FIGDIR = Path("results/figures"); FIGDIR.mkdir(parents=True, exist_ok=True)
DOMAIN_COLORS = {"CLI": "#E53935", "TOP": "#1E88E5", "SOL": "#8D6E63", "LAC": "#43A047"}
DOMAIN_OF = {"CLI": "Climate", "TOP": "Topography", "SOL": "Soil", "LAC": "Land Cover"}


def domain_prefix(v):
    for p in DOMAIN_COLORS:
        if p in v:
            return p
    return None


def build_matrix(all_importances):
    """Full variable x species importance matrix (union of each species' variables)."""
    species = list(all_importances.keys())
    allvars = sorted(set().union(*[set(imp["variable"]) for imp in all_importances.values()]))
    M = pd.DataFrame(0.0, index=allvars, columns=species)
    for sp, imp in all_importances.items():
        d = dict(zip(imp["variable"], imp["importance"]))
        for v in allvars:
            M.loc[v, sp] = d.get(v, 0.0)
    return M, species


def keep_rows(M, all_importances, threshold, top_n=5):
    """Keep var if max importance across species > threshold OR it is top-5 in any species."""
    above = M.max(axis=1) > threshold
    protected = set()
    for sp, imp in all_importances.items():
        protected.update(imp.sort_values("importance", ascending=False).head(top_n)["variable"])
    keep = above | M.index.to_series().isin(protected)
    return keep


def draw_heatmap(M, species, glossary, title, path, fontsize):
    order = sorted(M.index, key=lambda v: -M.loc[v].max())
    M = M.loc[order]
    labels = [f"{v}: {translate(v, glossary)[:42]}" if glossary else v for v in M.index]
    short = [s.split(" ")[0][0] + ". " + s.split(" ")[1] for s in species]
    fig, ax = plt.subplots(figsize=(8, max(4, len(M) * (0.34 if len(M) <= 30 else 0.18))))
    im = ax.imshow(M.values, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, style="italic", fontsize=fontsize + 1,
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(M))); ax.set_yticklabels(labels, fontsize=fontsize)
    for i, v in enumerate(M.index):
        p = domain_prefix(v)
        if p:
            ax.get_yticklabels()[i].set_color(DOMAIN_COLORS[p])
    if len(M) <= 30:
        for i in range(len(M)):
            for j in range(len(species)):
                val = M.values[i, j]
                if val > 0.01:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=fontsize - 1, color="white" if val > 0.3 else "black")
    plt.colorbar(im, ax=ax, label="Gini importance", shrink=0.45, anchor=(0.0, 0.0), pad=0.02)
    ax.set_title(title, fontsize=fontsize + 3)
    patches = [mpatches.Patch(color=c, label=DOMAIN_OF[p]) for p, c in DOMAIN_COLORS.items()]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.15, 1.0),
              fontsize=fontsize, title="Domain", frameon=True, borderaxespad=0.0)
    plt.tight_layout()
    plt.subplots_adjust(right=0.72, bottom=0.12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.04)
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    glossary = load_glossary("data/raw/S2.xlsx")

    # RF Gini importances per species (stable per-variable importance; replaces single-tree DT)
    all_imp = {}
    for sp in ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus",
               "Faxonius virilis", "Faxonius rusticus"]:
        combined, clean = build_species_frame(dff, sp, env)
        X = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=42, n_jobs=-1).fit(X, y)
        all_imp[sp] = __import__("pandas").DataFrame(
            {"variable": clean, "importance": clf.feature_importances_})
    M, species = build_matrix(all_imp)
    n_old = len(M)

    keep = keep_rows(M, all_imp, args.threshold, args.top_n)
    M_red = M.loc[keep]
    n_new = len(M_red)

    print("\n" + "=" * 70)
    print("FIGURE 2 REDUCTION (Reviewer 1.2)")
    print("=" * 70)
    print(f"  THRESHOLD (max Gini across species) : {args.threshold}")
    print(f"  top-N protection                    : top {args.top_n} per species")
    print(f"  N_OLD (full heatmap rows)           : {n_old}")
    print(f"  N_NEW (reduced main-text rows)      : {n_new}")
    print(f"  dropped to supplement               : {n_old - n_new}")
    # domain breakdown of the reduced set
    doms = pd.Series([DOMAIN_OF.get(domain_prefix(v), "Other") for v in M_red.index]).value_counts()
    print("  reduced-set rows by domain          : "
          + ", ".join(f"{k}={v}" for k, v in doms.items()))

    draw_heatmap(M_red, species, glossary,
                 "Figure 2. Environmental drivers of native-vs-invaded differentiation\n"
                 f"(RF Gini importance; variables with max importance > {args.threshold} or top-{args.top_n} in any species)",
                 FIGDIR / "figure2_rf_gini_thr004.png", fontsize=8)
    draw_heatmap(M, species, glossary,
                 "Figure S12. Full feature-importance heatmap, RF Gini (all variables)",
                 FIGDIR / "figureS12_rf_gini_full.png", fontsize=5)

    print(f"\n  Saved: {FIGDIR/'figure2_reduced.png'}  (main text)")
    print(f"  Saved: {FIGDIR/'figureS12_full.png'}  (supplement)")
    print(f"\n>>> For rebuttal: THRESHOLD={args.threshold}, N_OLD={n_old}, N_NEW={n_new}")


if __name__ == "__main__":
    main()
