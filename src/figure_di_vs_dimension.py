"""
figure_di_vs_dimension.py — Task (vii): D and I as a function of the number of retained
PCs, per species (Reviewer 2 point 7, for Discussion 4.4).

Reads the D/I-vs-dimension trend already computed by dimensionality_control.py
(dimensionality_control_summary.json -> di_vs_dim_trend). If Warren's I is missing for
any PC count, it is recomputed with the same log-space k-NN estimator. Produces a
two-panel figure (Schoener D | Warren I) with one curve per species, coloured by
pathway, showing overlap collapsing toward zero as dimension rises.

Run:  python src/figure_di_vs_dimension.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame
from dimensionality_control import prep, pcs, di_knn, PC_GRID

TABLES = Path("results/tables"); FIGDIR = Path("results/figures"); FIGDIR.mkdir(parents=True, exist_ok=True)
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
INTER_COLORS = ["#B71C1C", "#E53935", "#EF9A9A"]   # reds = intercontinental
WITHIN_COLORS = ["#1565C0", "#42A5F5"]             # blues = within-continent


def main():
    # recompute D and I together (ensures both, same estimator/order as (f))
    cfg = load_config("config/species_config.yaml")
    dff = apply_quality_filters(load_geotraits("data/raw/combined_data_true_master.csv"), cfg)
    env = get_env_variables(dff)["all_env"]

    curves = {}
    print("\nD/I vs number of PCs (per species):")
    for sp in SPECIES:
        _, Xs, status = prep(dff, sp, env)
        ks, Ds, Is = [], [], []
        for k in PC_GRID:
            if k <= Xs.shape[1]:
                D, I = di_knn(pcs(Xs, k), status)
                ks.append(k); Ds.append(D); Is.append(I)
        curves[sp] = {"k": ks, "D": Ds, "I": Is}
        print(f"  {sp:<26} D: " + " ".join(f"{d:.3f}" for d in Ds))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    color = {}
    for i, sp in enumerate(INTERCONTINENTAL):
        color[sp] = INTER_COLORS[i]
    for i, sp in enumerate(WITHIN_CONTINENT):
        color[sp] = WITHIN_COLORS[i]

    for metric, ax, title in [("D", axes[0], "Schoener's D"), ("I", axes[1], "Warren's I")]:
        for sp in SPECIES:
            c = curves[sp]
            style = "-" if sp in INTERCONTINENTAL else "--"
            lbl = sp.split(" ")[0][0] + ". " + sp.split(" ")[1]
            ax.plot(c["k"], c[metric], style, marker="o", color=color[sp], label=lbl, linewidth=1.8, markersize=4)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Number of retained PCA axes")
        ax.set_ylabel(f"{title} (native vs invaded)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="solid = intercontinental\ndashed = within-continent",
                   title_fontsize=7, loc="upper right")
    fig.suptitle("Niche overlap collapses toward zero as dimensionality increases\n"
                 "(native and invaded distributions become near-separable in high dimension)",
                 fontsize=12)
    plt.tight_layout()
    out = FIGDIR / "figure_di_vs_dimension.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    (TABLES / "di_vs_dimension_curve.json").write_text(json.dumps(curves, indent=2, default=float))
    print(f"\nSaved: {out}")
    print(f"Saved: {TABLES/'di_vs_dimension_curve.json'}")


if __name__ == "__main__":
    main()
