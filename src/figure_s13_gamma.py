"""
figure_s13_gamma.py — Figure S13, option gamma per the senior author.

One estimator across the whole curve (k-NN, since the grid density is 2-D only), with the
Table S5 grid values drawn as SEPARATE marked anchor points at 2 PC and labelled as such.
The curve is therefore internally consistent (no estimator seam), while the canonical
Table S5 values remain visible for reference.

Framing: this is a demonstration that Schoener's D and Warren's I have no stable
high-dimensional form — overlap collapses toward zero as dimensions are added because the
native and invaded distributions become near-separable — NOT an ecological result about
niche overlap.

Output: results/figures/figureS13_di_knn_gamma.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from dimensionality_control import prep, pcs, di_knn

TABLES = Path("results/tables"); FIGDIR = Path("results/figures"); FIGDIR.mkdir(parents=True, exist_ok=True)
PC_GRID = [2, 3, 5, 8, 13, 21, 34]
INTER = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN = ["Faxonius virilis", "Faxonius rusticus"]
SP = INTER + WITHIN
INTER_C = ["#B71C1C", "#E53935", "#EF9A9A"]
WITHIN_C = ["#1565C0", "#42A5F5"]


def main():
    cfg = load_config("config/species_config.yaml")
    dff = apply_quality_filters(load_geotraits("data/raw/combined_data_true_master.csv"), cfg)
    env = get_env_variables(dff)["all_env"]

    s5raw = json.loads((TABLES / "niche_overlap_summary.json").read_text())
    s5list = s5raw if isinstance(s5raw, list) else s5raw.get("data", s5raw)
    S5 = {r["species"]: {"D": r["schoeners_D"], "I": r["warrens_I"]} for r in s5list}

    curves = {}
    print("Figure S13 (gamma): single k-NN estimator; Table S5 grid values as anchors")
    for sp in SP:
        _, Xs, status = prep(dff, sp, env)
        ks, Ds, Is = [], [], []
        for k in PC_GRID:
            if k <= Xs.shape[1]:
                d, i = di_knn(pcs(Xs, k), status)   # SAME estimator at every k, incl. 2
                ks.append(k); Ds.append(d); Is.append(i)
        curves[sp] = {"k": ks, "D": Ds, "I": Is,
                      "S5_D": S5[sp]["D"], "S5_I": S5[sp]["I"]}
        print(f"  {sp:<26} kNN D: " + " ".join(f"{d:.3f}" for d in Ds)
              + f"   | S5 grid D={S5[sp]['D']:.3f}")

    color = {**{s: INTER_C[i] for i, s in enumerate(INTER)},
             **{s: WITHIN_C[i] for i, s in enumerate(WITHIN)}}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)

    for metric, s5key, ax, panel, title in [
        ("D", "S5_D", axes[0], "(A)", "Schoener's $D$"),
        ("I", "S5_I", axes[1], "(B)", "Warren's $I$"),
    ]:
        for sp in SP:
            c = curves[sp]
            ls = "-" if sp in INTER else "--"
            lbl = sp.split()[0][0] + ". " + sp.split()[1]
            ax.plot(c["k"], c[metric], ls, marker="o", color=color[sp], lw=1.8, ms=4, label=lbl)
            # Table S5 grid anchor at 2 PC — separate marker, clearly not on the curve
            ax.scatter([2], [c[s5key]], marker="*", s=170, color=color[sp],
                       edgecolor="black", linewidth=0.7, zorder=6)
        ax.scatter([], [], marker="*", s=140, color="gray", edgecolor="black",
                   label="Table S5 (grid, 2 PC)")
        ax.set_title(f"{panel} {title}", loc="left", fontsize=11)
        ax.set_xlabel("Number of retained PCA axes")
        ax.set_ylabel(f"{title} (native vs invaded)")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="upper right",
                   title="solid = intercontinental\ndashed = within-continent", title_fontsize=6.5)

    fig.suptitle("Figure S13. Schoener's $D$ and Warren's $I$ have no stable high-dimensional form\n"
                 "Overlap collapses toward zero as axes are added, because native and invaded "
                 "distributions become near-separable;\nthis is a property of the statistic, not an "
                 "ecological result. Curves: k-NN density (one estimator throughout).",
                 fontsize=9.5, y=1.02)
    plt.tight_layout()
    out = FIGDIR / "figureS13_di_knn_gamma.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    (TABLES / "figureS13_curve_data.json").write_text(json.dumps(curves, indent=2, default=float))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
