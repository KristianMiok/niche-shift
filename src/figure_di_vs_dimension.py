"""
figure_di_vs_dimension.py — Task (vii), CORRECTED: D and I vs number of retained PCs.

Fix (Lucian item 2): the 2-PC point of each curve MUST reproduce the Table S5 overlap
value, which is computed with the GRID density estimator (niche_overlap_metrics). The
previous version used k-NN density throughout, so the 2-PC point disagreed with Table S5
(e.g. leniusculus grid 0.36 vs k-NN 0.64). k-NN over-estimates overlap under severe class
imbalance (leniusculus 117 native vs 4459 invaded). We now anchor the 2-PC point to the
grid value (= Table S5) and use k-NN only for k>2, where the 2-D grid does not generalise.
The two estimators are labelled; the point of the figure is the monotone collapse, and the
left-hand (2-PC) point matches Table S5 by construction.

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
from dimensionality_control import prep, pcs, di_knn
from niche_overlap_metrics import estimate_density_grid, schoeners_d, warrens_i

TABLES = Path("results/tables"); FIGDIR = Path("results/figures"); FIGDIR.mkdir(parents=True, exist_ok=True)
PC_GRID = [2, 3, 5, 8, 13, 21, 34]
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
INTER_COLORS = ["#B71C1C", "#E53935", "#EF9A9A"]
WITHIN_COLORS = ["#1565C0", "#42A5F5"]


def grid_di_2pc(P2, status):
    nat, inv = P2[status == 0], P2[status == 1]
    c = np.vstack([nat, inv])
    xe = np.linspace(c[:, 0].min(), c[:, 0].max(), 101)
    ye = np.linspace(c[:, 1].min(), c[:, 1].max(), 101)
    ng = estimate_density_grid(nat[:, 0], nat[:, 1], xe, ye, sigma=1.0)
    ig = estimate_density_grid(inv[:, 0], inv[:, 1], xe, ye, sigma=1.0)
    return schoeners_d(ng, ig), warrens_i(ng, ig)


def main():
    cfg = load_config("config/species_config.yaml")
    dff = apply_quality_filters(load_geotraits("data/raw/combined_data_true_master.csv"), cfg)
    env = get_env_variables(dff)["all_env"]

    s5raw = json.loads((TABLES / "niche_overlap_summary.json").read_text())
    s5list = s5raw if isinstance(s5raw, list) else s5raw.get("data", s5raw)
    S5 = {r["species"]: {"D": r["schoeners_D"], "I": r["warrens_I"]} for r in s5list}

    curves = {}
    print("D/I vs PCs (2-PC point = grid = Table S5; higher PC = k-NN):")
    for sp in SPECIES:
        _, Xs, status = prep(dff, sp, env)
        gD, gI = S5[sp]["D"], S5[sp]["I"]        # ANCHOR: canonical Table S5 (niche_overlap_summary)
        ks, Ds, Is = [2], [gD], [gI]
        for k in PC_GRID[1:]:
            if k <= Xs.shape[1]:
                dK, iK = di_knn(pcs(Xs, k), status)
                ks.append(k); Ds.append(dK); Is.append(iK)
        curves[sp] = {"k": ks, "D": Ds, "I": Is, "D_2pc_grid": gD, "I_2pc_grid": gI}
        print(f"  {sp:<26} 2PC grid D={gD:.3f} (Table S5)  then k-NN: " + " ".join(f"{d:.3f}" for d in Ds[1:]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    color = {**{sp: INTER_COLORS[i] for i, sp in enumerate(INTERCONTINENTAL)},
             **{sp: WITHIN_COLORS[i] for i, sp in enumerate(WITHIN_CONTINENT)}}
    for metric, ax, title in [("D", axes[0], "Schoener's D"), ("I", axes[1], "Warren's I")]:
        for sp in SPECIES:
            c = curves[sp]
            ls = "-" if sp in INTERCONTINENTAL else "--"
            lbl = sp.split(" ")[0][0] + ". " + sp.split(" ")[1]
            ax.plot(c["k"], c[metric], ls, marker="o", color=color[sp], label=lbl, lw=1.8, ms=4)
            ax.scatter([2], [c[metric][0]], s=90, facecolors="none", edgecolors="black",
                       linewidths=1.3, zorder=5)  # ring the Table S5 anchor
        ax.set_title(title); ax.set_xlabel("Number of retained PCA axes")
        ax.set_ylabel(f"{title} (native vs invaded)"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="solid=intercontinental  dashed=within-continent\n"
                   "ringed point at 2 PC = Table S5 (grid)", title_fontsize=6.5, loc="upper right")
    fig.suptitle("Niche overlap collapses toward zero as dimensionality increases\n"
                 "(2-PC point = Table S5 grid value; higher dimensions via k-NN)", fontsize=11)
    plt.tight_layout()
    out = FIGDIR / "figure_di_vs_dimension.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    (TABLES / "di_vs_dimension_curve.json").write_text(json.dumps(curves, indent=2, default=float))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
