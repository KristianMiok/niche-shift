"""
figure_s6_redraw.py — Task (viii): redraw Figure S6 with an independent x-axis per row.

Reviewer/Lucian: the six statistics shared one x-axis (-40..+45), so the grouped-
permutation rows (dAUC difference 0.106, ratio 3.54) collapsed under the marker — yet
those are now the rows carrying point 6. And the legend overlapped the RF-Gini observed
marker. Fix: one facet per (measure, statistic), each with its own x-scale, observed
grouping marked, and the exact one-sided P annotated per facet. No shared axis, no
legend collision.

Reads the enumerated values from exact_permutation.py's summary if present; otherwise
recomputes them from the same sources.

Run:  python src/figure_s6_redraw.py
"""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame
from sklearn.ensemble import RandomForestClassifier
from random_forest_shap import get_gini_importances, compute_importance_by_type, compute_shap_values
from cross_species_comparison import classify_variable

TABLES = Path("results/tables"); FIGDIR = Path("results/null_model"); FIGDIR.mkdir(parents=True, exist_ok=True)
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
OBSERVED_INTER = frozenset(INTERCONTINENTAL)
SEED, TREES = 42, 500
DOMAINS = ["Climate", "Topography", "Soil", "Land Cover"]


def cm_from_bytype(bt):
    return {sp: bt[sp]["Climate"] - bt[sp]["Topography"] for sp in bt}


def enumerate_vals(values, statistic):
    def st(inter):
        a = np.mean([values[s] for s in inter]); b = np.mean([values[s] for s in SPECIES if s not in inter])
        return (a - b) if statistic == "difference" else (a / b if b != 0 else np.nan)
    allv = [st(frozenset(c)) for c in combinations(SPECIES, 3)]
    obs = st(OBSERVED_INTER)
    finite = np.array([v for v in allv if np.isfinite(v)])
    rank = int(np.sum(finite >= obs))
    return allv, obs, rank, rank / len(finite)


def compute_sources():
    cfg = load_config("config/species_config.yaml")
    dff = apply_quality_filters(load_geotraits("data/raw/combined_data_true_master.csv"), cfg)
    env = get_env_variables(dff)["all_env"]
    gini, shap_bt = {}, {}
    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        X = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values
        clf = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1).fit(X, y)
        gt = compute_importance_by_type(get_gini_importances(clf, clean), "gini_importance")
        gini[sp] = {"Climate": gt.get("Climate", 0), "Topography": gt.get("Topography", 0)}
        shap_imp, _, _ = compute_shap_values(clf, X, clean)
        st = compute_importance_by_type(shap_imp, "mean_abs_shap")
        shap_bt[sp] = {"Climate": st.get("Climate", 0), "Topography": st.get("Topography", 0)}
    gp = json.loads((TABLES / "grouped_permutation_summary.json").read_text())
    gperm = {sp: {"Climate": gp[sp]["RF"]["Climate"]["drop_auc"],
                  "Topography": gp[sp]["RF"]["Topography"]["drop_auc"]} for sp in SPECIES}
    return {"RF-Gini": gini, "RF-SHAP": shap_bt, "grouped-permutation dAUC": gperm}


def main():
    sources = compute_sources()
    panels = []
    for name, bt in sources.items():
        metric = cm_from_bytype(bt)
        for stat in ("difference", "ratio"):
            allv, obs, rank, p = enumerate_vals(metric, stat)
            panels.append((f"{name}\n{stat}", allv, obs, rank, p))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.5 * n))
    rng = np.random.RandomState(0)
    for ax, (label, allv, obs, rank, p) in zip(axes, panels):
        finite = [v for v in allv if np.isfinite(v)]
        others = [v for v in finite if v != obs]
        ax.scatter(others, rng.uniform(-0.25, 0.25, len(others)), s=45,
                   color="#9E9E9E", alpha=0.8, zorder=2, label="null groupings")
        ax.scatter([obs], [0], s=170, marker="D", color="#E53935", edgecolor="black",
                   linewidth=0.7, zorder=3, label="observed")
        ax.set_yticks([]); ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        # independent x-scale with a little padding
        lo, hi = min(finite), max(finite); pad = (hi - lo) * 0.08 or 0.1
        ax.set_xlim(lo - pad, hi + pad)
        ax.annotate(f"observed = {obs:.3f}   rank {rank}/10   P = {p:.2f}",
                    xy=(0.5, 0.86), xycoords="axes fraction", ha="center", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3E0", ec="#E0A96D", lw=0.5))
        ax.grid(axis="x", alpha=0.3, zorder=1)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.9), ncol=2, fontsize=8, frameon=True)
    fig.suptitle("Figure S6. Exact permutation — all C(5,3)=10 attainable groupings per statistic\n"
                 "(observed in red; independent x-axis per row; exact one-sided P = rank/10)",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "figS6_exact_permutation_facets.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Panels (all should be rank 1/10):")
    for label, _, obs, rank, p in panels:
        print(f"  {label.splitlines()[0]:<28}{label.splitlines()[1]:<12} obs={obs:.4f}  rank {rank}/10  P={p:.2f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
