"""
exact_permutation.py — Task (c): exact permutation test by complete enumeration,
regenerated on the post-dedup dataset for full internal consistency.

Reviewer 2 point 4: with five species split 3/2 there are only C(5,3)=10 unique
groupings, so 999 random permutations cannot yield P=0.001. We enumerate all ten
groupings, report every attainable statistic value, the observed rank, and the exact
one-sided P, for Gini and SHAP (difference and ratio; Lucian's spec) plus grouped
permutation (difference and ratio; collinearity-immune robustness). Null-mean/null-SD
are dropped. Figure S6 becomes a strip plot of the ten attainable values.

RECONCILIATION (confirmed): the manuscript 41.20 is the full-model RF-Gini
group-difference statistic (mean climate-topography over intercontinental minus over
within-continent). Lucian's 41.37 from the Results 3.3 fold-mean table is P. clarkii's
INDIVIDUAL climate-topography value (~41.3), not the group statistic. The test used
the group statistic. This script prints the stale Apr-3 source once to confirm that
origin, then computes and reports everything FRESH on the current deduplicated pipeline.

Run:  python src/exact_permutation.py
"""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame
from random_forest_shap import (
    get_gini_importances, compute_importance_by_type, compute_shap_values,
)

INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
OBSERVED_INTER = frozenset(INTERCONTINENTAL)

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, TREES = 42, 500
OUTDIR = Path("results/null_model"); OUTDIR.mkdir(parents=True, exist_ok=True)
TABLES = Path("results/tables"); TABLES.mkdir(parents=True, exist_ok=True)


# ------------------------- statistics -------------------------
def group_diff(values, inter_set):
    inter = np.mean([values[s] for s in inter_set])
    within = np.mean([values[s] for s in SPECIES if s not in inter_set])
    return float(inter - within)


def enumerate_stat(values, statistic):
    def stat_for(inter_set):
        if statistic == "difference":
            return group_diff(values, inter_set)
        inter = np.mean([values[s] for s in inter_set])
        within = np.mean([values[s] for s in SPECIES if s not in inter_set])
        return float(inter / within) if within != 0 else np.nan
    rows = [(frozenset(c), stat_for(frozenset(c))) for c in combinations(SPECIES, 3)]
    observed = next(v for iset, v in rows if iset == OBSERVED_INTER)
    return rows, observed


def exact_p(rows, observed):
    vals = np.array([v for _, v in rows], dtype=float)
    finite = vals[np.isfinite(vals)]
    rank = int(np.sum(finite >= observed))   # 1 = most extreme; >= includes self
    return rank, len(finite), float(rank / len(finite))


def cm_series(by_type):
    return {sp: by_type[sp]["Climate"] - by_type[sp]["Topography"] for sp in by_type}


# ------------------------- data sources -------------------------
def compute_fresh(dff, env):
    """Fresh RF Gini + SHAP by-type per species from the current (dedup) pipeline."""
    print("\n--- FRESH (post-dedup) RF importances by type [%] ---")
    gini_src, shap_src, do_shap = {}, {}, True
    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        X = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values
        clf = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1).fit(X, y)
        gt = compute_importance_by_type(get_gini_importances(clf, clean), "gini_importance")
        gini_src[sp] = {"Climate": float(gt.get("Climate", 0)), "Topography": float(gt.get("Topography", 0))}
        line = f"  {sp:<26} Gini C={gini_src[sp]['Climate']:6.2f} T={gini_src[sp]['Topography']:6.2f} C-T={gini_src[sp]['Climate']-gini_src[sp]['Topography']:6.2f}"
        if do_shap:
            try:
                shap_imp, _, _ = compute_shap_values(clf, X, clean)
                st = compute_importance_by_type(shap_imp, "mean_abs_shap")
                shap_src[sp] = {"Climate": float(st.get("Climate", 0)), "Topography": float(st.get("Topography", 0))}
                line += f"  |  SHAP C={shap_src[sp]['Climate']:6.2f} T={shap_src[sp]['Topography']:6.2f}"
            except Exception as e:
                print(f"  [i] SHAP unavailable ({type(e).__name__}); reporting Gini + grouped-perm only.")
                do_shap, shap_src = False, None
        print(line)
    out = {"RF-Gini": gini_src}
    if shap_src:
        out["RF-SHAP"] = shap_src
    return out


def from_grouped_perm(path="results/tables/grouped_permutation_summary.json"):
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    out = {}
    for sp, res in data.items():
        if sp in SPECIES:
            rf = res.get("RF", {})
            out[sp] = {"Climate": float(rf.get("Climate", {}).get("drop_auc", 0)),
                       "Topography": float(rf.get("Topography", {}).get("drop_auc", 0))}
    return {"grouped-permutation RF dAUC": out} if out else {}


def confirm_reconciliation(path="results/tables/rf_shap_summary.json"):
    p = Path(path)
    if not p.exists():
        print("\n[reconciliation] stale rf_shap_summary.json not found — narrative stands from prior run.")
        return
    data = json.loads(p.read_text())
    by = {}
    for sp, res in data.items():
        if sp in SPECIES:
            g = res.get("gini_by_type") or res.get("importance_by_type") or {}
            by[sp] = {"Climate": float(g.get("Climate", 0)), "Topography": float(g.get("Topography", 0))}
    if len(by) == len(SPECIES):
        gd = group_diff(cm_series(by), OBSERVED_INTER)
        cl = by["Procambarus clarkii"]["Climate"] - by["Procambarus clarkii"]["Topography"]
        print("\n[reconciliation — stale Apr-3 source, confirms manuscript origin]")
        print(f"  RF-Gini group-difference statistic = {gd:.4f}   (manuscript 41.20)")
        print(f"  P. clarkii individual climate-topography = {cl:.4f}   (Lucian's 41.37 from 3.3 fold-mean table)")
        print("  -> 41.20 is the group statistic; 41.37 is one species' value. Test used the group statistic.")


# ------------------------- run -------------------------
def main():
    config = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), config)
    env = get_env_variables(dff)["all_env"]

    confirm_reconciliation()
    sources = {**compute_fresh(dff, env), **from_grouped_perm()}

    print("\n" + "#" * 88)
    print("# EXACT ENUMERATION over C(5,3)=10 groupings  (all numbers FRESH / post-dedup)")
    print("#" * 88)
    table_rows, strip_data = [], {}
    for name, by_type in sources.items():
        metric = cm_series(by_type)
        for statistic in ("difference", "ratio"):
            rows, observed = enumerate_stat(metric, statistic)
            rank, n, p = exact_p(rows, observed)
            vals = sorted([v for _, v in rows if np.isfinite(v)], reverse=True)
            strip_data[(name, statistic)] = ([v for _, v in rows], observed)
            role = "robustness (collinearity-immune)" if name.startswith("grouped") else "primary (point 4)"
            print(f"\n[{name}] {statistic}  ({role})")
            print(f"  observed = {observed:.4f} | rank {rank}/{n} | exact one-sided P = {p:.3f}")
            print(f"  all {n} values: " + ", ".join(f"{v:.3f}" for v in vals))
            table_rows.append({"statistic": statistic, "importance_measure": name,
                               "observed": round(observed, 4), "rank_among_10": f"{rank}/{n}",
                               "exact_one_sided_P": round(p, 3)})

    tbl = pd.DataFrame(table_rows)[["statistic", "importance_measure", "observed", "rank_among_10", "exact_one_sided_P"]]
    tbl.to_csv(TABLES / "TableS1_exact_permutation.csv", index=False)
    print("\n===== Table S1 (regenerated, post-dedup) =====")
    print(tbl.to_string(index=False))
    print(f"\nSaved: {TABLES / 'TableS1_exact_permutation.csv'}")

    keys = list(strip_data.keys())
    fig, ax = plt.subplots(figsize=(9, max(3, 0.7 * len(keys) + 1)))
    for i, k in enumerate(keys):
        all_vals, observed = strip_data[k]
        finite = [v for v in all_vals if np.isfinite(v)]
        jit = (np.random.RandomState(0).rand(len(finite)) - 0.5) * 0.12
        ax.scatter(finite, np.full(len(finite), i) + jit, s=40, color="#9E9E9E", alpha=0.8, zorder=2)
        ax.scatter([observed], [i], s=140, marker="D", color="#E53935", edgecolor="black",
                   linewidth=0.6, zorder=3, label="observed" if i == 0 else None)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{m}\n{s}" for (m, s) in keys], fontsize=7)
    ax.set_xlabel("Statistic value (all C(5,3)=10 attainable groupings)")
    ax.set_title("Figure S6. Exact permutation — 10 attainable groupings\n"
                 "(observed grouping in red; exact one-sided P = rank/10)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3, zorder=1)
    plt.tight_layout()
    fig.savefig(OUTDIR / "figS6_exact_permutation_stripplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTDIR / 'figS6_exact_permutation_stripplot.png'}")


if __name__ == "__main__":
    main()
