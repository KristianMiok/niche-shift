"""
pathway_refit_grouped.py — Task (vi): purity refit under grouped permutation.

The original purity refit (task d) reported decision-tree domain importance, but we now
retire the decision tree at the domain level (single-tree instability). So we cannot
defend the pathway labels with the method we discard. This reruns the same purity test —
drop the wrong-continent invaded records, refit, compare domain importance — using
GROUPED PERMUTATION dAUC (collinearity-immune, the method that now carries point 6),
with RF Gini reported alongside for continuity.

Species tested (the two with mixed pathway records, per the continent audit):
  F. virilis (within-continent): drop 43 European invaded -> NA-only.
    Q: does the topographic grouped-perm signal survive?
  P. clarkii (intercontinental): drop 2249 NA invaded -> intercontinental-only.
    Q: does climate dominance survive?

Continent per invaded record = offline reverse_geocoder on lat_or/long_or (as in tasks
a/d; 0 Americas/Old-World mismatches). Feature set frozen from the baseline so any change
is attributable to the records, not to variable selection.

Run:  python src/pathway_refit_grouped.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import load_geotraits, load_config, get_env_variables, COL_STATUS
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame, domain_index_map, DOMAINS
from random_forest_shap import get_gini_importances, compute_importance_by_type

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, TREES, REPEATS = 42, 500, 10
LAT, LON = "lat_or", "long_or"
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
CONT_NAME = {"NA": "North America", "SA": "South America", "EU": "Europe",
             "AS": "Asia", "AF": "Africa", "OC": "Oceania", "AN": "Antarctica"}
REFIT = {
    "Faxonius virilis": {"keep": {"North America"}, "label": "NA-only (drop European)", "focus": "Topography"},
    "Procambarus clarkii": {"keep": {"Europe", "Asia", "Africa", "South America"},
                            "label": "intercontinental-only (drop North American)", "focus": "Climate"},
}


def continents_for(df):
    import reverse_geocoder as rg
    from pycountry_convert import country_alpha2_to_continent_code
    coords = list(zip(df[LAT].astype(float), df[LON].astype(float)))
    out = []
    for r in rg.search(coords, mode=1):
        cc = (r.get("cc") or "").upper()
        try:
            out.append(CONT_NAME.get(country_alpha2_to_continent_code(cc), "Unknown"))
        except Exception:
            out.append("Unknown")
    return out


def grouped_perm_dauc(X, y, clean, seed=SEED, n_repeats=REPEATS):
    """Domain-level grouped-permutation dAUC on held-out 5-fold splits (as in task e)."""
    dmap = domain_index_map(clean)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    model = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
    base_auc, per = [], {d: [] for d in dmap}
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        Xte = X[te]
        b = roc_auc_score(y[te], m.predict_proba(Xte)[:, 1])
        base_auc.append(b)
        for d, cols in dmap.items():
            for _ in range(n_repeats):
                Xp = Xte.copy()
                perm = rng.permutation(Xp.shape[0])
                Xp[:, cols] = Xp[np.ix_(perm, cols)]
                per[d].append(b - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
    return {d: float(np.mean(per[d])) for d in dmap}, float(np.mean(base_auc))


def rf_gini_domain(X, y, clean):
    clf = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1).fit(X, y)
    bt = compute_importance_by_type(get_gini_importances(clf, clean), "gini_importance")
    tot = sum(bt.get(d, 0) for d in DOMAINS)
    return {d: 100 * bt.get(d, 0) / tot for d in DOMAINS}


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    summary = {}

    for sp, spec in REFIT.items():
        print("\n" + "=" * 88)
        print(f"{sp}  —  grouped-perm purity refit  [{spec['label']}]")
        print("=" * 88)
        combined, clean = build_species_frame(dff, sp, env)
        inv_mask = combined["range_label"] == "invasive"
        inv = combined[inv_mask].copy()
        inv["_cont"] = continents_for(inv)
        drop_idx = inv.index[~inv["_cont"].isin(spec["keep"])]
        print(f"  invaded by continent: " + ", ".join(f"{k}={v}" for k, v in inv['_cont'].value_counts().items()))
        print(f"  dropping {len(drop_idx)} invaded rows outside {sorted(spec['keep'])}")
        combined_refit = combined.drop(index=drop_idx).reset_index(drop=True)

        def block(frame, tag):
            X = frame[clean].values
            y = (frame["range_label"] == "invasive").astype(int).values
            gp, auc = grouped_perm_dauc(X, y, clean)
            gini = rf_gini_domain(X, y, clean)
            print(f"\n  --- {tag} (n_inv={int(y.sum())}, AUC={auc:.3f}) ---")
            print("    grouped-perm dAUC: " + "  ".join(f"{d.split()[0][:4]}={gp.get(d,0):.4f}" for d in DOMAINS))
            print("    RF Gini %        : " + "  ".join(f"{d.split()[0][:4]}={gini.get(d,0):5.1f}" for d in DOMAINS))
            return {"n_invaded": int(y.sum()), "auc": auc, "grouped_perm_dauc": gp, "rf_gini_pct": gini}

        base = block(combined, "baseline (all invaded)")
        refit = block(combined_refit, spec["label"])

        f = spec["focus"]
        print(f"\n  >>> FOCUS [{f}]  grouped-perm dAUC {base['grouped_perm_dauc'][f]:.4f} -> {refit['grouped_perm_dauc'][f]:.4f}"
              f"   |  RF Gini {base['rf_gini_pct'][f]:.1f}% -> {refit['rf_gini_pct'][f]:.1f}%")
        summary[sp] = {"focus": f, "n_dropped": int(len(drop_idx)), "baseline": base, "refit": refit}

    (OUTDIR / "pathway_refit_grouped_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved: {OUTDIR/'pathway_refit_grouped_summary.json'}")


if __name__ == "__main__":
    main()
