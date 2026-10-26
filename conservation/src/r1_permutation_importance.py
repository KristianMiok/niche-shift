"""
r1_permutation_importance.py — R1 item KM-3: permutation importance on held-out folds,
aggregated by environmental domain, next to Gini and SHAP.

Two variants, both computed on the held-out fold of each training split:

  grouped  (primary)  all variables of one domain are permuted together with one
                      shared row shuffle; the loss of balanced accuracy and of AUC is
                      the domain's importance. Within-domain correlation cannot leak
                      importance to a correlated sibling, which is the weakness of
                      per-variable permutation with ~200 intercorrelated predictors.
                      Same method as Paper 1 (src/grouped_permutation.py), here with
                      balanced accuracy as the score. 20 shuffles per domain and fold.
                      Stratified 5-fold CV (as submitted) and 100 km spatial blocks.
  variable            sklearn permutation_importance per variable (balanced accuracy,
                      5 shuffles), RF only, stratified CV; summed by domain as % of the
                      positive total, for direct comparison with the Gini/SHAP shares.

Outputs (results/revision/tables/)
  r1_permutation_grouped.csv         species x model x CV x domain
  r1_permutation_variable.csv        species x variable (mean over folds and shuffles)
  r1_importance_domain_comparison.csv  RF Gini / SHAP / per-variable permutation shares
                                       and grouped drops side by side (Table 3)

Run (from conservation/):  ../.venv/bin/python src/r1_permutation_importance.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from r1_common import (
    DOMAINS, LAT, LON, PRIMARY_BLOCK_KM, SEED, SPECIES, TABLES, build_species_frame,
    domain_index_map, domain_of, domain_shares, ensure_dirs, env_columns, load_filtered,
    make_dt, make_rf, spatial_splits, stratified_splits, xy,
)

N_GROUPED = 20
N_VARIABLE = 5


def auc_or_nan(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def grouped(model, X, y, cols, splits, rng):
    dmap = domain_index_map(cols)
    drops = {d: {"bacc": [], "auc": []} for d in dmap}
    base_b, base_a = [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        m = clone(model).fit(X[tr], y[tr])
        Xte, yte = X[te], y[te]
        b0 = balanced_accuracy_score(yte, m.predict(Xte))
        a0 = auc_or_nan(yte, m.predict_proba(Xte)[:, 1])
        base_b.append(b0); base_a.append(a0)
        for d, ix in dmap.items():
            for _ in range(N_GROUPED):
                Xp = Xte.copy()
                perm = rng.permutation(len(yte))
                Xp[:, ix] = Xp[np.ix_(perm, ix)]
                drops[d]["bacc"].append(b0 - balanced_accuracy_score(yte, m.predict(Xp)))
                drops[d]["auc"].append(a0 - auc_or_nan(yte, m.predict_proba(Xp)[:, 1]))
    out = []
    for d in dmap:
        b = np.array(drops[d]["bacc"]); a = np.array(drops[d]["auc"], dtype=float)
        out.append({"domain": d, "n_variables": int(len(dmap[d])),
                    "baseline_bacc": float(np.mean(base_b)), "baseline_auc": float(np.nanmean(base_a)),
                    "drop_bacc_mean": float(b.mean()), "drop_bacc_sd": float(b.std()),
                    "drop_auc_mean": float(np.nanmean(a)), "drop_auc_sd": float(np.nanstd(a))})
    return out


def per_variable(model, X, y, cols, splits, seed):
    imps = []
    for tr, te in splits:
        m = clone(model).fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], scoring="balanced_accuracy",
                                   n_repeats=N_VARIABLE, random_state=seed, n_jobs=1)
        imps.append(r.importances_mean)
    return np.mean(imps, axis=0), np.std(imps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--species", default=None)
    ap.add_argument("--skip-variable", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    dff = load_filtered(args.cache)
    env = env_columns(dff)
    species = [args.species] if args.species else SPECIES

    g_rows, v_rows, cmp_rows = [], [], []
    for sp in species:
        combined, cols = build_species_frame(dff, sp, env)
        X, y = xy(combined, cols)
        lat, lon = combined[LAT].to_numpy(float), combined[LON].to_numpy(float)
        rng = np.random.RandomState(SEED)
        strat = stratified_splits(X, y, SEED)
        spat = spatial_splits(X, y, lat, lon, PRIMARY_BLOCK_KM, SEED)
        for model_name, model in (("RF", make_rf()), ("DT", make_dt())):
            for cv_name, splits in (("stratified", strat), (f"spatial_{PRIMARY_BLOCK_KM}km", spat)):
                for r in grouped(model, X, y, cols, splits, rng):
                    g_rows.append({"species": sp, "model": model_name, "cv": cv_name, **r})
                    print(f"{sp[:20]:<20} {model_name} {cv_name:<14} {r['domain']:<11} "
                          f"dBAcc={r['drop_bacc_mean']:.4f}  dAUC={r['drop_auc_mean']:.4f}")
        if not args.skip_variable:
            mean, sd = per_variable(make_rf(), X, y, cols, strat, SEED)
            for v, m_, s_ in zip(cols, mean, sd):
                v_rows.append({"species": sp, "variable": v, "domain": domain_of(v),
                               "perm_bacc_mean": float(m_), "perm_bacc_sd_over_folds": float(s_)})
            perm_share = domain_shares(mean, cols)
        else:
            perm_share = {d: np.nan for d in DOMAINS}

        # side-by-side with the (pinned-environment) Gini and SHAP outputs of random_forest_shap.py
        safe = sp.replace(" ", "_").lower()
        gini = pd.read_csv(Path("results/tables") / f"{safe}_rf_gini_importances.csv")
        shap = pd.read_csv(Path("results/tables") / f"{safe}_rf_shap_importances.csv")
        g_share = domain_shares(gini.gini_importance.to_numpy(), gini.variable.tolist())
        s_share = domain_shares(shap.mean_abs_shap.to_numpy(), shap.variable.tolist())
        for d in DOMAINS:
            gr = [r for r in g_rows if r["species"] == sp and r["model"] == "RF"
                  and r["cv"] == "stratified" and r["domain"] == d]
            gs = [r for r in g_rows if r["species"] == sp and r["model"] == "RF"
                  and r["cv"] == f"spatial_{PRIMARY_BLOCK_KM}km" and r["domain"] == d]
            cmp_rows.append({"species": sp, "domain": d,
                             "rf_gini_pct": g_share[d], "rf_shap_pct": s_share[d],
                             "rf_perm_variable_pct": perm_share[d],
                             "rf_grouped_drop_bacc": gr[0]["drop_bacc_mean"] if gr else np.nan,
                             "rf_grouped_drop_auc": gr[0]["drop_auc_mean"] if gr else np.nan,
                             "rf_grouped_drop_bacc_spatial": gs[0]["drop_bacc_mean"] if gs else np.nan})

    pd.DataFrame(g_rows).to_csv(TABLES / "r1_permutation_grouped.csv", index=False)
    if v_rows:
        pd.DataFrame(v_rows).sort_values(["species", "perm_bacc_mean"], ascending=[True, False]) \
            .to_csv(TABLES / "r1_permutation_variable.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(TABLES / "r1_importance_domain_comparison.csv", index=False)
    print(f"Saved r1_permutation_*.csv and r1_importance_domain_comparison.csv to {TABLES}")


if __name__ == "__main__":
    main()
