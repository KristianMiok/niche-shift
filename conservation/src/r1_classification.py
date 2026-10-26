"""
r1_classification.py — R1 items KM-2 (class-imbalance metrics) and KM-4 (models
without climate), plus the spatially blocked CV that replaces the basin_id grouping
of the submitted Table S5 (KM-7).

For each species x feature set x model x CV scheme, predictions are pooled over the
five held-out folds and scored once: accuracy, balanced accuracy, MCC, AUC,
per-class precision / recall / F1 and the confusion matrix, next to the
majority-class baseline. Per-fold values are written as well.

Feature sets
  all          : all cleaned variables (the submitted models)
  no_climate   : Topography + Soil + Land Cover only (KM-4, Reviewer 3 main point)
  climate_only : Climate only (reference for KM-4)
  no_climate_no_cellstep : no_climate without l_TOP101-104 (cell step length, a
                 latitude proxy of the 3-arc-second grid)

CV schemes
  stratified   : StratifiedKFold(5, shuffle, random_state=42) - as submitted
  spatial_Bkm  : StratifiedGroupKFold(5, shuffle) on grid cells of B km
                 (B = 50, 100, 250, 500; 100 km primary, repeated over 5 seeds;
                 other sizes over 3 seeds). Same blocks as Paper 1.

Outputs (results/revision/tables/)
  r1_classification_metrics.csv    one row per species x set x model x scheme x seed
  r1_classification_summary.csv    mean / SD over seeds
  r1_classification_per_fold.csv   per held-out fold
  r1_noclimate_top_variables.csv   RF Gini top 15 of the no-climate model (full fit)

Run (from conservation/):  ../.venv/bin/python src/r1_classification.py
                           ../.venv/bin/python src/r1_classification.py --sets no_climate_no_cellstep
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from r1_common import (
    BLOCK_SIZES_KM, DOMAINS, LAT, LON, PRIMARY_BLOCK_KM, SEED, SPECIES, TABLES,
    build_species_frame, domain_of, domain_shares, ensure_dirs, env_columns,
    load_filtered, make_dt, make_rf, metrics, oof_predict, per_fold_metrics,
    spatial_splits, stratified_splits, xy,
)

# l_TOP101-104 ("cell upstream distance") are the length of one flow step between
# 3-arc-second cells; the east-west cell width shrinks with latitude, so these
# variables are a latitude proxy (P. leptodactylus: median 117 m at 35-40 N, 103 m at
# 60-65 N) rather than a habitat property.
CELL_STEP = {"l_TOP101", "l_TOP102", "l_TOP103", "l_TOP104", "u_TOP101", "u_TOP102", "u_TOP103", "u_TOP104"}

FEATURE_SETS = {
    "all": lambda v: True,
    "no_climate": lambda v: domain_of(v) in ("Topography", "Soil", "Land Cover"),
    "climate_only": lambda v: domain_of(v) == "Climate",
    "no_climate_no_cellstep": lambda v: domain_of(v) in ("Topography", "Soil", "Land Cover") and v not in CELL_STEP,
}


def schemes(full: bool):
    """(name, block_km or None, seeds)."""
    out = [("stratified", None, [SEED])]
    if full:
        for b in BLOCK_SIZES_KM:
            seeds = [SEED + i for i in range(5)] if b == PRIMARY_BLOCK_KM else [SEED + i for i in range(3)]
            out.append((f"spatial_{b}km", b, seeds))
    else:
        out.append((f"spatial_{PRIMARY_BLOCK_KM}km", PRIMARY_BLOCK_KM, [SEED + i for i in range(5)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=None)
    ap.add_argument("--cache", default=None, help="optional pickle of the filtered records")
    ap.add_argument("--sets", default=None, help="comma-separated feature sets to (re)run; "
                    "results for other sets already on disk are kept")
    args = ap.parse_args()
    run_sets = args.sets.split(",") if args.sets else list(FEATURE_SETS)
    ensure_dirs()

    dff = load_filtered(args.cache)
    env = env_columns(dff)
    species = [args.species] if args.species else SPECIES

    rows, fold_rows, top_rows = [], [], []
    for sp in species:
        combined, clean = build_species_frame(dff, sp, env)
        lat = combined[LAT].astype(float).to_numpy()
        lon = combined[LON].astype(float).to_numpy()
        assert not np.isnan(lat).any() and not np.isnan(lon).any()
        print(f"\n=== {sp}: n={len(combined)}, {len(clean)} variables")

        for fs_name, keep in FEATURE_SETS.items():
            if fs_name not in run_sets:
                continue
            cols = [v for v in clean if keep(v)]
            X, y = xy(combined, cols)
            for model_name, model in (("RF", make_rf()), ("DT", make_dt())):
                for scheme, block, seeds in schemes(full=(fs_name == "all")):
                    for seed in seeds:
                        if block is None:
                            splits = stratified_splits(X, y, seed)
                        else:
                            splits = spatial_splits(X, y, lat, lon, block, seed)
                        pred, prob, fold_id, _ = oof_predict(model, X, y, splits)
                        m = metrics(y, pred, prob)
                        m.update(species=sp, feature_set=fs_name, n_variables=len(cols),
                                 model=model_name, cv=scheme, block_km=block, seed=seed)
                        rows.append(m)
                        pf = per_fold_metrics(y, pred, prob, fold_id)
                        pf = pf.assign(species=sp, feature_set=fs_name, model=model_name,
                                       cv=scheme, block_km=block, seed=seed)
                        fold_rows.append(pf)
                    last = [r for r in rows if r["species"] == sp and r["feature_set"] == fs_name
                            and r["model"] == model_name and r["cv"] == scheme]
                    print(f"  {fs_name:<12} {model_name}  {scheme:<14} "
                          f"acc={np.mean([r['accuracy'] for r in last]):.3f}  "
                          f"bacc={np.mean([r['balanced_accuracy'] for r in last]):.3f}  "
                          f"mcc={np.mean([r['mcc'] for r in last]):.3f}  "
                          f"auc={np.mean([r['auc'] for r in last]):.3f}")

            if fs_name.startswith("no_climate"):
                rf = make_rf().fit(X, y)
                imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
                shares = domain_shares(rf.feature_importances_, cols)
                for rank, (v, val) in enumerate(imp.head(15).items(), start=1):
                    top_rows.append({"species": sp, "feature_set": fs_name, "rank": rank, "variable": v,
                                     "domain": domain_of(v), "rf_gini": float(val),
                                     **{f"model_share_{d}": shares[d] for d in DOMAINS}})

    res = pd.DataFrame(rows)
    folds = pd.concat(fold_rows, ignore_index=True)
    if args.sets:  # merge with results of the other feature sets already on disk
        for name, new in (("r1_classification_metrics.csv", res), ("r1_classification_per_fold.csv", folds)):
            path = TABLES / name
            if path.exists():
                old = pd.read_csv(path)
                old = old[~old.feature_set.isin(run_sets)]
                new = pd.concat([old, new], ignore_index=True)
            new.to_csv(path, index=False)
        res = pd.read_csv(TABLES / "r1_classification_metrics.csv")
    else:
        res.to_csv(TABLES / "r1_classification_metrics.csv", index=False)
        folds.to_csv(TABLES / "r1_classification_per_fold.csv", index=False)
    if top_rows:
        top = pd.DataFrame(top_rows)
        path = TABLES / "r1_noclimate_top_variables.csv"
        if args.sets and path.exists():
            old = pd.read_csv(path)
            if "feature_set" not in old:
                old["feature_set"] = "no_climate"
            top = pd.concat([old[~old.feature_set.isin(run_sets)], top], ignore_index=True)
        top.to_csv(path, index=False)

    keys = ["species", "feature_set", "n_variables", "model", "cv", "block_km"]
    num = ["accuracy", "balanced_accuracy", "mcc", "auc", "recall_native", "recall_nonnative",
           "precision_native", "precision_nonnative", "f1_native", "f1_nonnative", "majority_baseline"]
    g = res.groupby(keys, dropna=False)[num]
    summ = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_sd")).join(g.size().rename("n_seeds"))
    summ.reset_index().to_csv(TABLES / "r1_classification_summary.csv", index=False)
    print(f"\nSaved r1_classification_*.csv to {TABLES}")


if __name__ == "__main__":
    main()
