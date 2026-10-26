"""
r1_downsampling.py — R1 item KM-2 (second part): balanced subsamples for the two
species with unequal classes.

A. fulcisianus (227 native / 629 non-native) and A. pallipes (2,790 native / 231
non-native): in each of 100 replicates the majority class is randomly reduced to the
size of the minority class, and RF and DT are evaluated with stratified 5-fold CV and
with 100 km spatial blocks (seeds vary with the replicate). With balanced classes,
accuracy can no longer profit from predicting the majority class. RF Gini domain
shares from a fit on each balanced subsample are kept as well.

Outputs (results/revision/tables/)
  r1_downsampling_replicates.csv
  r1_downsampling_summary.csv   mean, SD, 2.5 and 97.5 percentiles over replicates

Run (from conservation/):  ../.venv/bin/python src/r1_downsampling.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from r1_common import (
    DOMAINS, LAT, LON, PRIMARY_BLOCK_KM, TABLES, build_species_frame, domain_shares,
    ensure_dirs, env_columns, load_filtered, make_dt, make_rf, metrics, oof_predict,
    spatial_splits, stratified_splits, xy,
)

TARGETS = ["Austropotamobius fulcisianus", "Austropotamobius pallipes"]
N_REP = 100
KEEP = ["accuracy", "balanced_accuracy", "mcc", "auc", "recall_native", "recall_nonnative"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--reps", type=int, default=N_REP)
    args = ap.parse_args()
    ensure_dirs()
    dff = load_filtered(args.cache)
    env = env_columns(dff)

    rows = []
    for sp in TARGETS:
        combined, cols = build_species_frame(dff, sp, env)
        X_all, y_all = xy(combined, cols)
        lat_all = combined[LAT].to_numpy(float); lon_all = combined[LON].to_numpy(float)
        n_min = int(min((y_all == 0).sum(), (y_all == 1).sum()))
        major = 0 if (y_all == 0).sum() > (y_all == 1).sum() else 1
        for r in range(args.reps):
            rs = np.random.RandomState(r)
            idx_major = rs.choice(np.flatnonzero(y_all == major), size=n_min, replace=False)
            idx = np.sort(np.r_[idx_major, np.flatnonzero(y_all != major)])
            X, y, lat, lon = X_all[idx], y_all[idx], lat_all[idx], lon_all[idx]
            for model_name, model in (("RF", make_rf(seed=r)), ("DT", make_dt(seed=r))):
                for cv_name, splits in (("stratified", stratified_splits(X, y, r)),
                                        (f"spatial_{PRIMARY_BLOCK_KM}km", spatial_splits(X, y, lat, lon, PRIMARY_BLOCK_KM, r))):
                    pred, prob, _, _ = oof_predict(model, X, y, splits)
                    m = metrics(y, pred, prob)
                    rows.append({"species": sp, "replicate": r, "model": model_name, "cv": cv_name,
                                 "n_per_class": n_min, **{k: m[k] for k in KEEP}, "n_unscored": m["n_unscored"]})
            shares = domain_shares(make_rf(seed=r).fit(X, y).feature_importances_, cols)
            rows.append({"species": sp, "replicate": r, "model": "RF", "cv": "full_fit_gini",
                         "n_per_class": n_min, **{f"gini_{d}": shares[d] for d in DOMAINS}})
            if r % 10 == 0:
                last = [x for x in rows if x["species"] == sp and x["replicate"] == r and x["cv"] == "stratified" and x["model"] == "RF"]
                print(f"{sp} rep {r}: RF stratified bacc={last[0]['balanced_accuracy']:.3f}", flush=True)

    rep = pd.DataFrame(rows)
    rep.to_csv(TABLES / "r1_downsampling_replicates.csv", index=False)
    num = KEEP + [f"gini_{d}" for d in DOMAINS]
    out = []
    for (sp, mdl, cv), g in rep.groupby(["species", "model", "cv"]):
        row = {"species": sp, "model": mdl, "cv": cv, "replicates": g.replicate.nunique(),
               "n_per_class": int(g.n_per_class.iloc[0])}
        for k in num:
            if k in g and g[k].notna().any():
                row[f"{k}_mean"] = g[k].mean(); row[f"{k}_sd"] = g[k].std(ddof=0)
                row[f"{k}_q025"] = g[k].quantile(0.025); row[f"{k}_q975"] = g[k].quantile(0.975)
        out.append(row)
    pd.DataFrame(out).to_csv(TABLES / "r1_downsampling_summary.csv", index=False)
    print(f"Saved r1_downsampling_*.csv to {TABLES}")


if __name__ == "__main__":
    main()
