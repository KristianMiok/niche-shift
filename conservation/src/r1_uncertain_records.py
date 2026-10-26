"""
r1_uncertain_records.py — R1 item KM-8: refit after excluding records whose
native / non-native assignment is less certain (Reviewer 3, major point 2).

Subsets (countries from reverse_geocoder on the record coordinates)
  A. fulcisianus : non-native records restricted to the Iberian Peninsula (ES, PT);
                   the few non-native records elsewhere (France, Austria, Germany,
                   Italy) are dropped. Native records unchanged.
  A. pallipes    : non-native records restricted to the island of Ireland (IE and
                   Northern Ireland, the documented medieval introduction); Great
                   Britain, France, Italy and Austria dropped. Native unchanged.
  P. leptodactylus: records lying within X km of the nearest record of the other
                   status are dropped on both sides, X = 25, 50, 100 km. This removes
                   the zone where the drainage-based boundary separates the ranges
                   and where misassignment is most likely, without needing the
                   boundary itself.

Each subset is re-cleaned exactly as the full data (data_preparation.py) and refitted
with the submitted RF and DT settings; stratified 5-fold CV and 100 km spatial blocks
(5 seeds). RF Gini domain shares from a fit on the whole subset.

Output: results/revision/tables/r1_uncertain_records.csv

Run (from conservation/):  ../.venv/bin/python src/r1_uncertain_records.py
"""
from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
import pandas as pd
import reverse_geocoder as rg

from data_loader import COL_SPECIES, COL_STATUS
from r1_common import (
    DOMAINS, LAT, LON, PRIMARY_BLOCK_KM, SEED, TABLES, build_species_frame, domain_shares,
    ensure_dirs, env_columns, load_filtered, make_dt, make_rf, metrics,
    oof_predict, spatial_splits, stratified_splits, xy,
)
from sklearn.neighbors import BallTree
from species_selector import STATUS_ALIEN, STATUS_NATIVE

KEEP = ["accuracy", "balanced_accuracy", "mcc", "auc", "recall_native", "recall_nonnative"]


def geocode(df):
    res = rg.search(list(zip(df[LAT].astype(float), df[LON].astype(float))), mode=1, verbose=False)
    return pd.Series([r["cc"] for r in res], index=df.index), pd.Series([r["admin1"] for r in res], index=df.index)


def opposite_status_distance_km(sp_df):
    out = pd.Series(np.nan, index=sp_df.index)
    for s, o in ((STATUS_NATIVE, STATUS_ALIEN), (STATUS_ALIEN, STATUS_NATIVE)):
        a, b = sp_df[sp_df[COL_STATUS] == s], sp_df[sp_df[COL_STATUS] == o]
        tree = BallTree(np.radians(b[[LAT, LON]].to_numpy(float)), metric="haversine")
        d, _ = tree.query(np.radians(a[[LAT, LON]].to_numpy(float)), k=1)
        out.loc[a.index] = d[:, 0] * 6371.0
    return out


def subsets(dff):
    """Yield (species, subset_name, description, filtered table for that species)."""
    for sp in ["Pontastacus leptodactylus", "Austropotamobius fulcisianus", "Austropotamobius pallipes"]:
        sp_df = dff[dff[COL_SPECIES] == sp].copy()
        yield sp, "full", "all records (submitted data)", sp_df
        if sp == "Austropotamobius fulcisianus":
            cc, _ = geocode(sp_df)
            keep = (sp_df[COL_STATUS] == STATUS_NATIVE) | cc.isin(["ES", "PT"])
            yield sp, "nonnative_iberia_only", "non-native restricted to Spain and Portugal", sp_df[keep]
        if sp == "Austropotamobius pallipes":
            cc, adm = geocode(sp_df)
            ireland = (cc == "IE") | ((cc == "GB") & (adm == "Northern Ireland"))
            keep = (sp_df[COL_STATUS] == STATUS_NATIVE) | ireland
            yield sp, "nonnative_ireland_only", "non-native restricted to the island of Ireland", sp_df[keep]
        if sp == "Pontastacus leptodactylus":
            d = opposite_status_distance_km(sp_df)
            for x in (25, 50, 100):
                yield sp, f"boundary_buffer_{x}km", f"records within {x} km of the other status removed", sp_df[d > x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    ensure_dirs()
    dff = load_filtered(args.cache)
    env = env_columns(dff)

    rows = []
    for sp, name, desc, sub in subsets(dff):
        with contextlib.redirect_stdout(io.StringIO()):
            combined, cols = build_species_frame(sub, sp, env)
        X, y = xy(combined, cols)
        lat, lon = combined[LAT].to_numpy(float), combined[LON].to_numpy(float)
        base = {"species": sp, "subset": name, "description": desc,
                "n_native": int((y == 0).sum()), "n_nonnative": int((y == 1).sum()), "n_variables": len(cols)}
        shares = domain_shares(make_rf().fit(X, y).feature_importances_, cols)
        for model_name, model in (("RF", make_rf()), ("DT", make_dt())):
            for cv_name, seeds in (("stratified", [SEED]), (f"spatial_{PRIMARY_BLOCK_KM}km", [SEED + i for i in range(5)])):
                ms = []
                for s in seeds:
                    splits = stratified_splits(X, y, s) if cv_name == "stratified" else spatial_splits(X, y, lat, lon, PRIMARY_BLOCK_KM, s)
                    pred, prob, _, _ = oof_predict(model, X, y, splits)
                    ms.append(metrics(y, pred, prob))
                row = {**base, "model": model_name, "cv": cv_name, "seeds": len(seeds)}
                for k in KEEP:
                    row[k] = float(np.mean([m[k] for m in ms]))
                    row[f"{k}_sd"] = float(np.std([m[k] for m in ms]))
                if model_name == "RF":
                    row.update({f"rf_gini_{d}": shares[d] for d in DOMAINS})
                rows.append(row)
                print(f"{sp[:18]:<18} {name:<24} n={base['n_native']}/{base['n_nonnative']} {model_name} {cv_name:<14} "
                      f"acc={row['accuracy']:.3f} bacc={row['balanced_accuracy']:.3f} mcc={row['mcc']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "r1_uncertain_records.csv", index=False)
    print(f"Saved {TABLES / 'r1_uncertain_records.csv'}")


if __name__ == "__main__":
    main()
