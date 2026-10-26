"""
r1_basin_id_audit.py — why the R1 spatial CV uses grid blocks instead of basin_id
(KM-7), and what the submitted basin-grouped CV (Table S5) gave.

The submitted spatial_cv_rf.py grouped records by the GeoTraits column basin_id with
GroupKFold(5). For part of the records basin_id is not a spatially coherent drainage
basin: records thousands of km apart share an id (e.g. one id holds Danube-basin
records together with records in central France, Iran and Uzbekistan). Grouping by it
therefore does not keep nearby records together, and for P. leptodactylus it puts
native and non-native records into the same "basin". GroupKFold also balances fold
sizes, not classes, so several held-out folds contained a single class.

This script reports, per species: number of basin_id groups, groups holding both
statuses and their records, the largest distance of a record from the median position
of its group, and the single-class folds of the submitted GroupKFold; plus, over the
whole GeoTraits table, the share of records lying > 1,000 km from their group's median.

Output: results/revision/tables/r1_basin_id_audit.csv

Run (from conservation/):  ../.venv/bin/python src/r1_basin_id_audit.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_loader import COL_SPECIES, COL_STATUS, load_geotraits
from r1_common import INPUT, LAT, LON, SPECIES, TABLES, build_species_frame, ensure_dirs, env_columns, haversine_km, load_filtered, xy


def spread(df):
    med = df.groupby("basin_id")[[LAT, LON]].transform("median")
    return haversine_km(df[LAT].to_numpy(float), df[LON].to_numpy(float), med[LAT].to_numpy(float), med[LON].to_numpy(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    ensure_dirs()
    dff = load_filtered(args.cache)
    env = env_columns(dff)
    rows = []
    for sp in SPECIES:
        combined, cols = build_species_frame(dff, sp, env)
        X, y = xy(combined, cols)
        d = spread(combined)
        st = combined.groupby("basin_id")[COL_STATUS].nunique()
        mixed = st[st > 1].index
        single = 0
        for _, te in GroupKFold(n_splits=5).split(X, y, combined["basin_id"]):
            single += int(len(np.unique(y[te])) < 2)
        rows.append({"scope": sp, "records": len(combined), "basin_id_groups": combined.basin_id.nunique(),
                     "groups_with_both_statuses": len(mixed),
                     "records_in_mixed_groups": int(combined.basin_id.isin(mixed).sum()),
                     "max_km_from_group_median": float(d.max()),
                     "records_over_1000km_from_group_median": int((d > 1000).sum()),
                     "submitted_groupkfold_single_class_test_folds": single})
    raw = load_geotraits(INPUT).dropna(subset=[LAT, LON, "basin_id"])
    d = spread(raw)
    rows.append({"scope": "all GeoTraits records", "records": len(raw), "basin_id_groups": raw.basin_id.nunique(),
                 "max_km_from_group_median": float(d.max()),
                 "records_over_1000km_from_group_median": int((d > 1000).sum()),
                 "groups_with_records_over_1000km": int(raw.loc[d > 1000, "basin_id"].nunique())})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "r1_basin_id_audit.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
