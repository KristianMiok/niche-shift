"""
r1_status_by_country.py — R1 item 10 (how native / non-native statuses were assigned):
records used in the models by species, country and GeoTraits status.

The models used the Status column of the GeoTraits table as it stands in
data/raw/combined_data_true_master.csv; this table lets the range-assignment text in
Section 2.1 be checked against the data actually modelled (e.g. P. leptodactylus in
Turkey, Greece, Belarus, Russia and Poland, where the drainage-based boundary runs).
Countries from reverse_geocoder on the record coordinates (offline, admin level 0;
Northern Ireland reported separately from Great Britain).

Output: results/revision/tables/r1_status_by_country.csv

Run (from conservation/):  ../.venv/bin/python src/r1_status_by_country.py
"""
from __future__ import annotations

import argparse

import pandas as pd
import reverse_geocoder as rg

from data_loader import COL_SPECIES, COL_STATUS
from r1_common import LAT, LON, SPECIES, TABLES, ensure_dirs, load_filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    ensure_dirs()
    dff = load_filtered(args.cache)
    d = dff[dff[COL_SPECIES].isin(SPECIES)].copy()
    res = rg.search(list(zip(d[LAT].astype(float), d[LON].astype(float))), mode=1, verbose=False)
    d["country"] = [("GB-NIR" if (r["cc"] == "GB" and r["admin1"] == "Northern Ireland") else r["cc"]) for r in res]
    t = (d.groupby([COL_SPECIES, "country", COL_STATUS]).size().unstack(fill_value=0)
           .rename(columns={"Introduced": "non_native", "Native": "native"}).reset_index()
           .rename(columns={COL_SPECIES: "species"}))
    t = t.sort_values(["species", "native", "non_native"], ascending=[True, False, False])
    t.to_csv(TABLES / "r1_status_by_country.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
