"""
continent_audit.py
------------------
Continent audit of INVADED (Alien) records for the five focal invasive
crayfish species. Addresses Reviewer 2 (point 8) and the Methods statement
that F. virilis is "invasive in western North America AND Europe" while it
was classified as a within-continent invader.

PRIMARY  (post-filter): data/processed/<species>_invasive.csv
         = exactly the invaded records that entered the native-vs-invaded
           classifier, so the breakdown matches the records carrying the signal.
CONTEXT  (pre-filter):  all Status=="Alien" records for the 5 species in the
           raw master, to show how filtering redistributed records by continent.

Continent = offline reverse geocoding (nearest populated place -> country ->
continent) via reverse_geocoder (GeoNames). No GDAL. A longitude-based
Americas/Old-World cross-check is printed to catch gross errors and is used as
an automatic fallback if reverse_geocoder is unavailable.

Outputs:
  results/tables/continent_audit_records.csv   (one row per invaded record + continent)
  results/tables/continent_audit_summary.csv   (species x continent, post-filter)
"""
from pathlib import Path
import pandas as pd

MASTER = "data/raw/combined_data_true_master.csv"
PROCESSED = Path("data/processed")
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)

SPECIES_COL = "Crayfish_scientific_name"
LAT, LON, WOC = "lat_or", "long_or", "WoCID"

FOCAL = {
    "Procambarus clarkii":      ("procambarus_clarkii",      "intercontinental"),
    "Faxonius limosus":         ("faxonius_limosus",         "intercontinental"),
    "Pacifastacus leniusculus": ("pacifastacus_leniusculus", "intercontinental"),
    "Faxonius virilis":         ("faxonius_virilis",         "within-continent"),
    "Faxonius rusticus":        ("faxonius_rusticus",        "within-continent"),
}
CONT_NAME = {"NA":"North America","SA":"South America","EU":"Europe",
             "AS":"Asia","AF":"Africa","OC":"Oceania","AN":"Antarctica"}

def load_master():
    keep = {SPECIES_COL, "Status", LAT, LON, WOC}
    return pd.read_csv(MASTER, usecols=lambda c: c in keep, low_memory=False)

def invaded_records(species, stem, master):
    """post-filter invaded records + lat/lon.
    1: coords in processed file  2: WoCID -> merge master  3: fallback master Alien (pre-filter)."""
    f = PROCESSED / f"{stem}_invasive.csv"
    if f.exists():
        inv = pd.read_csv(f, low_memory=False)
        if {LAT, LON}.issubset(inv.columns):
            return inv[[c for c in (WOC, LAT, LON) if c in inv.columns]].copy(), "processed(has coords)"
        if WOC in inv.columns and WOC in master.columns:
            m = master[[WOC, LAT, LON]].drop_duplicates(WOC)
            return inv[[WOC]].merge(m, on=WOC, how="left"), "processed(WoCID)->master coords"
    sub = master[(master[SPECIES_COL] == species) & (master["Status"] == "Alien")]
    return sub[[c for c in (WOC, LAT, LON) if c in sub.columns]].copy(), "FALLBACK master Alien (PRE-filter!)"

def _heur(lat, lon):
    if lon < -30: return "North America" if lat >= 13 else "South America"
    if lon > 60:  return "Asia"
    return "Europe" if lat >= 35 else "Africa"

def geocode(df):
    d = df.dropna(subset=[LAT, LON]).copy()
    coords = list(zip(d[LAT].astype(float), d[LON].astype(float)))
    try:
        import reverse_geocoder as rg
        from pycountry_convert import country_alpha2_to_continent_code
        res = rg.search(coords, mode=1)
        conts = []
        for r in res:
            cc = (r.get("cc") or "").upper()
            try: conts.append(CONT_NAME.get(country_alpha2_to_continent_code(cc), "Unknown"))
            except Exception: conts.append("Unknown")
        d["continent"] = conts; method = "reverse_geocoder"
    except Exception as e:
        print(f"  [!] reverse_geocoder unavailable ({e}); longitude heuristic used.")
        d["continent"] = [_heur(la, lo) for la, lo in coords]; method = "longitude-heuristic"
    d["_am_geo"] = d["continent"].isin(["North America", "South America"])
    d["_am_lon"] = d[LON].astype(float) < -30
    return d, method

def main():
    master = load_master()
    parts = []
    print("=== Path used to obtain invaded records (per species) ===")
    for sp, (stem, pathway) in FOCAL.items():
        recs, path = invaded_records(sp, stem, master)
        recs["species"] = sp; recs["pathway_as_classified"] = pathway
        parts.append(recs)
        print(f"  {sp:<26} n={len(recs):>6}  via {path}")
    rec = pd.concat(parts, ignore_index=True)

    rec, gm = geocode(rec)
    print(f"\nContinent method: {gm}  |  rows with usable coords: {len(rec)}")
    n_bad = int((rec["_am_geo"] != rec["_am_lon"]).sum())
    print(f"Americas/Old-World cross-check mismatches: {n_bad} of {len(rec)}")
    if n_bad:
        print(rec.loc[rec["_am_geo"] != rec["_am_lon"], ["species", LAT, LON, "continent"]].head(10).to_string(index=False))

    summ = pd.crosstab(rec["species"], rec["continent"], margins=True, margins_name="TOTAL")
    print("\n===== POST-FILTER  species x continent  (records the classifier actually saw) =====")
    print(summ.to_string())

    msub = master.loc[master[SPECIES_COL].isin(list(FOCAL)), [SPECIES_COL, "Status", LAT, LON]]
    msub = msub[msub["Status"] == "Alien"].rename(columns={SPECIES_COL: "species"})
    msub, _ = geocode(msub)
    print("\n===== PRE-FILTER  species x continent  (all Alien in raw master, context) =====")
    print(pd.crosstab(msub["species"], msub["continent"], margins=True, margins_name="TOTAL").to_string())

    print("\n===== DECISIVE (post-filter) =====")
    for sp in ["Faxonius virilis", "Procambarus clarkii"]:
        vc = rec.loc[rec["species"] == sp, "continent"].value_counts()
        print(f"  {sp}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))

    rec.drop(columns=["_am_geo", "_am_lon"]).to_csv(OUTDIR / "continent_audit_records.csv", index=False)
    summ.to_csv(OUTDIR / "continent_audit_summary.csv")
    print(f"\nSaved: {OUTDIR/'continent_audit_records.csv'}\nSaved: {OUTDIR/'continent_audit_summary.csv'}")

if __name__ == "__main__":
    main()
