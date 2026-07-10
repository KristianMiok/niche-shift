"""
spatial_block_cv.py — Task (b): spatially blocked cross-validation.

Reviewer 2 point 5: spatial autocorrelation / geographic separation may inflate
classification performance. For the primary native-vs-invaded classifier (all five
species) we replace random k-fold with spatial block CV: occurrences are assigned to
grid cells of side B km, and folds are formed at the CELL level via StratifiedGroupKFold
(keeps every cell intact within one fold while balancing the native/invaded ratio).
No cell straddles train and test, so nearby, environmentally similar records cannot leak.

We sweep B in {50,100,250,500} km (do not defend one size; show conclusions are stable).
Per species and block size: blocked accuracy, blocked AUC, and domain-level grouped
permutation ΔAUC (Climate/Topography/Soil/Land Cover, collinearity-immune, as in task e)
recomputed under blocking, plus the pathway contrast. A random-CV baseline is computed
in the same run for a like-for-like comparison.

Grid: latitude-corrected equirectangular. dlat = B/111; per band dlon = B/(111*cos(lat)).
No projection library, no GDAL.

Expectation (Lucian): blocking bites hardest on F. virilis and F. rusticus (native and
invaded interdigitate); ocean-separated intercontinental species barely move. A drop
there is the honest, interesting result — not a failure.

Run:  python src/spatial_block_cv.py --species "Faxonius virilis"   # smoke test
      python src/spatial_block_cv.py                                  # all five
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import (
    load_geotraits, load_config, get_env_variables, COL_SPECIES, COL_STATUS,
)
from species_selector import apply_quality_filters, STUDY_SPECIES
from grouped_permutation import build_species_frame, domain_index_map, DOMAINS

INPUT = "data/raw/combined_data_true_master.csv"
CONFIG = "config/species_config.yaml"
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
LAT, LON = "lat_or", "long_or"
BLOCKS_KM = [50, 100, 250, 500]
PATHWAY = {
    "Procambarus clarkii": "intercontinental",
    "Faxonius limosus": "intercontinental",
    "Pacifastacus leniusculus": "intercontinental",
    "Faxonius virilis": "within-continent",
    "Faxonius rusticus": "within-continent",
}


def assign_cells(lat, lon, block_km):
    """Latitude-corrected equirectangular grid -> integer cell code per record."""
    dlat = block_km / 111.0
    lat_band = np.floor(lat / dlat).astype(np.int64)
    band_center = (lat_band + 0.5) * dlat
    coslat = np.clip(np.cos(np.radians(band_center)), 0.05, None)
    dlon = block_km / (111.0 * coslat)
    lon_band = np.floor(lon / dlon).astype(np.int64)
    # Bijective integer pairing of the two bands (shift to non-negative first),
    # then factorize the resulting 1-D integer array -> contiguous cell codes.
    a = lat_band - lat_band.min()
    b = lon_band - lon_band.min()
    paired = a * (b.max() + 1) + b
    codes, _ = pd.factorize(paired)
    return codes


def _auc(y_true, proba):
    return np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, proba)


def eval_split(model, X, y, clean, splitter, groups, n_repeats, seed):
    """Fold loop: acc + AUC on held-out, and per-domain block-permutation ΔAUC."""
    dmap = domain_index_map(clean)
    rng = np.random.RandomState(seed)
    accs, aucs, n_single, n_folds = [], [], 0, 0
    dom = {d: [] for d in dmap}
    splits = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue  # cannot train a two-class model on this fold
        n_folds += 1
        m = clone(model).fit(X[tr], y[tr])
        Xte = X[te]
        accs.append(accuracy_score(y[te], m.predict(Xte)))
        base = _auc(y[te], m.predict_proba(Xte)[:, 1])
        if np.isnan(base):
            n_single += 1
            continue  # AUC + grouped-perm undefined on single-class held-out
        aucs.append(base)
        for d, cols in dmap.items():
            for _ in range(n_repeats):
                Xp = Xte.copy()
                perm = rng.permutation(Xp.shape[0])   # one shared shuffle
                Xp[:, cols] = Xp[np.ix_(perm, cols)]  # whole domain moves together
                pa = _auc(y[te], m.predict_proba(Xp)[:, 1])
                if not np.isnan(pa):
                    dom[d].append(base - pa)
    out = {
        "acc": float(np.mean(accs)) if accs else np.nan,
        "auc": float(np.mean(aucs)) if aucs else np.nan,
        "n_folds": int(n_folds),
        "n_single_class_folds": int(n_single),
    }
    for d in dmap:
        out[d] = float(np.mean(dom[d])) if dom[d] else np.nan
    return out


def _fmt(v, w=9, p=3):
    return f"{'  --':>{w}}" if (isinstance(v, float) and np.isnan(v)) else f"{v:>{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--species", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=500)
    args = ap.parse_args()

    config = load_config(CONFIG)
    df = load_geotraits(INPUT)
    dff = apply_quality_filters(df, config)
    env = get_env_variables(dff)["all_env"]
    species_list = [args.species] if args.species else STUDY_SPECIES
    model = RandomForestClassifier(n_estimators=args.trees, class_weight="balanced",
                                   random_state=args.seed, n_jobs=-1)
    summary = {}

    for sp in species_list:
        print("\n" + "=" * 96)
        print(f"{sp}  [{PATHWAY[sp]}]  —  spatial block CV")
        print("=" * 96)
        combined, clean = build_species_frame(dff, sp, env)
        assert LAT in combined.columns and LON in combined.columns, "coordinates lost in pipeline"

        lat = combined[LAT].astype(float).values
        lon = combined[LON].astype(float).values
        ok = ~(np.isnan(lat) | np.isnan(lon))
        X = combined[clean].values[ok]
        y = (combined["range_label"] == "invasive").astype(int).values[ok]
        lat, lon = lat[ok], lon[ok]
        dmap = domain_index_map(clean)
        dropped = int((~ok).sum())
        print(f"  n={len(y)} ({(y==0).sum()} nat / {(y==1).sum()} inv)"
              + (f"  [dropped {dropped} rows w/o coords]" if dropped else "")
              + f" | {len(clean)} vars | domain vars: "
              + ", ".join(f"{d.split()[0][:4]}={len(dmap.get(d, []))}" for d in DOMAINS))

        rows = {"random": eval_split(
            model, X, y, clean,
            StratifiedKFold(5, shuffle=True, random_state=args.seed),
            None, args.repeats, args.seed)}
        for B in BLOCKS_KM:
            cells = assign_cells(lat, lon, B)
            ncell = int(len(np.unique(cells)))
            if ncell < 2:
                rows[f"{B}km"] = {"acc": np.nan, "auc": np.nan, "n_cells": ncell, "note": "too few cells"}
                continue
            k = min(5, ncell)
            try:
                res = eval_split(
                    model, X, y, clean,
                    StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=args.seed),
                    cells, args.repeats, args.seed)
                res["n_cells"] = ncell
            except Exception as e:
                res = {"acc": np.nan, "auc": np.nan, "n_cells": ncell, "note": type(e).__name__}
            rows[f"{B}km"] = res

        hdr = (f"  {'CV':<9}{'cells':>7}{'folds':>7}{'acc':>9}{'AUC':>9}"
               + "".join(f"{d.split()[0][:5]:>9}" for d in DOMAINS))
        print("\n" + hdr)
        for key in ["random"] + [f"{B}km" for B in BLOCKS_KM]:
            r = rows[key]
            line = (f"  {key:<9}{str(r.get('n_cells','-')):>7}{str(r.get('n_folds','-')):>7}"
                    f"{_fmt(r.get('acc', np.nan))}{_fmt(r.get('auc', np.nan))}"
                    + "".join(_fmt(r.get(d, np.nan), 9, 4) for d in DOMAINS))
            if r.get("note"):
                line += f"   [{r['note']}]"
            print(line)
        for B in BLOCKS_KM:
            nsc = rows[f"{B}km"].get("n_single_class_folds", 0)
            if nsc:
                print(f"    note {B}km: {nsc} single-class fold(s) -> AUC excluded there")

        print("\n  >>> Topography ΔAUC   random " + _fmt(rows['random'].get('Topography', np.nan), 7, 4)
              + "  |  " + "  ".join(f"{B}km" + _fmt(rows[f'{B}km'].get('Topography', np.nan), 7, 4) for B in BLOCKS_KM))
        print("  >>> AUC               random " + _fmt(rows['random'].get('auc', np.nan), 7)
              + "  |  " + "  ".join(f"{B}km" + _fmt(rows[f'{B}km'].get('auc', np.nan), 7) for B in BLOCKS_KM))
        summary[sp] = {"pathway": PATHWAY[sp], "rows": rows}

    (OUTDIR / "spatial_block_cv_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved: {OUTDIR/'spatial_block_cv_summary.json'}")

    if len(species_list) > 1:
        for metric, key, p in [("Topography ΔAUC (does the contrast survive blocking?)", "Topography", 4),
                               ("AUC (random -> blocked; performance inflation from autocorrelation)", "auc", 3)]:
            print("\n" + "=" * 96)
            print("CROSS-SPECIES  " + metric)
            print("=" * 96)
            print(f"{'species':<26}{'pathway':<18}{'random':>9}" + "".join(f"{str(B)+'km':>9}" for B in BLOCKS_KM))
            for sp in species_list:
                r = summary[sp]["rows"]
                vals = [r["random"].get(key, np.nan)] + [r[f"{B}km"].get(key, np.nan) for B in BLOCKS_KM]
                print(f"{sp:<26}{PATHWAY[sp]:<18}" + "".join(_fmt(v, 9, p) for v in vals))


if __name__ == "__main__":
    main()
