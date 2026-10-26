"""
r1_separate_models_audit.py — R1 items KM-6 (threshold shift: formula and uncertainty)
and the background part of KM-7 (geographic limits of the pseudo-absences).

What the submitted separate niche models do (separate_niche_models.py)
  presence   : records of the focal species in one range (native or non-native)
  background : a 1:1 random sample of records of all OTHER species that carry the same
               status (Native, or Introduced), drawn from the whole GeoTraits table,
               with no geographic restriction (random_state=42)
  model      : CART, depth 4, class_weight="balanced", random_state=42, variables
               re-cleaned on presence + background
  shift      : for variables among the 10 highest-impurity-decrease splits of BOTH the
               native and the non-native tree (if a variable splits more than once, its
               split with the largest decrease is used):
                   shift_pct = (t_nonnative - t_native) / |t_native| x 100

Part 1 - where the background comes from: share of background records farther than
500 / 1000 km from every presence record of that range, and their continents.

Part 2 - stability: 200 replicates, each with a bootstrap resample of the presence
records, a fresh background draw and a different tree seed. For every variable we
record how often it is shared between the native and non-native tree and the
distribution of its shift_pct (median, 2.5-97.5 % interval, share negative).
Run twice: with the submitted background ("status"), and with a geographically
constrained background ("geo100": other species' records, any status, within 100 km
of a presence record of that range).

Outputs (results/revision/tables/)
  r1_background_geography.csv
  r1_threshold_stability_replicates.csv
  r1_threshold_stability_summary.csv
  r1_separate_models_geo100.csv          Part 3: canonical refit with the 100 km background
  r1_separate_models_geo100_shared.csv   (drop-in replacement for Table S4) and its shared splits

Run (from conservation/):  ../.venv/bin/python src/r1_separate_models_audit.py
"""
from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
import pandas as pd
import pycountry_convert as pc
import reverse_geocoder as rg
from sklearn.neighbors import BallTree
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from data_loader import COL_SPECIES, COL_STATUS
from data_preparation import handle_missing_values, remove_constant_variables, remove_highly_correlated
from r1_common import DOMAINS, LAT, LON, SPECIES, TABLES, domain_shares, ensure_dirs, env_columns, load_filtered
from separate_niche_models import compare_thresholds, extract_top_splits
from species_selector import STATUS_ALIEN, STATUS_NATIVE

N_REP = 200
EARTH_KM = 6371.0
SUBMITTED = {  # shared variables reported in the submitted Section 3.4
    "Pontastacus leptodactylus": {"l_TOP102": -11.5, "l_CLI14": -23.2},
}


def nearest_km(lat, lon, ref_lat, ref_lon):
    tree = BallTree(np.radians(np.c_[ref_lat, ref_lon]), metric="haversine")
    d, _ = tree.query(np.radians(np.c_[lat, lon]), k=1)
    return d[:, 0] * EARTH_KM


def continent(cc: str) -> str:
    try:
        return pc.convert_continent_code_to_continent_name(pc.country_alpha2_to_continent_code(cc))
    except Exception:
        return "unknown"


def background_pool(dff, species, status, presence, mode):
    others = dff[dff[COL_SPECIES] != species]
    if mode == "status":
        return others[others[COL_STATUS] == status]
    # geo100: other species, any status, within 100 km of a presence record
    d = nearest_km(others[LAT].to_numpy(), others[LON].to_numpy(),
                   presence[LAT].to_numpy(), presence[LON].to_numpy())
    return others[d <= 100]


def fit_tree(presence, background, env, seed):
    p = presence.copy(); p["_t"] = 1
    b = background.copy(); b["_t"] = 0
    comb = pd.concat([p, b], ignore_index=True)
    with contextlib.redirect_stdout(io.StringIO()):
        comb, cols = handle_missing_values(comb, env, max_missing_pct=0.3)
        cols = remove_constant_variables(comb, cols)
        cols = remove_highly_correlated(comb, cols, threshold=0.98)
    clf = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=seed)
    clf.fit(comb[cols].to_numpy(), comb["_t"].to_numpy())
    return clf, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--reps", type=int, default=N_REP)
    args = ap.parse_args()
    ensure_dirs()
    dff = load_filtered(args.cache)
    env = env_columns(dff)

    # ---------------- Part 1: geography of the submitted background ----------------
    geo_rows = []
    for sp in SPECIES:
        for status, rng in ((STATUS_NATIVE, "native"), (STATUS_ALIEN, "non-native")):
            pres = dff[(dff[COL_SPECIES] == sp) & (dff[COL_STATUS] == status)]
            pool = background_pool(dff, sp, status, pres, "status")
            bg = pool.sample(n=min(len(pres), len(pool)), random_state=42, replace=False)
            d = nearest_km(bg[LAT].to_numpy(), bg[LON].to_numpy(), pres[LAT].to_numpy(), pres[LON].to_numpy())
            ccs = [r["cc"] for r in rg.search(list(zip(bg[LAT], bg[LON])), mode=1, verbose=False)]
            cont = pd.Series([continent(c) for c in ccs]).value_counts(normalize=True)
            geo_rows.append({
                "species": sp, "range": rng, "n_presence": len(pres), "n_background": len(bg),
                "background_pool_size": len(pool), "background_n_species": bg[COL_SPECIES].nunique(),
                "share_bg_over_500km": float((d > 500).mean()), "share_bg_over_1000km": float((d > 1000).mean()),
                "median_bg_distance_km": float(np.median(d)),
                "bg_continents": "; ".join(f"{k} {100 * v:.1f}%" for k, v in cont.items()),
            })
            print(f"{sp:<30} {rng:<10} bg {len(bg)}: >1000 km {100 * (d > 1000).mean():.1f}%  {geo_rows[-1]['bg_continents']}")
    pd.DataFrame(geo_rows).to_csv(TABLES / "r1_background_geography.csv", index=False)

    # ---------------- Part 2: replicate stability of the threshold shifts ----------------
    rep_rows = []
    for mode in ("status", "geo100"):
        for sp in SPECIES:
            sp_df = dff[dff[COL_SPECIES] == sp]
            pres = {s: sp_df[sp_df[COL_STATUS] == s] for s in (STATUS_NATIVE, STATUS_ALIEN)}
            pools = {s: background_pool(dff, sp, s, pres[s], mode) for s in pres}
            for r in range(args.reps):
                rs = np.random.RandomState(1000 + r)
                splits = {}
                for s in pres:
                    p = pres[s].sample(n=len(pres[s]), replace=True, random_state=rs)
                    b = pools[s].sample(n=min(len(pres[s]), len(pools[s])), replace=False, random_state=rs)
                    clf, cols = fit_tree(p, b, env, seed=1000 + r)
                    splits[s] = (extract_top_splits(clf, cols, n_top=10), cols[clf.tree_.feature[0]])
                comps = compare_thresholds(splits[STATUS_NATIVE][0], splits[STATUS_ALIEN][0])
                shared = [c for c in comps if c["in_both"]]
                base = {"mode": mode, "species": sp, "replicate": r,
                        "root_native": splits[STATUS_NATIVE][1], "root_nonnative": splits[STATUS_ALIEN][1],
                        "n_shared": len(shared)}
                if not shared:
                    rep_rows.append({**base, "variable": None, "shift_pct": np.nan})
                for c in shared:
                    rep_rows.append({**base, "variable": c["variable"], "shift_pct": c["shift_pct"],
                                     "native_threshold": c["native_threshold"],
                                     "nonnative_threshold": c["invasive_threshold"]})
            reps = pd.DataFrame([x for x in rep_rows if x["mode"] == mode and x["species"] == sp])
            print(f"[{mode}] {sp}: >=1 shared variable in {100 * (reps.groupby('replicate').n_shared.first() > 0).mean():.0f}% "
                  f"of replicates; most often shared: {reps.variable.value_counts().head(3).to_dict()}")

    reps = pd.DataFrame(rep_rows)
    reps.to_csv(TABLES / "r1_threshold_stability_replicates.csv", index=False)

    summ = []
    for (mode, sp), g in reps.groupby(["mode", "species"]):
        nrep = g.replicate.nunique()
        any_shared = float((g.groupby("replicate").n_shared.first() > 0).mean())
        roots_nat = g.groupby("replicate").root_native.first().value_counts(normalize=True)
        roots_non = g.groupby("replicate").root_nonnative.first().value_counts(normalize=True)
        variables = set(g.variable.dropna()) | set(SUBMITTED.get(sp, {}))
        for v in variables:
            s = g[g.variable == v].shift_pct.dropna()
            summ.append({
                "mode": mode, "species": sp, "variable": v, "replicates": nrep,
                "share_replicates_any_shared": any_shared,
                "share_replicates_shared": len(s) / nrep,
                "shift_pct_median": float(s.median()) if len(s) else np.nan,
                "shift_pct_q025": float(s.quantile(0.025)) if len(s) else np.nan,
                "shift_pct_q975": float(s.quantile(0.975)) if len(s) else np.nan,
                "share_negative": float((s < 0).mean()) if len(s) else np.nan,
                "submitted_shift_pct": SUBMITTED.get(sp, {}).get(v, np.nan),
                "root_native_mode": f"{roots_nat.index[0]} ({100 * roots_nat.iloc[0]:.0f}%)",
                "root_nonnative_mode": f"{roots_non.index[0]} ({100 * roots_non.iloc[0]:.0f}%)",
            })
    out = pd.DataFrame(summ).sort_values(["mode", "species", "share_replicates_shared"], ascending=[True, True, False])
    out.to_csv(TABLES / "r1_threshold_stability_summary.csv", index=False)

    # ---------------- Part 3: one canonical refit with the geographic background ----------------
    # Same settings as the submitted models (1:1, depth 4, seed 42) but background drawn from
    # other species' records within 100 km of the presences: a drop-in replacement for Table S4.
    can_rows, shared_rows = [], []
    for sp in SPECIES:
        sp_df = dff[dff[COL_SPECIES] == sp]
        splits = {}
        for status, rng in ((STATUS_NATIVE, "native"), (STATUS_ALIEN, "non-native")):
            pres = sp_df[sp_df[COL_STATUS] == status]
            pool = background_pool(dff, sp, status, pres, "geo100")
            bg = pool.sample(n=min(len(pres), len(pool)), random_state=42, replace=False)
            clf, cols = fit_tree(pres, bg, env, seed=42)
            p = pres.copy(); p["_t"] = 1
            b = bg.copy(); b["_t"] = 0
            comb = pd.concat([p, b], ignore_index=True)
            with contextlib.redirect_stdout(io.StringIO()):
                comb, _ = handle_missing_values(comb, env, max_missing_pct=0.3)
            X, y = comb[cols].to_numpy(), comb["_t"].to_numpy()
            cv = cross_validate(DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42), X, y,
                                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring=["accuracy", "balanced_accuracy", "roc_auc"])
            imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
            shares = domain_shares(clf.feature_importances_, cols)
            top5 = imp.head(5)
            can_rows.append({
                "species": sp, "range": rng, "n_presence": len(pres), "n_background": len(bg),
                "background_pool_size": len(pool), "background_n_species": bg[COL_SPECIES].nunique(),
                "cv_accuracy": cv["test_accuracy"].mean(), "cv_balanced_accuracy": cv["test_balanced_accuracy"].mean(),
                "cv_auc": cv["test_roc_auc"].mean(),
                **{f"share_{d}": shares[d] for d in DOMAINS},
                "top5": "; ".join(f"{v} ({val:.3f})" for v, val in top5.items()),
            })
            splits[status] = extract_top_splits(clf, cols, n_top=10)
        for c in compare_thresholds(splits[STATUS_NATIVE], splits[STATUS_ALIEN]):
            if c["in_both"]:
                shared_rows.append({"species": sp, "variable": c["variable"], "native_threshold": c["native_threshold"],
                                    "nonnative_threshold": c["invasive_threshold"], "shift_pct": c["shift_pct"]})
    pd.DataFrame(can_rows).to_csv(TABLES / "r1_separate_models_geo100.csv", index=False)
    pd.DataFrame(shared_rows, columns=["species", "variable", "native_threshold", "nonnative_threshold", "shift_pct"]) \
        .to_csv(TABLES / "r1_separate_models_geo100_shared.csv", index=False)
    print(f"Saved r1_background_geography.csv, r1_threshold_stability_*.csv, r1_separate_models_geo100*.csv to {TABLES}")


if __name__ == "__main__":
    main()
