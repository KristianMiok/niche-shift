"""
background_matched_refit.py — Task (g): background-matched classifier (Reviewer 2 point 2).

Option A: restrict BOTH native and invaded records to their region of overlap in
environmental space (intersection of the two groups' convex hulls in the first k PCA
axes), then refit and measure grouped-permutation dAUC. If climate still dominates
INSIDE the shared envelope, the climate signal is not merely a continental background
contrast. Reported at k=5 AND k=10 (Lucian: report sensitivity, because the PCA basis is
fitted on pooled native+invaded data and is not a neutral coordinate system).

Convex-hull membership via halfspace equations (vectorized, any dimension), QJ joggle
against degeneracies, wrapped in try/except. If a species' overlap is too sparse
(<40 per class) or the hull fails, Option B is used for it.

Option B (fallback + independent point-2 check, run for all five): add continent as a
one-hot covariate and refit. Continent columns are never permuted (they sit outside the
four thematic domains), so they are always available to absorb the continental contrast;
if climate grouped-perm dAUC survives with continent present, climate carries information
beyond continent.

Internal control (Lucian): the within-continent species share a continent between native
and invaded ranges, so they are the built-in control — climate dominance there cannot be
a continental artefact.

Run:  python src/background_matched_refit.py
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame, domain_index_map, DOMAINS

warnings.filterwarnings("ignore")
INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, TREES, REPEATS = 42, 500, 10
K_LEVELS = [5, 10]
MIN_PER_CLASS = 40
LAT, LON = "lat_or", "long_or"
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
CONT_NAME = {"NA": "North America", "SA": "South America", "EU": "Europe",
             "AS": "Asia", "AF": "Africa", "OC": "Oceania", "AN": "Antarctica"}


def _in_hull_qhull(pts, hp):
    h = ConvexHull(hp, qhull_options="QJ")
    eq = h.equations
    return np.all(pts @ eq[:, :-1].T + eq[:, -1] <= 1e-9, axis=1)

def _in_hull_lp(pts, hp):
    """Point-in-convex-hull via LP feasibility (no hull construction -> no facet blow-up
    in high dimension). x in conv(hp) iff exists w>=0, sum(w)=1, hp.T@w = x. One LP/point."""
    from scipy.optimize import linprog
    n, d = hp.shape
    A_eq = np.vstack([hp.T, np.ones(n)])                 # (d+1) x n
    c = np.zeros(n)
    out = np.zeros(len(pts), dtype=bool)
    for i, x in enumerate(pts):
        b_eq = np.concatenate([x, [1.0]])
        r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        out[i] = r.success
    return out

def in_hull(pts, hull_pts, seed=SEED):
    """Which of `pts` lie inside conv(hull_pts). QHull halfspaces for low dimension
    (<=6, fast and exact); LP feasibility for higher dimension (QHull's facet count blows
    up in ~10D). Hull points are subsampled (QHull <=1500, LP <=400) since only extreme
    points define the boundary; all `pts` are still tested against it."""
    d = hull_pts.shape[1]
    rng = np.random.RandomState(seed)
    if d <= 6:
        hp = hull_pts if len(hull_pts) <= 1500 else hull_pts[rng.choice(len(hull_pts), 1500, replace=False)]
        return _in_hull_qhull(pts, hp)
    hp = hull_pts if len(hull_pts) <= 400 else hull_pts[rng.choice(len(hull_pts), 400, replace=False)]
    return _in_hull_lp(pts, hp)


def grouped_perm_dauc(X, y, clean, seed=SEED):
    """Domain-level grouped-permutation dAUC on 5-fold held-out (as in task e)."""
    if min(np.bincount(y)) < 5:
        return None, None
    dmap = domain_index_map(clean)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    model = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
    base, per = [], {d: [] for d in dmap}
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        b = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]); base.append(b)
        for d, cols in dmap.items():
            for _ in range(REPEATS):
                Xp = X[te].copy(); perm = rng.permutation(Xp.shape[0])
                Xp[:, cols] = Xp[np.ix_(perm, cols)]
                per[d].append(b - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
    return {d: float(np.mean(per[d])) for d in dmap}, float(np.mean(base))


def continents_for(df):
    import reverse_geocoder as rg
    from pycountry_convert import country_alpha2_to_continent_code
    coords = list(zip(df[LAT].astype(float), df[LON].astype(float)))
    out = []
    for r in rg.search(coords, mode=1):
        cc = (r.get("cc") or "").upper()
        try: out.append(CONT_NAME.get(country_alpha2_to_continent_code(cc), "Unknown"))
        except Exception: out.append("Unknown")
    return out


def option_a(combined, clean, status, k):
    """Restrict to convex-hull overlap in first k PCs; refit; grouped-perm dAUC."""
    Xs = np.nan_to_num(StandardScaler().fit_transform(combined[clean].values),
                       nan=0.0, posinf=0.0, neginf=0.0)
    Xs = Xs[:, Xs.std(axis=0) > 1e-8]
    P = PCA(n_components=k, svd_solver="full", random_state=SEED).fit_transform(Xs)
    nat, inv = P[status == 0], P[status == 1]
    try:
        t0 = time.time()
        overlap = in_hull(P, nat) & in_hull(P, inv)
        dt = time.time() - t0
    except (MemoryError, Exception) as e:
        return {"k": k, "status": f"hull_failed:{type(e).__name__}"}
    yo = status[overlap]
    n_nat, n_inv = int((yo == 0).sum()), int((yo == 1).sum())
    res = {"k": k, "hull_seconds": round(dt, 1), "n_native": n_nat, "n_invaded": n_inv}
    if min(n_nat, n_inv) < MIN_PER_CLASS:
        res["status"] = "too_sparse"; return res
    Xo = combined[clean].values[overlap]
    gp, auc = grouped_perm_dauc(Xo, yo, clean)
    res.update({"status": "ok", "auc": auc, "grouped_perm_dauc": gp})
    return res


def option_b(combined, clean, status):
    """Add continent one-hot covariates (never permuted); grouped-perm dAUC of climate."""
    conts = continents_for(combined)
    dummies = pd.get_dummies(pd.Series(conts, name="cont"), prefix="CONT")
    X = np.hstack([combined[clean].values, dummies.values.astype(float)])
    clean_ext = list(clean) + list(dummies.columns)  # CONT_* match no domain -> never permuted
    gp, auc = grouped_perm_dauc(X, status, clean_ext)
    return {"auc": auc, "grouped_perm_dauc": gp, "continent_cols": list(dummies.columns)}


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    summary = {}

    for sp in SPECIES:
        pw = "intercontinental" if sp in INTERCONTINENTAL else "within-continent"
        print("\n" + "=" * 96)
        print(f"{sp}  [{pw}]  —  background-matched refit (point 2)")
        print("=" * 96)
        combined, clean = build_species_frame(dff, sp, env)
        status = (combined["range_label"] == "invasive").astype(int).values

        # full-set climate/topography dAUC for reference
        gp_full, auc_full = grouped_perm_dauc(combined[clean].values, status, clean)
        print(f"  FULL set (n_nat={int((status==0).sum())}, n_inv={int((status==1).sum())}, AUC={auc_full:.3f})")
        print(f"    climate dAUC={gp_full['Climate']:.4f}   topography dAUC={gp_full['Topography']:.4f}")

        # Option A at k=5 and k=10
        a_res = {}
        for k in K_LEVELS:
            r = option_a(combined, clean, status, k)
            a_res[f"k{k}"] = r
            if r["status"] == "ok":
                print(f"  Option A k={k}: overlap nat={r['n_native']} inv={r['n_invaded']} "
                      f"(hull {r['hull_seconds']}s) AUC={r['auc']:.3f}  "
                      f"climate dAUC={r['grouped_perm_dauc']['Climate']:.4f}  "
                      f"topo dAUC={r['grouped_perm_dauc']['Topography']:.4f}")
            else:
                print(f"  Option A k={k}: {r['status']}"
                      + (f" (nat={r.get('n_native')} inv={r.get('n_invaded')})" if 'n_native' in r else ""))

        # Option B (continent covariate) — all species
        b = option_b(combined, clean, status)
        print(f"  Option B (continent covariate): AUC={b['auc']:.3f}  "
              f"climate dAUC={b['grouped_perm_dauc']['Climate']:.4f}  "
              f"topo dAUC={b['grouped_perm_dauc']['Topography']:.4f}")

        summary[sp] = {"pathway": pw, "full": {"auc": auc_full, "grouped_perm_dauc": gp_full},
                       "option_a": a_res, "option_b": b}

        # verdict line for climate
        cf = gp_full["Climate"]
        parts = [f"full={cf:.4f}"]
        for k in K_LEVELS:
            r = a_res[f"k{k}"]
            if r["status"] == "ok":
                parts.append(f"A_k{k}={r['grouped_perm_dauc']['Climate']:.4f}")
        parts.append(f"B={b['grouped_perm_dauc']['Climate']:.4f}")
        print(f"  >>> CLIMATE dAUC:  " + "  ".join(parts))

    (OUTDIR / "background_matched_refit_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved: {OUTDIR/'background_matched_refit_summary.json'}")
    print("\n>>> Read: does climate dAUC survive inside the shared envelope (Option A k=5 and k=10)")
    print(">>> and with continent as a covariate (Option B)? If yes, climate is not merely a")
    print(">>> continental background contrast -> point 2 addressed. Within-continent species are")
    print(">>> the internal control (climate there cannot be a continental artefact).")


if __name__ == "__main__":
    main()
