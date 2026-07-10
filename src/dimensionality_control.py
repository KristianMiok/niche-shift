"""
dimensionality_control.py — Task (f): 2x2 dimensionality control (Reviewer 2 point 7).

Honest resolution: classical D/I have NO numerically stable high-dimensional form.
In high dimension the native and invaded record clouds become near-disjoint (the same
fact that gives RF AUC ~0.99), so Schoener's D under any density estimator collapses to
a degenerate floor. This is curse-of-dimensionality, inherent to scalar overlap, not a
bug. We therefore report the DEFENSIBLE cells and DOCUMENT the collapse as a trend rather
than forcing a single unreliable high-dim number.

Reported:
  var_2pc, n_pc for 80% variance                     -> answers "modest fraction / how many"
  D,I at 2 PC (original grid method, reliable)        -> reproduces the manuscript metric
  RF AUC on 2 PC vs on ~400 vars (top-right cell)     -> ML separates even at 2 PC
  D,I vs number of PCs (k-NN log-density trend)       -> "do overlap values change with dim?"
  pathway separation of D,I at 2 PC (exact 3/2)       -> D/I DO separate (validates retraction)

Self-check: k-NN D at 2 PC is printed next to grid D at 2 PC; if they disagree grossly the
k-NN trend is untrustworthy and we fall back to the 2-PC + RF-2PC + variance cells only.

Run:  python src/dimensionality_control.py
"""
import json, warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame
from niche_overlap_metrics import estimate_density_grid, schoeners_d, warrens_i

warnings.filterwarnings("ignore", category=UserWarning)
INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, TREES, VAR_TARGET = 42, 500, 0.80
PC_GRID = [2, 3, 5, 8, 13, 21, 34]
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
OBSERVED_INTER = frozenset(INTERCONTINENTAL)


def prep(dff, sp, env):
    combined, clean = build_species_frame(dff, sp, env)
    status = (combined["range_label"] == "invasive").astype(int).values
    X = combined[clean].values
    Xs = np.nan_to_num(StandardScaler().fit_transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    Xs = Xs[:, Xs.std(axis=0) > 1e-8]  # drop constant cols -> stabilises SVD
    return X, Xs, status


def pcs(Xs, n):
    return PCA(n_components=n, svd_solver="full", random_state=SEED).fit_transform(Xs)


def di_grid_2d(P2, status):
    nat, inv = P2[status == 0], P2[status == 1]
    c = np.vstack([nat, inv])
    xe = np.linspace(c[:, 0].min(), c[:, 0].max(), 101)
    ye = np.linspace(c[:, 1].min(), c[:, 1].max(), 101)
    ng = estimate_density_grid(nat[:, 0], nat[:, 1], xe, ye, sigma=1.0)
    ig = estimate_density_grid(inv[:, 0], inv[:, 1], xe, ye, sigma=1.0)
    return schoeners_d(ng, ig), warrens_i(ng, ig)


def _logdens(sample, data, k=15):
    d = data.shape[1]
    nn = NearestNeighbors(n_neighbors=min(k, len(data))).fit(data)
    dist, _ = nn.kneighbors(sample)
    r = np.where(dist[:, -1] <= 0, np.finfo(float).tiny, dist[:, -1])
    return -d * np.log(r) - np.log(len(data))


def di_knn(P, status, k=15, n_mc=4000):
    rng = np.random.RandomState(SEED)
    nat, inv = P[status == 0], P[status == 1]
    pool = np.vstack([nat, inv])
    S = pool[rng.choice(len(pool), min(n_mc, len(pool)), replace=False)]
    def norm(l):
        w = np.exp(l - l.max()); return w / w.sum()
    pa, pb = norm(_logdens(S, nat, k)), norm(_logdens(S, inv, k))
    return float(1 - 0.5 * np.abs(pa - pb).sum()), float(np.sqrt(pa * pb).sum())


def sep(vals):
    def gd(inter):
        return np.mean([vals[s] for s in inter]) - np.mean([vals[s] for s in SPECIES if s not in inter])
    allv = [gd(frozenset(c)) for c in combinations(SPECIES, 3)]
    obs = gd(OBSERVED_INTER)
    rank = int(np.sum(np.array(allv) >= obs))
    return float(obs), rank, float(rank / len(allv))


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]

    rows, D2, I2, trend = [], {}, {}, {}
    print("\n" + "=" * 100)
    print("2x2 DIMENSIONALITY CONTROL (point 7)   [D_hi omitted: no stable high-dim form]")
    print("=" * 100)
    print(f"  {'species':<26}{'n_pc80':>7}{'var2%':>7}{'var80%':>7}"
          f"{'D_2pc':>8}{'I_2pc':>8}{'D_2pc_knn':>10}{'RF400_AUC':>10}{'RF2pc_AUC':>10}")

    for sp in SPECIES:
        X, Xs, status = prep(dff, sp, env)
        cum = np.cumsum(PCA(svd_solver="full", random_state=SEED).fit(Xs).explained_variance_ratio_)
        n80 = int(np.searchsorted(cum, VAR_TARGET) + 1)
        var2 = float(cum[1])
        P2 = pcs(Xs, 2)
        d2, i2 = di_grid_2d(P2, status)
        d2k, _ = di_knn(P2, status)                 # self-check vs grid
        D2[sp], I2[sp] = d2, i2

        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        rf = lambda: RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1)
        u400 = cross_validate(rf(), X, status, cv=cv, scoring=["roc_auc"])["test_roc_auc"].mean()
        u2 = cross_validate(rf(), P2, status, cv=cv, scoring=["roc_auc"])["test_roc_auc"].mean()

        # D/I vs dimension trend (k-NN)
        tr = {}
        for kpc in PC_GRID:
            if kpc <= Xs.shape[1]:
                dk, ik = di_knn(pcs(Xs, kpc), status)
                tr[kpc] = {"D": dk, "I": ik}
        trend[sp] = tr

        print(f"  {sp:<26}{n80:>7}{var2*100:>6.1f}%{cum[n80-1]*100:>6.1f}%"
              f"{d2:>8.3f}{i2:>8.3f}{d2k:>10.3f}{u400:>10.3f}{u2:>10.3f}")
        rows.append({"species": sp, "pathway": "intercontinental" if sp in INTERCONTINENTAL else "within-continent",
                     "n_pc_80pct": n80, "var_2pc": var2, "var_80pct": float(cum[n80-1]),
                     "D_2pc_grid": d2, "I_2pc_grid": i2, "D_2pc_knn_check": d2k,
                     "RF_400var_auc": float(u400), "RF_2pc_auc": float(u2)})

    print("\n" + "-" * 100)
    print("D/I vs number of PCs (k-NN log-density trend) — does overlap change with dimension?")
    print("-" * 100)
    print(f"  {'species':<26}" + "".join(f"{'D@'+str(k):>8}" for k in PC_GRID))
    for sp in SPECIES:
        print(f"  {sp:<26}" + "".join(
            f"{trend[sp].get(k, {}).get('D', float('nan')):>8.3f}" for k in PC_GRID))

    print("\n" + "-" * 100)
    print("PATHWAY SEPARATION by D/I at 2 PC (exact 3/2; rank 1/10 => most extreme)")
    print("-" * 100)
    sepres = {}
    for name, vals in [("Schoener D (2 PC)", D2), ("Warren I (2 PC)", I2)]:
        obs, rank, p = sep(vals)
        sepres[name] = {"obs_group_diff": obs, "rank": rank, "p": p}
        tag = "SEPARATES (validates retracting 'D/I fail')" if rank == 1 else "does not separate"
        print(f"  {name:<22}{obs:>10.4f}  rank {rank}/10  P={p:.3f}   {tag}")

    (OUTDIR / "dimensionality_control_summary.json").write_text(
        json.dumps({"per_species": rows, "di_vs_dim_trend": trend, "pathway_separation_2pc": sepres},
                   indent=2, default=float))
    pd.DataFrame(rows).to_csv(OUTDIR / "dimensionality_control.csv", index=False)
    print(f"\nSaved: {OUTDIR/'dimensionality_control.csv'}")
    print(f"Saved: {OUTDIR/'dimensionality_control_summary.json'}")
    print("\n>>> Check: is 'D_2pc_knn' close to 'D_2pc'? If yes, the k-NN trend is trustworthy")
    print(">>> and the D@k row documents the honest collapse toward 0 as dimension rises.")


if __name__ == "__main__":
    main()
