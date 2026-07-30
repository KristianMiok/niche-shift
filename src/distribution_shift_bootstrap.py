"""
distribution_shift_bootstrap.py — final version of the model-free distribution-shift
analysis, matching what the Methods and the response letter claim.

Two corrections to the earlier run:
  1. Variables whose NATIVE interquartile range is zero are now EXCLUDED, not given a
     fallback denominator. The earlier script substituted the native SD (and then 1.0),
     which silently kept undefined shifts in the averages.
  2. The bootstrap is over VARIABLES, resampled within domain, giving a CI on the
     topography-to-climate ratio of mean absolute standardised shift. (The earlier
     bootstrap file resampled CV folds of the grouped-permutation dAUC ratio — a
     different quantity, correct in itself but not this one.)

Shift per variable: (median_invaded - median_native) / IQR_native.
Species summary: mean |shift| within each domain; ratio = topography / climate.
Sensitivity: same ratio restricted to variables whose bootstrap CI excludes zero.
Pathway test: exact enumeration of all ten C(5,3) partitions on the species ratios.

Run:  python src/distribution_shift_bootstrap.py
"""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame
from cross_species_comparison import classify_variable

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
N_BOOT, SEED = 2000, 42
OUT = Path("results/tables"); OUT.mkdir(parents=True, exist_ok=True)
INTER = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN = ["Faxonius virilis", "Faxonius rusticus"]
SP = INTER + WITHIN
OBS = frozenset(INTER)


def shift(nat, inv):
    """Standardised shift, or None when the native IQR is zero (variable excluded)."""
    iqr = np.subtract(*np.percentile(nat, [75, 25]))
    if iqr <= 0:
        return None
    return (np.median(inv) - np.median(nat)) / iqr


def var_ci(nat, inv, rng):
    vals = []
    for _ in range(500):
        s = shift(rng.choice(nat, len(nat)), rng.choice(inv, len(inv)))
        if s is not None:
            vals.append(s)
    if len(vals) < 50:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ratio_from(topo, clim):
    if len(topo) == 0 or len(clim) == 0:
        return np.nan
    c = np.mean(np.abs(clim))
    return np.mean(np.abs(topo)) / c if c > 1e-12 else np.nan


def boot_ratio(topo, clim, rng, n=N_BOOT):
    topo, clim = np.asarray(topo), np.asarray(clim)
    out = []
    for _ in range(n):
        t = rng.choice(topo, len(topo), replace=True)
        c = rng.choice(clim, len(clim), replace=True)
        r = ratio_from(t, c)
        if np.isfinite(r):
            out.append(r)
    return np.array(out)


def enum_test(vals, sign=+1):
    def gd(inter):
        a = np.mean([vals[s] for s in inter]); b = np.mean([vals[s] for s in SP if s not in inter])
        return sign * (a - b)
    allv = [gd(frozenset(c)) for c in combinations(SP, 3)]
    obs = gd(OBS)
    rank = int(np.sum(np.array(allv) >= obs))
    return float(obs), rank, rank / len(allv), sorted(allv, reverse=True)


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    rng = np.random.RandomState(SEED)

    rows, per_species = [], {}
    print("\n" + "=" * 100)
    print("DISTRIBUTION SHIFT — zero-IQR variables excluded, bootstrap over variables")
    print("=" * 100)
    print(f"  {'species':<26}{'n_topo':>7}{'n_clim':>7}{'excl':>6}"
          f"{'|topo|':>8}{'|clim|':>8}{'ratio':>8}{'95% CI':>20}")

    for sp in SP:
        combined, clean = build_species_frame(dff, sp, env)
        nat_df = combined[combined["range_label"] == "native"]
        inv_df = combined[combined["range_label"] == "invasive"]
        topo, clim, excluded, sig_topo, sig_clim = [], [], 0, [], []
        for v in clean:
            dom = classify_variable(v)
            if dom not in ("Topography", "Climate"):
                continue
            nat, inv = nat_df[v].values, inv_df[v].values
            s = shift(nat, inv)
            if s is None:
                excluded += 1
                continue
            lo, hi = var_ci(nat, inv, rng)
            sig = lo is not None and (lo > 0 or hi < 0)
            rows.append({"species": sp, "variable": v, "domain": dom, "std_shift": s,
                         "ci_low": lo, "ci_high": hi, "ci_excludes_zero": bool(sig)})
            (topo if dom == "Topography" else clim).append(s)
            if sig:
                (sig_topo if dom == "Topography" else sig_clim).append(s)

        r = ratio_from(topo, clim)
        b = boot_ratio(topo, clim, rng)
        lo, hi = np.percentile(b, [2.5, 97.5])
        r_sig = ratio_from(sig_topo, sig_clim)
        per_species[sp] = {"ratio": float(r), "ci_low": float(lo), "ci_high": float(hi),
                           "ratio_significant_only": float(r_sig) if np.isfinite(r_sig) else None,
                           "n_topo": len(topo), "n_clim": len(clim), "n_excluded_zero_iqr": excluded,
                           "mean_abs_topo": float(np.mean(np.abs(topo))),
                           "mean_abs_clim": float(np.mean(np.abs(clim))),
                           "pathway": "intercontinental" if sp in INTER else "within-continent"}
        print(f"  {sp:<26}{len(topo):>7}{len(clim):>7}{excluded:>6}"
              f"{np.mean(np.abs(topo)):>8.3f}{np.mean(np.abs(clim)):>8.3f}{r:>8.3f}"
              f"   [{lo:.3f}, {hi:.3f}]")

    pd.DataFrame(rows).to_csv(OUT / "distribution_shift_clean.csv", index=False)

    print("\n  ratio using only variables whose CI excludes zero (sensitivity):")
    for sp in SP:
        v = per_species[sp]["ratio_significant_only"]
        print(f"    {sp:<26}{v:.3f}" if v is not None else f"    {sp:<26}   n/a")

    print("\n" + "-" * 100)
    print("PATHWAY TEST on topography:climate ratio (exact enumeration, within - inter)")
    print("-" * 100)
    ratios = {s: per_species[s]["ratio"] for s in SP}
    obs, rank, p, allv = enum_test(ratios, sign=-1)
    print(f"  observed group difference = {obs:+.4f}   rank {rank}/10   exact one-sided P = {p:.2f}")
    print(f"  all ten values: " + ", ".join(f"{v:+.3f}" for v in allv))
    tag = "SEPARATES" if rank == 1 else "does NOT separate"
    print(f"  -> {tag}")

    grp = {}
    for name, g in [("intercontinental", INTER), ("within-continent", WITHIN)]:
        vals = [per_species[s]["ratio"] for s in g]
        grp[name] = {"mean_ratio": float(np.mean(vals))}
        print(f"  {name:<18} mean ratio = {np.mean(vals):.3f}")

    (OUT / "distribution_shift_bootstrap.json").write_text(json.dumps(
        {"per_species": per_species, "group_means": grp,
         "pathway_test": {"observed": obs, "rank": rank, "p": p}}, indent=2, default=float))
    print(f"\nSaved: {OUT/'distribution_shift_clean.csv'}")
    print(f"Saved: {OUT/'distribution_shift_bootstrap.json'}")


if __name__ == "__main__":
    main()
