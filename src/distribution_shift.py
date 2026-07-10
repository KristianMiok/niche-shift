"""
distribution_shift.py — Decision-1 replacement for Table S6 / Figure 5.

Threshold shifts are single-tree artefacts (threshold_stability_v2 confirmed: SD 76-142%
under both first-split and max-importance-split selection, sign flips throughout). Replace
them with a stable, genuinely ecological quantity: for each species and each environmental
variable, the shift in the variable's DISTRIBUTION between native and invaded records,
median(invaded) - median(native), with a bootstrap 95% CI, standardised by the native IQR
so shifts are comparable across variables and species.

Then the same 3/2 exact enumeration (C(5,3)=10) test Lucian requires: do TOPOGRAPHIC
distribution shifts separate the pathway groups (rank <=2/10)? If yes, this is a stable
replacement for Figure 5. If no, distribution shifts don't carry the dichotomy either and
we drop the whole thread, keeping only the domain-level result (Figure 1C / Table S11).

Run:  python src/distribution_shift.py
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
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
OBSERVED_INTER = frozenset(INTERCONTINENTAL)


def standardized_shift(nat, inv):
    """(median_inv - median_nat) / native IQR; robust, comparable across variables."""
    iqr = np.subtract(*np.percentile(nat, [75, 25]))
    if iqr <= 0:
        iqr = np.std(nat) if np.std(nat) > 0 else 1.0
    return (np.median(inv) - np.median(nat)) / iqr


def boot_ci(nat, inv, rng):
    vals = [standardized_shift(rng.choice(nat, len(nat)), rng.choice(inv, len(inv)))
            for _ in range(N_BOOT)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rank_test(values):
    """Exact 3/2: is the observed grouping the most extreme of the 10 in |group-mean diff|?"""
    def gd(inter, sign):
        a = np.mean([values[s] for s in inter]); b = np.mean([values[s] for s in SPECIES if s not in inter])
        return sign * (a - b)
    best = None
    for sign in (+1, -1):
        allv = [gd(frozenset(c), sign) for c in combinations(SPECIES, 3)]
        obs = gd(OBSERVED_INTER, sign)
        rank = int(np.sum(np.array(allv) >= obs))
        if best is None or rank < best[1]:
            best = (obs, rank, rank / len(allv), sign)
    return best


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    rng = np.random.RandomState(SEED)

    # per-species, per-variable standardized distribution shift (topography focus)
    per_species_topo = {}   # species -> mean |shift| over topographic vars (for the group test)
    per_species_clim = {}
    rows = []
    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        nat_df = combined[combined["range_label"] == "native"]
        inv_df = combined[combined["range_label"] == "invasive"]
        topo_shifts, clim_shifts = [], []
        for v in clean:
            dom = classify_variable(v)
            if dom not in ("Topography", "Climate"):
                continue
            nat, inv = nat_df[v].values, inv_df[v].values
            s = standardized_shift(nat, inv)
            lo, hi = boot_ci(nat, inv, rng)
            sig = (lo > 0) or (hi < 0)   # CI excludes zero
            rows.append({"species": sp, "variable": v, "domain": dom,
                         "std_shift": s, "ci_low": lo, "ci_high": hi, "sig": sig})
            (topo_shifts if dom == "Topography" else clim_shifts).append(abs(s))
        per_species_topo[sp] = float(np.mean(topo_shifts)) if topo_shifts else 0.0
        per_species_clim[sp] = float(np.mean(clim_shifts)) if clim_shifts else 0.0
        print(f"  {sp:<26} mean|topo shift|={per_species_topo[sp]:.3f}  mean|clim shift|={per_species_clim[sp]:.3f}")

    pd.DataFrame(rows).to_csv(OUTDIR / "distribution_shift.csv", index=False)

    print("\n=== 3/2 rank test on mean |distribution shift| (does it separate pathway groups?) ===")
    for name, vals in [("topography", per_species_topo), ("climate", per_species_clim)]:
        obs, rank, p, sign = rank_test(vals)
        tag = "SEPARATES -> stable replacement for Fig 5" if rank <= 2 else "does NOT separate -> drop this thread too"
        print(f"  {name:<12} observed={obs:+.4f}  rank {rank}/10  P={p:.2f}   {tag}")

    (OUTDIR / "distribution_shift_summary.json").write_text(
        json.dumps({"per_species_topo": per_species_topo, "per_species_clim": per_species_clim,
                    "topo_test": rank_test(per_species_topo), "clim_test": rank_test(per_species_clim)},
                   indent=2, default=float))
    print(f"\nSaved: {OUTDIR/'distribution_shift.csv'}, distribution_shift_summary.json")


if __name__ == "__main__":
    main()
