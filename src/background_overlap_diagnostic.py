"""
background_overlap_diagnostic.py — Task (g) FEASIBILITY ONLY (Reviewer 2 point 2).

Lucian asked whether Option A (environmentally-restricted classifier) is FEASIBLE before
committing to it, because it may weaken the result and he wants the honest answer first.
This script does NOT run the restricted refit. It only measures how many native and
invaded records survive inside the shared environmental envelope, per species, so we can
judge feasibility from numbers rather than guesswork.

Shared envelope = intersection region in the first k PCA axes (k for ~80% variance),
approximated per-axis: a record is "in overlap" if, on every retained PCA axis, its score
lies within the [min,max] range spanned by BOTH native and invaded (i.e. within the
per-axis overlap of the two groups' ranges). This is a generous convex-ish proxy; the true
intersection is smaller, so these counts are UPPER bounds on what a restricted refit keeps.

Reports per species: n_native/n_invaded total, how many fall in the shared envelope, and
the resulting class balance. If intersection counts are tiny (especially for the
intercontinental species, whose native NA and invaded EU ranges are climatically distinct),
Option A is not viable there and Option B (continent as covariate) is the honest fallback.

Run:  python src/background_overlap_diagnostic.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, VAR_TARGET = 42, 0.80
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]

    rows = []
    print("\n" + "=" * 96)
    print("TASK (g) FEASIBILITY — shared environmental envelope, native vs invaded")
    print("=" * 96)
    print(f"  {'species':<26}{'pathway':<18}{'n_nat':>7}{'n_inv':>7}{'k_pc':>6}"
          f"{'nat_in':>8}{'inv_in':>8}{'%nat':>7}{'%inv':>7}{'balance':>9}")

    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        status = (combined["range_label"] == "invasive").astype(int).values
        Xs = np.nan_to_num(StandardScaler().fit_transform(combined[clean].values),
                           nan=0.0, posinf=0.0, neginf=0.0)
        Xs = Xs[:, Xs.std(axis=0) > 1e-8]

        cum = np.cumsum(PCA(svd_solver="full", random_state=SEED).fit(Xs).explained_variance_ratio_)
        k = int(np.searchsorted(cum, VAR_TARGET) + 1)
        P = PCA(n_components=k, svd_solver="full", random_state=SEED).fit_transform(Xs)

        nat, inv = P[status == 0], P[status == 1]
        # per-axis overlap band shared by both groups
        lo = np.maximum(nat.min(axis=0), inv.min(axis=0))
        hi = np.minimum(nat.max(axis=0), inv.max(axis=0))
        in_band = lambda A: np.all((A >= lo) & (A <= hi), axis=1)
        nat_in, inv_in = int(in_band(nat).sum()), int(in_band(inv).sum())
        pn = 100 * nat_in / len(nat) if len(nat) else 0
        pi = 100 * inv_in / len(inv) if len(inv) else 0
        bal = f"{nat_in}:{inv_in}"
        pw = "intercontinental" if sp in INTERCONTINENTAL else "within-continent"
        print(f"  {sp:<26}{pw:<18}{len(nat):>7}{len(inv):>7}{k:>6}"
              f"{nat_in:>8}{inv_in:>8}{pn:>6.1f}%{pi:>6.1f}%{bal:>9}")
        rows.append({"species": sp, "pathway": pw, "n_native": len(nat), "n_invaded": len(inv),
                     "k_pc_80": k, "native_in_envelope": nat_in, "invaded_in_envelope": inv_in,
                     "pct_native_in": pn, "pct_invaded_in": pi})

    pd.DataFrame(rows).to_csv(OUTDIR / "background_overlap_diagnostic.csv", index=False)
    (OUTDIR / "background_overlap_diagnostic.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nSaved: {OUTDIR/'background_overlap_diagnostic.csv'}")
    print("\n>>> Read: for each species, are BOTH nat_in and inv_in large enough (say >=40 each,")
    print(">>> ideally >=100) to refit inside the shared envelope? If the intercontinental")
    print(">>> species collapse to tiny counts, Option A is not viable there -> recommend Option B.")


if __name__ == "__main__":
    main()
