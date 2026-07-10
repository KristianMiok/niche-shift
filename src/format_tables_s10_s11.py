"""
format_tables_s10_s11.py — assemble Table S11 (grouped permutation, blocker ii) and
Table S10 (blocked CV, blocker iii) from the JSON summaries already produced by
grouped_permutation.py and spatial_block_cv.py. No new computation; formatting only.

Table S11: grouped-permutation RF dAUC, 5 species x 4 domains (+ accuracy-drop dAcc).
           The full table the new headline sentence must be written from.
Table S10: spatially blocked CV per species per block size (50/100/250/500 km):
           blocked accuracy, blocked AUC, and per-domain grouped-perm dAUC (the
           recomputed pathway contrast). Random-CV baseline row included.

Run:  python src/format_tables_s10_s11.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

TABLES = Path("results/tables")
DOMAINS = ["Climate", "Topography", "Soil", "Land Cover"]
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
BLOCKS = ["random", "50km", "100km", "250km", "500km"]


def pw(sp):
    return "intercontinental" if sp in INTERCONTINENTAL else "within-continent"


def fmt(v, p=4):
    return "  --" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{p}f}"


# ---------------- Table S11: grouped permutation ----------------
def table_s11():
    data = json.loads((TABLES / "grouped_permutation_summary.json").read_text())
    rows = []
    for sp in SPECIES:
        rf = data[sp]["RF"]
        r = {"species": sp, "pathway": pw(sp)}
        for d in DOMAINS:
            r[f"{d}_dAUC"] = float(rf.get(d, {}).get("drop_auc", np.nan))
            r[f"{d}_dAcc"] = float(rf.get(d, {}).get("drop_acc", np.nan))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "TableS11_grouped_permutation_full.csv", index=False)

    print("\n" + "=" * 96)
    print("TABLE S11  — grouped-permutation RF importance (dAUC = drop in AUC when domain permuted)")
    print("=" * 96)
    print(f"  {'species':<26}{'pathway':<18}" + "".join(f"{d[:5]:>9}" for d in DOMAINS))
    for _, r in df.iterrows():
        print(f"  {r['species']:<26}{r['pathway']:<18}"
              + "".join(f"{r[d+'_dAUC']:>9.4f}" for d in DOMAINS))
    print("\n  (dAUC: topography ~0 for the three intercontinental invaders, nonzero for both")
    print("   within-continent; climate largest for every species — the new-headline table.)")
    # group means for the headline sentence
    inter = df[df.pathway == "intercontinental"]; within = df[df.pathway == "within-continent"]
    print("\n  group means (dAUC):")
    for d in DOMAINS:
        print(f"    {d:<12} intercontinental={inter[d+'_dAUC'].mean():.4f}   "
              f"within-continent={within[d+'_dAUC'].mean():.4f}")
    return df


# ---------------- Table S10: blocked CV ----------------
def table_s10():
    data = json.loads((TABLES / "spatial_block_cv_summary.json").read_text())
    rows = []
    for sp in SPECIES:
        r = data[sp]["rows"]
        for blk in BLOCKS:
            b = r.get(blk, {})
            row = {"species": sp, "pathway": pw(sp), "cv": blk,
                   "n_cells": b.get("n_cells", "-"),
                   "accuracy": b.get("acc", np.nan), "auc": b.get("auc", np.nan)}
            for d in DOMAINS:
                row[f"{d}_dAUC"] = b.get(d, np.nan)
            note = b.get("note") or (f"{b.get('n_single_class_folds')} single-class fold(s)"
                                     if b.get("n_single_class_folds") else "")
            row["note"] = note
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "TableS10_blocked_cv_full.csv", index=False)

    print("\n" + "=" * 110)
    print("TABLE S10  — spatially blocked CV per species per block size")
    print("=" * 110)
    for sp in SPECIES:
        sub = df[df.species == sp]
        print(f"\n  {sp}  [{pw(sp)}]")
        print(f"    {'CV':<8}{'cells':>7}{'acc':>8}{'AUC':>8}"
              + "".join(f"{d[:4]+'dA':>9}" for d in DOMAINS) + "   note")
        for _, r in sub.iterrows():
            print(f"    {r['cv']:<8}{str(r['n_cells']):>7}{fmt(r['accuracy'],3):>8}{fmt(r['auc'],3):>8}"
                  + "".join(f"{fmt(r[d+'_dAUC']):>9}" for d in DOMAINS)
                  + (f"   [{r['note']}]" if r['note'] else ""))
    print("\n  (Topography dAUC row is the recomputed pathway contrast under blocking:")
    print("   ~0 for intercontinental at every block size, nonzero for within-continent.)")
    return df


def main():
    table_s11()
    table_s10()
    print(f"\nSaved: {TABLES/'TableS11_grouped_permutation_full.csv'}")
    print(f"Saved: {TABLES/'TableS10_blocked_cv_full.csv'}")


if __name__ == "__main__":
    main()
