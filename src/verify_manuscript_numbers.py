"""
verify_manuscript_numbers.py — assemble every reportable number from the repo outputs
into one reference sheet, so the manuscript draft can be checked against it before submission.

Pulls from the canonical post-dedup JSON/CSV summaries and prints a single labelled list:
record counts, accuracy, theme importances (Table S2), grouped permutation (S11), exact
enumeration (S1), blocked CV (S10), D/I overlap (S5), continent-only (S12), pathway refit,
background-matched (g), dimensionality. Anything the manuscript states should match a line here.
"""
import json
from pathlib import Path
import numpy as np

T = Path("results/tables")
def load(name):
    p = T / name
    return json.loads(p.read_text()) if p.exists() else None

SP = ["Procambarus clarkii","Faxonius limosus","Pacifastacus leniusculus","Faxonius virilis","Faxonius rusticus"]
DOM = ["Climate","Topography","Soil","Land Cover"]

print("="*80); print("MANUSCRIPT NUMBER REFERENCE SHEET (post-dedup, canonical)"); print("="*80)

# --- Table S5 / record counts / D-I ---
s5 = load("tables_s2_s5_summary.json")
nov = load("niche_overlap_summary.json")
print("\n[Record counts + D/I overlap]")
if s5:
    tot_nat = sum(r["n_native"] for r in s5["table_s5"]); tot_inv = sum(r["n_invaded"] for r in s5["table_s5"])
    print(f"  totals: native={tot_nat}, invaded={tot_inv}, all={tot_nat+tot_inv}")
    for r in s5["table_s5"]:
        print(f"  {r['species']:<26} nat={r['n_native']:>5} inv={r['n_invaded']:>5} feats={r['n_features']}")
if nov:
    novl = nov if isinstance(nov, list) else nov.get("data", [])
    for r in novl:
        print(f"  {r['species']:<26} D={r['schoeners_D']:.3f}  I={r['warrens_I']:.3f}  PC1={r['pc1_variance_explained']:.3f} PC2={r['pc2_variance_explained']:.3f}")

# --- accuracy ---
print("\n[Classification accuracy, 5-fold CV]")
if s5:
    accs = [r["cv_accuracy_mean"] for r in s5["accuracy"]]
    print(f"  range: {min(accs)*100:.1f}%-{max(accs)*100:.1f}%")
    for r in s5["accuracy"]:
        print(f"  {r['species']:<26} acc={r['cv_accuracy_mean']*100:.1f}%  AUC={r['cv_auc_mean']:.3f}")

# --- Table S2 theme importance ---
print("\n[Table S2: theme-level RF Gini %, full-model mean (fold SD)]")
if s5:
    maxsd = 0
    for r in s5["table_s2"]:
        line = f"  {r['species']:<26} " + " ".join(f"{d[:4]}={r[d+'_mean']:.1f}({r[d+'_foldSD']:.1f})" for d in DOM)
        print(line); maxsd = max(maxsd, max(r[d+'_foldSD'] for d in DOM))
    wc = [r for r in s5["table_s2"] if r["pathway"]=="within-continent"]
    print(f"  within-continent means: " + ", ".join(f"{d}={np.mean([r[d+'_mean'] for r in wc]):.1f}%" for d in DOM))
    print(f"  max fold SD: {maxsd:.2f}%")

# --- grouped permutation S11 ---
print("\n[Table S11: grouped-permutation RF dAUC by domain]")
gp = load("grouped_permutation_summary.json")
if gp:
    inter=["Procambarus clarkii","Faxonius limosus","Pacifastacus leniusculus"]
    for sp in SP:
        rf=gp[sp]["RF"]; print(f"  {sp:<26} " + " ".join(f"{d[:4]}={rf[d]['drop_auc']:.4f}" for d in DOM))
    for grp,name in [(inter,"intercontinental"),([s for s in SP if s not in inter],"within-continent")]:
        print(f"  {name} mean: " + ", ".join(f"{d}={np.mean([gp[s]['RF'][d]['drop_auc'] for s in grp]):.4f}" for d in DOM))

# --- exact enumeration S1 ---
print("\n[Table S1: exact permutation, observed + rank + P]")
s1 = None
p = T/"TableS1_exact_permutation.csv"
if p.exists():
    import csv
    for row in csv.DictReader(p.open()):
        print(f"  {row['statistic']:<12}{row['importance_measure']:<26} obs={row['observed']:>8} rank={row['rank_among_10']} P={row['exact_one_sided_P']}")

# --- background-matched (g) ---
print("\n[Background-matched (g): climate dAUC full -> A_k5 -> B]")
bg = load("background_matched_refit_summary.json")
if bg:
    for sp in SP:
        d=bg[sp]; full=d["full"]["grouped_perm_dauc"]["Climate"]
        a5=d["option_a"].get("k5",{}).get("grouped_perm_dauc",{}).get("Climate")
        b=d["option_b"]["grouped_perm_dauc"]["Climate"]
        a5s=f"{a5:.4f}" if a5 is not None else d["option_a"].get("k5",{}).get("status","?")
        print(f"  {sp:<26} full={full:.4f}  A_k5={a5s}  B={b:.4f}")

# --- continent-only S12 ---
print("\n[Table S12: continent-only accuracy]")
co = load("continent_only_accuracy.json")
if co:
    for r in co:
        print(f"  {r['species']:<26} acc={r['continent_only_accuracy']*100:.1f}%  maj={r['majority_baseline']*100:.1f}%  inv_on_native={r['pct_invaded_on_native_continent']:.1f}%")

# --- pathway refit (d/vi) ---
print("\n[Pathway refit (grouped-perm): baseline -> refit, focus domain]")
pr = load("pathway_refit_grouped_summary.json")
if pr:
    for sp,d in pr.items():
        f=d["focus"]; b=d["baseline"]["grouped_perm_dauc"][f]; r=d["refit"]["grouped_perm_dauc"][f]
        print(f"  {sp:<26} {f}: {b:.4f} -> {r:.4f}  (dropped {d['n_dropped']})")

print("\n" + "="*80)
print("Compare each manuscript figure/number against the matching line above.")
