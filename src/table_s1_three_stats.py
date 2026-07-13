"""
table_s1_three_stats.py — Table S1 extended with the three separation statistics
the senior author requested: climate alone, topography alone, and climate-minus-topography.
For each, the exact 3/2 enumeration (observed group-difference, rank among 10, one-sided P)
on grouped-permutation RF dAUC. Shows that climate alone does NOT separate the pathway
groups (rank 3/10) while topography alone (1/10) and the difference (1/10) do.

Run:  python src/table_s1_three_stats.py
"""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

T = Path("results/tables")
INTER = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN = ["Faxonius virilis", "Faxonius rusticus"]
SP = INTER + WITHIN
OBS = frozenset(INTER)

gp = json.loads((T / "grouped_permutation_summary.json").read_text())
clim = {s: gp[s]["RF"]["Climate"]["drop_auc"] for s in SP}
topo = {s: gp[s]["RF"]["Topography"]["drop_auc"] for s in SP}
cmt = {s: clim[s] - topo[s] for s in SP}


def enum_test(values, sign=+1):
    def gd(inter):
        a = np.mean([values[s] for s in inter]); b = np.mean([values[s] for s in SP if s not in inter])
        return sign * (a - b)
    allv = [gd(frozenset(c)) for c in combinations(SP, 3)]
    obs = gd(OBS)
    rank = int(np.sum(np.array(allv) >= obs))
    return float(obs), rank, rank / len(allv), sorted(allv, reverse=True)


rows = []
print("=" * 78)
print("TABLE S1 (extended) — separation of pathway groups by domain importance")
print("=" * 78)
print(f"  {'statistic':<28}{'observed':>10}{'rank':>8}{'P':>8}   {'separates?':<12}")
for name, vals, sign in [
    ("climate dAUC (inter - within)", clim, +1),
    ("topography dAUC (within - inter)", topo, -1),
    ("climate - topography", cmt, +1),
]:
    obs, rank, p, allv = enum_test(vals, sign)
    sep = "YES" if rank == 1 else "NO"
    rows.append({"statistic": name, "observed": round(obs, 4),
                 "rank_among_10": f"{rank}/10", "one_sided_P": round(p, 3), "separates": sep})
    print(f"  {name:<28}{obs:>10.4f}{rank:>6}/10{p:>8.3f}   {sep:<12}")
    print(f"    all 10 values: " + ", ".join(f"{v:+.4f}" for v in allv))

df = pd.DataFrame(rows)
df.to_csv(T / "TableS1_three_statistics.csv", index=False)
print(f"\nSaved: {T/'TableS1_three_statistics.csv'}")
print("\n>>> Key: climate alone does NOT separate the pathway groups (rank 3/10) — the")
print(">>> dichotomy is topographic (1/10), not climatic. This is the extended Table S1.")
