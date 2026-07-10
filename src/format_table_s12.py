"""
format_table_s12.py — reformat continent-only accuracy as standalone Table S12.

Lucian: continent-only accuracy is conceptually distinct from the blocked-CV columns
(it measures label-continent confounding, not spatial autocorrelation), so it belongs in
its own table. Assembles Table S12 from continent_only_accuracy.json with interpretive
columns making the non-identifiability point explicit for Reviewer 2 point 2.
"""
import json
from pathlib import Path
import pandas as pd

TABLES = Path("results/tables")
data = json.loads((TABLES / "continent_only_accuracy.json").read_text())
rows = []
for r in data:
    acc = r["continent_only_accuracy"] * 100
    maj = r["majority_baseline"] * 100
    lift = acc - maj
    # non-identifiability flag: continent alone near-perfect AND well above majority
    conf = "severe" if acc >= 99 else ("moderate" if lift >= 5 else "low")
    rows.append({
        "Species": r["species"],
        "Pathway": r["pathway"],
        "Continent-only accuracy (%)": round(acc, 1),
        "Majority baseline (%)": round(maj, 1),
        "Lift over majority (pp)": round(lift, 1),
        "Invaded on native continent (%)": round(r["pct_invaded_on_native_continent"], 1),
        "Label-continent confounding": conf,
    })
df = pd.DataFrame(rows)
df.to_csv(TABLES / "TableS12_continent_only.csv", index=False)
print("===== TABLE S12: Continent-only classification (label-continent confounding) =====\n")
print(df.to_string(index=False))
print("\nNote for caption: for P. leniusculus and F. limosus continent alone classifies")
print("native-vs-invaded at ~100% (severe confounding), so entering continent as a covariate")
print("(Option B) is entering a copy of the response; the climate collapse there is")
print("non-identifiability, not evidence that climate is a continental artefact.")
print(f"\nSaved: {TABLES/'TableS12_continent_only.csv'}")
