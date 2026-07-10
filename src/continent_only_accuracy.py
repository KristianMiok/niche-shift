"""
continent_only_accuracy.py — item 1: continent-only classification accuracy per species.

Reviewer-2-point-2 reinterpretation (Lucian): the Option B climate collapse for F. limosus
and P. leniusculus is non-identifiability, not sparse sampling — for those species continent
alone predicts native-vs-invaded almost perfectly, so entering continent as a covariate is
entering a copy of the response. This quantifies that: 5-fold CV accuracy of a classifier
using ONLY the continent label (one-hot) to predict native-vs-invaded status, per species.
Also reports % of invaded records on the native continent (from the continent audit logic).

Column to add to Table S10/S11.  Run:  python src/continent_only_accuracy.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED = 42
LAT, LON = "lat_or", "long_or"
OUTDIR = Path("results/tables")
CONT_NAME = {"NA": "North America", "SA": "South America", "EU": "Europe",
             "AS": "Asia", "AF": "Africa", "OC": "Oceania", "AN": "Antarctica"}
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT
NATIVE_CONTINENT = {sp: "North America" for sp in SPECIES}  # all five are North American natives


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


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    rows = []
    print(f"\n{'species':<26}{'pathway':<18}{'cont_only_acc':>14}{'majority':>10}{'inv_on_native%':>16}")
    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        y = (combined["range_label"] == "invasive").astype(int).values
        conts = continents_for(combined)
        Xc = pd.get_dummies(pd.Series(conts)).values.astype(float)
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        acc = cross_val_score(LogisticRegression(max_iter=1000), Xc, y, cv=cv, scoring="accuracy").mean()
        maj = cross_val_score(DummyClassifier(strategy="most_frequent"), Xc, y, cv=cv, scoring="accuracy").mean()
        inv = np.array(conts)[y == 1]
        on_native = 100 * np.mean(inv == NATIVE_CONTINENT[sp]) if len(inv) else 0
        pw = "intercontinental" if sp in INTERCONTINENTAL else "within-continent"
        print(f"{sp:<26}{pw:<18}{acc*100:>13.1f}%{maj*100:>9.1f}%{on_native:>15.1f}%")
        rows.append({"species": sp, "pathway": pw, "continent_only_accuracy": float(acc),
                     "majority_baseline": float(maj), "pct_invaded_on_native_continent": float(on_native)})
    pd.DataFrame(rows).to_csv(OUTDIR / "continent_only_accuracy.csv", index=False)
    (OUTDIR / "continent_only_accuracy.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nSaved: {OUTDIR/'continent_only_accuracy.csv'}")
    print("\n>>> High continent-only accuracy (limosus/leniusculus ~100%) => continent is a copy")
    print(">>> of the label => Option B climate dAUC ~0 is non-identifiability, not ecology.")


if __name__ == "__main__":
    main()
