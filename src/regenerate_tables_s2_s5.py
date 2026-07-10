"""
regenerate_tables_s2_s5.py — post-dedup regeneration of Table S2, Table S5, and the
classification-accuracy range, all from ONE set of per-species RF fits so they are
mutually consistent (blocker (i), plus the shifted Table S5 / accuracy that the
occurrence-level dedup moves).

Table S5  : n_native / n_invaded per species (post-dedup counts).
accuracy  : 5-fold CV accuracy and AUC per species (the "96.5-99.9%" range moves).
Table S2  : theme-level RF Gini importance, 5 species x 4 domains, as mean +/- fold SD.
            Importance is aggregated to domain WITHIN each CV fold (fit on the fold's
            train split), then summarised across the 5 folds -> genuine fold SD, not
            across-tree SD. The full-data domain means are also printed for reference
            (these are the numbers that feed the abstract / 3.2 / Figure 1).

Run:  python src/regenerate_tables_s2_s5.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame
from cross_species_comparison import classify_variable

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, TREES = 42, 500
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
DOMAINS = ["Climate", "Topography", "Soil", "Land Cover"]
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT


def domain_pct(imp_vec, feature_names):
    """Aggregate a Gini importance vector to domain percentages (grand-total normalised)."""
    s = pd.Series(imp_vec, index=feature_names)
    by = {d: 0.0 for d in DOMAINS + ["Other"]}
    for f, v in s.items():
        by[classify_variable(f) if classify_variable(f) in DOMAINS else "Other"] += v
    tot = sum(by.values())
    return {d: 100 * by[d] / tot for d in DOMAINS}  # 4 domains, % of grand total


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]

    s5_rows, acc_rows, s2_rows = [], [], []
    print("\n" + "=" * 92)
    print("REGENERATING Table S5, accuracy, Table S2  (post-dedup, one set of fits)")
    print("=" * 92)

    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        X = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values
        pw = "intercontinental" if sp in INTERCONTINENTAL else "within-continent"

        # ---- Table S5 counts ----
        n_nat, n_inv = int((y == 0).sum()), int((y == 1).sum())
        s5_rows.append({"species": sp, "pathway": pw, "n_native": n_nat, "n_invaded": n_inv,
                        "n_total": n_nat + n_inv, "n_features": len(clean)})

        # ---- per-fold: accuracy, AUC, domain importance ----
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        accs, aucs, fold_domains = [], [], []
        for tr, te in cv.split(X, y):
            m = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], m.predict(X[te])))
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
            fold_domains.append(domain_pct(m.feature_importances_, clean))
        acc_rows.append({"species": sp, "pathway": pw,
                         "cv_accuracy_mean": float(np.mean(accs)), "cv_accuracy_sd": float(np.std(accs)),
                         "cv_auc_mean": float(np.mean(aucs)), "cv_auc_sd": float(np.std(aucs))})

        # ---- full-data domain means (feed abstract / 3.2 / Figure 1) ----
        full = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1).fit(X, y)
        full_dom = domain_pct(full.feature_importances_, clean)
        fd = pd.DataFrame(fold_domains)  # 5 folds x 4 domains
        row = {"species": sp, "pathway": pw}
        for d in DOMAINS:
            row[f"{d}_mean"] = float(full_dom[d])          # full-model value
            row[f"{d}_foldmean"] = float(fd[d].mean())     # mean across folds
            row[f"{d}_foldSD"] = float(fd[d].std())        # genuine fold SD
        s2_rows.append(row)

    s5 = pd.DataFrame(s5_rows); acc = pd.DataFrame(acc_rows); s2 = pd.DataFrame(s2_rows)

    print("\n----- Table S5 (post-dedup counts) -----")
    print(s5.to_string(index=False))
    print(f"\n  invaded total across 5 species: {s5['n_invaded'].sum()}  (was 18,245 in the audit)")
    print(f"  native total across 5 species : {s5['n_native'].sum()}")

    print("\n----- Classification accuracy (5-fold CV) -----")
    for _, r in acc.iterrows():
        print(f"  {r['species']:<26} acc={r['cv_accuracy_mean']:.3f}±{r['cv_accuracy_sd']:.3f}  "
              f"AUC={r['cv_auc_mean']:.3f}±{r['cv_auc_sd']:.3f}")
    print(f"\n  accuracy range: {acc['cv_accuracy_mean'].min()*100:.1f}%–{acc['cv_accuracy_mean'].max()*100:.1f}%  "
          f"(manuscript said 96.5–99.9%)")

    print("\n----- Table S2 (theme-level RF importance, % ; full-model mean, fold SD) -----")
    print(f"  {'species':<26}" + "".join(f"{d[:4]+'(SD)':>16}" for d in DOMAINS))
    for _, r in s2.iterrows():
        print(f"  {r['species']:<26}" +
              "".join(f"{r[d+'_mean']:>7.1f} ({r[d+'_foldSD']:>4.1f})" for d in DOMAINS))
    print(f"\n  max fold SD across all cells: {max(r[d+'_foldSD'] for _,r in s2.iterrows() for d in DOMAINS):.2f}%"
          f"  (manuscript 3.3 claimed SD <= 1.6%)")

    # within-continent domain means (for the abstract '42% each' fix)
    wc = s2[s2['pathway'] == 'within-continent']
    print("\n  within-continent domain means (abstract fix): "
          + ", ".join(f"{d}={wc[d+'_mean'].mean():.1f}%" for d in DOMAINS))

    s5.to_csv(OUTDIR / "TableS5_record_counts_postdedup.csv", index=False)
    acc.to_csv(OUTDIR / "classification_accuracy_postdedup.csv", index=False)
    s2.to_csv(OUTDIR / "TableS2_theme_importance_postdedup.csv", index=False)
    (OUTDIR / "tables_s2_s5_summary.json").write_text(
        json.dumps({"table_s5": s5_rows, "accuracy": acc_rows, "table_s2": s2_rows}, indent=2, default=float))
    print(f"\nSaved: TableS5_record_counts_postdedup.csv, classification_accuracy_postdedup.csv, "
          f"TableS2_theme_importance_postdedup.csv")


if __name__ == "__main__":
    main()
