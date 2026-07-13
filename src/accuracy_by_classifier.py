"""
accuracy_by_classifier.py — per-species, per-classifier CV accuracy and AUC from the
single canonical post-dedup pipeline (the last unverified quantity in the manuscript).

The "98.2-100.0%" reported earlier was RANDOM FOREST ONLY and was not labelled as such.
Results 3.1 quotes both a decision-tree range (0.965-0.999) and a random-forest range
(0.979-0.999), both pre-dedup. This produces the full five-species x two-classifier table
so every accuracy figure in the paper syncs to one source.

Both classifiers share the same folds: StratifiedKFold(5, shuffle=True, random_state=42)
on the same post-dedup per-species frames (apply_quality_filters -> build_species_frame).
Decision tree: max_depth=5, class_weight balanced, seed 42 (as used elsewhere).
Random forest: 500 trees, class_weight balanced, seed 42 — identical to
regenerate_tables_s2_s5.py, so the RF column reproduces the canonical accuracy exactly.

Run:  python src/accuracy_by_classifier.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, TREES, DEPTH = 42, 500, 5
OUT = Path("results/tables"); OUT.mkdir(parents=True, exist_ok=True)
INTER = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN = ["Faxonius virilis", "Faxonius rusticus"]
SP = INTER + WITHIN


def cv_scores(make_model, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    accs, aucs = [], []
    for tr, te in cv.split(X, y):
        m = make_model().fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], m.predict(X[te])))
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(aucs)), float(np.std(aucs))


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]

    rows = []
    for sp in SP:
        combined, clean = build_species_frame(dff, sp, env)
        X = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values

        dt_acc, dt_acc_sd, dt_auc, dt_auc_sd = cv_scores(
            lambda: DecisionTreeClassifier(max_depth=DEPTH, class_weight="balanced",
                                           random_state=SEED), X, y)
        rf_acc, rf_acc_sd, rf_auc, rf_auc_sd = cv_scores(
            lambda: RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                           random_state=SEED, n_jobs=-1), X, y)

        rows.append({"species": sp,
                     "pathway": "intercontinental" if sp in INTER else "within-continent",
                     "n_native": int((y == 0).sum()), "n_invaded": int((y == 1).sum()),
                     "DT_accuracy": dt_acc, "DT_accuracy_sd": dt_acc_sd,
                     "DT_auc": dt_auc, "DT_auc_sd": dt_auc_sd,
                     "RF_accuracy": rf_acc, "RF_accuracy_sd": rf_acc_sd,
                     "RF_auc": rf_auc, "RF_auc_sd": rf_auc_sd})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "accuracy_by_classifier.csv", index=False)

    print("\n" + "=" * 92)
    print("CLASSIFICATION ACCURACY — 5-fold CV, post-dedup, single canonical pipeline")
    print("=" * 92)
    print(f"  {'species':<26}{'DT acc':>9}{'DT AUC':>9}{'RF acc':>9}{'RF AUC':>9}")
    for _, r in df.iterrows():
        print(f"  {r['species']:<26}{r['DT_accuracy']:>9.3f}{r['DT_auc']:>9.3f}"
              f"{r['RF_accuracy']:>9.3f}{r['RF_auc']:>9.3f}")

    print("\n  RANGES for the manuscript:")
    print(f"    decision tree : {df.DT_accuracy.min():.3f}-{df.DT_accuracy.max():.3f}  "
          f"({df.DT_accuracy.min()*100:.1f}%-{df.DT_accuracy.max()*100:.1f}%)")
    print(f"    random forest : {df.RF_accuracy.min():.3f}-{df.RF_accuracy.max():.3f}  "
          f"({df.RF_accuracy.min()*100:.1f}%-{df.RF_accuracy.max()*100:.1f}%)")
    print(f"\n  Manuscript currently says: DT 0.965-0.999, RF 0.979-0.999 (both PRE-dedup)")
    print(f"  Earlier '98.2-100.0%' was the RF column only, unlabelled.")
    print(f"\n  Saved: {OUT/'accuracy_by_classifier.csv'}")


if __name__ == "__main__":
    main()
