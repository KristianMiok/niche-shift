"""
grouped_permutation.py — Task (e): domain-level grouped permutation importance.

Answers Reviewer 2 point 6 ("feature importance is not causal") without relying
on Gini or SHAP. For each thematic domain (Climate/Topography/Soil/Land Cover),
permute ALL of that domain's columns TOGETHER with a single shared row shuffle,
breaking the domain<->label link while preserving within-domain correlation
structure. The resulting drop in held-out performance is a domain-level
importance that is immune to within-domain collinearity by construction.

Contrast with sklearn's permutation_importance (per-column, independent): under
correlated predictors (climate variables are strongly intercorrelated) the model
compensates via a correlated sibling column and importance leaks. Shuffling the
whole domain at once destroys the domain jointly -> clean domain effect.

Held-out: importances computed per fold on the 5-fold CV test partition, then
averaged, so drops reflect generalisation, not memorised training signal.

Models: RandomForest (primary, 500 trees, matches manuscript) and Decision Tree
(depth 5) to see whether grouped permutation agrees with the DT-Gini magnitudes
or contradicts them.

Run:  python src/grouped_permutation.py
      python src/grouped_permutation.py --repeats 20 --species "Faxonius virilis"
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import (
    load_geotraits, load_config, get_env_variables, COL_SPECIES, COL_STATUS,
)
from species_selector import (
    apply_quality_filters, STUDY_SPECIES, STATUS_NATIVE, STATUS_ALIEN,
)
from data_preparation import (
    handle_missing_values, remove_constant_variables, remove_highly_correlated,
)
from cross_species_comparison import classify_variable

INPUT = "data/raw/combined_data_true_master.csv"
CONFIG = "config/species_config.yaml"
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
DOMAINS = ["Climate", "Topography", "Soil", "Land Cover"]


def build_species_frame(df_filtered, species, env_vars):
    sp = df_filtered[df_filtered[COL_SPECIES] == species].copy()
    native = sp[sp[COL_STATUS] == STATUS_NATIVE]
    invasive = sp[sp[COL_STATUS] == STATUS_ALIEN]
    combined = pd.concat([native, invasive], ignore_index=True)
    combined, clean = handle_missing_values(combined, env_vars)
    clean = remove_constant_variables(combined, clean)
    clean = remove_highly_correlated(combined, clean, threshold=0.98)
    combined["range_label"] = combined[COL_STATUS].map(
        {STATUS_NATIVE: "native", STATUS_ALIEN: "invasive"}
    )
    return combined.reset_index(drop=True), clean


def domain_index_map(clean):
    """Map each domain -> integer column positions in `clean`."""
    dom = {}
    for j, v in enumerate(clean):
        d = classify_variable(v)
        dom.setdefault(d, []).append(j)
    return {d: np.array(idx) for d, idx in dom.items() if d in DOMAINS}


def grouped_perm_one_model(model, X, y, clean, n_repeats, seed):
    """Return dict domain -> (mean_drop_acc, sd_drop_acc, mean_drop_auc, sd_drop_auc)."""
    dmap = domain_index_map(clean)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)

    base_acc, base_auc = [], []
    per_dom = {d: {"acc": [], "auc": []} for d in dmap}

    for tr, te in cv.split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        Xte = X[te]
        base_acc.append(accuracy_score(y[te], m.predict(Xte)))
        base_auc.append(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))

        for d, cols in dmap.items():
            for _ in range(n_repeats):
                Xp = Xte.copy()
                perm = rng.permutation(Xp.shape[0])   # ONE shared shuffle
                Xp[:, cols] = Xp[np.ix_(perm, cols)]  # whole domain moves together
                per_dom[d]["acc"].append(accuracy_score(y[te], m.predict(Xp)))
                per_dom[d]["auc"].append(roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))

    b_acc, b_auc = np.mean(base_acc), np.mean(base_auc)
    out = {"_baseline": {"acc": float(b_acc), "auc": float(b_auc)}}
    for d in dmap:
        da = b_acc - np.array(per_dom[d]["acc"])
        du = b_auc - np.array(per_dom[d]["auc"])
        out[d] = {
            "drop_acc": float(da.mean()), "drop_acc_sd": float(da.std()),
            "drop_auc": float(du.mean()), "drop_auc_sd": float(du.std()),
            "n_vars": int(len(dmap[d])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--species", default=None, help="run a single species")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = load_config(CONFIG)
    df = load_geotraits(INPUT)
    dff = apply_quality_filters(df, config)
    env = get_env_variables(dff)["all_env"]
    species_list = [args.species] if args.species else STUDY_SPECIES

    models = {
        "RF": RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=args.seed, n_jobs=-1),
        "DT": DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                     random_state=args.seed),
    }

    summary = {}
    for sp in species_list:
        print("\n" + "=" * 78)
        print(f"{sp}  —  grouped permutation importance  (repeats={args.repeats})")
        print("=" * 78)
        combined, clean = build_species_frame(dff, sp, env)
        X = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values
        dmap = domain_index_map(clean)
        print(f"  n={len(y)} ({(y==0).sum()} nat / {(y==1).sum()} inv) | "
              f"{len(clean)} vars | domain vars: "
              + ", ".join(f"{d.split()[0][:4]}={len(dmap.get(d, []))}" for d in DOMAINS))

        summary[sp] = {}
        for mname, model in models.items():
            res = grouped_perm_one_model(model, X, y, clean, args.repeats, args.seed)
            summary[sp][mname] = res
            b = res["_baseline"]
            print(f"\n  [{mname}] baseline acc={b['acc']:.3f} AUC={b['auc']:.3f}")
            print(f"    {'domain':<12}{'Δacc':>10}{'±sd':>8}{'ΔAUC':>10}{'±sd':>8}{'vars':>6}")
            order = sorted([d for d in res if d != "_baseline"],
                           key=lambda d: -res[d]["drop_auc"])
            for d in order:
                r = res[d]
                print(f"    {d:<12}{r['drop_acc']:>10.4f}{r['drop_acc_sd']:>8.4f}"
                      f"{r['drop_auc']:>10.4f}{r['drop_auc_sd']:>8.4f}{r['n_vars']:>6}")

        # cross-species contrast helper: RF topography ΔAUC
        rf = summary[sp]["RF"]
        if "Topography" in rf and "Climate" in rf:
            print(f"\n  >>> RF ΔAUC  Climate={rf['Climate']['drop_auc']:.4f}  "
                  f"Topography={rf['Topography']['drop_auc']:.4f}  "
                  f"(topo-climate = {rf['Topography']['drop_auc']-rf['Climate']['drop_auc']:+.4f})")

    out = OUTDIR / "grouped_permutation_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved: {out}")

    # compact cross-species table (RF ΔAUC by domain) — the contrast that matters
    if len(species_list) > 1:
        print("\n" + "=" * 78)
        print("CROSS-SPECIES  RF ΔAUC by domain  (grouped permutation, collinearity-immune)")
        print("=" * 78)
        print(f"{'species':<26}" + "".join(f"{d.split()[0][:5]:>9}" for d in DOMAINS))
        for sp in species_list:
            rf = summary[sp]["RF"]
            print(f"{sp:<26}" + "".join(f"{rf.get(d, {}).get('drop_auc', 0):>9.4f}" for d in DOMAINS))


if __name__ == "__main__":
    main()
