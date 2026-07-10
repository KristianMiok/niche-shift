"""
threshold_stability.py — item 2b: is the +46.8% threshold shift (Table S6 / Figure 5)
stable, or a single-tree artefact?

separate_niche_models.train_niche_tree uses a fixed random_state, so every reported
threshold shift comes from ONE tree. shift_pct = (inv_thr - nat_thr)/|nat_thr|*100 also
divides by a possibly-small native threshold, amplifying tiny split changes. We refit the
native and invasive depth-limited trees over 20 seeds per species and record the
distribution of shift_pct for each shared variable. Wide spread (or the variable dropping
out of the shared set across seeds) => the shift is not a stable ecological quantity.

Run:  python src/threshold_stability.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_geotraits, load_config, get_env_variables, COL_STATUS
from species_selector import apply_quality_filters, STATUS_NATIVE, STATUS_ALIEN
from grouped_permutation import build_species_frame

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
N_SEEDS, DEPTH = 20, 4
OUTDIR = Path("results/tables")
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT


def first_threshold(tree, feat_idx):
    """First split threshold on feature feat_idx in a fitted sklearn tree, or None."""
    t = tree.tree_
    for node in range(t.node_count):
        if t.feature[node] == feat_idx:
            return float(t.threshold[node])
    return None


def top_split_feature(tree):
    """Root split feature index (the dominant variable of this tree)."""
    return int(tree.tree_.feature[0])


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    summary = {}

    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        nat = combined[combined["range_label"] == "native"]
        inv = combined[combined["range_label"] == "invasive"]
        # presence-vs-background framing per group, as in separate_niche_models:
        # here we approximate with native-vs-invasive shared-variable thresholds across seeds
        Xn = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values

        shifts = {}   # variable -> list of shift_pct across seeds
        roots = {"native": [], "invasive": []}
        for seed in range(N_SEEDS):
            # native tree: native presence vs background (invasive as contrast), and vice versa
            dt_nat = DecisionTreeClassifier(max_depth=DEPTH, random_state=seed,
                                            class_weight="balanced").fit(Xn, 1 - y)
            dt_inv = DecisionTreeClassifier(max_depth=DEPTH, random_state=seed,
                                            class_weight="balanced").fit(Xn, y)
            roots["native"].append(clean[top_split_feature(dt_nat)])
            roots["invasive"].append(clean[top_split_feature(dt_inv)])
            for j, v in enumerate(clean):
                tn, ti = first_threshold(dt_nat, j), first_threshold(dt_inv, j)
                if tn is not None and ti is not None and abs(tn) > 1e-9:
                    shifts.setdefault(v, []).append((ti - tn) / abs(tn) * 100)

        # keep variables present in a majority of seeds
        stable_vars = {v: s for v, s in shifts.items() if len(s) >= N_SEEDS // 2}
        rows = []
        for v, s in sorted(stable_vars.items(), key=lambda kv: -abs(np.median(kv[1]))):
            a = np.array(s)
            rows.append({"variable": v, "n_seeds_present": len(s),
                         "shift_median": float(np.median(a)), "shift_min": float(a.min()),
                         "shift_max": float(a.max()), "shift_sd": float(a.std()),
                         "sign_flips": bool(a.min() < 0 < a.max())})
        summary[sp] = {"root_native_mode": pd.Series(roots["native"]).mode().tolist(),
                       "root_invasive_mode": pd.Series(roots["invasive"]).mode().tolist(),
                       "root_native_unique": pd.Series(roots["native"]).nunique(),
                       "root_invasive_unique": pd.Series(roots["invasive"]).nunique(),
                       "variables": rows}

        print("\n" + "=" * 92)
        print(f"{sp}  —  threshold-shift stability over {N_SEEDS} seeds")
        print("=" * 92)
        print(f"  root split (native tree): {summary[sp]['root_native_unique']} distinct features across seeds "
              f"(mode {summary[sp]['root_native_mode']})")
        print(f"  root split (invasive tree): {summary[sp]['root_invasive_unique']} distinct features across seeds")
        print(f"  {'variable':<22}{'n_seed':>7}{'median%':>9}{'min%':>9}{'max%':>9}{'SD%':>8}{'flip':>6}")
        for r in rows[:6]:
            print(f"  {r['variable']:<22}{r['n_seeds_present']:>7}{r['shift_median']:>9.1f}"
                  f"{r['shift_min']:>9.1f}{r['shift_max']:>9.1f}{r['shift_sd']:>8.1f}"
                  f"{'  YES' if r['sign_flips'] else '   no':>6}")

    (OUTDIR / "threshold_stability_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved: {OUTDIR/'threshold_stability_summary.json'}")
    print("\n>>> Read: if shift medians swing widely (min/max span, high SD, sign flips) or the root")
    print(">>> split feature changes across seeds, the threshold shifts are single-tree artefacts")
    print(">>> => Table S6 / Figure 5 cannot report a stable magnitude like +46.8%.")


if __name__ == "__main__":
    main()
