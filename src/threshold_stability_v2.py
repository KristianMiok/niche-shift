"""
threshold_stability_v2.py — answer Lucian's _first_threshold question before finalising
the Decision-1 cut.

His concern: threshold_stability used the FIRST node where a variable appears, which may
be a minor deep split. Maybe the shifts are unstable only because of that choice, and a
threshold taken at the variable's HIGHEST-IMPORTANCE split is stable. This compares both
selection rules over 20 seeds; if max-importance is materially more stable, the cut is
premature and we salvage Figure 5 with the better threshold.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
N_SEEDS, DEPTH = 20, 4
INTERCONTINENTAL = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN_CONTINENT = ["Faxonius virilis", "Faxonius rusticus"]
SPECIES = INTERCONTINENTAL + WITHIN_CONTINENT


def thresholds_for_feature(tree, j):
    """All (threshold, node_importance) for feature j in the tree."""
    t = tree.tree_
    imp = t.compute_feature_importances(normalize=False)  # per-feature total; use node impurity drop
    out = []
    for node in range(t.node_count):
        if t.feature[node] == j:
            # weighted impurity decrease at this node
            n = t.n_node_samples[node]
            dec = (t.weighted_n_node_samples[node] / t.weighted_n_node_samples[0]) * (
                t.impurity[node]
                - (t.n_node_samples[t.children_left[node]] / n) * t.impurity[t.children_left[node]]
                - (t.n_node_samples[t.children_right[node]] / n) * t.impurity[t.children_right[node]]
            )
            out.append((float(t.threshold[node]), float(dec)))
    return out


def collect(tree, j, rule):
    ths = thresholds_for_feature(tree, j)
    if not ths:
        return None
    if rule == "first":
        return ths[0][0]
    return max(ths, key=lambda x: x[1])[0]  # max-importance split


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    out = {}
    for sp in SPECIES:
        combined, clean = build_species_frame(dff, sp, env)
        X = combined[clean].values
        y = (combined["range_label"] == "invasive").astype(int).values
        shifts = {"first": {}, "maximp": {}}
        for seed in range(N_SEEDS):
            dt_n = DecisionTreeClassifier(max_depth=DEPTH, random_state=seed, class_weight="balanced").fit(X, 1 - y)
            dt_i = DecisionTreeClassifier(max_depth=DEPTH, random_state=seed, class_weight="balanced").fit(X, y)
            for rule in ("first", "maximp"):
                for j, v in enumerate(clean):
                    tn, ti = collect(dt_n, j, rule), collect(dt_i, j, rule)
                    if tn is not None and ti is not None and abs(tn) > 1e-9:
                        shifts[rule].setdefault(v, []).append((ti - tn) / abs(tn) * 100)
        print(f"\n{sp}")
        print(f"  {'variable':<22}{'rule':<8}{'n':>4}{'median%':>9}{'min%':>8}{'max%':>8}{'SD%':>8}{'flip':>6}")
        rowsummary = {}
        for rule in ("first", "maximp"):
            stable = {v: s for v, s in shifts[rule].items() if len(s) >= N_SEEDS // 2}
            top = sorted(stable.items(), key=lambda kv: -abs(np.median(kv[1])))[:3]
            for v, s in top:
                a = np.array(s)
                flip = a.min() < 0 < a.max()
                print(f"  {v:<22}{rule:<8}{len(s):>4}{np.median(a):>9.1f}{a.min():>8.1f}{a.max():>8.1f}{a.std():>8.1f}{'YES' if flip else 'no':>6}")
            rowsummary[rule] = {"median_SD_top3": float(np.mean([np.std(s) for _, s in top])) if top else None,
                                "any_flip_top3": bool(any(np.array(s).min() < 0 < np.array(s).max() for _, s in top))}
        out[sp] = rowsummary
        f_sd = rowsummary["first"]["median_SD_top3"]; m_sd = rowsummary["maximp"]["median_SD_top3"]
        if f_sd and m_sd:
            print(f"  >>> mean SD of top-3 shifts:  first={f_sd:.1f}%   max-importance={m_sd:.1f}%  "
                  f"({'max-imp MORE stable' if m_sd < f_sd*0.7 else 'both similarly unstable'})")
    Path("results/tables/threshold_stability_v2.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nSaved: results/tables/threshold_stability_v2.json")
    print(">>> If max-importance SD is not materially lower, the first-vs-max choice is not the")
    print(">>> cause; the shifts are unstable regardless -> the cut stands.")


if __name__ == "__main__":
    main()
