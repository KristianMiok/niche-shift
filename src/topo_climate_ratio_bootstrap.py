"""
topo_climate_ratio_bootstrap.py — item 6b: bootstrap CI for the topography-to-climate
grouped-permutation dAUC ratio, per pathway group, before it goes in the supplementary.

For each species, ratio = topography dAUC / climate dAUC (grouped permutation). Bootstrap
over the CV folds' held-out records to get a 95% CI on the per-group mean ratio. Confirms
the within-continent group has a materially higher topo:climate ratio than the
intercontinental group, with uncertainty, rather than as a point estimate.

Run:  python src/topo_climate_ratio_bootstrap.py
"""
import json
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from data_loader import load_geotraits, load_config, get_env_variables
from species_selector import apply_quality_filters
from grouped_permutation import build_species_frame, domain_index_map, DOMAINS

INPUT, CONFIG = "data/raw/combined_data_true_master.csv", "config/species_config.yaml"
SEED, TREES, N_BOOT = 42, 300, 1000
INTER = ["Procambarus clarkii", "Faxonius limosus", "Pacifastacus leniusculus"]
WITHIN = ["Faxonius virilis", "Faxonius rusticus"]
SP = INTER + WITHIN


def fold_ratios(sp, dff, env):
    """Per-fold topography/climate dAUC ratio for one species (held-out)."""
    combined, clean = build_species_frame(dff, sp, env)
    X = combined[clean].values
    y = (combined["range_label"] == "invasive").astype(int).values
    dmap = domain_index_map(clean)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rng = np.random.RandomState(SEED)
    ratios = []
    for tr, te in cv.split(X, y):
        m = RandomForestClassifier(n_estimators=TREES, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        b = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        def dauc(dom):
            drops = []
            for _ in range(5):
                Xp = X[te].copy(); perm = rng.permutation(Xp.shape[0])
                Xp[:, dmap[dom]] = Xp[np.ix_(perm, dmap[dom])]
                drops.append(b - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
            return np.mean(drops)
        c = dauc("Climate")
        t = dauc("Topography")
        ratios.append(t / c if c > 1e-6 else np.nan)
    return np.array(ratios)


def main():
    cfg = load_config(CONFIG)
    dff = apply_quality_filters(load_geotraits(INPUT), cfg)
    env = get_env_variables(dff)["all_env"]
    rng = np.random.RandomState(SEED)

    per_species = {}
    print("Topography:climate dAUC ratio (per species, 5 folds):")
    for sp in SP:
        r = fold_ratios(sp, dff, env)
        per_species[sp] = r
        print(f"  {sp:<26} folds: " + " ".join(f"{x:.3f}" for x in r) + f"   mean={np.nanmean(r):.3f}")

    print("\nBootstrap 95% CI on per-GROUP mean ratio:")
    out = {}
    for name, grp in [("intercontinental", INTER), ("within-continent", WITHIN)]:
        pooled = np.concatenate([per_species[s] for s in grp])
        pooled = pooled[~np.isnan(pooled)]
        boot = [np.mean(rng.choice(pooled, len(pooled))) for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"mean": float(np.mean(pooled)), "ci_low": float(lo), "ci_high": float(hi)}
        print(f"  {name:<18} mean ratio={np.mean(pooled):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    sep = out["within-continent"]["ci_low"] > out["intercontinental"]["ci_high"]
    print(f"\n  CIs {'DO NOT overlap' if sep else 'OVERLAP'} -> "
          f"{'within-continent ratio is significantly higher' if sep else 'difference not clean'}")
    Path("results/tables/topo_climate_ratio_bootstrap.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: results/tables/topo_climate_ratio_bootstrap.json")


if __name__ == "__main__":
    main()
