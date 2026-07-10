"""
pathway_refit.py — Task (d): decisive test for Reviewer 2, point 8.

Refit the primary native-vs-invaded classifier for the two species whose
invaded records mix pathway types (continent audit, this session):

  * Faxonius virilis (classified within-continent): drop the European invaded
    records -> refit on North-America-only invaded + all native.
    Q: does topographic dominance (~64%) survive? If yes, it was NOT an
    artefact of the intercontinental contamination.
  * Procambarus clarkii (classified intercontinental): drop the North-American
    invaded records -> refit on intercontinental invaded (EU/AS/AF/SA) + native.
    Q: does climate dominance survive (or strengthen)?

METHOD (apples-to-apples): baseline reuses the exact production pipeline, so
baseline per-domain importances MUST reproduce the manuscript (self-check).
The refit FREEZES that feature set and removes only wrong-continent invaded
rows, so any change is attributable to the records, not to variable selection.
Continent per invaded record = offline reverse_geocoder on lat_or/long_or
(deterministic; 0 Americas/Old-World mismatches in the audit). No GDAL.

Run:  python src/pathway_refit.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import (
    load_geotraits, load_config, get_env_variables, COL_SPECIES, COL_STATUS,
)
from species_selector import apply_quality_filters, STATUS_NATIVE, STATUS_ALIEN
from data_preparation import (
    handle_missing_values, remove_constant_variables, remove_highly_correlated,
)
from decision_tree import (
    train_final_tree, extract_feature_importances, cross_validate_tree,
)
from random_forest_shap import train_and_evaluate_rf, get_gini_importances
from cross_species_comparison import classify_variable

INPUT = "data/raw/combined_data_true_master.csv"
CONFIG = "config/species_config.yaml"
LAT, LON = "lat_or", "long_or"
OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)
DOMAINS = ["Climate", "Topography", "Soil", "Land Cover"]
CONT_NAME = {"NA": "North America", "SA": "South America", "EU": "Europe",
             "AS": "Asia", "AF": "Africa", "OC": "Oceania", "AN": "Antarctica"}

REFIT = {
    "Faxonius virilis": {
        "keep_invaded": {"North America"},
        "label": "NA-only invaded (dropped European)",
        "focus": "Topography",
    },
    "Procambarus clarkii": {
        "keep_invaded": {"Europe", "Asia", "Africa", "South America"},
        "label": "intercontinental-only invaded (dropped North American)",
        "focus": "Climate",
    },
}
_SHAP_WARNED = False


def build_species_frame(df_filtered, species, env_vars):
    """Reproduce the production per-species dataset EXACTLY (baseline)."""
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


def continents_for(df):
    coords = list(zip(df[LAT].astype(float), df[LON].astype(float)))
    import reverse_geocoder as rg
    from pycountry_convert import country_alpha2_to_continent_code
    out = []
    for r in rg.search(coords, mode=1):
        cc = (r.get("cc") or "").upper()
        try:
            out.append(CONT_NAME.get(country_alpha2_to_continent_code(cc), "Unknown"))
        except Exception:
            out.append("Unknown")
    return out


def per_domain_pct(imp, col):
    """Same normalisation as the manuscript figure (grand total incl. 'Other')."""
    d = imp.copy()
    d["type"] = d["variable"].apply(classify_variable)
    by = d.groupby("type")[col].sum()
    tot = by.sum()
    return {t: float(by.get(t, 0.0) / tot * 100) for t in DOMAINS + ["Other"]}


def shap_domain(rf, X, clean):
    global _SHAP_WARNED
    try:
        from random_forest_shap import compute_shap_values
        shap_imp, _, _ = compute_shap_values(rf, X, clean)
        return per_domain_pct(shap_imp, "mean_abs_shap")
    except Exception as e:
        if not _SHAP_WARNED:
            print(f"  [i] SHAP skipped ({type(e).__name__}); `uv pip install shap` to include it.")
            _SHAP_WARNED = True
        return None


def fit_block(combined, clean, label):
    X = combined[clean].values
    y = (combined["range_label"] == "invasive").astype(int).values

    dt = train_final_tree(X, y, clean, max_depth=5, random_state=42)
    dt_dom = per_domain_pct(extract_feature_importances(dt, clean), "importance")
    dt_cv = cross_validate_tree(X, y, max_depth=5)

    rf, rf_cv = train_and_evaluate_rf(X, y, clean)
    rf_dom = per_domain_pct(get_gini_importances(rf, clean), "gini_importance")

    return {
        "label": label,
        "n_native": int((y == 0).sum()), "n_invasive": int((y == 1).sum()),
        "dt_acc": dt_cv["accuracy_mean"], "dt_auc": dt_cv["roc_auc_mean"],
        "rf_acc": rf_cv["accuracy_mean"], "rf_auc": rf_cv["roc_auc_mean"],
        "dt_domain": dt_dom, "rf_domain": rf_dom,
        "shap_domain": shap_domain(rf, X, clean),
    }


def _print_block(b):
    print(f"\n  --- {b['label']} ---")
    print(f"    n: {b['n_native']} native, {b['n_invasive']} invaded")
    print(f"    DT  acc={b['dt_acc']:.3f} AUC={b['dt_auc']:.3f}   "
          f"RF acc={b['rf_acc']:.3f} AUC={b['rf_auc']:.3f}")
    dd, rd = b["dt_domain"], b["rf_domain"]
    print("    DT   %: " + "  ".join(f"{t.split()[0][:4]}={dd[t]:5.1f}" for t in DOMAINS))
    print("    RF   %: " + "  ".join(f"{t.split()[0][:4]}={rd[t]:5.1f}" for t in DOMAINS))
    if b["shap_domain"]:
        sd = b["shap_domain"]
        print("    SHAP %: " + "  ".join(f"{t.split()[0][:4]}={sd[t]:5.1f}" for t in DOMAINS))


def main():
    config = load_config(CONFIG)
    df = load_geotraits(INPUT)
    dff = apply_quality_filters(df, config)
    env = get_env_variables(dff)["all_env"]
    print(f"Filtered master: {dff.shape[0]} rows | env vars: {len(env)}")

    summary = {}
    for sp, spec in REFIT.items():
        print("\n" + "=" * 74)
        print(f"{sp}  —  task (d) refit  [{spec['label']}]")
        print("=" * 74)

        combined, clean = build_species_frame(dff, sp, env)
        print(f"  cleaned feature set: {len(clean)} vars (frozen for baseline AND refit)")

        inv = combined[combined["range_label"] == "invasive"].copy()
        inv["_cont"] = continents_for(inv)
        vc = inv["_cont"].value_counts()
        print("  invaded by continent (this modeling set): "
              + ", ".join(f"{k}={v}" for k, v in vc.items())
              + "   [audit: virilis NA=457/EU=43 ; clarkii NA=2249]")
        am_geo = inv["_cont"].isin(["North America", "South America"])
        am_lon = inv[LON].astype(float) < -30
        print(f"  geo/lon Americas cross-check mismatches: {int((am_geo != am_lon).sum())} of {len(inv)}")

        keep = spec["keep_invaded"]
        drop_idx = inv.index[~inv["_cont"].isin(keep)]
        print(f"  dropping {len(drop_idx)} invaded rows outside {sorted(keep)}")
        combined_refit = combined.drop(index=drop_idx).reset_index(drop=True)

        base = fit_block(combined, clean, "baseline (all invaded)")
        refit = fit_block(combined_refit, clean, spec["label"])
        _print_block(base)
        _print_block(refit)

        f = spec["focus"]
        print(f"\n  >>> FOCUS [{f}]   "
              f"DT {base['dt_domain'][f]:.1f}% -> {refit['dt_domain'][f]:.1f}%   |   "
              f"RF {base['rf_domain'][f]:.1f}% -> {refit['rf_domain'][f]:.1f}%")

        summary[sp] = {"n_dropped": int(len(drop_idx)), "focus": f,
                       "baseline": base, "refit": refit}

    out = OUTDIR / "pathway_refit_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
