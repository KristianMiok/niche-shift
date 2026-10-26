"""
r1_tables.py — assemble the manuscript tables of the revision from the R1 outputs, so
every number in Tables 2, 3 and S5 traces to one file.

  r1_Table2.csv   records, variables, DT and RF performance (stratified CV, pooled
                  out-of-fold), minority-class recall, majority baseline, RF balanced
                  accuracy under 100 km spatial blocks
  r1_Table3.csv   domain shares: DT Gini, RF Gini, RF SHAP, RF per-variable permutation,
                  and grouped permutation loss of balanced accuracy
  r1_TableS5.csv  RF and DT under stratified CV and 50/100/250/500 km spatial blocks
  r1_table_feature_sets.csv  RF and DT with all variables, without climate, without
                  climate and the cell-step variables, and climate only (stratified CV
                  and 100 km blocks): the new supplementary table of KM-4
  r1_table_ecospat.csv  ecospat results at the primary 100 km background, per species
                  and variable set: PCA-env variance, D, I, equivalency and similarity
                  tests (p for both alternatives), expansion, stability, unfilling
  r1_table_ecospat_sensitivity.csv  D, I and the dynamic indices for 50, 100 and 200 km
                  backgrounds and intersection 0.1 and 0
  r1_submitted_vs_regenerated.csv  submitted values next to the regenerated ones

Run (from conservation/):  ../.venv/bin/python src/r1_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from r1_common import DOMAINS, SPECIES, TABLES, domain_shares, ensure_dirs

ECOSPAT_SETS = ["climate", "non_climate", "all"]
FEATURE_SETS = ["all", "no_climate", "no_climate_no_cellstep", "climate_only"]

RES = Path("results/tables")
SUBMITTED = {  # Tables 2 and 3 of the R1 manuscript draft (values carried from the first submission)
    "Pontastacus leptodactylus": dict(dt_acc=0.829, rf_acc=0.835, rf_auc=0.903, dt_clim=67.5, dt_topo=27.9,
                                      rf_gini_clim=41.6, rf_gini_topo=32.2, shap_clim=49.4, shap_topo=26.2,
                                      D=0.422, spatial_acc=0.782),
    "Austropotamobius fulcisianus": dict(dt_acc=0.977, rf_acc=0.989, rf_auc=0.999, dt_clim=98.0, dt_topo=0.6,
                                         rf_gini_clim=55.4, rf_gini_topo=18.1, shap_clim=58.3, shap_topo=18.3,
                                         D=0.229, spatial_acc=0.953),
    "Austropotamobius pallipes": dict(dt_acc=0.996, rf_acc=0.997, rf_auc=1.000, dt_clim=94.4, dt_topo=3.5,
                                      rf_gini_clim=50.3, rf_gini_topo=20.2, shap_clim=50.8, shap_topo=23.9,
                                      D=0.070, spatial_acc=0.993),
}


def safe(sp):
    return sp.replace(" ", "_").lower()


def main():
    ensure_dirs()
    summ = pd.read_csv(TABLES / "r1_classification_summary.csv")
    cmp = pd.read_csv(TABLES / "r1_importance_domain_comparison.csv")
    eco_files = [TABLES / f"r1_ecospat_results_{s}.csv" for s in ECOSPAT_SETS]
    eco = pd.concat([pd.read_csv(f) for f in eco_files if f.exists()], ignore_index=True) if any(
        f.exists() for f in eco_files) else None

    def get(sp, model, cv, col, fs="all"):
        r = summ[(summ.species == sp) & (summ.model == model) & (summ.cv == cv) & (summ.feature_set == fs)]
        return float(r[col].iloc[0])

    t2, t3, s5, cmp_rows, fsets = [], [], [], [], []
    for sp in SPECIES:
        r = summ[(summ.species == sp) & (summ.feature_set == "all") & (summ.model == "RF") & (summ.cv == "stratified")].iloc[0]
        minority = "recall_native_mean" if sp == "Austropotamobius fulcisianus" else "recall_nonnative_mean"
        t2.append({
            "species": sp, "n_variables": int(r.n_variables),
            "dt_accuracy": get(sp, "DT", "stratified", "accuracy_mean"),
            "dt_balanced_accuracy": get(sp, "DT", "stratified", "balanced_accuracy_mean"),
            "rf_accuracy": r.accuracy_mean, "rf_balanced_accuracy": r.balanced_accuracy_mean,
            "rf_mcc": r.mcc_mean, "rf_auc": r.auc_mean,
            "rf_recall_native": r.recall_native_mean, "rf_recall_nonnative": r.recall_nonnative_mean,
            "minority_class": "native" if sp == "Austropotamobius fulcisianus" else "non-native",
            "rf_minority_recall": float(r[minority]),
            "majority_baseline": r.majority_baseline_mean,
            "rf_balanced_accuracy_spatial100": get(sp, "RF", "spatial_100km", "balanced_accuracy_mean"),
            "rf_balanced_accuracy_spatial100_sd": get(sp, "RF", "spatial_100km", "balanced_accuracy_sd"),
            "rf_balanced_accuracy_no_climate": get(sp, "RF", "stratified", "balanced_accuracy_mean", "no_climate"),
            "rf_balanced_accuracy_no_climate_spatial100": get(sp, "RF", "spatial_100km", "balanced_accuracy_mean", "no_climate"),
            "rf_balanced_accuracy_no_climate_no_cellstep": get(sp, "RF", "stratified", "balanced_accuracy_mean",
                                                               "no_climate_no_cellstep"),
            "rf_balanced_accuracy_no_climate_no_cellstep_spatial100": get(sp, "RF", "spatial_100km",
                                                                          "balanced_accuracy_mean",
                                                                          "no_climate_no_cellstep"),
        })
        dt = pd.read_csv(RES / f"{safe(sp)}_feature_importances.csv")
        dts = domain_shares(dt.importance.to_numpy(), dt.variable.tolist())
        c = cmp[cmp.species == sp].set_index("domain")
        row3 = {"species": sp}
        for d in DOMAINS:
            k = d.split()[0].lower()
            row3[f"dt_{k}_pct"] = dts[d]
            row3[f"rf_gini_{k}_pct"] = c.loc[d, "rf_gini_pct"]
            row3[f"rf_shap_{k}_pct"] = c.loc[d, "rf_shap_pct"]
            row3[f"rf_perm_{k}_pct"] = c.loc[d, "rf_perm_variable_pct"]
            row3[f"rf_grouped_drop_bacc_{k}"] = c.loc[d, "rf_grouped_drop_bacc"]
            row3[f"rf_grouped_drop_bacc_{k}_spatial100"] = c.loc[d, "rf_grouped_drop_bacc_spatial"]
        t3.append(row3)
        for model in ("RF", "DT"):
            for cv in ["stratified", "spatial_50km", "spatial_100km", "spatial_250km", "spatial_500km"]:
                q = summ[(summ.species == sp) & (summ.feature_set == "all") & (summ.model == model) & (summ.cv == cv)].iloc[0]
                s5.append({"species": sp, "model": model, "cv": cv, "allocations": int(q.n_seeds),
                           **{m: q[f"{m}_mean"] for m in ("accuracy", "balanced_accuracy", "mcc", "auc")},
                           "balanced_accuracy_sd": q.balanced_accuracy_sd})
        sub = SUBMITTED[sp]
        new = {"dt_acc": t2[-1]["dt_accuracy"], "rf_acc": t2[-1]["rf_accuracy"], "rf_auc": t2[-1]["rf_auc"],
               "dt_clim": dts["Climate"], "dt_topo": dts["Topography"],
               "rf_gini_clim": c.loc["Climate", "rf_gini_pct"], "rf_gini_topo": c.loc["Topography", "rf_gini_pct"],
               "shap_clim": c.loc["Climate", "rf_shap_pct"], "shap_topo": c.loc["Topography", "rf_shap_pct"],
               "spatial_acc": get(sp, "RF", "spatial_100km", "accuracy_mean")}
        notes = {"spatial_acc": "submitted: basin_id GroupKFold; regenerated: 100 km grid blocks, mean of 5 allocations",
                 "rf_auc": "regenerated: pooled out-of-fold AUC"}
        for k, v in sub.items():
            if k == "D":
                continue
            cmp_rows.append({"species": sp, "quantity": k, "submitted": v, "regenerated": new.get(k), "note": notes.get(k, "")})
        if eco is not None:
            for s in ECOSPAT_SETS:
                e = eco[(eco.species == sp) & (eco.buffer_km == 100) & (eco.variable_set == s)]
                if len(e):
                    cmp_rows.append({"species": sp, "quantity": f"D_{s}", "submitted": sub["D"],
                                     "regenerated": float(e.D.iloc[0]),
                                     "note": "submitted: occurrence-only PCA of all variables, Gaussian-smoothed "
                                             f"histogram; regenerated: ecospat PCA-env ({s} variables), 100 km background"})
        for fs in FEATURE_SETS:
            for model in ("RF", "DT"):
                for cv in ("stratified", "spatial_100km"):
                    q = summ[(summ.species == sp) & (summ.feature_set == fs) & (summ.model == model) & (summ.cv == cv)]
                    if len(q):
                        q = q.iloc[0]
                        fsets.append({"species": sp, "feature_set": fs, "n_variables": int(q.n_variables),
                                      "model": model, "cv": cv,
                                      **{m: q[f"{m}_mean"] for m in ("accuracy", "balanced_accuracy", "mcc", "auc")}})

    pd.DataFrame(t2).to_csv(TABLES / "r1_Table2.csv", index=False)
    pd.DataFrame(t3).to_csv(TABLES / "r1_Table3.csv", index=False)
    pd.DataFrame(s5).to_csv(TABLES / "r1_TableS5.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(TABLES / "r1_submitted_vs_regenerated.csv", index=False)
    pd.DataFrame(fsets).to_csv(TABLES / "r1_table_feature_sets.csv", index=False)
    if eco is not None:
        eco = eco.assign(variable_set=pd.Categorical(eco.variable_set, ECOSPAT_SETS, ordered=True),
                         species=pd.Categorical(eco.species, SPECIES, ordered=True))
        main_cols = ["variable_set", "species", "n_occ_native", "n_occ_nonnative", "n_bg_native", "n_bg_nonnative",
                     "n_variables", "pc1_var", "pc2_var", "pc1_top_loadings", "pc2_top_loadings", "D", "I",
                     "replicates", "equivalency_D_sim_mean", "equivalency_p_lower", "equivalency_p_higher",
                     "similarity_nat_to_non_D_sim_mean", "similarity_nat_to_non_p_higher", "similarity_nat_to_non_p_lower",
                     "similarity_non_to_nat_D_sim_mean", "similarity_non_to_nat_p_higher", "similarity_non_to_nat_p_lower",
                     "expansion_i01", "stability_i01", "unfilling_i01"]
        eco[eco.buffer_km == 100].sort_values(["variable_set", "species"])[main_cols].to_csv(
            TABLES / "r1_table_ecospat.csv", index=False)
        sens_cols = ["variable_set", "species", "buffer_km", "n_bg_native", "n_bg_nonnative", "pc1_var", "pc2_var",
                     "D", "I", "expansion_i01", "stability_i01", "unfilling_i01", "expansion_i0", "stability_i0",
                     "unfilling_i0"]
        eco.sort_values(["variable_set", "species", "buffer_km"])[sens_cols].to_csv(
            TABLES / "r1_table_ecospat_sensitivity.csv", index=False)
    print(pd.DataFrame(t2).round(3).to_string(index=False))
    print(pd.DataFrame(cmp_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
