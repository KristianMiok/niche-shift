# Paper 2 — niche retention in historically translocated crayfish

Code and processed outputs for:

> Pârvulescu L., Petko O. N., Miok K. *Historically translocated crayfish populations
> retain most of their native environmental niche: implications for conservation
> assessment.* Biodiversity and Conservation (in revision; submission ID
> 9468c89f-7df4-4107-a27e-78698586f69c).

Native vs non-native ("Introduced" in the GeoTraits `Status` column) occurrences of
*Pontastacus leptodactylus*, *Austropotamobius fulcisianus* and *A. pallipes*.

The repository root holds Paper 1 (Miok et al. 2026, *Journal of Biogeography*,
doi:10.1111/jbi.70332). This folder is self-contained: it carries its own copy of the
pipeline, because Paper 2 changes the species list and status labels in
`src/species_selector.py`, which would otherwise change Paper 1's published results.

## Running

All paths in the scripts are relative to this folder, and `data` is a link to the
repository's shared `data/` folder (raw data are not tracked). From the repository root:

```bash
cd conservation
../.venv/bin/python src/random_forest_shap.py --input data/raw/combined_data_true_master.csv
```

Python 3.12 with the versions pinned in `../requirements.txt`.

Input: `data/raw/combined_data_true_master.csv`, the GeoTraits master table (file of
18 February 2026; 115,191 records, 419 columns, 264,508,600 bytes, MD5
0670eba9ae3098cb6c816493d8b13f79), and `data/raw/S2.xlsx` (variable glossary). Neither
is tracked.

## Scripts and outputs (submitted version)

| Script | Outputs in `results/` | Manuscript |
|---|---|---|
| `decision_tree.py` | `tables/*_decision_rules.*`, `*_feature_importances.csv`, `dt_summary.json`; `figures/*_decision_tree.png`, `*_feature_importance.png` | Table 2 (DT), Table 3 (DT), Fig. S1, Fig. S2, Table S1 |
| `random_forest_shap.py` | `tables/*_rf_gini_importances.csv`, `*_rf_shap_importances.csv`, `rf_shap_summary.json`; `figures/*_shap_summary.png` | Table 2 (RF), Table 3 (RF), Fig. S3 |
| `cross_species_comparison.py` | `figures/cross_species_*.png`, `rf_importance_by_type_stacked.png` | Fig. 2, Fig. S4 |
| `niche_overlap_metrics.py`, `niche_overlap_conservation_figure.py` | `tables/niche_overlap_summary.*`; `figures/niche_overlap/*` | Section 3.3, Fig. 3, Fig. S7 |
| `separate_niche_models.py`, `separate_niche_table_s4.py` | `tables/*_threshold_comparison.csv`, `separate_niche_summary.json`, `table_s4_expanded*` | Section 3.4, Table S4 |
| `cv_importance_stability.py` | `tables/cv_stability/*`; `figures/cv_stability/*` | Table S2, Fig. S5 |
| `pseudoabsence_sensitivity.py` | `pseudoabsence_sensitivity/*`; `figures/pseudoabsence_sensitivity/*` | Table S3, Fig. S6 |
| `spatial_cv_rf.py` | `tables/spatial_cv_rf_*` | Table S5 |
| `cross_paper_comparison.py` | `figures/cross_paper_*.png` (uses `results/paper1_reference_*.json`) | not in manuscript |

`null_model_dichotomy.py`, `sample_size_sensitivity.py` and `eda.py` are part of the
inherited pipeline; their outputs were not regenerated for this paper.

The outputs above were regenerated in the pinned environment for the revision. Record
counts, decision trees and D/I reproduce the first submission exactly. Random-forest
values in Tables 2 and 3 move by up to about 2 percentage points, and the basin-grouped
spatial accuracies of Table S5 by up to 3.4 (earlier scikit-learn version). The separate
niche models reproduce Table S4, but four lower splits of the native *A. pallipes*
presence-background tree differ. Figures are saved at 300 dpi.

## Revision R1 (October 2026)

New scripts, all prefixed `r1_` and sharing `src/r1_common.py` (settings identical to
the submitted pipeline). Outputs go to `results/revision/tables/` and
`results/revision/figures/`.

| Script | Outputs | Revision item |
|---|---|---|
| `r1_classification.py` | `r1_classification_{metrics,summary,per_fold}.csv`, `r1_noclimate_top_variables.csv` | KM-2 (balanced accuracy, MCC, class metrics), KM-4 (models without climate, and without climate and the cell-step variables), KM-7 (grid-block spatial CV) |
| `r1_downsampling.py` | `r1_downsampling_{replicates,summary}.csv` | KM-2 (100 balanced subsamples) |
| `r1_permutation_importance.py` | `r1_permutation_{grouped,variable}.csv`, `r1_importance_domain_comparison.csv` | KM-3 |
| `r1_ecospat_prepare.py`, `r1_ecospat.R`, `r1_ecospat_figure.R` | `r1_ecospat_background_summary.csv`, `r1_ecospat_results_<set>.csv`, `r1_ecospat_null_D_<set>_<species>.csv`, `r1_ecospat_versions_<set>.json` (sets: `climate`, `non_climate`, `all`); `figures/Fig3_ecospat_niche_dynamics.png`, `FigS7_ecospat_niche_dynamics_all_variables.png`, `FigS8_ecospat_null_distributions.png` | KM-5, item 19 |
| `r1_separate_models_audit.py` | `r1_background_geography.csv`, `r1_threshold_stability_*.csv`, `r1_separate_models_geo100*.csv` | KM-6, item 11 (background limits) |
| `r1_basin_id_audit.py` | `r1_basin_id_audit.csv` | KM-7 (why basin_id was replaced) |
| `r1_uncertain_records.py` | `r1_uncertain_records.csv` | KM-8 |
| `r1_status_by_country.py` | `r1_status_by_country.csv` | item 10 |
| `r1_figures.py` | `figures/Fig2_domain_importance.png`, `figures/FigS2_top20_dt_by_domain.png` | items 16-17 |
| `r1_tables.py` | `r1_Table2.csv`, `r1_Table3.csv`, `r1_TableS5.csv`, `r1_table_feature_sets.csv`, `r1_table_ecospat.csv`, `r1_table_ecospat_sensitivity.csv`, `r1_submitted_vs_regenerated.csv` | Tables 2, 3, S5 and the new tables |

Run order (from inside `conservation/`): `r1_classification.py` (then
`--sets no_climate_no_cellstep`), `r1_permutation_importance.py`, `r1_downsampling.py`,
`r1_uncertain_records.py`, `r1_separate_models_audit.py`, `r1_basin_id_audit.py`,
`r1_status_by_country.py`, `r1_ecospat_prepare.py`,
`ECOSPAT_SET=<set> Rscript src/r1_ecospat.R` for `climate`, `non_climate` and `all`,
`Rscript src/r1_ecospat_figure.R`, `r1_figures.py`, `r1_tables.py`. The ecospat step
needs R with ecospat (4.1.4 used, 1000 replicates; 30-45 min per set on one core); its
record-level inputs in `results/revision/ecospat_input/` are not tracked, in line with
the World of Crayfish data policy, and are rebuilt by `r1_ecospat_prepare.py`.

Item 18: `decision_tree.py`, `niche_overlap_metrics.py` and `pseudoabsence_sensitivity.py`
now label the second class "non-native" in figures and rule files (outputs regenerated;
values unchanged). Internal variable and JSON key names of the inherited pipeline still
say "invasive".

## Provenance

Merged into `main` from the former branch `conservation-paper` (last commit 4d5a99f,
6 April 2026). The earlier public repository github.com/KristianMiok/niche-conservation
stops one commit before that and does not contain `spatial_cv_rf.py` (Table S5),
`separate_niche_table_s4.py` (Table S4) or the Figure 3 panel script.
