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

## Provenance

Merged into `main` from the former branch `conservation-paper` (last commit 4d5a99f,
6 April 2026). The earlier public repository github.com/KristianMiok/niche-conservation
stops one commit before that and does not contain `spatial_cv_rf.py` (Table S5),
`separate_niche_table_s4.py` (Table S4) or the Figure 3 panel script.
