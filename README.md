# Crayfish Niche Shift Analysis

Machine-learning analysis of **native vs. invasive niche reorganisation** in freshwater crayfish using network-aware environmental predictors from the **Global Crayfish Database of Geospatial Traits (GeoTraits)**.

This repository accompanies the manuscript:

**Invasion pathway predicts the axis of ecological niche reorganisation in freshwater crayfish**

The project tests whether native and invasive ranges are differentiated along different environmental axes, and whether those axes depend on invasion pathway type (**intercontinental vs. within-continent**).

## Main study system

Five invasive crayfish species were analysed after quality filtering:

- **Procambarus clarkii**
- **Faxonius limosus**
- **Pacifastacus leniusculus**
- **Faxonius virilis**
- **Faxonius rusticus**

These include:

- **Intercontinental invaders**: *P. clarkii*, *F. limosus*, *P. leniusculus*
- **Within-continent invaders**: *F. virilis*, *F. rusticus*

## Core analyses implemented

The repository contains the full workflow for:

1. **Data filtering and species selection**
2. **Decision-tree classification** of native vs. invasive occurrences
3. **Separate niche models** for native-only and invasive-only ranges
4. **Random forest + SHAP robustness analysis**
5. **Null model / permutation test** for the pathway dichotomy
6. **Classical niche overlap metrics** (Schoener’s D, Warren’s I)
7. **Cross-validation importance stability**
8. **Sample-size sensitivity analysis**
9. **Pseudo-absence sensitivity analysis** for separate niche models

## Project structure

```text
crayfish-niche-shift/
├── config/
│   └── species_config.yaml
├── data/
│   ├── raw/                     # Original GeoTraits exports (not tracked in git)
│   ├── interim/                 # Intermediate processing outputs
│   └── processed/               # Species-level prepared datasets
├── results/
│   ├── figures/                 # Main and supplementary figures
│   ├── tables/                  # Main and supplementary result tables
│   ├── null_model/              # Permutation-test outputs
│   ├── pseudoabsence_sensitivity/
│   └── sample_size_sensitivity/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── species_selector.py
│   ├── data_preparation.py
│   ├── eda.py
│   ├── decision_tree.py
│   ├── cross_species_comparison.py
│   ├── separate_niche_models.py
│   ├── variable_glossary.py
│   ├── random_forest_shap.py
│   ├── null_model_dichotomy.py
│   ├── niche_overlap_metrics.py
│   ├── cv_importance_stability.py
│   ├── sample_size_sensitivity.py
│   └── pseudoabsence_sensitivity.py
├── requirements.txt
├── .gitignore
└── README.md
