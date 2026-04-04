# Niche Reorganisation in Invasive Freshwater Crayfish

This repository accompanies the manuscript:
> **Invasion pathway predicts the axis of ecological niche reorganisation in freshwater crayfish**

## Overview

Biological invasions involve not only changes in niche position but a reorganisation of the environmental axes that distinguish species' distributions. This study tests whether the dominant axes of niche reorganisation differ systematically with invasion pathway type, using decision trees and random forests trained on ~400 hydrologically resolved environmental variables from GeoTraits.

## Study species

Five invasive freshwater crayfish species were analysed:

| Species | Invasion type | Native range | Invasive range |
|---|---|---|---|
| *Procambarus clarkii* | Intercontinental | South-central USA | Europe, Africa, Asia |
| *Faxonius limosus* | Intercontinental | Eastern North America | Europe |
| *Pacifastacus leniusculus* | Intercontinental | Pacific Northwest | Europe |
| *Faxonius virilis* | Within-continent | Central-eastern North America | Western N. America, Europe |
| *Faxonius rusticus* | Within-continent | Ohio River basin | Great Lakes, NE USA |

## Analyses

| Analysis | Script | Description |
|---|---|---|
| Data filtering | `data_loader.py`, `species_selector.py`, `data_preparation.py` | Quality filtering, species selection, variable cleaning |
| Decision tree classification | `decision_tree.py` | Native vs. invasive classification with interpretable thresholds |
| Random forest + SHAP | `random_forest_shap.py` | Ensemble robustness check with SHAP feature attributions |
| Cross-species comparison | `cross_species_comparison.py` | Variable-type importance aggregation across species |
| Separate niche models | `separate_niche_models.py` | Independent native/invasive presence–background models |
| Niche overlap metrics | `niche_overlap_metrics.py` | Schoener's D and Warren's I in PCA space |
| Null model | `null_model_dichotomy.py` | Permutation test for the pathway dichotomy |
| CV importance stability | `cv_importance_stability.py` | Feature importance variance across cross-validation folds |
| Sample-size sensitivity | `sample_size_sensitivity.py` | Subsampling test for small native samples |
| Pseudo-absence sensitivity | `pseudoabsence_sensitivity.py` | Alternative background strategy comparison |
| Variable glossary | `variable_glossary.py` | Maps coded variable names to ecological descriptions using S2 metadata |

## Project structure
```text
├── config/
│   └── species_config.yaml          # Species list, filtering thresholds
├── data/
│   ├── raw/                         # GeoTraits data + S2 glossary (not tracked)
│   ├── interim/                     # Intermediate outputs
│   └── processed/                   # Per-species analysis-ready datasets
├── results/
│   ├── figures/                     # Main figures
│   │   ├── cv_stability/            # CV fold stability plots
│   │   ├── niche_overlap/           # PCA scatters and density maps
│   │   └── pseudoabsence_sensitivity/
│   ├── tables/                      # Result tables and JSON summaries
│   │   └── cv_stability/
│   ├── null_model/
│   └── sample_size_sensitivity/
├── src/                             # All analysis scripts
├── requirements.txt
└── README.md
```

## Reproducing the analyses

### Setup
```bash
git clone https://github.com/KristianMiok/NicheReorganisation.git
cd NicheReorganisation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data

Place `combined_data_true_master.csv` and `S2.xlsx` in `data/raw/`. The GeoTraits dataset is freely available at [https://world.crayfish.ro/](https://world.crayfish.ro/).

### Run

All scripts are run from the project root with the `--input` flag:
```bash
# Core pipeline
python src/data_loader.py --input data/raw/combined_data_true_master.csv
python src/species_selector.py --input data/raw/combined_data_true_master.csv
python src/data_preparation.py --input data/raw/combined_data_true_master.csv
python src/decision_tree.py --input data/raw/combined_data_true_master.csv
python src/cross_species_comparison.py --input data/raw/combined_data_true_master.csv
python src/separate_niche_models.py --input data/raw/combined_data_true_master.csv

# Robustness analyses
python src/random_forest_shap.py --input data/raw/combined_data_true_master.csv
python src/niche_overlap_metrics.py --input data/raw/combined_data_true_master.csv
python src/null_model_dichotomy.py --input data/raw/combined_data_true_master.csv
python src/cv_importance_stability.py --input data/raw/combined_data_true_master.csv
python src/sample_size_sensitivity.py --input data/raw/combined_data_true_master.csv
python src/pseudoabsence_sensitivity.py --input data/raw/combined_data_true_master.csv
```

## Requirements

- Python ≥ 3.10
- pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, shap, pyyaml, openpyxl

See `requirements.txt` for pinned versions.

## Data availability

The Global Crayfish Database of Geospatial Traits is freely available through the World of Crayfish® platform ([https://world.crayfish.ro/](https://world.crayfish.ro/)). Raw occurrence coordinates are withheld per WoC® data policy; the unique WoC® identifier enables authorised re-linkage.

## License

[To be added]
