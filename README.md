# Niche Reorganisation in Invasive Freshwater Crayfish

This repository accompanies the manuscript:
> **Beyond niche overlap: axis-specific reorganisation of environmental space in freshwater crayfish invasions**
>
> *Journal of Biogeography* (accepted 2026), manuscript 2253676.

The repository root is this paper (Paper 1). Two companion papers that use the same
pipeline live in their own self-contained folders and leave the root untouched:

| Folder | Paper |
|---|---|
| [`conservation/`](conservation/) | Paper 2 — niche retention in historically translocated crayfish (*Biodiversity and Conservation*, in revision) |
| [`lineage/`](lineage/) | Paper 3 — environmental structure of *Austropotamobius torrentium* lineages |

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


Python 3.12. Install pinned dependencies with `pip install -r requirements.txt`.

The analysis is version-sensitive: random forest tie-breaking differs between
scikit-learn releases, which shifts reported values in the third decimal. The results in
the paper were produced with **scikit-learn 1.9.0** and **NumPy 2.2.6**, as pinned in
`requirements.txt` and stated in Methods 2.5. Reproducing the published numbers exactly
requires those versions; other versions reproduce the analysis but not every decimal.

## Data availability


The underlying dataset is the Global Crayfish Database of Geospatial Traits, published as
Petko et al. (2026), *Network-aware environmental trait database for global freshwater
crayfish*, **Scientific Data**, DOI [10.1038/s41597-026-07845-5](https://doi.org/10.1038/s41597-026-07845-5),
and available through the World of Crayfish® platform ([world.crayfish.ro](https://world.crayfish.ro/)).
Raw occurrence coordinates are withheld under WoC® data policy; the unique WoC® identifier
enables authorised re-linkage.

This repository contains the analysis code and the processed outputs it produces
(`results/tables/`, `results/figures/`). An archived, citable version of the repository is
deposited on Zenodo; the DOI appears in the data accessibility statement of the paper.
To reproduce the analyses, place the master table at `data/raw/combined_data_true_master.csv`.

## License


MIT License. See [LICENSE](LICENSE).

