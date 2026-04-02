# Crayfish Niche Shift Analysis

Comparative analysis of ecological niche thresholds between native and invasive
ranges of freshwater crayfish, using machine learning decision rules derived from
the Global Crayfish Database of Geospatial Traits (GeoTraits).

## Project Structure

```
crayfish-niche-shift/
├── config/
│   └── species_config.yaml      # Species selection criteria & parameters
├── data/
│   ├── raw/                     # Original GeoTraits exports (not tracked in git)
│   ├── interim/                 # Intermediate processing outputs
│   └── processed/               # Final analysis-ready datasets
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load and validate raw GeoTraits data
│   ├── species_selector.py      # Filter species by inclusion criteria
│   ├── data_preparation.py      # Prepare native/invasive splits, clean, encode
│   └── eda.py                   # Exploratory data analysis & summary stats
├── notebooks/
│   └── 01_data_exploration.py   # Initial data exploration (run as script or notebook)
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone and set up environment
git clone <repo-url>
cd crayfish-niche-shift
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data

Place `combined_data_true_master.csv` in `data/raw/`. This is the Full Integrated
Dataset from the Global Crayfish Database of Geospatial Traits, containing
occurrence-level records with ~400 environmental variables.

This file is not tracked in git due to size.

## Usage

```bash
# Step 1: Load data and see species/status counts
python src/data_loader.py --input data/raw/combined_data_true_master.csv

# Step 2: Select candidate species meeting inclusion criteria
python src/species_selector.py --input data/raw/combined_data_true_master.csv

# Step 3: Prepare analysis-ready datasets per species
python src/data_preparation.py --input data/raw/combined_data_true_master.csv

# Step 4: Run EDA for a specific species
python src/eda.py --species "Procambarus clarkii" \
                  --data data/processed/procambarus_clarkii_combined.csv \
                  --vars data/processed/procambarus_clarkii_env_vars.txt
```
