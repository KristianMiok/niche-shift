"""
r1_common.py — shared settings and helpers for the Biodiversity and Conservation
revision (R1) analyses of Paper 2.

Everything here reuses the submitted pipeline unchanged (quality filters, per-species
variable cleaning, model settings), so the R1 analyses are directly comparable with
the submitted Tables 2-3 and S1-S5:

  records   : Native vs Introduced, High accuracy, snapping <= 1 km,
              one record per species x status x stream segment (species_selector.py)
  variables : per species, drop > 30 % missing, drop records > 50 % missing,
              median-impute the rest, drop constants, drop one of each |r| > 0.98 pair
              (data_preparation.py)
  RF        : 500 trees, class_weight="balanced", max_features="sqrt",
              min_samples_leaf=1, bootstrap, Gini criterion, random_state=42
  DT        : CART, Gini, max_depth=5, class_weight="balanced", random_state=42
  CV        : StratifiedKFold(5, shuffle=True, random_state=42)  (as submitted)

New in R1: spatial blocking on a latitude-corrected grid (as in Paper 1,
src/spatial_block_cv.py at the repository root), because the GeoTraits basin_id
column is not spatially coherent for part of the records (see r1_basin_id_audit.py).
Positive class (1) = non-native, 0 = native.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_config, load_geotraits, get_env_variables, COL_SPECIES, COL_STATUS
from species_selector import apply_quality_filters, STATUS_NATIVE, STATUS_ALIEN, STUDY_SPECIES
from data_preparation import handle_missing_values, remove_constant_variables, remove_highly_correlated
from cross_species_comparison import classify_variable

INPUT = "data/raw/combined_data_true_master.csv"
CONFIG = "config/species_config.yaml"
OUT = Path("results/revision")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SEED = 42
DOMAINS = ["Climate", "Topography", "Soil", "Land Cover"]
LAT, LON = "lat_or", "long_or"
SPECIES = list(STUDY_SPECIES)
SHORT = {
    "Pontastacus leptodactylus": "P. leptodactylus",
    "Austropotamobius fulcisianus": "A. fulcisianus",
    "Austropotamobius pallipes": "A. pallipes",
}
PRIMARY_BLOCK_KM = 100
BLOCK_SIZES_KM = [50, 100, 250, 500]


def ensure_dirs():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def load_filtered(cache: str | None = None) -> pd.DataFrame:
    """Quality-filtered records of all species (as in the submitted pipeline)."""
    if cache and Path(cache).exists():
        return pd.read_pickle(cache)
    df = apply_quality_filters(load_geotraits(INPUT), load_config(CONFIG))
    if cache:
        df.to_pickle(cache)
    return df


def env_columns(df: pd.DataFrame) -> list[str]:
    return get_env_variables(df)["all_env"]


def build_species_frame(dff: pd.DataFrame, species: str, env_vars: list[str]):
    """Pooled native + non-native records of one species with the cleaned variable set.

    Identical to the submitted random_forest_shap.prepare_species_data / Paper 1
    grouped_permutation.build_species_frame.
    """
    sp = dff[dff[COL_SPECIES] == species].copy()
    native = sp[sp[COL_STATUS] == STATUS_NATIVE]
    nonnative = sp[sp[COL_STATUS] == STATUS_ALIEN]
    combined = pd.concat([native, nonnative], ignore_index=True)
    combined, clean = handle_missing_values(combined, env_vars)
    clean = remove_constant_variables(combined, clean)
    clean = remove_highly_correlated(combined, clean, threshold=0.98)
    combined["range_label"] = np.where(combined[COL_STATUS] == STATUS_NATIVE, "native", "non-native")
    return combined.reset_index(drop=True), clean


def xy(combined: pd.DataFrame, cols: list[str]):
    X = combined[cols].to_numpy(dtype=float)
    y = (combined["range_label"] == "non-native").astype(int).to_numpy()
    return X, y


def domain_of(var: str) -> str:
    return classify_variable(var)


def domain_index_map(cols: list[str]) -> dict[str, np.ndarray]:
    m: dict[str, list[int]] = {}
    for j, v in enumerate(cols):
        m.setdefault(domain_of(v), []).append(j)
    return {d: np.array(ix) for d, ix in m.items() if d in DOMAINS}


def domain_shares(values: np.ndarray, cols: list[str]) -> dict[str, float]:
    """Sum non-negative importances by domain, as % of the four-domain total."""
    s = pd.Series(np.clip(values, 0, None), index=cols)
    by = {d: float(s[[v for v in cols if domain_of(v) == d]].sum()) for d in DOMAINS}
    tot = sum(by.values())
    return {d: (100 * by[d] / tot if tot > 0 else np.nan) for d in DOMAINS}


def make_rf(seed: int = SEED) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def make_dt(seed: int = SEED, depth: int = 5) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=seed)


# ---------------------------------------------------------------------------
# Spatial blocks (identical construction to Paper 1, src/spatial_block_cv.py)
# ---------------------------------------------------------------------------

def assign_cells(lat, lon, block_km):
    """Latitude-corrected equirectangular grid -> integer cell code per record.

    Copied verbatim from Paper 1 (src/spatial_block_cv.py) so both papers use the
    same blocks.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat = block_km / 111.0
    lat_band = np.floor(lat / dlat).astype(np.int64)
    band_center = (lat_band + 0.5) * dlat
    coslat = np.clip(np.cos(np.radians(band_center)), 0.05, None)
    dlon = block_km / (111.0 * coslat)
    lon_band = np.floor(lon / dlon).astype(np.int64)
    a = lat_band - lat_band.min()
    b = lon_band - lon_band.min()
    paired = a * (b.max() + 1) + b
    codes, _ = pd.factorize(paired)
    return codes


def stratified_splits(X, y, seed=SEED, k=5):
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y))


def spatial_splits(X, y, lat, lon, block_km, seed=SEED, k=5):
    groups = assign_cells(lat, lon, block_km)
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y, groups))


# ---------------------------------------------------------------------------
# Metrics on pooled out-of-fold predictions
# ---------------------------------------------------------------------------

def oof_predict(model, X, y, splits):
    """Fit per fold, return pooled out-of-fold class predictions and P(non-native)."""
    pred = np.full(len(y), -1)
    prob = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    fitted = []
    for f, (tr, te) in enumerate(splits):
        fold_id[te] = f
        if len(np.unique(y[tr])) < 2:
            # all records of one class sit in the held-out cells: no two-class model can
            # be trained; these records stay unscored (reported as n_unscored)
            fitted.append(None)
            continue
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        prob[te] = m.predict_proba(X[te])[:, 1]
        fitted.append(m)
    return pred, prob, fold_id, fitted


def metrics(y, pred, prob) -> dict:
    n_all = len(y)
    ok = pred >= 0
    y, pred, prob = y[ok], pred[ok], prob[ok]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)), "n_unscored": int(n_all - len(y)),
        "n_native": int((y == 0).sum()), "n_nonnative": int((y == 1).sum()),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "precision_native": precision_score(y, pred, pos_label=0, zero_division=0),
        "recall_native": recall_score(y, pred, pos_label=0, zero_division=0),
        "f1_native": f1_score(y, pred, pos_label=0, zero_division=0),
        "precision_nonnative": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall_nonnative": recall_score(y, pred, pos_label=1, zero_division=0),
        "f1_nonnative": f1_score(y, pred, pos_label=1, zero_division=0),
        "tn_native_as_native": int(tn), "fp_native_as_nonnative": int(fp),
        "fn_nonnative_as_native": int(fn), "tp_nonnative_as_nonnative": int(tp),
        "majority_baseline": max((y == 0).mean(), (y == 1).mean()),
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def per_fold_metrics(y, pred, prob, fold_id) -> pd.DataFrame:
    rows = []
    for f in np.unique(fold_id):
        ix = fold_id == f
        if not (pred[ix] >= 0).any():
            rows.append({"fold": int(f), "n": 0, "n_unscored": int(ix.sum()),
                         "single_class_test_fold": bool(len(np.unique(y[ix])) < 2)})
            continue
        r = metrics(y[ix], pred[ix], prob[ix])
        r["fold"] = int(f)
        r["single_class_test_fold"] = bool(len(np.unique(y[ix])) < 2)
        rows.append(r)
    return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
