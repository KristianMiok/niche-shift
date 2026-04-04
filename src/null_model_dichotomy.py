"""
null_model_dichotomy.py

Permutation test for the intercontinental vs. within-continent dichotomy.

Question:
Is the observed climate-vs-topography contrast stronger than expected
if invasion-type labels are randomly assigned across the five species?

We use decision-tree and/or random-forest variable-type importance summaries
and compute an observed statistic:

    mean(climate - topography in intercontinental species)
    -
    mean(climate - topography in within-continent species)

Then we shuffle the invasion-type labels across species many times
(default 999 permutations) and recompute the statistic.

Usage:
    python src/null_model_dichotomy.py
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

INTERCONTINENTAL = [
    "Procambarus clarkii",
    "Faxonius limosus",
    "Pacifastacus leniusculus",
]

WITHIN_CONTINENT = [
    "Faxonius virilis",
    "Faxonius rusticus",
]

SPECIES_ORDER = INTERCONTINENTAL + WITHIN_CONTINENT


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_rf_summary(
    filepath: str = "results/tables/rf_shap_summary.json",
    importance_key: str = "gini",
) -> pd.DataFrame:
    """
    Load RF summary JSON and extract climate/topography percentages.

    importance_key:
        - "gini"  -> use RF Gini importance by type
        - "shap"  -> use RF SHAP importance by type
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"RF summary file not found: {filepath}\n"
            f"Run random_forest_shap.py first."
        )

    with open(path, "r") as f:
        data = json.load(f)

    rows = []
    for species, res in data.items():
        # Try common nested keys
        by_type = None

        if importance_key == "gini":
            by_type = (
                res.get("importance_by_type_gini")
                or res.get("gini_by_type")
                or res.get("importance_by_type")
            )
        elif importance_key == "shap":
            by_type = (
                res.get("importance_by_type_shap")
                or res.get("shap_by_type")
            )

        if by_type is None:
            raise KeyError(
                f"Could not find '{importance_key}' importance-by-type block "
                f"for species: {species}"
            )

        climate = float(by_type.get("Climate", 0.0))
        topo = float(by_type.get("Topography", 0.0))
        soil = float(by_type.get("Soil", 0.0))
        land = float(by_type.get("Land Cover", 0.0))

        rows.append({
            "species": species,
            "climate": climate,
            "topography": topo,
            "soil": soil,
            "land_cover": land,
            "climate_minus_topography": climate - topo,
            "climate_to_topography_ratio": climate / topo if topo > 0 else np.nan,
        })

    df = pd.DataFrame(rows)

    # Keep only focal species, in the intended order
    df = df[df["species"].isin(SPECIES_ORDER)].copy()
    df["species"] = pd.Categorical(df["species"], categories=SPECIES_ORDER, ordered=True)
    df = df.sort_values("species").reset_index(drop=True)

    # Add true labels
    df["invasion_type"] = df["species"].apply(
        lambda s: "intercontinental" if s in INTERCONTINENTAL else "within_continent"
    )

    return df


def compute_observed_statistic(
    df: pd.DataFrame,
    metric_col: str = "climate_minus_topography",
) -> float:
    """
    Observed statistic:
        mean(metric in intercontinental)
        -
        mean(metric in within_continent)
    """
    inter = df[df["invasion_type"] == "intercontinental"][metric_col].mean()
    within = df[df["invasion_type"] == "within_continent"][metric_col].mean()
    return float(inter - within)


def permutation_test(
    df: pd.DataFrame,
    metric_col: str = "climate_minus_topography",
    n_perm: int = 999,
    random_state: int = 42,
) -> dict:
    """
    Permute invasion-type labels across species while preserving group sizes:
    3 intercontinental, 2 within-continent.
    """
    rng = np.random.default_rng(random_state)

    observed = compute_observed_statistic(df, metric_col=metric_col)

    species = df["species"].tolist()
    values = dict(zip(df["species"], df[metric_col]))

    perm_stats = []
    for _ in range(n_perm):
        shuffled = species.copy()
        rng.shuffle(shuffled)

        perm_inter = set(shuffled[:3])   # preserve 3 vs 2 split
        perm_within = set(shuffled[3:])

        inter_mean = np.mean([values[s] for s in perm_inter])
        within_mean = np.mean([values[s] for s in perm_within])

        perm_stats.append(float(inter_mean - within_mean))

    perm_stats = np.array(perm_stats)

    # two-sided empirical p-value
    p_two_sided = (np.sum(np.abs(perm_stats) >= abs(observed)) + 1) / (n_perm + 1)

    # one-sided empirical p-value for "observed is larger than random"
    p_one_sided = (np.sum(perm_stats >= observed) + 1) / (n_perm + 1)

    return {
        "observed_statistic": float(observed),
        "permutation_mean": float(np.mean(perm_stats)),
        "permutation_sd": float(np.std(perm_stats, ddof=1)),
        "p_two_sided": float(p_two_sided),
        "p_one_sided": float(p_one_sided),
        "perm_stats": perm_stats,
    }


def plot_permutation_distribution(
    perm_stats: np.ndarray,
    observed: float,
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(perm_stats, bins=20, edgecolor="black")
    plt.axvline(observed, linestyle="--", linewidth=2, label=f"Observed = {observed:.2f}")
    plt.xlabel("Permutation statistic")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_summary_text(
    model_name: str,
    metric_name: str,
    result: dict,
) -> str:
    return (
        f"{model_name} | metric = {metric_name}\n"
        f"Observed statistic: {result['observed_statistic']:.4f}\n"
        f"Permutation mean:   {result['permutation_mean']:.4f}\n"
        f"Permutation SD:     {result['permutation_sd']:.4f}\n"
        f"Empirical p (one-sided): {result['p_one_sided']:.4f}\n"
        f"Empirical p (two-sided): {result['p_two_sided']:.4f}\n"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    out_dir = Path("results/null_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1) RF Gini
    # -------------------------------------------------------------
    df_gini = load_rf_summary(
        filepath="results/tables/rf_shap_summary.json",
        importance_key="gini",
    )

    print("\n" + "=" * 80)
    print("RANDOM FOREST (GINI) - SPECIES LEVEL INPUT")
    print("=" * 80)
    print(df_gini[[
        "species", "invasion_type", "climate", "topography",
        "climate_minus_topography", "climate_to_topography_ratio"
    ]].to_string(index=False))

    res_gini_diff = permutation_test(
        df_gini,
        metric_col="climate_minus_topography",
        n_perm=999,
        random_state=42,
    )

    print("\n" + "=" * 80)
    print("NULL MODEL: RF GINI | CLIMATE - TOPOGRAPHY")
    print("=" * 80)
    print(make_summary_text(
        model_name="RF Gini",
        metric_name="climate_minus_topography",
        result=res_gini_diff
    ))

    plot_permutation_distribution(
        res_gini_diff["perm_stats"],
        res_gini_diff["observed_statistic"],
        title="Null model: RF Gini | Climate - Topography",
        output_path=str(out_dir / "rf_gini_climate_minus_topography.png"),
    )

    res_gini_ratio = permutation_test(
        df_gini,
        metric_col="climate_to_topography_ratio",
        n_perm=999,
        random_state=42,
    )

    print("\n" + "=" * 80)
    print("NULL MODEL: RF GINI | CLIMATE / TOPOGRAPHY RATIO")
    print("=" * 80)
    print(make_summary_text(
        model_name="RF Gini",
        metric_name="climate_to_topography_ratio",
        result=res_gini_ratio
    ))

    plot_permutation_distribution(
        res_gini_ratio["perm_stats"],
        res_gini_ratio["observed_statistic"],
        title="Null model: RF Gini | Climate / Topography ratio",
        output_path=str(out_dir / "rf_gini_climate_to_topography_ratio.png"),
    )

    # -------------------------------------------------------------
    # 2) RF SHAP
    # -------------------------------------------------------------
    df_shap = load_rf_summary(
        filepath="results/tables/rf_shap_summary.json",
        importance_key="shap",
    )

    print("\n" + "=" * 80)
    print("RANDOM FOREST (SHAP) - SPECIES LEVEL INPUT")
    print("=" * 80)
    print(df_shap[[
        "species", "invasion_type", "climate", "topography",
        "climate_minus_topography", "climate_to_topography_ratio"
    ]].to_string(index=False))

    res_shap_diff = permutation_test(
        df_shap,
        metric_col="climate_minus_topography",
        n_perm=999,
        random_state=42,
    )

    print("\n" + "=" * 80)
    print("NULL MODEL: RF SHAP | CLIMATE - TOPOGRAPHY")
    print("=" * 80)
    print(make_summary_text(
        model_name="RF SHAP",
        metric_name="climate_minus_topography",
        result=res_shap_diff
    ))

    plot_permutation_distribution(
        res_shap_diff["perm_stats"],
        res_shap_diff["observed_statistic"],
        title="Null model: RF SHAP | Climate - Topography",
        output_path=str(out_dir / "rf_shap_climate_minus_topography.png"),
    )

    res_shap_ratio = permutation_test(
        df_shap,
        metric_col="climate_to_topography_ratio",
        n_perm=999,
        random_state=42,
    )

    print("\n" + "=" * 80)
    print("NULL MODEL: RF SHAP | CLIMATE / TOPOGRAPHY RATIO")
    print("=" * 80)
    print(make_summary_text(
        model_name="RF SHAP",
        metric_name="climate_to_topography_ratio",
        result=res_shap_ratio
    ))

    plot_permutation_distribution(
        res_shap_ratio["perm_stats"],
        res_shap_ratio["observed_statistic"],
        title="Null model: RF SHAP | Climate / Topography ratio",
        output_path=str(out_dir / "rf_shap_climate_to_topography_ratio.png"),
    )

    # -------------------------------------------------------------
    # Save compact outputs
    # -------------------------------------------------------------
    summary = {
        "rf_gini_climate_minus_topography": {
            "observed_statistic": res_gini_diff["observed_statistic"],
            "permutation_mean": res_gini_diff["permutation_mean"],
            "permutation_sd": res_gini_diff["permutation_sd"],
            "p_one_sided": res_gini_diff["p_one_sided"],
            "p_two_sided": res_gini_diff["p_two_sided"],
        },
        "rf_gini_climate_to_topography_ratio": {
            "observed_statistic": res_gini_ratio["observed_statistic"],
            "permutation_mean": res_gini_ratio["permutation_mean"],
            "permutation_sd": res_gini_ratio["permutation_sd"],
            "p_one_sided": res_gini_ratio["p_one_sided"],
            "p_two_sided": res_gini_ratio["p_two_sided"],
        },
        "rf_shap_climate_minus_topography": {
            "observed_statistic": res_shap_diff["observed_statistic"],
            "permutation_mean": res_shap_diff["permutation_mean"],
            "permutation_sd": res_shap_diff["permutation_sd"],
            "p_one_sided": res_shap_diff["p_one_sided"],
            "p_two_sided": res_shap_diff["p_two_sided"],
        },
        "rf_shap_climate_to_topography_ratio": {
            "observed_statistic": res_shap_ratio["observed_statistic"],
            "permutation_mean": res_shap_ratio["permutation_mean"],
            "permutation_sd": res_shap_ratio["permutation_sd"],
            "p_one_sided": res_shap_ratio["p_one_sided"],
            "p_two_sided": res_shap_ratio["p_two_sided"],
        },
    }

    with open(out_dir / "null_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    df_gini.to_csv(out_dir / "species_level_input_rf_gini.csv", index=False)
    df_shap.to_csv(out_dir / "species_level_input_rf_shap.csv", index=False)

    print("\nSaved outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()