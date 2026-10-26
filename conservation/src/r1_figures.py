"""
r1_figures.py — R1 figures (items 16-18 of the revision list), 300 dpi PNG.

  Fig 2   (A) decision-tree Gini importance by domain; (B) random forest Gini and SHAP
          importance by domain; (C) random forest grouped
          permutation importance (loss of balanced accuracy when a whole domain is
          shuffled; filled = stratified CV, open = 100 km spatial blocks)
  Fig S2  top 20 decision-tree variables per species, bars coloured by domain
  Fig S4  decision-tree Gini importance across species (heatmap; top 15 variables per
          species with importance >= 0.005), domain shown as a colour strip
  Fig S5  cross-validation stability of domain-level random-forest Gini importance:
          mean +/- SD over the 5 stratified folds, one PNG per species and a 3-panel
          version (inputs: results/tables/cv_stability/*_type_summary.csv)

Domain colours are a colour-vision-deficiency-safe set (validated adjacent and all
pairs; red/blue/green semantics of the submitted figures kept, soil moved from brown
to ochre). Identity never rests on colour alone: legend plus direct labels.

Inputs: results/tables/*_feature_importances.csv (DT), *_rf_gini_importances.csv,
*_rf_shap_importances.csv, results/revision/tables/r1_importance_domain_comparison.csv,
r1_permutation_grouped.csv.

Run (from conservation/):  ../.venv/bin/python src/r1_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch

from r1_common import DOMAINS, FIGURES, SHORT, SPECIES, TABLES, domain_of, domain_shares, ensure_dirs
from variable_glossary import load_glossary, make_label

DOMAIN_COLORS = {"Climate": "#e34948", "Topography": "#2a78d6", "Soil": "#eda100", "Land Cover": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
RES = Path("results/tables")


def safe(sp):
    return sp.replace(" ", "_").lower()


def stacked(ax, rows, labels, title):
    """Horizontal 100 % stacked bars, one row per entry, domain order fixed."""
    y = np.arange(len(rows))[::-1]
    for yi, shares in zip(y, rows):
        left = 0.0
        for d in DOMAINS:
            w = shares[d]
            ax.barh(yi, w, left=left, height=0.62, color=DOMAIN_COLORS[d], edgecolor="white", linewidth=1.0)
            if w >= 8:
                ax.text(left + w / 2, yi, f"{w:.0f}", ha="center", va="center", fontsize=7, color=INK)
            left += w
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of importance (%)")
    ax.set_title(title, loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def fig2():
    cmp = pd.read_csv(TABLES / "r1_importance_domain_comparison.csv")
    grp = pd.read_csv(TABLES / "r1_permutation_grouped.csv")
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.5, 1.5], hspace=0.55)

    # (A) decision trees
    ax = fig.add_subplot(gs[0])
    rows, labels = [], []
    for sp in SPECIES:
        dt = pd.read_csv(RES / f"{safe(sp)}_feature_importances.csv")
        rows.append(domain_shares(dt.importance.to_numpy(), dt.variable.tolist()))
        labels.append(SHORT[sp])
    stacked(ax, rows, [f"$\\it{{{l.split()[0]}}}$ $\\it{{{l.split()[1]}}}$" for l in labels],
            "A  Decision trees (Gini)")

    # (B) random forests: Gini, SHAP, per-variable permutation
    ax = fig.add_subplot(gs[1])
    rows, labels = [], []
    for sp in SPECIES:
        c = cmp[cmp.species == sp].set_index("domain")
        # per-variable permutation is left out: shuffling one variable at a time dilutes the
        # importance of correlated groups (in P. leptodactylus the many climate variables stand
        # in for each other, so topography comes first), and in the near-perfect A. pallipes
        # model only 5 of 271 variables get positive importance; panel C shows the grouped
        # version instead
        for col, name in (("rf_gini_pct", "Gini"), ("rf_shap_pct", "SHAP")):
            if c[col].notna().all():
                rows.append({d: float(c.loc[d, col]) for d in DOMAINS})
                g, s = SHORT[sp].split()
                labels.append(f"$\\it{{{g}}}$ $\\it{{{s}}}$  {name}")
    stacked(ax, rows, labels, "B  Random forests (Gini, SHAP)")

    # (C) grouped permutation, loss of balanced accuracy
    ax = fig.add_subplot(gs[2])
    rf = grp[grp.model == "RF"]
    width = 0.18
    x = np.arange(len(SPECIES))
    for k, d in enumerate(DOMAINS):
        st = [rf[(rf.species == sp) & (rf.cv == "stratified") & (rf.domain == d)].drop_bacc_mean.iloc[0] for sp in SPECIES]
        sd = [rf[(rf.species == sp) & (rf.cv == "stratified") & (rf.domain == d)].drop_bacc_sd.iloc[0] for sp in SPECIES]
        spt = [rf[(rf.species == sp) & (rf.cv.str.startswith("spatial")) & (rf.domain == d)].drop_bacc_mean.iloc[0] for sp in SPECIES]
        pos = x + (k - 1.5) * width
        ax.bar(pos, st, width * 0.9, color=DOMAIN_COLORS[d], edgecolor="white", linewidth=0.8, label=d)
        ax.errorbar(pos, st, yerr=sd, fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5)
        ax.scatter(pos, spt, s=18, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\it{{{SHORT[sp].split()[0]}}}$ $\\it{{{SHORT[sp].split()[1]}}}$" for sp in SPECIES])
    ax.set_ylabel("Loss of balanced accuracy")
    ax.set_title("C  Random forests: grouped permutation by domain", loc="left")
    ax.text(0.01, 0.98, "Bars: stratified CV (mean ± SD,\n5 folds × 20 shuffles)\n"
            "Circles: 100 km spatial blocks", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=INK2, linespacing=1.4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    handles = [Patch(facecolor=DOMAIN_COLORS[d], label=d) for d in DOMAINS]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK,
                              label="100 km spatial blocks"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995), fontsize=7.5)
    out = FIGURES / "Fig2_domain_importance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figS2():
    glossary = load_glossary("data/raw/S2.xlsx")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), gridspec_kw={"wspace": 1.35})
    for ax, sp in zip(axes, SPECIES):
        dt = pd.read_csv(RES / f"{safe(sp)}_feature_importances.csv")
        top = dt[dt.importance > 0].head(20).iloc[::-1]
        cols = [DOMAIN_COLORS.get(domain_of(v), "#9b9a94") for v in top.variable]
        ax.barh(range(len(top)), 100 * top.importance, color=cols, height=0.7, edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([make_label(v, glossary, max_len=48) for v in top.variable], fontsize=6.5)
        ax.set_xlabel("Decision-tree Gini importance (%)")
        g, s = SHORT[sp].split()
        ax.set_title(f"$\\it{{{g}}}$ $\\it{{{s}}}$", loc="left")
        ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for yi, v in enumerate(100 * top.importance):
            if v >= 5:
                ax.text(v + 0.8, yi, f"{v:.1f}", va="center", fontsize=6.5, color=INK2)
    handles = [Patch(facecolor=DOMAIN_COLORS[d], label=d) for d in DOMAINS]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    out = FIGURES / "FigS2_top20_dt_by_domain.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figS4(n_top=15, min_imp=0.005):
    glossary = load_glossary("data/raw/S2.xlsx")
    imp = {sp: pd.read_csv(RES / f"{safe(sp)}_feature_importances.csv").set_index("variable").importance
           for sp in SPECIES}
    rows = set()
    for sp, s in imp.items():
        rows.update(s[s >= min_imp].sort_values(ascending=False).head(n_top).index)
    m = pd.DataFrame({sp: [float(imp[sp].get(v, 0.0)) for v in rows] for sp in SPECIES}, index=list(rows))
    m = m.loc[m.max(axis=1).sort_values(ascending=False).index]
    fig = plt.figure(figsize=(7.6, 0.24 * len(m) + 1.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.035, 1, 0.05], wspace=0.04, bottom=0.07, top=0.93)
    ax_strip, ax, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for i, v in enumerate(m.index):
        ax_strip.add_patch(plt.Rectangle((0, i - 0.5), 1, 1, color=DOMAIN_COLORS[domain_of(v)], linewidth=0))
    ax_strip.set_xlim(0, 1); ax_strip.set_ylim(len(m) - 0.5, -0.5)
    ax_strip.set_xticks([]); ax_strip.set_yticks(range(len(m)))
    ax_strip.set_yticklabels([make_label(v, glossary, max_len=52) for v in m.index], fontsize=6.5, color=INK)
    for s in ax_strip.spines.values():
        s.set_visible(False)
    ax_strip.tick_params(axis="y", length=0)
    im = ax.imshow(m.values, cmap="Greys", norm=PowerNorm(0.5, vmin=0, vmax=1), aspect="auto")
    for i in range(len(m)):
        for j in range(len(SPECIES)):
            val = m.values[i, j]
            if val >= 0.005:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if val > 0.25 else INK)
    ax.set_xticks(range(len(SPECIES)))
    ax.set_xticklabels([f"$\\it{{{SHORT[sp].split()[0]}}}$ $\\it{{{SHORT[sp].split()[1]}}}$" for sp in SPECIES])
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Decision-tree Gini importance (square-root colour scale)")
    handles = [Patch(facecolor=DOMAIN_COLORS[d], label=d) for d in DOMAINS]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.55, 0.055), fontsize=7.5)
    out = FIGURES / "FigS4_dt_importance_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figS5():
    """Fold stability of domain-level RF Gini importance, one panel per species."""
    data = {sp: pd.read_csv(RES / "cv_stability" / f"{safe(sp)}_type_summary.csv").set_index("type") for sp in SPECIES}
    ymax = max(float((d.mean_pct + d.sd_pct).max()) for d in data.values())
    ymax = 10 * np.ceil(ymax / 10 + 0.05)

    def panel(ax, sp):
        d = data[sp].reindex(DOMAINS)
        x = np.arange(len(DOMAINS))
        ax.bar(x, d.mean_pct, 0.62, color=[DOMAIN_COLORS[k] for k in DOMAINS], edgecolor="white", linewidth=0.8)
        ax.errorbar(x, d.mean_pct, yerr=d.sd_pct, fmt="none", ecolor=INK, elinewidth=0.9, capsize=3)
        for xi, (mu, sd) in enumerate(zip(d.mean_pct, d.sd_pct)):
            ax.text(xi, mu + sd + ymax * 0.015, f"{mu:.1f} \u00b1 {sd:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(DOMAINS)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Share of random-forest Gini importance (%)")
        g, s = SHORT[sp].split()
        ax.set_title(f"$\\it{{{g}}}$ $\\it{{{s}}}$", loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    outs = []
    for sp in SPECIES:
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        panel(ax, sp)
        out = FIGURES / f"FigS5_type_stability_{safe(sp)}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outs.append(out)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, sp in zip(axes, SPECIES):
        panel(ax, sp)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.text(0.5, -0.04, "Mean \u00b1 SD over the five stratified cross-validation folds", ha="center", fontsize=7.5, color=INK2)
    out = FIGURES / "FigS5_type_stability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outs.append(out)
    return outs


if __name__ == "__main__":
    ensure_dirs()
    print(fig2())
    print(figS2())
    print(figS4())
    for o in figS5():
        print(o)
