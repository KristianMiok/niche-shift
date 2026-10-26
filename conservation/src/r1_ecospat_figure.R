# r1_ecospat_figure.R — figures for R1 item KM-5 (and item 19), 300 dpi PNG.
#
#   figures/Fig3_ecospat_niche_dynamics.png
#       niche dynamics per species (columns) for the climatic and the non-climatic
#       variables (rows); 100 km background, intersection = 0.1. Replaces Figure 3.
#   figures/FigS7_ecospat_niche_dynamics_all_variables.png
#       the same for the full variable set of the classifiers. Replaces Figure S7.
#   figures/FigS8_ecospat_null_distributions.png
#       null distributions of D (equivalency; similarity in both directions) against
#       the observed D, as cumulative distributions, for every species and variable set.
#
# The 100 km grids are rebuilt exactly as in r1_ecospat.R (same PCA-env and kernel
# settings; deterministic), because terra rasters cannot be saved with saveRDS. The
# simulated D values are read from r1_ecospat_null_D_<set>_<species>.csv.
#
# Reading the dynamics panels (ecospat.plot.niche.dyn, ecospat 4.1.4): coloured cells
# are cells of the 100 x 100 environmental grid occupied in at least one range; within
# the environments shared by both ranges (each range's background without its 10 % most
# marginal environments) they are stability (occupied in both), expansion (non-native
# only) or unfilling (native only); outside them, occupied cells are native-only
# environments ("abandoned"), non-native-only environments ("pioneering") or marginal
# to both (grey). Contours: each range's background environment, full extent (solid)
# and without its 10 % most marginal environments (dashed). Grey shading: density of
# non-native occurrences. Arrows: shift of the centroid of occurrences (solid) and of
# the background (dashed) from the native to the non-native range.
#
# Run (from conservation/):  Rscript src/r1_ecospat_figure.R
suppressPackageStartupMessages({ library(ecospat); library(ade4) })

in_root <- "results/revision/ecospat_input"
tab_dir <- "results/revision/tables"
fig_dir <- "results/revision/figures"
species <- c(pontastacus_leptodactylus = "P. leptodactylus",
             austropotamobius_fulcisianus = "A. fulcisianus",
             austropotamobius_pallipes = "A. pallipes")
full <- c(pontastacus_leptodactylus = "Pontastacus leptodactylus",
          austropotamobius_fulcisianus = "Austropotamobius fulcisianus",
          austropotamobius_pallipes = "Austropotamobius pallipes")
set_labels <- c(climate = "climatic variables", non_climate = "non-climatic variables", all = "all variables")
CELL_STEP <- c("l_TOP101", "l_TOP102", "l_TOP103", "l_TOP104", "u_TOP101", "u_TOP102", "u_TOP103", "u_TOP104")
select_vars <- function(v, set) switch(set, all = v, climate = v[grepl("CLI", v)],
                                       non_climate = v[!grepl("CLI", v) & !(v %in% CELL_STEP)])
COL_NAT <- "#e67e22"; COL_NON <- "#7b3294"
COL_STAB <- "#2a78d6"; COL_EXP <- "#e34948"; COL_UNF <- "#eda100"
ALPHA <- 0.30   # ecospat.plot.niche.dyn transparency = 70

build <- function(key, set, B = 100) {
  d <- file.path(in_root, key, paste0(B, "km"))
  vars <- select_vars(readLines(file.path(d, "variables.txt")), set)
  bg1 <- read.csv(file.path(d, "bg_native.csv"));  bg2 <- read.csv(file.path(d, "bg_nonnative.csv"))
  oc1 <- read.csv(file.path(d, "occ_native.csv")); oc2 <- read.csv(file.path(d, "occ_nonnative.csv"))
  glob <- rbind(bg1[, vars], bg2[, vars])
  keep <- vars[apply(glob, 2, sd) > 0]
  pca <- dudi.pca(glob[, keep], center = TRUE, scale = TRUE, scannf = FALSE, nf = 2)
  s <- lapply(list(bg1 = bg1, bg2 = bg2, oc1 = oc1, oc2 = oc2), function(x) suprow(pca, x[, keep])$lisup)
  z1 <- ecospat.grid.clim.dyn(glob = pca$li, glob1 = s$bg1, sp = s$oc1, R = 100, th.sp = 0, th.env = 0)
  z2 <- ecospat.grid.clim.dyn(glob = pca$li, glob1 = s$bg2, sp = s$oc2, R = 100, th.sp = 0, th.env = 0)
  list(z1 = z1, z2 = z2, s = s, var = pca$eig / sum(pca$eig))
}

sets <- names(set_labels)[file.exists(file.path(tab_dir, paste0("r1_ecospat_results_", names(set_labels), ".csv")))]
res <- do.call(rbind, lapply(sets, function(st) read.csv(file.path(tab_dir, paste0("r1_ecospat_results_", st, ".csv")))))
res <- res[res$buffer_km == 100, ]

dyn_legend <- function() {
  par(mar = c(0, 0, 0, 0)); plot.new()
  legend("center", ncol = 2, bty = "n", cex = 0.8, x.intersp = 0.6, y.intersp = 1.05,
         legend = c("Stability: occupied in both ranges", "Expansion: occupied in the non-native range only",
                    "Unfilling: occupied in the native range only",
                    "Native niche in environments absent from the non-native range",
                    "Non-native niche in environments absent from the native range",
                    "Native background: full extent (solid), without 10% most marginal (dashed)",
                    "Non-native background: full extent (solid), without 10% most marginal (dashed)",
                    "Centroid shift: occurrences (solid), background (dashed)",
                    "Grey shading: density of non-native occurrences",
                    "Occupied, in environments marginal to both ranges"),
         fill = c(adjustcolor(c(COL_STAB, COL_EXP, COL_UNF, "lightgreen", "pink"), alpha.f = ALPHA), NA, NA, NA,
                  "grey35", adjustcolor("grey", alpha.f = ALPHA)),
         border = c(rep("grey40", 5), NA, NA, NA, "grey35", "grey40"),
         col = c(rep(NA, 5), COL_NAT, COL_NON, "black", NA, NA), lwd = c(rep(NA, 5), 2, 2, 2, NA, NA))
}

dyn_figure <- function(plot_sets, file) {
  # ecospat draws with terra::plot and hard-codes its margins, so each panel's two title
  # lines get their own layout cell above the panel instead of a main title
  ns <- length(plot_sets)
  cells <- matrix(0, nrow = 2 * ns, ncol = 3)
  for (i in seq_len(ns)) {
    cells[2 * i - 1, ] <- (i - 1) * 6 + 1:3   # title cells
    cells[2 * i, ] <- (i - 1) * 6 + 4:6       # plot cells
  }
  png(file.path(fig_dir, file), width = 3 * 1050, height = ns * 900 + 520, res = 300)
  layout(rbind(cells, rep(6 * ns + 1, 3)), heights = c(rep(c(0.16, 1), ns), 0.62))
  par(cex = 0.75)
  k <- 0
  for (st in plot_sets) {
    g_list <- lapply(names(species), function(key) build(key, st))
    names(g_list) <- names(species)
    par(mar = c(0, 0, 0, 0))
    for (key in names(species)) {
      r <- res[res$variable_set == st & res$species == full[[key]], ]
      k <- k + 1
      plot.new()
      text(0.5, 0.68, bquote(bold(.(LETTERS[k])) ~~ italic(.(species[[key]])) ~ "(" * .(set_labels[[st]]) * ")"),
           cex = 0.95)
      text(0.5, 0.18, sprintf("D = %.3f; expansion %.3f, stability %.3f, unfilling %.3f",
                              r$D, r$expansion_i01, r$stability_i01, r$unfilling_i01), cex = 0.8)
    }
    par(mgp = c(1.9, 0.6, 0), cex = 0.75)
    for (key in names(species)) {
      g <- g_list[[key]]
      ecospat.plot.niche.dyn(g$z1, g$z2, intersection = 0.1, interest = 2, title = "",
                             name.axis1 = sprintf("PC1 (%.1f%%)", 100 * g$var[1]),
                             name.axis2 = sprintf("PC2 (%.1f%%)", 100 * g$var[2]),
                             col.unf = COL_UNF, col.exp = COL_EXP, col.stab = COL_STAB,
                             colZ1 = COL_NAT, colZ2 = COL_NON)
      ecospat.shift.centroids(g$s$oc1, g$s$oc2, g$s$bg1, g$s$bg2, col = "black")
    }
  }
  dyn_legend()
  dev.off()
  file.path(fig_dir, file)
}

out <- c(dyn_figure(intersect(c("climate", "non_climate"), sets), "Fig3_ecospat_niche_dynamics.png"))
if ("all" %in% sets) out <- c(out, dyn_figure("all", "FigS7_ecospat_niche_dynamics_all_variables.png"))

# ---- null distributions ---------------------------------------------------------
cols <- c(equivalency = "grey35", similarity_nat_to_non = COL_STAB, similarity_non_to_nat = COL_NAT)
labs <- c(equivalency = "Equivalency (records reallocated between ranges)",
          similarity_nat_to_non = "Similarity, native to non-native (non-native niche shifted at random)",
          similarity_non_to_nat = "Similarity, non-native to native (native niche shifted at random)")
f_tests <- "FigS8_ecospat_null_distributions.png"
png(file.path(fig_dir, f_tests), width = 3 * 1050, height = length(sets) * 950 + 330, res = 300)
layout(rbind(matrix(seq_len(3 * length(sets)), ncol = 3, byrow = TRUE), rep(3 * length(sets) + 1, 3)),
       heights = c(rep(1, length(sets)), 0.36))
par(mar = c(3.6, 3.6, 2.6, 0.8), mgp = c(2.0, 0.6, 0), cex = 0.8)
k <- 0
for (st in sets) for (key in names(species)) {
  k <- k + 1
  sims <- read.csv(file.path(tab_dir, paste0("r1_ecospat_null_D_", st, "_", key, ".csv")))
  r <- res[res$variable_set == st & res$species == full[[key]], ]
  xr <- range(c(sims$D, r$D)); xr <- c(max(0, xr[1] - 0.02), min(1, xr[2] + 0.02))
  plot(NA, xlim = xr, ylim = c(0, 1), xlab = "Schoener's D", ylab = "Cumulative proportion")
  title(main = bquote(bold(.(LETTERS[k])) ~~ italic(.(species[[key]])) ~ "(" * .(set_labels[[st]]) * ")"),
        font.main = 1, cex.main = 1.05)
  for (t in names(cols)) lines(ecdf(sims$D[sims$test == t]), col = cols[[t]], do.points = FALSE, verticals = TRUE, lwd = 1.8)
  abline(v = r$D, col = COL_EXP, lwd = 2, lty = 2)
}
par(mar = c(0, 0, 0, 0)); plot.new()
legend("center", ncol = 2, bty = "n", cex = 0.8, legend = c(labs, "Observed D"),
       col = c(cols, COL_EXP), lwd = 2, lty = c(1, 1, 1, 2))
dev.off()
out <- c(out, file.path(fig_dir, f_tests))
cat("Saved", out, sep = "\n  ")
