# r1_ecospat.R — R1 item KM-5, step 2: niche overlap, equivalency and similarity tests
# and niche dynamics (stability, expansion, unfilling) with ecospat
# (Broennimann et al. 2012; Guisan et al. 2014; Di Cola et al. 2017).
#
# Input: results/revision/ecospat_input/<species>/<B>km/ from r1_ecospat_prepare.py
#
# Per species and buffer B:
#   PCA-env  : ade4::dudi.pca on the pooled background of both ranges, centred and
#              scaled (correlation matrix), first two axes kept; background and
#              occurrences projected with suprow().
#   Densities: ecospat.grid.clim.dyn, R = 100 grid cells per axis, kernel.method =
#              "adehabitat" (bivariate normal kernel, reference bandwidth h_ref),
#              th.sp = 0, th.env = 0.
#   Overlap  : Schoener's D and Warren's I on prevalence-corrected densities (z.cor).
#   Tests    : equivalency (records reallocated between ranges) and similarity
#              (rand.type = 2, the second range's density shifted at random within its
#              background), in both directions; 1000 replicates; p-values for both
#              alternatives computed from the simulated distributions as (k + 1)/(n + 1).
#   Dynamics : ecospat.niche.dyn.index with intersection = 0.1 (primary: environments
#              shared by both ranges after removing the 10 % most marginal environments
#              of each range's background) and intersection = 0 (the full intersection).
#              In the tests, intersection affects only the dynamic indices: observed and
#              simulated D are computed over the whole environmental space (ecospat 4.1.4).
# Tests are run for the primary buffer (100 km); 50 and 200 km give overlap and
# dynamics only.
#
# Variable sets (environment variable ECOSPAT_SET), each with its own PCA-env:
#   all          the species' full cleaned variable set (as the classifiers); its first
#                two axes are dominated by the many topographic slope/elevation-
#                difference variables and by soil, not by climate
#   climate      climatic variables only: the climatic niche of the niche-shift
#                literature (Broennimann et al. 2012; Petitpierre et al. 2012)
#   non_climate  topography + soil + land cover, without the cell-step variables
#                l_TOP101-104 (latitude proxy): a direct habitat-niche comparison
# Outputs carry the set name: r1_ecospat_results_<set>.csv and
# r1_ecospat_null_D_<set>_<species>.csv (the 1000 simulated D values of each test).
# Figures: src/r1_ecospat_figure.R (Fig3_ecospat_niche_dynamics.png,
# FigS6_ecospat_niche_dynamics_all_variables.png, FigS7_ecospat_null_distributions.png),
# which rebuilds the 100 km grids (terra rasters cannot be saved with saveRDS) and reads
# the simulated D values from these files.
#
# Run (from conservation/):  ECOSPAT_SET=climate Rscript src/r1_ecospat.R
suppressPackageStartupMessages({ library(ecospat); library(ade4); library(jsonlite) })
set.seed(20261026)

in_root  <- "results/revision/ecospat_input"
tab_dir  <- "results/revision/tables"
fig_dir  <- "results/revision/figures"
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

species <- c(pontastacus_leptodactylus = "Pontastacus leptodactylus",
             austropotamobius_fulcisianus = "Austropotamobius fulcisianus",
             austropotamobius_pallipes = "Austropotamobius pallipes")
buffers <- c(50, 100, 200)
primary <- 100
REP <- as.integer(Sys.getenv("ECOSPAT_REP", "1000"))
NCORES <- as.integer(Sys.getenv("ECOSPAT_NCORES", "1"))  # terra grids cannot be sent to parallel workers
SET <- Sys.getenv("ECOSPAT_SET", "all")
CELL_STEP <- c("l_TOP101", "l_TOP102", "l_TOP103", "l_TOP104", "u_TOP101", "u_TOP102", "u_TOP103", "u_TOP104")
select_vars <- function(v) switch(SET,
  all = v,
  climate = v[grepl("CLI", v)],
  non_climate = v[!grepl("CLI", v) & !(v %in% CELL_STEP)],
  stop("ECOSPAT_SET must be all, climate or non_climate"))

pvals <- function(obs, sim) {
  n <- length(sim)
  c(higher = (sum(sim >= obs) + 1) / (n + 1), lower = (sum(sim <= obs) + 1) / (n + 1))
}

rows <- list()
for (key in names(species)) {
  sp <- species[[key]]
  for (B in buffers) {
    d <- file.path(in_root, key, paste0(B, "km"))
    vars <- select_vars(readLines(file.path(d, "variables.txt")))
    bg1 <- read.csv(file.path(d, "bg_native.csv"));   bg2 <- read.csv(file.path(d, "bg_nonnative.csv"))
    oc1 <- read.csv(file.path(d, "occ_native.csv"));  oc2 <- read.csv(file.path(d, "occ_nonnative.csv"))
    glob <- rbind(bg1[, vars], bg2[, vars])
    # variables constant over the pooled background cannot be scaled
    keep <- vars[apply(glob, 2, sd) > 0]
    pca <- dudi.pca(glob[, keep], center = TRUE, scale = TRUE, scannf = FALSE, nf = 2)
    var_expl <- pca$eig / sum(pca$eig)
    s_glob <- pca$li
    s_bg1 <- suprow(pca, bg1[, keep])$lisup;  s_bg2 <- suprow(pca, bg2[, keep])$lisup
    s_oc1 <- suprow(pca, oc1[, keep])$lisup;  s_oc2 <- suprow(pca, oc2[, keep])$lisup
    z1 <- ecospat.grid.clim.dyn(glob = s_glob, glob1 = s_bg1, sp = s_oc1, R = 100, th.sp = 0, th.env = 0)
    z2 <- ecospat.grid.clim.dyn(glob = s_glob, glob1 = s_bg2, sp = s_oc2, R = 100, th.sp = 0, th.env = 0)
    ov_c <- ecospat.niche.overlap(z1, z2, cor = TRUE)
    ov_u <- ecospat.niche.overlap(z1, z2, cor = FALSE)
    dyn01 <- ecospat.niche.dyn.index(z1, z2, intersection = 0.1)$dynamic.index.w
    dyn0  <- ecospat.niche.dyn.index(z1, z2, intersection = 0)$dynamic.index.w
    # top loadings of the two axes (for the methods/figure caption)
    co <- pca$co; top1 <- head(rownames(co)[order(-abs(co[, 1]))], 5); top2 <- head(rownames(co)[order(-abs(co[, 2]))], 5)
    r <- list(variable_set = SET, species = sp, buffer_km = B, n_occ_native = nrow(oc1), n_occ_nonnative = nrow(oc2),
              n_bg_native = nrow(bg1), n_bg_nonnative = nrow(bg2), n_variables = length(keep),
              pc1_var = var_expl[1], pc2_var = var_expl[2],
              pc1_top_loadings = paste(top1, collapse = ";"), pc2_top_loadings = paste(top2, collapse = ";"),
              D = ov_c$D, I = ov_c$I, D_uncorrected = ov_u$D, I_uncorrected = ov_u$I,
              expansion_i01 = unname(dyn01["expansion"]), stability_i01 = unname(dyn01["stability"]),
              unfilling_i01 = unname(dyn01["unfilling"]),
              expansion_i0 = unname(dyn0["expansion"]), stability_i0 = unname(dyn0["stability"]),
              unfilling_i0 = unname(dyn0["unfilling"]))
    if (B == primary) {
      cat(sprintf("%s: tests with %d replicates...\n", sp, REP))
      eq   <- ecospat.niche.equivalency.test(z1, z2, rep = REP, intersection = 0.1, ncores = NCORES)
      s12  <- ecospat.niche.similarity.test(z1, z2, rep = REP, intersection = 0.1, rand.type = 2, ncores = NCORES)
      s21  <- ecospat.niche.similarity.test(z2, z1, rep = REP, intersection = 0.1, rand.type = 2, ncores = NCORES)
      p_eq <- pvals(eq$obs$D, eq$sim$D); p12 <- pvals(s12$obs$D, s12$sim$D); p21 <- pvals(s21$obs$D, s21$sim$D)
      r <- c(r, list(replicates = REP,
                     equivalency_D_sim_mean = mean(eq$sim$D), equivalency_p_higher = p_eq[["higher"]],
                     equivalency_p_lower = p_eq[["lower"]],
                     similarity_nat_to_non_D_sim_mean = mean(s12$sim$D), similarity_nat_to_non_p_higher = p12[["higher"]],
                     similarity_nat_to_non_p_lower = p12[["lower"]],
                     similarity_non_to_nat_D_sim_mean = mean(s21$sim$D), similarity_non_to_nat_p_higher = p21[["higher"]],
                     similarity_non_to_nat_p_lower = p21[["lower"]]))
      write.csv(data.frame(test = rep(c("equivalency", "similarity_nat_to_non", "similarity_non_to_nat"), each = length(eq$sim$D)),
                           D = c(eq$sim$D, s12$sim$D, s21$sim$D)),
                file.path(tab_dir, paste0("r1_ecospat_null_D_", SET, "_", key, ".csv")), row.names = FALSE)
    }
    rows[[length(rows) + 1]] <- r
    cat(sprintf("[%s] %-30s %3d km  D=%.3f  exp=%.3f stab=%.3f unf=%.3f  PC1+2=%.1f%%\n", SET, sp, B, ov_c$D,
                r$expansion_i01, r$stability_i01, r$unfilling_i01, 100 * (var_expl[1] + var_expl[2])))
  }
}
all_cols <- unique(unlist(lapply(rows, names)))
res <- do.call(rbind, lapply(rows, function(x) {
  x[setdiff(all_cols, names(x))] <- NA
  as.data.frame(x[all_cols], stringsAsFactors = FALSE)
}))
write.csv(res, file.path(tab_dir, paste0("r1_ecospat_results_", SET, ".csv")), row.names = FALSE)

cat(toJSON(list(R = R.version.string, ecospat = as.character(packageVersion("ecospat")),
                ade4 = as.character(packageVersion("ade4")), adehabitatHR = as.character(packageVersion("adehabitatHR")),
                replicates = REP, variable_set = SET), auto_unbox = TRUE),
    file = file.path(tab_dir, paste0("r1_ecospat_versions_", SET, ".json")))
cat("Saved", file.path(tab_dir, paste0("r1_ecospat_results_", SET, ".csv")), "\n")
