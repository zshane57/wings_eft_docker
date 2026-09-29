#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# process_features.R
#
# Accepts a new sample coordinate CSV (with dm_x, dm_y, pa2r_x, pa2r_y columns),
# converts into EFD coefficients, reads testing.csv as dataframe and appends
# the new sample into the testing df. Performs LDA projection on the whole
# testing set together (using pre-trained LDA from fold 1). Then reads the
# fold-1 training rows (weight = 1) from the base guide data file
# (guide_input/combined_fd1_dmpa2r_data.txt) and writes the FULL GUIDE input
# file: training rows (weight = 1) on top, followed by the testing rows
# (weight = 0; the last row is the new sample), with header format
# (sample_name, species, family, dm-LD1..dm-LD12, pa2r-LD1..pa2r-LD12, weight).
#
# Usage:
#   Rscript process_features.R <coord_csv> <output_file> [species] [family]
#
#   <coord_csv>   : path to contour coordinates CSV (4 cols: dm_x, dm_y, pa2r_x, pa2r_y)
#   <output_file> : path where the FULL guide data file is written (training
#                   rows on top, testing rows below - ready for GUIDE)
#   [species]     : optional class label for the new sample row. GUIDE cannot
#                   parse a row whose class label is not a training level, so
#                   the new sample needs a placeholder species label
#                   (default "C.vicina", any training species works).
#   [family]      : optional family label; derived from [species] if omitted.
# -----------------------------------------------------------------------------

# Load required packages
suppressMessages({
  if (!requireNamespace("Momocs", quietly = TRUE)) {
    if (!requireNamespace("remotes", quietly = TRUE)) {
      install.packages("remotes", repos = "https://cloud.r-project.org")
    }
    if (!requireNamespace("sf", quietly = TRUE)) {
      install.packages("sf", repos = "https://cloud.r-project.org")
    }
    if (!requireNamespace("s2", quietly = TRUE)) {
      install.packages("s2", repos = "https://cloud.r-project.org")
    }
    library(remotes)
    remotes::install_version("Matrix", version = "1.6-5")
    remotes::install_version("RRPP", version = "2.0.0")
    remotes::install_version("geomorph", version = "4.0.7")
    remotes::install_github("MomX/Momocs", upgrade = "never", quiet = TRUE)
  }
  library(Momocs)
  library(MASS)
})

# --- Argument parsing ---------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript process_features.R <coord_csv> <output_file> [species] [family]")
}
coord_csv   <- args[[1]]
output_file <- args[[2]]

# GUIDE requires every row (including weight-0 test rows) to carry a class
# label that exists among the training species; a label like "unknown"
# aborts the GUIDE run. The new sample is therefore given a placeholder
# species label (default C.vicina), overridable on the command line.
new_species_label <- if (length(args) >= 3) args[[3]] else "C.vicina"

family_map <- c(
  "C.vicina" = "Calliphoridae", "Ch.albiceps" = "Calliphoridae",
  "Ch.bezziana" = "Calliphoridae", "Ch.megacephala" = "Calliphoridae",
  "Ch.nigripes" = "Calliphoridae", "Ch.rufifacies" = "Calliphoridae",
  "L.sericata" = "Calliphoridae", "A.gressitti" = "Sarcophagidae",
  "B.karnyi" = "Sarcophagidae", "Le.alba" = "Sarcophagidae",
  "Z.aquila" = "Sarcophagidae", "S.princeps" = "Sarcophagidae",
  "Sy.nudiseta" = "Muscidae"
)
if (length(args) >= 4) {
  new_family_label <- args[[4]]
} else if (new_species_label %in% names(family_map)) {
  new_family_label <- family_map[[new_species_label]]
} else {
  stop(sprintf("Species '%s' not in family map; pass its family as the 4th argument.",
               new_species_label))
}

if (!file.exists(coord_csv)) {
  stop(sprintf("Coordinate CSV not found: %s", coord_csv))
}

# Source the custom normalization helpers
source("normalisation_scale_rotate_phase.R")

H <- 10  # number of harmonics

# --- Helper: extract EFD coefficients from contour coordinates ----------------
extract_efourier_coefficients <- function(raw_coe, harmonics = H) {
  if (!is.list(raw_coe)) {
    stop("efourier() output must be a list")
  }

  coeff_names <- c("an", "bn", "cn", "dn")
  coeffs <- list()

  for (name in coeff_names) {
    if (!name %in% names(raw_coe)) {
      stop(sprintf("efourier() output is missing '%s' coefficient list", name))
    }

    values <- raw_coe[[name]]
    if (is.list(values)) {
      values <- unlist(values, use.names = FALSE)
    }
    values <- as.numeric(values)
    if (length(values) < harmonics) {
      stop(sprintf("Expected at least %d coefficients for '%s', got %d", harmonics, name, length(values)))
    }

    coeffs[[name]] <- values[seq_len(harmonics)]
  }

  coeffs
}

# Canonicalise the traversal direction of a contour so that its EFD
# coefficients land in the same convention as the LDA training database.
# EFD coefficients are sensitive to the direction in which the contour
# points are emitted (reversing the order flips the sign of every b_n and
# d_n after normalisation); the scale/rotate/phase normalisation cannot
# absorb a reversed traversal. Every row of the training database
# (h10_dm_norm_new.csv / h10_pa2r_norm_new.csv) has d1 < 0, so we enforce
# d1 < 0: if a first pass yields d1 > 0, the extraction emitted the points
# the other way round and we reverse the sequence.
canonicalise_direction <- function(m) {
  norm_d1 <- function(mm) {
    ef <- tryCatch(efourier(mm, nb.h = H, norm = TRUE),
                   error = function(e) efourier(Momocs::coo(mm), nb.h = H, norm = TRUE))
    co <- extract_efourier_coefficients(ef, H)
    df <- as.data.frame(t(c(co$an, co$bn, co$cn, co$dn)), stringsAsFactors = FALSE)
    names(df) <- c(paste0("a", 1:H), paste0("b", 1:H),
                   paste0("c", 1:H), paste0("d", 1:H))
    res <- normalize_row(df[1, , drop = FALSE], H)
    res$final$dn[1]
  }
  if (norm_d1(m) > 0) {
    cat("  [direction] d1 > 0: reversing contour point order to match training convention\n")
    m[nrow(m):1, , drop = FALSE]
  } else {
    m
  }
}

# Process contour coordinates from combined CSV
process_contour_csv <- function(coord_csv) {
  coords <- read.csv(coord_csv, header = TRUE, stringsAsFactors = FALSE)

  # Check for required columns
  required_cols <- c("dm_x", "dm_y", "pa2r_x", "pa2r_y")
  missing_cols <- setdiff(required_cols, names(coords))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s. Available: %s",
                 paste(missing_cols, collapse = ", "),
                 paste(names(coords), collapse = ", ")))
  }

  # Extract DM coordinates
  dm_matrix <- cbind(as.numeric(coords$dm_x), as.numeric(coords$dm_y))
  dm_matrix <- na.omit(dm_matrix)

  # Extract PA2R coordinates
  pa2r_matrix <- cbind(as.numeric(coords$pa2r_x), as.numeric(coords$pa2r_y))
  pa2r_matrix <- na.omit(pa2r_matrix)

  # Align both contours to the training-data traversal-direction convention
  dm_matrix <- canonicalise_direction(dm_matrix)
  pa2r_matrix <- canonicalise_direction(pa2r_matrix)

  if (nrow(dm_matrix) < 6) {
    stop(sprintf("DM: needs at least 6 contour points for EFD (10 harmonics), got %d", nrow(dm_matrix)))
  }
  if (nrow(pa2r_matrix) < 6) {
    stop(sprintf("PA2R: needs at least 6 contour points for EFD (10 harmonics), got %d", nrow(pa2r_matrix)))
  }

  # Compute EFDs with norm = TRUE to match docker notebook pipeline
  # (docker notebook uses norm=TRUE then applies custom normalization)
  tryCatch({
    raw_dm <- efourier(dm_matrix, H, norm = TRUE)
    raw_pa2r <- efourier(pa2r_matrix, H, norm = TRUE)
  }, error = function(e) {
    dm_coo <- Momocs::coo(dm_matrix)
    pa2r_coo <- Momocs::coo(pa2r_matrix)
    raw_dm <<- efourier(dm_coo, H, norm = TRUE)
    raw_pa2r <<- efourier(pa2r_coo, H, norm = TRUE)
  })

  coeff_dm <- extract_efourier_coefficients(raw_dm, H)
  coeff_pa2r <- extract_efourier_coefficients(raw_pa2r, H)

  # Pack into wide format dataframes
  vec_dm <- c(coeff_dm$an, coeff_dm$bn, coeff_dm$cn, coeff_dm$dn)
  vec_pa2r <- c(coeff_pa2r$an, coeff_pa2r$bn, coeff_pa2r$cn, coeff_pa2r$dn)

  df_dm <- as.data.frame(t(vec_dm), stringsAsFactors = FALSE)
  colnames(df_dm) <- c(paste0("a", 1:H), paste0("b", 1:H),
                       paste0("c", 1:H), paste0("d", 1:H))
  df_dm$name <- "new_sample"
  df_dm$species <- new_species_label
  df_dm$family <- new_family_label
  df_dm <- df_dm[, c("name", "species", "family",
                     paste0("a", 1:H), paste0("b", 1:H),
                     paste0("c", 1:H), paste0("d", 1:H))]

  df_pa2r <- as.data.frame(t(vec_pa2r), stringsAsFactors = FALSE)
  colnames(df_pa2r) <- c(paste0("a", 1:H), paste0("b", 1:H),
                         paste0("c", 1:H), paste0("d", 1:H))
  df_pa2r$name <- "new_sample"
  df_pa2r$species <- new_species_label
  df_pa2r$family <- new_family_label
  df_pa2r <- df_pa2r[, c("name", "species", "family",
                         paste0("a", 1:H), paste0("b", 1:H),
                         paste0("c", 1:H), paste0("d", 1:H))]

  return(list(dm = df_dm, pa2r = df_pa2r))
}

# Apply normalization pipeline to a single row
apply_pipeline <- function(df_row) {
  results <- normalize_row(df_row[1, , drop = FALSE], H)
  list(
    an = results$final$an,
    bn = results$final$bn,
    cn = results$final$cn,
    dn = results$final$dn
  )
}

# Project and center using LDA loadings and mean
project_and_center_lda <- function(vec_40, loadings_path, mean_rds_path, cell_name) {
  coef_names <- c(paste0("a", 1:H), paste0("b", 1:H),
                  paste0("c", 1:H), paste0("d", 1:H))
  names(vec_40) <- coef_names

  # Drop a1, b1, c1 (retain 37 features including d1)
  cols_to_remove <- c("a1", "b1", "c1")
  vec_37 <- vec_40[!names(vec_40) %in% cols_to_remove]

  # Load 37-row LDA loadings matrix
  if (!file.exists(loadings_path)) {
    stop(sprintf("LDA loadings file not found: %s", loadings_path))
  }
  loadings_mat <- as.matrix(read.table(loadings_path, header = TRUE, row.names = 1,
                                       sep = "\t", check.names = FALSE))

  # Ensure feature order matches loadings row names exactly
  if (!all(names(vec_37) == rownames(loadings_mat))) {
    vec_37 <- vec_37[rownames(loadings_mat)]
  }

  # Raw LDA projection (1x37 %*% 37x12 = 1x12)
  ld_raw <- as.numeric(matrix(vec_37, nrow = 1) %*% loadings_mat)

  # Subtract training set LDA mean
  if (!file.exists(mean_rds_path)) {
    stop(sprintf("Mean LDA RDS file not found: %s", mean_rds_path))
  }
  mean_ld_train <- readRDS(mean_rds_path)

  ld_centered <- ld_raw - mean_ld_train
  names(ld_centered) <- paste0(cell_name, "-LD", 1:12)
  return(ld_centered)
}

# --- Main processing ----------------------------------------------------------

cat("Processing contour coordinates...\n")
coords_list <- process_contour_csv(coord_csv)
dm_df_raw <- coords_list$dm
pa2r_df_raw <- coords_list$pa2r

# Apply normalization
cat("Applying normalization pipeline...\n")
norm_dm <- apply_pipeline(dm_df_raw)
norm_pa2r <- apply_pipeline(pa2r_df_raw)

# Assemble 40-element vectors
vec_dm <- c(norm_dm$an, norm_dm$bn, norm_dm$cn, norm_dm$dn)
vec_pa2r <- c(norm_pa2r$an, norm_pa2r$bn, norm_pa2r$cn, norm_pa2r$dn)

# Paths to pre-trained LDA models (fold 1) — relative to the working directory
# (same files as /Users/zshane/Documents/um_msc/{dm,pa2r}_h10_guide_new/, kept
# as local copies so the script runs unchanged on the host and in Docker)
dm_loadings_path   <- "lda_loadings_fold_1_dm.txt"
dm_mean_path       <- "mean_ld_train_fd_1_dm.rds"
pa2r_loadings_path <- "lda_loadings_fold_1_pa2r.txt"
pa2r_mean_path     <- "mean_ld_train_fd_1_pa2r.rds"

# Base guide data file holding the fold-1 training set (its weight = 1 rows).
# See flywingproj.md Task 4: the combined training file is the GUIDE base;
# the script keeps its training rows and rebuilds the testing section below.
base_training_path <- "guide_input/combined_fd1_dmpa2r_data.txt"

# Project and center both cells
cat("Projecting DM using LDA...\n")
dm_ld <- project_and_center_lda(vec_dm, dm_loadings_path, dm_mean_path, "dm")

cat("Projecting PA2R using LDA...\n")
pa2r_ld <- project_and_center_lda(vec_pa2r, pa2r_loadings_path, pa2r_mean_path, "pa2r")

# Combine 24 features
features_24 <- c(dm_ld, pa2r_ld)
if (length(features_24) != 24) {
  stop(sprintf("Expected 24 features, got %d.", length(features_24)))
}

# --- Read existing testing data and project all samples using LDA -------------
cat("Reading existing testing data...\n")
dm_testing <- read.csv("dm_testing.csv", stringsAsFactors = FALSE)
pa2r_testing <- read.csv("pa2r_testing.csv", stringsAsFactors = FALSE)

# Project all existing testing samples using LDA
project_testing_samples <- function(testing_df, loadings_path, mean_rds_path, cell_name) {
  results_list <- list()

  for (i in 1:nrow(testing_df)) {
    row <- testing_df[i, ]
    # Extract 40 EFD coefficients
    vec_40 <- as.numeric(row[paste0("a", 1:H)])
    vec_40 <- c(vec_40, as.numeric(row[paste0("b", 1:H)]))
    vec_40 <- c(vec_40, as.numeric(row[paste0("c", 1:H)]))
    vec_40 <- c(vec_40, as.numeric(row[paste0("d", 1:H)]))
    names(vec_40) <- c(paste0("a", 1:H), paste0("b", 1:H),
                       paste0("c", 1:H), paste0("d", 1:H))

    # Project and center using the existing function
    ld_centered <- project_and_center_lda(vec_40, loadings_path, mean_rds_path, cell_name)

    # Create result row with proper column names
    result_row <- data.frame(
      sample_name = row$name,
      species = row$species,
      family = row$family,
      weight = 0,
      stringsAsFactors = FALSE
    )
    # Add LD columns
    for (j in 1:12) {
      result_row[[paste0(cell_name, "-LD", j)]] <- ld_centered[j]
    }
    results_list[[i]] <- result_row
  }

  do.call(rbind, results_list)
}

cat("Projecting existing DM testing samples...\n")
dm_testing_ld <- project_testing_samples(dm_testing, dm_loadings_path, dm_mean_path, "dm")

cat("Projecting existing PA2R testing samples...\n")
pa2r_testing_ld <- project_testing_samples(pa2r_testing, pa2r_loadings_path, pa2r_mean_path, "pa2r")

# Create new sample row
new_sample_row <- data.frame(
  sample_name = dm_df_raw$name,
  species = dm_df_raw$species,
  family = dm_df_raw$family,
  t(features_24),
  weight = 0,
  stringsAsFactors = FALSE
)

# Combine all testing data
# Merge dm and pa2r testing data by sample_name
dm_cols <- c("sample_name", "species", "family", paste0("dm-LD", 1:12), "weight")
pa2r_cols <- c("sample_name", paste0("pa2r-LD", 1:12))

# Ensure dm_testing_ld has correct columns
dm_testing_ld <- dm_testing_ld[, dm_cols]
pa2r_testing_ld <- pa2r_testing_ld[, pa2r_cols]

# Merge
combined_testing <- merge(dm_testing_ld, pa2r_testing_ld, by = "sample_name", all = TRUE)

# Add new sample
# Ensure new_sample_row has correct column names
expected_cols <- c("sample_name", "species", "family",
                   paste0("dm-LD", 1:12), paste0("pa2r-LD", 1:12), "weight")
names(new_sample_row) <- expected_cols

combined_testing <- rbind(combined_testing, new_sample_row)

# Reorder columns to match expected output format
output_cols <- c("sample_name", "species", "family",
                 paste0("dm-LD", 1:12), paste0("pa2r-LD", 1:12), "weight")
combined_testing <- combined_testing[, output_cols]

# --- Training data: fold-1 training rows (weight = 1) from the base file -----
# The base guide data file contains the SMOTE-balanced fold-1 training set
# (weight = 1) followed by the fold-1 testing rows (weight = 0). Only the
# training rows are taken from it: they cannot be regenerated without
# re-running the whole 5-fold LDA + SMOTE pipeline. The testing rows are
# re-projected fresh above and the new sample is appended at the very end
# (flywingproj.md Tasks 3-4).
if (!file.exists(base_training_path)) {
  stop(sprintf("Base training data file not found: %s", base_training_path))
}
base_data <- read.delim(base_training_path, header = TRUE,
                        stringsAsFactors = FALSE, check.names = FALSE)
base_weight <- suppressWarnings(as.numeric(base_data$weight))
if (any(is.na(base_weight))) {
  stop("Non-numeric weight values found in the base training data file.")
}
train_data <- base_data[base_weight == 1, output_cols, drop = FALSE]
cat(sprintf("Loaded %d training rows (weight = 1) from %s\n",
            nrow(train_data), base_training_path))

# --- Assemble the full GUIDE input: training on top, testing below ----------
rownames(train_data) <- NULL
rownames(combined_testing) <- NULL
full_data <- rbind(train_data, combined_testing)

# --- Write output -------------------------------------------------------------
cat(sprintf("Writing %d training rows + %d testing rows (total %d) to %s\n",
            nrow(train_data), nrow(combined_testing), nrow(full_data), output_file))
write.table(full_data,
            file = output_file,
            sep = "\t",
            quote = FALSE,
            row.names = FALSE,
            col.names = TRUE)

cat("Done.\n")