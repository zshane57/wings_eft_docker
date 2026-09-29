#!/usr/bin/env Rscript
# normalisation_scale_rotate.R
# -----------------------------------------------------------------------------
# Applies three-step EFD normalisation (size, rotation, and starting point)
# to a CSV containing elliptic Fourier coefficients for harmonic 10.
#
# INPUT CSV structure (wide format):
#   id,species,family,a1,a2,...,a10,b1,...,b10,c1,...,c10,d1,...,d10
# Each row represents one shape; a1..a10 etc. are raw (un-normalized) harmonic
# coefficients (Kuhl & Giardina form). Rows may contain additional metadata
# columns before the coefficient block; this script assumes exactly 3 metadata
# columns as described.
#
# OUTPUT:
#   Three data frames / CSVs:
#     *_size_norm.csv      : coefficients after size normalization only.
#     *_rot_size_norm.csv  : coefficients after size + rotation normalization.
#     *_final_norm.csv     : coefficients after all three normalizations.
#   All retain id,species,family and coefficients in SAME ordering
#   (a1..a10,b1..b10,c1..c10,d1..d10) but values replaced by normalized ones.
#
# USAGE EXAMPLES (inside R):
#   source("normalisation_scale_rotate.R")
#   result <- process_efd_csv("efd_coeffs_raw.csv")
#   # result is a list with $size_norm, $rot_size_norm, and $final_norm data.frames
#
# From command line:
#   Rscript normalisation_scale_rotate.R input.csv size_norm_out.csv rot_size_norm_out.csv final_norm_out.csv
# -----------------------------------------------------------------------------

# The helper is sourced directly by the feature script, so we avoid importing
# optional package dependencies that are not needed for the normalization logic.

# ---------- Utility helpers --------------------------------------------------
extract_coeff_vectors <- function(row, H = 10) {
	# row is a single-row data.frame or matrix.
	# returns list(an,bn,cn,dn) numeric length H
	get_seq <- function(prefix) {
		cols <- paste0(prefix, seq_len(H))
		vals <- as.numeric(unlist(row[ , cols, drop = TRUE]))
		if (length(vals) != H || any(is.na(vals))) {
			stop(sprintf("Failed to extract full sequence for prefix '%s'.", prefix))
		}
		vals
	}
	list(
		an = get_seq("a"),
		bn = get_seq("b"),
		cn = get_seq("c"),
		dn = get_seq("d")
	)
}

# ---------- Size normalization (semi-major axis scaling) ---------------------
size_normalize_coeffs <- function(an, bn, cn, dn, tol = 1e-9) {
	a1 <- an[1]; b1 <- bn[1]; c1 <- cn[1]; d1 <- dn[1]
	S1 <- a1^2 + b1^2 + c1^2 + d1^2
	D  <- (a1^2 + c1^2 - b1^2 - d1^2)
	E  <- 2 * (a1 * b1 + c1 * d1)
	disc <- sqrt(D^2 + E^2)
	lambda1 <- (S1 + disc)/2
	if (lambda1 < tol) stop("Degenerate first harmonic: semi-major axis ~ 0")
	A_len <- sqrt(lambda1)
	# Scale all harmonics by A_len
	list(
		an = an / A_len,
		bn = bn / A_len,
		cn = cn / A_len,
		dn = dn / A_len,
		meta = list(semi_major = A_len)
	)
}

# ---------- Rotation normalization (spatial) ---------------------------------
rotation_normalize_coeffs <- function(an, bn, cn, dn) {
	# Left rotation using first harmonic only
	a1 <- an[1]; b1 <- bn[1]; c1 <- cn[1]; d1 <- dn[1]
	A11 <- a1*a1 + b1*b1
	A22 <- c1*c1 + d1*d1
	A12 <- a1*c1 + b1*d1
	phi <- 0.5 * atan2(2 * A12, A11 - A22)
	co <- cos(phi); si <- sin(phi)
	a_rot <- co*an + si*cn
	b_rot <- co*bn + si*dn
	c_rot <- -si*an + co*cn
	d_rot <- -si*bn + co*dn
	applied_left_reflect <- FALSE
	#if (a_rot[1] < 0) {
	#	a_rot <- -a_rot
	#	b_rot <- -b_rot
	#	applied_left_reflect <- TRUE
	#}
	list(
		an = a_rot,
		bn = b_rot,
		cn = c_rot,
		dn = d_rot,
		meta = list(rotation_radians = phi,
								rotation_degrees = phi * 180 / pi,
								applied_left_reflection = applied_left_reflect)
	)
}

# ---------- Starting point normalization (parametric phase) ------------------
starting_point_normalize_coeffs <- function(an, bn, cn, dn, H = 10) {
    # This function aligns the parametric starting point so that the major axis
    # of the first harmonic ellipse corresponds to t=0.
    # It does so by rotating the parametric phase of all harmonics.
    a1 <- an[1]; b1 <- bn[1]
    # Calculate the phase of the first harmonic's x-component
    phi <- atan2(b1, a1)
    
    # We apply a parametric phase shift of -phi to all harmonics.
    # For harmonic 'n', this corresponds to a rotation of -n*phi.
    an_new <- numeric(H)
    bn_new <- numeric(H)
    cn_new <- numeric(H)
    dn_new <- numeric(H)
    
    for (n in 1:H) {
        th <- -n * phi
        co <- cos(th)
        si <- sin(th)
        
        # Apply rotation to coefficients of harmonic n
        an_new[n] <- co * an[n] - si * bn[n]
        bn_new[n] <- si * an[n] + co * bn[n]
        cn_new[n] <- co * cn[n] - si * dn[n]
        dn_new[n] <- si * cn[n] + co * dn[n]
    }
    
    list(
        an = an_new,
        bn = bn_new,
        cn = cn_new,
        dn = dn_new,
        meta = list(start_point_phase_rad = -phi,
                    start_point_phase_deg = -phi * 180 / pi)
    )
}


# ---------- Row processing pipeline ------------------------------------------
normalize_row <- function(row, H = 10) {
	coeffs <- extract_coeff_vectors(row, H)
	size_out <- size_normalize_coeffs(coeffs$an, coeffs$bn, coeffs$cn, coeffs$dn)
	rot_out  <- rotation_normalize_coeffs(size_out$an, size_out$bn, size_out$cn, size_out$dn)
	final_out <- starting_point_normalize_coeffs(rot_out$an, rot_out$bn, rot_out$cn, rot_out$dn, H)
	list(size = size_out, rot_size = rot_out, final = final_out)
}

# ---------- Apply to entire data.frame ---------------------------------------
apply_normalizations <- function(df, H = 10) {
	# df includes metadata columns then coefficients
	n <- nrow(df)
	# Pre-allocate result data.frames
	coeff_cols <- c(paste0("a", 1:H), paste0("b", 1:H), paste0("c", 1:H), paste0("d", 1:H))
	
	# Create empty dataframes for results
	df_size_norm <- df
	df_rot_size_norm <- df
	df_final_norm <- df
	
	# Loop over rows and apply normalizations
	for (i in 1:n) {
		results <- normalize_row(df[i, ], H)
		
		# Populate the dataframes
		df_size_norm[i, coeff_cols] <- c(results$size$an, results$size$bn, results$size$cn, results$size$dn)
		df_rot_size_norm[i, coeff_cols] <- c(results$rot_size$an, results$rot_size$bn, results$rot_size$cn, results$rot_size$dn)
		df_final_norm[i, coeff_cols] <- c(results$final$an, results$final$bn, results$final$cn, results$final$dn)
	}
	
	list(size_norm = df_size_norm, rot_size_norm = df_rot_size_norm, final_norm = df_final_norm)
}

# ---------- Main execution block ---------------------------------------------
process_efd_csv <- function(input_file, size_norm_file, rot_size_norm_file, final_norm_file, H = 10) {
	cat("Reading input file:", input_file, "\n")
	if (!file.exists(input_file)) stop("Input file not found.")
	
	df <- read.csv(input_file, stringsAsFactors = FALSE, check.names = FALSE)
	
	cat("Applying normalizations to", nrow(df), "rows...\n")
	norm_results <- apply_normalizations(df, H)
	
	cat("Writing output file:", size_norm_file, "\n")
	write.csv(norm_results$size_norm, size_norm_file, row.names = FALSE, quote = TRUE)
	
	cat("Writing output file:", rot_size_norm_file, "\n")
	write.csv(norm_results$rot_size_norm, rot_size_norm_file, row.names = FALSE, quote = TRUE)
	
	cat("Writing output file:", final_norm_file, "\n")
	write.csv(norm_results$final_norm, final_norm_file, row.names = FALSE, quote = TRUE)
	
	cat("Done.\n")
	invisible(norm_results)
}

# Command-line execution logic
if (sys.nframe() == 0) {
	# Check if any arguments were provided
	args <- commandArgs(trailingOnly = TRUE)
	
	if (length(args) > 0) {
		# Use optparse if available for nicer CLI
		if (requireNamespace("optparse", quietly = TRUE)) {
			option_list <- list(
				optparse::make_option(c("-i", "--input"), type="character", help="Input CSV file with raw coefficients"),
				optparse::make_option(c("-s", "--size_out"), type="character", help="Output file for size-normalized data"),
				optparse::make_option(c("-r", "--rot_size_out"), type="character", help="Output file for rotation+size-normalized data"),
				optparse::make_option(c("-f", "--final_out"), type="character", help="Output file for final (all three) normalized data")
			)
			parser <- optparse::OptionParser(usage = "%prog -i input.csv -s size.csv -r rot.csv -f final.csv",
																			 option_list = option_list)
			opts <- optparse::parse_args(parser)
			
			if (!is.null(opts$input) && !is.null(opts$size_out) && !is.null(opts$rot_size_out) && !is.null(opts$final_out)) {
				process_efd_csv(opts$input, opts$size_out, opts$rot_size_out, opts$final_out)
				quit(save = "no") # Exit after CLI processing
			}
		} 
		
		# Prevent automatic execution when sourced inside another script.
		if (sys.nframe() == 0 && !interactive() && length(commandArgs(trailingOnly = TRUE)) > 0) {
			args <- commandArgs(trailingOnly = TRUE)
			
			# Only run CLI if explicitly provided 4 arguments
			if (length(args) == 4) {
				process_efd_csv(args[1], args[2], args[3], args[4])
			}
		}
	}
	# If no arguments provided, fall through to the hardcoded calls below
}

# Call process_efd_csv("path/to/file.csv")

path <- "/Users/zshane/Documents/um_msc/appdev/"

#process_efd_csv(paste0(path,"dm_docker_testonly.csv"), 
#                size_norm_file = paste0(path, "dm_size_norm_test.csv"),
#                rot_size_norm_file = paste0(path, "dm_rot_size_norm_test.csv"),
#                final_norm_file = paste0(path, "dm_final_norm_docker_test.csv"))

#process_efd_csv(paste0(path,"pa2r_docker_testonly.csv"), 
#                size_norm_file = paste0(path, "pa2r_size_norm_test.csv"),
#                rot_size_norm_file = paste0(path, "pa2r_rot_size_norm_test.csv"),
#                final_norm_file = paste0(path, "pa2r_final_norm_docker_test.csv"))

#process_efd_csv(paste0(path,"h10_efd_coef_pa2r_docker.csv"), 
#                size_norm_file = paste0(path, "h10_pa2r_size_norm.csv"),
#                rot_size_norm_file = paste0(path, "h10_pa2r_rot_size_norm.csv"),
#                final_norm_file = paste0(path, "h10_pa2r_final_norm_docker.csv"))