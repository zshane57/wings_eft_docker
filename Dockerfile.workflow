# --------------------------------------------------------------
# Forensic Fly Wing Species Classification — Workflow image
# Tasks 2-5: EFD -> normalisation -> LDA -> GUIDE classification
# Base: ubuntu:22.04 (jammy) + R (CRAN apt repo) + GUIDE Ubuntu22 binary
#
# Build (Apple Silicon needs the amd64 platform for the GUIDE binary):
#   docker build --platform linux/amd64 -f Dockerfile.workflow -t flywing-workflow .
#
# Run:
#   docker run --rm flywing-workflow \
#       ./run_workflow.sh contours_coordinates_cvicina.csv C.vicina
# --------------------------------------------------------------

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- 1. System dependencies (spatial / graphics stack for Momocs & sf) ----
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    software-properties-common \
    dirmngr \
    gnupg \
    ca-certificates \
    wget \
    gzip \
    git \
    build-essential \
    cmake \
    gfortran \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libudunits2-dev \
    libssl-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libgit2-dev \
    libuv1-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libx11-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. R (current release from the CRAN apt repository for jammy) ----
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
        | tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc > /dev/null \
    && echo "deb [signed-by=/etc/apt/trusted.gpg.d/cran_ubuntu_key.asc] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
        > /etc/apt/sources.list.d/cran-r.list \
    && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- 3. GUIDE binary (Ubuntu 22 build) ----
ARG GUIDE_URL=https://pages.stat.wisc.edu/~loh/treeprogs/guide/linux/64bit/Ubuntu22/guide.gz
RUN cd /usr/local/bin \
    && wget -q "$GUIDE_URL" -O guide.gz \
    && gunzip -f guide.gz \
    && chmod +x guide \
    && ls -la /usr/local/bin/guide

# ---- 4. R packages: Posit binary repo (jammy) + Momocs from GitHub ----
RUN Rscript -e " \
  options(repos = c(CRAN = 'https://packagemanager.posit.co/cran/__linux__/jammy/latest')); \
  install.packages(c('remotes', 'optparse', 'sf', 's2', 'ape', 'jpeg', \
                     'ggplot2', 'plotly', 'rgl', 'htmlwidgets', 'fs')); \
  remotes::install_github('MomX/Momocs', upgrade = 'never'); \
  suppressMessages(library(Momocs)); \
  cat('Momocs', as.character(packageVersion('Momocs')), 'OK\n')"

# ---- 5. Pipeline assets ----
WORKDIR /app
COPY ["normalisation_scale_rotate_phase.R", "/app/"]
COPY process_features.R run_workflow.sh /app/
COPY lda_loadings_fold_1_dm.txt lda_loadings_fold_1_pa2r.txt /app/
COPY mean_ld_train_fd_1_dm.rds mean_ld_train_fd_1_pa2r.rds /app/
COPY dm_testing.csv pa2r_testing.csv /app/
COPY guide_input/ /app/guide_input/
# Any sample CSV following the naming convention is baked in at build time:
# drop new files (contours_coordinates*.csv) into this directory and rebuild.
COPY contours_coordinates*.csv /app/
RUN chmod +x /app/run_workflow.sh \
    && rm -f /app/guide_input/*.out /app/guide_input/*prob*.txt /app/guide_input/guide_run.log \
    && rm -f "/app/guide_input/.DS_Store"

# ---- 6. Default entrypoint ----
ENTRYPOINT ["/app/run_workflow.sh"]
CMD ["contours_coordinates_cvicina.csv", "C.vicina"]
