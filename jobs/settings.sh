# Settings read by the batch jobs in jobs/, apart from the copy-number jobs, which take theirs from qsub -v.
# Set the paths for your system here, or in a copy of this file given to every job as SETTINGS_FILE
# (qsub -v SETTINGS_FILE=/path/to/copy,...), then check them on the login node:
#   bash jobs/check_settings.sh            (training or pipeline after it checks only that part)
# A value passed with "qsub -v NAME=value" takes precedence over the value here.

# Training data, only read: the pickle of array beta values (docs/TRAINING.md), and optionally a list of
# Sample_IDs to leave out and a CSV of groups for grouped cross-validation (scripts/check_data.py and
# scripts/merge_groups.py write such files). Leave the last two empty when not used.
: "${DATA_PATH:=/path/to/arrays.pkl}"
: "${EXCLUDE_IDS=}"
: "${GROUPS_FILE=}"
# Classes left out of the model, by prefix, for scripts/check_data.py; the training recipe lists its own.
: "${EXCLUDE_PREFIXES:=MPAL AML_NOS B-ALL_NOS}"

: "${RUNS_DIR:=$HOME/sparsh_runs}"     # every training run gets its own new folder here
: "${CONDA_ENV:=sparsh}"              # environment with the packages of requirements.txt
: "${CONDA_BASE:=$HOME/miniconda3}"   # where conda is installed
: "${CUDA_MODULE=}"                   # environment module loaded before activation, for example cuda/12.3

# The single-sample pipeline (scripts/run_sample.py, jobs/run_sample.pbs; docs/PIPELINE.md). Leave CN_PANEL empty
# to call copy number without a stored reference (a first look only). SPARSH was tested with modkit 0.6.1 and
# dorado 1.3.0; dorado is needed only for reads that are not yet aligned to REFERENCE_FASTA, and minimap2 only with
# ALIGNER=minimap2 (leave MINIMAP2 empty to take minimap2 from PATH).
: "${REFERENCE_FASTA:=/path/to/hg19.fasta}"         # GRCh37, with its .fai index
: "${EPIC_MANIFEST:=/path/to/EPIC-8v2-0_A2.csv}"    # Illumina's MethylationEPIC v2.0 manifest
: "${PIPELINE_MODEL:=$RUNS_DIR/sparsh_model_0.4.0}" # the model folder
: "${CN_PANEL=}"                                    # stored copy-number reference (scripts/karyotype.py panel)
: "${SAMTOOLS:=samtools}"
: "${MODKIT:=modkit}"
: "${DORADO:=dorado}"
: "${MINIMAP2=}"
