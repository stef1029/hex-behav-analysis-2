#!/bin/bash -l

#SBATCH --job-name=cohort_pipeline
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# =============================================================================
# Everything a cohort still needs, submitted from the cluster
# =============================================================================
# What the control app's post-processing submits once a cohort is on CephFS. It does
# no analysis itself: it runs run_dlc_pipeline.py's command line, which works out what
# still needs DeepLabCut, submits an array per model, and (with --then-nwb) one job for
# pose and NWB that waits for them. It is a job rather than a command over ssh so that
# reading a large cohort happens here, not on the login node, and the submission
# returns at once.
#
# Usage:
#   sbatch --chdir=COHORT cohort_pipeline.sh --scripts-dir SLURM_DIR COHORT \
#       --mouse-model T1=/path/config.yaml [--mouse-model ...] [--gpus 8] [--then-nwb] \
#       [--env ENV]
#
# --scripts-dir comes first and is required: under sbatch this runs from a Slurm spool
# copy, so it cannot find its own folder. --env, anywhere after it, is the environment
# every job in the chain runs in (see activate_env.sh). Everything else is handed to
# run_dlc_pipeline.py unchanged.
# =============================================================================

umask 002

if [ "$1" != "--scripts-dir" ] || [ -z "$2" ]; then
    echo "Error: first argument must be --scripts-dir SLURM_DIR"
    exit 1
fi
SCRIPTS_DIR="$2"
shift 2

PIPELINE="$SCRIPTS_DIR/../run_dlc_pipeline.py"
if [ ! -f "$PIPELINE" ]; then
    echo "Error: run_dlc_pipeline.py not found beside $SCRIPTS_DIR"
    exit 1
fi

echo "=== Cohort pipeline ==="
echo "Starting at: $(date)"
echo "Running on node: $(hostname)"

source "$SCRIPTS_DIR/activate_env.sh"
take_env_arg "$@" || exit 1
set -- "${ARGS_LEFT[@]}"
activate_env || exit 1

python "$PIPELINE" "$@" --scripts-dir "$SCRIPTS_DIR"
EXIT_CODE=$?

echo "Finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
