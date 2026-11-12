#!/bin/bash -l

#SBATCH --job-name=after_dlc
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# =============================================================================
# Pose, the session index and NWB for a cohort, once DeepLabCut has finished
# =============================================================================
# Submitted by run_dlc_pipeline.py with --dependency=afterany on the cohort's
# DeepLabCut arrays, so it runs whatever happened to individual videos: sessions with
# output are built, sessions without are skipped with a reason.
#
# Usage:
#   sbatch [--dependency=afterany:ID[:ID...]] after_dlc.sh --scripts-dir SLURM_DIR \
#       COHORT --mouse-model T1=/path/config.yaml [--mouse-model ...] [--env ENV]
#
# --scripts-dir comes first and is required, for the same reason as in
# cohort_pipeline.sh. --env is the environment to run in; without it, the one the
# submitting job ran in (see activate_env.sh). Everything else is handed to after_dlc.py
# unchanged.
# =============================================================================

umask 002

if [ "$1" != "--scripts-dir" ] || [ -z "$2" ]; then
    echo "Error: first argument must be --scripts-dir SLURM_DIR"
    exit 1
fi
SCRIPTS_DIR="$2"
shift 2

AFTER_DLC="$SCRIPTS_DIR/../after_dlc.py"
if [ ! -f "$AFTER_DLC" ]; then
    echo "Error: after_dlc.py not found beside $SCRIPTS_DIR"
    exit 1
fi

echo "=== After DeepLabCut: pose, index, NWB ==="
echo "Starting at: $(date)"
echo "Running on node: $(hostname)"

source "$SCRIPTS_DIR/activate_env.sh"
take_env_arg "$@" || exit 1
set -- "${ARGS_LEFT[@]}"
activate_env || exit 1

python "$AFTER_DLC" "$@"
EXIT_CODE=$?

echo "Finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
