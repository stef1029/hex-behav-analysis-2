#!/bin/bash -l

#SBATCH --job-name=DLC_unified_pipeline
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Create files group-writable (umask 002 -> 664 files / 775 dirs) so other
# lab members can read and overwrite the pipeline's outputs on their own runs. Inherited by all child processes (make_vid_list.py,
# the array job, batch_analyse.py).
umask 002

# Where the pipeline's scripts are: --scripts-dir, which run_dlc_pipeline.py always
# passes. Without it, this folder, which only works when this script is run directly:
# under sbatch it runs from a Slurm spool copy with no siblings.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
SCRIPTS_BASE_DIR="$SELF_DIR"

# =============================================================================
# Unified DeepLabCut Analysis Pipeline
# =============================================================================
# Usage (cohort mode - generates video list automatically):
#   sbatch unified_DLC_analysis_submission.sh --cohort COHORT_FOLDER --config MODEL_CONFIG [--gpus NUM_GPUS] [--env ENV]
#
# Usage (videolist mode - uses provided video list):
#   sbatch unified_DLC_analysis_submission.sh --videolist VIDEO_LIST_FILE --config MODEL_CONFIG [--gpus NUM_GPUS] [--env ENV]
#
# --env is the environment this job and the array it submits run in (see
# activate_env.sh); without it, HBA2_ENV from the submitting shell.
#
# Examples:
#   sbatch unified_DLC_analysis_submission.sh --cohort /path/to/cohort --config /path/to/config.yaml
#   sbatch unified_DLC_analysis_submission.sh --videolist /path/to/my_videos.txt --config /path/to/config.yaml --gpus 4
# =============================================================================

# Function to display usage
show_usage() {
    echo "Usage:"
    echo "  Cohort mode (auto-generate video list):"
    echo "    sbatch $0 --cohort COHORT_FOLDER --config MODEL_CONFIG [--gpus NUM_GPUS]"
    echo ""
    echo "  Video list mode (use provided video list):"
    echo "    sbatch $0 --videolist VIDEO_LIST_FILE --config MODEL_CONFIG [--gpus NUM_GPUS]"
    echo ""
    echo "Options:"
    echo "  --cohort     Path to cohort folder (will generate video list automatically)"
    echo "  --videolist  Path to existing video list (skips generation). Give --cohort"
    echo "               as well and per-session logs still land in the cohort folder."
    echo "  --config     Path to the DLC model's config.yaml (required)"
    echo "  --gpus       Number of GPUs to use (default: 8)"
    echo "  --env        venv folder or conda env name to run in (default: \$HBA2_ENV)"
    echo "  --scripts-dir Directory containing the pipeline scripts (default: beside this script)"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 --cohort /path/to/cohort --config /path/to/config.yaml"
    echo "  sbatch $0 --videolist /path/to/videos.tsv --cohort /path/to/cohort --gpus 4"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    show_usage
    exit 1
fi

# Initialize variables
COHORT_FOLDER=""
VIDEO_LIST_FILE=""
MODEL_CONFIG=""
NUM_GPUS=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cohort)
            COHORT_FOLDER="$2"
            shift 2
            ;;
        --videolist)
            VIDEO_LIST_FILE="$2"
            shift 2
            ;;
        --config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --env)
            # Exported, so the array submitted below inherits it too.
            export HBA2_ENV="$2"
            shift 2
            ;;
        --scripts-dir)
            SCRIPTS_BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            show_usage
            exit 1
            ;;
    esac
done

# Either is enough on its own, and both together is the normal case when
# run_dlc_pipeline.py submits: it has already built the list, and names the cohort so
# that per-session logs still land there beside everything else about the cohort.
if [ -z "$COHORT_FOLDER" ] && [ -z "$VIDEO_LIST_FILE" ]; then
    echo "Error: Must specify --cohort, --videolist, or both"
    show_usage
    exit 1
fi

echo "=== DeepLabCut Unified Pipeline ==="
echo "Starting at: $(date)"
echo "Cohort folder: ${COHORT_FOLDER:-<none>}"
echo "Video list file: ${VIDEO_LIST_FILE:-<will be generated>}"
echo "Model config: $MODEL_CONFIG"
echo "Number of GPUs: $NUM_GPUS"
echo "Environment: ${HBA2_ENV:-<not set>}"
echo "Running on node: $(hostname)"

# Check if model config exists
if [ -z "$MODEL_CONFIG" ]; then
    echo "Error: --config is required"
    show_usage
    exit 1
fi
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Error: Model config file does not exist: $MODEL_CONFIG"
    exit 1
fi

# =============================================================================
# Step 1: Generate or validate video list
# =============================================================================
if [ -z "$VIDEO_LIST_FILE" ]; then
    # No list handed in: build one from the cohort.
    echo ""
    echo "=== Step 1: Generating video list from cohort folder ==="

    # Check if cohort folder exists
    if [ ! -d "$COHORT_FOLDER" ]; then
        echo "Error: Cohort folder does not exist: $COHORT_FOLDER"
        exit 1
    fi

    source "$SCRIPTS_BASE_DIR/activate_env.sh"
    activate_env || exit 1

    # Use the configured script directory
    SCRIPT_DIR="$SCRIPTS_BASE_DIR"
    echo "Script directory: $SCRIPT_DIR"

    # Check if make_vid_list.py exists
    if [ ! -f "$SCRIPT_DIR/make_vid_list.py" ]; then
        echo "Error: make_vid_list.py not found in $SCRIPT_DIR"
        echo "Pass --scripts-dir with the folder this pipeline's scripts are in"
        echo "Contents of script directory:"
        ls -la "$SCRIPT_DIR" 2>/dev/null || echo "Directory does not exist"
        exit 1
    fi

    # The model is identified by the Task name inside its config, so the config is
    # the only thing that needs naming. The list goes in the cohort folder, beside the
    # logs the run will produce.
    VIDEO_LIST_FILE="$COHORT_FOLDER/dlc_videos.tsv"
    python "$SCRIPT_DIR/make_vid_list.py" "$COHORT_FOLDER" "$MODEL_CONFIG" "$VIDEO_LIST_FILE"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate video list"
        exit 1
    fi

    # Check if the video list file was created
    if [ ! -f "$VIDEO_LIST_FILE" ]; then
        echo "Error: $(basename $VIDEO_LIST_FILE) not found at $VIDEO_LIST_FILE"
        echo "Contents of cohort directory:"
        ls -la "$COHORT_FOLDER/"
        exit 1
    fi

    # Make the generated list group-owned + group-writable so other lab
    # members can overwrite it on their own pipeline runs (best-effort: only
    # works if we own the file, which we do when we just created it).
    chgrp trilab "$VIDEO_LIST_FILE" 2>/dev/null || true
    chmod g+rw "$VIDEO_LIST_FILE" 2>/dev/null || true

else
    echo ""
    echo "=== Step 1: Using provided video list ==="

    if [ ! -f "$VIDEO_LIST_FILE" ]; then
        echo "Error: Video list file does not exist: $VIDEO_LIST_FILE"
        exit 1
    fi

    echo "Video list file: $VIDEO_LIST_FILE"
fi

# Per-session logs are written to the job's working directory. Prefer the cohort, so
# everything about a run — the list, every log — sits in one place and can be cleared
# out in one go.
WORKING_DIR="${COHORT_FOLDER:-$(dirname "$VIDEO_LIST_FILE")}"

echo "Working directory: $WORKING_DIR"

# Step 2: Count videos
NUM_LINES=$(wc -l < "$VIDEO_LIST_FILE")
echo "Found $NUM_LINES videos to analyse"

if [ $NUM_LINES -eq 0 ]; then
    echo "No videos to analyse. Exiting."
    exit 0
fi

# Step 3: Submit array job
echo ""
echo "=== Step 2: Submitting analysis jobs ==="
echo "Submitting array job for $NUM_LINES videos using $NUM_GPUS GPUs"

# Ensure SCRIPT_DIR is set for videolist mode
SCRIPT_DIR="$SCRIPTS_BASE_DIR"

ANALYSIS_SCRIPT="$SCRIPT_DIR/submit_analysis_DLC3.sh"

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Analysis script not found: $ANALYSIS_SCRIPT"
    exit 1
fi

# Export environment variables for the analysis script
export DLC_VIDEO_LIST_FILE="$(basename $VIDEO_LIST_FILE)"
export DLC_MODEL_CONFIG="$MODEL_CONFIG"
export DLC_SCRIPTS_DIR="$SCRIPTS_BASE_DIR"

echo "Exported DLC_MODEL_CONFIG: $DLC_MODEL_CONFIG"

JOB_SUBMISSION=$(sbatch --array=1-${NUM_LINES}%${NUM_GPUS} --chdir="$WORKING_DIR" --parsable "$ANALYSIS_SCRIPT")
ARRAY_JOB_ID=$(echo $JOB_SUBMISSION | cut -d';' -f1)

echo "Array job submitted with ID: $ARRAY_JOB_ID"
echo "Job range: 1-$NUM_LINES with max $NUM_GPUS concurrent jobs"
echo "Video list file: $VIDEO_LIST_FILE"
echo "Model config: $MODEL_CONFIG"

echo ""
echo "=== Monitoring Commands ==="
echo "Check job status: squeue -j $ARRAY_JOB_ID"
echo "Check detailed status: scontrol show job $ARRAY_JOB_ID"
echo "Cancel jobs: scancel $ARRAY_JOB_ID"
echo ""
echo "=== Log Files ==="
echo "Individual job logs will be created in: $WORKING_DIR"
echo ""
echo "Pipeline setup completed at: $(date)"