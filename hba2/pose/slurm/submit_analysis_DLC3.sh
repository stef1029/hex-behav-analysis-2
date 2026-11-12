#!/bin/bash -l

#SBATCH --partition=agpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL

# Create files group-writable (umask 002 -> 664) so other lab members can read
# and overwrite this job's logs and DLC analysis outputs. Inherited by the
# environment and batch_analyse.py launched below.
umask 002

# Define the index of the video this particular job should process
VIDEO_INDEX=$SLURM_ARRAY_TASK_ID

# Get the video list file from environment variable (set by the unified submission
# script)
VIDEO_LIST="${DLC_VIDEO_LIST_FILE:-videos_to_analyse.tsv}"

# Get the model config from environment variable (set by the unified submission script)
MODEL_CONFIG="${DLC_MODEL_CONFIG}"
if [ -z "$MODEL_CONFIG" ]; then
    echo "ERROR: DLC_MODEL_CONFIG is not set; it names the model's config.yaml"
    exit 1
fi

# One line per video: video path, destination folder, session ID, tab separated.
# make_vid_list.py fills all three because only the cohort knows the last two: the
# destination depends on the model, and the session ID cannot be recovered from the
# video path in a nested session, where the video sits in raw/ rather than the session
# folder. "cut -s" returns nothing when there are no tabs, so a hand-written list of
# bare video paths still works — with the old, layout-dependent guesses.
LINE=$(sed -n "${VIDEO_INDEX}p" "$VIDEO_LIST")
VIDEO_PATH=$(printf '%s' "$LINE" | cut -f1)
DEST_FOLDER=$(printf '%s' "$LINE" | cut -s -f2)
SESSION_ID=$(printf '%s' "$LINE" | cut -s -f3)

if [ -z "$SESSION_ID" ]; then
    SESSION_ID=$(basename "$(dirname "$VIDEO_PATH")")
fi

echo "Processing video: $VIDEO_PATH"
echo "Session: $SESSION_ID"
echo "Using model config: $MODEL_CONFIG"
echo "Destination: ${DEST_FOLDER:-<beside the video>}"

# Get the current directory where the script is running from
CURRENT_DIR=$(pwd)

# Create a temporary log file using session ID
TEMP_LOG="${CURRENT_DIR}/${SESSION_ID}_temp.log"

# Redirect all output to the temporary log file
exec > "$TEMP_LOG" 2>&1

echo "Starting job at $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Video: $VIDEO_PATH"
echo "Session ID: $SESSION_ID"
echo "Model Config: $MODEL_CONFIG"
echo "Destination: ${DEST_FOLDER:-<beside the video>}"

# Load CUDA module and capture which version was loaded
echo "=== Loading CUDA Module ==="
if module load cuda/12.8 2>/dev/null; then
    echo "Successfully loaded CUDA 12.8"
    CUDA_MODULE_VERSION="12.8"
elif module load cuda/12.4 2>/dev/null; then
    echo "Successfully loaded CUDA 12.4"
    CUDA_MODULE_VERSION="12.4"
elif module load cuda/11.8 2>/dev/null; then
    echo "Successfully loaded CUDA 11.8"
    CUDA_MODULE_VERSION="11.8"
elif module load cuda/12.1 2>/dev/null; then
    echo "Successfully loaded CUDA 12.1"
    CUDA_MODULE_VERSION="12.1"
else
    echo "WARNING: No CUDA module could be loaded"
    CUDA_MODULE_VERSION="None"
fi

# Print system CUDA version if available
echo "=== System CUDA Version Check ==="
if command -v nvcc &> /dev/null; then
    echo "NVCC version:"
    nvcc --version | grep "release" || echo "Could not parse NVCC version"
else
    echo "NVCC not found in PATH"
fi

# Show what SLURM has set for GPU visibility
echo "=== SLURM GPU Configuration ==="
echo "SLURM GPU allocation: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES (set by SLURM): $CUDA_VISIBLE_DEVICES"
echo "Loaded CUDA module version: $CUDA_MODULE_VERSION"

# Don't override CUDA_VISIBLE_DEVICES - let SLURM handle it
# SLURM automatically sets this when you use --gres=gpu:1

# The scripts directory comes from DLC_SCRIPTS_DIR, set by whatever submitted this
# array. Without it this script looks beside itself, which only works when it is run
# directly: as an array task it runs from a Slurm spool copy with no siblings.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
SCRIPTS_DIR="${DLC_SCRIPTS_DIR:-$SELF_DIR}"

if [ ! -f "$SCRIPTS_DIR/batch_analyse.py" ]; then
    echo "ERROR: batch_analyse.py not found in '$SCRIPTS_DIR'. Set DLC_SCRIPTS_DIR."
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_SCRIPTS_DIR_FAILED.log"
    exec >&- 2>&-
    mv "$TEMP_LOG" "$FINAL_LOG"
    exit 1
fi
echo "Scripts directory: $SCRIPTS_DIR"

# The environment: HBA2_ENV, which whatever submitted this array hands on (its --env).
source "$SCRIPTS_DIR/activate_env.sh"
if ! activate_env; then
    echo "ERROR: could not activate the environment"
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_ENV_FAILED.log"
    exec >&- 2>&-
    mv "$TEMP_LOG" "$FINAL_LOG"
    exit 1
fi

# Enhanced CUDA and PyTorch version check
echo "=== PyTorch and CUDA Compatibility Check ==="
python -c "
import torch
import os
import sys

print('=== PyTorch Version Information ===')
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')

print('\n=== CUDA Availability ===')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count visible to PyTorch: {torch.cuda.device_count()}')
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')

if torch.cuda.is_available():
    print('\n=== GPU Information ===')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  - Compute capability: {props.major}.{props.minor}')
        print(f'  - Total memory: {props.total_memory / 1024**3:.2f} GB')
        
    # Fail fast if this PyTorch build has no kernels for the allocated GPU.
    # Allocating a tensor does not launch a compiled kernel, so a build that
    # lacks the GPU's sm_XX target passes a tensor-creation test and then dies
    # mid-inference with 'no kernel image is available' (seen on RTX 5090 /
    # sm_120 nodes fmg59-68 with a cu126 torch build). A non-zero exit here
    # renames the log to *_CUDA_FAILED.log via the check below.
    # A GPU runs kernels built for its own major version at an equal or lower minor
    # (CUDA binary compatibility): an RTX 4090 (sm_89) runs a build's sm_86 kernels.
    # Requiring an exact match turned 4090 nodes away from a cu128 build that serves
    # them; the cu126-on-sm_120 failure above still fails, having no sm_12x at all.
    major, minor = torch.cuda.get_device_capability(0)
    gpu_arch = f'sm_{major}{minor}'
    arch_list = torch.cuda.get_arch_list()
    runnable = [a for a in arch_list
                if a.startswith(f'sm_{major}') and len(a) == len(gpu_arch)
                and int(a[len(f'sm_{major}'):]) <= minor]
    print('\n=== GPU Architecture Check ===')
    print(f'GPU arch: {gpu_arch}')
    print(f'PyTorch compiled arch list: {arch_list}')
    if not runnable:
        print(f'\n✗ This PyTorch build has no kernels {gpu_arch} can run (none for sm_{major}x at or below it). Install a build that targets it (a cu128 wheel for sm_120).')
        sys.exit(1)
    print(f'✓ {gpu_arch} runs this build\'s {runnable[-1]} kernels')

    # Launch a real kernel and synchronise so any launch failure surfaces now.
    try:
        x = torch.randn(1, 3, 64, 64, device='cuda')
        w = torch.randn(8, 3, 3, 3, device='cuda')
        y = torch.nn.functional.conv2d(x, w)
        torch.cuda.synchronize()
        print(f'✓ CUDA kernel launch test (conv2d) passed, output shape {tuple(y.shape)}')
    except Exception as e:
        print(f'\n✗ CUDA kernel launch test failed: {e}')
        sys.exit(1)
else:
    print('\n✗ CUDA is not available to PyTorch')
    sys.exit(1)
"

# Check if the Python CUDA check failed
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA availability check failed"
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_CUDA_FAILED.log"
    exec >&- 2>&-
    mv "$TEMP_LOG" "$FINAL_LOG"
    exit 1
fi

# Run the Python script - pass GPU ID as 0 since SLURM maps the allocated GPU to index 0
# Also pass the model config path as the third argument
cd "$SCRIPTS_DIR"
echo "Running batch_analyse.py..."
echo "Arguments: $VIDEO_PATH 0 $MODEL_CONFIG $DEST_FOLDER"
python batch_analyse.py "$VIDEO_PATH" 0 "$MODEL_CONFIG" "$DEST_FOLDER"
PYTHON_EXIT_CODE=$?

# Check if the Python script failed and rename the log file accordingly
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script failed with exit code $PYTHON_EXIT_CODE"
    echo "Job failed at $(date)"
    
    # Rename the temporary log file to include FAILED in the name
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_FAILED.log"
else
    echo "Python script completed successfully"
    echo "Job completed at $(date)"
    
    # Rename the temporary log file to include SUCCESS in the name
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_SUCCESS.log"
fi

# Close the file descriptors before moving the file
exec >&- 2>&-

# Move the temporary log to the final log with the appropriate status in the filename
mv "$TEMP_LOG" "$FINAL_LOG"

# Make the log group-owned + group-writable for other lab members (best-effort).
chgrp trilab "$FINAL_LOG" 2>/dev/null || true
chmod g+rw "$FINAL_LOG" 2>/dev/null || true

# Exit with the Python script's exit code
exit $PYTHON_EXIT_CODE