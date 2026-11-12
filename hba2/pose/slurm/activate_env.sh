# Sourced by the job scripts beside it; defines take_env_arg and activate_env.
#
# The environment hba2 runs in is a virtual environment's folder (made with uv or venv)
# or a conda environment's name. Whoever submits names it with --env:
#
#   sbatch cohort_pipeline.sh --scripts-dir SLURM_DIR --env ~/envs/hba2 COHORT ...
#
# The script exports it as HBA2_ENV, and sbatch hands a job's variables on to the jobs
# it submits, so the DeepLabCut array and the after-DLC job run in the same one without
# being told again. Without --env, HBA2_ENV from the submitting shell is used, so
# setting it in ~/.bashrc on the cluster gives a default.
#
#   uv venv --python 3.10 ~/envs/hba2
#   uv pip install --python ~/envs/hba2 -e "<checkout>[dlc]" --torch-backend=cu128
#
# Usage, in a job script:
#   source "$SCRIPTS_DIR/activate_env.sh"
#   take_env_arg "$@" || exit 1
#   set -- "${ARGS_LEFT[@]}"
#   activate_env || exit 1
#
# With neither --env nor HBA2_ENV the job stops: there is no default environment.

# Take --env VALUE out of the arguments, wherever it is, and export it as HBA2_ENV.
# The other arguments are left, in order, in the array ARGS_LEFT.
take_env_arg() {
    ARGS_LEFT=()
    while [ $# -gt 0 ]; do
        if [ "$1" = "--env" ]; then
            if [ -z "$2" ]; then
                echo "Error: --env needs a venv folder or a conda env name"
                return 1
            fi
            export HBA2_ENV="$2"
            shift 2
        else
            ARGS_LEFT+=("$1")
            shift
        fi
    done
}

activate_env() {
    local env="$HBA2_ENV"
    if [ -z "$env" ]; then
        echo "Error: no environment. Pass --env with a venv folder or a conda env name,"
        echo "or set HBA2_ENV in the submitting shell."
        return 1
    fi
    if [ -f "$env/bin/activate" ]; then
        source "$env/bin/activate" || return 1
    else
        conda activate "$env" || return 1
    fi
    echo "Environment: $env"
    echo "Python: $(command -v python)"
}
