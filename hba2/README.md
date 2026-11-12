# hba2

From what the hex rig records to one table of trials.

The rig writes a DAQ file, a video, a weight log and its own account of each trial, each
on its own clock. This package puts them all on the DAQ's clock, adds where the mouse's
head was from DeepLabCut, writes one NWB file per session, and reads those files back as
a single pandas table with one row per trial. That table is where it hands over. What you
do with it after that is up to you: plain pandas, or a plotting library of your own built
on top.

## Starting a notebook

```python
from hba2 import load_sessions, qc_trials

trials = load_sessions("/path/to/cohort", include=True)
trials = qc_trials(trials, min_ear_likelihood=0.6, max_abs_frame_lag_s=0.1)

kept = trials[trials["keep"] == 1]
kept.groupby(["mouse_id", "laser_stim"])["correct"].mean()
```

- `load_sessions` picks sessions by the columns of the cohort's `sessions.csv`
  (`include=True`, `mouse=["M1", "M2"]`, `min_date="2026-06-01"`) and returns every
  trial in them, with each session's labels as columns.
- `qc_trials` removes nothing. It marks the trials not to use in a `keep` column, and
  prints what each rule cost.

[Loading data](docs/loading.md) has the rest, and
[The NWB files](docs/nwb-files.md#the-trials-table) lists every column.

## What it does, in order

```
raw session ─► postprocess ─► processed/ ─► DeepLabCut ─► pose ─► build_nwb ─► <session>.nwb
                (rig PC)                     (cluster)  (cluster)  (cluster)        │
                                                                                     ▼
                                                     load_sessions ─► ONE TRIALS TABLE
```

| stage | what it makes | where it runs |
|---|---|---|
| `postprocess_cohort` | `processed/`: events, trials, frame times, scales, laser pulses, all on the DAQ clock | the rig PC, from the control app, before the cohort is copied to CephFS |
| DeepLabCut (`pose/run_dlc_pipeline.py`) | `dlc/<model>/`: body-part coordinates per frame | the cluster, submitted by the control app |
| `build_pose` | head position, bearing and the angle to every port, per frame and per trial | the cluster, after DeepLabCut |
| `build_index` | `sessions.csv`: one row per session, with your labels | the cluster, after pose; or by hand |
| `build_nwb` | one NWB file per session, in a named batch | the cluster, after the index |
| `process_ephys_cohort` | spike sorts → good units, on the DAQ clock | by hand, per ephys cohort |
| `load_sessions` | the trials table | your notebook |

Each stage skips what is already up to date, so running it again costs a scan rather than
a rebuild. [The pipeline](docs/pipeline.md) covers how each one is run.

## The package

```
hba2/
├── cohort/           where every file is: Cohort, sessions.csv, neurons.csv
├── postprocessing/   raw rig output onto one clock
│   └── repairs/      one-off fixes for sessions recorded before a bug was found
├── pose/             DeepLabCut on the cluster, then head angles
│   └── slurm/        the cluster job scripts
├── nwb/              build_nwb: the files everything is read from
├── load/             the files to one trials table; QC; events, notes, spikes
├── ephys/            spike sorts, curations, good units, neuron names
└── legacy/           2024–2025 cohorts, converted into the same processed/ files
```

## Documentation

- [Principles](docs/principles.md): the handful of rules the package is built on, and why
- [The pipeline](docs/pipeline.md): running each stage for a cohort, and what to do after
  a bug
- [The cohort folder](docs/cohort-folder.md): the layout on disk, `Cohort`, and the two
  label tables
- [The processed files](docs/processed-files.md): what post-processing writes, column by
  column
- [The NWB files](docs/nwb-files.md): batches, what is inside a file, and the trials
  table's columns
- [Loading data](docs/loading.md): `load_sessions`, QC, events, notes, frames and spikes
- [pose/README.md](pose/README.md): DeepLabCut and the head-angle stage in detail
- [ephys/\_\_init\_\_.py](ephys/__init__.py): the ephys layers, and
  [ephys/TOOLS.md](ephys/TOOLS.md) for other labs' tools that read these files

Every module also opens with a docstring that explains what it does and why it is built
that way. Treat those docstrings as the detailed reference; these pages give the overview.

AI agents working in this package: read [AGENTS.md](AGENTS.md) first.

## Installing

With [uv](https://docs.astral.sh/uv/), from the folder holding `pyproject.toml`:

```
uv venv --python 3.10 .venv
uv pip install -e .                                  # loading and post-processing
uv pip install -e ".[dlc]" --torch-backend=cu128     # plus DeepLabCut, for the cluster
```

- **`[dlc]` pins one DeepLabCut version.** Every lab member's analysis then runs the
  same DeepLabCut. It also restricts the environment to what DeepLabCut allows: numpy
  below 2 and matplotlib below 3.9.
- **`--torch-backend`** picks the PyTorch build for the cluster's CUDA. Without it, a
  plain install may fetch one that does not match the GPUs.
- **On the cluster,** name the environment when you submit (`--env /path/to/.venv`,
  or `env=` from Python), and every job in the chain runs in it. Setting `HBA2_ENV`
  in `~/.bashrc` instead makes it the default. See
  [the pipeline](docs/pipeline.md#2-deeplabcut-and-pose-where-the-head-was).

Inside the lab's own repository it is installed from `hex_behav_analysis/` instead.
