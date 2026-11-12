# The pipeline

How a cohort gets from raw rig output to NWB files, and how to run each step. For a
normal cohort on the current rigs most of this happens without you: the control app runs
post-processing and submits the cluster stages.

```
raw session ─► postprocess ─► processed/ ─► DeepLabCut ─► pose ─► index ─► build_nwb
```

Every stage:

- takes a cohort folder, or a `Cohort`
- skips the sessions whose output is already up to date
- reports what it skipped and why, rather than stopping at the first bad session
- is a function with every choice as an argument, and has a command line
  (`python <stage>.py --help`)

So re-running a stage is always safe, and it is the normal way to pick up sessions that
have just arrived.

**The package holds no lab's choices.** No cohort, model, account or host is written into
it. Keep yours in scripts of your own, outside the package, that call these functions
with every argument visible. Those scripts are what you edit and run day to day.

## 1. Post-processing: onto one clock

```python
from hba2 import Cohort, postprocess_cohort

result = postprocess_cohort(Cohort("/path/to/cohort"))
print(result.describe())
```

For every session that has not been processed yet, this writes `<session>/processed/`:

- the behaviour events
- the trial windows
- the video frame times
- the weight readings
- the laser pulses
- the protocol's notes
- the session's metadata

Everything except the notes is on the DAQ clock. [The processed files](processed-files.md)
lists the files column by column.

- **There is nothing to configure.** The DAQ records the same 24 channels every
  session, so a channel's role follows from its name (`channel_role` in
  [`postprocessing/pipeline.py`](../postprocessing/pipeline.py)). Three channels are
  required, because each is the only way to place one stream on the clock: `CAMERA`,
  `SCALES` and `DAQ_LINK1` (the trial marker).
- **Where it runs.** On the rig PC, from the control app's post-processing panel, before
  the cohort is copied to CephFS. It can also be run anywhere by hand, normally from a
  short script per cohort that calls it with that cohort's path.
- **Options.** `only=(...)` limits it to some sessions. `reprocess=True` redoes sessions
  that already have output.
- **Ephys cohorts.** It also writes `ephys_sync.csv`, which maps the Open Ephys clock
  onto the DAQ's. That means one pass over a recording of about 60 GB, so it is done
  only when the file is missing. `sync_ephys(record)` redoes it on its own.

## 2. DeepLabCut and pose: where the head was

Both run on the cluster.

- **DeepLabCut.** [`pose/run_dlc_pipeline.py`](../pose/run_dlc_pipeline.py) submits an
  array job for the videos that do not yet have output from a given model.
  - Output lands in `<session>/dlc/<model>/`.
  - A model is named by the `Task` field of its config.
- **After DeepLabCut.** [`pose/after_dlc.py`](../pose/after_dlc.py) is queued behind
  those jobs. It runs pose, refreshes the session index and adds the sessions to an NWB
  batch.

The control app's last post-processing step submits both, through
`pose/slurm/cohort_pipeline.sh`.

**Which environment the jobs run in.** The one you name when you submit: `--env` to
the job scripts, `env=` to `submit_dlc` and `run_cohort`, a virtual environment's folder
or a conda environment's name. One environment with `hba2[dlc]` installed runs every
stage, DeepLabCut included. Each script exports it as `HBA2_ENV` and sbatch hands that
to the jobs it submits, so it is given once
([`pose/slurm/activate_env.sh`](../pose/slurm/activate_env.sh)). Without `--env`,
`HBA2_ENV` from the submitting shell is used, so a line in `~/.bashrc` makes a default.

`build_pose(cohort, model)` turns the DeepLabCut coordinates into:

- the head's position and bearing on every frame
- the angle to each of the six ports
- the same at the start and at the response of every trial

It applies no confidence threshold: the likelihood travels beside every angle, and a
threshold is a filter you apply when you load. The port positions come from a
calibrated table, `pose/port_coordinates/port_coordinates.json`, measured by
[`pose/calibrate_ports.py`](../pose/calibrate_ports.py) from where the head is at the
moment each beam breaks.

[pose/README.md](../pose/README.md) covers all of this in detail, including running
DeepLabCut by hand.

## 3. The session index: `sessions.csv`

```python
from hba2 import build_index
build_index(cohort)
```

This writes one row per session into `<cohort>/sessions.csv`:

- who, when and which protocol
- the laser settings
- the trial count
- whether the video is complete or has been repaired
- which DeepLabCut models have output

Columns starting with `_` are rebuilt every time. Any other column is yours: add one in
a spreadsheet to label sessions (`include`, `stim_position`, `notes`), and no rebuild
will touch it. See [the cohort folder](cohort-folder.md#the-two-label-tables).

`after_dlc` runs this for you. Run it by hand after editing nothing but the processed
files, or to see a new cohort's sessions before DeepLabCut has finished.

## 4. NWB files: one per session, in a batch

```
python nwb/build_nwb.py <cohort> <model task> --batch 2026-11_v1 [--sort S --curation C]
```

Or call `build_nwb(cohort, model, batch=...)` from a script of your own. That is also the
way to give different mice different models: `model` can be `{mouse: task}`.

- **What a batch is.** Every file belongs to a named **batch**: one file per session,
  all built from the same choices. A batch records the model, and the sort and curation
  when there are units.
- **Where it goes.** Each session's file goes in its own folder, at
  `<session>/nwb/<batch>/<session>.nwb`. The batch's recipe goes at
  `<cohort>/nwb/<batch>/recipe.json`.
- **A batch is never built two ways.** Topping up an existing batch with different
  settings stops before anything is written. Start a new batch instead.
- **Session overrides.** A session's own `sort` and `curation` columns in
  `sessions.csv`, where you have filled them in, replace the batch's for that session.

The file is assembled from the readable files written by the earlier stages, never from
raw data, and it is written to a temporary name and then renamed into place. So a killed
run leaves the previous file intact, and a suspect file is rebuilt rather than
investigated. [The NWB files](nwb-files.md) describes what is inside.

## 5. Ephys: from a spike sort to good units

```
python -m hba2.ephys run <cohort> --curation C [--sort S]
```

`process_ephys_cohort` does each of these steps where its output is missing or out of
date:

- syncs the ephys clock
- adopts sorts into `ephys_sorts/<sort>/`
- freezes the curations you name out of Phy
- exports their good units (`units.csv`, `spikes.parquet` on the DAQ clock,
  `waveforms.npz`)
- matches neurons across sessions (`neuron_names.csv`)
- rebuilds `<cohort>/neurons.csv`

Curating in Phy is the one step only a person can do. Freezing is done only for the
sessions you name, because only the curator knows which those are.

Then build an NWB batch with `--sort` and `--curation`, and the units sit in each file
beside the behaviour. The [`ephys/__init__.py`](../ephys/__init__.py) docstring describes
every layer and command. A cohort's own script calls `process_ephys_cohort` with that
cohort's sort and curations.

## Repairs: sessions recorded before a bug was found

[`postprocessing/repairs/`](../postprocessing/repairs/) holds one-off fixes for data that
was recorded or converted wrongly. Each repair:

- runs **after** post-processing, on its output exactly as post-processing left it
- applies a fix only when its own checks pass, and records the reason when they don't
- keeps the original beside what it changed

| repair | for |
|---|---|
| `repair_video_timing.py` | videos compressed by the old bin→avi converter, whose frames were paired with the wrong times. Run it after every post-processing run on such a cohort |
| `repair_video_offset.py` | sessions whose video sits a constant 0.1–5 s from the DAQ clock (some pre-2026 recordings). Run it after `repair_video_timing` |
| `repair_video_clock.py` | sessions whose camera never reached the DAQ, so frames have no pulses to count |
| `repair_scales_timing.py` | rigs whose scales were never wired to the DAQ |
| `repair_mouse_id.py` | legacy sessions whose typed mouse ID was wrong |

Each one's docstring explains what went wrong, how the fix was validated, and the
command to run. After a repair, rebuild the session's NWB files, normally into a new
batch.

## Legacy cohorts: 2024–2025

Sessions recorded before the controller rewrite have no metadata file, no trials CSV and
no trial-marker channel, so post-processing cannot run on them.
[`legacy/`](../legacy/__init__.py) does the same job by other means:

- trials are found on the DAQ from cue onsets
- the old rig's behaviour log is paired with them by time

It then writes the same `processed/` files with post-processing's own writers. From there
pose, the index, NWB and loading run unchanged. Each legacy cohort has its own port
calibration, in `<cohort>/port_coordinates/`, because the old rig numbers do not mean the
same hardware. Each legacy cohort is converted by a short script of its own, kept with
the lab's other scripts, that calls `legacy` with that cohort's settings.
