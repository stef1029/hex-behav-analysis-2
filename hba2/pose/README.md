# Pose: DeepLabCut and head angles

Running DeepLabCut on a cohort's videos, and turning the coordinates into head angles.

Two things you use. Everything else is machinery they drive; the cluster side lives in
`slurm/`.

```
pose/
├── run_dlc_pipeline.py   submit analysis for a cohort      ← you run this
├── pose.py               head angles from the coordinates  ← you import this
├── after_dlc.py          pose, session index, NWB — the job that follows DeepLabCut
├── frame_lookup.py       event time → video frame, and how far off it is
├── calibrate_ports.py    port positions from where the mice touched
├── port_coordinates/     the per-rig port table
└── slurm/
    ├── cohort_pipeline.sh                   what the control app submits: DLC, then after_dlc
    ├── after_dlc.sh                         runs after_dlc.py once the arrays finish
    ├── unified_DLC_analysis_submission.sh   builds the list, submits the array job
    ├── make_vid_list.py                     works out which videos still need running
    ├── submit_analysis_DLC3.sh              one array task: set up a GPU, run one video
    └── batch_analyse.py                     calls DeepLabCut, records what it used
```

This folder used to be `dlc_scripts/`. `dlc_scripts/utils` is kept as a link to `slurm/`
because each rig's `rigs.yaml` names it as `cluster_analysis.scripts_dir`; once those
point at `pose/slurm`, the link can go.

---

## Running an analysis

Edit the parameters at the bottom of `run_dlc_pipeline.py` and run it, or call it:

```python
from hba2.pose.run_dlc_pipeline import submit_dlc

submit_dlc("/path/to/cohort", "/path/to/config.yaml")
```

Only sessions that have a compressed video **and** no output yet for that model are
submitted, so re-running is safe and cheap — it finds nothing to do.

Grouping is an argument, not a different script:

```python
submit_dlc(cohort, config)                           # one model for the cohort
submit_dlc(cohort, {"M1": cfg_a, "M2": cfg_b})   # a model per mouse
submit_dlc(cohort, config, users={...})              # split over several accounts' quota
submit_dlc(cohort, {...}, users={...})               # both at once
```

Each group becomes one video list and one SLURM array job. `users` maps a username to
its SSH settings (`{"ssh_host": "cluster-login", "ssh_user": "someone"}`); each user submits as
itself and gets an equal share of the work.

`dry_run=True` prints what would be written and submitted, and stops.

---

## How a model is named

**By the `Task` field in its config, and nothing else.** That is the short name you
chose when you made the model, and it is the name used everywhere:

| | |
|---|---|
| Submitting | you give a config path; the Task is read out of it |
| On disk | coordinates go to `<session>/dlc/<task>/` |
| Asking for them later | you quote the Task |

DeepLabCut generates a longer name of its own around the Task —
`HrnetW32_260527_array_mice_behavMay27shuffle1_snapshot_best-190` — which it embeds in
every filename. You never have to type that. It is recorded in `model.json` beside the
coordinates:

```json
{
  "task": "my_model",
  "model": "HrnetW32_my_modelMay27shuffle1_snapshot_best-190",
  "config": ".../my_model-Me-2026-05-27/config.yaml",
  "video": ".../260623_131531_M1_video.avi",
  "analysed": "2026-09-16T14:53:44"
}
```

So the exact snapshot stays a recorded fact rather than something parsed back out of a
filename, and the name you work with stays short.

### Finding out what exists

```python
cohort.models                      # {'260527_array_mice_behav': 42, ...}
record.model_record("260527_array_mice_behav")   # the JSON above, or None
```

Nothing needs a config file to work out which coordinates are already there.

### Sessions analysed before this

Their folders are named with DeepLabCut's long name instead. Asking by Task still finds
them: `SessionRecord.dlc_key` matches an exact folder name first, then a folder whose
name contains the Task. If a name is ambiguous it raises and lists the candidates rather
than choosing one — two models sharing a Task substring are different answers, and
guessing is how a figure ends up built from a model nobody meant to use.

---

## Where everything lands

```
<cohort>/
├── dlc_videos_<task>.tsv          what this run will analyse
├── <session>_SUCCESS.log          one per session ...
├── <session>_FAILED.log           ... named for the session
└── <group>/<session>/
    └── dlc/<task>/
        ├── <session>_videoDLC_<model>.csv    the coordinates
        ├── <session>_videoDLC_<model>.h5
        └── model.json                        what produced them
```

**Coordinates go in the model's own folder**, not beside the video — in a nested session
the video lives in `raw/`, which the pipeline must never write to.

**Everything else a run produces goes in the cohort folder**: the video list, and one
log per session named for that session. So the state of a run is visible in one place
and clears out in one command. `_temp.log` while a job is running, then renamed to
`_SUCCESS.log`, `_FAILED.log` or `_CUDA_FAILED.log`.

---

## Head angles

Once coordinates exist, `pose.py` turns them into head position, bearing and the angle
to each port, on every frame:

```python
from hba2 import Cohort, build_pose

build_pose(Cohort(path), "260527_array_mice_behav")
```

Output goes in the same `dlc/<task>/` folder — `pose.csv` (per frame), `trial_pose.csv`
(per trial, the file most analysis wants) and `pose.json` (the port coordinates used and
what it was built from). See the docstring at the top of `pose.py`; it is the reference
for what the angles mean.

This is a separate stage on purpose: DeepLabCut lands hours to days after a session, and
nothing upstream waits for it.

### How close each frame is to its event

`trial_pose.csv` reads the head at two moments per trial, and each comes with a lag —
that frame's own time minus the event's, in seconds:

| column | the frame used | its lag |
|---|---|---|
| `frame` | first frame at or after `t_start` | `frame_lag_s` |
| `response_frame` | first frame at or after `t_response` | `response_frame_lag_s` |

Normally the lag is under one frame period (0.033 s at 30 fps). Where the camera dropped
frames or the video lost some, the nearest image can be far from the event, and the lag
is how you see that. Every value is still written; filter when you analyse, e.g.
`frame_lag_s <= 0.1`. A negative lag means the event came after the video ended.

Every event lookup goes through `frames_at` in `frame_lookup.py` — use it for any new
one. `pose.json` summarises the lags per session under `lookups`.

**The lag cannot see a video whose frames are paired with the wrong times**, which is
what the old bin→avi converter did to sessions with camera drops. Run
`postprocessing/repairs/repair_video_timing.py` after post-processing and before this; pose notices the
repaired `frame_times.csv` and rebuilds on its own.

---

## The chain

```
run_dlc_pipeline.py            your machine — picks videos, writes a list per group
        │  sbatch (optionally over SSH)
unified_DLC_analysis_submission.sh    cpu partition, 8G, 30 min
        │      └─ make_vid_list.py    only when no list was handed in
        │  sbatch --array --chdir=<cohort>
submit_analysis_DLC3.sh        agpu partition, 1 GPU, 100G — one task per video
        └─ batch_analyse.py    runs DLC, writes model.json
```

Every job runs in one environment with `hba2[dlc]` installed, named by whoever submits:
`--env` to the scripts, `env=` to `submit_dlc` and `run_cohort`. Each script exports it
as `HBA2_ENV`, and sbatch hands that on to the jobs it submits, so it is given once
([`slurm/activate_env.sh`](slurm/activate_env.sh)). Without `--env`, `HBA2_ENV` from
the submitting shell is used, so a line in `~/.bashrc` on the cluster makes a default.

### From the control app: DeepLabCut, then pose and NWB

The control app's post-processing ends, after the rsync to CephFS, by submitting one job
over SSH. Each mouse's model comes from `analysis_dlc_model` in the rigs YAML; the host
and scripts folder from its `global.cluster_analysis` block, and the environment from
that block's `env`.

```
control app                    ssh <host> sbatch --parsable cohort_pipeline.sh …
        │
cohort_pipeline.sh             cpu, 8G, 1 h — runs run_dlc_pipeline.py's command line
        │  sbatch --array --chdir=<cohort>   one per model, directly, so the IDs come back
submit_analysis_DLC3.sh        as above
        │  sbatch --dependency=afterany:<every array>
after_dlc.sh                   cpu, 32G, 12 h — after_dlc.py: pose → sessions.csv → NWB
```

The same thing by hand, on the cluster:

```bash
python run_dlc_pipeline.py /path/to/cohort \
    --mouse-model T1=/path/model_a/config.yaml --mouse-model T2=/path/model_b/config.yaml \
    --scripts-dir /path/to/pose/slurm --env /path/to/envs/hba2 --gpus 8 --then-nwb \
    [--dry-run]
```

- **One after-DLC job per cohort**, not per model, since each would rewrite
  `sessions.csv`. It waits for every array, and is submitted with no dependency when
  there was no DeepLabCut to run — so sessions analysed earlier but never built are.
- **`afterany`, not `afterok`.** Whatever DeepLabCut succeeded on is built; a session it
  failed on has no output and is skipped with its reason. A stage that fails outright
  is logged and the next still runs.
- **Re-running is cheap.** DeepLabCut, pose and NWB each skip what is already current.
- A mouse with sessions in the cohort but no model is named in the log, not analysed.
- The job's own logs land in the cohort: `cluster_analysis_<id>.log`,
  `after_dlc_<id>.log`, beside the per-session DeepLabCut logs.

### The video list

Three tab-separated columns, because only the cohort knows the last two:

```
<video path>	<destination folder>	<session id>
```

The destination depends on the model, and the session ID cannot be recovered from the
video path in a nested session. A hand-written list of bare video paths still works —
DeepLabCut then writes beside the video and the session ID is guessed from the path, the
way it used to be.

### Running the shell scripts by hand

```bash
sbatch slurm/unified_DLC_analysis_submission.sh \
    --cohort /path/to/cohort \
    --config /path/to/config.yaml \
    --gpus 4 \
    --scripts-dir /path/to/pose/slurm
```

`--videolist` takes a prepared list instead of building one. Give `--cohort` as well and
the per-session logs still land in the cohort folder.

**Pass `--scripts-dir`.** SLURM copies a batch script to a spool directory on the
compute node before running it, so the script cannot reliably find its own siblings;
`run_dlc_pipeline.py` passes the path for exactly this reason. Without the flag the
scripts fall back to a fixed path that is only right on one machine.

The pipeline assumes the scripts are reachable at the same path from wherever you submit
and from the cluster, which holds because `/lmb` homes are shared.
