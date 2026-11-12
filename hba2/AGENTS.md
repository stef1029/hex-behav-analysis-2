# Rules for AI agents: the core package

This page is for an AI coding assistant working with `hba2`, either using
it to analyse data or changing it. Read it before writing code. The package's own
documentation is [README.md](README.md) and [docs/](docs/); this page is the rules.

## The one rule

**Use the package; do not change it without asking.**

Everything an analysis needs is already here:

- the files, put on one clock
- a trials table with every column a figure needs
- a QC gate
- loaders for events, notes, frames and spikes

If a task seems to need a change to anything under `hba2/`, stop and say
what you want to change and why. Do not make the change and report it afterwards. The
user will often say yes. The rule is *never change it silently*, not *never change it*.

Why this matters: every past figure was made from what this package wrote. A change to
how a column is computed silently changes what a rebuilt file means, and a change to
where a file lives can break the rigs. The rigs' cluster jobs run the scripts in
`pose/slurm/`.

## Using it

**Start every analysis from the trials table.**

```python
from hba2 import Cohort, load_sessions, qc_trials
cohort = Cohort(path, batch="2026-10_v1")      # the batch is chosen here, once
trials = qc_trials(load_sessions(cohort, include=True), min_ear_likelihood=0.6)
```

- **Do not read `processed/*.csv`, DAQ files, DeepLabCut CSVs or videos** in analysis
  code. The NWB files carry all of it, and the loaders read it with the right clock
  and the right frame pairing (`load_frames`, `load_events`, `load_notes`,
  `load_units`, `load_spikes`).
- **Reading raw files is for checking a session by hand**, not for building a figure.

**Get every path from `Cohort`.** `Cohort(path).session(id)` hands back a record with
every file under a named key.

- **Never build a path** from a session ID, glob for a suffix, or hardcode a cohort's
  layout.
- **Never hardcode the rig's port positions.** The calibrated ones are in the NWB, and
  `load_ports` reads them.

**The table already has the answer more often than you expect.** Check its columns
(`describe_trials(trials)`) before computing anything from events or frames:

- `cue_angle_deg`, `touched_angle_deg` and `angle_port0`–`5`
- `response_time`, `correct`, `timed_out`
- `laser_stim` and `laser_offset_s`
- the session's labels

A new analysis almost always needs a new **column**, made in the notebook with pandas,
not a new pathway into the files.

**Mark trials; never delete them.**

- **QC is a `keep` column.** Drop `keep == 0` at the point you compute, not before.
- **Never drop rows from the table you pass around.** Neighbouring-trial measures
  ("the trial before") need every row.
- **Run `qc_trials` once, at the top**, with its thresholds visible.

**One clock: the DAQ's.** Every time in the trials, events, frames and spikes is DAQ
seconds.

- **`pc_offset_*` columns are on the PC clock.** Never add one to a DAQ time, because
  no mapping between the two exists.
- **Use them only to order a trial's notes**, or for the interval between two notes on
  the same trial.

**Know what the default columns mean.**

| column | what to watch for |
|---|---|
| `correct` | NaN for timeouts and trials with no cue, by design. `hit = outcome == "success"` counts timeouts as misses; `correct.mean()` leaves them out. Say which one you used |
| `cue_angle_deg` | Positive is the mouse's **left** (anticlockwise in the overhead image), negative its right, 0 straight ahead, in [-180, 180) |
| `t_response` | The chosen port's beam break. `attach_events(response=...)` would overwrite it, so name your own events something else |
| `frame_lag_s`, `ear_likelihood`, `response_distance_px`, `head_speed_px_s` | The pose quality checks. None are applied for you; they are what `qc_trials` thresholds |
| `_video_usable` (sessions.csv) | False where the video timing is damaged and not repaired. Pose from those sessions is wrong, not just noisy |

**Labels go in `sessions.csv`, not in code.** A condition, a drug or an exclusion is a
column the user types into the cohort's `sessions.csv`.

- **Columns starting with `_` are rebuilt.** Never write into them.
- **Never put a dict of session IDs in a notebook** when a label column would do.

## Scripts that run it go outside it

A script that runs the package for someone goes in that person's own folder, outside
`hba2/`. That covers an edit-and-run `main()`, a wrapper holding one cohort's paths
and models, a status report across cohorts and a batch of stages run in order. Even a
general-looking convenience script goes outside.

- **Looking for a tool to do a job is not a reason to add one here.** Find the function
  or command line that does it and call it from a script of your own. Write that script
  beside the user's other ones, such as a `lab/` folder next to the package, or ask
  where it should go.
- **A change belongs in `hba2/` only when it changes what the package can do.** That
  means a new argument a script needs, a bug fix, or a stage or loader any lab would
  use. Such a change still needs the user's yes under the one rule above.
- **Why:** a convenience script is someone's choices: their cohorts, their models and
  their order of steps. Added here, it turns into a second way to do something the
  package already does, and the package grows a layer that is right for one lab only.

## Running stages: they write into real cohorts

`postprocess_cohort`, `build_pose`, `build_index`, `build_nwb`, the repairs and the ephys
pipeline all **write into the cohort folder**, which is shared lab data on CephFS.

- **Ask before running any of them on a real cohort.**
- **Never pass `reprocess=True` or `--force` without being told to.**
- **To rebuild NWB files, build a new batch** rather than overwriting the one figures
  were made from.
- **Never write to shared lab data from a test.** For an experiment, use a disposable
  copy of a cohort.
- **Reading is always fine.** `load_sessions`, `Cohort(...).describe()` and
  `describe_sessions` change nothing.
- **Submit SLURM jobs from a clean login shell.** A notebook or agent session's PATH
  can break the job's environment. Jobs activate the env given by `--env`, or else
  `$HBA2_ENV` ([pose/slurm/activate_env.sh](pose/slurm/activate_env.sh)).

## Changing it, once the user has said yes

- **The core never imports anything built on top of it**, such as a plotting library.
  The dependency only runs one way, because this package is meant to stand alone.
- **The core holds no lab's values.** No cohort paths, model paths, account names or
  hosts. Every one is an argument, every stage has a command line, and a lab's own
  choices live in its own scripts that call these functions.
- **Keep the principles** in [docs/principles.md](docs/principles.md):
  - post-processing synchronises and does not interpret
  - one clock, by counting pulses
  - missing is NaN, never 0
  - what varies is data
  - no hardcoded paths
  - ports are 0-based
- **Write code the way the rest of the package is written.** `cohort/cohort.py` is the
  worked example:
  - files read top to bottom in the order the data moves
  - one main function in numbered steps
  - helpers only when shared
  - no type hints beyond dataclass fields and `-> X | None`
  - docstrings that explain why, not history
- **Keep the docs true.** A change to a column, a file or a stage updates the page in
  `docs/` that describes it, and the module docstring.
- **Files the rig or hexcontrol depends on** need the user's say-so even more than the
  rest. That covers the postprocessing entry points the control app calls
  (`postprocess_cohort`), `pose/slurm/` and `cohort/cohort.py`'s `scan_session` /
  `is_session` / `REQUIRED`.

## Where to look things up

1. [README.md](README.md): what the package does and its layout.
2. [docs/](docs/): the pipeline, the cohort folder, every file's columns, loading.
3. The module docstrings. Each opens with what it does and why it is built that way, and
   they are the detailed reference.
4. The source. When the docs and the code disagree, the code is right, and the docs need
   fixing. Say so.
