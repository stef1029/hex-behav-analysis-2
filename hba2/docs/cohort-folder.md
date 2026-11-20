# The cohort folder

A cohort is a folder of sessions, and the folder is all you ever pass in. Every other
path derives from it, so moving a cohort changes one line and nothing stores its name.

## The layout on disk

```
<cohort>/
├── sessions.csv                one row per session, and your labels    (build_index)
├── neurons.csv                 one row per neuron, and your labels     (ephys cohorts)
├── nwb/<batch>/recipe.json     what each NWB batch was built from
└── <group>/                    one per concurrent run across rigs, named by start time
    ├── <session>/              e.g. 260702_105918_M1
    │   ├── raw/                what the rig wrote (loose in the session folder on older sessions)
    │   ├── processed/          what post-processing wrote
    │   ├── dlc/<model>/        DeepLabCut's output, and the pose files built from it
    │   ├── nwb/<batch>/        <session>.nwb and its build record, one folder per batch
    │   └── ephys_sorts/<sort>/ a spike sort, and the curations frozen from it
    └── ephys/<..._mouse>/      raw ephys, one subfolder per session: Rig<n>_<mouse> for
                                Open Ephys, <session> for SpikeGLX (Neuropixels)
```

- **Sessions are grouped by when they started.** The rigs start concurrent sessions
  together, so each group folder holds the mice that ran at the same time. Raw ephys sits
  at group level because one recording covers them all. Each session's subfolder is
  found by its `_<mouse>` ending.
- **Everything about a session lives inside its own folder.** Copy the folder and its
  derived data comes with it. Only the cohort-wide tables and batch recipes sit at the
  top.
- **How a session is recognised.** A directory is a session if it contains a
  `*-metadata.json`, so group-level `ephys/` folders and `processed/` folders are
  skipped without being named.
- **Two raw layouts.** `nested` puts the rig's files in `raw/`; `flat` (older sessions)
  leaves them loose in the session folder. Which one a session uses is detected, and
  output goes to `processed/` either way.

The rig writes, per session:

- `-metadata.json`: setup, protocol and every parameter
- `-ArduinoDAQ.h5`: the DAQ, the master clock
- `-trials.csv`: the rig's own account of each trial
- `-blocks.json`: which blocks ran, and their parameters; newer sessions only
- `_Tracker_data.json`: camera metadata and frame IDs
- `_scales_data.csv`: the weight log
- the video: `.avi`, `.mkv` or `.mp4`

## `Cohort`: where every file is

```python
from hba2 import Cohort

cohort = Cohort("/path/to/cohort")            # newest NWB batch attached
cohort = Cohort("/path/to/cohort", batch="2026-10_v1")
print(cohort.describe())                      # what is there, per kind of file

record = cohort.session("260702_105918_M1")
record.raw["daq"]                             # full path, or None if it has no DAQ file
record.processed["trials"]
record.dlc                                    # model -> {"csv": ..., "h5": ...}
record.nwb                                    # this session's file in the batch

cohort.with_files(dlc=True, pose=False)       # sessions with DLC output but no pose yet
record.missing("ephys")                       # what one session lacks
```

`Cohort` walks the tree with directory listings only. It opens no file, so it stays fast
over CephFS.

- **Every path comes from the record.** A `SessionRecord` holds every path a session has,
  under a named key. A role the session has no file for is `None` rather than missing,
  so the record also lists what a session *could* have.
- **Use the record; never build a path yourself.** Nothing in the package joins a session
  ID onto a folder or globs for a suffix, and code built on it should not either.

## The two label tables

`sessions.csv` and `neurons.csv` hold everything known about every session, or every
neuron, plus the things only you know. They share one rule:

> **Columns starting with `_` belong to the file. Every other column belongs to you.**

```
session_id, _mouse_id, _date, _protocol, _n_trials, _video_usable, …, include, stim_position, notes
└ the key   └──────────────── rebuilt every time ───────────────────┘ └── never touched ──┘
```

- **What a rebuild does.** It recomputes every `_` column and adds rows for new sessions.
  It leaves every other column exactly as it was.
- **Adding a label.** Type a new column heading in a spreadsheet and fill it in. Nothing
  needs registering, and no rebuild will touch it.
- **Using a label.** It becomes something to choose sessions by,
  `load_sessions(cohort, include=True, stim_position=1)`. It is also a column on every
  trial.

**What `sessions.csv` records for you.** These come from the small files only (the
metadata JSON, `session.json` and `trials.csv`), so a rebuild never opens a DAQ file:

- who, when and on which rig
- the protocol and its phase
- the laser settings and windows
- the trial count and duration
- the trackers and blocks that ran
- the cue and wait durations
- frame counts and dropped frames
- whether the video is complete, repaired, and usable (`_video_usable`)
- which DeepLabCut models have output

Run `describe_sessions(cohort)` to list the columns with example values.

**What `neurons.csv` adds.** Labels follow cells rather than numbers.

- **Why that matters.** Re-running the cross-session matching can renumber neurons, so
  a label cannot be carried by `neuron_id` alone.
- **How labels are carried.** `_members` records which (session, unit) pairs make each
  neuron, and a rebuild moves your labels to whichever new neuron holds the same members.
- **Splits, merges and disappearances** are reported, never resolved silently.

See the docstring of [`cohort/neuron_index.py`](../cohort/neuron_index.py).

**A labelled copy travels with each NWB.** Each NWB file stores a copy of its session's
row, as it was when the file was built. `load_sessions(..., labels="snapshot")` reads
that copy instead of the live table, for files that have left the cohort folder.
