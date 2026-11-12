"""
The NWB files everything is plotted from.
========================================

One file per session holding everything a figure needs: the trials with their head
angles, the head on every frame, every hardware event, the laser pulses, the port
coordinates the angles were measured from, and the session's own facts and labels.
Nothing downstream has to open a CSV, a DAQ file or a video.

The previous library's ``<session>.nwb`` is a different file in a different place and is
never touched.

Batches
-------
Every file is built as part of a **batch**: one file per session, all from the same
choices, each in its session's ``nwb/`` folder under the batch's name, and one recipe for
the whole batch at cohort level::

    python build_nwb.py <cohort> <model task> --batch 2026-10_first_ephys \\
                        --sort ks4_camnt --curation 2026-09_manual

    from hba2.nwb.build_nwb import build_nwb       # the same, from Python
    build_nwb(cohort, model, batch="2026-10_first_ephys", sort=..., curation=...)

    <cohort>/nwb/2026-10_first_ephys/
    └── recipe.json             the parameters, the code version, and per session what
                                it resolved to (curation, units, names) or why not built

    <session>/nwb/2026-10_first_ephys/
    ├── <session>.nwb
    └── <session>.json          that file's build record and check, as above

The parameters are whatever this run was given; nothing reads ``recipe.json`` back to
build from. A batch name already used with other parameters stops the run. A session's
``sort`` / ``curation`` columns in ``sessions.csv``, where filled in, replace the
parameters for that session — for the one that only has ``auto``, say.

With ``--sort`` and ``--curation`` the file also holds that curation's good units (see
``ephys/``)::

    /general/devices/probe, /general/extracellular_ephys/shank0..7
    /general/extracellular_ephys/electrodes   every channel: id = Phy's channel index,
                                              rel_x/rel_y in um, open_ephys_channel
    /units                       one row per good unit, id = Phy's cluster ID:
                                 spike_times on the DAQ clock (this file's clock),
                                 obs_intervals, electrodes (peak channel), electrode_group
                                 (shank), waveform_mean (peak channel, volts), and the
                                 columns in UNIT_COLUMNS — metrics, neuron_id
    /general/scratch/neuron_labels   this file's neurons' rows of the cohort's
                                     neurons.csv when built — your labels for them

A session in the batch without that sort or curation still gets its file, without
units, and the run and the recipe say why.

What is inside
--------------
::

    /general
    ├── session_id, session_start_time, institution
    ├── subject                  mouse, and its weight that day
    ├── devices/rig_N            rig number, camera serial, fps
    └── scratch/
        ├── session_facts        everything post-processing recorded about the session
        ├── session_labels       this session's row of sessions.csv — your labels
        └── build_record         sources, fingerprints and counts (the check
                                 against them is in the sidecar JSON, written
                                 after the file itself)

    /intervals
    ├── trials                   one row per trial, with the pose at its start
    ├── epochs                   one row per block
    └── pulses_laser             one row per laser pulse, and the trial it fell in

    /processing/behaviour
    ├── Position/head_position       SpatialSeries (n, 2) px
    ├── CompassDirection/head_bearing    SpatialSeries (n,) degrees
    ├── port_angles                  TimeSeries (n, 6) degrees
    ├── ear_likelihood               TimeSeries (n,)
    ├── head_speed, turn_rate        TimeSeries (n,) px/s, degrees/s from the frame before
    ├── bearing_method               TimeSeries (n,) code per frame
    ├── video_frame, camera_frame_id TimeSeries (n,) — back to the AVI and the camera
    ├── events                       one row per hardware transition
    ├── notes                        what the protocol said, on the PC clock
    └── ports                        six rows: the calibrated coordinates

    /acquisition
    └── scales                       what the scales under the arena read, ~85 Hz

Every per-frame series shares one timestamps array, stored once and linked. Two things
follow: there is no second copy of the clock to drift, and frames the old converter never
wrote show up as jumps in it — so anything drawn against time breaks at a gap by itself.

Built from the readable files, not from raw
------------------------------------------
Post-processing, the video-timing repair and the pose stage have already done the work;
this only assembles it. So an NWB is always reproducible from files you can open and
read, and a suspect one is rebuilt rather than investigated. The manifest fingerprints
every input, and a rebuild is skipped when nothing has changed.

Written once, never appended to: assembled in memory, written to a temporary file, then
renamed into place. A killed run leaves the previous file intact, and the operation that
silently drops appended data in later pynwb versions never happens.

Deliberately left out
---------------------
* **The raw DAQ channels** — 13 MB a session, and the events table already holds every
  transition on them.
* **DeepLabCut's other body parts** — the head is what the angles use; a tail or spine
  overlay is a job for the video and the DLC file, which both stay where they are.
* **The video** — referenced by path only.

Each of those means a rebuild if a plot later needs it, which is cheap: this stage reads
small files and takes a few seconds a session. ``--no-scales`` drops the scales the same
way, for a cohort where nobody wants them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable, VectorData
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import CompassDirection, Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.misc import Units

# Runnable as a file as well as a module, like the other scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hba2.cohort import BATCHES_DIR, RECIPE, Cohort, model_for  # noqa: E402
from hba2.pose.pose import N_PORTS, _readable_json  # noqa: E402

# The page reads in the order the data moves (a comment rather than part of the
# docstring above, which is also the command line's --help):
#
#     1. the numbers and names   columns, their types, the laser window
#     2. a whole cohort          build_nwb: the batch's recipe, which sessions, each one
#                                in turn, the recipe again; is_current to skip the rest
#     3. one session             build_session_nwb: read, assemble, write, check
#     4. the pieces of one file  one function per thing that ends up in it
#     5. the check               verify, verify_units: read it back and compare
#     6. what comes out          EphysSource, SessionNWB, NWBResult
#     7. the command line

# =====================================================================
# 1. The numbers and names
# =====================================================================

#: Where the rig is, for the NWB metadata every file carries.
INSTITUTION = "MRC Laboratory of Molecular Biology"

#: The rigs record wall-clock time without an offset, and they sit in Cambridge.
TIMEZONE = ZoneInfo("Europe/London")

#: How long before a trial's marker a laser pulse still counts as that trial's.
#: The protocols arm stimulation *relative to the trial*, in windows that mostly start
#: before it — -400..-395 ms and -150..-145 ms on this cohort — so a pulse belonging to a
#: trial normally lands a few hundred milliseconds before its marker goes high, and a rule
#: of "inside the trial" would find none of them. A second covers every window the rig can
#: arm, with slack; ``laser_offset_s`` on each trial says where the pulse actually fell,
#: for a plot that wants to be exact.
LASER_PRE_S = 1.0

#: Compression for the per-frame arrays. Level 4 is the usual sweet spot: a two-hour
#: session's pose comes down from about 12 MB to 6 without noticeably slower reads.
GZIP = 4

#: Trials whose rows are read straight from ``trials.csv``, with their descriptions. The
#: pose columns are added from ``trial_pose.csv`` — see :func:`_trials`.
#: column name -> description
TRIAL_COLUMNS = {
    "t_response": "when the chosen port's beam broke, NaN if it never did (s)",
    "response_time": "t_response - start_time, NaN without a response (s)",
    "tracker": "the named counter this trial was recorded on",
    "sub_tracker": "the sub-tracker within it, 'default' where there is only one",
    "block": "the block of parameters that was active, from the trials file's stage",
    "trial_number": "the trial's number within its tracker, as the rig counted it",
    "outcome": "what the controller's own log called it",
    "correct_port": "the port the cue asked for, 0-5, -1 if none",
    "chosen_port": "the port the animal touched, 0-5, -1 if it touched none",
    "correct": "1 chose the cued port, 0 chose another, NaN when not defined for this "
               "trial: no cued port, or no choice at all. The default reading only; a "
               "plot is free to define correctness its own way",
    "timed_out": "1 where the animal chose no port at all",
    "stimulus_log": "what the controller presented, as it recorded it",
    "event_log": "what else the protocol noted as the trial ran, as it recorded it: "
                 "entries of {t, name, ...} with t relative to the trial marker and "
                 "negative where the thing happened before it. The names and their "
                 "keys are the protocol's own and vary between protocols, so nothing "
                 "here reads them — the laser columns below come from the pulses",
    "details_json": "everything else the protocol attached to this trial",
    "laser_stim": "1 where a laser pulse belongs to this trial — see LASER_PRE_S for "
                  "what belongs means",
    "n_laser_pulses": "laser pulses belonging to this trial",
    "t_laser_on": "the first laser pulse belonging to this trial, NaN if none (s)",
    "laser_offset_s": "t_laser_on - start_time: where that pulse fell relative to the "
                      "trial marker, normally negative (s)",
}

#: Added from ``trial_pose.csv``: the head at the trial marker going high, which is what
#: start_time is, plus the quality of that reading. Same names as the CSV.
#: column name -> description
POSE_COLUMNS = {
    "head_x": "head position at the trial start frame (px)",
    "head_y": "head position at the trial start frame (px)",
    "bearing_deg": "head bearing at the trial start frame (degrees, 0 = right, 90 = up)",
    "cue_angle_deg": "where the cued port sat relative to straight ahead at the trial "
                     "start frame (degrees, [-180, 180), 0 = straight ahead)",
    "touched_angle_deg": "where the port the animal went on to touch sat at the trial "
                         "start frame (degrees)",
    "touched_distance_px": "how far the touched port was from the head at the trial "
                           "start frame (px)",
    "frame_lag_s": "the trial start frame's own time minus start_time. Normally under "
                   "one frame period; larger means the video has a gap there (s)",
    "ear_likelihood": "DeepLabCut's confidence in the ears at the trial start frame",
    "response_frame": "the video frame showing the response, NaN without one",
    "response_frame_lag_s": "that frame's time minus t_response (s)",
    "response_distance_px": "head to the touched port at the moment its beam broke. A "
                            "poke is a few tens of pixels; hundreds means the head was "
                            "elsewhere (px)",
    "response_ear_likelihood": "DeepLabCut's confidence in the ears at that frame",
    "head_speed_px_s": "how fast the head moved around the trial start frame: the larger "
                       "of the steps into and out of it. A label that jumped reads as a "
                       "speed no mouse reaches (px/s)",
    "turn_rate_deg_s": "how fast the bearing turned around the trial start frame, the "
                       "same way and in size only. A bearing that swapped ends reads as "
                       "a turn no neck allows (degrees/s)",
    "response_head_speed_px_s": "head_speed_px_s at the response frame (px/s)",
    "response_turn_rate_deg_s": "turn_rate_deg_s at the response frame (degrees/s)",
}

#: The angle to every port at the trial start frame, so an angle plot never has to open
#: the per-frame data.
PORT_ANGLE_COLUMNS = tuple(f"angle_port{port}" for port in range(N_PORTS))

#: A table column holds one type, so each has to be declared. Text columns write "" where
#: there is nothing; integer columns cannot hold a missing value and are only used where
#: there is always an answer (a port is -1 for "none"); everything else is a float, and
#: missing is NaN.
TEXT_COLUMNS = frozenset({"tracker", "sub_tracker", "block", "outcome", "stimulus_log",
                          "event_log", "details_json"})
INT_COLUMNS = frozenset({"correct_port", "chosen_port", "trial_number", "laser_stim",
                         "n_laser_pulses", "timed_out"})

#: Unit columns from the curation's ``units.csv`` and ``neuron_names.csv``, with what
#: they mean. ``spike_times``, ``obs_intervals``, ``electrodes``, ``electrode_group`` and
#: ``waveform_mean`` are NWB's own columns and need no declaring; the table's ``id`` is
#: Phy's cluster ID.
#: column name -> description
UNIT_COLUMNS = {
    "shank": "the probe shank, 0-7, from the peak channel",
    "peak_channel": "Phy's index (0-127) of the channel with the largest waveform; also "
                    "the electrodes table's id",
    "open_ephys_channel": "which recorded channel the peak channel is (CH1..CH128)",
    "x_um": "the peak channel's position across the probe (um)",
    "depth_um": "the peak channel's position along its shank, from the tip (um)",
    "n_spikes": "spikes with a DAQ time — the ones in spike_times",
    "firing_rate_hz": "n_spikes over the observation interval",
    "amplitude_uv": "peak-to-trough of waveform_mean (uV)",
    "snr": "SpikeInterface's signal-to-noise ratio; NaN for a cluster made in Phy",
    "isi_violations_ratio": "SpikeInterface's refractory-violation ratio",
    "isi_violations_count": "SpikeInterface's count of refractory violations",
    "presence_ratio": "SpikeInterface's fraction of the recording the unit fired in",
    "amplitude_cutoff": "SpikeInterface's estimate of spikes lost below threshold",
    "quality": "the curation's label — always 'good'; nothing else is exported",
    "keep": "1 here; a figure marks a unit 0 to leave it out without deleting it",
    "neuron_id": "which neuron this unit is, across sessions; units of different "
                 "sessions with one name are one cell. Empty where no matching named it",
    "neuron_n_sessions": "how many sessions the matching found this neuron in, -1 if "
                         "unnamed",
    "neuron_matching": "the matching run that named it",
    "neuron_level": "which of that run's settings named it",
}
#: The unit columns' types, as for the trials: text "" and integer -1 where missing,
#: float NaN for the rest.
UNIT_TEXT = frozenset({"open_ephys_channel", "quality", "neuron_id", "neuron_matching",
                       "neuron_level"})
UNIT_INT = frozenset({"shank", "peak_channel", "n_spikes", "keep", "neuron_n_sessions"})

BY = "hba2/nwb/build_nwb.py"


# =====================================================================
# 2. A whole cohort
# =====================================================================

def _print_now(message):
    """What build_nwb reports to when no log_fn is given."""
    print(message, flush=True)


def _log_one(log, number, total, outcome):
    """One line for one session's outcome."""
    prefix = f"  [{number}/{total}] {outcome.session_id}:"
    if outcome.error:
        log(f"{prefix} FAILED — {outcome.error}")
        return
    if outcome.skipped:
        log(f"{prefix} skipped — {outcome.skipped}")
        return

    line = (f"{prefix} {outcome.counts.get('trials', 0)} trials, "
            f"{outcome.counts.get('frames', 0):,} frames")
    units = outcome.counts.get("units")
    if units:
        line += f", {units} units"
    line += f", {outcome.size_mb} MB"
    for note in outcome.notes:
        line += f" — {note}"
    if not outcome.verified:
        line += f" — DID NOT VERIFY: {'; '.join(outcome.problems)}"
    log(line)


def _one(record, model, labels, scales, plans, batch, neuron_labels=None):
    """
    One session, with its own labels and plan picked out — the unit the pool runs, so it
    has to be a module-level function. ``model`` is the batch's: one name or a dict by
    mouse.
    """
    return build_session_nwb(record, model_for(model, record),
                             labels=labels.get(record.session_id),
                             scales=scales, batch=batch, neuron_labels=neuron_labels,
                             **plans[record.session_id])


def build_nwb(cohort, model, *, batch, sort=None, curation=None, only=None,
              reprocess=False, scales=True, workers=1, log_fn=None):
    """
    Write a batch: an NWB file for every session in a cohort with pose from ``model``.

    Args:
        cohort: a :class:`Cohort`, or a path (Path or string) to build one from.
        model: which DeepLabCut model's pose to use, by its Task name. Or ``{mouse:
            model}`` with ``"*"`` for every other mouse — when mice were tracked with
            different models, as on an ephys cohort — and a mouse's entry may be a list, in
            order of preference.
        batch: the batch's name. Each session's file is
            ``<session>/nwb/<batch>/<session>.nwb``, and
            ``<cohort>/nwb/<batch>/recipe.json`` records what the batch was built from.
        sort, curation: put this curation's good units in each file. A session's own
            ``sort`` / ``curation`` columns in ``sessions.csv``, where filled in, take
            its place for that session.
        only: a tuple of session IDs: just these.
        reprocess: rebuild even where the file is still current.
        scales: include the scales readings.
        workers: sessions at once. Each is independent.
        log_fn: called with each line as it happens. Without it, lines are printed.

    A batch name already used with different settings stops the run before anything is
    written, saying what differs: one name, one set of choices. Delete the recipe's folder
    or pick another name to change them.

    Returns an :class:`NWBResult`.
    """
    log = log_fn or _print_now
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    if not batch:
        raise ValueError("every NWB file belongs to a batch — pass batch=")
    result = NWBResult(base=cohort.base, model=model, batch=batch)

    records = []
    for record in cohort:
        if not only or record.session_id in only:
            records.append(record)

    # ---- 1. Your labels, from sessions.csv ----------------------------
    # Every session's row of the cohort's index, by session ID. Read as text and kept as
    # text: these are your labels, and this stage has no business deciding that
    # ``include`` is a boolean or that a blank means anything in particular.
    labels = {}
    sessions_csv = cohort.base / "sessions.csv"
    if sessions_csv.exists():
        index = pd.read_csv(sessions_csv, dtype=str, keep_default_na=False)
        for _, row in index.iterrows():
            labels[row["session_id"]] = dict(row)

    # ---- 2. The batch's recipe: start it, or check this run matches it -
    parameters = {"model": model, "sort": sort, "curation": curation, "scales": bool(scales)}
    recipe_folder = cohort.base / BATCHES_DIR / batch
    recipe_path = recipe_folder / RECIPE
    if recipe_path.exists():
        existing = json.loads(recipe_path.read_text()).get("parameters", {})
        if existing != parameters:
            differs = []
            for name in sorted(set(existing) | set(parameters)):
                old, new = existing.get(name), parameters.get(name)
                if old != new:
                    differs.append(f"{name}: {old!r} there, {new!r} now")
            raise ValueError(
                f"batch '{batch}' was built with other settings — "
                + ", ".join(differs)
                + f". Use another batch name, or delete {recipe_folder} first.")
    else:
        # The commit this code is at, so a batch says which version built it.
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git_commit = None
        recipe_folder.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(_readable_json({
            "batch": batch,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "parameters": parameters,
            "per_session_overrides": "sessions.csv columns 'sort' and 'curation', where "
                                     "filled in, replace the parameters for that session",
            "code": {"by": BY, "git_commit": git_commit},
            "sessions": {},
        }))

    line = f"{cohort.base.name}: {len(records)} session(s), pose from {model!r}"
    if sort:
        line += f", units from {sort}/{curation}"
    line += f", batch '{batch}'"
    if not scales:
        line += ", without scales"
    if labels:
        line += f", labels from sessions.csv ({len(labels)} rows)"
    else:
        line += ", no sessions.csv to take labels from"
    log(line)

    # ---- 3. Each session's plan, and which still need building --------
    # Where its file goes, and whose units it takes: its own sessions.csv sort and
    # curation where filled in, the batch's otherwise.
    #: session ID -> {"path", "sort", "curation"}
    plans = {}
    for record in records:
        own = labels.get(record.session_id) or {}
        plans[record.session_id] = {
            "path": cohort.batch_path(batch, record.session_id),
            "sort": own.get("sort") or sort or None,
            "curation": own.get("curation") or curation or None,
        }

    todo = []
    for record in records:
        if reprocess:
            todo.append(record)
        elif not is_current(record, model_for(model, record), scales=scales,
                            **plans[record.session_id]):
            todo.append(record)
    if len(todo) != len(records):
        log(f"  {len(records) - len(todo)} already current")

    # ---- 4. Your neuron labels, from neurons.csv ----------------------
    # By neuron ID, as text, for the same reason as the session labels.
    neuron_labels = {}
    neurons_csv = cohort.base / "neurons.csv"
    if neurons_csv.exists():
        index = pd.read_csv(neurons_csv, dtype=str, keep_default_na=False)
        for _, row in index.iterrows():
            neuron_labels[row["neuron_id"]] = dict(row)

    # ---- 5. Build each one --------------------------------------------
    job = partial(_one, model=model, labels=labels, scales=scales, plans=plans, batch=batch,
                  neuron_labels=neuron_labels)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            outcomes = pool.map(job, todo)
            for number, outcome in enumerate(outcomes, start=1):
                _log_one(log, number, len(todo), outcome)
                result.results.append(outcome)
    else:
        for number, record in enumerate(todo, start=1):
            outcome = job(record)
            _log_one(log, number, len(todo), outcome)
            result.results.append(outcome)

    # ---- 6. The recipe: what each session built actually resolved to --
    recipe = json.loads(recipe_path.read_text())
    for outcome in result.results:
        entry = {"built_at": datetime.now().isoformat(timespec="seconds")}
        if outcome.ok:
            entry.update(outcome.resolved)
        else:
            entry["not_built"] = outcome.skipped or outcome.error
        recipe["sessions"][outcome.session_id] = entry
    recipe["sessions"] = dict(sorted(recipe["sessions"].items()))
    recipe_path.write_text(_readable_json(recipe))

    log("\n" + result.describe())
    return result


def _ephys_source(record, sort, curation):
    """
    The units to put in this session's file, as ``(EphysSource, None)``, or
    ``(None, why not)``. ``why not`` is None too when no units were asked for, or the
    session has no probe to have them from.

    Absence is a reason rather than an error: a batch spans sessions with and without a
    probe, and one sorted but not yet curated is a normal state to build through.
    """
    if not sort or not curation:
        return None, None
    if sort not in record.sorts:
        if record.has_ephys:
            return None, f"no sort '{sort}'"
        return None, None
    files = record.sorts[sort]["curations"].get(curation)
    if files is None:
        return None, f"sort '{sort}' has no curation '{curation}'"
    missing = [role for role in ("units", "spikes", "waveforms") if files.get(role) is None]
    if missing:
        return None, f"curation '{curation}' is not exported ({', '.join(missing)} missing)"
    if pd.read_csv(files["units"]).empty:
        return None, f"curation '{curation}' has no good units"
    source = EphysSource(sort=sort, curation=curation, files=files,
                         phy=Path(record.sorts[sort]["phy"]),
                         sort_record=record.sorts[sort].get("record"))
    return source, None


def _fingerprints(record, pose_dir, ephys=None):
    """
    Enough about every input to notice it has changed, without reading it again: file
    name -> {"size", "mtime"}. ``ephys`` is the session's EphysSource, or None.
    """
    files = {
        "trials": record.path("trials"),
        "events": record.path("events"),
        "frame_times": record.path("frame_times"),
        "session_json": record.path("session_json"),
        "pose": pose_dir / "pose.csv",
        "trial_pose": pose_dir / "trial_pose.csv",
        "pose_json": pose_dir / "pose.json",
    }
    for name, path in record.pulses.items():
        files[f"pulses_{name}"] = path
    if ephys is not None:
        for role in ("units", "spikes", "waveforms", "names"):
            if ephys.files.get(role) is not None:
                files[f"ephys_{role}"] = ephys.files[role]

    prints = {}
    for name, path in files.items():
        stat = Path(path).stat()
        prints[name] = {"size": stat.st_size, "mtime": stat.st_mtime}
    return prints


def is_current(record, model, *, path, scales=True, sort=None, curation=None):
    """
    True when a rebuild would produce the same file as the one already there.

    ``path`` is the file in question, in its batch, as a Path. ``sort`` and ``curation``
    must be what it was built with.
    """
    key = record.dlc_key(model)
    if key is None:
        return False
    try:
        stored = json.loads(path.with_suffix(".json").read_text())
        if not path.exists():
            return False
        includes = stored.get("includes", {})
        if includes.get("scales") != bool(scales):
            return False
        if (includes.get("sort"), includes.get("curation")) != (sort, curation):
            return False
        if not stored.get("verified"):
            return False
        ephys, _ = _ephys_source(record, sort, curation)
        return stored.get("sources") == _fingerprints(
            record, record.directory / "dlc" / key, ephys)
    except (OSError, ValueError, KeyError):
        return False


# =====================================================================
# 3. One session
# =====================================================================

def build_session_nwb(record, model, *, path, batch, labels=None, scales=True,
                      sort=None, curation=None, neuron_labels=None):
    """
    Write one session's NWB file, and check it against the files it came from.

    Args:
        record: every path this session has, already resolved.
        model: which DeepLabCut model's pose to use, by its Task name.
        path: the file to write, as a Path or a string — :meth:`Cohort.batch_path`. Its
            sidecar JSON goes beside it.
        batch: the batch this file belongs to, recorded in it.
        labels: this session's row of ``sessions.csv``, stored in the file so it can be
            grouped without reading the index. None writes no labels.
        scales: include the scales readings — see :func:`_scales`.
        sort, curation: put this curation's good units in the file too. A session
            without them still gets a file, and :attr:`SessionNWB.notes` says why it has
            no units.
        neuron_labels: ``neurons.csv`` by neuron ID, as text. The rows for this file's
            neurons are stored in it, as the session's labels are.

    Failures are returned rather than raised, so a cohort run reports a bad session and
    carries on. Check :attr:`SessionNWB.ok`.
    """
    key = record.dlc_key(model)
    path = Path(path)
    out = path.parent
    result = SessionNWB(session_id=record.session_id, model=model)

    if key is None:
        result.skipped = f"no output for model '{model}'"
        return result
    if not record.is_processed:
        result.skipped = "not post-processed"
        return result
    pose_dir = record.directory / "dlc" / key
    if not (pose_dir / "pose.csv").exists():
        result.skipped = f"no pose built for '{model}' — run build_pose first"
        return result

    try:
        # ---- 1. READ -------------------------------------------------
        facts = json.loads(record.path("session_json").read_text())
        pose_record = json.loads((pose_dir / "pose.json").read_text())
        trials = pd.read_csv(record.path("trials"))
        events = pd.read_csv(record.path("events"))
        # Optional: sessions post-processed before notes.csv existed have none, which is
        # absence rather than an error — the protocol annotations are still in the
        # trials' event_log and stimulus_log columns, verbatim. They get an empty table.
        if record.processed.get("notes"):
            notes = pd.read_csv(record.path("notes"), dtype={"value": str})
        else:
            notes = pd.DataFrame({
                "trial": pd.Series(dtype="int64"),
                "note": pd.Series(dtype="int64"),
                "kind": pd.Series(dtype="object"),
                "note_name": pd.Series(dtype="object"),
                "pc_offset_s": pd.Series(dtype="float64"),
                "key": pd.Series(dtype="object"),
                "value": pd.Series(dtype="object"),
            })
        frame_times = pd.read_csv(record.path("frame_times"))
        pose = pd.read_csv(pose_dir / "pose.csv")
        trial_pose = pd.read_csv(pose_dir / "trial_pose.csv")
        #: channel -> its pulses
        pulses = {}
        for name, pulses_path in record.pulses.items():
            pulses[name] = pd.read_csv(pulses_path)
        if record.processed.get("video_repair"):
            repair = json.loads(record.path("video_repair").read_text())
        else:
            repair = None
        ephys, no_units = _ephys_source(record, sort, curation)
        if no_units:
            result.notes.append(f"no units: {no_units}")

        # ---- 2. ASSEMBLE ---------------------------------------------
        # When the session started, as an aware datetime — NWB insists on the timezone.
        # The rig writes wall-clock time with no offset. Failing that, the session ID is
        # itself a timestamp (YYMMDD_HHMMSS), which is how the folder got its name.
        stamp = facts.get("start_time")
        if stamp:
            start = datetime.fromisoformat(stamp).replace(tzinfo=TIMEZONE)
        else:
            start = datetime.strptime(record.session_id[:13],
                                      "%y%m%d_%H%M%S").replace(tzinfo=TIMEZONE)
        weight = None
        if facts.get("mouse_weight"):
            weight = f"{facts['mouse_weight']} g"

        nwb = NWBFile(
            session_description=f"{facts.get('protocol_name', 'unknown protocol')}, "
                                f"phase {facts.get('phase')}",
            identifier=f"{record.session_id}/{batch}",
            session_start_time=start,
            session_id=record.session_id,
            institution=INSTITUTION,
            subject=Subject(subject_id=str(facts.get("mouse_id", record.mouse_id)),
                            species="Mus musculus", weight=weight),
        )
        rig = facts.get("rig", {})
        nwb.create_device(
            name=f"rig_{rig.get('number', 'unknown')}",
            description=f"{rig.get('name', 'hex arena')}, camera {rig.get('camera_serial')} "
                        f"at {facts.get('camera', {}).get('fps')} fps")

        # A session with no trials still has pose and events worth keeping, and gets a
        # file without a trials table rather than no file. The early training sessions
        # are like this: their trial marker channel never pulsed, so post-processing
        # found no trials to detect. A column with no rows has no type NWB can store,
        # which is the other reason not to write an empty table.
        laser_pulses = pulses.get("laser", pd.DataFrame(columns=["t_on", "t_off"]))
        if trials.empty:
            n_trials = 0
        else:
            n_trials = _trials(nwb, trials, trial_pose, pose, laser_pulses)
        counts = {"trials": n_trials,
                  "epochs": _epochs(nwb, trials),
                  "events": len(events)}
        for name, table in pulses.items():
            counts[f"pulses_{name}"] = _pulses(nwb, name, table, trials)

        interfaces, frame_counts = _pose_series(pose, frame_times)
        counts.update(frame_counts)
        behaviour = nwb.create_processing_module(
            name="behaviour",
            description="the head frame by frame, the hardware events, and the ports")
        for interface in interfaces:
            behaviour.add(interface)
        behaviour.add(_events(events))
        behaviour.add(_notes(notes))
        behaviour.add(_ports(np.asarray(pose_record["ports"]["coords"], dtype=float)))

        if scales:
            weights = _scales(record)
            if weights is not None:
                nwb.add_acquisition(weights)
            for series in _stillness(record):
                nwb.add_acquisition(series)

        units_provenance = None
        if ephys is not None:
            unit_counts, units_provenance = _units(nwb, ephys)
            counts.update(unit_counts)
            # This file's neurons' rows of neurons.csv.
            here = {str(n) for n in nwb.units["neuron_id"].data if n}
            these_labels = {}
            for neuron, row in (neuron_labels or {}).items():
                if neuron in here:
                    these_labels[neuron] = row
            nwb.add_scratch(json.dumps(these_labels),
                            name="neuron_labels",
                            description="this file's neurons' rows of the cohort's "
                                        "neurons.csv when this was built, as JSON — your "
                                        "labels for them, by neuron_id")

        video_repair = None
        if repair:
            video_repair = {"status": repair.get("status"),
                            "check": (repair.get("check") or {}).get("passed")}
        model_entry = {"task": model, "resolved": key, "record": record.model_record(model)}
        pose_entry = {"params": pose_record.get("params"),
                      "lookups": pose_record.get("lookups"),
                      "ports": pose_record.get("ports", {}).get("geometry")}
        built_at = datetime.now().isoformat(timespec="seconds")
        sources = _fingerprints(record, pose_dir, ephys)
        record_of_build = {
            "session_id": record.session_id,
            "batch": batch,
            "model": model_entry,
            "ephys": units_provenance,
            "built_at": built_at,
            "by": BY,
            "counts": counts,
            "notes": result.notes,
            "includes": {"scales": bool(scales), "sort": sort, "curation": curation},
            "params": {"laser_pre_s": LASER_PRE_S},
            "pose": pose_entry,
            "video_repair": video_repair,
            "sources": sources,
        }
        nwb.add_scratch(json.dumps(facts), name="session_facts",
                        description="everything post-processing recorded about this "
                                    "session, as JSON")
        nwb.add_scratch(json.dumps(labels or {}), name="session_labels",
                        description="this session's row of the cohort's sessions.csv "
                                    "when this was built, as JSON — your labels")
        nwb.add_scratch(json.dumps(record_of_build), name="build_record",
                        description="what this file was built from, as JSON. Whether it "
                                    "then matched those files is in the sidecar JSON "
                                    "beside it — that check can only run once this file "
                                    "exists, and this file is never reopened to add to it")

        # ---- 3. WRITE ------------------------------------------------
        # Whole file at once, then renamed: a killed run leaves the old file in place.
        out.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".nwb.tmp")
        with NWBHDF5IO(str(temporary), "w") as io:
            io.write(nwb)
        os.replace(temporary, path)

        # ---- 4. CHECK IT AGAINST WHAT IT CAME FROM -------------------
        result.problems = verify(path, trials, trial_pose, pose, events, counts)
        if ephys is not None:
            result.problems += verify_units(path, ephys)
        result.verified = not result.problems
        record_of_build["verified"] = result.verified
        record_of_build["problems"] = result.problems

        manifest = path.with_suffix(".json")
        manifest.write_text(_readable_json(record_of_build))

        # ---- 5. WHAT IT RESOLVED TO, for the batch's recipe ----------
        resolved_sort = None
        resolved_curation = None
        if ephys is not None:
            resolved_sort = ephys.sort
            resolved_curation = ephys.curation
        neuron_names = None
        if units_provenance:
            neuron_names = units_provenance.get("names")
        result.resolved = {
            "model": key,
            "sort": resolved_sort,
            "curation": resolved_curation,
            "units": counts.get("units", 0),
            "units_named": counts.get("units_named", 0),
            "neuron_names": neuron_names,
            "verified": result.verified,
            "notes": result.notes,
        }

        result.path = path
        result.counts = counts
        result.size_mb = round(path.stat().st_size / 1024 ** 2, 1)

    except Exception as exc:                      # noqa: BLE001 — reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"

    return result


# =====================================================================
# 4. The pieces of one file
#
# One function per thing that ends up in the file, each named for what it writes, in
# the order build_session_nwb adds them.
# =====================================================================

def _compressed(values):
    """Gzip anything with a row per frame (an array); leave small things alone."""
    if len(values) > 1000:
        return H5DataIO(values, compression="gzip", compression_opts=GZIP)
    return values


def _trials(nwb, trials, pose, frames, pulses):
    """
    One row per trial: the behaviour, the head at the trial start, and the laser.

    ``trials`` is trials.csv, ``pose`` trial_pose.csv, ``frames`` pose.csv and
    ``pulses`` the laser's pulses. Returns how many trials were written.

    ``start_time`` is the trial marker going high, which exists for every trial however
    it was cued — the one anchor that means the same thing all session. Anything measured
    from a cue instead belongs in the plot that knows which cue it wants.
    """
    table = trials.merge(pose, on="trial", how="left", suffixes=("", "_pose"))

    # ---- 1. The angle to every port --------------------------------------
    # A per-frame quantity, so it comes from pose.csv, read at the frame trial_pose.csv
    # recorded for the trial start. Without this an angle plot would have to open the
    # per-frame data for a number that costs six floats a trial.
    angles = frames.set_index("frame")[list(PORT_ANGLE_COLUMNS)]
    table[list(PORT_ANGLE_COLUMNS)] = angles.reindex(table["frame"]).to_numpy()

    table["response_time"] = table["t_response"] - table["t_start"]
    table["block"] = table["stage"]

    # ---- 2. The default reading of correctness ---------------------------
    # A trial is correct when it touched the port the cue asked for. It is **undefined**
    # — NaN — when the trial had no cued port, or when the animal touched nothing: a
    # timeout is a missing answer rather than a wrong one, and which of the two it should
    # count as belongs to the plot, not to the file.
    chosen = table["chosen_port"].to_numpy(dtype=float)
    cued = table["correct_port"].to_numpy(dtype=float)
    timed_out = chosen < 0
    correct = np.where(chosen == cued, 1.0, 0.0)
    correct[timed_out | (cued < 0)] = np.nan
    table["correct"], table["timed_out"] = correct, timed_out.astype(int)

    # ---- 3. The laser, per trial -----------------------------------------
    # So splitting a plot by stimulation needs no join. A pulse counts as a trial's when
    # it falls in [start - LASER_PRE_S, end] — see that constant for why the window
    # reaches back before the trial.
    starts = trials["t_start"].to_numpy(dtype=float)
    stops = trials["t_end"].to_numpy(dtype=float)
    if len(pulses):
        on = pulses["t_on"].to_numpy(dtype=float)
    else:
        on = np.empty(0)

    laser_counts = np.zeros(len(trials), dtype=int)
    first = np.full(len(trials), np.nan)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        belonging = on[(on >= start - LASER_PRE_S) & (on <= stop)]
        laser_counts[trial] = len(belonging)
        if len(belonging):
            first[trial] = belonging[0]

    table["laser_stim"] = (laser_counts > 0).astype(int)
    table["n_laser_pulses"] = laser_counts
    table["t_laser_on"] = first
    table["laser_offset_s"] = first - starts

    # ---- 4. Declare the columns ------------------------------------------
    #: column name -> description, in the order they are written
    described = {}
    described.update(TRIAL_COLUMNS)
    described.update(POSE_COLUMNS)
    for name in PORT_ANGLE_COLUMNS:
        described[name] = "the angle to this port at the trial start frame (degrees)"

    # A column the rig did not write for this session — details_json on the older
    # protocols — still gets written, empty, so every file has the same columns.
    for name in described:
        if name not in table:
            if name in TEXT_COLUMNS:
                table[name] = ""
            else:
                table[name] = np.nan
    for name, description in described.items():
        nwb.add_trial_column(name=name, description=description)

    # ---- 5. Each column as plain Python of a single type -----------------
    # See TEXT_COLUMNS.
    #: column name -> its values, one per trial
    typed = {}
    for name in described:
        values = table[name]
        if name in TEXT_COLUMNS:
            typed[name] = ["" if pd.isna(value) else str(value) for value in values]
        elif name in INT_COLUMNS:
            typed[name] = [int(value)
                           for value in pd.to_numeric(values, errors="coerce").fillna(-1)]
        else:
            typed[name] = [float(value) for value in pd.to_numeric(values, errors="coerce")]

    # ---- 6. The rows -----------------------------------------------------
    for trial in range(len(table)):
        row = {name: values[trial] for name, values in typed.items()}
        nwb.add_trial(start_time=float(table["t_start"].iloc[trial]),
                      stop_time=float(table["t_end"].iloc[trial]),
                      **row)
    return len(table)


def _epochs(nwb, trials):
    """
    One row per block — a run of consecutive trials sharing a block name. Returns how
    many.

    Derived here rather than read, because no session in the cohort this was written
    against recorded its blocks; the trials file's ``stage`` is the only record of which
    block was running. A session that does record them can be read directly instead.
    """
    if "stage" not in trials or trials.empty:
        return 0

    stage = trials["stage"].fillna("")
    changed = stage.ne(stage.shift()).cumsum()
    nwb.add_epoch_column(name="block", description="the block of parameters running")
    nwb.add_epoch_column(name="n_trials", description="trials in this block")

    count = 0
    for _, run in trials.groupby(changed, sort=True):
        nwb.add_epoch(start_time=float(run["t_start"].iloc[0]),
                      stop_time=float(run["t_end"].iloc[-1]),
                      block=str(run["stage"].iloc[0]), n_trials=len(run))
        count += 1
    return count


def _pulses(nwb, name, pulses, trials):
    """
    One row per pulse on a stimulation channel, with the trial it landed in. Returns how
    many.
    """
    if pulses.empty:
        return 0

    intervals = TimeIntervals(name=f"pulses_{name}",
                              description=f"every pulse on the {name} channel")
    intervals.add_column(name="trial", description="the trial it fell inside, -1 if none")

    starts = trials["t_start"].to_numpy(dtype=float)
    stops = trials["t_end"].to_numpy(dtype=float)
    for _, pulse in pulses.iterrows():
        # The same window as the trials' laser columns, so the table and they agree.
        belongs = np.flatnonzero((starts - LASER_PRE_S <= pulse["t_on"])
                                 & (stops >= pulse["t_on"]))
        if len(belongs):
            trial = int(trials["trial"].iloc[belongs[0]])
        else:
            trial = -1
        intervals.add_row(start_time=float(pulse["t_on"]), stop_time=float(pulse["t_off"]),
                          trial=trial)

    nwb.add_time_intervals(intervals)
    return len(pulses)


def _pose_series(pose, frame_times):
    """
    The head on every frame of the video, on the behaviour clock: the series to add, and
    the frame counts.

    Frames with no time are left out — an untimed frame cannot be placed on the clock,
    and a NaN in a timestamps array is a trap for every reader. ``video_frame`` keeps the
    way back to the AVI, so nothing is lost by leaving them out.

    All of these share one timestamps array: the first series holds it and the rest link,
    so the clock is stored once.
    """
    timed = pose.dropna(subset=["t"]).merge(
        frame_times[["frame", "frame_id"]], on="frame", how="left")
    timestamps = H5DataIO(timed["t"].to_numpy(dtype=float), compression="gzip",
                          compression_opts=GZIP)

    head = SpatialSeries(
        name="head_position",
        data=_compressed(timed[["head_x", "head_y"]].to_numpy(dtype=np.float32)),
        description="the ear midpoint pushed 40 px forward along the head bearing — the "
                    "point every angle in this file is measured from",
        reference_frame="top-left corner of the video image, x right, y down",
        unit="px", timestamps=timestamps)

    def series(name, values, unit, description):
        """A per-frame TimeSeries, its timestamps linked to the head's."""
        return TimeSeries(name=name, data=_compressed(values), unit=unit,
                          description=description, timestamps=head)

    bearing = SpatialSeries(
        name="head_bearing",
        data=_compressed(timed["bearing_deg"].to_numpy(dtype=np.float32)),
        description="which way the head faced, from the ear vector with the spine used "
                    "to catch the frames where it pointed backwards",
        reference_frame="the video image with y negated: 0 is to the right, 90 is up",
        unit="degrees", timestamps=head)

    # A string per frame would be a megabyte of text, so the method is a code and the
    # description says what the codes mean.
    methods = sorted(timed["method"].dropna().unique())
    codes = {name: number for number, name in enumerate(methods)}

    interfaces = [
        Position(spatial_series=head),
        CompassDirection(spatial_series=bearing),
        series("port_angles",
               timed[list(PORT_ANGLE_COLUMNS)].to_numpy(dtype=np.float32), "degrees",
               "where each of the six ports sat relative to straight ahead, one column "
               "per port, [-180, 180), 0 = straight ahead"),
        series("ear_likelihood", timed["ear_likelihood"].to_numpy(dtype=np.float32),
               "n/a", "DeepLabCut's confidence in the ears, the lower of the two"),
    ]
    # Pose built before these columns existed has neither; the series are then absent
    # rather than written blank, and load_frames gives blanks in their place.
    if {"head_speed_px_s", "turn_rate_deg_s"} <= set(timed.columns):
        interfaces.append(series(
            "head_speed", timed["head_speed_px_s"].to_numpy(dtype=np.float32), "px/s",
            "how fast the head moved to reach this frame from the one before. A "
            "label that jumped reads as a speed no mouse reaches"))
        interfaces.append(series(
            "turn_rate", timed["turn_rate_deg_s"].to_numpy(dtype=np.float32), "degrees/s",
            "how fast the bearing turned to reach this frame from the one before, "
            "positive anticlockwise. A bearing that swapped ends reads as a turn "
            "no neck allows"))

    code_list = ", ".join(f"{number} = {name}" for name, number in codes.items())
    interfaces.append(series(
        "bearing_method", timed["method"].map(codes).to_numpy(dtype=np.uint8), "n/a",
        "how the bearing was derived on this frame: " + code_list))
    interfaces.append(series(
        "video_frame", timed["frame"].to_numpy(dtype=np.int32), "n/a",
        "the row of the AVI, and of the DeepLabCut file, this frame is"))
    interfaces.append(series(
        "camera_frame_id", timed["frame_id"].to_numpy(dtype=np.int64), "n/a",
        "the camera's own frame ID, which counts frames it dropped too"))

    counts = {"frames": int(len(timed)), "frames_untimed": int(len(pose) - len(timed))}
    return interfaces, counts


def _events(events):
    """
    Every transition post-processing found on the behaviour channels, as it found them.

    A plain table rather than one time series per channel: it is what ``events.csv`` is,
    it reads back as a DataFrame, and it needs no NWB extension to be installed.
    """
    descriptions = {
        "t": "when it happened (s)",
        "event": "sensor_touch, sensor_release, led_on, led_off, led_dim_on, "
                 "led_dim_off (a PWM-dimmed period), valve_open, valve_close, "
                 "link_high, link_low",
        "channel": "the DAQ channel it happened on",
        "port": "the port that channel belongs to, 0-5, -1 if none",
        "in_trial": "1 where it fell inside a trial",
        "trial": "the trial it fell inside, -1 if none",
    }
    renamed = events.rename(columns={"index": "port"})
    # A session with no transitions at all still gets the table, empty — but an empty
    # list has no type for HDF5 to store, so each column says what it would hold.
    kinds = {"t": np.float64, "event": "U1", "channel": "U1", "port": np.int64,
             "in_trial": np.int64, "trial": np.int64}

    columns = []
    for name, description in descriptions.items():
        if len(renamed):
            data = renamed[name].to_list()
        else:
            data = np.array([], dtype=kinds[name])
        columns.append(VectorData(name=name, description=description, data=data))
    return DynamicTable(name="events", description="one row per hardware transition",
                        columns=columns)


def _notes(notes):
    """
    What the protocol annotated, on the **PC clock** — never the DAQ's.

    Carried through as its own table rather than as trial columns because its times
    cannot be placed on the clock everything else in this file uses: they are measured
    from when the *software* opened the trial, and ``start_time`` is when the *DAQ line*
    went high. See ``postprocessing/outputs/notes.py`` for the full argument.

    The column is called ``pc_offset_s`` for that reason, and there is deliberately no
    ``t`` here to be mistaken for one.
    """
    descriptions = {
        "trial": "the trial this annotation belongs to",
        "note": "0-based index within the trial, so repeated names stay apart",
        "kind": "stimulus (what the animal was shown) or event (anything else)",
        "note_name": "the modality for a stimulus, the protocol's own name for an "
                     "event. Not 'name', because a DynamicTable has one of its own "
                     "and a column called that reads back as the table's",
        "pc_offset_s": "seconds from the SOFTWARE trial start, on the PC wall clock. "
                       "NOT the DAQ clock and not comparable with start_time — the two "
                       "have different zeros, separated by an unmeasured serial latency. "
                       "Differences between two annotations of one trial are valid",
        "key": "the detail's name, empty where the annotation carries none",
        "value": "the detail, as text; JSON for anything that is not a plain string",
    }

    # Each column as plain Python, with blanks where a value is missing.
    columns = []
    for name, description in descriptions.items():
        if name not in notes or notes.empty:
            # Empty, but typed: an empty list has no type for HDF5 to store, and a
            # session with no trials has no notes — the early training sessions, the
            # optotagging runs.
            if name in ("trial", "note"):
                kind = np.int64
            elif name == "pc_offset_s":
                kind = np.float64
            else:
                kind = "U1"
            data = np.array([], dtype=kind)
        elif name in ("trial", "note"):
            data = [int(value) for value in notes[name].fillna(-1)]
        elif name == "pc_offset_s":
            data = [float(value) for value in notes[name]]
        else:
            data = ["" if pd.isna(value) else str(value) for value in notes[name]]
        columns.append(VectorData(name=name, description=description, data=data))

    return DynamicTable(
        name="notes",
        description="what the protocol recorded as each trial ran, on the PC clock",
        columns=columns)


def _ports(coords):
    """The calibrated port coordinates the angles in this file were measured from: an
    (n_ports, 2) array of x, y."""
    return DynamicTable(
        name="ports",
        description="the calibrated port coordinates used, in video pixels",
        columns=[
            VectorData(name="port", description="port number, 0-5",
                       data=list(range(N_PORTS))),
            VectorData(name="x", description="x in the video image (px)",
                       data=[float(x) for x, _ in coords]),
            VectorData(name="y", description="y in the video image (px)",
                       data=[float(y) for _, y in coords]),
        ])


def _scales(record) -> TimeSeries | None:
    """
    What the scales under the arena read, about 85 times a second. None for a session
    without them.

    On the behaviour clock like everything else, so a weight can be read at any moment of
    a trial. The readings are raw: the arena's own weight is in them, and the animal
    stepping on and off swings them about.
    """
    path = record.processed.get("scales")
    if path is None:
        return None
    scales = pd.read_csv(path).dropna(subset=["t"])
    return TimeSeries(
        name="scales", data=_compressed(scales["weight_g"].to_numpy(dtype=np.float32)),
        unit="g", description="what the scales under the arena read, as read: the "
                              "arena's own weight is included and the animal stepping "
                              "on and off swings it",
        timestamps=H5DataIO(scales["t"].to_numpy(dtype=float), compression="gzip",
                            compression_opts=GZIP))


def _stillness(record):
    """What the rig's stillness model made of the scales, as it ran: a list of
    TimeSeries, empty for a session without them.

    Acquisition rather than processing: a protocol may have waited on these, so
    they are part of what the session did, not a conclusion drawn afterwards.
    Re-scoring the weights later with a different model would give different
    numbers and would not be a record of this session. Which model produced
    them is in ``session_facts["stillness"]``, carried whole from the rig.

    Shares the scales' timestamps, because the rig scores once per weight
    reading, so no separate alignment is involved.
    """
    path = record.processed.get("scales")
    if path is None:
        return []
    frame = pd.read_csv(path).dropna(subset=["t"])
    if "p_still" not in frame.columns:
        return []           # session recorded before stillness, or with it off

    times = H5DataIO(frame["t"].to_numpy(dtype=float),
                     compression="gzip", compression_opts=GZIP)

    series = [TimeSeries(
        name="stillness_score",
        data=_compressed(frame["p_still"].to_numpy(dtype=np.float32)),
        unit="probability", timestamps=times,
        description="the rig's estimate that the animal was standing still, as "
                    "it ran. NaN where nothing was on the platform or the "
                    "window had not filled -- never 0, which would mean "
                    "'certainly moving'",
    )]

    if "still_for" in frame.columns:
        series.append(TimeSeries(
            name="still_for",
            data=_compressed(frame["still_for"].to_numpy(dtype=np.float32)),
            unit="seconds", timestamps=times,
            description="seconds of unbroken stillness at this reading, reset "
                        "the moment the animal moved or stepped off. This is "
                        "the number a protocol waits on",
        ))

    if "baseline_g" in frame.columns:
        series.append(TimeSeries(
            name="scales_baseline",
            data=_compressed(frame["baseline_g"].to_numpy(dtype=np.float32)),
            unit="g", timestamps=times,
            description="what the model took the empty platform to read. The "
                        "tare drifts, so occupancy is judged against this "
                        "rather than an absolute weight",
        ))

    return series


def _read_json(path) -> dict | None:
    """A JSON file's contents, or None when there is no such file (or no path)."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, TypeError, ValueError):
        return None


def _units(nwb, source):
    """
    The curation's good units, their spikes on the DAQ clock, and the probe they are on.
    ``source`` is an :class:`EphysSource`. Returns the counts, and what the units came
    from.

    The electrodes table is every channel of the probe, ``id`` = Phy's channel index, one
    ElectrodeGroup per shank. The positions are the sort's; whether the headstage wiring
    that put them there was right is recorded in ``sort.json`` and repeated in the build
    record.
    """
    # ---- 1. Read -----------------------------------------------------
    units = pd.read_csv(source.files["units"])
    spikes = pd.read_parquet(source.files["spikes"])
    waves = np.load(source.files["waveforms"])
    names = None
    if source.files.get("names"):
        names = pd.read_csv(source.files["names"], dtype={"neuron_id": str})
    sample_rate = float(waves["sample_rate"])

    positions = np.load(source.phy / "channel_positions.npy")
    shanks = np.load(source.phy / "channel_groups.npy")
    # What each channel was recorded as: SpikeInterface's names where the sort wrote
    # them, else built from Phy's 0-based channel map.
    if (source.phy / "channel_map_si.npy").exists():
        labels = np.load(source.phy / "channel_map_si.npy")
    else:
        labels = np.array([f"CH{c + 1}" for c in np.load(source.phy / "channel_map.npy")])

    # ---- 2. The probe and its channels -------------------------------
    # What the probe was, if the sort says: a tetrode drive is not a silicon probe, and
    # its channel groups are tetrodes, not shanks. The groups keep the name "shank<n>"
    # either way, so readers find them in one place; only the descriptions change.
    sort_record = _read_json(source.sort_record) or {}
    probe_info = sort_record.get("probe") or {}
    probe = nwb.create_device(name="probe", description=probe_info.get(
        "description", "silicon probe; shank and depth as the sort's "
                       "channel_positions.npy gives them"))
    group_word = probe_info.get("group", "shank")
    #: shank number -> its ElectrodeGroup
    groups = {}
    for shank in np.unique(shanks):
        groups[int(shank)] = nwb.create_electrode_group(
            name=f"shank{int(shank)}", description=f"{group_word} {int(shank)} of the probe",
            location="unknown", device=probe)
    nwb.add_electrode_column(name="open_ephys_channel",
                             description="the recorded channel this contact is on")
    for channel in range(len(positions)):
        nwb.add_electrode(id=channel, location="unknown", group=groups[int(shanks[channel])],
                          rel_x=float(positions[channel, 0]),
                          rel_y=float(positions[channel, 1]), rel_z=0.0,
                          open_ephys_channel=str(labels[channel]))

    # ---- 3. The units table ------------------------------------------
    nwb.units = Units(name="units", description="good units of one curation, spike times "
                      "on this file's clock (the DAQ's)", electrode_table=nwb.electrodes,
                      waveform_rate=sample_rate, waveform_unit="volts",
                      resolution=1.0 / sample_rate)
    for name, description in UNIT_COLUMNS.items():
        nwb.units.add_column(name=name, description=description)

    if names is not None:
        names = names.set_index("unit_id")
    wave_of = {int(u): w for u, w in zip(waves["unit_id"], waves["waveform"])}
    times = spikes.dropna(subset=["t_daq"])
    times_of = {int(u): g["t_daq"].to_numpy() for u, g in times.groupby("unit_id")}

    # ---- 4. One row per unit -----------------------------------------
    named = 0
    for _, unit in units.iterrows():
        unit_id = int(unit["unit_id"])
        named_here = names is not None and unit_id in names.index
        named += int(named_here)

        values = {name: unit[name] for name in UNIT_COLUMNS if name in unit}
        if named_here:
            values["neuron_id"] = names.at[unit_id, "neuron_id"]
            values["neuron_n_sessions"] = names.at[unit_id, "n_sessions"]
            values["neuron_matching"] = names.at[unit_id, "matching"]
            values["neuron_level"] = names.at[unit_id, "level"]
        else:
            values["neuron_id"] = ""
            values["neuron_n_sessions"] = -1
            values["neuron_matching"] = ""
            values["neuron_level"] = ""

        # Each value as the single type its column holds — see UNIT_TEXT, UNIT_INT.
        typed = {}
        for name, value in values.items():
            if name in UNIT_TEXT:
                typed[name] = "" if pd.isna(value) else str(value)
            elif name in UNIT_INT:
                typed[name] = -1 if pd.isna(value) else int(value)
            else:
                typed[name] = float(value) if pd.notna(value) else float("nan")

        nwb.units.add_unit(
            id=unit_id,
            spike_times=np.sort(times_of.get(unit_id, np.empty(0))),
            obs_intervals=[[float(unit["obs_start"]), float(unit["obs_stop"])]],
            electrodes=[int(unit["peak_channel"])],
            electrode_group=groups[int(unit["shank"])],
            waveform_mean=np.nan_to_num(wave_of[unit_id]) * 1e-6,
            **typed)

    # ---- 5. What came out, and from what -----------------------------
    counts = {"units": int(len(units)), "spikes": int(len(times)),
              "spikes_unsynced": int(len(spikes) - len(times)), "units_named": named}
    names_used = None
    if names is not None:
        names_used = {"matching": sorted(names["matching"].astype(str).unique()),
                      "level": sorted(names["level"].astype(str).unique())}
    provenance = {
        "sort": source.sort,
        "curation": source.curation,
        "sort_record": _read_json(source.sort_record),
        "curation_record": _read_json(source.files["record"]),
        "names": names_used,
    }
    return counts, provenance


# =====================================================================
# 5. The check
# =====================================================================

def _spread(n):
    """Twenty row numbers spread through n rows, or all of them if there are fewer."""
    return np.unique(np.linspace(0, n - 1, min(20, n)).astype(int))


def verify(path, trials, trial_pose, pose, events, counts):
    """
    Read the file back and compare it with the tables it was built from. Returns the
    problems found, as a list of strings — empty when it matches.

    Counts first, then actual values on twenty trials and twenty frames spread through
    the session: a file with the right number of wrong rows is the failure worth
    catching, and it is not caught by counting.
    """
    problems = []
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        behaviour = nwb.processing["behaviour"]
        # No trials table at all is right for a session with no trials, wrong otherwise.
        if nwb.trials is None:
            stored = pd.DataFrame()
        else:
            stored = nwb.trials.to_dataframe()
        head = behaviour["Position"]["head_position"]

        if len(stored) != len(trials):
            problems.append(f"{len(stored)} trials in the file, {len(trials)} in trials.csv")
        if len(behaviour["events"].id) != len(events):
            problems.append(f"{len(behaviour['events'].id)} events in the file, "
                            f"{len(events)} in events.csv")
        if len(head.timestamps) != counts["frames"]:
            problems.append(f"{len(head.timestamps)} timed frames in the file, "
                            f"{counts['frames']} expected")

        # Trials: times from trials.csv, angles from trial_pose.csv.
        if len(stored) == len(trials) and len(trials):
            picks = _spread(len(trials))
            checks = [("start_time", trials["t_start"]), ("t_response", trials["t_response"])]
            if "cue_angle_deg" in trial_pose:
                checks.append(("cue_angle_deg", trial_pose["cue_angle_deg"]))
            for column, expected in checks:
                left = pd.to_numeric(stored[column], errors="coerce").to_numpy(dtype=float)
                right = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
                if not np.allclose(left[picks], right[picks], atol=1e-6, equal_nan=True):
                    problems.append(f"trial column '{column}' does not match the CSV")

        # Frames: the clock and the head itself.
        timed = pose.dropna(subset=["t"])
        if len(head.timestamps) == len(timed) and len(timed):
            picks = _spread(len(timed))
            if not np.allclose(head.timestamps[picks],
                               timed["t"].to_numpy(dtype=float)[picks], atol=1e-6):
                problems.append("frame timestamps do not match pose.csv")
            if not np.allclose(np.asarray(head.data[picks], dtype=float),
                               timed[["head_x", "head_y"]].to_numpy(dtype=float)[picks],
                               atol=0.05, equal_nan=True):
                problems.append("head positions do not match pose.csv")

    return problems


def verify_units(path, source):
    """
    Read the units back and compare them with the curation's export (an
    :class:`EphysSource`). Returns the problems found, empty when they match.

    Every unit's spike count, and the actual times of a spread of spikes from each: the
    same check as the trials get, for the same reason.
    """
    problems = []
    units = pd.read_csv(source.files["units"])
    spikes = pd.read_parquet(source.files["spikes"]).dropna(subset=["t_daq"])
    with NWBHDF5IO(str(path), "r") as io:
        stored = io.read().units
        if stored is None:
            return ["no units table in the file"]
        ids = list(stored.id[:])
        if sorted(ids) != sorted(units["unit_id"].astype(int)):
            problems.append(f"{len(ids)} units in the file, {len(units)} in units.csv")
            return problems
        for row, unit_id in enumerate(ids):
            expected = np.sort(spikes.loc[spikes["unit_id"] == unit_id, "t_daq"].to_numpy())
            got = np.asarray(stored.get_unit_spike_times(row))
            if len(got) != len(expected):
                problems.append(f"unit {unit_id}: {len(got)} spikes in the file, "
                                f"{len(expected)} exported")
            elif len(got) and not np.allclose(got[_spread(len(got))],
                                              expected[_spread(len(got))], atol=1e-9):
                problems.append(f"unit {unit_id}: spike times do not match the export")
    return problems


# =====================================================================
# 6. What comes out
# =====================================================================

@dataclass(frozen=True)
class EphysSource:
    """Which sort and curation a session's units come from, and their files."""

    sort: str
    curation: str
    files: dict
    phy: Path
    sort_record: Path | None


@dataclass
class SessionNWB:
    """What one session produced, and anything noticed while producing it."""

    session_id: str
    model: str
    path: Path | None = None
    counts: dict = field(default_factory=dict)
    size_mb: float = 0.0
    verified: bool | None = None
    problems: list[str] = field(default_factory=list)
    skipped: str | None = None
    error: str | None = None
    #: Built, but worth knowing — a session asked for units that it could not have.
    notes: list[str] = field(default_factory=list)
    #: What this session actually resolved to, for a batch's recipe.
    resolved: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.error is None and self.skipped is None


@dataclass
class NWBResult:
    """What one sweep produced, per session."""

    base: Path
    model: str
    batch: str
    results: list[SessionNWB] = field(default_factory=list)

    @property
    def ok(self):
        return [r for r in self.results if r.ok]

    def describe(self):
        built = self.ok
        failed = [r for r in self.results if r.error]
        unverified = [r for r in built if not r.verified]
        skipped = [r for r in self.results if r.skipped]
        noted = [r for r in built if r.notes]

        lines = [f"{self.base.name} · {self.batch} — {len(built)} written, "
                 f"{len(failed)} failed, {len(skipped)} skipped"]
        if built:
            n_trials = sum(r.counts.get("trials", 0) for r in built)
            n_frames = sum(r.counts.get("frames", 0) for r in built)
            size_gb = sum(r.size_mb for r in built) / 1024
            lines.append(f"  {n_trials:,} trials, {n_frames:,} frames, "
                         f"{size_gb:.1f} GB in total")
            units = sum(r.counts.get("units", 0) for r in built)
            if units:
                n_spikes = sum(r.counts.get("spikes", 0) for r in built)
                n_named = sum(r.counts.get("units_named", 0) for r in built)
                lines.append(f"  {units} good units, {n_spikes:,} spikes, "
                             f"{n_named} with a neuron name")
        if unverified:
            lines += ["", "  DID NOT MATCH THE FILES THEY CAME FROM"]
            lines += [f"    {r.session_id}  {'; '.join(r.problems)}" for r in unverified]
        if noted:
            lines += ["", "  BUILT, BUT"]
            lines += [f"    {r.session_id}  {'; '.join(r.notes)}" for r in noted]
        if skipped:
            #: why -> how many sessions
            reasons = {}
            for item in skipped:
                reasons[item.skipped] = reasons.get(item.skipped, 0) + 1
            lines += ["", "  SKIPPED"]
            lines += [f"    {count} session(s): {why}" for why, count in sorted(reasons.items())]
        if failed:
            lines += ["", "  FAILED"]
            lines += [f"    {r.session_id}  {r.error}" for r in failed]
        return "\n".join(lines)


# =====================================================================
# 7. The command line
# =====================================================================

def cli(argv=None):
    """
    The command line. ``argv`` is a list of arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cohort", help="cohort folder")
    parser.add_argument("model", help="DeepLabCut model, by its Task name")
    parser.add_argument("--batch", required=True,
                        help="the batch to write: <session>/nwb/<batch>/ in each session")
    parser.add_argument("--sort", help="put this sort's units in")
    parser.add_argument("--curation", help="with --sort: which of its curations")
    parser.add_argument("--only", nargs="+", help="just these session IDs")
    parser.add_argument("--reprocess", action="store_true",
                        help="rebuild even where the file is still current")
    parser.add_argument("--no-scales", dest="scales", action="store_false",
                        help="leave the scales readings out")
    parser.add_argument("--workers", type=int, default=1, help="sessions at once")
    args = parser.parse_args(argv)

    only = None
    if args.only:
        only = tuple(args.only)
    result = build_nwb(args.cohort, args.model, batch=args.batch, sort=args.sort,
                       curation=args.curation, only=only,
                       reprocess=args.reprocess, scales=args.scales,
                       workers=args.workers)
    if any(r.error for r in result.results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli())
