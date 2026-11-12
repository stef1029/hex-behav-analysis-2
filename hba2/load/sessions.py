"""
From a cohort to the trials of the sessions you picked.
=======================================================

The step between "which sessions do I want" and "plot this"::

    from hba2 import load_sessions

    trials = load_sessions(cohort, include=True, stim_position="all",
                           laser_window="-400..-395")

Per-frame data — where the head was through a trial — is a separate call,
:func:`load_frames`, because it is a hundred times the size and most plots never touch it.

**It loads sessions, and what comes back is every trial in them.** Nothing here decides
which *trials* are worth using — no timeouts dropped, no confidence threshold, no
scoring. That sorting happens in the plot, where you can see it and change it. This is
the raw material.

One row per trial, every session stacked into one DataFrame, with the session's own facts
and your labels from ``sessions.csv`` attached as columns. That table is what every
plotter takes, and what you filter and group with ordinary pandas — so selecting sessions
is a query rather than a hand-written list, and a new experimental condition is a new
value in a column rather than new code.

**It reads the NWB files and nothing else.** Everything a plot needs is in them: the
trials with their head angles, the labels, the rig and the mouse. Build them with
``build_nwb`` first.

The one exception is ``where``, which is run against the cohort's live ``sessions.csv``,
because that is the file you edit when you change your mind about a session. Each NWB
also carries the labels as they were when it was built; ``labels="snapshot"`` uses those
instead, which is what to do with files that have travelled somewhere else.

Nothing is filtered out here. Timeouts, low-confidence pose and trials whose video frame
was far from the event all arrive, with the columns to spot them by — ``correct`` is NaN
where correctness is not defined, ``frame_lag_s`` says how close the pose reading was,
``response_distance_px`` how far the head was from the port it is recorded as touching.
Marking which of them are worth using is ``qc_trials``, once, in the notebook.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from ..cohort import Cohort
from ..cohort.session_index import index_path
from .rules import bare as _bare_rule, passes

#: The file each trial came from, carried on the table so :func:`load_frames` needs
#: nothing but the trials it is given. Underscored because it is plumbing, not data.
NWB_COLUMN = "_nwb"

#: Shorter names for the index columns worth filtering on, so a selection reads as
#: ``mouse="M1"`` rather than ``_mouse_id="M1"``. Any column of ``sessions.csv`` can be
#: used by its own name as well, with or without the leading underscore.
#: short name -> index column
ALIASES = {
    "mouse": "_mouse_id",
    "date": "_date",
    "rig": "_rig",
    "protocol": "_protocol",
    "phase": "_phase",
    "laser": "_laser_enabled",
    "laser_window": "_laser_windows_ms",
    "duration": "_duration_s",
    "trials": "_n_trials",
    "video_usable": "_video_usable",
    "video_complete": "_video_complete",
    "video_repair": "_video_repair",
    "video_repair_check": "_video_repair_check",
    "cue_duration": "_cue_duration_ms",
    "wait_duration": "_wait_duration_ms",
    "catch": "_catch_type",
}

#: Session facts pulled out of each file and attached to every one of its trials. The
#: rest stay in the file; these are the ones you group and split by.
#: column -> the keys leading to it inside the file's session facts
FACTS = {
    "mouse_id": ("mouse_id",),
    "rig": ("rig", "number"),
    "protocol": ("protocol_name",),
    "phase": ("phase",),
    "session_start": ("start_time",),
}

#: Per-frame trajectory checks in the NWB, and the column each becomes.
MOTION_SERIES = {"head_speed": "head_speed_px_s", "turn_rate": "turn_rate_deg_s"}


# =====================================================================
# From a cohort to its trials
# =====================================================================

def _say_nothing(message):
    """What a loader reports to when nobody is listening."""


def _index(cohort):
    """The cohort's index, or an empty frame when it has none."""
    path = index_path(cohort)
    if not path.exists():
        return pd.DataFrame(columns=["session_id"])
    return pd.read_csv(path, dtype={"session_id": str})


def _read(path):
    """
    One session's trials with its facts, and the labels stored beside them, as a
    ``(table, labels dict)`` pair.
    """
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        # A session whose trial marker never pulsed has pose but no trials — see
        # build_nwb. There is nothing here for a trial table.
        if nwb.trials is None:
            return pd.DataFrame(), {}
        trials = nwb.trials.to_dataframe().reset_index().rename(columns={"id": "trial"})
        facts = json.loads(nwb.scratch["session_facts"].data)
        labels = json.loads(nwb.scratch["session_labels"].data)
        trials.insert(0, "session_id", nwb.session_id)

    for name, keys in FACTS.items():
        value = facts
        for key in keys:
            value = (value or {}).get(key)
        trials.insert(1, name, value)

    return trials, labels


def _typed(value):
    """
    A label as the type it reads as, whichever way it arrived.

    The index is read by pandas and arrives typed; the snapshot inside a file is text,
    because that stage has no business deciding what a label of yours means. Here they
    become the same thing, so ``include == True`` works either way.
    """
    if not isinstance(value, str):
        return value
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            continue
    return value


def load_sessions(cohort, *, batch=None, where=None, only=None, labels="index",
                  log_fn=None, **filters):
    """
    The sessions you asked for, as every trial in them.

    Args:
        cohort: a :class:`Cohort`, or a path (Path or string) to build one from.
        batch: which batch's files to read, from ``<session>/nwb/<batch>/`` — the ones
            with units in, for :func:`~.spikes.load_units` and ``load_spikes``. Left
            out, it is the cohort's own batch: the one named in ``Cohort(path,
            batch=...)``, or else the newest.
        **filters: how sessions are chosen — a column of ``sessions.csv`` and what it
            must hold, in the same vocabulary ``qc_trials`` uses::

                include=True, stim_position="all"      equal to
                mouse=["M1", "M2"]                   any of
                not_mouse="test"                       anything but
                min_duration=1800, max_n_trials=900    at or above, at or below
                min_date="2026-06-01"                  text orders too

            Columns can be named by their alias (:data:`ALIASES`), their own name, or
            without the leading underscore, so ``mouse``, ``_mouse_id`` and ``mouse_id``
            are one column. A name that is not a column raises, listing the ones that
            are; a value nothing has raises, listing the values it holds.
        where: a pandas query string, for the one thing keywords cannot say — an ``or``
            across two columns. Needing it usually means the index is missing a column:
            ``_video_usable`` exists because "complete, or repaired and checked" was
            being written out every time.
        only: a tuple of session IDs. Given with filters, it narrows them further.
        labels: ``"index"`` reads your labels from ``sessions.csv`` as it is now;
            ``"snapshot"`` reads the copy stored in each file when it was built. Use the
            snapshot for files that have left this machine, the index otherwise — it is
            the one you edit.
        log_fn: called with a line per step; ``print`` to watch it.

    Returns:
        One row per trial, unsorted and unfiltered. Trial columns as
        :mod:`hba2.nwb.build_nwb` describes them, plus ``session_id``, the
        facts in :data:`FACTS`, and every label column from the index.
    """
    log = log_fn or _say_nothing
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort, batch=batch)

    # ---- 1. Which batch: the one asked for, or the cohort's own ----------
    if batch is None:
        batch = cohort.batch
    if batch is None:
        raise ValueError(f"{cohort.base.name} has no NWB batch yet — run build_nwb "
                         "with --batch")
    if batch not in cohort.batches:
        raise ValueError(f"no batch {batch!r} in {cohort.base.name}. "
                         f"Batches: {', '.join(cohort.batches) or 'none'}")

    # ---- 2. Which sessions ----------------------------------------------
    # Nothing given means every session that has a file; each of the three ways of
    # choosing narrows what the others left.
    index = _index(cohort)
    wanted = None
    if filters:
        chosen = index[_matching(index, filters)]
        wanted = set(chosen["session_id"])
    if where:
        queried = set(index.query(where)["session_id"])
        wanted = queried if wanted is None else wanted & queried
    if only:
        wanted = set(only) if wanted is None else wanted & set(only)

    records = [r for r in cohort if wanted is None or r.session_id in wanted]
    asked = [f"{name}={value!r}" for name, value in filters.items()]
    if where:
        asked.append(where)
    line = f"{cohort.base.name}: {len(records)} session(s) selected"
    if asked:
        line += f" by {', '.join(asked)}"
    line += f" of {len(index) or len(cohort)}"
    log(line)

    # ---- 3. Each session's file -----------------------------------------
    tables, snapshots, missing, empty = [], {}, [], []
    for record in records:
        path = cohort.batch_path(batch, record.session_id)
        if not path.exists():
            missing.append(record.session_id)
            continue
        table, snapshot = _read(path)
        if table.empty:
            empty.append(record.session_id)
            continue
        # Where this session's per-frame data is, so a plot that wants the time course
        # can find it from the table alone — see load_frames.
        table[NWB_COLUMN] = str(path)
        tables.append(table)
        snapshots[record.session_id] = snapshot

    if missing:
        if len(missing) > 1:
            which = f"{missing[0]} and {len(missing) - 1} more"
        else:
            which = missing[0]
        log(f"  {len(missing)} session(s) have no NWB for batch '{batch}' — run "
            f"build_nwb: {which}")
    if empty:
        log(f"  {len(empty)} session(s) have no trials")
    if not tables:
        raise ValueError("no trials loaded — check the query, the batch name, "
                         "and that build_nwb has run")

    # ---- 4. Your labels -------------------------------------------------
    # Attached once, to the whole table: a session whose label is blank would otherwise
    # contribute an empty column that pandas then has to reconcile with the filled ones.
    trials = pd.concat(tables, ignore_index=True)
    if labels == "index":
        label_table = index
    elif not snapshots:
        label_table = pd.DataFrame(columns=["session_id"])
    else:
        # The labels each file stored when it was built, as one table.
        label_table = pd.DataFrame.from_dict(snapshots, orient="index")
        label_table["session_id"] = label_table.index
        label_table = label_table.reset_index(drop=True).map(_typed)
    if not label_table.empty:
        spare = [name for name in label_table.columns
                 if name == "session_id" or name not in trials.columns]
        trials = trials.merge(label_table[spare], on="session_id", how="left")

    log(f"  {len(trials):,} trials from {len(tables)} session(s), "
        f"{trials['mouse_id'].nunique()} mice")
    return trials


def _matching(index, filters):
    """
    Which rows of the index match every keyword given, as a boolean Series.

    A value that is a list, tuple or set means any of them; anything else means equal to
    it. A keyword that matches nothing stops the call and says what the column does
    hold — silently loading no sessions is how a figure ends up describing a cohort it
    never read.
    """
    match = pd.Series(True, index=index.index)
    for rule, value in filters.items():
        here = passes(index, rule, value, column_for=_session_column)
        if not here.any():
            column = _session_column(index, _bare_rule(rule), rule)
            # The values the column takes, with counts, for an error worth reading.
            counts = index[column].value_counts(dropna=False)
            shown = ", ".join(f"{held} ({n})" for held, n in counts.head(8).items())
            if len(counts) > 8:
                shown += f", and {len(counts) - 8} more"
            raise ValueError(f"no sessions have {rule}={value!r}. {column} holds: {shown}")
        match &= here

    if not match.any():
        asked = ", ".join(f"{name}={value!r}" for name, value in filters.items())
        raise ValueError(f"no sessions match all of {asked} — each matches some on its "
                         "own, so it is the combination that is empty")
    return match


def _session_column(index, name, rule=""):
    """The index column a keyword means: its alias, its own name, or underscored."""
    for candidate in (ALIASES.get(name, name), name, f"_{name}"):
        if candidate in index.columns:
            return candidate
    raise KeyError(
        f"no column {name!r} in sessions.csv. Yours: "
        f"{', '.join(c for c in index.columns if not c.startswith('_'))}. "
        f"Generated: {', '.join(c for c in index.columns if c.startswith('_'))}. "
        f"Short names: {', '.join(ALIASES)}")


# =====================================================================
# What is behind the trials: frames and ports
# =====================================================================

def _needs(trials, column, who):
    """Stop when a table did not come from load_sessions and so lacks ``column``."""
    if column not in trials:
        raise KeyError(
            f"{who} needs the {column!r} column, which load_sessions adds. Pass it the "
            "table you loaded (or a slice of it) rather than one built another way.")


def load_frames(trials, *, before=0.0, after=0.0, port_angles=False, log_fn=None):
    """
    The head frame by frame, through the trials you hand it.

    The per-trial table has the head at two moments; this is the time course in between,
    and it is a separate call because it is a hundred times the data. Ask for the trials
    you are actually going to draw::

        frames = load_frames(trials.head(12), before=0.5, after=0.5)

    Args:
        trials: rows of a :func:`load_sessions` table — all of them, or a handful.
        before: seconds before each trial's start to include, for the approach.
        after: seconds after each trial's end.
        port_angles: also read the angle to each of the six ports on every frame.
        log_fn: called with a line per file opened.

    Returns:
        One row per frame per trial: ``session_id``, ``trial``, ``video_frame``, ``t``,
        ``t_from_start``, ``head_x``, ``head_y``, ``bearing_deg``, ``ear_likelihood``,
        ``head_speed_px_s``, ``turn_rate_deg_s`` — the last two how fast the head moved
        and turned arriving at that frame, the trajectory's own check on a frame:
        ``frames[frames["head_speed_px_s"] <= 2500]``.

    Frames the video never had are simply absent, so ``t`` can jump: a line drawn through
    it should break where the step is bigger than a frame or two, rather than inventing a
    straight run across the gap.
    """
    log = log_fn or _say_nothing
    _needs(trials, NWB_COLUMN, "load_frames")

    pieces = []
    # One open per file rather than per trial: opening is the expensive part, the slices
    # themselves are milliseconds.
    for path, rows in trials.groupby(NWB_COLUMN, sort=False):
        log(f"  {Path(path).name}: {len(rows)} trial(s)")
        with NWBHDF5IO(str(path), "r") as io:
            behaviour = io.read().processing["behaviour"]
            head = behaviour["Position"]["head_position"]
            series = {
                "head": head,
                "bearing_deg": behaviour["CompassDirection"]["head_bearing"],
                "ear_likelihood": behaviour["ear_likelihood"],
                "video_frame": behaviour["video_frame"],
            }
            # The trajectory checks, in files built since they existed. Older files give
            # blanks, which fail any limit set on them rather than pass it unseen.
            for name, column in MOTION_SERIES.items():
                if name in behaviour.data_interfaces:
                    series[column] = behaviour[name]
            if port_angles:
                series["port_angles"] = behaviour["port_angles"]
            stamps = head.timestamps[:]

            for _, trial in rows.iterrows():
                low = int(np.searchsorted(stamps, trial["start_time"] - before, "left"))
                high = int(np.searchsorted(stamps, trial["stop_time"] + after, "right"))
                if high <= low:
                    continue
                piece = pd.DataFrame({
                    "session_id": trial["session_id"],
                    "trial": trial["trial"],
                    "video_frame": series["video_frame"].data[low:high],
                    "t": stamps[low:high],
                    "t_from_start": stamps[low:high] - trial["start_time"],
                    "head_x": head.data[low:high, 0],
                    "head_y": head.data[low:high, 1],
                    "bearing_deg": series["bearing_deg"].data[low:high],
                    "ear_likelihood": series["ear_likelihood"].data[low:high],
                })
                for column in MOTION_SERIES.values():
                    if column in series:
                        piece[column] = series[column].data[low:high]
                    else:
                        piece[column] = np.nan
                if port_angles:
                    angles = series["port_angles"].data[low:high]
                    for port in range(angles.shape[1]):
                        piece[f"angle_port{port}"] = angles[:, port]
                pieces.append(piece)

    if not pieces:
        raise ValueError("no frames for those trials — check the window and the times")
    frames = pd.concat(pieces, ignore_index=True)
    log(f"  {len(frames):,} frames for {len(pieces)} trial(s)")
    return frames


def load_ports(trials):
    """
    The calibrated port coordinates behind each session, for drawing the arena.

    One row per port per session. They are a property of the rig, so sessions from the
    same rig agree exactly and a plot spanning one rig can use any of them; a plot
    spanning two rigs cannot, which is a thing worth noticing rather than averaging.
    """
    _needs(trials, NWB_COLUMN, "load_ports")

    rows = []
    for path, group in trials.groupby(NWB_COLUMN, sort=False):
        with NWBHDF5IO(str(path), "r") as io:
            nwb = io.read()
            ports = nwb.processing["behaviour"]["ports"].to_dataframe()
        ports.insert(0, "session_id", nwb.session_id)
        rows.append(ports)
    return pd.concat(rows, ignore_index=True)


# =====================================================================
# What there is: the menus
# =====================================================================

def describe_sessions(cohort):
    """
    What you can filter sessions by, read from the cohort itself.

    The menu for :func:`load_sessions`: every column of ``sessions.csv``, whether it is
    one of yours or one the pipeline generates, how many sessions have it filled in, and
    the values it takes where there are few enough to list.

    Read from the file rather than written down anywhere, so it is right for this cohort
    rather than for the one someone documented::

        describe_sessions(cohort)                     # the whole menu
        describe_sessions(cohort).query("kind == 'yours'")   # just your labels
    """
    index = _index(cohort)
    rows = []
    for column in index.columns:
        if column == "session_id":
            continue
        filled = index[column].dropna()
        distinct = filled.unique()

        # Few enough values to list them, up to six; otherwise how many there are.
        if len(distinct) <= 15:
            values = ", ".join(map(str, distinct[:6]))
            if len(distinct) > 6:
                values += " ..."
        else:
            values = f"{len(distinct)} distinct"

        # The short name a column can be filtered by, if it has one.
        filter_by = column
        for alias, target in ALIASES.items():
            if target == column:
                filter_by = alias
                break

        if column.startswith("_"):
            kind = "generated"
        else:
            kind = "yours"
        rows.append({
            "column": column,
            "kind": kind,
            "filter_by": filter_by,
            "filled": f"{len(filled)}/{len(index)}",
            "values": values,
        })
    return pd.DataFrame(rows)


def describe_trials(trials):
    """
    What each trial column means, and what it looks like.

    The descriptions come from the NWB files, where ``build_nwb`` wrote them, so they
    describe the data in front of you rather than a docstring someone forgot to update.

    Takes a loaded table (it finds a file through its ``_nwb`` column) or a path to one
    NWB file, as a Path or a string.
    """
    if isinstance(trials, (str, Path)):
        path, table = Path(trials), None
    else:
        _needs(trials, NWB_COLUMN, "describe_trials")
        path, table = Path(trials[NWB_COLUMN].iloc[0]), trials

    with NWBHDF5IO(str(path), "r") as io:
        stored = io.read().trials
        described = {column.name: column.description for column in stored.columns
                     if hasattr(column, "description")}
        if table is None:
            table = stored.to_dataframe()

    rows = []
    for column, description in described.items():
        if column not in table:
            continue
        values = table[column].dropna()
        if len(values):
            example = str(values.iloc[0])[:40]
        else:
            example = ""
        rows.append({"column": column, "description": description,
                     "dtype": str(table[column].dtype), "example": example})
    return pd.DataFrame(rows)


def read_session_trials(path):
    """
    One session's trials, with its facts and its own copy of your labels.

    For a single session on its own; :func:`load_sessions` does the labels its own way.
    """
    trials, labels = _read(path)
    for name, value in labels.items():
        if name != "session_id" and name not in trials.columns:
            trials[name] = _typed(value)
    return trials
