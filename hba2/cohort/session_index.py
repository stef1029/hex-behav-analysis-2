"""
The session index — one row per session, and the labels you attach to them.
==========================================================================

A single CSV in the cohort folder, ``sessions.csv``, holding everything known about
every session, plus the things only you know: whether to include it, what drug was
given, which fibre position was used.

**Columns starting with ``_`` belong to this file. Everything else belongs to you.**

    session_id, _mouse_id, _date, _n_trials, …, include, stim_position, notes
    └ the key   └────────── rebuilt every time ──────────┘ └── never touched ──┘

That one rule is the whole design. Rebuilding recomputes every ``_`` column, appends
rows for sessions that have appeared, and leaves every other column exactly as it was —
so **nothing you type is ever overwritten**, and adding a label means typing a new
heading in a spreadsheet, with nothing to register anywhere.

Most of what an experiment needs labelled is recorded by the rig already: the laser
windows, which trackers ran, how long the session was. What is left is what no machine
could know — which fibre was plugged in, which syringe was picked up, whether the
session was any good — and that is a few columns in a table rather than code.

Reading it back::

    cohort = Cohort(path)
    index = read_index(cohort)
    sessions = index.query("include and stim_position == 1")

Nothing here opens a DAQ file. At 13 MB a session that would turn a rebuild into a
coffee break, so the columns are drawn from the small files only: the rig's metadata
JSON, and — once post-processing has run — ``session.json`` and ``trials.csv``.
Sessions that have not been processed simply have those columns blank, which is a
useful thing to be able to see.

The page reads in the order the data moves::

    1. where it is            index_path, read_index
    2. building               build_index: read every session, merge, write
    3. one session's row      session_row, and the files it is drawn from
    4. merging and writing    _read_for_merge, _write (shared with neuron_index.py)
    5. what comes out         IndexResult
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .cohort import Cohort
from ..postprocessing.sources import metadata

INDEX_NAME = "sessions.csv"

#: Marks a column this file owns. The contract with everything else in the file.
OWNED = "_"

#: The key. Bare, because it is the row's address rather than something known about it,
#: and because it is what every merge joins on.
KEY = "session_id"

#: Every column this file writes, in the order they appear. Anything not here and not
#: the key is yours.
GENERATED = (
    "_present",
    # who, when, what
    "_mouse_id", "_date", "_time", "_rig", "_protocol", "_phase",
    "_mouse_weight", "_requested_trials",
    # the laser, as the protocol armed it — the intended stimulation, not what the
    # hardware actually delivered, which lives on the DAQ channel
    "_laser_enabled", "_laser_windows_ms", "_laser_hz", "_laser_fixed_pulse",
    # the cue, as the protocol set it — the pre-rewrite rigs' "Cue duration" etc.
    "_cue_duration_ms", "_wait_duration_ms",
    "_catch_type", "_catch_brightness", "_catch_wait_s",
    # what happened, once post-processing has run
    "_is_processed", "_n_trials", "_duration_s",
    "_trackers", "_trial_types", "_blocks",
    "_n_frames", "_fps", "_n_dropped_frames", "_daq_reliability",
    # which stillness model scored the scales readings, and at what threshold
    "_has_stillness", "_still_model", "_still_model_version", "_still_model_digest",
    "_still_threshold", "_still_threshold_off_model",
    # Whether the video holds every frame the camera recorded. It usually does; when it
    # does not, every pose timestamp after the first missing frame is wrong.
    "_video_frames", "_frames_lost", "_video_complete",
    # Most incomplete videos lost their frames to a bug in the old converter, and their
    # timing is repairable — postprocessing/repairs/repair_video_timing.py. "repaired" means the frame
    # times are right again (the missing frames are still missing); "unexplained" means
    # the loss is something else and the timing is not to be trusted. The check is
    # whether heads sit on ports at beam breaks once repaired: "pass" or "fail".
    # A usable video is _video_complete, or _video_repair == "repaired" with a passed check.
    "_video_repair", "_video_repair_check", "_video_usable",
    # what else exists for this session
    "_has_video", "_dlc_models", "_has_pose", "_has_ephys", "_layout",
)

#: Started empty for you to fill. Deliberately few — a column costs one keystroke to
#: add, and columns that do not apply to a cohort are just clutter in every row.
LABELS = ("include", "exclude_reason", "notes")

#: How many stimulation slots a protocol can arm.
N_STIM_SLOTS = 6

#: The pre-rewrite rigs' names for the catch settings, which changed spelling partway
#: through 2024.
_CATCH_KEYS = {"type": ("Catch trial type", "Catch_trial_type"),
               "brightness": ("Catch brightness", "Catch_brightness"),
               "wait": ("Catch wait time", "Catch_wait")}
_CATCH_TYPES = {"w": "wait", "l": "led"}


# =====================================================================
# 1. Where it is, and reading it back
# =====================================================================

def index_path(cohort):
    """Where a cohort's index lives, given a :class:`Cohort` or its folder as a Path or
    a string. Always in the cohort folder, never elsewhere."""
    if isinstance(cohort, Cohort):
        base = cohort.base
    else:
        base = Path(cohort)
    return base / INDEX_NAME


def read_index(cohort):
    """
    The index, with types inferred — what analysis should use. ``cohort`` is a
    :class:`Cohort` or its folder as a Path or a string.

    Raises:
        FileNotFoundError: if the cohort has no index yet. Build one first.
    """
    path = index_path(cohort)
    if not path.exists():
        raise FileNotFoundError(
            f"no index at {path} — run build_index(cohort) to create it")
    return pd.read_csv(path, dtype={KEY: str})


# =====================================================================
# 2. Building — which is merging, never rewriting
# =====================================================================

def _say_nothing(message):
    """What build_index reports to when nobody is listening."""


def build_index(cohort, *, log_fn=None):
    """
    Create or refresh a cohort's session index.

    Args:
        cohort: a :class:`Cohort`, or a path (Path or string) to build one from.
        log_fn: called with each line as it happens.

    Safe to run whenever. Generated columns are recomputed, new sessions are appended,
    and every column you added is carried through untouched — including for sessions
    that have since been moved off disk, which are kept and marked ``_present`` false
    rather than dropped, so labelling is never lost to a reorganised folder.

    Returns an :class:`IndexResult`.
    """
    log = log_fn or _say_nothing
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    path = index_path(cohort)
    result = IndexResult(path=path)

    # ---- 1. Every session's generated columns, fresh ------------------
    log(f"{cohort.base.name}: reading {len(cohort)} session(s)")
    fresh = pd.DataFrame([session_row(record) for record in cohort])
    if fresh.empty:
        fresh = pd.DataFrame(columns=[KEY, *GENERATED])
    for column in GENERATED:
        if column not in fresh:
            fresh[column] = pd.NA
    fresh = fresh.set_index(KEY)

    # ---- 2. What is on disk already -----------------------------------
    existing = _read_for_merge(path)

    if existing is None:
        # ---- 3a. No index yet: the fresh rows, and empty labels -------
        merged = fresh.copy()
        for label in LABELS:
            merged[label] = ""
        result.n_new = len(merged)
        yours = list(LABELS)
    else:
        # ---- 3b. An index already: merge into it ----------------------
        existing = existing.set_index(KEY)
        # Anything this file does not own, in the order you left it.
        yours = [c for c in existing.columns if c not in GENERATED]

        unexpected = [c for c in yours if c.startswith(OWNED)]
        if unexpected:
            result.warnings.append(
                f"the leading underscore is reserved for generated columns, but "
                f"{', '.join(unexpected)} is not one — kept and left untouched")

        # Generated values that are not what they were. Worth reporting rather than
        # hiding: a trial count that moved means the session was reprocessed, and a
        # value that moved because somebody edited a generated column is exactly the
        # mistake this file cannot otherwise warn about.
        shared = existing.index.intersection(fresh.index)
        for column in GENERATED:
            if column == "_present" or column not in existing:
                continue
            before = existing.loc[shared, column].fillna("").astype(str)
            after = fresh.loc[shared, column].fillna("").astype(str)
            for session_id in shared[(before != after).to_numpy()]:
                was, now = before[session_id], after[session_id]
                result.changed.append(f"{session_id}  {column}: {was or '<blank>'} -> "
                                      f"{now or '<blank>'}")

        # Start from what is on disk so your columns and their order survive, then
        # append the sessions that are new and overwrite every generated column.
        merged = existing.copy()
        appeared = fresh.index.difference(merged.index)
        result.n_new = len(appeared)
        if len(appeared):
            merged = pd.concat(
                [merged, pd.DataFrame(index=appeared, columns=merged.columns)])

        for column in GENERATED:
            if column not in merged:
                merged[column] = pd.NA
        merged.loc[fresh.index, list(GENERATED)] = fresh[list(GENERATED)]
        merged["_present"] = merged.index.isin(fresh.index)

        for label in LABELS:
            if label not in merged:
                merged[label] = ""
                yours.append(label)

    # ---- 4. Write ------------------------------------------------------
    result.n_missing = int((~merged["_present"].astype(bool)).sum())
    result.n_sessions = len(merged)
    result.label_columns = [c for c in yours if c != KEY]

    merged = merged[[*GENERATED, *result.label_columns]].sort_index()
    merged.index.name = KEY
    _write(merged.reset_index(), path)

    # A line, not the whole report — the caller prints describe() when it wants it,
    # and postprocess_cohort logs at the same grain.
    line = f"{path.name}: {result.n_sessions} session(s)"
    if result.n_new:
        line += f", {result.n_new} new"
    if result.changed:
        line += f", {len(result.changed)} value(s) changed"
    log(line)
    return result


# =====================================================================
# 3. What one session contributes
# =====================================================================

def _joined(values):
    """A list of labels as one cell — sorted, deduplicated, semicolon separated."""
    return ";".join(sorted({str(v) for v in values if v not in (None, "")}))


def session_row(record):
    """Every generated column for one session (a SessionRecord), as a dict."""
    row = {
        KEY: record.session_id,
        "_present": True,
        "_is_processed": record.is_processed,
        "_has_video": record.raw.get("video") is not None,
        "_dlc_models": _joined(record.dlc_models),
        "_has_pose": bool(record.pose_models),
        "_has_ephys": record.has_ephys,
        "_layout": record.layout,
        # Fallbacks from the folder name, overwritten when metadata is readable.
        "_mouse_id": record.mouse_id,
    }
    row.update(_from_metadata(record))
    processed = _from_processed(record)
    row.update(processed)
    row.update(_video_integrity(record, processed.get("_n_frames")))
    row.update(_video_repair(record))
    # One column for "the video timing can be trusted", because every analysis was
    # spelling it out as two and risked spelling it differently.
    row["_video_usable"] = bool(
        row.get("_video_complete")
        or (row.get("_video_repair") == "repaired"
            and row.get("_video_repair_check") == "pass"))
    return row


def _number(value) -> float | None:
    """The value as a float, or None if it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(parameters, names):
    """The value of the first of ``names`` the parameters hold, or None."""
    return next((parameters[n] for n in names if n in parameters), None)


def _from_metadata(record):
    """Setup facts, from the small JSON the rig writes at session start."""
    path = record.raw.get("metadata")
    if path is None:
        return {}

    try:
        meta = metadata.read(path)
    except (OSError, ValueError):
        return {}

    # ---- 1. Who, when, what --------------------------------------------
    start = meta.start_time
    # ISO, so it sorts lexically and no spreadsheet reinterprets it as a number.
    date = ""
    time = ""
    if start:
        date = start.date().isoformat()
        time = start.time().isoformat(timespec="seconds")
    if meta.rig_number is not None:
        rig = meta.rig_number
    else:
        rig = meta.rig_name

    facts = {
        "_mouse_id": meta.mouse_id,
        "_date": date,
        "_time": time,
        "_rig": rig,
        "_protocol": meta.protocol_name,
        "_phase": meta.phase,
        "_mouse_weight": meta.mouse_weight,
        "_requested_trials": meta.requested_trials,
    }
    parameters = meta.protocol_parameters

    # ---- 2. The armed stimulation, flattened into four columns ----------
    # _laser_windows_ms is a string like "-500..-20" rather than a pair of numbers
    # because it is the natural name for a condition — it is what you would split a
    # plot by, and on the cohort this was built against it distinguishes every
    # experimental group exactly.
    enabled = bool(parameters.get("laser_stim_enabled", False))

    windows = []
    for slot in range(1, N_STIM_SLOTS + 1):
        if not parameters.get(f"stim_slot_{slot}_enabled"):
            continue
        start_ms = parameters.get(f"stim_slot_{slot}_start_ms")
        end_ms = parameters.get(f"stim_slot_{slot}_end_ms")
        if start_ms is None or end_ms is None:
            continue
        windows.append(f"{float(start_ms):g}..{float(end_ms):g}")

    facts["_laser_enabled"] = enabled
    if enabled:
        facts["_laser_windows_ms"] = ";".join(windows)
        facts["_laser_hz"] = parameters.get("laser_pulse_frequency_hz")
        facts["_laser_fixed_pulse"] = bool(parameters.get("laser_use_fixed_pulse"))
    else:
        facts["_laser_windows_ms"] = ""
        facts["_laser_hz"] = None
        facts["_laser_fixed_pulse"] = None

    # ---- 3. The cue, and what the catch trials changed -------------------
    # _cue_duration_ms 0 means the cue stayed on until the animal chose — the rigs wrote
    # 0 for that, and phase 9 wrote no cue duration at all because it had no other
    # setting, so a phase-9 session with none recorded is 0 too. Blank means the
    # protocol had no cue duration (a training phase).
    cue = _number(parameters.get("Cue duration"))
    if cue is None and str(meta.phase) == "9":
        cue = 0.0
    facts["_cue_duration_ms"] = cue
    facts["_wait_duration_ms"] = _number(parameters.get("Wait duration"))

    kind = _first_present(parameters, _CATCH_KEYS["type"])
    if kind:
        facts["_catch_type"] = _CATCH_TYPES.get(str(kind).lower(), str(kind))
        facts["_catch_brightness"] = _number(
            _first_present(parameters, _CATCH_KEYS["brightness"]))
        facts["_catch_wait_s"] = _number(_first_present(parameters, _CATCH_KEYS["wait"]))
    else:
        # No catch trials.
        facts["_catch_type"] = ""
        facts["_catch_brightness"] = None
        facts["_catch_wait_s"] = None

    return facts


def _from_processed(record):
    """
    What actually happened, from this package's own output.

    Empty until a session has been post-processed, which is the honest answer: the
    number of trials the rig *ran* is not knowable from the file it wrote before
    starting.
    """
    facts = {}

    # ---- 1. session.json ---------------------------------------------
    session_json = record.processed.get("session_json")
    if session_json is not None:
        try:
            stored = json.loads(Path(session_json).read_text())
        except (OSError, ValueError):
            stored = {}
        camera, clock = stored.get("camera", {}), stored.get("clock", {})

        # Which stillness model scored the session, as columns to group a cohort by.
        # _still_model_digest is the one to filter on. It is computed from what the
        # model does, so two sessions sharing it were scored by the same function,
        # whatever their artefacts were called. Name and version are here for reading;
        # they are only as trustworthy as whoever remembered to bump them, which is
        # exactly why the digest exists.
        stillness = stored.get("stillness") or {}
        model = stillness.get("model") or {}
        config = stillness.get("config_used") or {}
        facts["_has_stillness"] = bool(stillness.get("scored"))
        facts["_still_model"] = model.get("name")
        facts["_still_model_version"] = model.get("version")
        facts["_still_model_digest"] = model.get("digest")
        facts["_still_threshold"] = config.get("still_threshold")
        # True where the session ran at a threshold the model did not recommend. One
        # column, because every analysis comparing the two would otherwise spell the
        # comparison out for itself.
        facts["_still_threshold_off_model"] = "threshold_differs_from_model" in stillness

        duration = None
        if clock.get("duration_s") is not None:
            duration = round(clock["duration_s"], 1)
        reliability = None
        if clock.get("reliability") is not None:
            reliability = round(clock["reliability"], 3)
        facts["_duration_s"] = duration
        facts["_n_frames"] = camera.get("n_frames")
        facts["_fps"] = camera.get("fps")
        facts["_n_dropped_frames"] = camera.get("n_dropped")
        facts["_daq_reliability"] = reliability

    # ---- 2. trials.csv -----------------------------------------------
    trials_csv = record.processed.get("trials")
    if trials_csv is not None:
        try:
            trials = pd.read_csv(trials_csv)
        except (OSError, ValueError, pd.errors.ParserError):
            return facts
        facts["_n_trials"] = len(trials)
        for column, name in (("tracker", "_trackers"),
                             ("sub_tracker", "_trial_types"),
                             ("stage", "_blocks")):
            if column in trials:
                facts[name] = _joined(trials[column].dropna().unique())

    return facts


def _video_integrity(record, tracker_frames):
    """
    Whether the compressed video is missing frames, and how many. ``tracker_frames`` is
    how many frames the camera tracker logged, or None if unknown.

    This matters more than it sounds. The video is mapped onto the behaviour clock by
    position — frame *i* of the file is the *i*-th frame the camera tracker logged — so
    a video that lost frames misplaces every event after the first gap, by a margin that
    grows through the session. It still plays, and it still looks right.

    The frame count is taken from DeepLabCut's row count where that exists, because DLC
    decoded every frame one at a time. The container's own header is used only as a
    fallback: on damaged files it reports the frame count the video was *meant* to have.
    """
    unknown = {"_video_frames": None, "_frames_lost": None, "_video_complete": None}
    if record.raw.get("video") is None or not tracker_frames:
        return unknown

    # DeepLabCut's row count, as the pose stage recorded it.
    frames = None
    for model in record.pose_models or record.dlc_models:
        pose_json = record.directory / "dlc" / model / "pose.json"
        try:
            frames = json.loads(pose_json.read_text())["frames"]["n"]
            break
        except (OSError, ValueError, KeyError):
            continue

    # Failing that, the video's header.
    if frames is None:
        try:
            import cv2
            capture = cv2.VideoCapture(str(record.raw["video"]))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
        except Exception:                          # noqa: BLE001 — absence is a state
            return unknown
        if frames <= 0:
            return unknown

    lost = int(tracker_frames) - int(frames)
    return {"_video_frames": int(frames), "_frames_lost": lost,
            "_video_complete": lost == 0}


def _video_repair(record):
    """What the video-timing repair decided about this session, if it looked at it."""
    stored = record.processed.get("video_repair")
    try:
        if stored:
            repair = json.loads(Path(stored).read_text())
        else:
            repair = {}
    except (OSError, ValueError):
        repair = {"status": "unreadable"}

    # The check only counts if it ran.
    check = repair.get("check") or {}
    passed = None
    if check.get("ran"):
        passed = check.get("passed")
    if passed is None:
        verdict = None
    elif passed:
        verdict = "pass"
    else:
        verdict = "fail"
    return {"_video_repair": repair.get("status"), "_video_repair_check": verdict}


# =====================================================================
# 4. Reading what is there, and writing it back
#
# Shared with neuron_index.py, which keeps neurons.csv under the same rules.
# =====================================================================

def _read_for_merge(path) -> pd.DataFrame | None:
    """
    The existing index at ``path`` as plain text, or None if there is none yet.

    Everything is read as a string deliberately: generated columns are about to be
    replaced anyway, and reading your columns without interpretation is what stops a
    rebuild from quietly turning ``1`` into ``1.0`` or an empty cell into ``NaN``.
    """
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"{path} exists but could not be read: {exc}") from exc


def _write(frame, path):
    """
    Replace the index at ``path`` with ``frame`` atomically, keeping the previous
    version.

    This file is hand-entered work that exists nowhere else, so a half-written CSV is
    the one failure here that cannot be recovered by re-running anything.
    """
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    if path.exists():
        # Contents only: copying the timestamps too (copy2) fails when the .bak belongs to
        # another user of a shared cohort folder, and then the index is never replaced.
        shutil.copyfile(path, path.with_name(path.name + ".bak"))
    temporary.replace(path)


# =====================================================================
# 5. What comes out
# =====================================================================

@dataclass
class IndexResult:
    """What one build did."""

    path: Path
    n_sessions: int = 0
    n_new: int = 0
    n_missing: int = 0
    label_columns: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self):
        heading = f"{self.path.name}: {self.n_sessions} session(s)"
        if self.n_new:
            heading += f", {self.n_new} new"
        if self.n_missing:
            heading += f", {self.n_missing} no longer on disk"
        lines = [heading]
        if self.label_columns:
            lines.append("  your columns: " + ", ".join(self.label_columns))
        else:
            lines.append("  your columns: none")

        if self.changed:
            lines += ["", f"  CHANGED ({len(self.changed)})"]
            lines += [f"    {line}" for line in self.changed[:15]]
            if len(self.changed) > 15:
                lines.append(f"    ... and {len(self.changed) - 15} more")

        if self.warnings:
            lines += ["", "  WARNINGS"]
            lines += [f"    {w}" for w in self.warnings]

        return "\n".join(lines)
