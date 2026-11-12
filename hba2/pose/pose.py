"""
Head angles, from DeepLabCut output.
====================================

What this stage produces, for one session and one DeepLabCut model: where the animal's
head was and which way it faced on **every** frame, and where each of the six ports sat
relative to that. The question nearly every analysis asks — *how far from straight ahead
was the cue when it appeared* — then becomes a column lookup rather than a calculation.

It runs **after** DeepLabCut, which lands hours to days after a session, and after
post-processing, whose ``events.csv``, ``frame_times.csv`` and ``trials.csv`` it reads.
It blocks neither of them, and neither of them waits for it.

Output goes inside the model's own folder::

    <session>/dlc/<model>/
    ├── <session>_videoDLC_<model>.csv    DeepLabCut wrote this
    ├── pose.csv          one row per frame
    ├── trial_pose.csv    one row per trial — the file most analysis wants
    └── pose.json         the port coordinates used, and what this was built from

so two models are two folders that cannot collide, and deleting a model deletes what was
derived from it. The path is the provenance.

**A head position is the midpoint of the ears pushed forward along the head bearing.**
That is the point the previous library measures from, so an angle from here means the
same thing as an angle from there. The bearing comes from the ear vector, which is
ambiguous by 180 degrees — the animal could face either way along it — so the spine is
used to catch and flip the frames where it points backwards.

**Confidence is not applied here.** Every geometrically computable frame gets an angle,
and the ear likelihood travels beside it in the table. Raising the confidence floor is
then a filter at load time rather than a rebuild, and nobody reprocesses a cohort
because they changed their mind about a threshold.

**Nor is a jump threshold.** Likelihood is DeepLabCut's opinion; whether the head moved
like a head is a fact about the trajectory, and the better test of a frame — a frame can
be low-confidence and exactly right, or confident and on the wrong side of the arena.
So every frame also carries how fast the head moved and turned to get there from the
frame before (``head_speed_px_s``, ``turn_rate_deg_s``), and every looked-up moment
carries the same at its frame. A label that jumps shows as a speed no mouse reaches, a
bearing that swaps ends as a turn rate no neck allows; the limits are again a filter at
load time — ``qc_trials(max_head_speed_px_s=..., max_turn_rate_deg_s=...)``.

The page, top to bottom::

    the numbers            what defines a head position, and the file names
    where the ports are    the calibrated table, read, never derived
    build_pose             every session in a cohort with output from one model
    build_session_pose     one session: read, head, moments, ports, write, warnings
    reading DeepLabCut     the only code that knows what a DLC file looks like
    the geometry           pure array maths
    the three files        pose.csv, trial_pose.csv, pose.json
    SessionPose,
    PoseResult             what comes back
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..cohort import Cohort, SessionRecord
from .frame_lookup import MAX_LAG_S, frames_at

# =====================================================================
# 1. The numbers that define a head position
#
# Both come from the previous library, and both are recorded in every pose.json so a
# table always says what produced it rather than relying on this file not having moved.
# =====================================================================

N_PORTS = 6

#: How far forward of the ear midpoint the reference point sits, in pixels. Puts it at
#: roughly the eyes, nearer the part of the animal that actually reaches a port.
EYES_OFFSET_PX = 40.0

#: Ears closer together than this are almost always one ear detected twice, and the
#: bearing derived from them points anywhere. Those frames fall through to the spine.
MIN_EAR_DISTANCE_PX = 50.0

PARAMS = {"eyes_offset_px": EYES_OFFSET_PX,
          "min_ear_distance_px": MIN_EAR_DISTANCE_PX,
          # Touches whose nearest frame is further than this are left out of the port
          # scatter — a summary cannot carry a lag per touch. The per-trial lookups keep
          # every value and write the lag beside it instead.
          "max_touch_lag_s": MAX_LAG_S,
          # Recorded so that pose built without these columns reads as stale and is
          # rebuilt — see head_motion and moment_motion.
          "motion": "head_speed_px_s, turn_rate_deg_s: from the frame before; at a "
                    "moment, the larger of the steps into and out of its frame"}

#: The post-processed files this stage reads. Their fingerprints go in pose.json, so a
#: re-run of post-processing — or a repaired frame_times.csv — makes the output stale.
INPUTS = ("frame_times", "trials", "events")

#: Above this, the head is too far from a port when that port's beam breaks for the
#: touches to be pokes. Flagged rather than fatal: the angles are still right, but the
#: session's chosen ports and response times are not to be trusted.
MAX_PORT_SCATTER_PX = 40.0

POSE_CSV = "pose.csv"
TRIAL_POSE_CSV = "trial_pose.csv"
POSE_JSON = "pose.json"

SPINE_PARTS = ("spine_1", "spine_2", "spine_3", "spine_4")

#: Key meaning "use this table whichever rig the session ran on".
ANY_RIG = -1

#: The calibrated port coordinates, produced by ``calibrate_ports.py`` and checked by
#: eye against the picture beside them. In the package and under version control, which
#: is the only location that resolves the same on every machine with no configuration —
#: and makes a recalibration a diff someone can read and date.
PORT_TABLE = Path(__file__).resolve().parent / "port_coordinates" / "port_coordinates.json"


# =====================================================================
# 2. Where the ports are
#
# Read, never derived. The coordinates come from the calibrated table that
# ``calibrate_ports.py`` produces by pooling a whole cohort and which somebody has
# looked at. Deriving them per session would mean two analyses silently measuring
# angles from different origins, so this file does not do it at all.
# =====================================================================

def load_port_table(path=PORT_TABLE):
    """The calibrated table exactly as written (a dict), or an empty one if there is none."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def approved_ports(path=PORT_TABLE):
    """``{rig number: coordinates}`` — the table every analysis measures angles from."""
    return {int(name.removeprefix("rig_")): np.asarray(entry["coords"], dtype=float)
            for name, entry in load_port_table(path).items()
            if isinstance(entry, dict) and entry.get("coords")}


def _supplied_tables(ports):
    """
    Turn a supplied port table into ``{rig number: coords}``.

    Accepts a path to JSON (a Path or a string), a mapping of rig number to six
    coordinate pairs, or a bare list of six pairs meaning every rig.
    """
    if isinstance(ports, (str, Path)):
        ports = json.loads(Path(ports).read_text())

    if isinstance(ports, dict):
        # A table calibrate_ports.py wrote — {"rig_1": {"coords": ..., ...}} — as well as
        # a plain {rig: coords}. The legacy rigs' tables, one per period, are of the
        # first kind (port_coordinates/legacy/<cohort>/port_coordinates.json).
        tables = {}
        for rig, entry in ports.items():
            if isinstance(entry, dict):
                # A calibrate_ports.py entry; one without coordinates is left out.
                if not entry.get("coords"):
                    continue
                coords = entry["coords"]
            else:
                # A plain list of coordinate pairs.
                coords = entry
            tables[int(str(rig).removeprefix("rig_"))] = np.asarray(coords, dtype=float)
        return tables

    coords = np.asarray(ports, dtype=float)
    if coords.shape != (N_PORTS, 2):
        raise ValueError(f"expected {N_PORTS} coordinate pairs, got {coords.shape}")
    return {ANY_RIG: coords}


def _rig_number(record):
    """Which rig a session ran on, from its post-processed session facts."""
    facts = json.loads(record.path("session_json").read_text())
    return int(facts.get("rig", {}).get("number", ANY_RIG))


# =====================================================================
# 3. A whole cohort
# =====================================================================

def _say_nothing(message):
    """What build_pose reports to when nobody is listening."""


def _never_cancel():
    """What build_pose polls when nobody can cancel it."""
    return False


def _model_paths(record, model):
    """
    Resolve a model name to this session's DeepLabCut CSV and output folder.

    Returns ``(csv path, folder)``, both None when the session has no output from the
    model.

    Models are named by **Task** — the short name from the config, which is what the
    folder is called. :meth:`SessionRecord.dlc_key` also accepts the longer generated
    name that sessions analysed before that convention are keyed by, so one Task works
    across both.
    """
    key = record.dlc_key(model)
    if key is None:
        return None, None
    return (record.dlc.get(key) or {}).get("csv"), record.directory / "dlc" / key


def _fingerprint(path):
    """Enough about a file (a Path or a string) to notice it has changed, without reading it."""
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def _inputs(record):
    """Fingerprints of the post-processed files this stage read."""
    return {role: _fingerprint(record.path(role)) for role in INPUTS}


def _is_current(out, record, source, ports):
    """
    True when a rebuild would produce the same thing as what is already there.

    ``ports`` is the coordinate array the rebuild would use, or None to skip that check.
    """
    try:
        stored = json.loads((out / POSE_JSON).read_text())
        inputs = _inputs(record)
        if ports is not None:
            stored_ports = stored["ports"]["coords"]
    except (OSError, ValueError, KeyError, TypeError):
        # Unreadable, or written before it recorded something checked here: rebuild.
        return False

    fingerprint = _fingerprint(source)
    if {key: stored.get("source", {}).get(key) for key in fingerprint} != fingerprint:
        return False
    # Post-processing re-run, or frame_times.csv repaired, since this was built.
    if stored.get("inputs") != inputs:
        return False
    if stored.get("params") != PARAMS:
        return False
    if ports is not None:
        return np.allclose(stored_ports, np.round(ports, 1), atol=0.11)
    return True


def build_pose(cohort, model, *, ports="approved", only=None, reprocess=False,
               log_fn=None, progress_fn=None, should_cancel=None):
    """
    Build head angles for every session in a cohort that has output from ``model``.

    Args:
        cohort: a :class:`Cohort`, or a path (Path or string) to build one from.
        model: which DeepLabCut model's output to read. One call, one model; two models
            are two calls writing to two folders.
        ports: where port coordinates come from.

            * ``"approved"`` (default) — the calibrated table shipped with the package,
              looked up by rig. Produced and eyeballed once by ``calibrate_ports.py``,
              then used unchanged by every cohort, so two analyses never silently
              measure angles from different origins.
            * a table — a path to JSON, a mapping of rig number to six coordinate
              pairs, or a bare list of six pairs for every rig. For trying something
              out without touching the approved file.

        only: a tuple of session IDs to build, and nothing else.
        reprocess: rebuild even where the existing output is still current.
        log_fn: called with each line as it happens. Without it the run is silent until
            it returns, which for a large cohort is a long time.
        progress_fn: called with a short status string per session, for a UI label.
        should_cancel: called with no arguments between sessions; stop when it returns
            True. Checked between rather than within, so a session is never left
            half-written.

    Returns a :class:`PoseResult`.

    Ports are never measured here. Calibration is a deliberate act with a picture to
    check — see ``calibrate_ports.py`` — and keeping it out of the build is what stops
    the coordinates quietly depending on which sessions happened to be in a cohort.

    Deliberately serial, for the same reason post-processing is: sessions are
    independent and would parallelise, but serial keeps the failure modes simple.
    """
    log = log_fn or _say_nothing
    progress = progress_fn or _say_nothing
    cancelled = should_cancel or _never_cancel

    # ---- 1. Which sessions -----------------------------------------------
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    result = PoseResult(base=cohort.base, model=model)

    candidates = [record for record in cohort.with_files()
                  if record.has_dlc and record.dlc_key(model) is not None]
    if only:
        candidates = [r for r in candidates if r.session_id in only]

    log(f"{cohort.base.name}: {len(candidates)} session(s) with '{model}' "
        f"({len(cohort)} in cohort)")

    # ---- 2. The port table, before anything is built -----------------------
    if ports == "approved":
        tables = approved_ports()
    else:
        tables = _supplied_tables(ports)
    if not tables:
        raise FileNotFoundError(
            "no calibrated port coordinates — run calibrate_ports.py first, "
            "or pass ports= explicitly")
    result.ports = tables
    log("  ports: " + ", ".join(f"rig {rig}" for rig in sorted(tables)
                                if rig != ANY_RIG))

    # ---- 3. Each session in turn -------------------------------------------
    total = len(candidates)
    for number, record in enumerate(candidates, start=1):
        if cancelled():
            log(f"Cancelled after {number - 1} of {total}.")
            break

        source, out = _model_paths(record, model)

        # A model folder holding no CSV (an .h5 alone) has nothing to build from.
        if source is None:
            result.results.append(SessionPose(
                session_id=record.session_id, model=model, out_dir=out,
                skipped=f"no DeepLabCut CSV for '{model}'"))
            continue

        # Before the table, because which rig a session ran on is recorded by
        # post-processing — an unprocessed session has no rig to look up, and saying
        # "no calibration" about it would send you to the wrong problem.
        if not record.is_processed:
            result.results.append(SessionPose(
                session_id=record.session_id, model=model, out_dir=out,
                skipped="not post-processed"))
            continue

        # The table that applies to this session — the catch-all, or its rig's.
        if ANY_RIG in tables:
            table = tables[ANY_RIG]
        else:
            try:
                table = tables.get(_rig_number(record))
            except (OSError, ValueError, KeyError, TypeError):
                # TypeError: session.json records the rig as null — a converted legacy
                # session whose rig was never logged. No rig, no table; reported per
                # session, not fatal.
                table = None
        if table is None:
            result.results.append(SessionPose(
                session_id=record.session_id, model=model, out_dir=out,
                error="no calibrated port coordinates for this session's rig — "
                      "run calibrate_ports.py on a cohort that includes it"))
            continue

        if not reprocess and _is_current(out, record, Path(source), table):
            result.results.append(SessionPose(
                session_id=record.session_id, model=model, out_dir=out,
                skipped="already current"))
            continue

        progress(f"Pose {number}/{total}: {record.session_id}")
        outcome = build_session_pose(record, model, ports=table)
        result.results.append(outcome)

        if outcome.ok:
            line = (f"  [{number}/{total}] {record.session_id}: "
                    f"{outcome.n_valid:,}/{outcome.n_frames:,} frames, "
                    f"{outcome.n_trials} trials")
            if outcome.warnings:
                line += f", {len(outcome.warnings)} warning(s)"
            log(line)
        elif outcome.error:
            log(f"  [{number}/{total}] {record.session_id}: FAILED — {outcome.error}")

    # ---- 4. The summary ------------------------------------------------------
    line = f"{cohort.base.name}: {len(result.ok)} built, {len(result.failed)} failed"
    if result.skipped:
        line += f", {len(result.skipped)} skipped"
    log(line)
    return result


# =====================================================================
# 4. One session
# =====================================================================

def _touches(record, frame_times, n_frames):
    """
    Every beam break in the session: ``(port touched, FrameLookup)``.

    ``frame_times`` is ``frame_times.csv`` as read; ``n_frames`` the video's frame count.
    """
    events = pd.read_csv(record.path("events"))
    touched = events[(events["event"] == "sensor_touch") & (events["index"] >= 0)]
    return (touched["index"].to_numpy(dtype=int),
            frames_at(frame_times, touched["t"].to_numpy(dtype=float), n_frames))


def build_session_pose(record, model, *, ports, out_dir=None):
    """
    Head position, bearing and port angles for one session and one model.

    Args:
        record: every path this session has, already resolved.
        model: which DeepLabCut model's output to read.
        ports: the calibrated coordinates for this session's rig, six (x, y) pairs.
            Required — without them there is no answer, and a session-derived
            substitute would be a different origin wearing the same name.
        out_dir: where to write, a Path or a string. Defaults to ``<session>/dlc/<model>``.

    Returns a :class:`SessionPose`. Failures are returned rather than raised, so a
    cohort run reports a bad session and carries on. Check :attr:`SessionPose.ok`.
    """
    source, folder = _model_paths(record, model)
    if out_dir is not None:
        out = Path(out_dir)
    else:
        out = folder or record.directory
    result = SessionPose(session_id=record.session_id, model=model, out_dir=out)

    if source is None:
        result.skipped = f"no output for model '{model}'"
        return result
    if not record.is_processed:
        result.skipped = "not post-processed"
        return result

    try:
        # ---- 1. Read -------------------------------------------------
        dlc = read_dlc(source)
        frame_times = pd.read_csv(record.path("frame_times"))
        trials = pd.read_csv(record.path("trials"))

        # ---- 2. The head, frame by frame -----------------------------
        bearing, method = head_bearing(dlc)
        position = head_position(dlc, bearing, method)
        times = (frame_times.set_index("frame")["t"]
                 .reindex(np.arange(dlc.n_frames)).to_numpy(dtype=float))
        motion = head_motion(position, bearing, times)

        # ---- 3. The moments that matter, as frames --------------------
        # Each event becomes a frame and a lag — how far that frame's own time is from
        # the event. See frame_lookup.py.
        #
        # A response is when the trial's chosen port's beam broke, NaN where there was
        # no response. A trials.csv without ``t_response`` gives NaN throughout rather
        # than a guess.
        if "t_response" in trials:
            response_times = pd.to_numeric(trials["t_response"],
                                           errors="coerce").to_numpy(dtype=float)
        else:
            response_times = np.full(len(trials), np.nan)
        lookups = {
            "trial_starts": frames_at(frame_times, trials["t_start"], dlc.n_frames),
            "responses": frames_at(frame_times, response_times, dlc.n_frames),
        }
        ports_touched, lookups["touches"] = _touches(record, frame_times, dlc.n_frames)

        # ---- 4. The ports --------------------------------------------
        table = np.asarray(ports, dtype=float)
        close = lookups["touches"].usable(MAX_LAG_S)
        counts, scatter = touch_scatter(
            ports_touched[close], lookups["touches"].frame[close], position, table)

        # ---- 5. The answer -------------------------------------------
        angles = angles_to_ports(position, bearing, table)

        # ---- 6. Write ------------------------------------------------
        out.mkdir(parents=True, exist_ok=True)
        result.written = [
            _write_pose(out, dlc, times, position, bearing, method, angles, motion),
            _write_trial_pose(out, trials, lookups, position, bearing,
                              method, angles, dlc, table, motion),
            _write_manifest(out, record, model, source, table, counts, scatter,
                            dlc, bearing, method, lookups),
        ]

        result.ports = table
        result.n_frames = dlc.n_frames
        result.n_valid = int(np.isfinite(bearing).sum())
        result.n_trials = len(trials)

        # ---- 7. Everything worth saying about a session that still succeeded --
        warnings = []

        if dlc.n_frames != len(frame_times):
            warnings.append(f"{dlc.n_frames:,} DLC rows but "
                            f"{len(frame_times):,} frames on the clock — the video lost "
                            "frames; see postprocessing/repairs/repair_video_timing.py")

        starts = lookups["trial_starts"]
        far = int((~starts.usable(MAX_LAG_S)).sum())
        if far > 0.01 * len(starts.frame):
            warnings.append(f"{far} of {len(starts.frame)} trial starts have no frame "
                            f"within {MAX_LAG_S:g} s")

        valid = np.isfinite(bearing).mean()
        if valid < 0.95:
            warnings.append(f"only {valid:.1%} of frames yielded a bearing")

        flipped = (method == "ears_flipped").mean()
        if flipped > 0.1:
            warnings.append(f"{flipped:.1%} of frames needed the spine flip")

        loose = [port for port in range(N_PORTS)
                 if np.isfinite(scatter[port]) and scatter[port] > MAX_PORT_SCATTER_PX]
        if loose:
            warnings.append(
                f"port(s) {loose} register touches a long way from the port itself "
                f"(up to {np.nanmax(scatter):.0f} px) — the sensor is probably tripping "
                "as the animal passes rather than pokes")

        result.warnings = warnings

    except Exception as exc:                      # noqa: BLE001 — reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"

    return result


# =====================================================================
# 5. Reading DeepLabCut
#
# The only place in this file that knows what a DLC file looks like.
# =====================================================================

def read_dlc(path):
    """
    Read a DeepLabCut CSV (a Path or a string) into the parts this stage needs, as a
    :class:`DLCData`.

    The file has three header rows — scorer, body part, coordinate — and the first two
    columns of the third are the frame index. Rows are frames, in order, one per frame
    of the video.
    """
    return dlc_from_table(read_dlc_table(path))


def read_dlc_table(path):
    """The whole DeepLabCut CSV — every body part — for callers that want more than this stage uses."""
    return pd.read_csv(path, header=[1, 2], index_col=0)


def dlc_from_table(frame):
    """The parts this stage needs, as a :class:`DLCData`, from a table already read by :func:`read_dlc_table`."""
    present = set(frame.columns.get_level_values(0))

    for part in ("left_ear", "right_ear"):
        if part not in present:
            raise ValueError(f"DLC output has no '{part}' — cannot derive a bearing")

    def xy(part):
        return frame[part][["x", "y"]].to_numpy(dtype=float)

    likelihood = np.minimum(
        frame["left_ear"]["likelihood"].to_numpy(dtype=float),
        frame["right_ear"]["likelihood"].to_numpy(dtype=float),
    )

    spine_parts = [part for part in SPINE_PARTS if part in present]
    if spine_parts:
        spine = np.stack([xy(part) for part in spine_parts], axis=1)
    else:
        spine = np.empty((len(frame), 0, 2))

    return DLCData(left_ear=xy("left_ear"), right_ear=xy("right_ear"),
                   ear_likelihood=likelihood, spine=spine)


@dataclass(frozen=True)
class DLCData:
    """The body parts this stage uses, as plain arrays, one row per frame."""

    left_ear: np.ndarray         # (n, 2)
    right_ear: np.ndarray        # (n, 2)
    ear_likelihood: np.ndarray   # (n,) — the lower of the two, the binding one
    spine: np.ndarray            # (n, k, 2), k == 0 when the model has no spine

    @property
    def n_frames(self):
        return len(self.left_ear)


# =====================================================================
# 6. The geometry
#
# Pure array maths. Nothing here opens a file or knows what a session is.
#
# Angles are in degrees on the image with y negated, so 0 is to the right and 90 is up
# the screen — the convention the previous library uses, kept so the two are comparable.
# =====================================================================

def head_bearing(dlc):
    """
    Which way the head faced on each frame, and how that was decided.

    The head points perpendicular to the line between the ears — 90 degrees clockwise
    of it — which fixes the axis but not the direction along it: the same two ears
    describe an animal facing forwards and one facing backwards. The spine breaks the
    tie. When the ear-derived bearing disagrees with the spine's direction by more than
    a right angle *and* most spine points lie behind the ear line, the bearing is
    flipped.

    Returns:
        ``(bearing_deg, method)`` — ``bearing_deg`` is NaN where no method applied;
        ``method`` is one of ``ears``, ``ears_flipped``, ``spine``, ``none``.

    One deliberate difference from the previous library: when the ears are unusable
    this takes the spine's direction vector directly, rather than least-squares fitting
    a line through the spine points. A fit needs special cases for vertical and
    degenerate spines, and this is the same quantity the flip check already uses.
    """
    ear = dlc.right_ear - dlc.left_ear
    usable = np.linalg.norm(ear, axis=1) >= MIN_EAR_DISTANCE_PX

    bearing = np.full(dlc.n_frames, np.nan)
    method = np.full(dlc.n_frames, "none", dtype="<U12")

    # The perpendicular, with y negated for image coordinates. Written out: the head
    # vector is (ear_y, -ear_x), and atan2(-head_y, head_x) is atan2(ear_x, ear_y).
    bearing[usable] = np.degrees(
        np.arctan2(ear[usable, 0], ear[usable, 1])) % 360
    method[usable] = "ears"

    if dlc.spine.shape[1] < 2:
        return bearing, method

    # Spine_1 is nearest the head, so spine_1 - spine_last points the way the animal is
    # facing — independently of the ears, which is what makes it a usable check.
    spine_vector = dlc.spine[:, 0, :] - dlc.spine[:, -1, :]
    spine_bearing = np.degrees(
        np.arctan2(-spine_vector[:, 1], spine_vector[:, 0])) % 360

    difference = np.abs(bearing - spine_bearing)
    difference = np.minimum(difference, 360 - difference)

    # Which side of the ear line each spine point falls on. A body behind the ears is
    # the normal case; a body in front of them means the ear labels are swapped.
    to_spine = dlc.spine - dlc.left_ear[:, None, :]
    cross = (ear[:, None, 0] * to_spine[:, :, 1]
             - ear[:, None, 1] * to_spine[:, :, 0])
    votes = (cross < 0).sum(axis=1)

    flip = usable & (difference > 90) & (votes > dlc.spine.shape[1] / 2)
    bearing[flip] = (bearing[flip] + 180) % 360
    method[flip] = "ears_flipped"

    # Ears unusable but a spine to fall back on.
    fallback = ~usable & np.isfinite(spine_bearing)
    bearing[fallback] = spine_bearing[fallback]
    method[fallback] = "spine"

    return bearing, method


def head_position(dlc, bearing, method):
    """
    The point every angle is measured from: a body reference, pushed forward.

    Normally that reference is the ear midpoint. On frames where the ears were too
    close to give a bearing it is ``spine_1`` instead — having just rejected the ears
    as unreliable, deriving a position from them anyway would be inconsistent. The
    previous library makes the same choice, and it matters for the small percentage of
    frames that take the spine path.

    y is subtracted rather than added because the image's y axis runs downwards while
    the bearing is expressed in ordinary axes.
    """
    origin = (dlc.left_ear + dlc.right_ear) / 2
    if dlc.spine.shape[1]:
        from_spine = method == "spine"
        origin[from_spine] = dlc.spine[from_spine, 0, :]

    radians = np.radians(bearing)
    return origin + np.column_stack([EYES_OFFSET_PX * np.cos(radians),
                                     -EYES_OFFSET_PX * np.sin(radians)])


def head_motion(position, bearing, times):
    """
    How fast the head moved and turned arriving at each frame, from the frame before.

    Returns ``(speed_px_s, turn_rate_deg_s)``, one value per frame each.

    Rates, not steps: a frame the camera dropped doubles the time between two video
    frames, and a step would read that as a jump. ``turn_rate_deg_s`` is signed — positive
    is anticlockwise in the image, the way the bearing grows — and takes the short way
    round, so 350 to 10 degrees is +20, not -340.

    The first frame, and any frame next to one with no time or no bearing, has no value:
    there is no step to measure, and a blank fails any limit set on it.
    """
    times = np.asarray(times, dtype=float)
    dt = np.diff(times)
    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.hypot(*np.diff(position, axis=0).T)
        turn = (np.diff(bearing) + 180.0) % 360.0 - 180.0
        timed = dt > 0
        speed = np.full(len(times), np.nan)
        rate = np.full(len(times), np.nan)
        speed[1:][timed] = step[timed] / dt[timed]
        rate[1:][timed] = turn[timed] / dt[timed]
    return speed, rate


def touch_scatter(ports_touched, frames, position, ports):
    """
    How far from each port the head actually was when that port's beam broke.

    Measured against the **calibrated** coordinates, so nothing here estimates a
    position — it only asks whether the touches agree with where the port is known to
    be. A clean port gives around 10-20 px. A sensor that trips as the animal passes
    rather than pokes gives well over 100, which is worth knowing before trusting that
    session's chosen ports or response times.

    Returns:
        ``(counts, scatter_px)``, both per port, NaN where a port was never touched.
    """
    counts = np.zeros(N_PORTS, dtype=int)
    scatter = np.full(N_PORTS, np.nan)

    for port in range(N_PORTS):
        xy = position[frames[ports_touched == port]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        counts[port] = len(xy)
        scatter[port] = float(np.median(np.linalg.norm(xy - ports[port], axis=1)))

    return counts, scatter


def angles_to_ports(position, bearing, ports):
    """
    Where each port sat relative to where the animal was looking.

    Returns an ``(n_frames, N_PORTS)`` array in degrees on [-180, 180): **0 means the
    port was straight ahead**, and the sign says which side it was on.
    """
    ports = np.asarray(ports, dtype=float)
    dx = ports[None, :, 0] - position[:, None, 0]
    dy = ports[None, :, 1] - position[:, None, 1]
    from_head = np.degrees(np.arctan2(-dy, dx))
    return (from_head - bearing[:, None] + 180) % 360 - 180


def port_geometry(coords):
    """
    The checks that say whether a set of six coordinates is believable, as a dict.

    A real hexagon puts its ports at one radius and 60 degrees apart, so the scatter in
    those two is a direct measure of how much to trust the set — without needing any
    reference table to compare against.
    """
    if not np.isfinite(coords).all():
        return {"radius_px": None, "radius_sd_px": None, "spacing_sd_deg": None}

    offset = coords - coords.mean(axis=0)
    radii = np.linalg.norm(offset, axis=1)
    angles = np.sort(np.degrees(np.arctan2(-offset[:, 1], offset[:, 0])) % 360)
    spacing = np.diff(np.concatenate([angles, angles[:1] + 360]))

    return {"radius_px": round(float(radii.mean()), 1),
            "radius_sd_px": round(float(radii.std()), 1),
            "spacing_sd_deg": round(float(spacing.std()), 2)}


# =====================================================================
# 7. The three files
#
# Thin enough to live beside the session that produces them, so this file says what a
# session produces without sending you anywhere else. Each returns the path it wrote.
# =====================================================================

def _write_pose(out, dlc, times, position, bearing, method, angles, motion):
    """One row per frame — the complete record, for trajectories and re-derivation."""
    speed, rate = motion
    table = pd.DataFrame({
        "frame": np.arange(dlc.n_frames),
        "t": times.round(6),
        "head_x": position[:, 0].round(1),
        "head_y": position[:, 1].round(1),
        "bearing_deg": bearing.round(2),
        "ear_likelihood": dlc.ear_likelihood.round(3),
        "head_speed_px_s": speed.round(1),
        "turn_rate_deg_s": rate.round(1),
        "method": method,
    })
    for port in range(N_PORTS):
        table[f"angle_port{port}"] = angles[:, port].round(2)

    path = out / POSE_CSV
    table.to_csv(path, index=False)
    return path


def _named_port(column):
    """
    A port column (a Series) as ``(port indices, which rows actually name a port)``.

    The controller writes ``-1`` for "no port" — a timed-out trial chose nothing. Left
    as a number that would index an array, it silently selects the *last* port and
    produces an angle that looks entirely plausible, which is the failure mode this
    exists to prevent. Anything outside 0..N_PORTS-1 is absent.
    """
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    known = np.isfinite(values) & (values >= 0) & (values < N_PORTS)
    ports = np.zeros(len(values), dtype=int)
    ports[known] = values[known].astype(int)
    return ports, known


def moment_motion(values, frames):
    """
    A motion value at looked-up frames: the larger in size of the steps into and out of it.

    Both sides, because a single mislabelled frame shows as a jump on the way in *and* on
    the way out, and the frame read for an event is exactly the one whose own position
    matters. Blank where there is no frame (-1) or neither step has a value.
    """
    values = np.abs(np.asarray(values, dtype=float))
    frames = np.asarray(frames)
    out = np.full(len(frames), np.nan)
    found = frames >= 0
    into = values[frames[found]]
    after = np.minimum(frames[found] + 1, len(values) - 1)
    out_of = np.where(frames[found] + 1 < len(values), values[after], np.nan)
    with np.errstate(invalid="ignore"):
        out[found] = np.fmax(into, out_of)
    return out


def _write_trial_pose(out, trials, lookups, position, bearing, method,
                      angles, dlc, ports, motion):
    """
    One row per trial — where the animal was looking when the cue appeared.

    Keyed on ``trial`` so it left-joins straight onto the post-processed ``trials.csv``,
    which keeps that file pure behaviour and independent of any model.

    Both angles are read at the frame of the **trial marker** going high, which is what
    ``t_start`` is. That edge exists for every trial regardless of how the trial was
    cued — not every trial starts with an LED — so it is the one anchor that means the
    same thing throughout a session. Anything measured relative to a cue instead belongs
    in the plot that knows which cue it wants.

    ``touched_angle_deg`` is therefore where the port they went on to touch sat at the
    start of the trial, not when they touched it.

    The ``response_*`` columns are the exception, read at the frame where that port's
    beam actually broke. ``response_distance_px`` is the check on whether the touch was
    a poke at all.

    Both frames come with a lag — ``frame_lag_s`` and ``response_frame_lag_s``, that
    frame's own time minus the event's. Every value is still written when the lag is
    large; filtering on it is left to the analysis, so the threshold stays a choice
    rather than something baked in here.

    The same goes for the trajectory checks: ``head_speed_px_s`` and ``turn_rate_deg_s``
    at the trial start frame, and ``response_head_speed_px_s`` and
    ``response_turn_rate_deg_s`` at the response frame, each the larger in size of the
    steps into and out of that frame (:func:`moment_motion`). They judge the moment, not
    the trial: a mouse up the wall halfway through spoils neither end.
    """
    speed, rate = motion
    anchor = trials["t_start"].to_numpy(dtype=float)
    starts = lookups["trial_starts"].frame
    if (starts < 0).any():
        raise ValueError("some trial starts have no video frame at all")

    def angle_at(column):
        """The angle, at each trial's start, of the port a trials.csv column names."""
        port, known = _named_port(trials[column])
        out_ = np.full(len(trials), np.nan)
        out_[known] = angles[starts[known], port[known]]
        return out_

    chosen, known = _named_port(trials["chosen_port"])
    distance = np.full(len(trials), np.nan)
    distance[known] = np.linalg.norm(
        ports[chosen[known]] - position[starts[known]], axis=1)

    response = lookups["responses"].frame
    responded = response >= 0
    # How far the head was from the port the animal is recorded as choosing, at the
    # moment that port's beam broke. A real poke is a few tens of pixels. Hundreds means
    # the head was elsewhere — a tail in the port, or a bad DeepLabCut frame, which
    # ``response_ear_likelihood`` tells apart.
    response_distance = np.full(len(trials), np.nan)
    usable = responded & known
    response_distance[usable] = np.linalg.norm(
        ports[chosen[usable]] - position[response[usable]], axis=1)

    response_likelihood = np.full(len(trials), np.nan)
    response_likelihood[responded] = dlc.ear_likelihood[response[responded]]

    table = pd.DataFrame({
        "trial": trials["trial"].to_numpy(),
        "frame": starts,
        "t": anchor.round(6),
        "frame_lag_s": lookups["trial_starts"].lag_s.round(4),
        "head_x": position[starts, 0].round(1),
        "head_y": position[starts, 1].round(1),
        "bearing_deg": bearing[starts].round(2),
        "ear_likelihood": dlc.ear_likelihood[starts].round(3),
        "head_speed_px_s": moment_motion(speed, starts).round(1),
        "turn_rate_deg_s": moment_motion(rate, starts).round(1),
        "method": method[starts],
        "cue_angle_deg": angle_at("correct_port").round(2),
        "touched_angle_deg": angle_at("chosen_port").round(2),
        "touched_distance_px": distance.round(1),
        "response_frame": np.where(responded, response, np.nan),
        "response_frame_lag_s": lookups["responses"].lag_s.round(4),
        "response_distance_px": response_distance.round(1),
        "response_ear_likelihood": response_likelihood.round(3),
        "response_head_speed_px_s": moment_motion(speed, response).round(1),
        "response_turn_rate_deg_s": moment_motion(rate, response).round(1),
    })

    path = out / TRIAL_POSE_CSV
    table.to_csv(path, index=False)
    return path


def _lag_summary(lookup):
    """How close one kind of lookup (a FrameLookup) got to its events, for pose.json."""
    found = lookup.frame >= 0
    lags = lookup.lag_s[found]
    median_lag = None
    if found.any():
        median_lag = round(float(np.median(lags)), 4)
    return {
        "n": int(found.sum()),
        "median_lag_s": median_lag,
        "n_over_max_lag": int((np.abs(lags) > MAX_LAG_S).sum()),
        "n_after_video_end": int((lags < 0).sum()),
    }


def _write_manifest(out, record, model, source, table, counts, scatter,
                    dlc, bearing, method, lookups):
    """
    What this was built from — the file that makes the other two trustworthy months on.

    It holds the port coordinates used, so a folder never has to look anything up, and
    enough about the source for the next run to tell whether anything has changed.

    The only port numbers here are the calibrated ones. There is deliberately no
    session-derived alternative sitting beside them to be picked up by mistake.
    """
    used, counted = np.unique(method, return_counts=True)

    # Whether frame_times.csv had been repaired for the old converter's frame loss
    # when this was built. None: that repair never looked at this session.
    repair_file = record.processed.get("video_repair")
    repair = None
    if repair_file:
        try:
            repair = json.loads(Path(repair_file).read_text())["status"]
        except (OSError, ValueError, KeyError):
            repair = "unreadable"

    # How far the head was from each port when its beam broke — a check on the
    # sensors, not an estimate of anything.
    scatter_px = []
    for value in scatter:
        if np.isfinite(value):
            scatter_px.append(round(float(value), 1))
        else:
            scatter_px.append(None)
    port_scatter = None
    if np.isfinite(scatter).any():
        port_scatter = round(float(np.nanmedian(scatter)), 1)

    # How far each looked-up frame was from its event. A session with many over the
    # limit has gaps in its video right where the analysis wants to look.
    lag_summaries = {"max_lag_s": MAX_LAG_S}
    for name, lookup in lookups.items():
        lag_summaries[name] = _lag_summary(lookup)

    source_record = {"csv": Path(source).name, **_fingerprint(source)}
    by_method = {name: int(count) for name, count in zip(used, counted)}

    manifest = {
        "session_id": record.session_id,
        "model": model,
        "source": source_record,
        "inputs": _inputs(record),
        "video_repair": repair,
        "params": PARAMS,
        "ports": {
            "coords": np.round(table, 1).tolist(),
            "geometry": port_geometry(table),
            "n_touches": counts.tolist(),
            "scatter_px": scatter_px,
            "port_scatter_px": port_scatter,
        },
        "trials": {"anchored_on": "t_start"},
        "lookups": lag_summaries,
        "frames": {
            "n": dlc.n_frames,
            "n_valid": int(np.isfinite(bearing).sum()),
            "by_method": by_method,
        },
    }

    path = out / POSE_JSON
    path.write_text(_readable_json(manifest))
    return path


def _readable_json(payload):
    """
    A dict as pretty-printed JSON text, but with short lists of numbers kept on one line.

    ``json.dumps(indent=2)`` gives every element of every array its own line, which
    spreads one coordinate pair over four and makes a port table genuinely hard to
    read — the opposite of the point of writing JSON rather than something binary.
    Innermost numeric lists collapse; the list of pairs stays one pair per line.
    """
    collapsible = r"\[\s+((?:(?:-?[\d.eE+]+|null),?\s+)+)\]"
    return re.sub(collapsible,
                  lambda match: "[" + " ".join(match.group(1).split()) + "]",
                  json.dumps(payload, indent=2))


# =====================================================================
# 8. What comes back
# =====================================================================

@dataclass
class SessionPose:
    """What one session produced, and anything noticed while producing it."""

    session_id: str
    model: str
    out_dir: Path
    written: list[Path] = field(default_factory=list)
    n_frames: int = 0
    n_valid: int = 0
    n_trials: int = 0
    ports: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    skipped: str | None = None

    @property
    def ok(self):
        return self.error is None and self.skipped is None


@dataclass
class PoseResult:
    """What one sweep produced, per session."""

    base: Path
    model: str
    results: list[SessionPose] = field(default_factory=list)
    ports: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def ok(self):
        return [r for r in self.results if r.ok]

    @property
    def failed(self):
        return [r for r in self.results if r.error]

    @property
    def skipped(self):
        return [r for r in self.results if r.skipped]

    def describe(self):
        heading = (f"{self.base.name} · {self.model} — {len(self.ok)} built, "
                   f"{len(self.failed)} failed")
        if self.skipped:
            heading += f", {len(self.skipped)} skipped"
        lines = [heading, ""]

        for rig, coords in sorted(self.ports.items()):
            if rig == ANY_RIG:
                name = "all rigs"
            else:
                name = f"rig {rig}"
            shape = port_geometry(coords)
            lines.append(f"  {name}: radius {shape['radius_px']} px "
                         f"(sd {shape['radius_sd_px']}), "
                         f"spacing sd {shape['spacing_sd_deg']} deg")

        if self.ok:
            valid = np.mean([r.n_valid / r.n_frames for r in self.ok if r.n_frames])
            lines.append(f"  frames with a bearing: {valid:.1%} on average")

        flagged = []                               # (session ID, warning)
        for item in self.ok:
            for warning in item.warnings:
                flagged.append((item.session_id, warning))
        if flagged:
            lines += ["", "  WARNINGS"]
            lines += [f"    {sid}  {w}" for sid, w in flagged[:20]]
            if len(flagged) > 20:
                lines.append(f"    ... and {len(flagged) - 20} more")

        if self.skipped:
            reasons = {}                           # reason -> how many sessions
            for item in self.skipped:
                reasons[item.skipped] = reasons.get(item.skipped, 0) + 1
            lines += ["", "  SKIPPED"]
            lines += [f"    {count} session(s): {why}"
                      for why, count in sorted(reasons.items())]

        if self.failed:
            lines += ["", "  FAILED"]
            lines += [f"    {r.session_id}  {r.error}" for r in self.failed]

        return "\n".join(lines)
