"""
Repair the video timing of sessions compressed by the old bin→avi converter.
===========================================================================

**Run after post-processing, on sessions exactly as it left them**::

    python repair_video_timing.py <cohort> <model task>

Safe to re-run: a session already decided is skipped, and one that post-processing has
rewritten since is decided again. Run it after every post-processing run. ``--force``
decides again from post-processing's original, and puts that original back wherever a
repair is no longer accepted.

What went wrong
---------------
The converter split a session's frames into chunks, one per worker. Each worker seeked
the raw ``.bin`` to its chunk's first *camera frame ID* as if that were a *position in
the file*, then read the chunk's length in order (fixed in commit 0aa1175). The two only
differ once the camera has dropped a frame, and from then on:

* every chunk starts as many frames too far into the file as the camera had dropped
  before it, so each image in it shows a moment later than the time it is given;
* the frames between one chunk's read and the next are never written;
* the last chunk runs off the end of the file — the only place the damage shows up, as a
  video shorter than the tracker's list.

Post-processing pairs video frame *i* with the *i*-th frame the tracker logged, so on
these sessions every image after the first affected chunk carries a time that is too
early: by seconds early on, and by minutes late in a bad session.

What the repair does
--------------------
The chunking is deterministic, so replaying it on the tracker's frame IDs says exactly
which position in the ``.bin`` every video frame was read from — and that position's
time is the frame's real time. ``frame_times.csv`` is rewritten with that pairing: the
same columns, one row per video frame, the time jumping forward over the frames that were
never written. **No timestamp is changed, only which video frame it is paired with.**

The video and DeepLabCut output are untouched and nothing needs re-running. Pose notices
the new ``frame_times.csv`` on its own and rebuilds; the gaps show up there as lags on
the lookups that land in them (see ``pose/frame_lookup.py``).

A repair is applied only when **both**

1. the replay reproduces the video's frame count exactly, or does once the chunks that
   were never written are found (below) — otherwise ``unexplained``;
2. the check below does not show the repair making the timing significantly worse —
   otherwise ``rejected``, or ``unchecked`` when there is nothing to check it with.

**Chunks never written.** Some videos are shorter than the replay by whole chunks —
exactly *k* of the 40 — with or without camera drops: a worker failed and its chunk was
never written, so every frame after it is paired with a time a chunk too early. Which
chunks are missing is found from the touches (the check below, as a search): every
choice of *k* chunks is scored, in one pass over the chunks in order, by touches with
the head at the port less touches with it away, and the best taken. Because the search
and the check use the same touches, this is only accepted when the search run on
alternate halves of the touches picks the same chunks both times, **and** the result
passes the check outright; otherwise the session stays ``unexplained``. It needs pose,
so it is decided on the run after DeepLabCut, and it needs frame times, so a session
with no camera pulses cannot be searched here.

The second condition is not a formality. On one 2026 cohort, four early training
sessions with a handful of camera drops match the replay's length exactly, yet the repair
turns hundreds of touches with the head at the port into touches with it elsewhere: they
lost those frames some other way, without the shift. Matching a length is necessary, and
not enough.

What it writes, and what it never touches
------------------------------------------
::

    <session>/processed/
    ├── frame_times.csv              rewritten with the repaired pairing, if accepted
    ├── frame_times_unrepaired.csv   post-processing's original, copied before that
    └── video_repair.json            what was decided and why, and the check

    <cohort>/video_repair/
    ├── <session>.png                the check, drawn, for each session it ran on
    ├── summary.csv                  every session this has a decision about
    └── summary.png

Nothing else is written. The rig's raw files, the video, DeepLabCut's output and the
previous library's files — the ``.nwb``, ``_video_frame_times.json``, the ``.pkl`` — are
only ever read, and keep the old timing.

The check
---------
Each candidate repair is checked the way the problem was found: when a port's beam
breaks, the head should be at that port. Every touch is classed near (head within 100 px
of the port) or far, under post-processing's timing and under the repaired timing, and
the touches whose class differs are counted both ways: **fixed** (far → near) and
**broken** (near → far).

* **Significantly more broken than fixed** (one-sided binomial test, p < 0.001) —
  ``rejected``; the file is left as post-processing wrote it.
* **Otherwise** — ``repaired``. It ``passed`` if at most 10% of touches are then far;
  undamaged sessions on that cohort sit at 2–4%. A repair still over that line is
  kept, being the better timing; the failure says something else is wrong too.

Why counts rather than comparing the two far percentages: a session with a few camera
drops moves its frames by a few positions, the percentages differ by noise, and a rule of
"no higher after" rejects good repairs at random. Why this direction is trustworthy: right
after a beam break the head is still in the port, so a correct repair almost never turns a
near touch far, while a wrong one — picking frames from before the touch, as the animal
approaches — does so often. The threshold is strict because a cohort tests every damaged
session at once; the four sessions it rejects on that cohort break 40–312 touches
against 1–102 fixed.

The picture shows the distance at every touch through the session under both timings, and
a few video frames with the frame each timing picks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

# Runnable as a file as well as a module, like the scripts in pose/.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hba2.cohort import Cohort  # noqa: E402
from hba2.pose.pose import (  # noqa: E402
    _readable_json, _rig_number, approved_ports, dlc_from_table, head_bearing,
    head_position, read_dlc_table,
)
from hba2.pose.frame_lookup import (  # noqa: E402
    MAX_LAG_S, frames_at)

# =====================================================================
# The numbers
# =====================================================================

#: Cores on the machine that compressed the damaged videos. The old converter made
#: ``max(n_frames // cores, 500)``-frame chunks, so this sets where every chunk began.
#: Established from the data on that cohort:
#: 40 is the only count that reproduces the length of every damaged video whose length
#: pins it down, and gives the fewest touches with the head away from the port pooled
#: over 48 sessions. A cohort compressed on another machine may need ``--cores``.
COMPRESSION_CORES = 40

#: The old converter's floor on chunk size.
MIN_CHUNK = 500

FRAME_TIMES = "frame_times.csv"
UNREPAIRED_CSV = "frame_times_unrepaired.csv"
REPAIR_JSON = "video_repair.json"
IMAGE_DIR = "video_repair"

#: A touch with the head further than this from the port was not a poke on this frame.
FAR_PX = 100.0
#: The check passes when no more than this share of touches are that far away.
MAX_FAR_FRACTION = 0.10
#: A repair is rejected when it breaks significantly more touches than it fixes, at this
#: one-sided p. Strict, because a cohort tests every damaged session at once.
REJECT_P = 0.001
#: Video frames drawn per session in the check picture.
N_EXAMPLES = 4

BY = "hba2/postprocessing/repairs/repair_video_timing.py"
BEFORE, AFTER, TOUCHED = "#eb6834", "#2a78d6", "#1baf7a"


# =====================================================================
# 1. A whole cohort
# =====================================================================

def _print_flushed(message):
    print(message, flush=True)


def _log_one(log, number, total, outcome):
    """One line of progress for a session that was decided, or failed."""
    prefix = f"  [{number}/{total}] {outcome.session_id}:"
    if outcome.error:
        log(f"{prefix} ERROR — {outcome.error}")
    elif outcome.status == "repaired":
        checked = ""
        if outcome.passed is not None:
            verdict = "passed" if outcome.passed else "FAILED"
            checked = (f"; head far from port {outcome.far_before:.1%} → "
                       f"{outcome.far_after:.1%} ({verdict})")
        log(f"{prefix} repaired, timing was off by up to "
            f"{outcome.max_shift_frames:,} frames{checked}")
    elif outcome.status in ("rejected", "unchecked", "unexplained"):
        log(f"{prefix} {outcome.status} — {outcome.reason}")


def repair_video_timing(cohort, model, *, cores=COMPRESSION_CORES, only=None, force=False,
                        dry_run=False, workers=1, ports=None, log_fn=None):
    """
    Repair every session in a cohort whose video the old converter damaged.

    Args:
        cohort: a :class:`Cohort`, or its folder as a Path or a string.
        model: DeepLabCut model (its Task) to check each repair with.
        cores: the compressing machine's core count — see :data:`COMPRESSION_CORES`.
        only: just these session IDs, a tuple.
        force: decide again from post-processing's originals.
        dry_run: decide and report, writing nothing.
        workers: sessions at once. Each is independent; checking reads a DeepLabCut file
            and a few video frames, so a cohort serially is the better part of an hour.
        ports: a port table (``port_coordinates.json``) to judge by instead of the
            package's, for a cohort whose rigs it does not describe; a Path or a string.
        log_fn: called with each line of progress; prints by default.
    """
    log = log_fn or _print_flushed
    cohort = cohort if isinstance(cohort, Cohort) else Cohort(cohort)
    records = [r for r in cohort if not only or r.session_id in only]
    result = RepairResult(base=cohort.base)
    image_dir = None if dry_run else cohort.base / IMAGE_DIR

    heading = (f"{cohort.base.name}: {len(records)} session(s), {cores}-core chunking, "
               f"checked with '{model}'")
    if dry_run:
        heading += " — DRY RUN, nothing written"
    log(heading)

    job = partial(repair_session, model=model, cores=cores, image_dir=image_dir,
                  force=force, dry_run=dry_run, ports=ports)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            outcomes = pool.map(job, records)
            for number, outcome in enumerate(outcomes, start=1):
                _log_one(log, number, len(records), outcome)
                result.results.append(outcome)
    else:
        for number, record in enumerate(records, start=1):
            outcome = job(record)
            _log_one(log, number, len(records), outcome)
            result.results.append(outcome)

    if not dry_run:
        log(f"wrote {write_summary(Cohort(cohort.base))}")
    log("\n" + result.describe())
    return result


# =====================================================================
# 2. One session
# =====================================================================

def _read_json(path):
    """A JSON file's contents, or an empty dict if it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _from_postprocessing(frame_times, frame_ids):
    """True when this is post-processing's pairing: one row per tracker frame, in order."""
    return (len(frame_times) == len(frame_ids)
            and np.array_equal(frame_times["frame_id"].to_numpy(dtype=int), frame_ids))


def _unexplained(result, manifest, reason, out, dry_run):
    """Record that frames were lost in a way the replay does not account for."""
    manifest.update(status="unexplained", reason=reason)
    if not dry_run:
        (out / REPAIR_JSON).write_text(_readable_json(manifest))
    result.status, result.reason = "unexplained", reason
    return result


def _lost_chunks(record, model, ports, frame_ids, cores, original, n_video):
    """
    The chunks never written, by :func:`find_lost_chunks` — ``(indices, "")`` — or
    ``(None, why not)``. Run on all the touches and on each alternate half; the halves
    must agree with each other and with the whole.
    """
    seen = head_and_touches(record, model, ports)
    if "why" in seen:
        return None, f"cannot look for chunks never written: {seen['why']}"
    t = original["t"].to_numpy(dtype=float)
    if not np.isfinite(t).any():
        return None, "cannot look for chunks never written: no frame is timed"
    chunks = chunk_reads(frame_ids, cores)
    picks = [find_lost_chunks(chunks, t, n_video, seen["position"], seen["touch_t"][half],
                              seen["touch_port"][half], seen["ports"])
             for half in (slice(None), slice(0, None, 2), slice(1, None, 2))]
    if picks[0] is None:
        return None, "no set of whole chunks accounts for the missing frames"
    if not (picks[0] == picks[1] == picks[2]):
        return None, (f"chunks never written not pinned down: all touches give {picks[0]}, "
                      f"alternate halves {picks[1]} and {picks[2]}")
    return picks[0], ""


def repair_session(record, model, *, cores=COMPRESSION_CORES, image_dir=None,
                   force=False, dry_run=False, ports=None):
    """
    Repair one session's ``frame_times.csv`` if the old converter damaged its video.

    Args:
        model: the DeepLabCut model (its Task) to check the repair with. Required: a
            repair is only applied once the check says it helps.
        cores: the compressing machine's core count — see :data:`COMPRESSION_CORES`.
        image_dir: where the check picture goes, a Path; None draws none.
        force: decide again from post-processing's original, even if decided before.
        dry_run: decide and report — the check included — but write nothing.
        ports: a port table to judge by instead of the package's, a Path or a string.
    """
    result = SessionRepair(session_id=record.session_id)
    if not record.is_processed:
        result.skipped = "not post-processed"
        return result
    for role in ("video", "tracker"):
        if record.raw.get(role) is None:
            result.skipped = f"no {role} file"
            return result

    out = record.out_dir
    try:
        # ---- 1. READ -------------------------------------------------
        frame_ids = np.asarray(
            json.loads(record.path("tracker").read_text())["frame_IDs"], dtype=int)
        current = pd.read_csv(out / FRAME_TIMES)
        previous = _read_json(out / REPAIR_JSON)

        # ---- 2. WHOSE frame_times.csv IS THIS ------------------------
        # Post-processing writes one row per tracker frame, in the tracker's order.
        # Anything else is this repair's output, and repairing that would shift it twice.
        if _from_postprocessing(current, frame_ids):
            original = current
            # A decision to leave this file alone stands until forced. A recorded repair
            # with post-processing's file in place means post-processing has run again
            # since, so that one is decided afresh; "unchecked" is retried every run in
            # case the model's output has arrived.
            if previous.get("status") in ("rejected", "unexplained") and not force:
                result.status = "already_decided"
                return result
        elif (out / UNREPAIRED_CSV).exists():
            # This repair's output from an earlier run — or a restore that was cut off.
            if previous.get("status") == "repaired" and not force:
                result.status = "already_decided"
                return result
            original = pd.read_csv(out / UNREPAIRED_CSV)
            if not _from_postprocessing(original, frame_ids):
                raise ValueError(f"{UNREPAIRED_CSV} does not match the tracker's frame IDs")
        else:
            raise ValueError(f"{FRAME_TIMES} matches neither the tracker's frame IDs nor "
                             "a recorded repair — left alone")

        manifest = {
            "session_id": record.session_id,
            "status": None,
            "reason": "",
            "decided_at": datetime.now().isoformat(timespec="seconds"),
            "by": BY,
            "tracker_frames": len(frame_ids),
            "camera_drops": int(frame_ids[-1] - frame_ids[0] + 1 - len(frame_ids)),
        }

        # ---- 3. HOW MANY FRAMES THE VIDEO HOLDS ----------------------
        n_video, source = count_video_frames(record)
        if n_video is None:
            return _unexplained(result, manifest, source, out, dry_run)
        manifest.update(video_frames=n_video, video_frames_from=source)
        if n_video == len(frame_ids):
            result.status = "not_needed"
            return result

        # ---- 4. REPLAY THE OLD CONVERTER -----------------------------
        positions = replay_chunks(frame_ids, cores)
        size = max(len(frame_ids) // cores, MIN_CHUNK)
        manifest["replay"] = {"cores": cores, "chunk_size": size,
                              "n_chunks": -(-len(frame_ids) // size),
                              "frames_written": len(positions)}
        lost = None
        if len(positions) != n_video:
            mismatch = (f"replaying the {cores}-core chunking writes {len(positions):,} "
                        f"frames but the video holds {n_video:,}")
            if len(positions) < n_video:
                return _unexplained(result, manifest, mismatch, out, dry_run)
            lost, why = _lost_chunks(record, model, ports, frame_ids, cores, original, n_video)
            if lost is None and why.startswith("cannot look"):
                # Nothing to search with yet (no pose, or no frame times): retried every run.
                manifest.update(status="unchecked", reason=f"{mismatch}; {why}")
                if not dry_run:
                    (out / REPAIR_JSON).write_text(_readable_json(manifest))
                result.status, result.reason = "unchecked", manifest["reason"]
                return result
            if lost is None:
                return _unexplained(result, manifest, f"{mismatch}; {why}", out, dry_run)
            chunks = chunk_reads(frame_ids, cores)
            positions = np.concatenate([c for i, c in enumerate(chunks) if i not in lost])
            manifest["lost_chunks"] = {"chunks": lost, "frames": int(sum(len(chunks[i]) for i in lost)),
                                       "why": mismatch}

        # ---- 5. THE REPAIRED PAIRING ---------------------------------
        # Video frame i came from .bin position positions[i]: that position's frame ID
        # and time are its own. Nothing is interpolated, nothing is invented.
        t_by_position = original["t"].to_numpy(dtype=float)
        repaired = pd.DataFrame({"frame": np.arange(n_video),
                                 "frame_id": frame_ids[positions],
                                 "t": t_by_position[positions]})
        shift = positions - np.arange(n_video)
        result.max_shift_frames = int(shift.max())
        manifest.update(
            frames_not_written=len(frame_ids) - n_video,
            max_shift_frames=result.max_shift_frames,
            skipped=skipped_runs(positions, len(frame_ids)),
        )

        # ---- 6. CHECK IT, AND DECIDE ---------------------------------
        check = check_repair(record, model, original, repaired, n_video, manifest,
                             None if dry_run else image_dir, ports=ports)
        manifest["check"] = check
        result.far_before = check.get("far_fraction_before")
        result.far_after = check.get("far_fraction_after")
        result.passed = check.get("passed")

        if lost is not None and not (check["ran"] and check["passed"]):
            status, reason = "unexplained", (
                f"chunks {lost} look never written, but the result does not pass the check "
                f"(head far from port {result.far_after if result.far_after is not None else float('nan'):.1%})")
        elif not check["ran"]:
            status, reason = "unchecked", f"replay matches, but nothing to check it with: {check['why']}"
        elif check["p_worse"] < REJECT_P:
            status, reason = "rejected", (
                f"replay matches the video's {n_video:,} frames, but the repair puts the head "
                f"away from the port on {check['touches_broken']} touches and back on it on "
                f"only {check['touches_fixed']} — not damaged by the chunking bug")
        elif lost is not None:
            status, reason = "repaired", (
                f"chunks {lost} were never written; with them left out the replay "
                f"reproduces the video's {n_video:,} frames; head far from port "
                f"{result.far_before:.1%} → {result.far_after:.1%} "
                f"({check['touches_fixed']} touches fixed, {check['touches_broken']} broken)")
        else:
            status, reason = "repaired", (
                f"replay reproduces the video's {n_video:,} frames exactly; head far from "
                f"port {result.far_before:.1%} → {result.far_after:.1%} "
                f"({check['touches_fixed']} touches fixed, {check['touches_broken']} broken)")
        manifest.update(status=status, reason=reason)
        result.status, result.reason = status, reason

        # ---- 7. WRITE ------------------------------------------------
        # Original first, then the decision, then the file itself: a run cut off at any
        # point leaves post-processing's file, its copy, or both, each of which the next
        # run recognises.
        if not dry_run:
            if status == "repaired" and original is current:
                shutil.copy2(out / FRAME_TIMES, out / UNREPAIRED_CSV)
            (out / REPAIR_JSON).write_text(_readable_json(manifest))
            if status == "repaired":
                # Same format as post-processing writes, swapped in whole so it is never
                # half-written.
                temporary = (out / FRAME_TIMES).with_suffix(".csv.tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.write("frame,frame_id,t\n")
                    for frame, frame_id, t in repaired.itertuples(index=False):
                        text = "" if np.isnan(t) else f"{t:.6f}"
                        handle.write(f"{frame},{frame_id},{text}\n")
                os.replace(temporary, out / FRAME_TIMES)
            elif original is current:
                # Post-processing's file is in place and stays: a leftover copy goes.
                (out / UNREPAIRED_CSV).unlink(missing_ok=True)
            else:
                # A repair accepted on an earlier run is not now: the kept copy goes back,
                # being post-processing's file again.
                os.replace(out / UNREPAIRED_CSV, out / FRAME_TIMES)

    except Exception as exc:                      # noqa: BLE001 — reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"

    return result


# =====================================================================
# 3. How many frames the video holds
# =====================================================================

def _header_frames(video) -> int | None:
    """The frame count in the video container's header, or None if it gives none."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return n if n > 0 else None


def count_video_frames(record):
    """
    The video's true frame count and how that is known — or ``(None, why not)``.

    DeepLabCut decoded every frame, so its row count is the authority. The container
    header is only a fallback: it agrees with DeepLabCut on every session the chunking bug
    damaged, but a truncated file's header reports the length it was *meant* to have, and
    a repair built on that would be wrong from start to finish.
    """
    # Frames per DeepLabCut model: its CSV's lines, less the three header rows.
    rows = {}
    for model, paths in record.dlc.items():
        if paths.get("csv"):
            with open(paths["csv"], "rb") as handle:
                rows[model] = sum(1 for _ in handle) - 3
    header = _header_frames(record.raw["video"])

    if len(set(rows.values())) > 1:
        return None, f"DeepLabCut models disagree on the frame count: {rows}"
    if rows:
        n = next(iter(rows.values()))
        if header is not None and header != n:
            return None, (f"the video header says {header:,} frames but DeepLabCut decoded "
                          f"{n:,} — a truncated or damaged file, not the chunking bug")
        return n, "deeplabcut_rows"
    if header is None:
        return None, "no DeepLabCut output, and the video header is unreadable"
    return header, "video_header"


# =====================================================================
# 4. The old converter, replayed
# =====================================================================

def replay_chunks(frame_ids, cores=COMPRESSION_CORES):
    """
    For each frame the old converter wrote, the position in the ``.bin`` it read it from.

    A model of the bug, line for line: chunks of ``max(n // cores, 500)`` entries of the
    tracker's frame-ID list; each seeks to its first frame **ID** as though it were a
    position, reads its chunk's length in order, and stops at the end of the file, which
    holds one frame per entry.
    """
    read = chunk_reads(frame_ids, cores)
    return np.concatenate(read) if read else np.empty(0, dtype=int)


def chunk_reads(frame_ids, cores=COMPRESSION_CORES):
    """:func:`replay_chunks` one chunk at a time: the ``.bin`` positions each read."""
    n = len(frame_ids)
    size = max(n // cores, MIN_CHUNK)
    read = []
    for start in range(0, n, size):
        seek = int(frame_ids[start])                    # the bug: an ID used as a position
        count = max(0, min(size, n - start, n - seek))  # the chunk, or up to the file's end
        read.append(np.arange(seek, seek + count))
    return read


def find_lost_chunks(chunks, t_by_position, n_video, position, touch_t, touch_port, ports):
    """
    Which chunks were never written, so that the rest add up to ``n_video`` frames: the
    choice that puts the head at the touched port on the most touches, less those it
    puts it away. Returns the chunk indices, or None when no choice adds up.

    One pass over the chunks in order, keeping for each number of frames lost so far the
    best score and how it was reached: a kept chunk's frames are the next ones in the
    video, a lost one's are not in it at all, so a kept chunk's score depends only on how
    much was lost before it.
    """
    lengths = np.array([len(c) for c in chunks])
    target = int(lengths.sum()) - n_video
    seeks = np.array([c[0] if len(c) else -1 for c in chunks])
    written_before = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    # Each touch's nearest .bin position by time, and the chunk that read it.
    r = np.clip(np.searchsorted(t_by_position, touch_t), 1, len(t_by_position) - 1)
    r = np.where(np.abs(t_by_position[r - 1] - touch_t) < np.abs(t_by_position[r] - touch_t),
                 r - 1, r)
    close = np.abs(t_by_position[r] - touch_t) <= MAX_LAG_S
    chunk = np.searchsorted(seeks, r, side="right") - 1
    inside = close & (chunk >= 0) & (r - seeks[np.maximum(chunk, 0)]
                                    < lengths[np.maximum(chunk, 0)])

    def score(c, lost):
        mine = inside & (chunk == c)
        frame = written_before[c] - lost + (r[mine] - seeks[c])
        ok = (frame >= 0) & (frame < n_video)
        d = np.linalg.norm(position[frame[ok]] - ports[touch_port[mine][ok]], axis=1)
        d = d[np.isfinite(d)]
        return int((d <= FAR_PX).sum()) - int((d > FAR_PX).sum())

    best = {0: (0, [])}
    for c, length in enumerate(lengths):
        step = {}
        for lost, (total, chosen) in best.items():
            kept = total + score(c, lost)
            if kept > step.get(lost, (-np.inf,))[0]:
                step[lost] = (kept, chosen)
            if lost + length <= target and total > step.get(lost + length, (-np.inf,))[0]:
                step[lost + length] = (total, chosen + [c])
        best = step
    return best[target][1] if target in best else None


def skipped_runs(positions, n_tracker):
    """
    Where the repaired video jumps over frames that were never written: a list of
    ``{"after_video_frame", "frames_skipped"}``.
    """
    jumps = np.flatnonzero(np.diff(positions) > 1)
    runs = [{"after_video_frame": int(i),
             "frames_skipped": int(positions[i + 1] - positions[i] - 1)} for i in jumps]
    tail = n_tracker - 1 - int(positions[-1])
    if tail:
        runs.append({"after_video_frame": len(positions) - 1, "frames_skipped": tail})
    return runs


# =====================================================================
# 5. The check
# =====================================================================

def _distances(lookup, position, ports, touch_port):
    """
    Head to touched port at each touch; NaN where there is no frame close enough.

    ``lookup`` is a :class:`FrameLookup` for the touches; the rest are arrays.
    """
    distance = np.full(len(touch_port), np.nan)
    usable = lookup.usable(MAX_LAG_S)
    distance[usable] = np.linalg.norm(
        position[lookup.frame[usable]] - ports[touch_port[usable]], axis=1)
    return distance


def head_and_touches(record, model, ports=None):
    """
    The head on every video frame and every beam break: a dict of ``key`` (the model's
    output), ``ports``, ``position``, ``dlc_table``, ``touch_t`` and ``touch_port`` — or
    of ``why`` there is nothing to judge with.
    """
    key = record.dlc_key(model)
    if key is None:
        return {"why": f"no output for model '{model}'"}
    rig = _rig_number(record)
    tables = approved_ports(ports) if ports is not None else approved_ports()
    if rig not in tables:
        return {"why": f"no calibrated port coordinates for rig {rig}"}
    dlc_table = read_dlc_table(record.dlc[key]["csv"])
    dlc = dlc_from_table(dlc_table)
    bearing, method = head_bearing(dlc)
    events = pd.read_csv(record.path("events"))
    touched = events[(events["event"] == "sensor_touch") & (events["index"] >= 0)]
    return {"key": key, "ports": tables[rig], "dlc_table": dlc_table,
            "position": head_position(dlc, bearing, method),
            "touch_t": touched["t"].to_numpy(dtype=float),
            "touch_port": touched["index"].to_numpy(dtype=int)}


def check_repair(record, model, before, after, n_video, manifest, image_dir, ports=None):
    """
    At every beam break, how far was the head from that port — as timed before and after?

    ``before`` and ``after`` are the two ``frame_times.csv`` tables. Returns the numbers
    for ``video_repair.json`` and, given an ``image_dir`` (a Path), draws the picture.
    ``ports`` is a port table (a Path or a string) to judge by instead of the package's —
    a cohort whose rig numbers mean different rigs (the 2026 red rigs are numbered 1 and
    2) has its own.
    """
    seen = head_and_touches(record, model, ports)
    if "why" in seen:
        return {"ran": False, "why": seen["why"]}
    key, ports, position, dlc_table = seen["key"], seen["ports"], seen["position"], seen["dlc_table"]
    touch_t, touch_port = seen["touch_t"], seen["touch_port"]

    # ---- every beam break, under each timing -------------------------
    old = frames_at(before, touch_t, n_video)
    new = frames_at(after, touch_t, n_video)
    d_old = _distances(old, position, ports, touch_port)
    d_new = _distances(new, position, ports, touch_port)

    def far(d) -> float | None:
        measured = d[np.isfinite(d)]
        return round(float(np.mean(measured > FAR_PX)), 4) if len(measured) else None

    check = {
        "ran": True,
        "model": key,
        "far_px": FAR_PX,
        "max_lag_s": MAX_LAG_S,
        "touches": int(len(touch_t)),
        "touches_measured_before": int(np.isfinite(d_old).sum()),
        "touches_measured_after": int(np.isfinite(d_new).sum()),
        "far_fraction_before": far(d_old),
        "far_fraction_after": far(d_new),
        "max_far_fraction": MAX_FAR_FRACTION,
    }
    if check["far_fraction_before"] is None or check["far_fraction_after"] is None:
        return {"ran": False, "why": "no touch had a video frame close enough to measure"}

    # Touches measured under both timings whose near/far class the repair changes.
    from scipy.stats import binomtest
    both = np.isfinite(d_old) & np.isfinite(d_new)
    fixed = int((both & (d_old > FAR_PX) & (d_new <= FAR_PX)).sum())
    broken = int((both & (d_old <= FAR_PX) & (d_new > FAR_PX)).sum())
    changed = fixed + broken
    p_worse = 1.0
    if changed:
        p_worse = float(f"{binomtest(broken, changed, 0.5, alternative='greater').pvalue:.3g}")
    check.update(
        touches_fixed=fixed,
        touches_broken=broken,
        p_worse=p_worse,
        reject_p=REJECT_P,
    )
    check["passed"] = (check["p_worse"] >= REJECT_P
                       and check["far_fraction_after"] <= MAX_FAR_FRACTION)

    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{record.session_id}.png"
        _draw_check(path, record, manifest, check, touch_t, touch_port, old, new,
                    d_old, d_new, ports, position, dlc_table, after)
        check["image"] = str(path)
    return check


def _draw_markers(axis, dlc_table, frame, head, ports, port):
    """Ports, the touched one ringed; DeepLabCut's ears, nose and spine on this frame."""
    axis.scatter(ports[:, 0], ports[:, 1], s=60, facecolors="none",
                 edgecolors="white", lw=1.2)
    axis.scatter([ports[port, 0]], [ports[port, 1]], s=420, facecolors="none",
                 edgecolors=TOUCHED, lw=3)

    parts = set(dlc_table.columns.get_level_values(0))

    def xy(part):
        return dlc_table[part]["x"].iloc[frame], dlc_table[part]["y"].iloc[frame]

    spine = [xy(p) for p in ("spine_1", "spine_2", "spine_3", "spine_4") if p in parts]
    if len(spine) >= 2:
        axis.plot(*zip(*spine), color="#ff4fd8", lw=2, alpha=0.9)
    for part, marker, colour in (("left_ear", "o", "#00e5ff"),
                                 ("right_ear", "o", "#00e5ff"),
                                 ("nose", "v", "#ffd400")):
        if part in parts:
            axis.scatter(*xy(part), s=40, marker=marker, color=colour,
                         edgecolors="black", lw=0.6)
    if np.isfinite(head).all():
        axis.scatter(*head, s=90, marker="x", color="white", lw=2.5)
        axis.plot([head[0], ports[port, 0]], [head[1], ports[port, 1]],
                  color="white", ls="--", lw=1.2)


def _draw_check(path, record, manifest, check, touch_t, touch_port, old, new,
                d_old, d_new, ports, position, dlc_table, after):
    """Distances through the session, and a few touches as the video actually shows them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(17, 11), constrained_layout=True)
    grid = figure.add_gridspec(3, N_EXAMPLES, height_ratios=[1.1, 1, 1])

    # ---- top: every touch, before and after --------------------------
    axis = figure.add_subplot(grid[0, :])
    for d, colour, label, fraction in (
            (d_old, BEFORE, "as post-processing timed it", check["far_fraction_before"]),
            (d_new, AFTER, "repaired", check["far_fraction_after"])):
        shown = np.isfinite(d)
        minutes = touch_t[shown] / 60
        axis.scatter(minutes, np.minimum(d[shown], 900), s=4, alpha=0.3, color=colour, lw=0)
        rolling = pd.Series(d[shown]).rolling(51, center=True, min_periods=10).median()
        share = "" if fraction is None else f" — {fraction:.1%} of touches over {FAR_PX:g} px"
        axis.plot(minutes, rolling, color=colour, lw=2, label=label + share)
    axis.axhline(FAR_PX, color="#8a8a8a", ls="--", lw=1)

    # where the repaired video jumps over frames that were never written
    timing = after["t"].to_numpy(dtype=float)
    for run in manifest.get("skipped", []):
        at = timing[run["after_video_frame"]]
        if np.isfinite(at):
            axis.axvline(at / 60, ymin=0.96, ymax=1, color="#555555", lw=1)

    counts = f"{check['touches_fixed']} touches fixed, {check['touches_broken']} broken"
    if check["p_worse"] < REJECT_P:
        verdict = f"REJECTED — {check['touches_broken']} touches broken, {check['touches_fixed']} fixed"
    elif check["passed"]:
        verdict = f"passed — {counts}"
    else:
        verdict = f"repaired, but over the 10% line — {counts}"
    axis.set(xlabel="time in session (min)", ylabel="head to touched port (px)",
             ylim=(0, 900),
             title=(f"{record.session_id} — {verdict}.  "
                    f"{manifest['camera_drops']:,} camera drops, "
                    f"{manifest['frames_not_written']:,} frames never written, "
                    f"timing off by up to {manifest['max_shift_frames']:,} frames before "
                    "repair  (ticks along the top: where the video skips)"))
    axis.legend(frameon=False, loc="upper left")

    # ---- below: the frames each timing picks at a few touches --------
    # A few touches spread through the session, preferring ones the repair moved.
    both = np.isfinite(d_old) & np.isfinite(d_new)
    moved = np.flatnonzero(both & (old.frame != new.frame))
    pool = moved if len(moved) else np.flatnonzero(both)
    if len(pool) == 0:
        figure.savefig(path, dpi=90)
        plt.close(figure)
        return
    picks = np.linspace(0, len(pool) - 1, min(N_EXAMPLES, len(pool))).round().astype(int)
    examples = [int(pool[i]) for i in np.unique(picks)]

    import cv2
    capture = cv2.VideoCapture(str(record.raw["video"]))
    left, right = ports[:, 0].min() - 170, ports[:, 0].max() + 170
    top, bottom = ports[:, 1].min() - 110, ports[:, 1].max() + 110
    try:
        for column, touch in enumerate(examples):
            for row, (lookup, d, name, colour) in enumerate(
                    ((old, d_old, "before", BEFORE), (new, d_new, "after", AFTER)), start=1):
                axis = figure.add_subplot(grid[row, column])
                frame = int(lookup.frame[touch])
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if ok:
                    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap="gray")
                _draw_markers(axis, dlc_table, frame, position[frame], ports,
                              touch_port[touch])
                axis.set_xlim(max(left, 0), right)
                axis.set_ylim(bottom, max(top, 0))
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_color(colour)
                    spine.set_linewidth(3)
                title = f"{name}: frame {frame:,} — head {d[touch]:.0f} px from port"
                if row == 1:
                    title += (f"\ntouch on port {touch_port[touch]} at "
                              f"{touch_t[touch] / 60:.1f} min")
                axis.set_title(title, fontsize=10, color="#222222")
    finally:
        capture.release()

    figure.savefig(path, dpi=90)
    plt.close(figure)


# =====================================================================
# 6. The cohort's summary
# =====================================================================

def write_summary(cohort):
    """
    One row per session this repair has a decision about, and a picture of the checks.
    Returns the path of ``summary.csv``.
    """
    rows = []
    for record in cohort:
        stored = _read_json(record.out_dir / REPAIR_JSON)
        if not stored:
            continue
        check = stored.get("check") or {}
        rows.append({
            "session_id": record.session_id,
            "status": stored.get("status"),
            "reason": stored.get("reason"),
            "tracker_frames": stored.get("tracker_frames"),
            "video_frames": stored.get("video_frames"),
            "camera_drops": stored.get("camera_drops"),
            "frames_not_written": stored.get("frames_not_written"),
            "max_shift_frames": stored.get("max_shift_frames"),
            "lost_chunks": (stored.get("lost_chunks") or {}).get("chunks"),
            "far_fraction_before": check.get("far_fraction_before"),
            "far_fraction_after": check.get("far_fraction_after"),
            "touches_fixed": check.get("touches_fixed"),
            "touches_broken": check.get("touches_broken"),
            "p_worse": check.get("p_worse"),
            "check_passed": check.get("passed"),
            "decided_at": stored.get("decided_at"),
        })

    folder = cohort.base / IMAGE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    path = folder / "summary.csv"
    table.to_csv(path, index=False)

    checked = table
    if len(table):
        checked = table.dropna(subset=["far_fraction_before", "far_fraction_after"])
    if len(checked):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        passed = checked["check_passed"].astype(bool)
        rejected = checked["status"] == "rejected"
        for mask, colour, label in ((passed, AFTER, "repaired, passed"),
                                    (~passed & ~rejected, TOUCHED, "repaired, over the line"),
                                    (rejected, BEFORE, "rejected")):
            axis.scatter(100 * checked.far_fraction_before[mask],
                         100 * checked.far_fraction_after[mask],
                         s=40, color=colour, alpha=0.8, lw=0,
                         label=f"{label} ({int(mask.sum())})")
        for _, row in checked[~passed].iterrows():
            axis.annotate(row.session_id, (100 * row.far_fraction_before,
                                           100 * row.far_fraction_after),
                          fontsize=7, xytext=(4, 4), textcoords="offset points")
        axis.axhline(100 * MAX_FAR_FRACTION, color="#8a8a8a", ls="--", lw=1,
                     label="pass line")
        axis.plot([0, 100], [0, 100], color="#bbbbbb", lw=1, label="no change")
        axis.set(xlim=(0, 100), ylim=(0, 100),
                 xlabel=f"touches with head > {FAR_PX:g} px from port, before (%)",
                 ylabel="… after repair (%)",
                 title=f"{cohort.base.name}: {len(checked)} sessions checked")
        axis.legend(frameon=False, loc="upper left")
        figure.savefig(folder / "summary.png", dpi=110)
        plt.close(figure)

    return path


# =====================================================================
# 7. What comes back
# =====================================================================

@dataclass
class SessionRepair:
    """What happened to one session."""

    session_id: str
    #: repaired · rejected · unchecked · unexplained · not_needed · already_decided
    status: str | None = None
    reason: str = ""
    max_shift_frames: int = 0
    far_before: float | None = None
    far_after: float | None = None
    passed: bool | None = None
    skipped: str | None = None
    error: str | None = None


@dataclass
class RepairResult:
    """What one sweep did, per session."""

    base: Path
    results: list[SessionRepair] = field(default_factory=list)

    def with_status(self, status):
        """The sessions given this status."""
        return [r for r in self.results if r.status == status]

    def describe(self):
        """The sweep, counted by outcome, then every session that needs a look."""
        repaired = self.with_status("repaired")
        checked = [r for r in repaired if r.passed is not None]
        failed = [r for r in checked if not r.passed]
        errors = [r for r in self.results if r.error]
        skipped = [r for r in self.results if r.skipped]

        first = f"{self.base.name} — {len(repaired)} repaired"
        if checked:
            first += f" ({len(checked) - len(failed)} passed, {len(failed)} over the line)"
        first += (f", {len(self.with_status('rejected'))} rejected, "
                  f"{len(self.with_status('unchecked'))} unchecked, "
                  f"{len(self.with_status('unexplained'))} unexplained, "
                  f"{len(self.with_status('already_decided'))} already decided, "
                  f"{len(self.with_status('not_needed'))} not needed, "
                  f"{len(skipped)} skipped, {len(errors)} errors")
        lines = [first]

        for status, heading in (
                ("rejected", "REJECTED — the repair made the timing significantly worse; left as it was"),
                ("unchecked", "UNCHECKED — nothing to check the repair with; left as it was")):
            if self.with_status(status):
                lines += ["", f"  {heading}"]
                lines += [f"    {r.session_id}  {r.reason}" for r in self.with_status(status)]

        if self.with_status("unexplained"):
            lines += ["", "  UNEXPLAINED — left as they were"]
            lines += [f"    {r.session_id}  {r.reason}"
                      for r in self.with_status("unexplained")]
        if failed:
            lines += ["", "  REPAIRED, BUT STILL OVER THE 10% LINE — something else is wrong too"]
            lines += [f"    {r.session_id}  head far from port on "
                      f"{r.far_before:.1%} of touches before, {r.far_after:.1%} after"
                      for r in failed]
        if errors:
            lines += ["", "  ERRORS"]
            lines += [f"    {r.session_id}  {r.error}" for r in errors]
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cohort", help="cohort folder")
    parser.add_argument("model", help="DeepLabCut model (its Task) to check repairs with")
    parser.add_argument("--cores", type=int, default=COMPRESSION_CORES,
                        help=f"cores on the compressing machine (default {COMPRESSION_CORES})")
    parser.add_argument("--only", nargs="+", help="just these session IDs")
    parser.add_argument("--force", action="store_true",
                        help="decide again from post-processing's originals")
    parser.add_argument("--dry-run", action="store_true", help="decide, write nothing")
    parser.add_argument("--workers", type=int, default=1, help="sessions at once")
    parser.add_argument("--ports", help="port table to judge by instead of the package's")
    args = parser.parse_args()

    result = repair_video_timing(args.cohort, args.model, cores=args.cores,
                                 only=tuple(args.only) if args.only else None,
                                 force=args.force, dry_run=args.dry_run,
                                 workers=args.workers, ports=args.ports)
    return 1 if any(r.error for r in result.results) else 0


if __name__ == "__main__":
    sys.exit(main())
