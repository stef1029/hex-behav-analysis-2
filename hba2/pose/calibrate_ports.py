"""
Where the ports are.
====================

Measures the six port positions for every rig in a cohort, and writes them out for
everything else to use.

The method is that **at the moment a beam breaks, the animal's head is at that port**,
so the median head position over every touch of a port *is* that port — in pixel
coordinates, from the same DeepLabCut model the angles will be measured with, and with
no calibration rig or chessboard involved. Pooling across a whole cohort is what makes
it solid: a port one session barely visited is still tens of thousands of touches across
a cohort.

This is a **deliberate act, not something that happens at analysis time**. Run it, read
the numbers, look at the picture, and if it is right the table becomes the one every
cohort uses::

    python calibrate_ports.py <cohort> <model task>

It writes, beside this script::

    port_coordinates/
    ├── port_coordinates.json   the table — what every analysis reads
    ├── rig_1.png               the same numbers drawn on that rig's own video
    └── rig_2.png

The pictures live with the table on purpose: a list of coordinates is impossible to
sanity-check by eye, and a picture of them sitting in the port mouths is immediate. They
are in the package and under version control, so a recalibration is a diff someone can
read, date and revert.

Only the rigs found in the given cohort are rewritten. A rig the cohort knows nothing
about keeps whatever it already had, so calibrating from one cohort never silently
discards another's work.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hba2 import Cohort, SessionRecord  # noqa: E402
from hba2.pose.frame_lookup import MAX_LAG_S  # noqa: E402

# Through the package, not as a loose module: pose.py imports the cohort relatively,
# which only resolves when it is loaded as part of the package it belongs to.
from hba2.pose.pose import (  # noqa: E402
    N_PORTS, PORT_TABLE, _readable_json, _rig_number, _touches, head_bearing,
    head_position, load_port_table, port_geometry, read_dlc,
)

#: Where the table and its pictures live: beside this script, in the package, in git.
OUTPUT_DIR = PORT_TABLE.parent
TABLE_NAME = PORT_TABLE.name

#: A session with fewer trials than this was aborted or a test, and its touches are not
#: a fair sample of how an animal approaches a port.
MIN_TRIALS = 50

#: How far a port's touches may sit from their own median, in pixels, for that port in
#: that session to be pooled. A clean port is around 10 px; a sensor breaking as the
#: animal passes rather than pokes gives well over 100. Applied per port rather than per
#: session, because one unreliable sensor should cost that sensor and nothing else.
MAX_SCATTER_PX = 30.0

#: Validated categorical slots, one per port (see the dataviz palette reference).
PORT_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED = "#33322e", "#8a8880"


# =====================================================================
# 1. Running it
# =====================================================================

def _print_flushed(message):
    """
    What calibrate reports to by default. Flushed, so progress is visible when the
    output is redirected to a file — this reads every DeepLabCut file in the cohort and
    takes minutes.
    """
    print(message, flush=True)


def calibrate(cohort, model, *, min_trials=MIN_TRIALS, max_scatter_px=MAX_SCATTER_PX,
              max_lag_s=MAX_LAG_S, out_dir=OUTPUT_DIR, fill_missing=None, log=None):
    """
    Measure every rig in a cohort, write the table, and draw the pictures.

    ``cohort`` is a :class:`Cohort` or a path (Path or string) to one; ``model`` the
    DeepLabCut model's Task. ``out_dir`` is a Path. ``fill_missing``, when a number,
    places ports with fewer touches than that from a fitted hexagon — see
    :func:`fill_from_hexagon`. ``log`` is called with each line of progress.

    Returns a list of :class:`RigPorts`, one per rig found, complete or not.
    """
    log = log or _print_flushed

    # ---- 1. Which sessions -----------------------------------------------
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    candidates = [r for r in cohort if r.is_processed and r.dlc_key(model)]
    log(f"{cohort.base.name}: {len(candidates)} of {len(cohort)} session(s) "
        f"have '{model}' and are post-processed")

    # ---- 2. Each session's touches, grouped by rig ---------------------------
    by_rig = {}        # rig number -> list of SessionPorts
    short = {}         # rig number -> how many sessions had too few trials
    unreadable = 0

    for number, record in enumerate(candidates, start=1):
        if number % 50 == 0:
            log(f"  measured {number}/{len(candidates)}")
        measured = measure_session(record, model, max_scatter_px, max_lag_s)
        if measured is None:
            unreadable += 1
            continue
        if measured.n_trials < min_trials:
            short[measured.rig] = short.get(measured.rig, 0) + 1
            continue
        if measured.points:
            by_rig.setdefault(measured.rig, []).append(measured)

    if unreadable:
        log(f"  {unreadable} session(s) could not be read and were skipped")

    # ---- 3. Pooled, per rig --------------------------------------------------
    rigs = []
    for rig_number in sorted(by_rig):
        rig = pool(by_rig[rig_number])
        rig.n_short = short.get(rig_number, 0)
        if fill_missing:
            rig = fill_from_hexagon(rig, fill_missing)
            if rig.filled:
                log(f"rig {rig.rig}: port(s) {rig.filled} placed from the fitted hexagon "
                    f"(fewer than {fill_missing} touches)")
        rigs.append(rig)
        log("\n" + rig.describe())
        if not rig.complete:
            log(f"  WARNING: rig {rig.rig} has a port with no usable touches — "
                "not written")

    # ---- 4. The pictures and the table, for the complete rigs ----------------
    complete = [r for r in rigs if r.complete]
    if complete:
        log("")
        for rig in complete:
            log(f"wrote {draw(rig, out_dir / f'rig_{rig.rig}.png')}")
        filters = {"min_trials": min_trials, "max_scatter_px": max_scatter_px,
                   "max_lag_s": max_lag_s}
        log(f"wrote {write_table(complete, cohort, model, filters, out_dir)}")
        log("\nLook at the pictures before trusting the numbers.")

    return rigs


# =====================================================================
# 2. One session's contribution
#
# This is the only place in the package that turns head positions into port
# coordinates. Analysis reads the calibrated table and never derives its own, so that
# two cohorts can never end up measuring angles from different origins.
# =====================================================================

def measure_ports(ports_touched, frames, position):
    """
    Median head position at each port's touches, with how tightly they agree.

    Median rather than mean throughout: a handful of wild detections is expected, and
    the median ignores them instead of being dragged by them. ``scatter`` is the median
    distance from that estimate, which says whether the touches describe a port at all.

    Returns:
        ``(coords, counts, scatter)`` — ``coords`` is ``(N_PORTS, 2)``, NaN for a port
        never touched.
    """
    coords = np.full((N_PORTS, 2), np.nan)
    counts = np.zeros(N_PORTS, dtype=int)
    scatter = np.full(N_PORTS, np.nan)

    for port in range(N_PORTS):
        xy = position[frames[ports_touched == port]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        coords[port] = np.median(xy, axis=0)
        counts[port] = len(xy)
        scatter[port] = float(np.median(np.linalg.norm(xy - coords[port], axis=1)))

    return coords, counts, scatter


def measure_session(record, model, max_scatter_px,
                    max_lag_s=MAX_LAG_S) -> SessionPorts | None:
    """
    Head position at every beam break, grouped by port, with the bad ports dropped.

    A touch counts only when the video has a frame within ``max_lag_s`` of it — a touch
    that fell in a gap would be measured on an image of the animal somewhere else.

    Returns None when the session cannot contribute at all — no output for this model,
    not post-processed, or unreadable.
    """
    key = record.dlc_key(model)
    if key is None or not record.is_processed:
        return None

    # ---- 1. Read, and find the head ------------------------------------------
    try:
        rig = _rig_number(record)
        dlc = read_dlc((record.dlc.get(key) or {}).get("csv"))
        bearing, method = head_bearing(dlc)
        position = head_position(dlc, bearing, method)

        frame_times = pd.read_csv(record.path("frame_times"))
        ports_touched, touches = _touches(record, frame_times, dlc.n_frames)
        n_trials = len(pd.read_csv(record.path("trials")))
    except Exception:                              # noqa: BLE001 — reported by caller
        return None

    # ---- 2. The touches with a frame close enough, measured per port ---------
    keep = touches.usable(max_lag_s)
    ports_touched, frames = ports_touched[keep], touches.frame[keep]
    _, counts, scatter = measure_ports(ports_touched, frames, position)

    # ---- 3. Keep each port's points, unless they scatter too much -----------
    result = SessionPorts(record=record, rig=rig, n_trials=n_trials,
                          scatter=scatter, counts=counts)
    for port in range(N_PORTS):
        xy = position[frames[ports_touched == port]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        if not np.isfinite(scatter[port]) or scatter[port] > max_scatter_px:
            result.rejected.append(port)
            continue
        result.points[port] = xy

    return result


# =====================================================================
# 3. Pooling, per rig
# =====================================================================

def pool(sessions):
    """
    One table, a :class:`RigPorts`, from many sessions (a list of :class:`SessionPorts`
    from one rig): the median of every kept touch, port by port.

    Median rather than mean because a handful of wild detections is expected and the
    median ignores them rather than being dragged by them.
    """
    rig = sessions[0].rig
    coords = np.full((N_PORTS, 2), np.nan)
    counts = np.zeros(N_PORTS, dtype=int)
    scatter = np.full(N_PORTS, np.nan)

    for port in range(N_PORTS):
        chunks = [s.points[port] for s in sessions if port in s.points]
        if not chunks:
            continue
        xy = np.concatenate(chunks)
        coords[port] = np.median(xy, axis=0)
        counts[port] = len(xy)
        scatter[port] = float(np.median(np.linalg.norm(xy - coords[port], axis=1)))

    return RigPorts(
        rig=rig, coords=coords, counts=counts, scatter=scatter,
        n_sessions=len(sessions),
        n_short=0,
        n_rejected_ports=sum(len(s.rejected) for s in sessions),
        sessions=sessions,
    )


def fill_from_hexagon(rig, min_touches):
    """
    Place ports the animals hardly touched, from the ports they did.

    Some cohorts used only some ports (2504's tetrode mice barely touched ports 1 and 2),
    so those have too few touches to measure. The arena is a regular hexagon, so a
    centre, radius, rotation and winding fitted to the well-measured ports puts the rest
    — the construction a hard-coded table uses for every port. Only ports with fewer
    than ``min_touches`` touches are replaced, and the entry says which, so a filled
    port is never mistaken for a measured one. Needs at least three measured ports.

    Returns the same :class:`RigPorts`, changed in place, or unchanged when there is
    nothing to fill or too little to fit.
    """
    from scipy.optimize import least_squares

    measured = [p for p in range(N_PORTS)
                if rig.counts[p] >= min_touches and np.isfinite(rig.coords[p]).all()]
    missing = [p for p in range(N_PORTS) if p not in measured]
    if not missing or len(measured) < 3:
        return rig
    xy = rig.coords[measured]
    best = None
    for winding in (1, -1):
        def residual(v):
            cx, cy, r, theta = v
            a = theta + winding * np.radians(60) * np.asarray(measured)
            return np.concatenate([cx + r * np.cos(a) - xy[:, 0],
                                   cy - r * np.sin(a) - xy[:, 1]])
        centre = xy.mean(axis=0)
        first = np.arctan2(-(xy[0, 1] - centre[1]), xy[0, 0] - centre[0]) \
            - winding * np.radians(60) * measured[0]
        fit = least_squares(residual, [*centre, np.linalg.norm(xy - centre, axis=1).mean(),
                                       first])
        if best is None or fit.cost < best[0].cost:
            best = (fit, winding)
    (cx, cy, r, theta), winding = best[0].x, best[1]
    coords = rig.coords.copy()
    for p in missing:
        a = theta + winding * np.radians(60) * p
        coords[p] = [cx + r * np.cos(a), cy - r * np.sin(a)]
    rig.coords, rig.filled = coords, missing
    return rig


# =====================================================================
# 4. The picture — what makes the numbers checkable
# =====================================================================

def video_frame(path, frame_number=1000):
    """
    One frame of a rig's own video (a Path or a string), as a backdrop. None if it
    cannot be read.
    """
    try:
        import cv2 as cv
    except ImportError:
        return None
    capture = cv.VideoCapture(str(path))
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            return None
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    finally:
        capture.release()


def draw(rig, out) -> Path | None:
    """
    The pooled ports on that rig's own video, with the touches they came from, saved
    to ``out`` (a Path).

    A list of twelve numbers cannot be checked by eye; six dots sitting in the port
    mouths can be, in a second. That is the whole reason this file exists.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    # The session that contributed most is the one most likely to have a usable video.
    ranked = sorted(rig.sessions, key=lambda s: -int(s.counts.sum()))
    backdrop, source = None, None
    for candidate in ranked[:5]:
        video = candidate.record.raw.get("video")
        if video is None:
            continue
        backdrop = video_frame(video)
        if backdrop is not None:
            source = candidate.record.session_id
            break

    figure, axis = plt.subplots(figsize=(7.2, 6.4))
    if backdrop is not None:
        axis.imshow(backdrop, cmap="gray")

    for port in range(N_PORTS):
        if not np.isfinite(rig.coords[port]).all():
            continue
        colour = PORT_COLOURS[port]
        chunks = [s.points[port] for s in rig.sessions if port in s.points]
        if chunks:
            cloud = np.concatenate(chunks)
        else:
            cloud = np.empty((0, 2))
        # Thin the cloud — hundreds of thousands of points render as a solid blob and
        # take a minute to draw.
        if len(cloud) > 4000:
            cloud = cloud[np.random.default_rng(0).choice(len(cloud), 4000, replace=False)]
        axis.scatter(cloud[:, 0], cloud[:, 1], s=3, alpha=0.10, color=colour,
                     linewidths=0)

        if port in rig.filled:
            # A port placed from the fitted hexagon is drawn hollow: not measured.
            face, edge, width = "none", colour, 2.2
        else:
            # A measured port: filled, with a white rim.
            face, edge, width = colour, "white", 1.6
        axis.scatter(*rig.coords[port], s=110, color=face, edgecolor=edge,
                     linewidth=width, zorder=3)
        # Outlined, because the backdrop runs from black surround to blown-out white
        # arena and a single ink colour is unreadable on one or the other.
        axis.annotate(f"{port}", rig.coords[port], color="white", fontsize=12,
                      weight="bold", xytext=(11, 11), textcoords="offset points",
                      zorder=4,
                      path_effects=[patheffects.withStroke(linewidth=3,
                                                           foreground="#1a1a1a")])

    shape = port_geometry(rig.coords)
    axis.set_title(f"Rig {rig.rig} — {int(rig.counts.sum()):,} touches from "
                   f"{rig.n_sessions} sessions", fontsize=12, color=INK)
    label = (f"radius {shape['radius_px']} px (sd {shape['radius_sd_px']}), "
             f"spacing sd {shape['spacing_sd_deg']}°")
    if source:
        label += f"   ·   frame from {source}"
    else:
        label += "   ·   no video available"
    axis.set_xlabel(label, fontsize=9, color=MUTED)
    axis.set_xticks([])
    axis.set_yticks([])

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out


# =====================================================================
# 5. The table
# =====================================================================

def write_table(rigs, cohort, model, filters, out_dir):
    """
    Merge these rigs (a list of :class:`RigPorts`) into the table in ``out_dir`` (a
    Path), leaving rigs this cohort knows nothing about. Returns the table's path.

    Each entry records what produced it, because a year from now the only way to judge
    whether to trust six numbers is to know how many touches from which sessions went
    into them.
    """
    path = out_dir / TABLE_NAME
    table = load_port_table(path)

    for rig in rigs:
        scatter_px = []
        for value in rig.scatter:
            if np.isfinite(value):
                scatter_px.append(round(float(value), 1))
            else:
                scatter_px.append(None)

        derived_from = {
            "cohort": cohort.base.name,
            "model": model,
            "sessions": rig.n_sessions,
            "touches": int(rig.counts.sum()),
            "touches_per_port": rig.counts.tolist(),
            "scatter_px": scatter_px,
            "filters": filters,
        }
        if rig.filled:
            derived_from["filled_from_hexagon"] = rig.filled
            derived_from["filled_note"] = ("too few touches to measure; placed by a "
                                           "regular hexagon fitted to the measured ports")

        table[f"rig_{rig.rig}"] = {
            "coords": np.round(rig.coords, 1).tolist(),
            "geometry": port_geometry(rig.coords),
            "derived_from": derived_from,
            "calibrated": date.today().isoformat(),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(_readable_json(dict(sorted(table.items()))))
    return path


# =====================================================================
# 6. What comes back
# =====================================================================

@dataclass
class SessionPorts:
    """Where one session says each port is, and how much it agrees with itself."""

    record: SessionRecord
    rig: int
    n_trials: int
    #: port -> the head positions at its touches, for the ports that passed
    points: dict[int, np.ndarray] = field(default_factory=dict)
    scatter: np.ndarray = field(default_factory=lambda: np.full(N_PORTS, np.nan))
    counts: np.ndarray = field(default_factory=lambda: np.zeros(N_PORTS, dtype=int))
    rejected: list[int] = field(default_factory=list)


@dataclass
class RigPorts:
    """The pooled answer for one rig, and what went into it."""

    rig: int
    coords: np.ndarray
    counts: np.ndarray
    scatter: np.ndarray
    n_sessions: int
    n_short: int
    n_rejected_ports: int
    sessions: list[SessionPorts] = field(default_factory=list)
    #: Ports placed by fitting a regular hexagon to the others — see fill_from_hexagon.
    filled: list[int] = field(default_factory=list)

    @property
    def complete(self):
        return bool(np.isfinite(self.coords).all())

    def describe(self):
        shape = port_geometry(self.coords)

        summary = (f"rig {self.rig}: {self.n_sessions} session(s) pooled, "
                   f"{int(self.counts.sum()):,} touches")
        if self.n_short:
            summary += f", {self.n_short} too short"
        if self.n_rejected_ports:
            summary += f", {self.n_rejected_ports} port(s) too scattered"

        hexagon_note = ""
        if shape['spacing_sd_deg'] is not None:
            hexagon_note = "(a perfect hexagon is 0)"

        lines = [
            summary,
            f"  radius {shape['radius_px']} px (sd {shape['radius_sd_px']}), "
            f"spacing sd {shape['spacing_sd_deg']} deg   {hexagon_note}",
            "  port        x        y    touches   scatter",
        ]
        for port in range(N_PORTS):
            x, y = self.coords[port]
            spread = self.scatter[port]
            row = f"    {port}   {x:7.1f}  {y:7.1f}   {self.counts[port]:8,}   "
            if np.isfinite(spread):
                row += f"{spread:5.1f} px"
            else:
                row += "   —"
            lines.append(row)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cohort", help="cohort folder to measure from")
    parser.add_argument("model", help="DeepLabCut model, by its Task name")
    parser.add_argument("--min-trials", type=int, default=MIN_TRIALS,
                        help=f"skip sessions with fewer trials (default {MIN_TRIALS})")
    parser.add_argument("--max-scatter-px", type=float, default=MAX_SCATTER_PX,
                        help="drop a port whose touches scatter more than this "
                             f"(default {MAX_SCATTER_PX:g})")
    parser.add_argument("--max-lag-s", type=float, default=MAX_LAG_S,
                        help="ignore a touch with no video frame this close to it "
                             f"(default {MAX_LAG_S:g})")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR,
                        help="where to write (default: beside this script)")
    args = parser.parse_args()

    rigs = calibrate(args.cohort, args.model, min_trials=args.min_trials,
                     max_scatter_px=args.max_scatter_px, max_lag_s=args.max_lag_s,
                     out_dir=args.out)
    return 0 if any(r.complete for r in rigs) else 1


if __name__ == "__main__":
    sys.exit(main())
