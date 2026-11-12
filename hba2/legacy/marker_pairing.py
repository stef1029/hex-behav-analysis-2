"""
Trial windows paired with the rig's rows by time, for sessions that lost a marker.
=================================================================================

Post-processing pairs the trial-marker pulses with the rows of the rig's trials CSV **by
order**, which is right when there is one pulse per row. When a pulse is missing from the
middle of a session — a DAQ message that was never saved — every row after it is given
the next trial's window, silently. A surplus pulse at the end (the session stopped
mid-trial) does no harm by order, but one in the middle does the same damage.

This pairs them by time instead, the way the pre-marker and legacy trials are paired: the
rows' own start times (``time_since_start_s - trial_duration_s``, PC clock) are lined up
with the marker onsets by the offset most of them agree on, refitted to a line, and
aligned in order allowing either side to skip (:func:`.trials._pair`). Then:

* a row paired with a marker gets that marker's window, exactly as post-processing would;
* a row with no marker gets the window its logged start and end put on the fitted line,
  marked ``"window_start": "rig_log"`` in its ``details_json`` — on the DAQ clock to the
  fit's accuracy, not the marker's;
* a marker with no row is left out of ``trials.csv`` and counted.

Rows are kept verbatim. ``events.csv``, ``trials.csv`` and ``notes.csv`` are rewritten
and ``session.json`` records how under ``trial_windows``. Only sessions whose marker count
differs from their row count are touched.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..cohort import Cohort
from ..postprocessing.pipeline import TRIAL_MARKER
from ..postprocessing.sources import daq as daq_source, trials_csv
from ..postprocessing.utils import clock
from . import trials as legacy_trials
from .premarker import _start, _write


def pair_cohort(base, log=print):
    """
    Re-pair by time every session in the cohort at ``base`` (a Path or a string) whose
    marker count differs from its row count. Returns ``{"done": {session ID: summary},
    "failed": {session ID: error}}``.
    """
    done, failed = {}, {}
    for record in Cohort(base):
        if not needs_pairing(record):
            continue
        try:
            s = pair_session(record)
            done[record.session_id] = s
            log(f"  {record.session_id}: {s['rows']} rows, {s['markers']} markers, "
                f"{s['paired']} paired; {len(s['rows_placed_from_rig_log'])} rows from "
                f"the rig log; markers without a row {s['markers_without_row']}; "
                f"residual median {s['pair_residual_ms_median']:.0f} ms, "
                f"max {s['pair_residual_ms_max']:.0f} ms")
        except Exception as exc:                    # noqa: BLE001 — reported per session
            failed[record.session_id] = f"{type(exc).__name__}: {exc}"
            log(f"  {record.session_id}: FAILED — {type(exc).__name__}: {exc}")
    log(f"{Path(base).name}: {len(done)} session(s) re-paired by time, {len(failed)} failed")
    return {"done": done, "failed": failed}


def needs_pairing(record):
    """Processed, with marker pulses, and a different number of them than rows."""
    if not record.is_processed or record.raw.get("trials_csv") is None:
        return False
    try:
        n_rows = len(pd.read_csv(record.path("trials_csv")))
    except Exception:                               # noqa: BLE001
        return False
    daq = daq_source.read(record.path("daq"))
    n_markers = len(clock.trial_windows(daq, TRIAL_MARKER)[0]) if daq.has(TRIAL_MARKER) else 0
    return n_rows > 0 and n_markers > 0 and n_markers != n_rows


def pair_session(record):
    """
    Pair one session's rows with its marker windows by time, and rewrite its files (a
    :class:`~..cohort.SessionRecord`). Returns what was done, as recorded in
    ``session.json``.
    """
    # ---- 1. Rows paired with marker onsets ------------------------------------
    daq = daq_source.read(record.path("daq"))
    starts, ends = clock.trial_windows(daq, TRIAL_MARKER)
    rows = trials_csv.read(record.path("trials_csv"))
    t_cued = [_start(r) for r in rows]
    t_done = [r.software_time if r.software_time is not None else np.nan for r in rows]
    marker_of_row, fit, passes = _pair_in_passes(starts, t_cued)
    if fit is None:
        raise ValueError("the rows' times could not be lined up with the marker onsets")
    pairs = {k: row for row, k in marker_of_row.items()}
    paired_rows = np.array(sorted(marker_of_row))
    offsets = np.array([starts[marker_of_row[r]] - t_cued[r] for r in paired_rows])

    def to_daq(index, t):
        # The offset of the nearest paired row before this one (after it, at the start):
        # the rig PC's clock can step mid-session, so one line for the whole session
        # would misplace everything on the far side of a step.
        k = np.searchsorted(paired_rows, index) - 1
        return t + offsets[max(k, 0)]

    # ---- 2. Each row's window ---------------------------------------------------
    new_starts, new_ends, kept, placed = [], [], [], []
    for index, row in enumerate(rows.rows):
        if index in marker_of_row:
            # Paired: the marker's window, exactly as post-processing would.
            k = marker_of_row[index]
            start, end, source = float(starts[k]), float(ends[k]), "marker"
        elif np.isfinite(t_cued[index]):
            # Not paired: its logged start and end, on the clock by its neighbours' offset.
            start = to_daq(index, t_cued[index])
            end = to_daq(index, t_done[index]) if np.isfinite(t_done[index]) else start
            source = "rig_log"
            placed.append(index)
        else:
            continue
        new_starts.append(start)
        new_ends.append(max(end, start))
        kept.append(replace(row, index=len(kept),
                            details={**(row.details or {}), "window_start": source}))

    # ---- 3. How good the pairing is ---------------------------------------------
    # How well each pair agrees with its neighbours' offset: a wrong pair stands out.
    residual = np.abs(offsets - pd.Series(offsets).rolling(9, center=True,
                                                          min_periods=1).median().to_numpy())
    unpaired_markers = sorted(set(range(len(starts))) - set(pairs))
    at_end = bool(unpaired_markers) and min(unpaired_markers) >= len(pairs)
    residual_median = residual_max = None
    if len(residual):
        residual_median = float(np.median(residual) * 1000)
        residual_max = float(np.max(residual) * 1000)

    # ---- 4. Write -------------------------------------------------------------------
    return _write(record, daq, rows, new_starts, new_ends, kept, {
        "made_by": "hba2.legacy.marker_pairing",
        "why": f"{len(starts)} marker pulses for {len(rows)} rows: paired by time, "
               f"not by order",
        "rule": "each row's logged start paired with a marker onset (offset, line fit, "
                "order-preserving alignment); a row with no marker placed from its "
                "logged start and end on the fitted line",
        "rows": len(rows), "markers": int(len(starts)), "paired": len(pairs),
        "rows_placed_from_rig_log": placed,
        "markers_without_row": [int(i) for i in unpaired_markers],
        "markers_without_row_at_end": at_end,
        "pairing_passes": passes,
        "pair_residual_ms_median": residual_median,
        "pair_residual_ms_max": residual_max,
        "pc_to_daq_fit": fit})


def _pair_in_passes(starts, t_cued, max_passes=4):
    """
    Row index -> marker index. Each pass pairs what the last left over, with its own
    offset, so a session whose PC clock stepped is paired on both sides of the step. A
    later pass's pair is kept only if it keeps rows and markers in the same order.

    Returns the pairs, the first pass's PC-to-DAQ line (None if nothing paired) and the
    number of passes made.
    """
    marker_of_row = {}
    first_fit, passes = None, 0
    for _ in range(max_passes):
        free_m = [k for k in range(len(starts)) if k not in marker_of_row.values()]
        free_r = [i for i, t in enumerate(t_cued) if np.isfinite(t) and i not in marker_of_row]
        if len(free_m) < 3 or len(free_r) < 3:
            break
        pairs, fit = legacy_trials._pair([starts[k] for k in free_m],
                                         [t_cued[i] for i in free_r])
        if fit is None or not pairs:
            break
        first_fit = first_fit or fit
        added = 0
        for a, b in pairs.items():
            row, marker = free_r[b], free_m[a]
            before = [m for r, m in marker_of_row.items() if r < row]
            after = [m for r, m in marker_of_row.items() if r > row]
            if (not before or max(before) < marker) and (not after or min(after) > marker):
                marker_of_row[row] = marker
                added += 1
        passes += 1
        if not added:
            break
    return dict(sorted(marker_of_row.items())), first_fit, passes
