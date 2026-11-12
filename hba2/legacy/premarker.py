"""
Trial windows for current-controller sessions recorded before the trial marker.
==============================================================================

The trial-marker channel (``DAQ_LINK1``) only started pulsing in April 2026 (later on
rig 4). Sessions before that have the rig's trials CSV but no marker, so
post-processing pairs the CSV with zero windows and writes no trials. Before the marker
the trials were run the same way they were analysed: **a trial starts when its LED
comes on and ends at the response — the first touch — or at the timeout.** This
recovers exactly that, on the DAQ:

* **start** — the cue onset (``trials.detect``'s cues: LED rising edges, merged when
  several light together);
* **end** — the first new beam break after it, before the rig logged the trial
  finished; with none, the rig's logged end on the fitted PC-to-DAQ line.

The rig writes each row as its trial *ends*, so ``time_since_start_s`` is the end and
``time_since_start_s - trial_duration_s`` the start.

Each CSV row is paired with a cue by time, exactly as the legacy Phase log is, and kept
**verbatim** — tracker, sub-tracker, stage, ports, outcome, stimulus log — just as
post-processing keeps it for sessions with a marker. Only the windows are new. A row no
cue paired with is placed where the fitted line puts it and flagged.

It rewrites ``events.csv`` (whose ``in_trial`` / ``trial`` depend on the windows),
``trials.csv`` and ``notes.csv``, and records how in ``session.json`` under
``trial_windows``. Everything else post-processing wrote is left alone. Only sessions
with no marker pulses and a non-empty trials CSV are touched.

    recover_cohort    every session that needs it
    needs_windows     whether one does
    recover_session   one session, cue onsets paired with the CSV rows
    _reward_only      one reward-training session, valve openings paired instead
    _write            the rewritten files (shared with marker_pairing.py)
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..cohort import Cohort
from ..postprocessing import outputs
from ..postprocessing.pipeline import TRIAL_MARKER, channel_role
from ..postprocessing.sources import daq as daq_source, trials_csv
from ..postprocessing.sources.trials_csv import TrialRows
from . import trials as legacy_trials

#: How long after the rig's logged end a touch may still be the response.
END_SLACK_S = legacy_trials.LOG_DONE_SLACK_S


def recover_cohort(base, log=print):
    """
    Give trial windows to every session in the cohort at ``base`` (a Path or a string)
    that needs them. Returns ``{"done": [session IDs], "failed": {session ID: error}}``.
    """
    cohort = Cohort(base)
    done, failed = [], {}
    for record in cohort:
        if not needs_windows(record):
            continue
        try:
            summary = recover_session(record)
            done.append(record.session_id)
            line = f"  {record.session_id}: {summary['rows']} rows, {summary['paired']} paired"
            if "cues" not in summary:
                line += " (reward training, by valve)"
            log(line)
        except Exception as exc:                    # noqa: BLE001 — reported per session
            failed[record.session_id] = f"{type(exc).__name__}: {exc}"
            log(f"  {record.session_id}: FAILED — {type(exc).__name__}: {exc}")
    log(f"{Path(base).name}: {len(done)} session(s) given trial windows, "
        f"{len(failed)} failed")
    return {"done": done, "failed": failed}


def needs_windows(record):
    """Processed, the rig wrote trials, and none were paired for want of a marker."""
    if not record.is_processed or record.raw.get("trials_csv") is None:
        return False
    try:
        n_rows = len(pd.read_csv(record.path("trials_csv")))
        n_done = len(pd.read_csv(record.path("trials")))
    except Exception:                               # noqa: BLE001
        return False
    return n_rows > 0 and n_done == 0


def _start(row):
    """A trials-CSV row's start on the PC clock: its end minus its duration; NaN if unknown."""
    if row.software_time is None:
        return np.nan
    return row.software_time - (row.software_duration or 0.0)


def recover_session(record):
    """
    Rewrite one session's windows (a :class:`~..cohort.SessionRecord`). Returns what
    was done, as recorded in ``session.json``. Raises if it has marker pulses after all.
    """
    daq = daq_source.read(record.path("daq"))
    if daq.has(TRIAL_MARKER) and len(daq.rising_edges(TRIAL_MARKER)):
        raise ValueError("the session has trial-marker pulses — post-processing's job")
    rows = trials_csv.read(record.path("trials_csv"))

    # ---- 1. Cue onsets paired with the rows ---------------------------------
    cues, _ = legacy_trials._cues(daq, phase="")
    onsets = [c.t_on for c in cues]
    # The rig writes a row when the trial *ends*: time_since_start_s is the end, and
    # the start is that minus trial_duration_s. (Checked: start against LED onset holds
    # to a few tens of milliseconds all session; the end-as-start reading drifts by
    # seconds.)
    t_cued = [_start(r) for r in rows]
    t_done = [r.software_time if r.software_time is not None else np.nan for r in rows]
    valid = [i for i, t in enumerate(t_cued) if np.isfinite(t)]
    pairs, fit = legacy_trials._pair(onsets, [t_cued[i] for i in valid])
    if fit is None and not onsets:
        return _reward_only(record, daq, rows, t_cued, t_done)
    if fit is None:
        raise ValueError("the trials CSV could not be lined up with any cue onsets")
    cue_of_row = {valid[j]: i for i, j in pairs.items()}

    # ---- 2. Each row's window -------------------------------------------------
    # Start at its cue, or where the fitted line puts its logged start; end at the
    # first beam break after it, before the next cue and the logged end (plus slack).
    touch_t, touch_port = legacy_trials._touches(daq)
    end_of_session = float(daq.timestamps[-1])

    starts, ends, kept, placed = [], [], [], 0
    for index, row in enumerate(rows.rows):
        if index in cue_of_row:
            start = onsets[cue_of_row[index]]
            source = "cue"
        elif np.isfinite(t_cued[index]):
            start = fit[0] * t_cued[index] + fit[1]
            source = "rig_log"
            placed += 1
        else:
            continue
        logged_end = None
        if np.isfinite(t_done[index]):
            logged_end = fit[0] * t_done[index] + fit[1]
        later = [o for o in onsets if o > start + legacy_trials.MERGE_S]
        next_cue = later[0] if later else end_of_session
        log_limit = logged_end + END_SLACK_S if logged_end is not None else np.inf
        limit = min(next_cue, log_limit)
        hit = np.flatnonzero((touch_t > start) & (touch_t < limit))
        if len(hit):
            end = float(touch_t[hit[0]])
        elif logged_end is not None:
            end = float(min(logged_end, limit))
        else:
            end = float(limit)
        starts.append(start)
        ends.append(end)
        kept.append(replace(row, index=len(kept),
                            details={**(row.details or {}), "window_start": source}))

    # ---- 3. Write -----------------------------------------------------------------
    return _write(record, daq, rows, starts, ends, kept, {
        "rule": "start at the cue onset; end at the first beam break after it, else "
                "at the rig's logged end",
        "rows": len(rows), "cues": len(onsets), "paired": len(pairs),
        "placed_from_rig_log": placed, "pc_to_daq_fit": fit})


def _reward_only(record, daq, rows, t_cued, t_done):
    """
    Reward training: no cue, the trial ends when the valve opens on the touch.

    The valve openings are the only trial events on the DAQ, so they are paired with
    the rows' logged *ends*; each trial ends at its valve and starts at the rig's logged
    start, placed on the line that pairing fits.
    """
    openings = [daq.rising_edges(f"VALVE{p}") for p in range(6) if daq.has(f"VALVE{p}")]
    if not openings:
        openings = [np.empty(0)]
    valves = np.sort(np.concatenate(openings))
    valid = [i for i, t in enumerate(t_done) if np.isfinite(t)]
    pairs, fit = legacy_trials._pair(list(valves), [t_done[i] for i in valid])
    if fit is None:
        raise ValueError("no cues, and the valve openings do not line up with the rows")
    valve_of_row = {valid[j]: i for i, j in pairs.items()}

    starts, ends, kept = [], [], []
    for index, row in enumerate(rows.rows):
        if not np.isfinite(t_cued[index]):
            continue
        start = fit[0] * t_cued[index] + fit[1]
        if index in valve_of_row:
            end = float(valves[valve_of_row[index]])
            window_end = "valve"
        else:
            end = fit[0] * t_done[index] + fit[1]
            window_end = "rig_log"
        starts.append(start)
        ends.append(max(end, start))
        kept.append(replace(row, index=len(kept), details={
            **(row.details or {}), "window_start": "rig_log", "window_end": window_end}))
    return _write(record, daq, rows, starts, ends, kept, {
        "rule": "reward training, no cue: end at the valve opening paired with the "
                "row's logged end; start at the rig's logged start on the fitted line",
        "rows": len(rows), "valves": int(len(valves)), "paired": len(pairs),
        "pc_to_daq_fit": fit})


def _write(record, daq, rows, starts, ends, kept, how):
    """
    Rewrite ``events.csv``, ``trials.csv`` and ``notes.csv`` with the new windows, in
    start order, and record ``how`` (a dict) in ``session.json`` under ``trial_windows``.

    ``rows`` is the session's TrialRows; ``kept`` the rows given a window, one per start.
    Returns the summary written, with the files' descriptions.
    """
    starts, ends = np.asarray(starts), np.asarray(ends)
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    kept = [replace(kept[i], index=n) for n, i in enumerate(order)]
    table = TrialRows(path=rows.path, rows=kept, columns=rows.columns,
                      renamed_columns=rows.renamed_columns)

    out = record.out_dir
    behaviour = [n for n in daq.channel_names
                 if channel_role(n) in ("behaviour", "trial_marker")]
    touches = outputs.events.sensor_touches(daq, behaviour)
    written = [outputs.events.write(out, daq, behaviour, starts, ends),
               outputs.trials.write(out, table, starts, ends, *touches),
               outputs.notes.write(out, table, len(starts))]

    summary = {
        "made_by": "hba2.legacy.premarker",
        "made_at": datetime.now().isoformat(timespec="seconds"),
        "why": "no trial-marker pulses: the session predates the marker",
        **how,
    }
    path = record.path("session_json")
    document = json.loads(path.read_text())
    document["trial_windows"] = summary
    path.write_text(json.dumps(document, indent=2, default=str))
    return {**summary, "written": [w.describe() for w in written]}
