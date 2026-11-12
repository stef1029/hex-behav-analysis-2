"""
``trials.csv`` — trial windows, and what the tracker itself recorded.

**Deliberately minimal.** Only two kinds of column exist here: the window, taken from the
trial-marker channel, and fields copied unchanged from the rig's own trials CSV. Nothing
in this file is a judgement made by this package.

No derived outcome and no correctness. Those differ by trial type — what counts as
correct in a scales-training block is not what counts in a six-port discrimination — and
they are cheap to compute from ``events.csv`` when the analysis knows which rules apply.

``t_response`` is not one of those. The controller has already decided which port was
chosen; all this does is find when that port's beam broke inside the trial, which puts
an event the controller recorded onto the master clock. That is this stage's whole job,
and it is why ``t_end`` cannot stand in for it: the marker channel can stay high for
seconds after the animal has responded.

``event_log`` is carried the same way: whatever the protocol noted as the trial ran,
copied through unread. The names in it and the keys beside them are the protocol's own
and vary between protocols, so nothing here interprets them — anything derived from
laser stimulation in this package comes from the DAQ pulses, which are what physically
happened.

``correct_port`` and ``chosen_port`` read like a correctness judgement but are not one:
they are what the tracker recorded. Whether a trial *was* correct means comparing them,
or comparing against the sensor events, and which is right depends on the trial type.

Trial start and end are the one exception to "no interpretation": every trial has them
whatever its type, so recording them here saves every analysis re-deriving the same thing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ._result import OutputResult

COLUMNS = (
    "trial", "t_start", "t_end", "t_response",
    "tracker", "sub_tracker", "outcome",
    "correct_port", "chosen_port", "trial_number",
    "stage", "stimulus_log", "event_log", "details_json",
)

MISSING_INT = -1


def _int(value):
    """Missing integers are -1: a NaN would turn the whole column into floats."""
    return MISSING_INT if value is None else int(value)


def _json_text(value):
    """An embedded JSON column, or blank when the rig recorded nothing in it."""
    if value:
        return json.dumps(value)
    return ""


def write(out_dir, rows, starts, ends, touch_times=None, touch_ports=None):
    """
    Pair trial windows with the controller's rows, by order.

    Safe because the controller emits one marker pulse per trial in the same sequence.
    If the counts differ the surplus is dropped and reported rather than guessed at.

    Args:
        touch_times, touch_ports: every beam break in the session, from
            :func:`events.sensor_touches`. Without them ``t_response`` is left empty.
    """
    path = Path(out_dir) / "trials.csv"
    n = min(len(starts), len(rows))

    # ---- 1. When each trial's chosen port was first touched -------------
    # NaN where nothing was. The *first* touch, not the last: animals carry on touching
    # after responding, so the last beam break inside a window is frequently a
    # different port. A trial whose chosen port is ``-1`` never chose anything, and
    # stays empty.
    responses = np.full(n, np.nan)
    if touch_times is not None and len(touch_times) != 0:
        for index in range(n):
            port = rows.rows[index].chosen_port
            if port is None or port < 0:
                continue
            hit = np.flatnonzero((touch_times >= starts[index])
                                 & (touch_times <= ends[index])
                                 & (touch_ports == port))
            if len(hit):
                responses[index] = touch_times[hit[0]]

    warnings = []
    if len(starts) != len(rows):
        warnings.append(
            f"{len(starts)} trial-marker pulses but {len(rows)} CSV rows; "
            f"paired the first {n} by order"
        )

    # ---- 2. Write ---------------------------------------------------------
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for index in range(n):
            row = rows.rows[index]
            response = ""
            if not np.isnan(responses[index]):
                response = f"{responses[index]:.6f}"
            writer.writerow((
                index,
                f"{starts[index]:.6f}",
                f"{ends[index]:.6f}",
                response,
                row.tracker,
                row.trial_type or "default",
                row.outcome,
                _int(row.correct_port),
                _int(row.chosen_port),
                _int(row.trial_number),
                row.stage,
                _json_text(row.stimuli),
                _json_text(row.events),
                _json_text(row.details),
            ))

    return OutputResult(name="trials", path=path, n_rows=n,
                        warnings=tuple(warnings))
