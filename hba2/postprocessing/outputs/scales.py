"""
``scales.csv`` — platform weight, and what the rig made of it, on the clock.

The rig's scales CSV has a column called ``message_id``, which is **not** a DAQ message
ID: it counts readings from zero, while real DAQ message IDs run into the tens of
millions. Treating it as a DAQ ID matches under 1% of readings; treating it as an index
into the ``SCALES`` pulse train matches all of them. It is the same mechanism as camera
frame IDs, which is why one function in :mod:`~.clock` handles both.

Stillness travels as columns here rather than as a file of its own. Every other output
split off because it is on a different clock — that is the whole reason ``notes.csv``
exists — but the rig scores stillness *per weight reading*, so these values share the
scales' rows and are placed on the DAQ clock by the same pulse match. A separate file
would need its own alignment and could then disagree with the readings it came from.

They are carried, not computed. A protocol may have waited on ``still_for``, which makes
these a record of what the session actually did rather than an interpretation that can be
redone — re-scoring later with a different model or threshold answers a different
question. Which model produced them is in the session's ``_scales_stillness.json``, and
is folded into ``session.json``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ._result import OutputResult

BASE_COLUMNS = ("t", "weight_g", "reading")

#: Appended only for a session the rig scored. Absent columns beat a block of
#: nulls: `has_stillness` on the source says so once, where a reader will see
#: it, instead of leaving it to be inferred from the data.
STILLNESS_COLUMNS = (
    "p_still", "occupied", "state", "still_for", "baseline_g", "still_threshold",
)


def _number(values, index, places):
    """Render one value, blank for absent or NaN.

    Blank rather than zero throughout: a zero ``p_still`` reads as "certainly
    moving" and a zero ``still_for`` as "on the platform but not settled", so
    neither can stand in for a reading the rig had no opinion about (rule 7 —
    nothing silently wrong).
    """
    if values is None:
        return ""
    value = values[index]
    return "" if np.isnan(value) else f"{value:.{places}f}"


def write(out_dir, placement, scales):
    """Write one row per weight reading."""
    path = Path(out_dir) / "scales.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Scales that logged no reading counter are numbered in file order.
    if scales.message_ids is not None:
        readings = scales.message_ids
    else:
        readings = np.arange(scales.n_readings)

    columns = BASE_COLUMNS
    if scales.has_stillness:
        columns = BASE_COLUMNS + STILLNESS_COLUMNS

    occupied = scales.occupied
    state = scales.state

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i, (t, w, r) in enumerate(zip(placement.times, scales.weights, readings)):
            stamp = "" if np.isnan(t) else f"{t:.6f}"
            line = (stamp, f"{w:.4f}", int(r))

            # The stillness fields, blank where the rig had no value.
            if scales.has_stillness:
                occupied_text = ""
                if occupied is not None:
                    occupied_text = "1" if occupied[i] else "0"
                state_text = ""
                if state is not None:
                    state_text = str(state[i])
                line += (
                    _number(scales.p_still, i, 4),
                    occupied_text,
                    state_text,
                    _number(scales.still_for, i, 3),
                    _number(scales.baseline, i, 4),
                    _number(scales.still_threshold, i, 4),
                )
            writer.writerow(line)

    warnings = ()
    if placement.n_unmatched:
        warnings = (
            f"{placement.n_unmatched} of {placement.n_total} scales readings "
            f"have no pulse and are untimed",
        )
    return OutputResult(name="scales", path=path, n_rows=placement.n_total,
                        n_unmatched=placement.n_unmatched, warnings=warnings)
