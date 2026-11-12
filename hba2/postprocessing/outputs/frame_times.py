"""
``frame_times.csv`` — video frame to timestamp.

Indexed by **row in the AVI**, which is what DeepLabCut produces, not by the camera's own
frame ID. The two differ whenever frames drop: the AVI contains only delivered frames, so
DLC row *N* is the *N*-th delivered frame — ``frame_ids[N]`` — not frame ID *N*. On
sessions where nothing dropped they coincide, which is exactly why a bug here would stay
invisible until the first session that did.

Frame IDs come from ``Tracker_data.json`` rather than the ``_frame_ids_backup.txt``
sidecar: the JSON is the primary record and carries the recording metadata too.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ._result import OutputResult

COLUMNS = ("frame", "frame_id", "t")
#: A session's pulse and frame counts differ by one or two at the ends; more is a fault.
EXTRA_PULSES_OK = 3


def write(out_dir, placement, camera):
    """Write one row per delivered frame."""
    path = Path(out_dir) / "frame_times.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(
            (row, int(frame_id), "" if np.isnan(t) else f"{t:.6f}")
            for row, (frame_id, t) in enumerate(zip(camera.frame_ids, placement.times))
        )

    warnings = ()
    if placement.n_unmatched:
        warnings = (
            f"{placement.n_unmatched} of {placement.n_total} frames have no camera "
            f"pulse and are untimed",
        )
    if camera.n_dropped:
        warnings += (f"camera dropped {camera.n_dropped} frames during acquisition",)
    if placement.n_skipped:
        warnings += (f"{placement.n_skipped} camera pulses before the recorder's start gap "
                     f"ignored (camera was already exposing when the DAQ started)",)
    # The reverse of untimed frames, and the easy one to miss: pulses the frame IDs do
    # not account for. If they came before frame 0, every frame is timed
    # that many pulses early — a constant video offset.
    extra = placement.n_pulses - (int(camera.frame_ids.max()) + 1) if camera.n_frames else 0
    if extra > EXTRA_PULSES_OK:
        warnings += (f"{extra} more camera pulses than frame IDs: if they came before "
                     f"frame 0 the video is {extra} frames early",)

    return OutputResult(name="frame_times", path=path, n_rows=placement.n_total,
                        n_unmatched=placement.n_unmatched, warnings=warnings)
