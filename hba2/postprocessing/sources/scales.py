"""
Reader for ``<session>_scales_data.csv``.

The platform scales, sampled continuously. Columns:

    timestamp_s   the scales server's own clock  (NOT the DAQ clock)
    weight_g      grams
    message_id    the DAQ message in flight when the reading was taken

``message_id`` is the important one: it is what lets every weight reading be
placed on the DAQ clock, by matching against rising edges of the ``SCALES``
channel. That matching is alignment, so it lives in
:mod:`hba2.postprocessing.utils.clock` — this module only parses.

Older wireless scales wrote no ``message_id``. Those readings can only ever sit
on their own clock, and :attr:`ScalesData.can_sync` says so up front rather than
letting it be discovered later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: The ScalesData fields that hold the rig's stillness record.
_STILLNESS_FIELDS = ("p_still", "occupied", "state", "still_for",
                     "baseline", "still_threshold")


def _number_column(frame, column) -> np.ndarray | None:
    """A numeric column as float64, or None if the CSV does not have it."""
    if column not in frame.columns:
        return None
    # Blank stays NaN rather than becoming zero: a zero score means
    # "certainly moving", which is not what a blank meant.
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def read(path):
    """
    Parse a scales CSV (a Path or a string) into :class:`ScalesData`.

    An empty or header-only file is returned as an empty ``ScalesData`` rather
    than raising: a session can legitimately run with the scales disabled.
    """
    path = Path(path)
    frame = pd.read_csv(path)

    if frame.empty:
        empty = np.empty(0, dtype=np.float64)
        return ScalesData(path=path, timestamps=empty, weights=empty, message_ids=None)

    # ---- 1. The readings ----------------------------------------------
    missing = {"timestamp_s", "weight_g"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required column(s): {sorted(missing)}"
        )

    message_ids = None
    if "message_id" in frame.columns and frame["message_id"].notna().any():
        message_ids = frame["message_id"].to_numpy(dtype=np.int64, na_value=-1)

    timestamps = frame["timestamp_s"].to_numpy(dtype=np.float64)
    weights = frame["weight_g"].to_numpy(dtype=np.float64)

    # ---- 2. The rig's stillness, or its absence as None -----------------
    # Presence is judged on ``p_still`` having at least one value. A file with
    # the columns but nothing in them — stillness configured but never able to
    # score — is the same situation as a file without them, and is reported the
    # same way rather than as a wall of nulls.
    stillness = {name: None for name in _STILLNESS_FIELDS}

    if "p_still" in frame.columns and frame["p_still"].notna().any():
        occupied = None
        if "occupied" in frame.columns:
            occupied = pd.to_numeric(frame["occupied"], errors="coerce") == 1
            occupied = occupied.to_numpy(dtype=bool)

        state = None
        if "state" in frame.columns:
            state = frame["state"].fillna("unknown").astype(str).to_numpy()

        stillness = {
            "p_still": _number_column(frame, "p_still"),
            "occupied": occupied,
            "state": state,
            "still_for": _number_column(frame, "still_for"),
            "baseline": _number_column(frame, "baseline_g"),
            "still_threshold": _number_column(frame, "still_threshold"),
        }

    return ScalesData(
        path=path,
        timestamps=timestamps,
        weights=weights,
        message_ids=message_ids,
        **stillness,
    )


@dataclass(frozen=True)
class ScalesData:
    """Weight readings as recorded, before alignment."""

    path: Path
    timestamps: np.ndarray      #: scales-server clock, seconds
    weights: np.ndarray         #: grams
    message_ids: np.ndarray | None  #: DAQ message IDs, None for wireless scales

    # What the rig's stillness estimator believed at each reading, or None for
    # a session recorded before stillness existed or with it switched off.
    # These are read rather than recomputed: a protocol may have gated on them,
    # which makes them a record of what happened rather than a judgement that
    # can be made again. Re-scoring later with a changed model or threshold
    # would answer a different question — see the session's
    # `_scales_stillness.json` for which model produced these.
    p_still: np.ndarray | None = None        #: score in [0, 1], NaN where undefined
    occupied: np.ndarray | None = None       #: bool, is anything on the platform
    state: np.ndarray | None = None          #: off | moving | still | unknown
    still_for: np.ndarray | None = None      #: s of unbroken stillness, NaN when off
    baseline: np.ndarray | None = None       #: empty-platform reading, grams
    still_threshold: np.ndarray | None = None  #: score compared against, per reading

    @property
    def n_readings(self):
        return len(self.weights)

    @property
    def can_sync(self):
        """True if these readings carry the message IDs needed for DAQ alignment."""
        return self.message_ids is not None

    @property
    def has_stillness(self):
        """True if the rig scored stillness alongside these readings.

        Stated up front, like :attr:`can_sync`, so a caller finds out before
        building anything on it rather than discovering a column of nulls.
        """
        return self.p_still is not None

    def describe(self):
        if self.n_readings == 0:
            return f"{self.path.name}: empty"
        if self.can_sync:
            kind = "wired (syncable)"
        else:
            kind = "wireless (no message IDs)"
        return (
            f"{self.path.name}: {self.n_readings:,} readings, {kind}, "
            f"{self.weights.min():.1f}..{self.weights.max():.1f} g"
        )
