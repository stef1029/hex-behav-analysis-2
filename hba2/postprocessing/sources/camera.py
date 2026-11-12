"""
Reader for ``<session>_Tracker_data.json``.

Camera metadata plus the ID of every frame the camera actually delivered. It
does **not** contain timing: each frame ID corresponds to a rising edge on the
DAQ ``CAMERA`` channel, and pairing the two is what puts frames on the clock.
That pairing is alignment and lives in
:mod:`hba2.postprocessing.utils.clock`.

Nothing here opens the video. The video is compressed hours later and is only
needed to cross-check that compression kept every frame — so post-processing
never waits on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

TIMESTAMP_FORMAT = "%y%m%d_%H%M%S"


def _parse_timestamp(value) -> datetime | None:
    """A ``YYMMDD_HHMMSS`` timestamp, or None if absent or unparseable."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value), TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read(path):
    """
    Parse a tracker JSON (a Path or a string) into :class:`CameraData`.

    Raises:
        ValueError: if ``frame_IDs`` is absent. Without it no frame can be
            placed on the clock, and video-based analysis is impossible.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc

    if "frame_IDs" not in document:
        raise ValueError(f"{path.name} has no 'frame_IDs'")

    frame_ids = np.asarray(document["frame_IDs"], dtype=np.int64)

    try:
        frame_rate = float(document.get("frame_rate"))
    except (TypeError, ValueError):
        frame_rate = None

    return CameraData(
        path=path,
        frame_ids=frame_ids,
        frame_rate=frame_rate,
        width=_as_int(document.get("image_width")),
        height=_as_int(document.get("image_height")),
        pixel_format=document.get("pixel_format"),
        start_time=_parse_timestamp(document.get("start_time")),
        end_time=_parse_timestamp(document.get("end_time")),
    )


@dataclass(frozen=True)
class CameraData:
    """What the camera recorded about itself."""

    path: Path
    frame_ids: np.ndarray       #: IDs of delivered frames, ascending
    frame_rate: float | None
    width: int | None
    height: int | None
    pixel_format: str | None
    start_time: datetime | None
    end_time: datetime | None

    @property
    def n_frames(self):
        return len(self.frame_ids)

    @property
    def n_dropped(self):
        """
        Frames the camera skipped, inferred from gaps in the ID sequence.

        A camera that delivered IDs 0..N without gaps dropped nothing; any
        shortfall against ``max(id) + 1`` was dropped at acquisition time.
        """
        if self.n_frames == 0:
            return 0
        return int(self.frame_ids.max()) + 1 - self.n_frames

    def describe(self):
        size = f"{self.width}x{self.height}" if self.width else "?"
        return (
            f"{self.path.name}: {self.n_frames:,} frames ({size} "
            f"{self.pixel_format}) @ {self.frame_rate}fps, {self.n_dropped} dropped"
        )
