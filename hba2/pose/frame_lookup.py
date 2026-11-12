"""
From a moment in time to the video frame that shows it — and how far off that frame is.
======================================================================================

Every analysis that asks *what did the animal look like when X happened* goes through
here: the head at trial start, the head when a beam broke, a frame for a picture.

The answer is always two numbers, never one::

    lookup = frames_at(frame_times, event_times, n_frames=dlc.n_frames)
    lookup.frame    # the first video frame exposed at or after each event
    lookup.lag_s    # that frame's own time minus the event's — how late the image is

**The lag is the whole point.** Normally it is under one frame period (0.033 s at 30 fps).
When the camera dropped frames, or the compressed video lost some, the nearest image may
be a long way from the event, and a lookup that only returned a frame would hand back
that distant image looking exactly as trustworthy as any other. With the lag beside it,
the gap is visible and filtering on it is one line::

    usable = np.abs(lookup.lag_s) <= MAX_LAG_S

The threshold is applied by whoever consumes the lookup, not here, so nothing is thrown
away at build time and changing your mind about it is a filter rather than a rebuild.

At-or-after rather than nearest, so a frame never predates the event it is meant to
show. The one exception is an event after the last frame of the video: it gets the last
frame and a **negative** lag, rather than being clamped to that frame silently.

The lag says nothing about whether ``frame_times.csv`` itself pairs frames with the right
times. A video damaged by the old converter needs ``postprocessing/repairs/repair_video_timing.py``
first; on an unrepaired file the lags look fine and the images are still from the wrong
moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The default for "this frame is close enough to the event". Three frames at 30 fps:
#: 80% of camera-drop gaps are three frames or fewer and get bridged; the rest hold most
#: of the missing frames and do not. On the cohort it was tuned on it sets aside about
#: 0.4% of trial-start and response lookups.
MAX_LAG_S = 0.1


def frames_at(frame_times, times, n_frames=None):
    """
    The first video frame at or after each moment, and how late it is, as a
    :class:`FrameLookup`.

    Args:
        frame_times: ``frame_times.csv`` as read — columns ``frame`` and ``t``.
        times: event times on the session clock. NaN means "no event" (a trial with no
            response) and comes back as frame -1.
        n_frames: how many frames the video actually has — DeepLabCut's row count, or
            None for no limit. Only frames below this are candidates, so a row in
            ``frame_times.csv`` that describes a frame the video does not contain can
            never be returned.
    """
    times = np.asarray(times, dtype=float)
    frame = np.full(times.shape, -1, dtype=int)
    lag = np.full(times.shape, np.nan)

    candidates = frame_times.dropna(subset=["t"])
    if n_frames is not None:
        candidates = candidates[candidates["frame"] < n_frames]
    stamps = candidates["t"].to_numpy(dtype=float)
    frames = candidates["frame"].to_numpy(dtype=int)

    asked = np.isfinite(times)
    if len(stamps) == 0 or not asked.any():
        return FrameLookup(frame=frame, lag_s=lag)

    # Past the last frame, the last frame — and the negative lag says so.
    index = np.searchsorted(stamps, times[asked]).clip(0, len(stamps) - 1)
    frame[asked] = frames[index]
    lag[asked] = stamps[index] - times[asked]
    return FrameLookup(frame=frame, lag_s=lag)


@dataclass(frozen=True)
class FrameLookup:
    """One lookup per event. ``frame`` is -1 and ``lag_s`` NaN where there was no event."""

    frame: np.ndarray    # int
    lag_s: np.ndarray    # float, seconds; positive means the image is after the event

    def usable(self, max_lag_s=MAX_LAG_S):
        """True where there is a frame and it is within ``max_lag_s`` of the event."""
        return (self.frame >= 0) & (np.abs(self.lag_s) <= max_lag_s)
