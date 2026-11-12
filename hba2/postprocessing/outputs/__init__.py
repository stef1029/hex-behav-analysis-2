"""One module per output file. Every ``write`` returns an :class:`OutputResult`."""

from . import ephys_sync, events, frame_times, notes, pulses, scales, session, trials
from ._result import OutputResult

__all__ = ["session", "events", "trials", "notes", "frame_times", "scales",
           "pulses", "ephys_sync", "OutputResult"]
