"""
Shared machinery the pipeline builds on.

``clock`` — placing streams on the DAQ clock.

Locating files is not here: that is the cohort's job, and the pipeline is handed the
paths it needs rather than looking for them.
"""

from . import clock

__all__ = ["clock"]
