"""
Post-processing: raw rig output -> a set of files on one clock.

Runs once per session, automatically, shortly after recording. Its whole job is
synchronisation; interpretation belongs to analysis code reading the outputs.

``pipeline.py`` is the readable spine — start there.
"""

from .pipeline import (
    CohortResult,
    SessionResult,
    postprocess_cohort,
    postprocess_session,
    sync_ephys,
)

__all__ = ["postprocess_session", "postprocess_cohort", "sync_ephys",
           "SessionResult", "CohortResult"]
