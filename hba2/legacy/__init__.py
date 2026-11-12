"""
Sessions recorded before the controller rewrite, made readable by this package.
==============================================================================

Cohorts from 2024–2025 predate ``-metadata.json``, the rig's trials CSV and the
trial-marker channel, so post-processing cannot run on them. This subpackage does its
job by other means — trials found on the DAQ from cue onsets, the old rig's
``_Phase_<X>_behaviour_data.json`` log paired with them by time — and then writes the
same ``processed/`` files with post-processing's own writers. From there pose, the
session index, NWB building and plotting run unchanged.

    phase_log.py   reads the Phase JSON: header, serial log, scales
    trials.py      finds trials on the DAQ and pairs the log with them
    convert.py     one session: metadata, symlinks, processed/
    run.py         a cohort: convert in parallel, then ports, pose, index, NWB

Every file it adds to a raw session folder is listed in that folder's
``LEGACY_CONVERSION.txt`` and marked ``converted_from``; nothing original is altered.

Each cohort is converted by a short script of its own, outside the package, that calls
``convert_cohort`` with that cohort's settings.
"""

from .convert import convert_session, find_sessions
from .run import RIG_NAMES, convert_cohort, finish_cohort

__all__ = ["convert_session", "find_sessions", "convert_cohort", "finish_cohort",
           "RIG_NAMES"]
