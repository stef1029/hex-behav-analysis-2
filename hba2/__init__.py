"""
hba2 — analysis stack for the v2 hex behaviour rig.

New work, deliberately independent of the older ``hex_behav_analysis`` package, which
stays frozen so existing notebooks keep running.

The pieces:

* :class:`Cohort` — where every file is. Ask it; never build a path yourself.
* :func:`postprocess_session` / :func:`postprocess_cohort` — put a session's streams on
  a single clock. Handed the paths; they scan nothing.
* :func:`build_pose` — head position, bearing and port angles from DeepLabCut output,
  once that has landed. A separate stage because it arrives on its own schedule.
* :func:`build_index` / :func:`read_index` — one row per session, and the labels you
  attach to them. Columns starting with ``_`` are rebuilt; the rest are yours.
* :func:`build_neuron_index` / :func:`read_neuron_index` — the same for neurons, in
  ``neurons.csv``; labels follow a neuron's member units, so renumbering cannot move them.
* :func:`repair_video_timing` — run after post-processing: re-pairs video frames with
  their real times on sessions whose video the old converter damaged.
* :func:`frames_at` — from an event time to the video frame showing it, and how far off
  that frame is. Every event lookup goes through it.
* :func:`build_nwb` — one NWB file per session in a named batch, holding everything a
  figure needs — with a sort and curation, its good units too. The standard entry
  point for plotting.
* :func:`load_sessions` / :func:`qc_trials` — from those files to one table of trials,
  and the trials in it worth using marked. Where the package hands over: everything after
  it is plain pandas, or a plotting library built on top (:mod:`.load`).
* :func:`process_ephys_cohort` — a cohort's spike sorts to named, exported good units
  and ``neurons.csv``, in one call that only does what is missing (:mod:`.ephys`).
* :func:`sync_ephys` — map a session's ephys clock onto its DAQ's on its own
  (post-processing and the ephys pipeline both do it for new sessions).

See README.md and docs/ beside this file.
"""

from .cohort import Cohort, SessionRecord, find_sessions
from .pose.pose import (
    PoseResult,
    SessionPose,
    build_pose,
    build_session_pose,
)
from .pose.frame_lookup import MAX_LAG_S, FrameLookup, frames_at
from .cohort.session_index import IndexResult, build_index, read_index
from .cohort.neuron_index import NeuronIndexResult, build_neuron_index, read_neuron_index
from .postprocessing import (
    CohortResult,
    SessionResult,
    postprocess_cohort,
    postprocess_session,
    sync_ephys,
)
from .nwb.build_nwb import NWBResult, SessionNWB, build_nwb, build_session_nwb
from .ephys import EphysCohortResult, process_ephys_cohort, process_ephys_session
from .postprocessing.repairs.repair_video_timing import RepairResult, repair_video_timing
# Everything in load/, so a notebook needs one import line.
from .load import (
    TrialRates, attach_events, attach_notes, bin_trials, count, count_notes,
    describe_events, describe_neurons, describe_notes, describe_sessions, describe_trials,
    describe_units, first, kept, last, load_events, load_frames, load_notes, load_ports,
    load_sessions, load_spikes, load_units, lump_neurons, matching, neuron_key, note, nth,
    passes, qc_trials, read_session_trials, record)

__version__ = "0.4.0"
__all__ = [
    "Cohort", "SessionRecord", "find_sessions",
    "postprocess_session", "postprocess_cohort", "sync_ephys",
    "SessionResult", "CohortResult",
    "build_pose", "build_session_pose", "SessionPose", "PoseResult",
    "build_index", "read_index", "IndexResult",
    "build_neuron_index", "read_neuron_index", "NeuronIndexResult",
    "repair_video_timing", "RepairResult",
    "build_nwb", "build_session_nwb", "NWBResult", "SessionNWB",
    "process_ephys_cohort", "process_ephys_session", "EphysCohortResult",
    "frames_at", "FrameLookup", "MAX_LAG_S",
    "load_sessions", "load_frames", "load_ports", "read_session_trials",
    "describe_sessions", "describe_trials",
    "qc_trials", "kept", "record", "passes", "matching",
    "load_events", "attach_events", "describe_events", "first", "last", "nth", "count",
    "load_notes", "attach_notes", "describe_notes", "note", "count_notes",
    "load_units", "load_spikes", "bin_trials", "TrialRates", "lump_neurons",
    "describe_units", "describe_neurons", "neuron_key",
]
