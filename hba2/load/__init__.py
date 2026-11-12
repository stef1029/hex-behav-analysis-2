"""
From the NWB files to one table of trials.
==========================================

The hand-off point of the package: everything before this builds the files, everything
after it is yours. A notebook starts the same way every time::

    from hba2 import load_sessions, qc_trials

    trials = load_sessions("/path/to/cohort", include=True)
    trials = qc_trials(trials, min_ear_likelihood=0.6, max_abs_frame_lag_s=0.1)

One row per trial, every session stacked, your ``sessions.csv`` labels as columns.
Nothing is dropped on the way in; ``qc_trials`` marks what to leave out in ``keep``.

* :mod:`sessions` — :func:`load_sessions`, the trials; :func:`load_frames`, the head
  frame by frame through them; :func:`load_ports`; ``describe_sessions`` /
  ``describe_trials``, what there is
* :mod:`qc` — :func:`qc_trials`, one sweep marking the trials that smell bad;
  :func:`kept`, the ones that passed
* :mod:`rules` — the one vocabulary (``min_``, ``max_``, ``not_``, lists) for saying
  which rows you mean, shared by session selection, QC and anything built on top
* :mod:`events` — the moment inside a trial a figure is about, off the hardware
* :mod:`notes` — what the protocol said it did, on its own clock
* :mod:`spikes` — the good units in those sessions, their spikes around a moment of each
  trial, and every neuron's rate binned through every trial
"""

from . import events, notes, qc, rules, sessions, spikes
from .events import attach_events, count, describe_events, first, last, load_events, nth
from .notes import attach_notes, count_notes, describe_notes, load_notes, note
from .qc import kept, qc_trials, record
from .rules import matching, passes
from .sessions import (
    describe_sessions, describe_trials, load_frames, load_ports, load_sessions,
    read_session_trials)
from .spikes import (
    TrialRates, bin_trials, describe_neurons, describe_units, load_spikes, load_units,
    lump_neurons, neuron_key)

__all__ = [
    "events", "notes", "qc", "rules", "sessions", "spikes",
    "load_sessions", "load_frames", "load_ports", "read_session_trials",
    "describe_sessions", "describe_trials",
    "qc_trials", "kept", "record", "passes", "matching",
    "load_events", "attach_events", "describe_events", "first", "last", "nth", "count",
    "load_notes", "attach_notes", "describe_notes", "note", "count_notes",
    "load_units", "load_spikes", "bin_trials", "TrialRates", "lump_neurons",
    "describe_units", "describe_neurons", "neuron_key",
]
