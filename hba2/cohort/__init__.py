"""
What sessions exist, where their files are, and the labels you give them.
=========================================================================

* :mod:`.cohort` — :class:`Cohort`, every session and file in a cohort folder. Ask it;
  never build a path yourself
* :mod:`.session_index` — ``sessions.csv``, one row per session: the facts read from the
  files, and the columns that are yours to fill in
* :mod:`.neuron_index` — ``neurons.csv``, the same for neurons
"""

from .cohort import (
    BATCHES_DIR, CURATION, CURATIONS_DIR, RECIPE, REQUIRED, SORT, SORTS_DIR, Cohort,
    SessionRecord, batch_cohort, find_sessions, is_session, model_for, scan_session)
from .session_index import IndexResult, build_index, index_path, read_index
from .neuron_index import NeuronIndexResult, build_neuron_index, read_neuron_index

__all__ = [
    "Cohort", "SessionRecord", "find_sessions", "is_session", "scan_session",
    "batch_cohort", "model_for",
    "REQUIRED", "BATCHES_DIR", "RECIPE", "SORT", "SORTS_DIR", "CURATION", "CURATIONS_DIR",
    "build_index", "read_index", "index_path", "IndexResult",
    "build_neuron_index", "read_neuron_index", "NeuronIndexResult",
]
