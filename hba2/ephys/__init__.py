"""
Spike sorts, their curations, and the clean units that come out of them.
========================================================================

The layers, from the probe to a figure::

    <group>/ephys/<..._mouse>/            the raw recording: Open Ephys, or SpikeGLX. Never touched
    <session>/processed/ephys_sync.csv    the ephys clock against the DAQ's (postprocessing)
    <session>/ephys_sorts/<sort>/         one sort
    ├── sort.json                         what produced it
    ├── sorter/  analyzer/                the sorter's and SpikeInterface's own output
    ├── phy/                              Phy's folder — the only place curation happens
    └── curations/<curation>/             a curation, frozen out of phy/
        ├── curation.json                 who, when, what Phy said, what was exported
        ├── cluster_group.tsv             Phy's labels, line endings fixed
        ├── cluster_info.tsv              Phy's summary of every cluster
        ├── spike_clusters.npy            which cluster every spike is in, after merges
        ├── units.csv                     the good units: one row each       } export
        ├── spikes.parquet                their spikes, on the DAQ clock     }
        ├── waveforms.npz                 their mean waveform, peak channel  }
        └── neuron_names.csv              unit -> neuron, from cross-session matching

and from there ``build_nwb --batch`` puts the good units into each session's NWB, beside
the behaviour, where :func:`~hba2.load_units` and
:func:`~hba2.load_spikes` read them.

**Open Ephys or SpikeGLX, the steps are the same.** Everything from ``phy/`` on reads
Phy's files and ``ephys_sync.csv``, which mean the same thing for both. Only the sync
reads the raw recording, and it tells them apart by the folder
(``postprocessing.sources.spikeglx.is_spikeglx``). A sorter finds a Neuropixels probe the
same way: ``spikeglx.locate(record.ephys).continuous`` is its ``.ap.bin`` (the folder
SpikeInterface's ``read_spikeglx`` takes, with ``stream_id`` its ``.stream``). A sort of
one should write ``sort.json``'s ``probe`` (``description``, ``group``), which
``build_nwb`` takes the probe's description from, and its ``pipeline.recording_index`` as
the probe's index.

**Run it as one step**: :func:`process_ephys_cohort` (``pipeline.py``, or
``python -m hba2.ephys run``) does all of the below for a cohort, each
step only where its output is missing. The steps, each also a function here and a
command of its own:

* :func:`adopt_sort` — move a sort from the pipeline's ``spikesorter_output_<x>/`` into
  ``ephys_sorts/<name>/``. Once per sort; new sorts should land there directly.
* :func:`freeze` — copy what Phy currently says into a named curation. Phy keeps
  working on ``phy/``; a frozen curation never changes.
* :func:`export` — a curation's good units, their metrics and their spikes on the DAQ
  clock, as plain files.
* :func:`write_names` — neuron names from a cross-session matching run, onto each
  session's curation. Every good unit gets one; a unit the run never matched is a
  single-session neuron.

Then ``build_neuron_index`` (``cohort/neuron_index.py``, or ``... ephys neurons``) gathers the
named units into the cohort's ``neurons.csv`` — one row per neuron, generated columns
underscored, any column you add yours to fill — and ``load_units`` joins it on.

**Optotagging** — :func:`tag_cohort` (``optotagging.py``) tests every good unit of a
curation against the laser pulses of its session: the previous pipeline's first-spike
test, on the DAQ clock.

**Only good units are exported.** ``mua`` and ``noise`` stay in Phy and never reach a
file or a figure. In the 2603 curation ``mua`` was not curated and is noise.
"""

from .adopt import adopt_sort
from .curation import freeze
from .export import export
from .names import name_unnamed, write_names
from .optotagging import Tags, tag_cohort, tag_session
from .pipeline import (EphysCohortResult, EphysSessionResult, process_ephys_cohort,
                       process_ephys_session)

__all__ = ["process_ephys_cohort", "process_ephys_session",
           "tag_cohort", "tag_session", "Tags",
           "EphysCohortResult", "EphysSessionResult",
           "adopt_sort", "freeze", "export", "write_names", "name_unnamed"]
