"""
The ephys pipeline: a cohort's sorts to clean, named units, in one call.
=======================================================================

**This file is the step**, as ``postprocessing/pipeline.py`` is for behaviour: read
:func:`process_ephys_cohort` and :func:`process_ephys_session` top to bottom and you have
seen everything it does. The work is in the step modules beside it.

Per session, in order, each step only when its output is missing::

    1. adopt    ephys_sorts/<sort>/             from the sorter's spikesorter_output_<x>/
    2. freeze   ephys_sorts/<sort>/curations/<c>/   what Phy says now — only when asked
    3. sync     processed/ephys_sync.csv        the ephys clock against the DAQ's
    4. export   units.csv, spikes.parquet, waveforms.npz   redone when its inputs changed

Sync comes after freeze so that only a session whose units will be used pays for it.
Every sorted session is still adopted, so a session curated later needs only its freeze.

then across the cohort::

    5. names    neuron_names.csv in every curation — from a matching run if given,
                else single-session names for sessions not yet named
    6. neurons  <cohort>/neurons.csv — rebuilt, your columns kept

So running it again is always safe and does only what is new. It stops short of the
NWB files: a batch is a deliberate, named choice (which pose model, which curation),
made with ``build_nwb --batch`` when you want one.

**Freezing is the one step never done by default.** A freeze says "Phy's labels for this
session are a curation called <c>", and only you know which sessions were actually
curated: a session nobody opened in Phy still has the sorter's own labels, and freezing
those under a curated-sounding name would quietly put unvetted units in a figure. So
``freeze`` names the sessions to freeze, or is ``True`` to mean all of them.

A session that cannot go on — no recording, no sort, not frozen — stops there and says
which; the rest carry on. Failures are reported, never raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..cohort import Cohort, scan_session
from ..cohort.neuron_index import NeuronIndexResult, build_neuron_index
from ..postprocessing.pipeline import sync_ephys
from .adopt import DEFAULT_NAME, DEFAULT_OLD, adopt_sort
from .curation import freeze as freeze_curation
from .export import export
from .names import name_unnamed, write_names


# =====================================================================
# A whole cohort
# =====================================================================

def _say_nothing(message):
    """What process_ephys_cohort reports to when nobody is listening."""


def process_ephys_cohort(cohort, *, curation, sort=DEFAULT_NAME, freeze=False,
                         tracks=None, matching=None, level=None, sync=True,
                         old=DEFAULT_OLD, only=None, reexport=False, log_fn=None):
    """
    Every ephys session of a cohort, as far as named, exported good units and
    ``neurons.csv``.

    Args:
        cohort: a :class:`Cohort`, or a path (Path or string) to build one from.
        curation: which curation the units come from, e.g. ``2026-09_manual``.
        sort: the sort's name under ``ephys_sorts/``.
        freeze: the session IDs (any iterable of strings) whose current Phy labels are
            this curation, to freeze where it does not exist yet; ``True`` for every
            session. Default: freeze nothing — use curations already frozen.
        tracks: a cross-session matching table (``tracks_all.csv``), as a Path or a
            string. With it, every session is named from it, and ``matching`` and
            ``level`` say what it was. Without it, sessions already named keep their
            names and the rest get single-session ones.
        sync: map ephys clocks that have not been mapped. Minutes per session.
        old: the sorter's own folder name, adopted where found.
        only: just these session IDs (any iterable of strings).
        reexport: export every session again, even where up to date.
        log_fn: called with a line (a string) per session as it goes. Without it the run
            is silent until it returns — and syncing alone is minutes a session.

    Returns an :class:`EphysCohortResult`.
    """
    log = log_fn or _say_nothing
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    if tracks is not None and not (matching and level):
        raise ValueError("a matching table needs matching= and level= to say what it is")
    to_freeze = freeze if isinstance(freeze, bool) else set(freeze)
    result = EphysCohortResult(base=cohort.base, sort=sort, curation=curation)

    # ---- Which sessions ------------------------------------------------
    # A session is an ephys one if it has a recording, a sort, or the sorter's own
    # folder still waiting to be adopted.
    wanted = set(only) if only else None
    records = []
    for r in cohort.with_files():
        if wanted is not None and r.session_id not in wanted:
            continue
        if r.has_ephys or r.has_sort or (r.directory / old).is_dir():
            records.append(r)
    log(f"{cohort.base.name}: {len(records)} session(s) with ephys, "
        f"units from {sort}/{curation}")
    if isinstance(to_freeze, set):
        unknown = sorted(to_freeze - {r.session_id for r in cohort})
        if unknown:
            log(f"  {len(unknown)} session(s) asked to freeze are not in this cohort "
                f"(e.g. {unknown[0]})")

    # ---- 1. to 4. Each session in turn ---------------------------------
    for number, record in enumerate(records, start=1):
        if isinstance(to_freeze, bool):
            freeze_here = to_freeze
        else:
            freeze_here = record.session_id in to_freeze
        outcome = process_ephys_session(record, sort=sort, curation=curation,
                                        freeze=freeze_here, sync=sync, old=old,
                                        reexport=reexport)
        result.results.append(outcome)

        line = f"  [{number}/{len(records)}] {record.session_id}: "
        if outcome.error:
            line += f"FAILED — {outcome.error}"
        elif outcome.stopped:
            line += f"stopped — {outcome.stopped}"
        elif outcome.done:
            line += f"{outcome.units} units ({', '.join(outcome.done)})"
        else:
            line += f"{outcome.units} units (up to date)"
        log(line)

    # ---- 5. NAMES and 6. NEURONS — across the cohort ------------------
    try:
        cohort.load()
        if tracks is not None:
            result.names = write_names(cohort, tracks, sort=sort, curation=curation,
                                       matching=matching, level=level)
        else:
            result.names = name_unnamed(cohort, sort=sort, curation=curation)
        cohort.load()
        result.neurons = build_neuron_index(cohort, sort=sort, curation=curation)
        log(f"  {result.neurons.path.name}: {result.neurons.n_neurons} neuron(s)")
    except Exception as exc:                      # noqa: BLE001 — reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"

    log("\n" + result.describe())
    return result


# =====================================================================
# One session
# =====================================================================

def process_ephys_session(record, *, sort=DEFAULT_NAME, curation, freeze=False,
                          sync=True, old=DEFAULT_OLD, reexport=False):
    """
    Take one session as far as exported good units, doing only what is missing.

    Args:
        record: the session's :class:`~hba2.cohort.SessionRecord`, as
            the cohort found it.
        sort: the sort's name under ``ephys_sorts/``.
        curation: the curation to use — and to freeze into, if ``freeze``.
        freeze: freeze Phy's current labels as ``curation`` if it does not exist. Say so
            only for a session you curated.
        sync: map the ephys clock if that has not been done. One pass over the raw
            recording: minutes.
        old: the sorter's own folder, adopted into ``ephys_sorts/<sort>`` if found.
        reexport: export even when the export is newer than everything it came from.

    Returns an :class:`EphysSessionResult`.
    """
    result = EphysSessionResult(session_id=record.session_id)
    try:
        # ---- 1. ADOPT ------------------------------------------------
        if sort not in record.sorts:
            if not (record.directory / old).is_dir():
                result.stopped = f"no sort '{sort}' and no {old}/ to adopt"
                return result
            adopt_sort(record.directory, old=old, name=sort)
            result.done.append("adopt")
            record = scan_session(record.directory)
        elif (record.directory / old).is_dir():
            result.warnings.append(f"both ephys_sorts/{sort}/ and {old}/ exist — "
                                   f"using ephys_sorts/{sort}/")

        # ---- 2. FREEZE -----------------------------------------------
        if curation not in record.sorts[sort]["curations"]:
            if not freeze:
                result.stopped = f"no curation '{curation}' (not asked to freeze one)"
                return result
            freeze_curation(record.sorts[sort]["dir"], curation,
                            note="frozen by process_ephys_cohort")
            result.done.append("freeze")
            record = scan_session(record.directory)

        # ---- 3. SYNC -------------------------------------------------
        if record.processed.get("ephys_sync") is None:
            if not record.has_ephys:
                result.stopped = "no ephys recording to sync"
                return result
            if not sync:
                result.stopped = "not synced (sync=False)"
                return result
            written = sync_ephys(record)
            result.done.append("sync")
            result.warnings.extend(written.warnings)
            record = scan_session(record.directory)

        # ---- 4. EXPORT -----------------------------------------------
        # Redone when asked, when it is missing, or when it is older than anything it
        # was made from. Its inputs are the curation's spike assignment and labels and
        # the session's sync; a sync redone, or a curation replaced by hand, makes the
        # export out of date.
        files = record.curation(sort, curation)
        outputs = [files.get(role) for role in ("units", "spikes", "waveforms")]
        if reexport:
            redo = True
        elif any(path is None for path in outputs):
            redo = True
        else:
            made = min(Path(path).stat().st_mtime for path in outputs)
            inputs = [files.get("spike_clusters"), files.get("cluster_group"),
                      record.processed.get("ephys_sync")]
            redo = any(path is not None and Path(path).stat().st_mtime > made
                       for path in inputs)

        if redo:
            summary = export(record, sort, curation)
            result.done.append("export")
            if summary["spikes_unsynced"]:
                result.warnings.append(f"{summary['spikes_unsynced']} spike(s) outside "
                                       "the DAQ's span have no DAQ time")
            result.units = summary["units"]
        else:
            # Up to date: the count is units.csv's rows, less its header.
            with open(files["units"]) as handle:
                result.units = max(sum(1 for _ in handle) - 1, 0)

    except Exception as exc:                      # noqa: BLE001 — reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"
    return result


# =====================================================================
# What comes back
# =====================================================================

@dataclass
class EphysSessionResult:
    """What one session went through, and where it stopped if it did."""

    session_id: str
    done: list[str] = field(default_factory=list)
    stopped: str | None = None          #: why it went no further — not an error
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    units: int = 0

    @property
    def ok(self):
        return self.error is None

    @property
    def exported(self):
        """Whether this session ends the run with good units to hand on."""
        return self.ok and self.stopped is None


@dataclass
class EphysCohortResult:
    """What one sweep did, per session and across the cohort."""

    base: Path
    sort: str
    curation: str
    results: list[EphysSessionResult] = field(default_factory=list)
    names: dict = field(default_factory=dict)
    neurons: NeuronIndexResult | None = None
    error: str | None = None

    def describe(self):
        exported = [r for r in self.results if r.exported]
        failed = [r for r in self.results if r.error]
        stopped = [r for r in self.results if r.ok and r.stopped]
        lines = [f"{self.base.name} · {self.sort}/{self.curation} — "
                 f"{len(exported)} session(s) with units, {len(stopped)} stopped short, "
                 f"{len(failed)} failed",
                 f"  {sum(r.units for r in exported)} good units"]

        #: step name -> how many sessions did it this run
        steps = {}
        for r in self.results:
            for step in r.done:
                steps[step] = steps.get(step, 0) + 1
        if steps:
            lines.append("  this run: " + ", ".join(f"{step} {n}" for step, n in steps.items()))

        # The naming report: a list or dict is shown by its length, a number as it is,
        # and an empty entry not at all.
        if self.names:
            counts = []
            for key, value in self.names.items():
                if not value:
                    continue
                if isinstance(value, (list, dict)):
                    value = len(value)
                counts.append(f"{key} {value}")
            lines.append("  names: " + ", ".join(counts))
        if self.neurons is not None:
            lines += [""] + ["  " + line for line in self.neurons.describe().splitlines()]

        if stopped:
            #: why it stopped -> the sessions that stopped for that reason
            reasons = {}
            for r in stopped:
                reasons.setdefault(r.stopped, []).append(r.session_id)
            lines += ["", "  STOPPED SHORT"]
            lines += [f"    {len(ids)} session(s): {why}" for why, ids in sorted(reasons.items())]

        warned = []
        for r in self.results:
            for warning in r.warnings:
                warned.append((r.session_id, warning))
        if warned:
            lines += ["", "  WARNINGS"]
            lines += [f"    {sid}  {w}" for sid, w in warned[:20]]
        if failed:
            lines += ["", "  FAILED"]
            lines += [f"    {r.session_id}  {r.error}" for r in failed]
        if self.error:
            lines += ["", f"  COHORT STEP FAILED — {self.error}"]
        return "\n".join(lines)
