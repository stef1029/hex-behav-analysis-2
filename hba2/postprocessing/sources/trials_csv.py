"""
Reader for ``<session>-trials.csv``.

The rig's own account of each trial, written at session end. It is the authority
for *what kind of trial this was* and *what the mouse chose* — but **not for
timing**: ``time_since_start_s`` is the controller's software clock, which drifts
against the DAQ. Trial times come from the DAQ link channel instead
(see :func:`hba2.postprocessing.utils.clock.trial_windows`).

Two conventions worth stating plainly, because both have caused bugs:

* **Ports here are already 0-based** and are kept that way. Nothing in this package
  maps them to 1-based strings to match DAQ channel names.
* **Block identity is not always in ``tracker``.** Most protocols make one
  tracker per block, but some share one tracker across several blocks (the audio
  phases) and one uses a single tracker for a whole session. ``stage`` carries
  the block name when the protocol passes it, and is empty when it does not — so
  the usable block identity is ``stage or tracker``, resolved once here as
  :attr:`TrialRow.block`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

#: Columns the schema gained over time. A session recorded before one was added simply
#: does not have it, which is absence rather than an error.
OPTIONAL_COLUMNS = ("trial_type", "stage", "stimulus_log", "details_json", "event_log")

#: Sub-tracker name used when a tracker has no sub-trackers — the controller's own
#: convention, and what an older CSV with no ``trial_type`` column means.
DEFAULT_SUB_TRACKER = "default"

#: Columns the controller has since renamed, translated on read. ``group`` was what a
#: tracker was called before the blocks rewrite; the values are the same thing under
#: both names (``scales_training``, ``cue_duration_100ms``, ``warm_up``), so a session
#: recorded under the old name is read without anything downstream needing to know.
_LEGACY_COLUMNS = {"group": "tracker"}


def _as_int(value) -> int | None:
    """Empty strings are normal here — a timeout has no chosen port."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_json(value, default):
    """Parse an embedded JSON column, falling back to ``default`` rather than failing."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def read(path):
    """
    Parse a trials CSV (a Path or a string) into :class:`TrialRows`.

    Tolerates missing optional columns and unknown extra ones — the schema has
    grown over time and will grow again. Raises only if the file has no header
    or lacks the columns without which a trial is meaningless.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        # ---- 1. The header ----------------------------------------------
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} is empty — no header row")
        columns = list(reader.fieldnames)

        renames = {old: new for old, new in _LEGACY_COLUMNS.items()
                   if old in columns and new not in columns}
        columns = [renames.get(name, name) for name in columns]

        missing = {"tracker", "outcome"} - set(columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing required column(s): {sorted(missing)}"
            )

        # ---- 2. Each row ------------------------------------------------
        rows = []
        for index, raw in enumerate(reader):
            # Under current column names.
            if renames:
                raw = {renames.get(key, key): value for key, value in raw.items()}
            rows.append(TrialRow(
                index=index,
                trial_number=_as_int(raw.get("trial_number")),
                tracker=(raw.get("tracker") or "").strip(),
                # No trial_type column at all in the earliest sessions; a tracker
                # without sub-trackers is "default", which is what the controller
                # itself writes.
                trial_type=(raw.get("trial_type") or "").strip() or DEFAULT_SUB_TRACKER,
                stage=(raw.get("stage") or "").strip(),
                outcome=(raw.get("outcome") or "").strip(),
                software_time=_as_float(raw.get("time_since_start_s")),
                software_duration=_as_float(raw.get("trial_duration_s")),
                correct_port=_as_int(raw.get("correct_port")),
                chosen_port=_as_int(raw.get("chosen_port")),
                stimuli=_as_json(raw.get("stimulus_log"), default=[]),
                events=_as_json(raw.get("event_log"), default=[]),
                details=_as_json(raw.get("details_json"), default={}),
                raw=dict(raw),
            ))

    return TrialRows(path=path, rows=rows, columns=columns,
                     renamed_columns=renames)


@dataclass(frozen=True)
class TrialRow:
    """One row of the trials CSV, parsed."""

    index: int                  #: 0-based position in the file
    trial_number: int | None    #: the rig's own global trial number
    tracker: str
    trial_type: str
    stage: str
    outcome: str

    software_time: float | None  #: seconds since session start, SOFTWARE clock
    software_duration: float | None

    correct_port: int | None    #: 0-based, None if unrecorded
    chosen_port: int | None     #: 0-based, None on a timeout

    stimuli: list[dict] = field(default_factory=list)

    #: What the protocol noted as the trial ran — each entry ``{t, name, ...}``,
    #: with ``t`` relative to the trial marker and negative where the thing
    #: happened before it. The names and the keys beside them are the
    #: protocol's own; nothing in this package interprets them.
    events: list[dict] = field(default_factory=list)

    details: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def block(self):
        """
        The block this trial ran under.

        ``stage`` when the protocol recorded it, otherwise the tracker name.
        This coalesce is load-bearing: within a single protocol some branches
        pass ``stage`` and others do not.
        """
        return self.stage or self.tracker


@dataclass(frozen=True)
class TrialRows:
    """Every row of one trials CSV, in file order."""

    path: Path
    rows: list[TrialRow]
    columns: list[str]
    #: Old column names translated on read (old -> current); empty for a current CSV.
    renamed_columns: dict = field(default_factory=dict)

    @property
    def absent_columns(self):
        """Optional columns this CSV does not have, for the record."""
        return [name for name in OPTIONAL_COLUMNS if name not in self.columns]

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    @property
    def blocks(self):
        """Distinct block names, in first-seen order."""
        seen = {}
        for row in self.rows:
            seen.setdefault(row.block, None)
        return list(seen)

    def describe(self):
        #: outcome -> how many trials had it
        outcomes = {}
        for row in self.rows:
            outcomes[row.outcome] = outcomes.get(row.outcome, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items()))

        line = (f"{self.path.name}: {len(self.rows)} trials, "
                f"{len(self.blocks)} blocks, {summary}")
        absent = self.absent_columns
        if absent:
            line += f", no {'/'.join(absent)}"
        return line
