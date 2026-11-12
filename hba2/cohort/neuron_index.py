"""
The neuron index — one row per neuron, and the labels you attach to them.
========================================================================

``neurons.csv`` in the cohort folder is to neurons what ``sessions.csv`` is to sessions,
under the same one rule: **columns starting with ``_`` belong to this file; everything
else belongs to you.**

    neuron_id, _mouse_id, _n_sessions, _members, _shank, _depth_um, …, cell_type, optotagged, include, notes
    └ the key  └─────────────── rebuilt every time ───────────────┘ └─────── never touched ───────┘

Add a label by typing a column heading. Rebuilding recomputes every ``_`` column from the
curations' exports, appends rows for neurons that have appeared, and never overwrites a
value you typed. ``load_units`` joins the table onto every unit, so a label becomes
something to select by (``load_units(trials, cell_type="narrow")``) and group by
(``group="cell_type"``) with nothing registered anywhere.

A neuron is a ``neuron_id`` from ``neuron_names.csv``: one cell, seen in one session or
in several. Every good unit has one (see ``ephys/names.py``).

Labels follow cells, not numbers
--------------------------------
Re-running the cross-session matching can renumber neurons — ``E4_n0127`` in one run
may be ``E4_n0131`` in the next, and ``E4_n0127`` may then be a different cell. So a
label cannot be carried by ``neuron_id`` alone.

``_members`` records which (session, unit) pairs make up each neuron. On a rebuild, your
labels go to whichever new neuron holds the same members:

* **renamed** — the members are the same, under a new ID: the labels move with them.
* **split** — one old neuron's members now form several: each part gets its labels.
* **merged** — several old neurons now form one: a label they agree on is kept; one they
  disagree on is left blank and reported, never chosen between.
* **gone** — no member of an old neuron is present any more (a session removed, a
  curation changed): the row stays, marked ``_present`` false, labels intact.

All four are reported by the build, so nothing moves without you seeing it.

Reading it back::

    neurons = read_neuron_index(cohort)
    narrow = neurons.query("cell_type == 'narrow' and _n_sessions > 1")

The page reads in the order the data moves::

    1. where it is            neuron_index_path, read_neuron_index
    2. building               build_neuron_index: gather, merge, write
    3. what the curations say unit_rows → neuron_rows
    4. carrying the labels    _carry_labels: your labels onto the new rows
    5. what comes out         NeuronIndexResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import Cohort
from .session_index import OWNED, _read_for_merge, _write

INDEX_NAME = "neurons.csv"
KEY = "neuron_id"

#: Every column this file writes, in order. Anything not here and not the key is yours.
GENERATED = (
    "_present",
    # who, and where it was seen
    "_mouse_id", "_n_sessions", "_sessions", "_first_date", "_last_date", "_members",
    # how it was named
    "_epoch", "_matching", "_level",
    # what it looks like — medians over its units, one per session
    "_shank", "_depth_um", "_x_um", "_amplitude_uv", "_waveform_width_ms",
    "_firing_rate_hz", "_snr", "_isi_violations_ratio", "_presence_ratio",
    # which units these came from
    "_sort", "_curation",
)

#: Started empty for you to fill. Add any column you like beside them.
LABELS = ("include", "notes")

#: Joins (session, unit) pairs in ``_members``.
MEMBER_SEP = ";"


# =====================================================================
# 1. Where it is, and reading it back
# =====================================================================

def neuron_index_path(cohort):
    """Where a cohort's neuron index lives, given a :class:`Cohort` or its folder as a
    Path or a string. Always the cohort folder."""
    if isinstance(cohort, Cohort):
        base = cohort.base
    else:
        base = Path(cohort)
    return base / INDEX_NAME


def read_neuron_index(cohort):
    """
    The index, with types inferred — what analysis should use. ``cohort`` is a
    :class:`Cohort` or its folder as a Path or a string.

    Raises:
        FileNotFoundError: if the cohort has none yet. Build one first.
    """
    path = neuron_index_path(cohort)
    if not path.exists():
        raise FileNotFoundError(f"no neuron index at {path} — run "
                                "build_neuron_index(cohort, sort=..., curation=...)")
    return pd.read_csv(path, dtype={KEY: str})


# =====================================================================
# 2. Building — merging, never rewriting
# =====================================================================

def _say_nothing(message):
    """What build_neuron_index reports to when nobody is listening."""


def build_neuron_index(cohort, *, sort, curation, log_fn=None):
    """
    Create or refresh ``neurons.csv`` from one curation of one sort, across the cohort.

    Args:
        cohort: a :class:`Cohort`, or a path (Path or string) to build one from.
        sort, curation: which sort's curation the units come from, by name.
        log_fn: called with each line as it happens.

    Needs each session's curation exported and named (``ephys export``, ``ephys names``).
    Safe to run whenever: generated columns are recomputed, new neurons appended, and
    every column you added is carried — by membership, not by ID, so renumbering a
    matching run cannot put your labels on the wrong cell. See the module docstring.

    Returns a :class:`NeuronIndexResult`.
    """
    log = log_fn or _say_nothing
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    path = neuron_index_path(cohort)
    result = NeuronIndexResult(path=path)

    # ---- 1. Every neuron's generated columns, fresh -------------------
    fresh = neuron_rows(unit_rows(cohort, sort, curation), sort, curation)
    log(f"{cohort.base.name}: {len(fresh)} neuron(s) from {sort}/{curation}")

    # ---- 2. What is on disk already -----------------------------------
    existing = _read_for_merge(path)

    # Nothing named under this curation is a wrong argument or a step not yet run, not
    # a cohort whose neurons all vanished: merging it would mark every row gone.
    if fresh.empty:
        result.warnings.append(f"no exported, named units for {sort}/{curation} — "
                               f"{path.name} left as it was")
        if existing is not None:
            result.n_neurons = len(existing)
            result.label_columns = [c for c in existing.columns
                                    if c != KEY and c not in GENERATED]
        log(f"{path.name}: nothing to index for {sort}/{curation}; left as it was")
        return result

    if existing is None:
        # ---- 3a. No index yet: the fresh rows, and empty labels -------
        merged = fresh.copy()
        yours = list(LABELS)
        for label in yours:
            merged[label] = ""
        result.n_new = len(merged)
    else:
        # ---- 3b. An index already: carry your labels across -----------
        existing = existing.set_index(KEY)
        yours = [c for c in existing.columns if c not in GENERATED]
        unexpected = [c for c in yours if c.startswith(OWNED)]
        if unexpected:
            result.warnings.append(
                f"the leading underscore is reserved for generated columns, but "
                f"{', '.join(unexpected)} is not one — kept and left untouched")
        for label in LABELS:
            if label not in yours:
                yours.append(label)
        merged = _carry_labels(existing, fresh, yours, result)

    # ---- 4. Write ------------------------------------------------------
    result.n_neurons = len(merged)
    result.n_gone = int((~merged["_present"].astype(str).isin(["True", "true"])).sum())
    result.label_columns = yours
    merged = merged[[*GENERATED, *yours]].sort_index()
    merged.index.name = KEY
    _write(merged.reset_index(), path)

    line = f"{path.name}: {result.n_neurons} neuron(s)"
    if result.n_new:
        line += f", {result.n_new} new"
    if result.renamed:
        line += f", {len(result.renamed)} relabelled"
    if result.conflicts:
        line += f", {len(result.conflicts)} conflict(s)"
    log(line)
    return result


# =====================================================================
# 3. What the curations say about each neuron
# =====================================================================

def _waveform_width_ms(wave, sample_rate):
    """
    Trough to the following peak, in milliseconds — the usual narrow/wide measure.

    Taken on the peak channel's mean waveform (an array). NaN where there is no
    waveform.
    """
    if not np.isfinite(wave).any():
        return np.nan
    trough = int(np.nanargmin(wave))
    after = wave[trough:]
    if len(after) < 2:
        return np.nan
    peak = trough + int(np.nanargmax(after))
    return (peak - trough) / sample_rate * 1000.0


def unit_rows(cohort, sort, curation):
    """
    Every named good unit in the cohort, one row each, with what the index summarises.

    Sessions without this curation, or not yet exported or named, contribute nothing.
    """
    rows = []
    for record in cohort.with_files():
        files = record.sorts.get(sort, {}).get("curations", {}).get(curation)
        if not files or not files.get("units") or not files.get("names"):
            continue
        units = pd.read_csv(files["units"])
        names = pd.read_csv(files["names"], dtype={"neuron_id": str, "epoch": str,
                                                   "track": str})
        table = units.merge(names, on="unit_id", how="inner")

        #: unit ID -> its waveform's width
        widths = {}
        if files.get("waveforms"):
            waves = np.load(files["waveforms"])
            rate = float(waves["sample_rate"])
            for unit_id, wave in zip(waves["unit_id"], waves["waveform"]):
                widths[int(unit_id)] = _waveform_width_ms(wave, rate)
        table["waveform_width_ms"] = table["unit_id"].map(widths)

        table["session_id"] = record.session_id
        table["mouse_id"] = record.mouse_id
        table["date"] = record.date
        rows.append(table)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def _joined(values):
    """Values as one cell — sorted, deduplicated, joined by MEMBER_SEP, blanks dropped."""
    return MEMBER_SEP.join(sorted({str(v) for v in values if pd.notna(v) and str(v)}))


def _median(values):
    """The median of a column, as a float; anything not a number is left out."""
    return float(pd.to_numeric(values, errors="coerce").median())


def neuron_rows(units, sort, curation):
    """The generated columns, one row per neuron, from :func:`unit_rows`."""
    if units.empty:
        return pd.DataFrame(columns=[KEY, *GENERATED]).set_index(KEY)

    out = []
    for neuron_id, group in units.groupby("neuron_id", sort=True):
        members = []
        for session_id, unit_id in zip(group["session_id"], group["unit_id"]):
            members.append(f"{session_id}:{int(unit_id)}")

        # One shank for a real cell; several means the matching joined units on
        # different shanks, which is worth seeing rather than averaging away.
        shanks = sorted({int(s) for s in group["shank"].dropna()})
        if len(shanks) == 1:
            shank = shanks[0]
        else:
            shank = MEMBER_SEP.join(map(str, shanks))

        out.append({
            KEY: neuron_id,
            "_present": True,
            "_mouse_id": group["mouse_id"].iloc[0],
            "_n_sessions": int(group["session_id"].nunique()),
            "_sessions": _joined(group["session_id"]),
            "_first_date": min(group["date"]),
            "_last_date": max(group["date"]),
            "_members": _joined(members),
            "_epoch": _joined(group["epoch"]),
            "_matching": _joined(group["matching"]),
            "_level": _joined(group["level"]),
            "_shank": shank,
            "_depth_um": _median(group["depth_um"]),
            "_x_um": _median(group["x_um"]),
            "_amplitude_uv": _median(group["amplitude_uv"]),
            "_waveform_width_ms": _median(group["waveform_width_ms"]),
            "_firing_rate_hz": _median(group["firing_rate_hz"]),
            "_snr": _median(group["snr"]),
            "_isi_violations_ratio": _median(group["isi_violations_ratio"]),
            "_presence_ratio": _median(group["presence_ratio"]),
            "_sort": sort,
            "_curation": curation,
        })
    return pd.DataFrame(out).set_index(KEY)


# =====================================================================
# 4. Carrying your labels onto the new rows
# =====================================================================

def _carry_labels(existing, fresh, yours, result):
    """
    The fresh rows, with your labels brought over from whichever old rows hold the same
    members, and old rows nothing claims kept as ``_present`` false.

    ``existing`` is neurons.csv as text, by neuron ID; ``yours`` the label columns; what
    moved is recorded on ``result``, a :class:`NeuronIndexResult`.
    """
    # ---- 1. Which old neuron each member belonged to ---------------------
    #: member -> old neuron_id
    old_of = {}
    for old_id, members in existing.get("_members", pd.Series(dtype=str)).items():
        for member in str(members).split(MEMBER_SEP):
            if member:
                old_of[member] = old_id

    #: old neuron_id -> {label column -> value}
    labels = {}
    for old_id, row in existing.iterrows():
        labels[old_id] = {c: row.get(c, "") for c in yours}

    merged = fresh.copy()
    for column in yours:
        merged[column] = ""

    # ---- 2. Each new neuron takes the labels of the old ones it holds ----
    #: old neuron_id -> the new neuron_ids it went to
    claimed = {}
    for new_id, row in fresh.iterrows():
        members = [m for m in str(row["_members"]).split(MEMBER_SEP) if m]
        sources = sorted({old_of[m] for m in members if m in old_of})

        # A row typed in by hand, no members yet: it keeps its own labels.
        if not sources and new_id in existing.index:
            if not str(existing.loc[new_id].get("_members", "")).strip():
                sources = [new_id]

        if not sources:
            result.n_new += 1
            continue
        for old_id in sources:
            claimed.setdefault(old_id, set()).add(new_id)
        if sources != [new_id]:
            result.renamed.append(f"{', '.join(sources)} -> {new_id}")

        # A label the sources agree on is kept; one they disagree on is left blank.
        for column in yours:
            values = {str(labels[s].get(column, "")) for s in sources} - {"", "nan"}
            if len(values) == 1:
                merged.at[new_id, column] = values.pop()
            elif len(values) > 1:
                each = " / ".join(f"{s}={labels[s][column]!r}" for s in sources)
                result.conflicts.append(f"{new_id}.{column}: " + each)

    for old_id, new_ids in claimed.items():
        if len(new_ids) > 1:
            result.split.append(f"{old_id} -> {', '.join(sorted(new_ids))}")

    # ---- 3. Old neurons nothing claims ----------------------------------
    # Kept, labels and all, so no typing is ever lost.
    gone = [old_id for old_id in existing.index if old_id not in claimed]
    for old_id in gone:
        row = existing.loc[old_id]
        key = old_id
        if key in merged.index:
            # The ID now names a different cell; the old one keeps its labels under a
            # name that cannot be mistaken for it.
            key = f"{old_id}_retired"
            result.warnings.append(f"{old_id} now names a different neuron; the old row "
                                   f"is kept as {key}")
        values = {}
        for column in GENERATED:
            values[column] = row.get(column, "")
        for column in yours:
            values[column] = row.get(column, "")
        values["_present"] = False
        merged.loc[key] = pd.Series(values)
    return merged


# =====================================================================
# 5. What comes out
# =====================================================================

@dataclass
class NeuronIndexResult:
    """What one build did, including every label that moved."""

    path: Path
    n_neurons: int = 0
    n_new: int = 0
    n_gone: int = 0
    label_columns: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)
    split: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self):
        heading = f"{self.path.name}: {self.n_neurons} neuron(s)"
        if self.n_new:
            heading += f", {self.n_new} new"
        if self.n_gone:
            heading += f", {self.n_gone} no longer present"
        lines = [heading, "  your columns: " + (", ".join(self.label_columns) or "none")]

        for title, items in (("LABELS MOVED TO A NEW ID", self.renamed),
                             ("SPLIT — LABELS COPIED TO EACH PART", self.split),
                             ("MERGED WITH DISAGREEING LABELS — LEFT BLANK", self.conflicts),
                             ("WARNINGS", self.warnings)):
            if items:
                lines += ["", f"  {title} ({len(items)})"]
                lines += [f"    {item}" for item in items[:20]]
                if len(items) > 20:
                    lines.append(f"    ... and {len(items) - 20} more")
        return "\n".join(lines)
