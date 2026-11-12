"""
What the protocol said it did — and why its times are not the DAQ's.
====================================================================

The controller lets a protocol annotate a trial as it runs: ``t.stimulus(port, modality)``
for what the animal was shown, ``t.event(name, **details)`` for anything else. That is
the only record of which laser slot fired, at what power, into which armed window — no
amount of reading ``pulses_laser.csv`` recovers those numbers::

    trials = attach_notes(trials,
        laser         = note("laser", "slot", "power_mw", "window_ms"),
        left_platform = note("mouse_left_platform"),
    )

adds ``laser_slot``, ``laser_power_mw``, ``laser_window_ms``, ``has_laser``,
``has_left_platform`` — and ``pc_offset_laser``, ``pc_offset_left_platform``.

**Those offsets are on the PC's clock, and it is not the DAQ's.** An annotation is
timestamped from when the *software* opened the trial; ``start_time`` is when the *DAQ
line went high*. Between them sits a serial write latency that varies trial to trial and
has never been measured, so there is no mapping between the two — not a rough one, not a
bad one. ``start_time + pc_offset_laser`` is meaningless, which is why the column is named
so that writing it looks wrong.

What the offsets *are* good for: ordering annotations within a trial, and the interval
between two of them, because those share a clock and a zero.

**The vocabulary is entirely the protocol's**, so ``describe_notes(trials)`` is how you
find out what a cohort even recorded — there is no fixed list to document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from .sessions import NWB_COLUMN, _needs, _say_nothing

#: Where the annotations live in the file.
TABLE = ("behaviour", "notes")

#: The trial columns the annotations came from, for a file built before the table existed.
LEGACY = {"event": "event_log", "stimulus": "stimulus_log"}

#: How an attached annotation is written onto the trials table. The offset's name is the
#: warning: it says which clock, and which zero, so it cannot be quietly added to a time.
FLAG = "has_{label}"
OFFSET = "pc_offset_{label}"
COUNT = "n_{label}"
FIELD = "{label}_{field}"

#: Ask for this instead of naming fields to take every field the annotation carries.
ALL_FIELDS = "*"


@dataclass(frozen=True)
class Note:
    """One protocol annotation to find per trial."""

    name: str
    fields: tuple[str, ...] = ()
    kind: str | None = None
    pick: str = "first"
    n: int = 0

    def how(self):
        """One line for the provenance record."""
        parts = [f"{self.pick} {self.kind or 'annotation'} {self.name!r}"]
        if self.fields:
            parts.append(f"fields {', '.join(self.fields)}")
        return " ".join(parts) + " (PC clock)"


def note(name, *fields, kind=None, pick="first", n=0):
    """
    One annotation, and the details to take off it.

    Args:
        name: the protocol's own name — ``describe_notes`` lists what there is.
        *fields: which details to attach as columns. ``"*"`` takes every field it
            carries; naming none attaches only the flag and the offset.
        kind: ``"stimulus"`` or ``"event"`` to disambiguate, if a protocol uses one name
            for both.
        pick: ``"first"``, ``"last"`` or ``"nth"``, when a trial has several.
        n: which one, 0-based, for ``pick="nth"``.
    """
    return Note(name=name, fields=tuple(fields), kind=kind, pick=pick, n=n)


def count_notes(name, kind=None):
    """How many times an annotation appeared in each trial."""
    return Note(name=name, kind=kind, pick="count")


def _empty():
    """The annotations table with nothing in it."""
    return pd.DataFrame(columns=["session_id", "trial", "note", "kind", "note_name",
                                 "pc_offset_s", "key", "value"])


def load_notes(trials, log_fn=None):
    """
    Every protocol annotation belonging to the trials you hand it.

    Returns:
        One row per detail per annotation: ``session_id``, ``trial``, ``note``, ``kind``,
        ``note_name``, ``pc_offset_s``, ``key``, ``value``. An annotation carrying no details
        is one row with ``key`` and ``value`` blank.

    There is no ``t`` column, deliberately — see the module docstring.
    """
    log = log_fn or _say_nothing
    _needs(trials, NWB_COLUMN, "load_notes")

    pieces = []
    for path, rows in trials.groupby(NWB_COLUMN, sort=False):
        table = _read(path)
        wanted = set(rows["trial"])
        piece = table[table["trial"].isin(wanted)].copy()
        piece.insert(0, "session_id", rows["session_id"].iloc[0])
        log(f"  {path.rsplit('/', 1)[-1]}: {len(piece):,} annotation rows")
        pieces.append(piece)

    if not pieces:
        return _empty()
    found = pd.concat(pieces, ignore_index=True)
    found["value"] = found["value"].map(_decode)
    return found


def attach_notes(trials, *, log_fn=None, **notes):
    """
    Take the named annotations off each trial and write them onto the table. Each is a
    :class:`Note`, made by :func:`note` or :func:`count_notes`.

    Each adds ``has_<label>`` and ``pc_offset_<label>``, plus a column per field asked
    for, and records what it meant in ``attrs["tags"]``.
    """
    if not notes:
        raise ValueError("attach_notes needs at least one, e.g. "
                         "laser=note('laser', 'power_mw')")
    for label, wanted in notes.items():
        if not isinstance(wanted, Note):
            raise TypeError(f"{label}= is {type(wanted).__name__}, not a note. Use "
                            "note(...) or count_notes(...).")

    found = load_notes(trials, log_fn=log_fn)
    attached = trials.copy()
    tags = dict(trials.attrs.get("tags", {}))
    for label, wanted in notes.items():
        attached = _apply(attached, found, label, wanted)
        tags[label] = f"note: {wanted.how()}"
    attached.attrs["tags"] = tags
    return attached


def describe_notes(trials):
    """
    What this data's protocols actually annotated — the menu for :func:`note`.

    Read from the files, because the names and their details are the protocol's own and
    vary between them; there is no list to keep up to date anywhere.
    """
    found = load_notes(trials)
    if found.empty:
        return pd.DataFrame(columns=["kind", "note_name", "trials", "fields"])

    rows = []
    for (kind, name), piece in found.groupby(["kind", "note_name"], sort=True):
        fields = sorted({key for key in piece["key"] if key})
        if fields:
            listed = ", ".join(fields)
        else:
            listed = "—"
        rows.append({
            "kind": kind,
            "note_name": name,
            "trials": piece.groupby(["session_id", "trial"]).ngroups,
            "fields": listed,
        })
    return pd.DataFrame(rows)


# =====================================================================
# The steps
# =====================================================================

def _apply(trials, found, label, wanted):
    """One annotation's columns, joined back onto the trials by identity."""
    keys = ["session_id", "trial"]
    matched = found
    if not matched.empty:
        keep = matched["note_name"] == wanted.name
        if wanted.kind is not None:
            keep &= matched["kind"] == wanted.kind
        matched = matched[keep]

    if wanted.pick == "count":
        column = COUNT.format(label=label)
        if matched.empty:
            return trials.assign(**{column: 0})
        # One annotation is several rows, one per detail, so count annotations not rows.
        counted = (matched.groupby(keys)["note"].nunique().rename(column).reset_index())
        joined = trials.merge(counted, on=keys, how="left")
        joined[column] = joined[column].fillna(0).astype(int)
        return joined

    flag, offset = FLAG.format(label=label), OFFSET.format(label=label)
    # The details to attach: those asked for, or every one it carries.
    if ALL_FIELDS not in wanted.fields:
        fields = [field for field in wanted.fields if field != ALL_FIELDS]
    elif matched.empty:
        fields = []
    else:
        fields = sorted({key for key in matched["key"] if key})

    if matched.empty:
        blanks = {flag: 0, offset: np.nan}
        blanks |= {FIELD.format(label=label, field=field): np.nan for field in fields}
        return trials.assign(**blanks)

    chosen = _pick(matched, keys, wanted)
    # The annotation itself: one row per (trial, note), carrying its offset.
    heads = (chosen.drop_duplicates(keys + ["note"])[keys + ["pc_offset_s"]]
             .rename(columns={"pc_offset_s": offset}))
    joined = trials.merge(heads, on=keys, how="left")
    joined[flag] = joined[offset].notna().astype(int)

    for field in fields:
        column = FIELD.format(label=label, field=field)
        values = (chosen[chosen["key"] == field][keys + ["value"]]
                  .rename(columns={"value": column}))
        joined = joined.merge(values.drop_duplicates(keys), on=keys, how="left")
    return joined


def _pick(matched, keys, wanted):
    """Which of a trial's annotations of this name to take, when it has several."""
    order = matched.sort_values(keys + ["note"], kind="stable")
    indices = order.groupby(keys, sort=False)["note"].transform(
        {"first": "min", "last": "max"}.get(wanted.pick, "min"))
    if wanted.pick == "nth":
        numbered = order.groupby(keys, sort=False)["note"].rank(method="dense") - 1
        return order[numbered == wanted.n]
    return order[order["note"] == indices]


def _read(path):
    """
    One session's annotations, from the notes table or from the trial columns.

    Files built before the notes table existed still carry the annotations verbatim as
    ``event_log`` and ``stimulus_log``, so they are reshaped on read rather than being
    unreadable until the file is rebuilt.
    """
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        module, table = TABLE
        behaviour = nwb.processing.get(module)
        if behaviour is not None and table in behaviour.data_interfaces:
            found = behaviour[table].to_dataframe().reset_index(drop=True)
            if len(found):
                return found
        if nwb.trials is None:
            return _empty()
        return _from_logs(nwb.trials.to_dataframe().reset_index())


def _from_logs(trials):
    """The same long table, rebuilt from the verbatim JSON on each trial."""
    rows = []
    for position, trial in trials.iterrows():
        index = int(trial.get("trial", position))
        note = 0
        for kind, column in LEGACY.items():
            # The column's JSON list of entries; anything else is no entries.
            text = trial.get(column)
            entries = []
            if isinstance(text, str) and text.strip():
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError):
                    parsed = []
                if isinstance(parsed, list):
                    entries = parsed

            for entry in entries:
                # An event is named by its name, a stimulus by its modality.
                if kind == "event":
                    own = ("t", "name")
                    name = str(entry.get("name"))
                else:
                    own = ("t", "modality")
                    name = str(entry.get("modality") or "stimulus")
                details = {k: v for k, v in entry.items() if k not in own}
                offset = entry.get("t")
                if not details:
                    rows.append((index, note, kind, name, offset, "", None))
                for key, value in details.items():
                    rows.append((index, note, kind, name, offset, key, value))
                note += 1
    return pd.DataFrame(rows, columns=["trial", "note", "kind", "note_name",
                                       "pc_offset_s", "key", "value"])


def _decode(value):
    """A stored detail as the type it was. Plain strings were written raw."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if not isinstance(value, str):
        return value
    if value == "":
        return np.nan
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value
