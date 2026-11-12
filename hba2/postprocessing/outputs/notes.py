"""
What the protocol said it did — on its own clock, and labelled as such.
=======================================================================

Every other file in this directory is on the DAQ clock. This one is not, and that is the
whole reason it exists as a separate file rather than as columns on ``trials.csv``.

The controller lets a protocol annotate a trial as it runs — ``t.stimulus(port, modality)``
for what the animal was shown, ``t.event(name, **details)`` for anything else worth
knowing. Both are timestamped as ``self._now() - self._trial_start_time``, where
``_now()`` is the **PC wall clock** and ``_trial_start_time`` is when the *software*
opened the trial. The trial marker in every other file is when the *DAQ line went high*.
Between those two instants sits the serial write latency, which varies trial to trial and
has never been measured. So a protocol annotation cannot be placed on the DAQ clock at
all — not badly, not approximately. There is no mapping.

**The file format carries that warning.** There is no ``t`` column here and there never
will be. The only time is ``pc_offset_s``, which names both the clock it came from and
the zero it was measured from, so nobody opening this file finds a column that looks like
it could be added to ``t_start``.

What *is* reliable: the fields. A protocol recording ``t.event("laser", slot=2,
power_mw=8.0)`` is the only record of which slot fired and at what power — no amount of
staring at ``pulses_laser.csv`` recovers those. And differences between two annotations on
the same trial are real, because they share a clock and a zero; only the offset to
hardware is meaningless.

Long rather than one row per annotation, because which keys a protocol uses is exactly
the thing that varies between protocols — so it is a column of values, never a set of
column names (§2.3), and the file stays greppable and sortable (§2.4). A session's worth
is a few thousand rows.

| column | meaning |
|---|---|
| ``trial`` | the trial it belongs to; annotations always have one |
| ``note`` | 0-based index within the trial, so two ``laser`` notes stay apart |
| ``kind`` | ``stimulus`` or ``event`` — the two calls that produce these |
| ``note_name`` | the modality for a stimulus, the protocol's own name for an event.
  Called ``note_name`` and not ``name`` because NWB's DynamicTable already has a
  ``name`` of its own, and a column called that reads back as the table's |
| ``pc_offset_s`` | seconds from the software trial start, on the PC clock |
| ``key`` / ``value`` | one detail per row; both empty for an annotation carrying none |

Values are written as JSON unless they are already plain strings, so an int comes back an
int and ``window_ms`` comes back a list. The one ambiguity that buys is a protocol whose
detail is literally the string ``"true"``, which reads back as a boolean.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ._result import OutputResult

COLUMNS = ("trial", "note", "kind", "note_name", "pc_offset_s", "key", "value")

#: What a stimulus is called when the protocol gave no modality.
UNNAMED_STIMULUS = "stimulus"

#: What an event is called when the protocol gave no name. Should not happen — the
#: controller requires one — but a malformed log is not a reason to lose the row.
UNNAMED_EVENT = "event"

#: Keys consumed by the table's own columns rather than written as details.
STIMULUS_OWN = ("t", "modality")
EVENT_OWN = ("t", "name")


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _notes_of(row):
    """
    One trial's annotations, stimuli first, as ``(kind, name, pc_offset_s, details)``.

    Stimuli come first because they are what the animal was shown, and a protocol that
    records both usually presents before it annotates. Within each kind the protocol's
    own order is kept — it is the order they happened.
    """
    for stimulus in row.stimuli or ():
        details = {key: value for key, value in stimulus.items()
                   if key not in STIMULUS_OWN}
        yield ("stimulus",
               str(stimulus.get("modality") or UNNAMED_STIMULUS),
               _as_float(stimulus.get("t")),
               details)

    for event in row.events or ():
        details = {key: value for key, value in event.items() if key not in EVENT_OWN}
        yield ("event",
               str(event.get("name") or UNNAMED_EVENT),
               _as_float(event.get("t")),
               details)


def _encode(value):
    """
    A detail as text that reads back as what it was.

    Strings go raw so the file stays readable; everything else goes through JSON, which
    is what makes an int come back an int and a window come back a list.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def write(out_dir, rows, n_trials):
    """
    One row per detail per annotation, for the first ``n_trials`` trials.

    Args:
        rows: the parsed trials CSV — its rows already hold ``stimuli`` and ``events`` as
            lists of dicts, so nothing is parsed twice.
        n_trials: how many trials were paired with marker pulses, so this file covers
            exactly the trials ``trials.csv`` does and the two join cleanly.

    Written even when empty. A protocol that annotated nothing produces a header and no
    rows, which says "nothing recorded"; the file being absent says "post-processing
    predates it", and those are different facts.
    """
    path = Path(out_dir) / "notes.csv"
    n = min(n_trials, len(rows.rows))
    written = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for trial in range(n):
            for note, (kind, name, offset, details) in enumerate(_notes_of(rows.rows[trial])):
                stamp = "" if offset is None else f"{offset:.6f}"
                if not details:
                    writer.writerow((trial, note, kind, name, stamp, "", ""))
                    written += 1
                    continue
                for key, value in details.items():
                    writer.writerow((trial, note, kind, name, stamp, key, _encode(value)))
                    written += 1

    return OutputResult(name="notes", path=path, n_rows=written)
