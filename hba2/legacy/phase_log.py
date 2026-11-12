"""
Reader for ``<session>_Phase_<X>_behaviour_data.json`` — the old rig's own session record.

Before the controller rewrite there was no ``-metadata.json`` and no ``-trials.csv``.
This file was both: a header of session facts, then ``Logs`` (the serial conversation
with the arena Arduino, one line per message) and ``Scales data``.

**Nothing in it is on the DAQ clock.** Every time is the PC's, in seconds from when the
script started. The trials converter uses the log only to say what the rig *thought*
happened — which port it cued, what it credited — and pairs it with the DAQ by time.

Log lines, ``;``-separated::

    OUT;<t>                         the PC sent a trial (or a follow-up) to the Arduino
    IN;<t>;R;<port>;R               the Arduino acknowledged: this port was cued
    IN;<t>;C;<port|F>;<T|F>         the trial completed: port credited, success flag;
                                    port ``F`` is a timeout
    IN;<t>;R;<port>;<ms>;R          the same, from the earliest sessions, which also
                                    carried the Arduino's own millisecond clock

Ports are 1-based here and converted to 0-based on read, like everything else in the
package. Lines from later sessions carry ANSI colour codes, stripped on read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: Header keys that are lists of records rather than session facts.
_BULK_KEYS = ("Logs", "Scales data")


def read(path):
    """The Phase JSON at ``path`` (a Path or a string), as a :class:`PhaseLog`."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    header = {k: v for k, v in document.items() if k not in _BULK_KEYS}

    # Scales readings: rows of PC time, grams and, on wired scales, the pulse index.
    # Rows are cut to the shortest, so a file mixing the two kinds reads as the shorter.
    clean = [r for r in (document.get("Scales data") or [])
             if isinstance(r, (list, tuple)) and len(r) >= 2]
    if clean:
        width = min(len(r) for r in clean)
        scales = np.asarray([r[:width] for r in clean], dtype=float)
    else:
        scales = np.empty((0, 2))

    return PhaseLog(path=path, header=header,
                    trials=parse_logs(document.get("Logs") or []), scales=scales)


def _as_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _port(value) -> int | None:
    """1-based port text -> 0-based int; ``F`` and anything else -> None."""
    number = _as_int(value)
    if number is None or number < 1:
        return None
    return number - 1


def parse_logs(lines):
    """
    Pair each acknowledgement (R) with the completion (C) that follows it.

    ``lines`` is the Phase JSON's ``Logs`` list; returns a list of :class:`LogTrial`.
    """
    trials = []
    pending = None              # (t_cued, cued_port) of an R waiting for its C

    for raw in lines:
        parts = [p.strip() for p in _ANSI.sub("", str(raw)).split(";")]
        if len(parts) < 4 or parts[0] != "IN":
            continue
        try:
            t = float(str(parts[1]).strip())
        except (TypeError, ValueError):
            continue
        kind, port = parts[2], parts[3]

        if kind == "R":
            if pending is not None:          # a cue with no completion logged
                trials.append(LogTrial(pending[0], pending[1], None, None, None))
            pending = (t, _port(port))
        elif kind == "C" and pending is not None:
            flag = parts[-1].upper()
            success = None
            if flag == "T":
                success = True
            if flag == "F":
                success = False
            trials.append(LogTrial(
                t_cued=pending[0], cued_port=pending[1], t_done=t,
                chosen_port=_port(port), success=success))
            pending = None

    if pending is not None:
        trials.append(LogTrial(pending[0], pending[1], None, None, None))
    return trials


# =====================================================================
# What comes out
# =====================================================================

@dataclass(frozen=True)
class LogTrial:
    """One cued trial as the old rig logged it. PC clock throughout."""

    t_cued: float            #: when the Arduino acknowledged the cue (R)
    cued_port: int | None    #: 0-based; None if unparseable
    t_done: float | None     #: when it reported completion (C); None if the log ends first
    chosen_port: int | None  #: 0-based; None on a timeout
    success: bool | None


@dataclass(frozen=True)
class PhaseLog:
    """Everything in one Phase JSON."""

    path: Path
    header: dict[str, Any]
    trials: list[LogTrial] = field(default_factory=list)
    #: ``(n, 2)`` or ``(n, 3)``: PC time, grams, and (wired scales only) the index of
    #: the SCALES pulse the reading belongs to.
    scales: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    @property
    def phase(self) -> str | None:
        value = self.header.get("Behaviour phase")
        if value is None:
            match = re.search(r"_Phase_(.+?)_behaviour_data", self.path.name)
            if match:
                value = match.group(1)
        if value is None:
            return None
        return str(value)

    @property
    def rig(self) -> int | None:
        return _as_int(self.header.get("Rig"))

    @property
    def scales_have_pulse_index(self):
        return self.scales.ndim == 2 and self.scales.shape[1] >= 3
