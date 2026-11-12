"""
``pulses_<name>.csv`` — one file per configured pulse source.

A pulse source is any channel that is not arena hardware or a timing line — laser
stimulation today, anything wired to a spare line tomorrow. The pipeline decides which
those are; this module just writes one. Each file is named after its channel, so
listing the directory tells you which trains exist and where each came from.

Laser lives here rather than in ``events.csv`` for two reasons: it is stimulation rather
than something the animal did, and at ~5,800 pulses per session it would swamp a
~25,000-row events file.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ._result import OutputResult

COLUMNS = ("pulse", "t_on", "t_off", "duration")


def write(out_dir, daq, channel) -> OutputResult | None:
    """
    Write the pulse train on ``channel`` as ``pulses_<channel>.csv``.

    Returns ``None`` — writing nothing — when the channel carries no pulses. A spare
    channel that was not wired up this session should not leave an empty file behind;
    the files that exist are what say which channels were live.
    """
    ons = daq.rising_edges(channel)
    offs = daq.falling_edges(channel)
    n = min(len(ons), len(offs))
    if n == 0:
        return None

    path = Path(out_dir) / f"pulses_{channel.lower()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(
            (i, f"{ons[i]:.6f}", f"{offs[i]:.6f}", f"{offs[i] - ons[i]:.6f}")
            for i in range(n)
        )

    warnings = ()
    if len(ons) != len(offs):
        warnings = (
            f"channel {channel!r} has {len(ons)} rises and {len(offs)} falls; "
            f"wrote the first {n} complete pulses",
        )
    return OutputResult(name=f"pulses_{channel.lower()}", path=path,
                        n_rows=n, warnings=warnings)
