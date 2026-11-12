"""
``events.csv`` — every behaviour channel transition, on the clock.

The primitive. Trials and any later interpretation are derived from this, which is what
makes changing an interpretation a seconds-long re-run rather than a re-synchronisation.

Events are named by channel **kind**, never by meaning — an LED is not necessarily a cue
and a sensor touch is not necessarily a response. Those differ per trial type and belong
to analysis.

The camera and scales channels are excluded: they are timing streams, not behaviour,
and would add ~1.8 million rows to a ~25,000-row file. They get their own files. The
trial-marker channel *is* included — it is doing
double duty, since the same transitions define ``in_trial``, but omitting it would make
this an incomplete record of the DAQ for the sake of 1,660 rows.

**A dimmed LED is one period, not its PWM.** An LED below full brightness is pulsed at
about 1 kHz, and the DAQ sees every pulse: the flicker protocol's dim distractor LEDs put
~80 million rows in a session's file and an NWB build ran out of memory reading it. A run
of short pulses (each, and each gap, under ``PWM_MAX_S``; at least ``PWM_MIN_PULSES`` of
them) is written as one ``led_dim_on`` at its first rise and one ``led_dim_off`` at its
last fall. ``led_on``/``led_off`` keep meaning a step to and from steady light, so a cue
found as "the first ``led_on``" is never a distractor's pulse. The duty cycle is in the
result's notes rather than a column.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from ..utils.clock import assign_to_windows
from ._result import OutputResult

COLUMNS = ("t", "event", "channel", "index", "in_trial", "trial")

#: Per-port channels carry their port in the name. Only these are indexed — a trailing
#: digit on DAQ_LINK1 or EXT_1 is part of the channel's identity, not a port number,
#: and treating it as one would make link events look like port 0 events.
_PORT_CHANNEL = re.compile(r"^(SENSOR|LED_|VALVE)(\d+)$")

#: A high pulse, and the low gap after it, both shorter than this are PWM, not light
#: switched on and off. LED PWM runs at ~1 kHz (2 ms at the Giga's 500 Hz); nothing a
#: protocol times is shorter than tens of milliseconds.
PWM_MAX_S = 0.005
#: Fewer short pulses in a row than this are written as they are.
PWM_MIN_PULSES = 3


def _port_of(channel):
    """
    The port a channel serves, read from its name; ``-1`` if it serves none.

    No arithmetic: names are 0-based, so the number in the name *is* the port.
    """
    match = _PORT_CHANNEL.match(channel)
    return int(match.group(2)) if match else -1


def sensor_touches(daq, channels):
    """
    Time and port of every beam break in the session, in time order, as two arrays.
    ``channels`` is a list of channel names; only the sensors among them are read.

    Sensors are active-low, so a touch is a *falling* edge. Shared with the trials
    writer, which needs to know when a trial's chosen port was first touched — the same
    edges, read once.
    """
    times, ports = [], []
    for channel in channels:
        if not channel.startswith("SENSOR") or not daq.has(channel):
            continue
        data = daq.channel(channel)
        changed = np.flatnonzero(data[1:] != data[:-1]) + 1
        falling = changed[data[changed] <= 0]
        if len(falling) == 0:
            continue
        times.append(daq.timestamps[falling])
        ports.append(np.full(len(falling), _port_of(channel), dtype=int))

    if not times:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)

    every, port = np.concatenate(times), np.concatenate(ports)
    order = np.argsort(every, kind="stable")
    return every[order], port[order]


def _collapse_pwm(t, rising):
    """
    An LED's transitions with each PWM run replaced by its first rise and last fall.

    Returns the times kept, their event names, the median duty cycle of the runs (NaN if
    none) and how many pulses went into runs.
    """
    names = np.where(rising, "led_on", "led_off").astype(object)
    # Complete pulses: a rise followed directly by a fall.
    first = 1 if len(rising) and not rising[0] else 0
    up = np.arange(first, len(t) - 1, 2)
    up = up[rising[up] & ~rising[up + 1]]
    if len(up) < PWM_MIN_PULSES:
        return t, list(names), np.nan, 0
    width = t[up + 1] - t[up]
    short = width < PWM_MAX_S
    # Two neighbouring pulses belong to one run when both are short and the gap is too.
    consecutive = up[1:] == up[:-1] + 2
    linked = short[:-1] & short[1:] & consecutive & (t[up[1:]] - t[up[:-1] + 1] < PWM_MAX_S)
    run = np.r_[0, np.cumsum(~linked)]
    keep = np.ones(len(t), dtype=bool)
    duties, in_runs = [], 0
    for label in np.flatnonzero(np.bincount(run) >= PWM_MIN_PULSES):
        pulses = up[run == label]
        start, stop = pulses[0], pulses[-1] + 1
        keep[start + 1:stop] = False
        names[start], names[stop] = "led_dim_on", "led_dim_off"
        if t[stop] > t[start]:
            duties.append(width[run == label].sum() / (t[stop] - t[start]))
        in_runs += len(pulses)

    duty = np.nan
    if duties:
        duty = float(np.median(duties))
    return t[keep], list(names[keep]), duty, in_runs


def write(out_dir, daq, channels, starts, ends):
    """
    Write every transition on ``channels`` (a list of channel names).

    Which channels those are is decided by the pipeline, so this module does not need
    to know what any channel is for — only how to name what it did.
    """
    path = Path(out_dir) / "events.csv"
    wanted = [name for name in channels if daq.has(name)]

    times, names, chans, indices = [], [], [], []
    notes = []
    for channel in wanted:
        data = daq.channel(channel)
        changed = np.flatnonzero(data[1:] != data[:-1]) + 1
        if len(changed) == 0:
            continue
        rising = data[changed] > 0
        index = _port_of(channel)

        stamps = daq.timestamps[changed]
        if channel.startswith("LED"):
            stamps, named, duty, in_runs = _collapse_pwm(stamps, rising)
            if in_runs:
                notes.append(f"{channel}: {in_runs:,} PWM pulses written as "
                             f"{named.count('led_dim_on'):,} dim period(s), duty ~{100 * duty:.0f}%")
        else:
            # Named by the channel's kind, not by what it was used for. Sensors are
            # **active-low**: the beam reads high when unbroken, so a touch is a
            # falling edge.
            if channel.startswith("SENSOR"):
                on, off = "sensor_release", "sensor_touch"
            elif channel.startswith("VALVE"):
                on, off = "valve_open", "valve_close"
            else:
                on, off = "link_high", "link_low"
            named = [on if up else off for up in rising]
        times.append(stamps)
        names.extend(named)
        chans.extend([channel] * len(stamps))
        indices.extend([index] * len(stamps))

    if not times:
        path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
        return OutputResult(name="events", path=path, n_rows=0,
                            warnings=("no behaviour channels had any transitions",))

    t = np.concatenate(times)
    order = np.argsort(t, kind="stable")
    t = t[order]
    names = [names[i] for i in order]
    chans = [chans[i] for i in order]
    indices = [indices[i] for i in order]

    trial = assign_to_windows(t, starts, ends)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(
            (f"{t[i]:.6f}", names[i], chans[i], indices[i],
             1 if trial[i] >= 0 else 0, trial[i])
            for i in range(len(t))
        )

    return OutputResult(name="events", path=path, n_rows=len(t), warnings=tuple(notes))
