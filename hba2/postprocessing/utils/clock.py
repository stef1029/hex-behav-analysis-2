"""
Placing streams on the DAQ clock.

**This is the only module in the package that derives a timestamp.** Sources parse; this
places; nothing downstream ever re-derives an alignment. If a time looks wrong, there is
exactly one file to read.

Only two streams actually need placing, and they use the identical mechanism: the camera
and the scales each number their entries in the order the DAQ pulsed for them, so entry
*n* happened at the *n*-th rising edge. Behaviour channels need nothing — they *are* the
DAQ. Ephys needs a mapping rather than a conversion, so behaviour stays on the DAQ clock
whether or not a probe was present: :func:`map_ephys_to_daq` makes it, and
:func:`ephys_to_daq` applies it — to spikes only, when a curation is exported.

Two failure modes are made explicit rather than hidden, because both occur in real
sessions:

* Counts never match exactly. A session with 240,661 frames may have 240,659 camera
  pulses. Entries beyond the pulses recorded get **NaN, never 0** — a zero would place
  them at the start of the recording and look entirely plausible in a plot.
* A stream may not be placeable at all, in which case that fact is returned rather than
  quietly substituted.

The page runs in the order the pipeline uses it:

    trial_windows, assign_to_windows   the trials, and which one a time falls in
    recording_start                    where the camera's frames begin
    place_by_pulse_index               the camera and the scales
    map_ephys_to_daq, ephys_to_daq     the ephys clock
    Placement, EphysMap                what comes back
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The camera recorder stops the camera and waits 0.5 s before it starts recording
#: (behaviour_camera, from October 2026). A camera left exposing by an earlier session
#: therefore shows on the DAQ as pulses, a silence, then this recording's frames — and
#: frame ID 0 is the first pulse after the silence, not the first pulse. Below the
#: recorder's 0.5 s so DAQ timing jitter cannot hide it; well above anything seen
#: mid-recording (DAQ stalls reach ~0.1 s).
START_GAP_S = 0.3
#: The silence can only be at the start: searched for this long after the first pulse.
START_WINDOW_S = 10.0
#: Pulses the frames may outnumber after a skip before the skip is judged wrong — the
#: one or two a pulse/frame count is off by at the ends of a normal session.
START_TOLERANCE = 3

#: A row whose offset is further than this from the median of the rows around it is
#: treated as a glitch. Real drift moves a few microseconds a second.
GLITCH_S = 0.005
GLITCH_NEIGHBOURS = 3

#: How far past the first or last row of ``ephys_sync.csv`` an ephys time may still be
#: mapped, by carrying the end rows' line on. The rows are a second apart and anchored
#: mid-window, so the first and last half-second of a session lie outside them; beyond a
#: couple of seconds there was no DAQ to pair with and NaN is the honest answer.
EPHYS_REACH_S = 2.0


# =====================================================================
# Trials
# =====================================================================

def trial_windows(daq, trial_marker):
    """
    Start and end times of every trial, from the marker channel.

    The controller holds this channel high for the duration of each trial, so the edges
    *are* the trials — on real sessions the pulse count matches the trials CSV row count
    exactly. Returns ``(starts, ends)``, equal-length arrays, truncated to whichever is
    shorter if the recording ended mid-trial.
    """
    starts = daq.rising_edges(trial_marker)
    ends = daq.falling_edges(trial_marker)
    n = min(len(starts), len(ends))
    return starts[:n], ends[:n]


def assign_to_windows(times, starts, ends):
    """
    Which trial window each time falls in; ``-1`` for none.

    Window containment is deliberately mechanical: a valve opening 50 ms after the
    marker drops is *not* assigned to the trial it rewarded. Widening that is an
    interpretation, and interpretations belong to analysis.
    """
    times = np.asarray(times, dtype=np.float64)
    if len(starts) == 0:
        return np.full(len(times), -1, dtype=np.int64)

    candidate = np.searchsorted(starts, times, side="right") - 1
    inside = (candidate >= 0) & (times <= ends[np.clip(candidate, 0, len(ends) - 1)])
    return np.where(inside, candidate, -1).astype(np.int64)


# =====================================================================
# The camera and the scales
# =====================================================================

def recording_start(pulse_times, frame_ids):
    """
    Index of the first camera pulse that belongs to this recording.

    0 unless the start gap is there to see. Usually it is not: a camera that was idle
    when the session began has no pulses before the recording, so the silence before it
    is just the DAQ's own start. It only shows when the camera was already exposing —
    the case that otherwise puts every frame on the wrong pulse, by as many pulses as
    came before the gap.

    The latest qualifying silence in the window is taken, and a skip is refused if it
    would leave fewer pulses than the frame IDs need: a skip that cannot be right is not
    made, so a misread here can only fall back to what was done before.
    """
    pulse_times = np.asarray(pulse_times, dtype=np.float64)
    frame_ids = np.asarray(frame_ids)
    if len(pulse_times) < 2 or len(frame_ids) == 0:
        return 0
    in_window = pulse_times < pulse_times[0] + START_WINDOW_S
    gaps = np.flatnonzero(np.diff(pulse_times[in_window]) >= START_GAP_S)
    if len(gaps) == 0:
        return 0
    start = int(gaps[-1]) + 1
    needed = int(frame_ids.max()) + 1
    if len(pulse_times) - start >= needed - START_TOLERANCE:
        return start
    return 0


def place_by_pulse_index(name, indices, pulse_times, skip=0):
    """
    Time entries that are numbered by the pulses which produced them; a :class:`Placement`.

    Used for both the camera (``frame_ID``) and the scales (the column the rig calls
    ``message_id``, which is in fact a reading counter — treating it as a real DAQ
    message ID matches under 1% of readings, treating it as an index matches all of
    them).

    ``skip``: pulses at the start that belong to no entry, so entry 0 is pulse
    ``skip`` — the camera's, from :func:`recording_start`.
    """
    pulse_times = np.asarray(pulse_times)[skip:]
    indices = np.asarray(indices)
    times = np.full(len(indices), np.nan, dtype=np.float64)
    in_range = (indices >= 0) & (indices < len(pulse_times))
    times[in_range] = pulse_times[indices[in_range]]

    return Placement(
        name=name,
        times=times,
        n_total=len(indices),
        n_unmatched=int((~in_range).sum()),
        n_pulses=len(pulse_times),
        n_skipped=skip,
    )


# =====================================================================
# The ephys clock
# =====================================================================

def map_ephys_to_daq(pulse_times, message_ids, daq_times, every_s=1.0):
    """
    Pair the ephys clock with the DAQ's, through the sync pulse each message produced;
    an :class:`EphysMap`.

    The DAQ pulses once per loop and records a message on a subset of loops, numbered by
    loop, so message *m* happened at sync pulse *m* — joined by counting, never by
    comparing times.

    The DAQ's own timestamps jitter by about 60 µs per message and occasionally by a few
    milliseconds, while the ephys pulses are exact to the sample. So each row takes the
    **median** offset of the messages in its ``every_s`` window rather than any single
    message's: the jitter goes, the drift (seconds-to-minutes scale) stays.
    """
    message_ids = np.asarray(message_ids, dtype=np.int64)
    matched = (message_ids >= 0) & (message_ids < len(pulse_times))
    ids = message_ids[matched]
    t_daq = np.asarray(daq_times, dtype=np.float64)[matched]
    t_ephys = np.asarray(pulse_times, dtype=np.float64)[ids]
    offset = t_ephys - t_daq

    window = np.floor((t_daq - t_daq[0]) / every_s).astype(np.int64)
    bounds = np.flatnonzero(np.diff(window)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [len(window)]))

    rows = []
    for start, stop in zip(starts, stops):
        middle = (start + stop) // 2
        pulse = ids[middle]
        median = float(np.median(offset[start:stop]))
        rows.append((pulse, pulse_times[pulse], pulse_times[pulse] - median, stop - start))
    pulse, ephys, daq, counts = (np.array(column) for column in zip(*rows))

    # The drift is smooth, so a row whose offset sits far from its neighbours' is a glitch
    # in the DAQ's timestamps, not the clocks moving. Dropping it lets the line run
    # straight through from the rows either side.
    offsets = ephys - daq
    local = np.array([np.median(offsets[max(0, i - GLITCH_NEIGHBOURS):i + GLITCH_NEIGHBOURS + 1])
                      for i in range(len(offsets))])
    sound = np.abs(offsets - local) <= GLITCH_S
    pulse, ephys, daq, counts = pulse[sound], ephys[sound], daq[sound], counts[sound]

    # Measured against the line exports use, ends extended: a flat clamp past the first
    # and last rows would charge up to half a window to the messages there.
    residual = np.abs(ephys_to_daq(t_ephys, ephys, daq) - t_daq) * 1000
    residual_ms_median = float(np.nanmedian(residual))
    residual_ms_p99 = float(np.nanpercentile(residual, 99))

    return EphysMap(pulse=pulse.astype(np.int64), t_ephys=ephys, t_daq=daq,
                    n_messages=counts.astype(np.int64), n_matched=int(matched.sum()),
                    n_unmatched=int((~matched).sum()),
                    residual_ms_median=residual_ms_median,
                    residual_ms_p99=residual_ms_p99,
                    n_glitch_rows=int((~sound).sum()))


def ephys_to_daq(t_ephys, rows_ephys, rows_daq, reach_s=EPHYS_REACH_S):
    """
    Ephys-clock times on the DAQ clock, through the rows :func:`map_ephys_to_daq` made.

    Straight lines between rows; the first and last segments carried on for ``reach_s``;
    **NaN beyond that, never a clamped value** — a spike from after the DAQ stopped has
    no DAQ time, and pinning it to the last row would invent one.
    """
    t_ephys = np.asarray(t_ephys, dtype=np.float64)
    out = np.interp(t_ephys, rows_ephys, rows_daq)
    if len(rows_ephys) >= 2:
        before = t_ephys < rows_ephys[0]
        after = t_ephys > rows_ephys[-1]
        slope_in = (rows_daq[1] - rows_daq[0]) / (rows_ephys[1] - rows_ephys[0])
        slope_out = (rows_daq[-1] - rows_daq[-2]) / (rows_ephys[-1] - rows_ephys[-2])
        out[before] = rows_daq[0] + (t_ephys[before] - rows_ephys[0]) * slope_in
        out[after] = rows_daq[-1] + (t_ephys[after] - rows_ephys[-1]) * slope_out
    out[(t_ephys < rows_ephys[0] - reach_s) | (t_ephys > rows_ephys[-1] + reach_s)] = np.nan
    return out


# =====================================================================
# What comes back
# =====================================================================

@dataclass(frozen=True)
class Placement:
    """One stream's entries, timed on the DAQ clock."""

    name: str
    times: np.ndarray      #: DAQ-clock seconds; NaN where unmatched
    n_total: int
    n_unmatched: int
    n_pulses: int
    #: Pulses ignored before the recording's start gap — see :func:`recording_start`.
    n_skipped: int = 0

    @property
    def fraction_matched(self):
        if self.n_total == 0:
            return 0.0
        return 1 - self.n_unmatched / self.n_total

    def describe(self):
        matched = self.n_total - self.n_unmatched
        line = (f"{self.name:<8} {matched:,}/{self.n_total:,} timed "
                f"({self.fraction_matched:.4%}) from {self.n_pulses:,} pulses")
        if self.n_skipped:
            line += f", {self.n_skipped:,} before the start gap ignored"
        return line


@dataclass(frozen=True)
class EphysMap:
    """
    The ephys clock against the DAQ's, sampled once per ``every_s`` of the session.

    ``t_daq`` at an ephys time between two rows is the straight line between them. The
    two clocks drift apart by a few parts per million — tens of milliseconds over two
    hours — and smoothly, so a row a second is far denser than the drift needs.
    """

    pulse: np.ndarray        #: the DAQ message ID — and sync pulse — each row is anchored on
    t_ephys: np.ndarray      #: that pulse's time on the ephys clock (s)
    t_daq: np.ndarray        #: the same moment on the DAQ clock (s)
    n_messages: np.ndarray   #: DAQ messages each row's offset was taken from
    n_matched: int           #: DAQ messages with a pulse
    n_unmatched: int         #: DAQ messages beyond the last pulse, or before the first
    #: How far single messages sit from the smoothed line, in milliseconds: the DAQ's
    #: own timestamp jitter, which the median removes.
    residual_ms_median: float
    residual_ms_p99: float
    #: Rows left out because their offset jumped away from their neighbours' — a DAQ
    #: glitch, seen at the very start of short sessions, not clock drift.
    n_glitch_rows: int = 0
