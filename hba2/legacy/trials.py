"""
Trials for a session recorded before the trial-marker channel existed.
=====================================================================

A current rig holds ``DAQ_LINK1`` high for each trial and writes its own trials CSV, and
post-processing copies both. The old rig did neither, so here the trials are **found on
the DAQ** — the hardware record, and the one clock — and the rig's serial log is used
only to say what the rig itself believed, beside it.

What a trial is, on the DAQ
---------------------------
* **Start** is a cue going on: an ``LED_<port>`` or ``GO_CUE`` rising edge. Onsets
  each within :data:`MERGE_S` of the one before are one trial — an audio trial on the red rigs
  lights ``LED_0`` as it sounds, and a phase-10 catch trial is a light and a tone
  together. In phases 1 and 2 nothing is cued: the trial is the free reward, so a
  ``VALVE<port>`` opening starts it.
* **Response** is the first *new* beam break after the start — a falling edge, since
  the sensors are active-low — and before the trial ended: the rig's logged completion
  where the log pairs, else the next trial's start. A beam already broken
  at the cue is not a response; the trial is kept and flagged, never dropped.
* **End** is the response. With no response it is where the rig's log says the trial
  finished, failing that the end of the tone, failing that the next trial's start.

``correct_port`` is the port whose LED lit. An audio trial lights none, so its port
comes from the rig's log where the two pair up, and is port 0 otherwise — the audio
port on every rig that ran them. ``chosen_port`` is the port of the response.
``outcome`` compares the two. All three are therefore read off the hardware; the log's
own view of each is kept in ``details_json`` and every disagreement listed there.

Pairing with the log
--------------------
The log's acknowledgement times are on the PC clock. They are paired with cue onsets by
finding the single offset that lines most of them up (the mode of every pairwise
difference), then aligning the two sequences in order, refitting a straight line to
absorb drift, and aligning again; no pair may be further apart than
:data:`PAIR_TOLERANCE_S`. Pairing keeps
both in order but lets either skip, so a missing or extra cue costs that one trial its
pair rather than shifting every trial after it.

Ports the DAQ could not see
---------------------------
On some rigs and dates a port's lines were not recorded: its LED never changes, or is
stuck high, and its sensor never changes (rig 3 in spring 2025 lost ports 1 and 2 this
way for weeks). The rig's log is then the only record for that port, and is used for it
and nothing else:

* a log trial cueing a **blind LED** that paired with no DAQ cue is added at the onset
  the fitted line puts it at, flagged ``onset_from_rig_log``;
* a trial whose log credits a **blind sensor** takes that port as its response, at the
  log's completion time on the fitted line, flagged ``response_from_rig_log``. The
  response is not added to ``events.csv``, which stays a record of the hardware.

    detect          the whole session: cues, pairing, one row per trial
    _cues           cue onsets on the DAQ, grouped into trials
    _touches        every beam break
    _pair, _align   cue onsets against the log's acknowledgements
    _describe       one trial: what was cued, touched, and where it ended
    LegacyTrials    what comes back
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..postprocessing.sources.trials_csv import TrialRow, TrialRows
from .phase_log import LogTrial

N_PORTS = 6

#: Cue onsets closer than this are one trial.
MERGE_S = 0.05

#: How far a paired log acknowledgement may sit from its cue, after the fit. Generous,
#: because the older rigs' PC timestamps jitter by a few hundred milliseconds; trials
#: are seconds apart, and pairing keeps their order, so it cannot cross a trial.
PAIR_TOLERANCE_S = 1.0

#: How long after the log's completion a touch may still be the response — the
#: completion message reaches the PC after the touch, and the line has some scatter.
LOG_DONE_SLACK_S = 0.5

#: What leaving a cue or a log entry unpaired costs, in seconds of mismatch.
GAP_COST_S = 0.6

#: Resolution of the first, coarse offset search.
OFFSET_BIN_S = 0.1

#: A cue channel high for more than this fraction of the session is stuck, not cueing.
STUCK_FRACTION = 0.5

#: Phases whose trials are free rewards, not cues.
REWARD_PHASES = ("1", "2")

#: The columns of a rig trials CSV this converter fills in, for ``session.json``'s
#: record of which were absent.
COLUMNS = ("tracker", "trial_type", "trial_number", "outcome", "correct_port",
           "chosen_port", "stage", "event_log", "details_json")


def detect(daq, phase, log, source, nonspatial_cues=()):
    """
    Find every trial on the DAQ and describe it.

    Args:
        daq: the session's DAQ, on its clock.
        phase: the behaviour phase as a string, or None.
        log: the session's :class:`~.phase_log.PhaseLog`, or None without one.
        source: the file (a Path) the rows are said to come from.
        nonspatial_cues: a tuple of channels that also start a trial but name no port —
            on one 2025 non-spatial-cue cohort the cue was driven from ``LASER``.
            Such a trial's port can only come from the rig's log.

    Returns a :class:`LegacyTrials`.
    """
    phase = (phase or "").strip()

    # ---- 1. Cues on the DAQ, paired with the log ---------------------------
    cues, stuck = _cues(daq, phase, nonspatial_cues)
    log_trials = log.trials if log is not None else []

    pairs, fit = _pair([c.t_on for c in cues], [t.t_cued for t in log_trials])
    for i, j in pairs.items():
        cues[i].from_log = log_trials[j]

    # ---- 2. Cues the hardware could not show --------------------------------
    # Log trials on a stuck channel that paired with nothing, placed where the fitted
    # line says they happened.
    blind_leds = {int(n.removeprefix("LED_")) for n in stuck if n.startswith("LED_")}
    blind_sensors = {int(n.removeprefix("SENSOR")) for n in stuck if n.startswith("SENSOR")}
    if fit is not None and blind_leds:
        used = set(pairs.values())
        for j, trial in enumerate(log_trials):
            if j in used or trial.cued_port not in blind_leds:
                continue
            cue = Cue(t_on=fit[0] * trial.t_cued + fit[1], leds={trial.cued_port},
                      from_log=trial, recovered=True)
            cues.append(cue)
        cues.sort(key=lambda c: c.t_on)

    # ---- 3. Each trial -------------------------------------------------------
    touch_t, touch_port = _touches(daq)
    sensors = [(port, daq.channel(f"SENSOR{port}")) for port in range(N_PORTS)
               if daq.has(f"SENSOR{port}")]
    end_of_session = float(daq.timestamps[-1])

    starts, ends, rows, extra = [], [], [], []
    for index, cue in enumerate(cues):
        next_on = cues[index + 1].t_on if index + 1 < len(cues) else end_of_session
        # Which ports' beams were already broken as the cue came on.
        at = int(np.searchsorted(daq.timestamps, cue.t_on, side="left")) - 1
        held = []
        if at >= 0:
            held = [port for port, data in sensors if data[at] == 0]
        row, t_end, from_log = _describe(index, cue, phase, next_on, touch_t,
                                         touch_port, held, blind_sensors, fit)
        starts.append(cue.t_on)
        ends.append(t_end)
        rows.append(row)
        if from_log is not None:
            extra.append(from_log)

    # ---- 4. What the log ran that the DAQ never showed -----------------------
    paired_logs = {id(c.from_log) for c in cues if c.from_log is not None}
    log_only = [(round(t.t_cued, 3), t.cued_port) for t in log_trials
                if id(t) not in paired_logs]
    log_touches = (np.array([t for t, _ in extra], dtype=float),
                   np.array([p for _, p in extra], dtype=int))

    return LegacyTrials(
        log_only=log_only,
        starts=np.asarray(starts, dtype=float),
        ends=np.asarray(ends, dtype=float),
        rows=TrialRows(path=source, rows=rows, columns=list(COLUMNS)),
        n_log_trials=len(log_trials),
        n_paired=len(pairs),
        stuck_channels=stuck,
        pc_to_daq=fit,
        log_touches=log_touches,
    )


# =====================================================================
# Finding cues
# =====================================================================

@dataclass
class Cue:
    """One trial as found on the DAQ, before anything is decided about it."""

    t_on: float
    leds: set[int] = field(default_factory=set)
    audio: bool = False
    valve: int | None = None
    #: A cue on a channel that names no port, e.g. a whole-arena light.
    nonspatial: bool = False
    t_audio_off: float | None = None
    from_log: LogTrial | None = None
    #: Placed from the rig log because its LED channel recorded nothing.
    recovered: bool = False


def _changes(daq, name):
    """Whether a channel ever changes state."""
    data = daq.channel(name)
    return bool(np.any(data[1:] != data[:-1]))


def _cues(daq, phase, nonspatial=()):
    """
    Every cue onset, grouped into trials: a list of :class:`Cue`. Also a list of which
    LED and sensor channels are stuck. ``nonspatial`` is a tuple of channel names.
    """
    onsets = []                 # (time, kind, port or None)
    stuck = []
    duration = max(daq.duration, 1e-9)

    if phase in REWARD_PHASES:
        for port in range(N_PORTS):
            name = f"VALVE{port}"
            if daq.has(name):
                onsets += [(t, "valve", port) for t in daq.rising_edges(name)]
    else:
        for port in range(N_PORTS):
            name = f"LED_{port}"
            if not daq.has(name):
                continue
            # Fraction of the session the channel spent high, from its edges.
            data = daq.channel(name)
            up, down = daq.rising_edges(name), daq.falling_edges(name)
            if data[0]:
                up = np.concatenate(([daq.timestamps[0]], up))
            if data[-1]:
                down = np.concatenate((down, [daq.timestamps[-1]]))
            n = min(len(up), len(down))
            high_fraction = float(np.sum(down[:n] - up[:n])) / duration
            if high_fraction > STUCK_FRACTION or not _changes(daq, name):
                stuck.append(name)
                continue
            onsets += [(t, "led", port) for t in daq.rising_edges(name)]
        if daq.has("GO_CUE"):
            onsets += [(t, "audio", None) for t in daq.rising_edges("GO_CUE")]
        for name in nonspatial:
            if daq.has(name):
                onsets += [(t, "nonspatial", None) for t in daq.rising_edges(name)]

    stuck += [f"SENSOR{port}" for port in range(N_PORTS)
              if daq.has(f"SENSOR{port}") and not _changes(daq, f"SENSOR{port}")]

    onsets.sort(key=lambda o: o[0])
    audio_off = daq.falling_edges("GO_CUE") if daq.has("GO_CUE") else np.empty(0)

    cues = []
    last = -np.inf
    for t, kind, port in onsets:
        # Chained: an onset joins the trial if it is close to the *previous* onset, so
        # a flickering cue — a phase-10 catch light pulses every 10 ms — stays one trial.
        if not cues or t - last > MERGE_S:
            cues.append(Cue(t_on=float(t)))
        last = t
        cue = cues[-1]
        if kind == "led":
            cue.leds.add(port)
        elif kind == "audio":
            cue.audio = True
            after = audio_off[audio_off > t]
            cue.t_audio_off = float(after[0]) if len(after) else None
        elif kind == "nonspatial":
            cue.nonspatial = True
        else:
            cue.valve = port
    return cues, stuck


def _touches(daq):
    """
    Every beam break, as arrays of times and ports, in time order. Active-low: falling
    edges.
    """
    times, ports = [], []
    for port in range(N_PORTS):
        name = f"SENSOR{port}"
        if daq.has(name):
            edges = daq.falling_edges(name)
            times.append(edges)
            ports.append(np.full(len(edges), port))
    if not times:
        return np.empty(0), np.empty(0, dtype=int)
    t, p = np.concatenate(times), np.concatenate(ports)
    order = np.argsort(t, kind="stable")
    return t[order], p[order]


# =====================================================================
# Pairing with the log
# =====================================================================

def _pair(onsets, cued):
    """
    Pair two lists of times: DAQ onsets and the log's PC times.

    Returns ``{onset index: log index}`` and the PC-to-DAQ line used, ``(slope,
    intercept)`` — or ``{}`` and None when nothing lines up.
    """
    if not onsets or not cued:
        return {}, None
    a, b = np.asarray(onsets), np.asarray(cued)

    # The offset that lines most acknowledgements up with a cue.
    diffs = (a[:, None] - b[None, :]).ravel()
    bins = np.round(diffs / OFFSET_BIN_S).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    # Neighbouring bins summed, so jitter straddling a bin edge still counts together.
    lookup = dict(zip(values, counts))
    smoothed = np.array([lookup.get(v - 1, 0) + c + lookup.get(v + 1, 0)
                         for v, c in zip(values, counts)])
    if smoothed.max() < min(3, len(a), len(b)):
        return {}, None
    fit = (1.0, float(values[np.argmax(smoothed)] * OFFSET_BIN_S))

    pairs = {}
    for _ in range(3):                      # align, refit to a line, align again
        pairs = _align(a, fit[0] * b + fit[1])
        if len(pairs) >= 3:
            i, j = np.array(list(pairs)), np.array(list(pairs.values()))
            slope, intercept = np.polyfit(b[j], a[i], 1)
            fit = (float(slope), float(intercept))
    return pairs, fit


def _align(a, mapped):
    """
    Pair two sorted arrays of times in order, allowing either to skip entries. Returns
    ``{index in a: index in mapped}``.

    Order-preserving alignment rather than nearest-time matching, because the PC's
    timestamps jitter by up to a few hundred milliseconds on the older rigs while trials
    are seconds apart: the order is certain even where single times are not. A pair
    costs its time difference; leaving an entry unpaired costs :data:`GAP_COST_S`; pairs
    further apart than :data:`PAIR_TOLERANCE_S` are not allowed at all.
    """
    n, m = len(a), len(mapped)
    cost = np.full((n + 1, m + 1), np.inf)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)     # 1 pair, 2 skip a, 3 skip b
    cost[0, :] = np.arange(m + 1) * GAP_COST_S
    cost[:, 0] = np.arange(n + 1) * GAP_COST_S
    move[1:, 0], move[0, 1:] = 2, 3
    for i in range(1, n + 1):
        gap_a = cost[i - 1, :] + GAP_COST_S
        for j in range(1, m + 1):
            best, step = gap_a[j], 2
            if cost[i, j - 1] + GAP_COST_S < best:
                best, step = cost[i, j - 1] + GAP_COST_S, 3
            d = abs(a[i - 1] - mapped[j - 1])
            if d <= PAIR_TOLERANCE_S and cost[i - 1, j - 1] + d < best:
                best, step = cost[i - 1, j - 1] + d, 1
            cost[i, j], move[i, j] = best, step

    pairs = {}
    i, j = n, m
    while i > 0 or j > 0:
        step = move[i, j]
        if step == 1:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif step == 2:
            i -= 1
        else:
            j -= 1
    return dict(sorted(pairs.items()))


# =====================================================================
# One trial
# =====================================================================

def _describe(index, cue, phase, next_on, touch_t, touch_port, held, blind_sensors, fit):
    """
    One trial's row, its end time and, when the response came from the rig log, that
    response as ``(time, port)`` (else None).

    ``held`` is the list of ports whose beams were broken as the cue came on;
    ``blind_sensors`` the set of ports whose sensors recorded nothing.
    """
    flags = []
    log = cue.from_log

    # ---- what was cued -------------------------------------------------
    if cue.valve is not None:
        kind, correct = "reward", cue.valve
    elif cue.nonspatial and not cue.leds and not cue.audio:
        kind = "nonspatial"
        correct = log.cued_port if log is not None else None
    elif cue.audio and (not cue.leds or (cue.leds == {0} and phase != "10")):
        kind = "audio"
        correct = log.cued_port if log is not None and log.cued_port is not None else 0
    elif cue.audio:
        kind = "catch"
        correct = min(cue.leds - {0}) if cue.leds - {0} else 0
    else:
        kind = "visual"
        correct = min(cue.leds) if cue.leds else None
        if len(cue.leds) > 1:
            flags.append(f"several_leds:{sorted(cue.leds)}")

    # ---- what was touched ----------------------------------------------
    # Catch trials are answered after the tone, as the rig scored them.
    search_from = cue.t_audio_off if kind == "catch" and cue.t_audio_off else cue.t_on
    # The trial is over when the rig says so — a touch after its timeout answered
    # nothing — so the search stops at the log's completion, where there is one.
    search_to = next_on
    if log is not None and log.t_done is not None and fit is not None:
        search_to = min(search_to, fit[0] * log.t_done + fit[1] + LOG_DONE_SLACK_S)
    window = (touch_t > search_from) & (touch_t < search_to)
    hit = np.flatnonzero(window)
    t_response = None
    chosen = None
    if len(hit):
        t_response = float(touch_t[hit[0]])
        chosen = int(touch_port[hit[0]])

    # A port the DAQ could not see: the log is the only record of a touch there.
    from_log = None
    if (log is not None and log.chosen_port in blind_sensors and log.t_done is not None
            and fit is not None):
        t_log = fit[0] * log.t_done + fit[1]
        if t_response is None or t_log < t_response:
            t_response, chosen = float(t_log), log.chosen_port
            from_log = (t_response, chosen)
            flags.append("response_from_rig_log")

    if held:
        flags.append(f"sensor_held_at_cue:{held}")

    if chosen is None:
        outcome = "timeout"
    elif correct is None:
        outcome = "failure"
        flags.append("no_cued_port")
    else:
        outcome = "success" if chosen == correct else "failure"

    # ---- where it ended ------------------------------------------------
    if t_response is not None:
        t_end = t_response
    elif log is not None and log.t_done is not None:
        t_end = min(cue.t_on + (log.t_done - log.t_cued), next_on)
    elif cue.t_audio_off is not None:
        t_end = min(cue.t_audio_off, next_on)
    else:
        t_end = next_on
        flags.append("end_is_next_trial")

    # ---- the rig's own view, and where it disagrees --------------------
    rig_log = None
    events = []
    if log is not None:
        rig_log = {"cued_port": log.cued_port, "chosen_port": log.chosen_port,
                   "success": log.success}
        events.append({"t": 0.0, "name": "rig_log_cue", "port": log.cued_port})
        if log.t_done is not None:
            events.append({"t": round(log.t_done - log.t_cued, 6),
                           "name": "rig_log_completion", "port": log.chosen_port,
                           "success": log.success})
        if kind not in ("audio", "nonspatial") and log.cued_port is not None \
                and log.cued_port != correct:
            flags.append("rig_log_cued_port_differs")
        if log.t_done is not None and log.chosen_port != chosen:
            flags.append("rig_log_chosen_port_differs")
        if log.success is not None and log.success != (outcome == "success"):
            flags.append("rig_log_outcome_differs")
    else:
        flags.append("no_rig_log_pair")
    if cue.recovered:
        flags.append("onset_from_rig_log")

    software_time = None
    software_duration = None
    if log is not None:
        software_time = log.t_cued
        if log.t_done is not None:
            software_duration = log.t_done - log.t_cued

    row = TrialRow(
        index=index,
        trial_number=index,
        tracker=f"phase_{phase}" if phase else "legacy",
        trial_type=kind,
        stage=phase or "",
        outcome=outcome,
        software_time=software_time,
        software_duration=software_duration,
        correct_port=correct,
        chosen_port=chosen,
        events=events,
        details={"source": "legacy_converter", "cue": kind,
                 "cue_leds": sorted(cue.leds), "cue_audio": cue.audio,
                 "rig_log": rig_log, "flags": flags},
    )
    return row, float(t_end), from_log


# =====================================================================
# What comes out
# =====================================================================

@dataclass
class LegacyTrials:
    """Trial windows on the DAQ clock and the rows that describe them."""

    starts: np.ndarray
    ends: np.ndarray
    rows: TrialRows
    n_log_trials: int
    n_paired: int
    stuck_channels: list[str]               #: LED and sensor channels that saw nothing
    pc_to_daq: tuple[float, float] | None   #: slope, intercept; None if unpaired
    #: Logged trials that no DAQ cue paired with and that could not be recovered:
    #: ``(PC time, cued port)``. The rig thought it ran them; the hardware shows nothing.
    log_only: list = field(default_factory=list)
    #: Responses on blind sensors, taken from the log: (times, ports). Passed to the
    #: trials writer beside the real beam breaks, so ``t_response`` is filled for them.
    log_touches: tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0, dtype=int))

    @property
    def warnings(self):
        """The lines worth a warning in the conversion report, as a list of strings."""
        out = []
        # A channel that recorded nothing only matters when a trial needed it; an
        # unused port's LED is flat in most sessions. Those are in session.json.
        flags = []
        for row in self.rows.rows:
            flags += row.details["flags"]
        recovered = flags.count("onset_from_rig_log")
        from_log = flags.count("response_from_rig_log")
        if recovered or from_log:
            out.append(f"DAQ blind on {', '.join(self.stuck_channels)}: {recovered} cue(s) "
                       f"and {from_log} response(s) taken from the rig log")
        if self.n_log_trials and self.n_paired < 0.9 * min(self.n_log_trials,
                                                             len(self.starts)):
            out.append(f"only {self.n_paired} of {len(self.starts)} DAQ trials paired "
                       f"with the rig log ({self.n_log_trials} logged)")
        return out
