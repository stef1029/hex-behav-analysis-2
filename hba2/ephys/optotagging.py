"""
Optotagging: which good units fire, reliably and on time, to the laser.
======================================================================

An opsin-expressing cell answers a brief light pulse with a spike a few milliseconds
later, nearly every time, at nearly the same delay. This finds the units that do, in a
session that flashed the laser at the probe — the ``Optotagging`` protocol::

    from hba2.ephys import tag_cohort

    tags = tag_cohort("/path/to/cohort", sort="ks4_camnt",
                      curation="2026-09_manual")
    tags.units[tags.units["tagged"]]

**The method is the previous pipeline's first-spike test** (``optotagging_v2``,
``latency_mode="first_spike"``):

1. **Laser pulses** are the rising edges of the DAQ's ``EXT_1`` line — on the v2 rigs
   the laser TTL is wired there; the channel named ``LASER`` is empty. On the 2025
   rig-3 DAQ (2504, legacy) there is no ``EXT_1`` and the TTL is on ``LASER``, which is
   read instead (:data:`LEGACY_LASER_CHANNEL`). Pulses are grouped
   into bursts wherever the gap between onsets exceeds ``burst_gap_s`` (0.2 s), and a
   lone pulse is not a burst and is left out.
2. **Per pulse, per unit**: the latency of the **first** spike within ``window_s``
   (15 ms) of the onset, or none. Only the first, so a cell that fires a locked spike and
   then a sustained train is judged by the locked spike rather than smeared by the train.
3. **Only pulses the probe recorded count.** A pulse outside the unit's recorded span
   (``obs_start``–``obs_stop``) is not a pulse it failed to answer — it was not being
   listened to — so it is left out of the reliability and counted in
   ``n_pulses_unrecorded``. On one cohort three optotagging sessions' recordings stop
   early (after 8, 17 and 138 s, with the laser running 14–529 s); the previous pipeline scored eleven units
   there 80–100% reliable at zero jitter, which is the laser train mapped onto a
   recording that had already ended rather than any response.
4. **Tested** if it answered at least ``min_responses`` (20) pulses.
5. **Tagged** if its **reliability** (pulses answered / pulses) is at least
   ``min_reliability`` (0.5) and its **jitter** — the median absolute deviation of its
   first-spike latencies — is at most ``max_jitter_ms`` (1 ms).
6. **Answered too soon** is reported, not judged: ``early_share`` is the share of a
   unit's first spikes under ``EARLY_MS`` (1 ms) after the onset, faster than any
   light-driven spike — the cell's ongoing firing landing on the pulse. A cell firing
   steadily through a burst has some; the verdict counts them as answers.
7. **Artefact suspected** where a tagged unit's jitter is below ``artefact_jitter_ms``
   (0.05 ms): a real cell has some jitter, while the laser's electrical or photoelectric
   artefact, sorted into a unit, lands on the pulse at the same delay every time. These
   stay unresolved until artefact handling (onset blanking, waveform match) exists.

**Where the times come from.** The laser edges are read from the session's DAQ file with
the same reader post-processing uses (``postprocessing.sources.daq``), so they are on the
DAQ clock; the spikes are the curation export's ``t_daq``, on the same clock. Reading the
DAQ here rather than from ``processed/`` is deliberate: an optotagging session has no
controller ``trials.csv``, so post-processing does not run on it. On the session it was
checked on this gives unit 96 the previous pipeline's numbers (99.8% reliable, 5.6 ms, jitter 0.9
ms), and laser edges put on the ephys clock through the sync pulses instead agree with
these to within 0.04 ms jitter. The DAQ samples once a millisecond, so every latency is
known only to that.

Only the curation's **good** units are tested — the export holds nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..cohort import Cohort
from ..postprocessing.sources import daq as daq_source, metadata as metadata_source

#: The DAQ line the laser TTL is wired to on the v2 rigs.
LASER_CHANNEL = "EXT_1"
#: Where it was before them: the 2025 rig-3 DAQ (the Axona tetrode cohort)
#: had a line named ``LASER`` and no ``EXT_1``. Read when the default line is absent or
#: never went high, so ``tag_cohort`` runs on those sessions unchanged — pass
#: ``protocol=None`` there, as their protocol is ``legacy phase 9c``, not Optotagging.
LEGACY_LASER_CHANNEL = "LASER"
#: The protocol that flashes the laser at the probe.
PROTOCOL = "Optotagging"

BURST_GAP_S = 0.2
WINDOW_S = 0.015
MIN_RESPONSES = 20
MIN_RELIABILITY = 0.5
MAX_JITTER_MS = 1.0
ARTEFACT_JITTER_MS = 0.05
#: A "first spike" this soon after the pulse onset (ms) is too fast to be driven by the
#: light — it is the cell's own firing that happened to land there (or an artefact).
#: Reported as ``early_share``, the share of answered pulses that were answered this
#: soon; not used in the verdict, which is the previous pipeline's.
EARLY_MS = 1.0
#: Spikes kept around each burst's first pulse for a raster (s).
RASTER_S = (-1.0, 2.0)

#: ``units``' columns, in order.
UNIT_COLUMNS = ("session_id", "unit_id", "neuron_id", "shank", "depth_um",
                "firing_rate_hz", "n_pulses", "n_pulses_unrecorded", "n_responses",
                "reliability",
                "median_latency_ms", "jitter_ms", "early_share", "tested", "tagged",
                "artefact_suspect",
                "reason")


@dataclass
class Tags:
    """What one run found: a row per unit, and the numbers behind each verdict."""

    units: pd.DataFrame        #: one row per good unit per session — see UNIT_COLUMNS
    responses: pd.DataFrame    #: one row per unit per pulse: recorded, latency_ms (NaN if none)
    rasters: pd.DataFrame      #: spikes around each burst: session, unit, burst, t_ms
    pulses: pd.DataFrame       #: the laser pulses used: session, pulse, burst, t_on, t_off
    sessions: pd.DataFrame     #: per session: pulses, bursts, units tested and tagged
    settings: dict = field(default_factory=dict)

    @property
    def tagged(self):
        """The tagged units, artefact suspects included — filter on that column."""
        return self.units[self.units["tagged"]]

    def describe(self):
        units = self.units
        clean = units["tagged"] & ~units["artefact_suspect"]
        lines = [
            f"{len(self.sessions)} session(s) with laser, {len(units)} good unit(s), "
            f"{int(units['tested'].sum())} tested",
            f"  tagged: {int(units['tagged'].sum())} — of which artefact-suspect "
            f"{int(units['artefact_suspect'].sum())}, clean {int(clean.sum())}",
        ]
        # Then a line per clean tagged unit.
        for row in units[clean].itertuples():
            lines.append(f"  {row.session_id} unit {row.unit_id} ({row.neuron_id}): "
                         f"{row.reliability:.0%} reliable, {row.median_latency_ms:.1f} ms, "
                         f"jitter {row.jitter_ms:.2f} ms")
        return "\n".join(lines)


# =====================================================================
# One session
# =====================================================================

def laser_pulses(record, *, channel=LASER_CHANNEL, burst_gap_s=BURST_GAP_S):
    """
    The laser pulses of one session (a
    :class:`~hba2.cohort.SessionRecord`) on the DAQ clock, numbered
    into bursts.

    Returns ``pulse``, ``burst``, ``t_on``, ``t_off`` — empty if the line never went high
    (a behaviour session, or the laser not connected). A pulse alone, more than
    ``burst_gap_s`` from its neighbours, is left out.
    """
    daq = daq_source.read(record.path("daq"))
    # The 2025 rig-3 DAQ: no live EXT_1, so the legacy LASER line instead.
    if (channel == LASER_CHANNEL and daq.has(LEGACY_LASER_CHANNEL)
            and not (daq.has(channel) and len(daq.rising_edges(channel)))):
        channel = LEGACY_LASER_CHANNEL
    if not daq.has(channel):
        return pd.DataFrame(columns=["pulse", "burst", "t_on", "t_off"])
    ons, offs = daq.rising_edges(channel), daq.falling_edges(channel)
    if not len(ons):
        return pd.DataFrame(columns=["pulse", "burst", "t_on", "t_off"])
    # Each rise's own fall: the first fall after it, if one came before the next rise.
    after = np.searchsorted(offs, ons, side="right")
    t_off = np.where(after < len(offs), offs[np.minimum(after, len(offs) - 1)], np.nan)
    following = np.append(ons[1:], np.inf)
    t_off[t_off > following] = np.nan
    burst = np.concatenate(([0], np.cumsum(np.diff(ons) > burst_gap_s)))
    pulses = pd.DataFrame({"t_on": ons, "t_off": t_off, "burst": burst})
    size = pulses.groupby("burst")["t_on"].transform("size")
    pulses = pulses[size > 1].copy()
    pulses["burst"] = pd.factorize(pulses["burst"])[0]
    pulses.insert(0, "pulse", np.arange(len(pulses)))
    return pulses.reset_index(drop=True)


def first_spike_latencies(spike_times, t_on, window_s=WINDOW_S):
    """
    Per pulse, the delay to the first spike at or after its onset within ``window_s``, in
    ms; NaN where there was none. ``spike_times`` and ``t_on`` are arrays of seconds;
    ``spike_times`` must be sorted. Returns an array, one value per pulse.
    """
    following = np.searchsorted(spike_times, t_on, side="left")
    latency = np.full(len(t_on), np.nan)
    has = following < len(spike_times)
    latency[has] = spike_times[following[has]] - t_on[has]
    latency[latency > window_s] = np.nan
    return latency * 1000.0


def judge(latencies_ms, *, min_responses=MIN_RESPONSES, min_reliability=MIN_RELIABILITY,
          max_jitter_ms=MAX_JITTER_MS, artefact_jitter_ms=ARTEFACT_JITTER_MS):
    """
    The verdict on one unit from its per-pulse first-spike latencies (an array of ms,
    NaN = none), as a dict of the ``units`` columns it fills.
    """
    answered = latencies_ms[~np.isnan(latencies_ms)]
    n_pulses, n_responses = len(latencies_ms), len(answered)
    if n_pulses:
        reliability = n_responses / n_pulses
    else:
        reliability = np.nan
    if n_responses:
        median = float(np.median(answered))
        jitter = float(np.median(np.abs(answered - median)))
        early = float(np.mean(answered < EARLY_MS))
    else:
        median = jitter = early = np.nan
    tested = n_responses >= min_responses
    reasons = []
    if not tested:
        reasons.append(f"too few responses ({n_responses} < {min_responses})")
    else:
        if reliability < min_reliability:
            reasons.append(f"low reliability ({reliability:.2f} < {min_reliability})")
        if jitter > max_jitter_ms:
            reasons.append(f"high jitter ({jitter:.2f} > {max_jitter_ms} ms)")
    tagged = tested and not reasons
    artefact = bool(tagged and jitter < artefact_jitter_ms)
    if artefact:
        reasons.append(f"jitter {jitter:.3f} ms is below {artefact_jitter_ms} ms — "
                       "likely the laser artefact")
    if reasons:
        reason = "; ".join(reasons)
    else:
        reason = "reliable, low-jitter first spike"
    return {"n_pulses": n_pulses, "n_responses": n_responses, "reliability": reliability,
            "median_latency_ms": median, "jitter_ms": jitter, "early_share": early,
            "tested": tested,
            "tagged": tagged, "artefact_suspect": artefact,
            "reason": reason}


def tag_session(record, *, sort, curation, channel=LASER_CHANNEL,
                burst_gap_s=BURST_GAP_S, window_s=WINDOW_S, min_responses=MIN_RESPONSES,
                min_reliability=MIN_RELIABILITY, max_jitter_ms=MAX_JITTER_MS,
                artefact_jitter_ms=ARTEFACT_JITTER_MS, raster_s=RASTER_S) -> Tags | None:
    """
    Every good unit of one session's curation, tested against its laser pulses.

    ``record`` is the session's :class:`~hba2.cohort.SessionRecord`;
    ``raster_s`` is a (start, stop) pair of seconds around each burst's first pulse.
    None when the session had no laser pulses. Raises when the curation has not been
    exported, since its spikes are what is tested.
    """
    # ---- 1. The pulses, and the curation's units and spikes -------------
    pulses = laser_pulses(record, channel=channel, burst_gap_s=burst_gap_s)
    if pulses.empty:
        return None
    files = record.curation(sort, curation)
    if files.get("units") is None or files.get("spikes") is None:
        raise FileNotFoundError(f"{record.session_id}: curation {curation!r} of {sort!r} "
                                "has not been exported — run the ephys pipeline first")
    units = pd.read_csv(files["units"])
    spikes = pd.read_parquet(files["spikes"], columns=["unit_id", "t_daq"])
    if files.get("names") is not None:
        names = pd.read_csv(files["names"], dtype={"neuron_id": str})[["unit_id", "neuron_id"]]
    else:
        names = pd.DataFrame(columns=["unit_id", "neuron_id"])
    units = units.merge(names, on="unit_id", how="left")

    # ---- 2. Each unit against every pulse -------------------------------
    t_on = pulses["t_on"].to_numpy(float)
    first = pulses.groupby("burst")["t_on"].min()
    rows, responses, rasters = [], [], []
    for unit in units.itertuples():
        times = np.sort(spikes.loc[spikes["unit_id"] == unit.unit_id, "t_daq"].dropna()
                        .to_numpy(float))
        latency = first_spike_latencies(times, t_on, window_s)
        recorded = (t_on >= unit.obs_start) & (t_on + window_s <= unit.obs_stop)
        verdict = judge(latency[recorded], min_responses=min_responses,
                        min_reliability=min_reliability, max_jitter_ms=max_jitter_ms,
                        artefact_jitter_ms=artefact_jitter_ms)
        rows.append({"session_id": record.session_id, "unit_id": int(unit.unit_id),
                     "neuron_id": unit.neuron_id, "shank": unit.shank,
                     "depth_um": unit.depth_um, "firing_rate_hz": unit.firing_rate_hz,
                     "n_pulses_unrecorded": int((~recorded).sum()), **verdict})
        responses.append(pd.DataFrame({"session_id": record.session_id,
                                       "unit_id": int(unit.unit_id),
                                       "pulse": pulses["pulse"], "burst": pulses["burst"],
                                       "recorded": recorded,
                                       "latency_ms": np.where(recorded, latency, np.nan)}))
        low = np.searchsorted(times, first.to_numpy() + raster_s[0])
        high = np.searchsorted(times, first.to_numpy() + raster_s[1])
        for burst, start, a, b in zip(first.index, first.to_numpy(), low, high):
            if b > a:
                rasters.append(pd.DataFrame({"session_id": record.session_id,
                                             "unit_id": int(unit.unit_id), "burst": burst,
                                             "t_ms": (times[a:b] - start) * 1000.0}))

    # ---- 3. The tables --------------------------------------------------
    table = pd.DataFrame(rows, columns=UNIT_COLUMNS)
    pulses = pulses.assign(session_id=record.session_id,
                           t_on_ms=(pulses["t_on"] - pulses["burst"].map(first)) * 1000.0,
                           t_off_ms=(pulses["t_off"] - pulses["burst"].map(first)) * 1000.0)

    # The session's summary row. The stretch every unit was recorded through, and the
    # most pulses any unit missed, need units to say.
    if len(units):
        recorded_s = [float(units["obs_start"].max()), float(units["obs_stop"].min())]
    else:
        recorded_s = None
    if len(table):
        most_unrecorded = int(table["n_pulses_unrecorded"].max())
    else:
        most_unrecorded = 0
    summary = pd.DataFrame([{
        "session_id": record.session_id, "n_pulses": len(pulses),
        "n_bursts": int(pulses["burst"].nunique()),
        "pulse_ms": float(np.nanmedian(pulses["t_off"] - pulses["t_on"]) * 1000.0),
        "pulse_hz": float(1.0 / np.median(np.diff(t_on)[np.diff(pulses["burst"]) == 0])),
        "recorded_s": recorded_s,
        "n_pulses_unrecorded": most_unrecorded,
        "n_units": len(table), "n_tested": int(table["tested"].sum()),
        "n_tagged": int(table["tagged"].sum()),
        "n_artefact_suspect": int(table["artefact_suspect"].sum())}])

    if responses:
        responses = pd.concat(responses, ignore_index=True)
    else:
        responses = pd.DataFrame()
    if rasters:
        rasters = pd.concat(rasters, ignore_index=True)
    else:
        rasters = pd.DataFrame(columns=["session_id", "unit_id", "burst", "t_ms"])
    return Tags(units=table, responses=responses, rasters=rasters, pulses=pulses,
                sessions=summary)


# =====================================================================
# A cohort
# =====================================================================

def _say_nothing(message):
    """What tag_cohort reports to when nobody is listening."""


def tag_cohort(cohort, *, sort, curation, protocol=PROTOCOL, only=None, log_fn=None,
               **settings):
    """
    :func:`tag_session` on every session of ``protocol`` that has ``curation`` exported.

    ``cohort`` is a :class:`Cohort`, or a Path or string. ``only`` restricts it to those
    session IDs (a list); ``protocol=None`` tries every session with the curation (a
    behaviour session simply has no pulses). Sessions without the curation are skipped
    and logged — an optotagging session nobody curated has no good units to test.
    ``log_fn`` is called with each line (a string). ``settings`` are
    :func:`tag_session`'s thresholds.
    """
    log = log_fn or _say_nothing
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)

    # ---- 1. Each session in turn ---------------------------------------
    found, skipped = [], {}
    for session_id in cohort.session_ids:
        if only is not None and session_id not in only:
            continue
        record = cohort.session(session_id)
        if protocol is not None:
            # The session's protocol name, from its metadata; None if it cannot be read.
            try:
                metadata = metadata_source.read(record.path("metadata"))
                session_protocol = metadata.protocol_name
            except (FileNotFoundError, KeyError, ValueError):
                session_protocol = None
            if session_protocol != protocol:
                continue
        try:
            files = record.curation(sort, curation)
        except KeyError:
            skipped[session_id] = f"no curation {curation!r}"
            continue
        if files.get("units") is None:
            skipped[session_id] = "curation not exported"
            continue
        tags = tag_session(record, sort=sort, curation=curation, **settings)
        if tags is None:
            skipped[session_id] = "no laser pulses"
            continue
        row = tags.sessions.iloc[0]
        log(f"  {session_id}: {row.n_pulses} pulses in {row.n_bursts} bursts, "
            f"{row.n_tested}/{row.n_units} units tested, {row.n_tagged} tagged")
        found.append(tags)
    for session_id, why in skipped.items():
        log(f"  {session_id}: skipped — {why}")
    if not found:
        raise ValueError(f"no session tested. Skipped: {skipped}")

    # ---- 2. Every session's tables, one under another ------------------
    # A session's empty table is left out rather than joined.
    #: table name -> the sessions' tables of that name, concatenated
    joined = {}
    for name in ("units", "responses", "rasters", "pulses", "sessions"):
        tables = [getattr(t, name) for t in found if not getattr(t, name).empty]
        joined[name] = pd.concat(tables, ignore_index=True)

    used = {"sort": sort, "curation": curation, "protocol": protocol,
            "channel": settings.get("channel", LASER_CHANNEL),
            "burst_gap_s": settings.get("burst_gap_s", BURST_GAP_S),
            "window_s": settings.get("window_s", WINDOW_S),
            "min_responses": settings.get("min_responses", MIN_RESPONSES),
            "min_reliability": settings.get("min_reliability", MIN_RELIABILITY),
            "max_jitter_ms": settings.get("max_jitter_ms", MAX_JITTER_MS),
            "artefact_jitter_ms": settings.get("artefact_jitter_ms", ARTEFACT_JITTER_MS),
            "skipped": skipped}
    return Tags(units=joined["units"], responses=joined["responses"],
                rasters=joined["rasters"], pulses=joined["pulses"],
                sessions=joined["sessions"], settings=used)
