"""
The post-processing pipeline: raw rig output -> files on one clock.

**This file is the step.** Read it top to bottom and you have seen everything
post-processing does:

    channel roles            what each DAQ channel is for
    postprocess_cohort       every session that needs it, one after another
    postprocess_session      one session: read, clock, write, check
    sync_ephys               the ephys clock, the one slow and optional part
    SessionResult,
    CohortResult             what comes back

It performs no work itself — every stage is a call into its own module:

    sources/        one reader per raw format; each parses, nothing more
    utils/clock.py  place the two indexed streams on the DAQ clock
    outputs/        one module per output file

A session is handed over as a :class:`~hba2.cohort.SessionRecord` —
every path it needs, already resolved — and this file never looks at a directory itself.
The cohort is scanned once, by whoever builds the record, so a cohort-wide run does one
walk rather than one per session. It also makes the record the only argument that
varies, which is what would let ``postprocess_cohort`` hand sessions to a process pool.

Its whole job is synchronisation. What counts as a success, a response or a reaction time
is specific to a trial type and belongs to analysis code reading ``events.csv`` later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import outputs
from ..cohort import Cohort, SessionRecord
from .sources import blocks, camera, daq, metadata, openephys, scales, spikeglx, trials_csv
from .utils import clock

# What each DAQ channel is for. The DAQ records the same 24 channels with the same
# names every session, so a channel's role follows from its name — there is nothing to
# configure, and a spare channel wired to something new becomes a pulse source on its
# own. Which spare was actually in use is answerable from the recording, because an
# unused channel simply has no pulses.
CAMERA = "CAMERA"
SCALES = "SCALES"
TRIAL_MARKER = "DAQ_LINK1"
AUX_BEHAVIOUR = "DAQ_LINK0"          # aux link line, carries cues on some protocols
ARENA = ("SENSOR", "LED_", "VALVE")  # per-port hardware

#: Without these a session cannot be put on one clock: each is the only route for one
#: stream.
REQUIRED_CHANNELS = (CAMERA, SCALES, TRIAL_MARKER)


def channel_role(channel):
    """
    What a channel is for: ``behaviour``, ``camera``, ``scales``, ``trial_marker`` or
    ``pulses``.

    Anything unrecognised is a pulse source. That default is deliberate: a channel
    nobody has told this code about is far more likely to be a spare wired to new
    equipment than to be arena hardware, and treating it as a pulse train costs one
    small file while losing it would be silent.
    """
    if channel.startswith(ARENA):
        return "behaviour"
    if channel == CAMERA:
        return "camera"
    if channel == SCALES:
        return "scales"
    if channel == TRIAL_MARKER:
        return "trial_marker"
    if channel == AUX_BEHAVIOUR:
        return "behaviour"
    return "pulses"


# =====================================================================
# A whole cohort
# =====================================================================

def _say_nothing(message):
    """What postprocess_cohort reports to when nobody is listening."""


def postprocess_cohort(cohort, *, only=None, reprocess=False, ephys_sync=True,
                       log_fn=None, progress_fn=None, should_cancel=None):
    """
    Post-process every session in a cohort that needs it.

    Args:
        cohort: a :class:`Cohort`, or a path (Path or string) to build one from.
        only: a tuple of session IDs: process just these.
        reprocess: also redo sessions that already have output.
        ephys_sync: map the ephys clock where it has not been — see
            :func:`postprocess_session`. False leaves it to the ephys pipeline, which
            only does it for sessions whose units are going to be used.
        log_fn: called with each line as it happens. Without it the run is silent
            until it returns, which for a large cohort is a long time — a session
            takes roughly half a minute.
        progress_fn: called with a short status string per session, for a UI label.
        should_cancel: called with no arguments between sessions; stop when it returns
            True. Checked between rather than within, so a session is never left
            half-written.

    The cohort is walked **once**, here, and each session is handed the paths it needs.
    Sessions missing required raw files are skipped and listed rather than attempted.

    Deliberately serial. Sessions are independent and :class:`SessionRecord` pickles, so
    a process pool would drop straight in — but that is worth proving on a rig before
    relying on it, and serial keeps the failure modes simple.
    """
    log = log_fn or _say_nothing
    progress = progress_fn or _say_nothing

    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    result = CohortResult(base=cohort.base)

    # ---- 1. Which sessions ---------------------------------------------
    if reprocess:
        candidates = list(cohort)
    else:
        candidates = cohort.with_files(processed=False)
    if only:
        candidates = [r for r in candidates if r.session_id in only]

    total = len(candidates)
    log(f"{cohort.base.name}: {total} session(s) to process ({len(cohort)} in cohort)")

    # ---- 2. Each one in turn -------------------------------------------
    for number, record in enumerate(candidates, start=1):
        if should_cancel is not None and should_cancel():
            log(f"Cancelled after {number - 1} of {total}.")
            break

        if not record.can_process:
            result.skipped.append(record)
            continue

        progress(f"Session {number}/{total}: {record.session_id}")
        outcome = postprocess_session(record, ephys_sync=ephys_sync)
        result.results.append(outcome)

        if not outcome.ok:
            log(f"  [{number}/{total}] {record.session_id}: FAILED — {outcome.error}")
            continue
        line = f"  [{number}/{total}] {record.session_id}: {len(outcome.written)} file(s) written"
        if outcome.warnings:
            line += f", {len(outcome.warnings)} warning(s)"
        log(line)

    # ---- 3. The summary ------------------------------------------------
    if result.skipped:
        # Collapsed: on a large cohort this is otherwise a wall of near-identical lines.
        example = result.skipped[0]
        log(f"  {len(result.skipped)} session(s) skipped, missing raw files "
            f"(e.g. {example.session_id}: {', '.join(example.missing_raw)})")

    line = f"{cohort.base.name}: {len(result.ok)} processed, {len(result.failed)} failed"
    if result.skipped:
        line += f", {len(result.skipped)} skipped"
    log(line)
    return result


# =====================================================================
# One session
# =====================================================================

def postprocess_session(record, out_dir=None, *, ephys_sync=True):
    """
    Put one session's streams on a single clock and write them out.

    Args:
        record: every path this session has, already resolved. Nothing is scanned here.
        out_dir: where to write, as a Path or string. Defaults to ``<session>/processed``.
        ephys_sync: for a session with a probe, also map the ephys clock — but only when
            ``ephys_sync.csv`` is not already there. It is one pass over the whole
            recording, minutes rather than seconds, and the raw ephys never changes, so
            reprocessing the behaviour does not redo it. :func:`sync_ephys` does it on
            its own, and redoes it if asked.

    Failures are returned rather than raised, so a cohort run reports a bad session and
    carries on. Check :attr:`SessionResult.ok`.
    """
    if out_dir is None:
        out = record.out_dir
    else:
        out = Path(out_dir)
    result = SessionResult(session_id=record.session_id, out_dir=out)

    if not record.can_process:
        result.error = f"missing required file(s): {', '.join(record.missing_raw)}"
        return result

    try:
        # ---- 1. Read -------------------------------------------------
        # Each source parses one format. None of them aligns anything, and
        # none of them touches a timestamp.
        meta = metadata.read(record.path("metadata"))
        daq_data = daq.read(record.path("daq"))
        rows = trials_csv.read(record.path("trials_csv"))
        camera_data = camera.read(record.path("tracker"))
        scales_data = scales.read(record.path("scales_csv"))
        # Optional: absent on every session recorded before the rig wrote it.
        block_record = blocks.read(record.raw.get("blocks"))

        recorded = set(daq_data.channel_names)
        absent = [name for name in REQUIRED_CHANNELS if name not in recorded]
        if absent:
            result.error = f"recording has no {', '.join(absent)} channel(s)"
            return result

        # ---- 2. Which channel is which -------------------------------
        # Classified once, here, and the answer handed to the outputs. They then write
        # what they are given rather than each re-deciding what a channel is for.
        roles = {name: channel_role(name) for name in daq_data.channel_names}
        behaviour = [n for n, r in roles.items() if r in ("behaviour", "trial_marker")]
        pulse_sources = [n for n, r in roles.items() if r == "pulses"]

        # ---- 3. Clock ------------------------------------------------
        # Only two streams need placing, by the same mechanism. Behaviour
        # channels are already on the clock; they *are* the DAQ.
        starts, ends = clock.trial_windows(daq_data, TRIAL_MARKER)
        # Read once here and shared: the trials writer needs the same beam breaks the
        # events writer does, to say when each trial's chosen port was touched.
        touches = outputs.events.sensor_touches(daq_data, behaviour)

        camera_pulses = daq_data.rising_edges(CAMERA)
        frames = clock.place_by_pulse_index(
            "frames", camera_data.frame_ids, camera_pulses,
            skip=clock.recording_start(camera_pulses, camera_data.frame_ids))

        # Scales that logged no pulse index cannot be placed: every reading is
        # unmatched (NaN time) rather than guessed.
        if scales_data.message_ids is not None:
            readings = scales_data.message_ids
        else:
            readings = np.full(scales_data.n_readings, -1)
        weights = clock.place_by_pulse_index(
            "scales", readings, daq_data.rising_edges(SCALES))

        # ---- 4. Write ------------------------------------------------
        # One module per file. This block is the list of what a session produces.
        result.written = [
            outputs.session.write(out, meta, daq_data, camera_data, rows,
                                  block_record, roles, record.layout,
                                  record.raw.get("scales_stillness"),
                                  scales_data.has_stillness),
            outputs.events.write(out, daq_data, behaviour, starts, ends),
            outputs.trials.write(out, rows, starts, ends, *touches),
            # Not on the DAQ clock, and deliberately a file of its own — see notes.py.
            outputs.notes.write(out, rows, len(starts)),
            outputs.frame_times.write(out, frames, camera_data),
            outputs.scales.write(out, weights, scales_data),
        ]

        # A pulse source with nothing on it was not wired up this session, so it gets
        # no file — the files that exist say which channels were live.
        for channel in pulse_sources:
            written = outputs.pulses.write(out, daq_data, channel)
            if written is not None:
                result.written.append(written)

        # The ephys clock is optional, slow, and must never cost the behaviour: a
        # recording it cannot read is a warning on an otherwise good session.
        wants_ephys = ephys_sync and record.ephys is not None
        if wants_ephys and not (Path(out) / "ephys_sync.csv").exists():
            try:
                result.written.append(sync_ephys(record, out, daq_data=daq_data,
                                                 use_cached=False))
            except Exception as exc:              # noqa: BLE001 — reported, not raised
                result.warnings.append(f"ephys sync not written: "
                                       f"{type(exc).__name__}: {exc}")

        # ---- 5. Check ------------------------------------------------
        # Each writer's own warnings, then the checks no single source could make,
        # because they compare independent producers: a session where every file
        # looks fine on its own can still be wrong. (Trial-marker pulses against CSV
        # rows is one of these, but trials.py already says it, with how it paired them.)
        for item in result.written:
            result.warnings.extend(item.warnings)

        if meta.scales_enabled and not daq_data.has(SCALES):
            result.warnings.append(
                "metadata says scales were enabled but the channel is absent")

        if meta.has_laser:
            carrying = [c for c in pulse_sources if len(daq_data.rising_edges(c))]
            if not carrying:
                result.warnings.append(
                    "protocol enabled the laser but no pulse source carries pulses")

        if daq_data.reliability is not None and daq_data.reliability < 99.0:
            result.warnings.append(
                f"DAQ reliability {daq_data.reliability:.2f}% is below 99%")

    except Exception as exc:                      # noqa: BLE001 — reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def sync_ephys(record, out_dir=None, recording_index=0, use_cached=True, daq_data=None):
    """
    Map one session's ephys clock onto its DAQ's, whether or not that was done before:
    read the sync pulses, pair them with the DAQ's messages, write the mapping.

    :func:`postprocess_session` runs it for a new session, handing over the DAQ it has
    already read. Called alone, it is for the case where the behaviour is processed and
    only the ephys needs (re)doing, and it reads the DAQ itself. Raises rather than
    returning a result — called on its own, a failure is the answer.

    Open Ephys and SpikeGLX recordings both end as the same ``ephys_sync.csv``: probe
    seconds against DAQ seconds. Which one this is, the ephys folder says
    (:func:`~.sources.spikeglx.is_spikeglx`). For SpikeGLX, ``recording_index`` picks
    the probe.

    ``use_cached``: take the pulses from the previous pipeline's
    ``*_ephys_sync_timestamps.h5`` where one sits beside the raw files and checks out
    (:func:`~.sources.openephys.cached_pulses`) — seconds rather than minutes. Anything
    wrong with it falls back to reading the recording; ``ephys_sync.json`` says which.
    Open Ephys only.
    """
    if record.ephys is None:
        raise FileNotFoundError(f"{record.session_id} has no ephys recording")
    if out_dir is None:
        out = record.out_dir
    else:
        out = Path(out_dir)
    if daq_data is None:
        daq_data = daq.read(record.path("daq"))

    if spikeglx.is_spikeglx(record.ephys):
        recording = spikeglx.locate(record.ephys, recording_index)
        pulses = spikeglx.sync_pulses(recording)
        source = "recording"
        _check_one_pulse_per_loop(pulses.times, daq_data)
    else:
        recording = openephys.locate(record.ephys, recording_index)

        # The pulses: from the cached file if asked and it checks out, else the recording.
        pulses = None
        source = "recording"
        cached = record.path("metadata").parent / f"{record.session_id}{openephys.CACHED_SUFFIX}"
        if use_cached and cached.exists():
            try:
                pulses = openephys.cached_pulses(cached, recording)
                source = str(cached)
            except Exception as exc:              # noqa: BLE001 — fall back, and say so
                source = f"recording (cached file rejected: {exc})"
        if pulses is None:
            pulses = openephys.sync_pulses(recording)

    mapping = clock.map_ephys_to_daq(pulses.times, daq_data.message_ids,
                                     daq_data.timestamps)
    return outputs.ephys_sync.write(out, recording, pulses, mapping, source=source)


def _check_one_pulse_per_loop(pulse_times, daq_data):
    """
    Refuse a pulse train that is not one pulse per DAQ loop.

    Pairing pulse *k* with message *k* assumes the DAQ pulses on every loop, as the 1 kHz
    firmware does. A firmware that pulses on every *n*-th loop would pair each message
    with the wrong pulse and still look smooth, so the two rates are compared instead.
    """
    ids = np.asarray(daq_data.message_ids, dtype=np.int64)
    times = np.asarray(daq_data.timestamps, dtype=np.float64)
    if len(ids) < 2 or len(pulse_times) < 2 or times[-1] <= times[0]:
        return
    loops_per_s = (ids[-1] - ids[0]) / (times[-1] - times[0])
    pulses_per_s = 1.0 / float(np.median(np.diff(pulse_times)))
    if not 0.9 < loops_per_s / pulses_per_s < 1.1:
        raise ValueError(f"the DAQ ran {loops_per_s:.0f} loops/s but pulsed "
                         f"{pulses_per_s:.0f} times/s — this firmware does not pulse on "
                         "every loop, and pairing by count needs it to")


# =====================================================================
# What comes back
# =====================================================================

@dataclass
class SessionResult:
    """What one run produced, and anything odd it noticed."""

    session_id: str
    out_dir: Path | None = None
    written: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self):
        return self.error is None

    def describe(self):
        if not self.ok:
            return f"{self.session_id}: FAILED — {self.error}"
        lines = [f"{self.session_id}:"]
        for item in self.written:
            lines.append(f"  {item.describe()}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


@dataclass
class CohortResult:
    """What one sweep produced, per session."""

    base: Path
    results: list[SessionResult] = field(default_factory=list)
    skipped: list[SessionRecord] = field(default_factory=list)

    @property
    def ok(self):
        return [r for r in self.results if r.ok]

    @property
    def failed(self):
        return [r for r in self.results if not r.ok]

    def describe(self):
        heading = f"{self.base.name} — {len(self.ok)} processed, {len(self.failed)} failed"
        if self.skipped:
            heading += f", {len(self.skipped)} skipped"
        lines = [heading, ""]

        # Per output, because "did every data stream behave" is the real question and
        # it is answered across sessions rather than within one.
        #: output name -> what each session wrote for it
        by_output = {}
        for result in self.ok:
            for item in result.written:
                by_output.setdefault(item.name, []).append(item)
        for name, items in sorted(by_output.items()):
            clean = sum(1 for i in items if not i.n_unmatched and not i.warnings)
            line = f"  {name:<16} {clean}/{len(items)} clean"
            if clean != len(items):
                line += f" · {len(items) - clean} with warnings"
            lines.append(line)

        flagged = []
        for result in self.ok:
            for warning in result.warnings:
                flagged.append((result.session_id, warning))
        if flagged:
            lines += ["", "  WARNINGS"]
            for session_id, warning in flagged[:20]:
                lines.append(f"    {session_id}  {warning}")
            if len(flagged) > 20:
                lines.append(f"    ... and {len(flagged) - 20} more")

        if self.skipped:
            lines += ["", "  SKIPPED — missing raw files"]
            for record in self.skipped[:10]:
                lines.append(f"    {record.session_id}: {', '.join(record.missing_raw)}")
            if len(self.skipped) > 10:
                lines.append(f"    ... and {len(self.skipped) - 10} more")

        if self.failed:
            lines += ["", "  FAILED"]
            for result in self.failed:
                lines.append(f"    {result.session_id}  {result.error}")

        return "\n".join(lines)
