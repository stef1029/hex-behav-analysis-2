"""
The 2024 recordings: an Open Ephys board as the clock.
=====================================================

The 2024 cohorts predate
the DAQ writing its own timestamps. Each rig's Arduino DAQ streamed its channel states to
``<session>-ArduinoDAQ.json`` — one bit string per message, no times — and pulsed an
Open Ephys acquisition board once per message. The camera pulsed the same board once per
frame. One board recorded both rigs of a group, in legacy ``.continuous`` format, in a
folder beside the sessions (``<time>_OEAB_recording/`` or ``YYYY-MM-DD_HH-MM-SS/``).

So the clock here is the Open Ephys board's, and this module builds from it the same
:class:`~..postprocessing.sources.daq.DaqData` a current DAQ file reads as: message *k*
happened at the *k*-th DAQ pulse, joined by counting, exactly as the current rig's ephys
sync works. The camera's pulses come out beside it, for frame *n* at the *n*-th.

Which ADC carried what, per rig (from the previous pipeline, confirmed by pulse counts
against message and frame counts):

    rig 1   ADC1 DAQ message pulse (rising)    ADC2 camera (falling)
    rig 2   ADC5 DAQ message pulse (rising)    ADC4 camera (falling)

The bit strings name no channels. Bit *i* counted from the right is :data:`CHANNELS`
[*i*] — checked against the previous pipeline's decoded output, every channel matching
exactly. Names are 1-based and translated to 0-based as for any old DAQ file.

Some message IDs are corrupt — a garbled serial read gives IDs in the hundreds of
millions. Only IDs that continue the running sequence are kept; the rest are counted.

    find_recordings   the Open Ephys folders beside a group's sessions
    build             one session's DAQ on the board's clock
    pulse_times       edges on one ``.continuous`` channel
    OEABClock         what comes back
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..postprocessing.sources import daq as daq_source
from ..postprocessing.sources.daq import DaqData

#: Bit i, counted from the right of each message string, is CHANNELS[i].
CHANNELS = (
    "SPOT2", "SPOT3", "SPOT4", "SPOT5", "SPOT6", "SPOT1",
    "SENSOR6", "SENSOR1", "SENSOR5", "SENSOR2", "SENSOR4", "SENSOR3",
    "BUZZER4", "LED_3", "LED_4", "BUZZER3", "BUZZER5", "LED_2", "LED_5", "BUZZER2",
    "BUZZER6", "LED_1", "LED_6", "BUZZER1",
    "VALVE4", "VALVE3", "VALVE5", "VALVE2", "VALVE6", "VALVE1", "GO_CUE", "NOGO_CUE",
    "CAMERA",
)

#: Rig -> (DAQ pulse ADC, camera ADC).
RIG_ADCS = {1: ("ADC1", "ADC2"), 2: ("ADC5", "ADC4")}

#: A message ID further than this past the last good one is a corrupt read.
MAX_ID_JUMP = 10_000

#: A recording that stopped before the DAQ did still times every message it saw. When no
#: recording covers the whole session, one covering at least this share of its messages
#: is used and the messages after it stopped are dropped — the end of the session is lost,
#: everything before it is timed exactly as usual. Seen on 2403 and 2407 sessions whose
#: recording ended a few seconds to half a minute early.
MIN_COVERAGE = 0.95

#: One record of a legacy ``.continuous`` file, after its 1024-byte text header.
_RECORD = np.dtype([("timestamp", "<i8"), ("n", "<u2"), ("recording", "<u2"),
                    ("samples", ">i2", 1024), ("marker", "u1", 10)])


def find_recordings(group):
    """
    Folders holding ``*_ADC1.continuous`` within a group (a Path), at any depth up to
    two. Session folders are not looked in, unless their name says ``OEAB``.
    """
    found = []
    for candidate in sorted(group.iterdir()):
        if not candidate.is_dir():
            continue
        # A session folder, not a recording.
        if re.match(r"^\d{6}_\d{6}_", candidate.name) and "OEAB" not in candidate.name:
            continue
        for node in [candidate, *sorted(p for p in candidate.iterdir() if p.is_dir())]:
            if any(p.name.endswith("_ADC1.continuous") for p in node.iterdir()):
                found.append(node)
    return found


def build(daq_json, recordings, rig):
    """
    The session's DAQ on the Open Ephys clock, from whichever recording fits it.

    Args:
        daq_json: the session's ``-ArduinoDAQ.json`` (a Path).
        recordings: the group's recording folders, from :func:`find_recordings`.
        rig: the old rig number, which says which ADCs to read (:data:`RIG_ADCS`).

    Returns an :class:`OEABClock`.
    """
    if rig not in RIG_ADCS:
        raise ValueError(f"no ADC wiring known for rig {rig!r} (only {sorted(RIG_ADCS)})")
    daq_adc, camera_adc = RIG_ADCS[rig]

    # ---- 1. The messages ----------------------------------------------------
    document = json.loads(daq_json.read_text())
    messages = document.get("messages") or []
    if not messages:
        raise ValueError(f"{daq_json.name} has no messages")
    # A message whose string is the wrong length was misread; dropping it is safer than
    # decoding it, since an all-zero sensor bit reads as a touch.
    lengths = np.array([len(m[1]) if isinstance(m[1], str) else 0 for m in messages])
    width = int(np.bincount(lengths).argmax())
    raw_ids = np.array([m[0] if isinstance(m[0], int) else -1 for m in messages],
                       dtype=np.int64)
    raw_ids[lengths != width] = -1

    # ---- 2. Only the IDs that continue the running sequence -----------------
    keep = np.zeros(len(raw_ids), dtype=bool)
    last = -1
    for i, value in enumerate(raw_ids):
        if last < value <= last + MAX_ID_JUMP:
            keep[i] = True
            last = value
    kept = np.flatnonzero(keep)
    ids = raw_ids[kept]
    n_corrupt = int((~keep).sum())
    strings = [messages[i][1] for i in kept]

    # ---- 3. The recording whose DAQ pulse train covers them most tightly ----
    best = partial = None            # (surplus, recording, pulses, camera file)
    for recording in recordings:
        files = {p.name.rsplit("_", 1)[-1].removesuffix(".continuous"): p
                 for p in recording.glob("*_ADC*.continuous")}
        if daq_adc not in files or camera_adc not in files:
            continue
        pulses = pulse_times(files[daq_adc])
        surplus = len(pulses) - (int(ids.max()) + 1)
        if surplus >= 0 and (best is None or surplus < best[0]):
            best = (surplus, recording, pulses, files[camera_adc])
        elif surplus < 0 and (partial is None or len(pulses) > len(partial[2])):
            partial = (surplus, recording, pulses, files[camera_adc])

    # None covers it all: one that stopped a little early, the messages after it dropped.
    n_truncated = 0
    if best is None and partial is not None \
            and len(partial[2]) >= MIN_COVERAGE * (int(ids.max()) + 1):
        best = partial
        covered = ids < len(partial[2])
        n_truncated = int((~covered).sum())
        ids = ids[covered]
        strings = [s for s, is_covered in zip(strings, covered) if is_covered]
    if best is None:
        raise ValueError(f"no Open Ephys recording in the group has enough {daq_adc} "
                         f"pulses for {len(ids):,} messages")
    _, recording, pulses, camera_file = best

    # ---- 4. The bit strings as named channels -------------------------------
    # Equal-length '0'/'1' strings -> (n, width) uint8.
    width = len(strings[0])
    raw = np.frombuffer("".join(strings).encode("ascii"), dtype=np.uint8)
    bits = (raw.reshape(len(strings), width) == ord("1")).astype(np.uint8)
    channels = {CHANNELS[i]: bits[:, width - 1 - i] for i in range(min(width, len(CHANNELS)))}
    renames = daq_source._legacy_renames(set(channels))
    channels = {renames.get(name, name): data for name, data in channels.items()}

    # ---- 5. The DAQ, message k at pulse k -----------------------------------
    daq = DaqData(
        path=daq_json,
        message_ids=ids.astype(np.uint32),
        timestamps=pulses[ids],
        channels=channels,
        attributes={"reliability": document.get("reliability"),
                    "messages_per_second": document.get("messages per second"),
                    "time": document.get("time"),
                    "clock": f"open ephys {recording.parent.name}/{daq_adc}"},
        n_error_messages=len(document.get("error messages") or []),
        renamed_channels=renames,
    )
    return OEABClock(daq=daq, camera_pulses=pulse_times(camera_file, falling=True),
                     recording=recording, n_pulses=len(pulses), n_corrupt_ids=n_corrupt,
                     n_truncated=n_truncated)


def pulse_times(path, falling=False):
    """
    Edge times in seconds from the recording's first sample, on one legacy
    ``.continuous`` channel (a Path). Rising edges, or falling ones if ``falling``.
    """
    # The sample rate is in the 1024-byte text header; the samples follow it.
    header = path.read_bytes()[:1024].decode("latin-1")
    match = re.search(r"sampleRate\s*=\s*([\d.]+)", header)
    rate = float(match.group(1)) if match else 30000.0
    records = np.memmap(path, dtype=_RECORD, mode="r", offset=1024)
    samples = np.asarray(records["samples"]).ravel()

    low, high = np.percentile(samples, [1, 99])
    high_state = samples > (low + high) / 2
    if falling:
        edges = np.flatnonzero(high_state[:-1] & ~high_state[1:]) + 1
    else:
        edges = np.flatnonzero(~high_state[:-1] & high_state[1:]) + 1
    return edges / rate


# =====================================================================
# What comes out
# =====================================================================

@dataclass
class OEABClock:
    daq: DaqData
    camera_pulses: np.ndarray
    recording: Path
    n_pulses: int
    n_corrupt_ids: int
    #: DAQ messages after the Open Ephys recording stopped, dropped — see MIN_COVERAGE.
    n_truncated: int = 0
