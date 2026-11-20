"""
Reader for a SpikeGLX recording — only as far as the sync pulses.

Neuropixels probes on the PXIe chassis are recorded by SpikeGLX, with an NI card
(PXIe-6341, BNC-2110 breakout) recording beside them. The rig's DAQ sends one pulse per
read of its loop into an NI input, so pulse *k* on that channel is DAQ loop *k*, which is
``message_id`` *k* — the same link as Open Ephys's ``ADC1``, but on the NI card's clock
rather than the probe's.

So there are two links to make, joined on the NI clock::

    DAQ message_id --(rig pulses on the NI card, counted)--> NI time
    NI time --(SpikeGLX's 1 Hz sync: on the NI card, and bit 6 of the probe's SY word)--> probe time

The 1 Hz square wave comes from the imec card's SMA. Every probe records it internally;
the NI card records it through a cable. Each of its edges therefore has a time on both
clocks, and the straight line through them carries any NI time onto the probe. The
pulses come back on the probe's clock, so ``ephys_sync.csv`` means the same thing as for
Open Ephys: probe seconds against DAQ seconds, and a spike's sample divided by the
probe's rate is its probe time.

Layout, below the session's ``ephys/<session id>/`` folder, as hexcontrol writes it::

    <session id>_g0/
    ├── <session id>_g0_t0.nidq.bin / .meta        NI card: AI0-7 then the digital word
    └── <session id>_g0_t0.imec0.ap.bin / .meta    probe: 384 channels then SY
    spikeglx_session.json                          which probe and NI channels are whose

The JSON names the rig's NI channel (``rig_sync_line``, e.g. ``AI4``) and the 1 Hz
channel (``sync.ni_channel_name``, e.g. ``AI0``); nothing here assumes either. A
recording made with SpikeGLX's sync off has no 1 Hz pulse to align on and is refused.

The probe's AP file is the expensive read: the SY channel is one column of a ~80 GB an
hour file, so finding its edges is one pass over it, in chunks. The NI file is small.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: What hexcontrol writes beside the recording.
SESSION_JSON = "spikeglx_session.json"

#: The probe's SY word carries the 1 Hz sync on this bit.
SY_SYNC_BIT = 6

#: Samples per read when scanning a file for edges.
CHUNK = 2_000_000

#: A channel whose whole range is narrower than this (raw int16 counts) carries no pulses.
MIN_SWING = 1000

#: The two clocks' 1 Hz edges are paired by nearest neighbour; a pair further apart than
#: this is not the same edge. Both files start saving at the same moment, within ms.
PAIR_WITHIN_S = 0.25

#: Rig pulses this close to the start of the NI file may be a train that began before
#: saving did, in which case counting from the first one is off by however many were lost.
EARLY_PULSE_S = 0.05


def is_spikeglx(ephys_dir):
    """Whether a session's ephys folder holds a SpikeGLX recording."""
    return ephys_dir is not None and (Path(ephys_dir) / SESSION_JSON).exists()


def locate(ephys_dir, index=0):
    """
    The ``index``-th probe of the SpikeGLX recording in a session's ephys folder, with the
    NI file beside it, as a :class:`Recording`.

    This is also where a spike sorter finds the probe's data: ``Recording.continuous`` is
    the ``.ap.bin``, and SpikeInterface's ``read_spikeglx`` takes its folder.
    """
    ephys_dir = Path(ephys_dir)
    info = json.loads((ephys_dir / SESSION_JSON).read_text())
    probes = info.get("probes") or []
    if index >= len(probes):
        raise IndexError(f"{ephys_dir} has {len(probes)} probe(s); asked for #{index}")
    probe = probes[index]
    ip = int(probe["imec"])

    # 1. The files: as the JSON names them, else found by their SpikeGLX names.
    if probe.get("files"):
        ap_bin = ephys_dir / probe["files"][0]
    else:
        ap_bin = _one(ephys_dir, f"*.imec{ip}.ap.bin")
    if info.get("ni_file"):
        ni_bin = ephys_dir / info["ni_file"]
    else:
        ni_bin = _one(ephys_dir, "*.nidq.bin")
    ap_meta, ni_meta = read_meta(ap_bin), read_meta(ni_bin)

    # 2. Shapes and rates, from the .meta files.
    ap_channels = int(ap_meta["nSavedChans"])
    ni_channels = int(ni_meta["nSavedChans"])
    sync = info.get("sync") or {}

    return Recording(
        directory=ephys_dir, continuous=ap_bin, stream=f"imec{ip}.ap",
        sample_rate=float(ap_meta["imSampRate"]),
        n_samples=int(ap_bin.stat().st_size // (2 * ap_channels)),
        index=index, n_recordings=len(probes),
        ap_channels=ap_channels, ap_meta=ap_meta,
        ni_bin=ni_bin, ni_meta=ni_meta, ni_channels=ni_channels,
        ni_sample_rate=float(ni_meta["niSampRate"]),
        ni_n_samples=int(ni_bin.stat().st_size // (2 * ni_channels)),
        rig_channel=str(info["rig_sync_line"]),
        sync_channel=sync.get("ni_channel_name"),
        session=info)


def sync_pulses(recording):
    """
    The rig's sync pulses, timed on the probe's clock; a :class:`SyncPulses`.

    1. Rising edges on the rig's NI channel, in NI seconds.
    2. The 1 Hz edges on the NI card and on the probe, paired.
    3. The rig's pulses carried through those pairs onto the probe.

    Raises when the recording cannot be aligned: no 1 Hz sync, no pulses on a channel, or
    too few edges in common.
    """
    if not recording.sync_channel:
        raise ValueError(f"{recording.directory}: SpikeGLX's sync was off for this "
                         "recording, so the probe cannot be aligned to the NI card")

    # ---- 1. The rig's pulses, on the NI clock ----------------------------
    rig_samples, threshold = _ni_edges(recording, recording.rig_channel)
    t_rig_ni = rig_samples / recording.ni_sample_rate
    warnings = []
    if len(t_rig_ni) and t_rig_ni[0] < EARLY_PULSE_S:
        warnings.append(f"the first {recording.rig_channel} pulse is {t_rig_ni[0] * 1000:.0f} ms "
                        "into the NI file — the train may have begun before saving, which "
                        "would offset every count")

    # ---- 2. The 1 Hz edges on both clocks, paired ------------------------
    ni_sync, _ = _ni_edges(recording, recording.sync_channel)
    probe_sync = _sy_edges(recording)
    t_ni = ni_sync / recording.ni_sample_rate
    t_probe = probe_sync / recording.sample_rate
    if len(t_ni) < 2 or len(t_probe) < 2:
        raise ValueError(f"{recording.directory}: too few 1 Hz edges to align "
                         f"({len(t_ni)} on {recording.sync_channel}, {len(t_probe)} on the "
                         "probe) — is the imec card's SMA cabled to the NI card?")
    nearest = np.clip(np.searchsorted(t_probe, t_ni), 1, len(t_probe) - 1)
    left_closer = (t_ni - t_probe[nearest - 1]) < (t_probe[nearest] - t_ni)
    nearest = nearest - left_closer
    paired = np.abs(t_probe[nearest] - t_ni) < PAIR_WITHIN_S
    pair_ni, pair_probe = t_ni[paired], t_probe[nearest[paired]]
    if len(pair_ni) < 2:
        raise ValueError(f"{recording.directory}: no 1 Hz edges pair up between the NI "
                         "card and the probe")

    # ---- 3. The rig's pulses on the probe's clock ------------------------
    times = _through(t_rig_ni, pair_ni, pair_probe)
    fit = np.polyfit(pair_ni, pair_probe, 1)
    residual_us = np.abs(np.polyval(fit, pair_ni) - pair_probe) * 1e6
    alignment = {
        "ni_file": str(recording.ni_bin),
        "ni_sample_rate": recording.ni_sample_rate,
        "sync_channel": recording.sync_channel,
        "sync_edges_ni": int(len(t_ni)),
        "sync_edges_probe": int(len(t_probe)),
        "sync_edges_paired": int(len(pair_ni)),
        "probe_minus_ni_s": float(np.median(pair_probe - pair_ni)),
        "probe_per_ni_second": float(fit[0]),
        "line_residual_us_max": round(float(residual_us.max()), 3),
    }
    return SyncPulses(channel=recording.rig_channel, times=times,
                      samples=np.round(times * recording.sample_rate).astype(np.int64),
                      threshold=threshold, alignment=alignment, warnings=tuple(warnings))


def read_meta(bin_path):
    """A SpikeGLX ``.meta`` file, beside its ``.bin``, as a dict of strings."""
    text = Path(bin_path).with_suffix(".meta").read_text()
    pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line)
    return {key.lstrip("~"): value for key, value in pairs}


def channel_column(meta, acquired):
    """The column in the ``.bin`` holding acquired channel ``acquired``, or None if not saved."""
    subset = meta.get("snsSaveChanSubset", "all")
    if subset == "all":
        return acquired
    saved = []
    for part in subset.split(","):
        if ":" in part:
            first, last = part.split(":")
            saved.extend(range(int(first), int(last) + 1))
        elif part:
            saved.append(int(part))
    return saved.index(acquired) if acquired in saved else None


def _one(folder, pattern):
    found = sorted(folder.rglob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {folder}, found {len(found)}")
    return found[0]


def _ni_edges(recording, name):
    """Rising-edge samples of one NI channel, ``AI<n>`` or ``P0.<n>``, and the threshold."""
    meta = recording.ni_meta
    mn, ma, xa, dw = (int(x) for x in meta["snsMnMaXaDw"].split(","))
    name = name.strip().upper()
    if name.startswith("AI"):
        acquired, bit = mn + ma + int(name[2:]), None
    elif name.startswith("P0."):
        line = int(name[3:])
        acquired, bit = mn + ma + xa + line // 16, line % 16
    else:
        raise ValueError(f"unknown NI channel {name!r}: use AI<n> or P0.<n>")
    column = channel_column(meta, acquired)
    if column is None:
        raise ValueError(f"{name} is not saved in {recording.ni_bin}")
    data = np.memmap(recording.ni_bin, dtype=np.int16, mode="r",
                     shape=(recording.ni_n_samples, recording.ni_channels))

    if bit is not None:
        return _edges(data, column, lambda chunk: (chunk.view(np.uint16) >> bit) & 1), bit

    # Analog: the threshold is the midpoint of the range the channel reaches.
    low, high = np.iinfo(np.int16).max, np.iinfo(np.int16).min
    for first in range(0, recording.ni_n_samples, CHUNK):
        chunk = np.asarray(data[first:first + CHUNK, column])
        low, high = min(low, int(chunk.min())), max(high, int(chunk.max()))
    if high - low < MIN_SWING:
        raise ValueError(f"{name} shows no pulses ({low}..{high}) in {recording.ni_bin}")
    threshold = (low + high) // 2
    return _edges(data, column, lambda chunk: chunk > threshold), threshold


def _sy_edges(recording):
    """Rising-edge samples of the 1 Hz sync on the probe's SY word."""
    meta = recording.ap_meta
    ap, lf, sy = (int(x) for x in meta["snsApLfSy"].split(","))
    if sy < 1:
        raise ValueError(f"{recording.continuous} records no SY channel")
    column = channel_column(meta, ap + lf)
    if column is None:
        raise ValueError(f"the SY channel is not saved in {recording.continuous}")
    data = np.memmap(recording.continuous, dtype=np.int16, mode="r",
                     shape=(recording.n_samples, recording.ap_channels))
    return _edges(data, column, lambda chunk: (chunk.view(np.uint16) >> SY_SYNC_BIT) & 1)


def _edges(data, column, high):
    """Samples where ``high(chunk)`` goes from false to true, in one chunked pass."""
    edges, previous = [], np.int8(0)
    for first in range(0, data.shape[0], CHUNK):
        state = np.asarray(high(np.asarray(data[first:first + CHUNK, column]))).astype(np.int8)
        if not len(state):
            break
        before = np.concatenate(([previous], state[:-1]))
        edges.append(np.flatnonzero((before == 0) & (state == 1)) + first)
        previous = state[-1]
    if not edges:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(edges).astype(np.int64)


def _through(t, x, y):
    """``t`` carried along the straight lines between points ``(x, y)``, the ends extended."""
    out = np.interp(t, x, y)
    before, after = t < x[0], t > x[-1]
    out[before] = y[0] + (t[before] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    out[after] = y[-1] + (t[after] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return out


@dataclass(frozen=True)
class Recording:
    """One probe of a SpikeGLX recording, and the NI file recorded beside it."""

    directory: Path
    continuous: Path         #: the probe's ``.ap.bin`` — what a sorter reads
    stream: str              #: SpikeGLX's name for it, e.g. ``imec0.ap``
    sample_rate: float       #: the probe's calibrated rate (Hz)
    n_samples: int
    index: int               #: which probe of the recording, counting from 0
    n_recordings: int        #: how many probes it has
    ap_channels: int
    ap_meta: dict
    ni_bin: Path
    ni_meta: dict
    ni_channels: int
    ni_sample_rate: float
    ni_n_samples: int
    rig_channel: str         #: the NI channel the rig's DAQ pulses on, e.g. ``AI4``
    sync_channel: str | None  #: the NI channel carrying the 1 Hz sync; None if it was off
    session: dict            #: ``spikeglx_session.json`` as read
    system: str = "spikeglx"


@dataclass(frozen=True)
class SyncPulses:
    """Every rising edge on the rig's NI channel, on the probe's clock."""

    channel: str
    times: np.ndarray        #: probe seconds, from the start of the probe's file
    samples: np.ndarray      #: the probe sample each edge falls on
    threshold: int           #: raw NI units separating low from high (the bit, if digital)
    alignment: dict = field(default_factory=dict)   #: how NI time was carried onto the probe
    warnings: tuple = ()
