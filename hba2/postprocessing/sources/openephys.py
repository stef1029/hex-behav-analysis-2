"""
Reader for an Open Ephys recording — only as far as the sync pulses.

The v2 rig's DAQ sends one pulse per read of its loop into the first analog input
(``ADC1``) of the Open Ephys acquisition board, so pulse *k* on that channel is DAQ loop
*k*, which is what the DAQ file calls ``message_id`` *k*. Those pulses are the whole
link between the two clocks; nothing else in the recording is read here. Spikes are the
sorter's business, and the neural channels never pass through this package.

Layout, below the session's ``ephys/<Rig_mouse>/`` folder::

    Record Node <n>/experiment<e>/recording<r>/
    ├── structure.oebin                   channel names, count and sample rate
    └── continuous/<...>acquisition_board/
        ├── continuous.dat                int16, samples x channels, interleaved
        └── timestamps.npy                the ephys clock, one per sample (s)

The neural stream carries 128 probe channels then ``ADC1..ADC8``, interleaved, so one
analog channel cannot be read without passing over the whole file: a two-hour session is
about 60 GB, read once, in chunks.

A session that was stopped and restarted has several ``recording<r>`` folders. The first
is used, as the sorter does, and the count is reported so a second one is never silent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Where the rig's sync line goes.
SYNC_CHANNEL = "ADC1"

#: Samples per read when scanning for pulses: about 1.4 GB of file per chunk at 136
#: channels, but only one column of it is kept.
CHUNK = 5_000_000

#: The threshold is set from this much of the recording, taken this far in, where the
#: pulse train is certainly running.
THRESHOLD_WINDOW_S = 5.0
THRESHOLD_AT = 0.4

#: What the previous pipeline named its copy of these same pulses.
CACHED_SUFFIX = "_ephys_sync_timestamps.h5"


def _numeric_order(path):
    """Sort key putting recording10 after recording9, not after recording1."""
    return tuple(int(n) if n.isdigit() else n for n in re.split(r"(\d+)", str(path)))


def locate(ephys_dir, index=0):
    """
    The neural stream of the ``index``-th recording under a session's ephys folder (a
    Path or a string), as a :class:`Recording`.

    Raises when there is none, rather than returning something half-found.
    """
    ephys_dir = Path(ephys_dir)
    found = sorted(ephys_dir.glob(
        "Record Node*/experiment*/recording*/continuous/*acquisition_board/continuous.dat"),
        key=_numeric_order)
    if not found:
        raise FileNotFoundError(f"no acquisition-board continuous.dat under {ephys_dir}")
    if index >= len(found):
        raise IndexError(f"{ephys_dir} has {len(found)} recording(s); asked for #{index}")

    continuous = found[index]
    oebin = continuous.parents[2] / "structure.oebin"

    # The oebin's entry for the stream stored in this folder.
    folder = continuous.parent.name
    stream = None
    for candidate in json.loads(oebin.read_text()).get("continuous", []):
        if str(candidate.get("folder_name", "")).rstrip("/") == folder:
            stream = candidate
            break
    if stream is None:
        raise ValueError(f"structure.oebin has no stream for {folder}")

    names = tuple(channel["channel_name"] for channel in stream["channels"])
    n_samples = continuous.stat().st_size // (2 * len(names))

    return Recording(directory=ephys_dir, continuous=continuous,
                     timestamps=continuous.parent / "timestamps.npy", oebin=oebin,
                     stream=continuous.parent.name, channel_names=names,
                     sample_rate=float(stream["sample_rate"]), n_samples=int(n_samples),
                     index=index, n_recordings=len(found))


def sync_pulses(recording, channel=SYNC_CHANNEL):
    """
    Rising edges of the sync square wave, in one streaming pass over the recording; a
    :class:`SyncPulses`.

    The line swings between two flat levels, so the threshold is the midpoint of what a
    few seconds of it reach. Raises if that window shows no swing at all, which means
    the channel is wrong or the cable was out.
    """
    if channel not in recording.channel_names:
        # Leaving out the probe's own CH... channels, which are not what was asked for.
        others = [n for n in recording.channel_names if not n.startswith("CH")]
        raise ValueError(f"no channel {channel!r} in {recording.oebin}; it has "
                         f"{', '.join(others)}")
    column = recording.channel_names.index(channel)
    data = np.memmap(recording.continuous, dtype=np.int16, mode="r",
                     shape=(recording.n_samples, recording.n_channels))

    start = int(THRESHOLD_AT * recording.n_samples)
    window = np.asarray(data[start:start + int(THRESHOLD_WINDOW_S * recording.sample_rate),
                             column])
    low, high = int(window.min()), int(window.max())
    if high - low < 100:
        raise ValueError(f"{channel} shows no pulses ({low}..{high}) in "
                         f"{recording.continuous}")
    threshold = (low + high) // 2

    edges, previous = [], np.int8(0)
    for first in range(0, recording.n_samples, CHUNK):
        state = (np.asarray(data[first:first + CHUNK, column]) > threshold).astype(np.int8)
        before = np.concatenate(([previous], state[:-1]))
        edges.append(np.flatnonzero((before == 0) & (state == 1)) + first)
        previous = state[-1]
    samples = np.concatenate(edges) if edges else np.empty(0, dtype=np.int64)

    clock = np.load(recording.timestamps, mmap_mode="r")
    return SyncPulses(channel=channel, times=np.asarray(clock[samples], dtype=np.float64),
                      samples=samples, threshold=threshold)


def _text(value):
    """An HDF5 attribute as a string, whether it was stored as bytes or not."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def cached_pulses(path, recording, channel=SYNC_CHANNEL):
    """
    The same pulses, from the previous pipeline's ``*_ephys_sync_timestamps.h5`` (a Path
    or a string), as a :class:`SyncPulses`.

    That pipeline found them the same way — rising edges on ``ADC1``, timed by
    ``timestamps.npy`` — and on the session checked against this module
    every one of its 7,115,294 pulses matched to the nanosecond. Reading it takes
    seconds where :func:`sync_pulses` takes minutes.

    It is only used if it is plainly the same thing, and raises otherwise so the caller
    can fall back to the recording: same channel, the same recording folder, and pulses
    that increase, sit inside the recording, and come once a millisecond.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        meta = dict(handle["metadata"].attrs) if "metadata" in handle else {}
        times = np.asarray(handle["pins/pin_0/high_timestamps"][:], dtype=np.float64)

    if _text(meta.get("source_channel", "")) != channel:
        raise ValueError(f"{path} was read from {meta.get('source_channel')!r}, not {channel}")
    # The recording it names, from 'Record Node' on — the rest of the path is wherever
    # the cohort sat when it was made.
    made_from = _text(meta.get("continuous_path", ""))
    tail = str(recording.continuous)[str(recording.continuous).find("Record Node"):]
    if not made_from.endswith(tail):
        raise ValueError(f"{path} was made from {made_from}, not this recording ({tail})")

    clock = np.load(recording.timestamps, mmap_mode="r")
    intervals = np.diff(times)
    if len(times) < 2 or (intervals <= 0).any():
        raise ValueError(f"{path}: pulse times do not increase")
    if times[0] < clock[0] or times[-1] > clock[-1]:
        raise ValueError(f"{path}: pulses fall outside the recording")
    if not 0.00099 < float(np.median(intervals)) < 0.00101:
        raise ValueError(f"{path}: pulses are not once a millisecond "
                         f"(median {np.median(intervals) * 1000:.4f} ms)")

    return SyncPulses(channel=channel, times=times,
                      samples=np.searchsorted(clock, times),
                      threshold=int(meta.get("threshold_raw", -1)))


@dataclass(frozen=True)
class Recording:
    """Where one Open Ephys recording's neural stream is, and its shape."""

    directory: Path
    continuous: Path
    timestamps: Path
    oebin: Path
    stream: str
    channel_names: tuple   #: every channel's name, in file order
    sample_rate: float
    n_samples: int
    #: Which ``recording<r>`` this is, counting from 0, and how many the folder holds.
    index: int
    n_recordings: int

    @property
    def n_channels(self):
        return len(self.channel_names)


@dataclass(frozen=True)
class SyncPulses:
    """Every rising edge on the sync channel, on the ephys clock."""

    channel: str
    times: np.ndarray        #: ephys seconds, from ``timestamps.npy``
    samples: np.ndarray      #: the sample each edge fell on
    threshold: int           #: raw ADC units separating low from high
