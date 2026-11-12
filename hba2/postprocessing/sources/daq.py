"""
Reader for ``<session>-ArduinoDAQ.h5``.

This file is the **authority for when anything happened**. Every other stream is
aligned to it, and its ``timestamps`` array is the master clock.

Layout (see the DAQLink data-format docs):

    message_ids          uint32   one per message, sequential
    timestamps           float64  seconds since acquisition start, monotonic
    channel_data/<name>  uint8    0/1 state per message, one array per channel
    error_messages       optional, present when reads were malformed
    (attributes)         mouse_ID, date_time, No_of_messages, reliability,
                         time_taken, messages_per_second

Channel names are 0-based and name the port they serve: ``SENSOR0`` is the sensor at
port 0, so the number in a channel name *is* the port.

Sessions recorded before the rename name them 1-based (``SENSOR1``..``SENSOR6``). That
is **detected from the names themselves and translated on read**, so everything
downstream sees one convention and ``events.csv`` means the same thing whichever vintage
a session is. Detecting from the names rather than from the folder layout matters: the
two changed at the same time, but a session recorded between them would fall through a
crack in any rule that assumed they were coupled.

The whole file is ~15 MB, so everything is read eagerly: about 70 MB in memory
for a two-hour session. That buys straightforward code and costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

#: Per-port channel families, which are the only names the rename touched.
_FAMILIES = ("SENSOR", "LED_", "VALVE")
N_PORTS = 6

#: Old name -> current name, e.g. SENSOR1 -> SENSOR0.
_LEGACY_NAMES = {}
for _family in _FAMILIES:
    for _port in range(N_PORTS):
        _LEGACY_NAMES[f"{_family}{_port + 1}"] = f"{_family}{_port}"


def _legacy_renames(names):
    """
    Old channel names present in a recording (a set of names), mapped to their current
    names.

    A ``...6`` channel is the tell: there is no port 6, so under current naming it
    cannot exist. Empty result means the recording already uses current names.

    Also used by the legacy converters, which build a :class:`DaqData` from older
    file formats.
    """
    if not any(f"{family}{N_PORTS}" in names for family in _FAMILIES):
        return {}
    return {old: new for old, new in _LEGACY_NAMES.items() if old in names}


def read(path):
    """
    Load a DAQ recording (a Path or a string) as a :class:`DaqData`.

    Raises:
        ValueError: if the file is structurally unusable — no timestamps, or
            arrays whose lengths disagree. Both make the clock meaningless.
    """
    path = Path(path)

    # ---- 1. Read the file -------------------------------------------
    with h5py.File(path, "r") as handle:
        if "timestamps" not in handle:
            raise ValueError(f"{path.name} has no 'timestamps' dataset")

        timestamps = np.asarray(handle["timestamps"][:], dtype=np.float64)
        if "message_ids" in handle:
            message_ids = np.asarray(handle["message_ids"][:], dtype=np.uint32)
        else:
            message_ids = np.arange(len(timestamps), dtype=np.uint32)
        channels = {
            name: np.asarray(handle["channel_data"][name][:])
            for name in handle.get("channel_data", {})
        }

        # HDF5 attribute values as plain Python, for JSON round-tripping.
        attributes = {}
        for key, value in handle.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            elif isinstance(value, np.generic):
                value = value.item()
            attributes[key] = value

        n_errors = len(handle["error_messages"]) if "error_messages" in handle else 0

    # ---- 2. Current channel names -----------------------------------
    renames = _legacy_renames(set(channels))
    if renames:
        channels = {renames.get(name, name): data for name, data in channels.items()}

    # ---- 3. Check -----------------------------------------------------
    # Fail loudly on anything that would make later alignment silently wrong.
    if len(timestamps) == 0:
        raise ValueError(f"{path.name} has an empty timestamps array")

    if len(message_ids) != len(timestamps):
        raise ValueError(
            f"{path.name}: {len(message_ids)} message IDs but "
            f"{len(timestamps)} timestamps"
        )

    wrong_length = {
        name: len(data) for name, data in channels.items()
        if len(data) != len(timestamps)
    }
    if wrong_length:
        raise ValueError(
            f"{path.name}: channels disagree with the timestamp count "
            f"({len(timestamps)}): {wrong_length}"
        )

    return DaqData(
        path=path,
        message_ids=message_ids,
        timestamps=timestamps,
        channels=channels,
        attributes=attributes,
        n_error_messages=n_errors,
        renamed_channels=renames,
    )


@dataclass(frozen=True)
class DaqData:
    """Everything the DAQ recorded, on the master clock."""

    path: Path
    message_ids: np.ndarray
    timestamps: np.ndarray
    #: channel name -> 0/1 state per message
    channels: dict
    attributes: dict = field(default_factory=dict)
    n_error_messages: int = 0
    #: Old channel names translated on read (old -> current); empty for a current
    #: recording.
    renamed_channels: dict = field(default_factory=dict)

    @property
    def used_legacy_names(self):
        """True if this recording named its channels 1-based."""
        return bool(self.renamed_channels)

    # ------------------------------------------------------------------
    # Basic facts
    # ------------------------------------------------------------------

    @property
    def channel_names(self):
        """Every channel's name, sorted."""
        return sorted(self.channels)

    @property
    def n_messages(self):
        return len(self.timestamps)

    @property
    def duration(self):
        """Recording length in seconds."""
        if not self.n_messages:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def reliability(self) -> float | None:
        """Percentage of messages received intact, as the DAQ reported it."""
        value = self.attributes.get("reliability")
        if value is None:
            return None
        return float(value)

    def has(self, name):
        return name in self.channels

    def channel(self, name):
        """Raise a clear error rather than a KeyError for a missing channel."""
        try:
            return self.channels[name]
        except KeyError:
            raise KeyError(
                f"No channel {name!r} in {self.path.name}. "
                f"Present: {', '.join(self.channel_names)}"
            ) from None

    # ------------------------------------------------------------------
    # Edges — how every other stream finds its place on the clock
    # ------------------------------------------------------------------

    def rising_edges(self, name):
        """Timestamps at which ``name`` went 0 -> 1."""
        data = self.channel(name)
        return self.timestamps[1:][(data[:-1] == 0) & (data[1:] == 1)]

    def falling_edges(self, name):
        """Timestamps at which ``name`` went 1 -> 0."""
        data = self.channel(name)
        return self.timestamps[1:][(data[:-1] == 1) & (data[1:] == 0)]

    def rising_message_ids(self, name):
        """Message IDs at which ``name`` went 0 -> 1.

        The message ID, not the timestamp, is what peripherals key against:
        the scales CSV and the ephys sync both record IDs.
        """
        data = self.channel(name)
        return self.message_ids[1:][(data[:-1] == 0) & (data[1:] == 1)]

    def describe(self):
        if self.reliability is None:
            return f"{self.path.name}: {self.n_messages:,} messages"
        return (
            f"{self.path.name}: {self.n_messages:,} messages, "
            f"{self.duration:.1f}s, {len(self.channels)} channels, "
            f"reliability {self.reliability:.3f}%"
        )
