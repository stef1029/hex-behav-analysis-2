"""
Reader for ``<session>-metadata.json``.

Written by the controller at session *start*, so it describes what was set up
rather than what happened. It is the authority for mouse identity, rig, protocol
and every protocol parameter.

Two things worth knowing about what comes back:

* ``mouse_id`` comes from the file, not from the folder name.
* The whole document is kept — mouse weight, protocol name, every protocol
  parameter, camera details and the ephys block — not a summary of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%y%m%d_%H%M%S"

#: Session layout as recorded by the rig, when it recorded one at all.
#:
#:   1  acquisition files loose in the session folder, channels named 1-based
#:   2  acquisition files in <session>/raw, channels named 0-based
#:
#: Read as provenance, never enforced. Which layout a session uses is detected from the
#: session itself — by the cohort, from the folder, and by the DAQ reader, from the
#: channel names — both of which are more reliable than a field that older sessions do
#: not carry at all.
LAYOUT_UNRECORDED = 1


# ----------------------------------------------------------------------
# Coercion helpers — the rig writes some fields inconsistently across
# versions, so every accessor tolerates a missing or oddly-typed value.
# ----------------------------------------------------------------------

def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value) -> str | None:
    if value is None:
        return None
    return str(value)


def read(path):
    """
    Parse a session metadata JSON (a Path or a string) into a :class:`SessionMeta`.

    Raises:
        ValueError: if the file is not valid JSON, or has no ``mouse_id``.
            Both mean the session cannot be identified, which is fatal.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc

    mouse_id = document.get("mouse_id")
    if not mouse_id:
        raise ValueError(f"{path.name} has no mouse_id")

    # Absent means the rig predates the field, which is layout 1 by definition.
    layout_version = _as_int(document.get("layout_version")) or LAYOUT_UNRECORDED

    # A ``YYMMDD_HHMMSS`` session timestamp, or None if absent or unparseable.
    start_time = None
    stamp = document.get("start_timestamp")
    if stamp:
        try:
            start_time = datetime.strptime(str(stamp), TIMESTAMP_FORMAT)
        except ValueError:
            start_time = None

    parameters = document.get("protocol_parameters") or {}
    rig = document.get("rig") or {}
    peripherals = document.get("peripherals") or {}
    window = peripherals.get("camera_window") or {}

    return SessionMeta(
        session_id=document.get("session_id") or path.name.split("-metadata")[0],
        mouse_id=str(mouse_id),
        start_time=start_time,
        layout_version=layout_version,
        protocol_name=document.get("protocol_name") or "",
        phase=_as_str(parameters.get("phase", parameters.get("behaviour_phase"))),
        mouse_weight=_as_float(document.get("mouse_weight")),
        requested_trials=_as_int(document.get("num_trials")),
        rig_name=rig.get("name"),
        rig_number=_as_int(rig.get("number")),
        camera_serial=_as_str(rig.get("camera_serial")),
        camera_fps=_as_float(peripherals.get("camera_fps")),
        camera_width=_as_int(window.get("width")),
        camera_height=_as_int(window.get("height")),
        scales_enabled=bool(peripherals.get("scales_enabled", False)),
        protocol_parameters=parameters,
        ephys=document.get("ephys"),
        raw=document,
    )


@dataclass(frozen=True)
class SessionMeta:
    """Session setup, as the rig recorded it."""

    session_id: str
    mouse_id: str
    start_time: datetime | None

    #: On-disk layout the rig recorded, or 1 when it recorded none. Provenance only.
    layout_version: int

    protocol_name: str
    phase: str | None
    mouse_weight: float | None
    requested_trials: int | None

    rig_name: str | None
    rig_number: int | None
    camera_serial: str | None

    camera_fps: float | None
    camera_width: int | None
    camera_height: int | None
    scales_enabled: bool

    #: Every protocol parameter, verbatim.
    protocol_parameters: dict = field(default_factory=dict)
    #: Ephys block when the session was recorded with a probe, else None.
    ephys: dict | None = None
    #: The entire document, so nothing is ever lost.
    raw: dict = field(default_factory=dict)

    @property
    def laser_parameters(self):
        """
        The laser-related protocol parameters, if any, as a dict.

        Covers both the pulse settings (``laser_pulse_frequency_hz`` and
        friends) and the stimulation slots (``stim_slot_N_*``), which give the
        *intended* stimulation window relative to the cue. What was actually
        delivered comes from the DAQ laser channel, not from here.
        """
        return {
            key: value
            for key, value in self.protocol_parameters.items()
            if key.startswith("laser_") or key.startswith("stim_slot_")
        }

    @property
    def has_laser(self):
        """True if the protocol had laser stimulation switched on."""
        return bool(self.protocol_parameters.get("laser_stim_enabled", False))
