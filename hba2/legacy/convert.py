"""
One legacy session -> the files the rest of the package reads.
================================================================

For a session recorded before the controller rewrite — an ``-ArduinoDAQ.h5`` with
1-based channel names, a ``_Phase_<X>_behaviour_data.json``, no ``-metadata.json``, no
trials CSV, no trial-marker channel. Post-processing proper cannot run on these, so this
does its job by other means and then hands over to the same writers, so what comes out
is the same files with the same meanings.

What it writes, and where
-------------------------
Into the session folder, beside the raw files:

* ``<session>-metadata.json`` — built from the Phase JSON header and the DAQ file, so
  the cohort can find the session. Marked ``"converted_from"``.
* ``<session>_video.avi`` — a symlink to the session's video when the rig named it
  something else (``_raw_MP.avi``, ``<time>_output.avi``).
* ``LEGACY_CONVERSION.txt`` — says which files in the folder this wrote and that none
  of them are original.

Into ``dlc/<model>/``: a symlink per DeepLabCut file, renamed the way the cohort
expects, for output DeepLabCut named after an old video name. Nothing is copied.

Into ``processed/``: the same files post-processing writes — ``session.json``,
``events.csv``, ``trials.csv``, ``notes.csv``, ``frame_times.csv``, ``scales.csv`` and
``pulses_<channel>.csv`` — using post-processing's own writers.

Nothing original is altered, moved or deleted.

    find_sessions         the session folders under a cohort
    convert_session       one session, as numbered steps
    find_files            what raw files a folder holds
    ConvertResult,
    LegacyFiles           what comes back
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from ..postprocessing import outputs
from ..postprocessing.sources import blocks, daq as daq_source, metadata
from ..postprocessing.sources.camera import CameraData
from ..postprocessing.sources.daq import DaqData
from ..postprocessing.sources.scales import ScalesData
from ..postprocessing.utils import clock
from . import oeab_daq, phase_log, trials as legacy_trials

#: Written into every converted metadata file and session.json.
CONVERTER = "hba2.legacy"
NOTE_NAME = "LEGACY_CONVERSION.txt"

SESSION_ID = re.compile(r"^\d{6}_\d{6}_.+$")
#: Everything after this in a DeepLabCut filename is the model name.
DLC_SPLIT = "DLC_"
VIDEO_SUFFIXES = ("_video.avi", "_raw_MP.avi", "_output.avi")

#: Channels that are cues or arena hardware; they go in events.csv.
BEHAVIOUR_PREFIXES = ("SENSOR", "LED_", "VALVE", "GO_CUE", "NOGO_CUE")

#: Red-rig channels left out of every output. ``SPOT<n>`` toggles hundreds of thousands
#: of times a session — the port spotlights' PWM, not anything the protocol did — and
#: as pulse files they made each NWB take hours to write. The role in session.json
#: says they were present and not written.
NOT_WRITTEN_PREFIXES = ("SPOT",)

#: A camera pulse this early means the camera was running before the DAQ, so the
#: pulses for the first frames were never recorded.
EARLY_CAMERA_S = 0.1


def find_sessions(base):
    """
    Legacy session folders under ``base`` (a Path or a string): ``<base>/<group>/<session>``,
    any depth up to 3. A session folder is one named like a session that holds a DAQ
    file. Returns a list of Paths.
    """
    found = []

    def walk(path, depth):
        for child in sorted(p for p in path.iterdir() if p.is_dir()):
            if child.name in ("processed", "dlc", "nwb", "ephys", "ephys_sorts",
                              "backup_files", "debug"):
                continue
            if SESSION_ID.match(child.name) and any(
                    p.name.endswith(("-ArduinoDAQ.h5", "-ArduinoDAQ.json"))
                    for p in child.iterdir()):
                found.append(child)
            elif depth < 3:
                walk(child, depth + 1)

    walk(Path(base), 0)
    return found


# =====================================================================
# One session
# =====================================================================

def convert_session(directory, *, rig_names, nonspatial_cues=(), overwrite=False):
    """
    Convert one session folder (a Path or a string). Failures are returned, not raised.

    Args:
        rig_names: what each old rig number was, e.g. ``{1: "Red rig 1"}``. Old rig
            numbers do not mean what new ones do, so the name is what says which.
        nonspatial_cues: a tuple of channels that cued a trial without naming a port —
            see :func:`.trials.detect`.
        overwrite: redo a session already converted.

    Returns a :class:`ConvertResult`.
    """
    files = find_files(directory)
    result = ConvertResult(session_id=files.session_id, directory=files.directory)
    out = files.directory / "processed"

    # ---- 1. Already done, or cannot be -----------------------------------
    if (out / "events.csv").exists() and not overwrite:
        result.warnings.append("already converted")
        try:
            result.n_trials = sum(1 for _ in (out / "trials.csv").open()) - 1
        except OSError:
            pass
        return result
    if files.missing:
        result.error = f"missing {', '.join(files.missing)}"
        return result

    try:
        # ---- 2. The rig's log --------------------------------------------
        log = phase_log.read(files.phase_json) if files.phase_json else None
        if log is None:
            result.warnings.append("no Phase JSON — trials have no rig log to pair with")

        # ---- 3. The DAQ, from whichever format this session has ----------
        oeab = None
        h5_mirror = False
        if files.daq is None:
            # The 2025 DAQ JSON holds the h5's arrays rather than bit strings.
            with files.daq_json.open("rb") as handle:
                head = handle.read(4096).decode("utf-8", "ignore")
            # An empty file fits neither format; say so, rather than letting the Open
            # Ephys path fail on something unrelated, like the rig's wiring.
            if not head.strip():
                raise ValueError(f"{files.daq_json.name} is empty — the DAQ log was "
                                 "never written")
            h5_mirror = '"message_ids"' in head or '"channel_data"' in head

        if files.daq is not None:
            daq = daq_source.read(files.daq)
            camera_pulses = daq.rising_edges("CAMERA") if daq.has("CAMERA") else np.empty(0)
        elif h5_mirror:
            # 2025 sessions whose h5 was never written: the JSON beside it holds the
            # same arrays.
            daq = _daq_from_json(files.daq_json)
            camera_pulses = daq.rising_edges("CAMERA") if daq.has("CAMERA") else np.empty(0)
        else:
            # The 2024 bit strings, timed by the group's Open Ephys board.
            rig_hint = None
            if log is not None:
                rig_hint = log.rig
            rig_hint = rig_hint or files.signal_rig
            oeab = oeab_daq.build(files.daq_json,
                                  oeab_daq.find_recordings(files.directory.parent),
                                  rig_hint)
            daq, camera_pulses = oeab.daq, oeab.camera_pulses
            if oeab.n_corrupt_ids:
                result.warnings.append(f"{oeab.n_corrupt_ids} corrupt DAQ message(s) dropped")
            if oeab.n_truncated:
                result.warnings.append(
                    f"the Open Ephys recording stopped before the DAQ: the last "
                    f"{oeab.n_truncated:,} DAQ message(s) have no time and were dropped, "
                    f"with any trials in them")

        clock_repair = None
        if files.daq is not None or oeab is None:
            daq, clock_repair = _repair_synthetic_clock(daq, files)
            if clock_repair:
                camera_pulses = daq.rising_edges("CAMERA")
                result.warnings.append(clock_repair["summary"])

        # ---- 4. The rig ---------------------------------------------------
        # The old rig number, from the Phase JSON, else the rig's signal file. Checked
        # against the DAQ's channel layout, which is independent of both: only the red
        # rigs (old 1 and 2) recorded SPOT and BUZZER channels.
        rig = log.rig if log is not None else None
        rig_source = "phase_json"
        if rig is None:
            rig, rig_source = files.signal_rig, "signal_file"
        red_layout = any(n.startswith(("SPOT", "BUZZER")) for n in daq.channel_names)
        if rig is None:
            # Not recorded: the layout says which kind.
            if red_layout:
                result.warnings.append("rig not recorded; DAQ layout says a red rig (1 or 2)")
            else:
                result.warnings.append("rig not recorded; DAQ layout says rig 3")
                rig = 3
        elif (rig in (1, 2)) != red_layout:
            # Recorded, and the layout disagrees.
            layout = "red rig" if red_layout else "rig 3"
            result.warnings.append(f"rig {rig} from {rig_source} disagrees with the DAQ "
                                   f"layout ({layout})")

        # ---- 5. The files the cohort needs to find this session -----------
        meta_path = _write_metadata(files, log, daq, rig, rig_names)
        result.links += _link_dlc(files)
        # The video, under the name the cohort looks for, when the rig named it otherwise.
        if files.video is not None and not files.video.name.endswith("_video.avi"):
            link = files.directory / f"{files.session_id}_video.avi"
            if not (link.exists() or link.is_symlink()):
                link.symlink_to(files.video.name)
            result.links.append(link.name)
        meta = metadata.read(meta_path)

        # ---- 6. Trials, on the DAQ ----------------------------------------
        found = legacy_trials.detect(daq, log.phase if log else None, log,
                                     files.phase_json or files.daq, nonspatial_cues)
        result.warnings += found.warnings
        result.n_trials = len(found.starts)

        # ---- 7. The camera's frames on the clock --------------------------
        cam, cam_note = _camera(files)
        if cam_note:
            result.warnings.append(cam_note)
        pulses = camera_pulses
        # Frame IDs as indices into the camera pulses. Normally frame ID n is pulse n.
        # When the camera was already running as the DAQ started, the first frames'
        # pulses were never recorded: the first pulse arrives immediately and there are
        # fewer pulses than frames, so frame ID n is pulse n - shift. Only the DAQ file
        # can miss them: the Open Ephys board records the camera whether or not the DAQ
        # has started.
        frame_ids, shift = cam.frame_ids, 0
        if oeab is None and len(pulses) and len(frame_ids):
            expected = int(frame_ids.max()) + 1
            first = float(pulses[0] - daq.timestamps[0])
            if len(pulses) < expected and first < EARLY_CAMERA_S:
                shift = expected - len(pulses)
                frame_ids = frame_ids - shift
        if shift:
            result.warnings.append(f"camera started {shift} pulse(s) before the DAQ; "
                                   f"frame IDs shifted to match")
        frames = clock.place_by_pulse_index("frames", frame_ids, pulses)

        # ---- 8. Weights on the clock, where they can be placed -----------
        # Wired scales logged, with each reading, the index of the SCALES pulse it
        # produced — the same mechanism the current rig uses. Wireless scales logged
        # only a PC time, which cannot be placed on the DAQ clock, so those sessions
        # get no scales file.
        scales_data = weights = None
        if log is not None and log.scales_have_pulse_index and daq.has("SCALES") \
                and len(log.scales) != 0:
            scales_data = ScalesData(path=files.phase_json, timestamps=log.scales[:, 0],
                                     weights=log.scales[:, 1],
                                     message_ids=log.scales[:, 2].astype(np.int64))
            weights = clock.place_by_pulse_index("scales", scales_data.message_ids,
                                                 daq.rising_edges("SCALES"))

        # ---- 9. Which channel is which ------------------------------------
        roles = {name: _role(name) for name in daq.channel_names}
        behaviour = [n for n, r in roles.items() if r == "behaviour"]
        pulse_sources = [n for n, r in roles.items() if r == "pulses"]
        touches = outputs.events.sensor_touches(daq, behaviour)
        # The DAQ's beam breaks, plus responses on blind ports taken from the rig log.
        touch_times = np.concatenate((touches[0], found.log_touches[0]))
        touch_ports = np.concatenate((touches[1], found.log_touches[1])).astype(int)
        order = np.argsort(touch_times, kind="stable")
        touch_times, touch_ports = touch_times[order], touch_ports[order]

        # ---- 10. Write, with post-processing's own writers ----------------
        written = [
            outputs.session.write(out, meta, daq, cam, found.rows, blocks.read(None),
                                  roles, "flat"),
            outputs.events.write(out, daq, behaviour, found.starts, found.ends),
            outputs.trials.write(out, found.rows, found.starts, found.ends,
                                 touch_times, touch_ports),
            outputs.notes.write(out, found.rows, len(found.starts)),
            outputs.frame_times.write(out, frames, cam),
        ]
        if weights is not None:
            written.append(outputs.scales.write(out, weights, scales_data))
        for channel in pulse_sources:
            pulse_file = outputs.pulses.write(out, daq, channel)
            if pulse_file is not None:
                written.append(pulse_file)
        result.written = written
        for item in written:
            result.warnings += list(item.warnings)

        # ---- 11. Say it was converted, and how -----------------------------
        _stamp_session_json(out / "session.json", files, found, shift, weights is not None,
                            oeab, clock_repair)
        _write_note(files, meta_path, result.links)

    except Exception as exc:                       # noqa: BLE001 — reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"
        result.warnings.append(traceback.format_exc(limit=3))
    return result


# =====================================================================
# Finding the raw files
# =====================================================================

def find_files(directory):
    """What raw files a session folder (a Path or a string) holds: a :class:`LegacyFiles`."""
    directory = Path(directory)
    sid = directory.name
    names = sorted(p.name for p in directory.iterdir() if p.is_file() or p.is_symlink())

    def first(predicate):
        hits = [n for n in names if predicate(n)]
        return directory / hits[0] if hits else None

    phases = [n for n in names if re.search(r"_Phase_.+_behaviour_data\.json$", n)]
    # More than one Phase JSON is a restarted session; the largest has the trials.
    phase = max(phases, key=lambda n: (directory / n).stat().st_size) if phases else None

    # The video, by the first suffix that has one; a link is one this converter made.
    videos = [n for n in names if n.endswith(VIDEO_SUFFIXES) and not os.path.islink(directory / n)]
    video = None
    for suffix in VIDEO_SUFFIXES:
        matching = [n for n in videos if n.endswith(suffix)]
        if matching:
            video = matching[0]
            break

    dlc = {}                    # model -> {"csv": path, "h5": path}
    for n in names:
        # Split-video fragments are partial outputs, not a session's tracking.
        if DLC_SPLIT not in n or not n.endswith((".csv", ".h5")) or "_videoDLC_" in n \
                or "_split" in n:
            continue
        stem = n.rsplit(".", 1)[0]
        model = stem.split(DLC_SPLIT, 1)[1]
        dlc.setdefault(model, {})[n.rsplit(".", 1)[1]] = directory / n

    signal = first(lambda n: re.match(r"rig_\d+_.*\.signal$", n) is not None)
    signal_rig = int(re.findall(r"rig_(\d+)", signal.name)[0]) if signal else None
    return LegacyFiles(
        directory=directory, session_id=sid,
        daq=first(lambda n: n.endswith("-ArduinoDAQ.h5")),
        daq_json=first(lambda n: n.endswith("-ArduinoDAQ.json")),
        phase_json=directory / phase if phase else None,
        tracker=first(lambda n: n.endswith("_Tracker_data.json")),
        frame_ids_backup=first(lambda n: n.endswith("frame_ids_backup.txt")),
        video=directory / video if video else None,
        dlc=dlc,
        signal_rig=signal_rig,
    )


# =====================================================================
# The DAQ
# =====================================================================

def _daq_from_json(path):
    """The 2025 DAQ JSON (a Path), which mirrors the h5's arrays, as DaqData."""
    document = json.loads(path.read_text())
    timestamps = np.asarray(document["timestamps"], dtype=float)
    ids = np.asarray(document.get("message_ids") or np.arange(len(timestamps)),
                     dtype=np.uint32)
    channels = {k: np.asarray(v, dtype=np.uint8)
                for k, v in (document.get("channel_data") or {}).items()}
    renames = daq_source._legacy_renames(set(channels))
    channels = {renames.get(k, k): v for k, v in channels.items()}
    attributes = {k: document.get(k) for k in ("reliability", "messages_per_second",
                                               "date_time", "mouse_ID") if k in document}
    return DaqData(path=path, message_ids=ids, timestamps=timestamps, channels=channels,
                   attributes=attributes,
                   n_error_messages=len(document.get("error_messages") or []),
                   renamed_channels=renames)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _repair_synthetic_clock(daq, files):
    """
    Real times for a DAQ file whose timestamps were invented.

    Some 2025 sessions crashed before the DAQ file was written, and it was rebuilt from
    the ``-backup.csv`` of message IDs and channel states, which carries no times. The
    rebuild spaced rows evenly — 1 ms apart, or the session's length spread over its
    rows — but the DAQ's real rate drifted through a session, so event order is right
    and every duration is wrong, by up to a quarter.

    The camera fixes it. It ran at a fixed frame rate and its pulses are on the DAQ by
    message ID, so camera pulse *k* happened at *k / fps*, and every message between
    two pulses is placed on the straight line between them.

    Returns the DAQ and a dict saying what was done — or the DAQ unchanged, and None,
    when the timestamps are real: real ones are never evenly spaced. ``files`` is the
    session's :class:`LegacyFiles`, for the camera's frame rate.
    """
    ts = daq.timestamps
    if len(ts) < 1000:
        return daq, None
    steps = np.diff(ts)
    if steps.max() - steps.min() > 1e-7:
        return daq, None
    if not daq.has("CAMERA"):
        raise ValueError("DAQ timestamps were invented (evenly spaced) and there is no "
                         "camera channel to recover real times from")

    fps = None
    if files.tracker is not None:
        fps = _float(json.loads(files.tracker.read_text()).get("frame_rate"))
    fps = fps or 30.0
    camera = daq.rising_message_ids("CAMERA").astype(np.float64)
    if len(camera) < 100:
        raise ValueError("DAQ timestamps were invented and too few camera pulses to fix them")

    ids = daq.message_ids.astype(np.float64)
    frame_time = np.arange(len(camera)) / fps
    fixed = np.interp(ids, camera, frame_time)
    # Before the first and after the last pulse, carry the end segments' rate on: the
    # rate over the first (or last) 100 pulses, or all of them in a shorter session.
    n_edge = min(100, len(camera) - 1)
    last_start = -101 if len(camera) > 100 else 0
    rate_in = (camera[n_edge] - camera[0]) / (n_edge / fps)
    rate_out = (camera[-1] - camera[last_start]) / (n_edge / fps)
    before, after = ids < camera[0], ids > camera[-1]
    fixed[before] = (ids[before] - camera[0]) / rate_in
    fixed[after] = frame_time[-1] + (ids[after] - camera[-1]) / rate_out

    rate = (camera[-1] - camera[0]) / frame_time[-1]
    repaired = replace(daq, timestamps=fixed,
                       attributes={**daq.attributes, "messages_per_second": rate})
    method = f"camera pulse k at k/{fps:g} s; messages between pulses on the line between them"
    summary = f"invented DAQ clock replaced from camera pulses (mean {rate:.1f} messages/s)"
    return repaired, {
        "reason": "timestamps were evenly spaced, invented when the file was rebuilt "
                  "from the backup CSV",
        "invented_step_s": float(steps[0]),
        "method": method,
        "mean_rate_messages_per_s": round(float(rate), 3),
        "summary": summary,
    }


# =====================================================================
# The files the cohort needs to find a session
# =====================================================================

def _write_metadata(files, log, daq, rig, rig_names):
    """``<session>-metadata.json``, from the Phase JSON header and the DAQ. Returns its Path."""
    header = dict(log.header) if log is not None else {}
    sid = files.session_id
    # The folder name, not the typed header: the ID was typed by hand on these rigs and
    # sometimes typed wrong (M106-1b for M106-3b, a stray comma), and the folder
    # was renamed to the right mouse afterwards. What was typed is kept beside it.
    typed = header.get("Mouse ID") or daq.attributes.get("mouse_ID")
    mouse = sid.split("_", 2)[2] if sid.count("_") >= 2 else typed
    phase = log.phase if log is not None else None
    tracker = json.loads(files.tracker.read_text()) if files.tracker else {}
    fps = header.get("FPS") or tracker.get("frame_rate")
    sources = [p.name for p in (files.phase_json, files.daq or files.daq_json, files.tracker)
               if p is not None]
    parameters = {"phase": phase}
    parameters.update({k: v for k, v in header.items() if k != "Headers"})
    rig_name = None
    if rig:
        rig_name = rig_names.get(rig, f"legacy rig {rig}")

    document = {
        "converted_from": {
            "by": CONVERTER,
            "at": datetime.now().isoformat(timespec="seconds"),
            "note": "Not written by the rig. Rebuilt from the files listed in 'sources' "
                    "so this pre-rewrite session can be read by hba2.",
            "sources": sources,
        },
        "session_id": sid,
        "mouse_id": str(mouse),
    }
    if typed and str(typed) != str(mouse):
        document["mouse_id_as_typed"] = str(typed)
    document.update({
        "mouse_weight": header.get("Mouse weight"),
        "num_trials": header.get("Total trials") or header.get("Total Trials"),
        "protocol_name": f"legacy phase {phase}" if phase else "legacy",
        "protocol_parameters": parameters,
        "start_timestamp": sid[:13],
        "rig": {"name": rig_name, "number": rig, "camera_serial": None},
        "peripherals": {
            "camera_fps": fps,
            "camera_window": {"width": tracker.get("image_width") or tracker.get("width"),
                              "height": tracker.get("image_height") or tracker.get("height")},
            "scales_enabled": daq.has("SCALES"),
        },
    })
    path = files.directory / f"{sid}-metadata.json"
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return path


def _link_dlc(files):
    """
    ``dlc/<model>/<session>_videoDLC_<model>.<ext>`` -> the original, relative. Returns
    the links made, as strings relative to the session folder.
    """
    made = []
    for model, kinds in files.dlc.items():
        folder = files.directory / "dlc" / model
        folder.mkdir(parents=True, exist_ok=True)
        for ext, target in kinds.items():
            link = folder / f"{files.session_id}_videoDLC_{model}.{ext}"
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(os.path.relpath(target, folder))
            made.append(str(link.relative_to(files.directory)))
    return made


# =====================================================================
# The camera
# =====================================================================

def _parse_stamp(value):
    try:
        return datetime.strptime(str(value), "%y%m%d_%H%M%S")
    except (TypeError, ValueError):
        return None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _camera(files):
    """
    The tracker JSON as CameraData, with its frame IDs from the backup text file if it
    lost them. Returns it and a note saying where the frame IDs came from, or None.
    """
    document = json.loads(files.tracker.read_text())
    ids = document.get("frame_IDs")
    note = None
    if not ids and document.get("data") and "frame" in (document.get("headers") or []):
        # The earliest tracker format: one row per delivered frame, frame number in a
        # column, instead of a list of IDs.
        column = document["headers"].index("frame")
        ids = [int(row[column]) for row in document["data"] if row[column] is not None]
        note = "frame IDs read from the tracker's per-frame rows"
    if not ids and files.frame_ids_backup is not None:
        ids = [int(x) for x in files.frame_ids_backup.read_text().split() if x.strip()]
        note = "frame IDs read from the frame_ids_backup text file"
    if not ids:
        raise ValueError("no frame IDs in the tracker JSON or a backup file")
    return CameraData(
        path=files.tracker,
        frame_ids=np.asarray(ids, dtype=np.int64),
        frame_rate=_float(document.get("frame_rate")),
        width=_int(document.get("image_width") or document.get("width")),
        height=_int(document.get("image_height") or document.get("height")),
        pixel_format=document.get("pixel_format"),
        start_time=_parse_stamp(document.get("start_time")),
        end_time=_parse_stamp(document.get("end_time")),
    ), note


# =====================================================================
# Channel roles
# =====================================================================

def _role(name):
    """What a DAQ channel is for: not_written, camera, scales, behaviour or pulses."""
    if name.startswith(NOT_WRITTEN_PREFIXES):
        return "not_written"
    if name == "CAMERA":
        return "camera"
    if name == "SCALES":
        return "scales"
    if name.startswith(BEHAVIOUR_PREFIXES):
        return "behaviour"
    return "pulses"


# =====================================================================
# Saying it was converted
# =====================================================================

def _stamp_session_json(path, files, found, shift, has_scales, oeab=None, clock_repair=None):
    """
    Say, in the session's own facts, that it was converted and how.

    ``found`` is the session's :class:`~.trials.LegacyTrials`, ``shift`` the camera index
    shift, ``oeab`` the :class:`~.oeab_daq.OEABClock` when the board was the clock.
    """
    document = json.loads(path.read_text())
    if oeab is None:
        daq_format = "legacy_daq_h5"
        clock_text = "the DAQ's own timestamps"
    else:
        daq_format = "legacy_daq_json_open_ephys_clock"
        clock_text = (f"Open Ephys board, {oeab.recording.relative_to(files.directory.parent)}; "
                      f"message k at DAQ pulse k ({oeab.n_pulses:,} pulses)")
    scales = "wired, by pulse index" if has_scales else "not placeable (none or wireless)"
    sources = [p.name for p in (files.phase_json, files.daq or files.daq_json,
                                files.tracker, files.frame_ids_backup)
               if p is not None]
    document["converted_from"] = {
        "by": CONVERTER,
        "at": datetime.now().isoformat(timespec="seconds"),
        "format": daq_format,
        "clock": clock_text,
        "trials": "found on the DAQ from cue onsets; rig log paired by time "
                  "(see trials.csv details_json)",
        "n_rig_log_trials": found.n_log_trials,
        "n_paired_with_rig_log": found.n_paired,
        "rig_log_trials_without_daq_cue": found.log_only,
        "pc_to_daq_fit": found.pc_to_daq,
        "stuck_channels": found.stuck_channels,
        "camera_index_shift": shift,
        "clock_repair": clock_repair,
        "scales": scales,
        "sources": sources,
    }
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")


def _write_note(files, meta_path, links):
    """
    ``LEGACY_CONVERSION.txt``: which files in the folder this wrote. ``links`` is the
    list of symlinks made, as strings.
    """
    lines = [
        "This session was recorded before the hex_behav controller rewrite.",
        f"On {datetime.now():%Y-%m-%d} {CONVERTER} added the files below so that",
        "hba2 can read it. NONE OF THEM ARE ORIGINAL RIG OUTPUT.",
        "Every original file in this folder is unchanged.",
        "",
        f"  {meta_path.name}    built from the Phase JSON header and DAQ file",
        "  processed/                        everything inside it",
    ]
    lines += [f"  {link}    symlink to the original" for link in links]
    (files.directory / NOTE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


# =====================================================================
# What comes out
# =====================================================================

@dataclass
class LegacyFiles:
    directory: Path
    session_id: str
    daq: Path | None
    #: The 2024 format: bit strings, no times — see oeab_daq.py.
    daq_json: Path | None
    phase_json: Path | None
    tracker: Path | None
    frame_ids_backup: Path | None
    video: Path | None
    dlc: dict[str, dict[str, Path]]
    signal_rig: int | None

    @property
    def missing(self):
        """Which required raw files are absent: a list naming ``daq`` and/or ``tracker``."""
        missing = []
        if not (self.daq or self.daq_json):
            missing.append("daq")
        if not self.tracker:
            missing.append("tracker")
        return missing


@dataclass
class ConvertResult:
    session_id: str
    directory: Path
    written: list = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    n_trials: int = 0

    @property
    def ok(self):
        return self.error is None

    def describe(self):
        if not self.ok:
            return f"{self.session_id}: FAILED — {self.error}"
        lines = [f"{self.session_id}: {self.n_trials} trials"]
        lines += [f"  {w.describe()}" for w in self.written]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)
