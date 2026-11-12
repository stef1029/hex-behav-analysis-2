"""
DAQ files from the red rigs in spring 2026, rewritten as they should have been recorded.
=======================================================================================

The red rigs ran the current controller from March to June 2026, so their sessions have
everything post-processing reads — but their ``-ArduinoDAQ.h5`` files were damaged in
three ways, none of which post-processing can see. This module rewrites the file, in the
same format, so the rest of the package reads the session as if none of it had happened.
It is for a **copy** of the cohort: it reads the original file and writes a new one.

What went wrong
---------------
**1. Firmware and decoder disagreed about which bit is which.** The DAQ firmware was
reflashed several times in March, and a rig ran whichever version was last put on it.

* A red rig on the 24-bit layout (``9e48ea9``) puts the camera on bit 22, which the
  decoder calls ``LASER``.
* A red rig on the 26-bit layout (``203532d``) puts the camera on bit 24 and the scales
  on bit 25 — but the writer only keeps bits 0–23, so the camera's *value* was thrown
  away. Its *edges* were not: the firmware sends a message whenever any of its 26 bits
  changes, so every camera edge is a saved message whose 24 bits equal the one before.
  The camera is rebuilt from those (:func:`camera_from_repeats`). The same layout puts
  the trial marker (pin 51) on bit 21, which the decoder calls ``EXT_1``.

Which channel holds what, per rig and date, is the cohort script's table — data, not
code — and every assignment is checked against the session (frames, trial rows).

**2. The serial reader cut packets apart.** ``serial_listen.py`` splits the stream on
``\\x02\\x01`` — one packet's end byte and the *next* packet's start byte — with a one
second timeout, and always drops the last two bytes it read. Two things follow:

* the two bytes can occur inside a packet's payload; the packet is then saved as two
  error records whose lengths add up to 7, and the two missing bytes are ``02 01``. It is
  rebuilt exactly (``split``).
* with no next packet within a second — a rig whose camera was not reaching the DAQ, so
  only behaviour changes were sent — the read times out. A lone packet is then saved as
  an "accepted" record shifted by one byte, with message ID 2**24 + n (``shifted``); two
  packets close together become a 10-byte error (the whole payload, ``err10``) and an
  8-byte one (all but the last byte, ``err8``). All three are decoded. ``err8`` and
  ``shifted`` lose the payload's last byte, which is bits 0–7: those bits are carried
  over from the message before and the row is marked in ``repair/unknown_bits``.

**3. Camera pulses the DAQ never saved.** A message lost before (2) was undone loses a
camera edge with it. After rebuilding, any camera interval longer than 1.5 periods **on
the firmware's own clock** (the message ID counts loop cycles) has pulses inserted on
the regular grid (``filled``). The PC timestamps are not used to decide this: they
arrive late now and then, and judged on them pulses would be added that never existed.

**4. Frame 0's pulse on the DAQ's first message.** When the DAQ and camera started
together, the DAQ's first message is frame 0's strobe, which as a first sample can never
be a rising edge: every frame would be timed by the next one's pulse. The pulse is put
on that message and an ``inserted`` copy of it with the camera low put before it
(:func:`frame_zero`).

What the new file holds
-----------------------
The same datasets as the rig's file, plus a ``repair`` group:

    channel_data/<name>        the arena channels as recorded, and the six link /
                               external channels under their current names, each
                               holding what the table says it is — zeros if nothing
    message_ids, timestamps    every message, rebuilt ones included, in firmware order
    error_messages             only the errors nothing could be made of
    repair/source              per row: record, split, err10, err8, shifted, inserted
    repair/unknown_bits        per row: bits carried from the message before
    repair/original_row        per row: its row in the rig's file, -1 if none
    repair/unassigned/<label>  any recorded channel the table gave no role
    repair/error_messages_original
    attrs["repair"]            JSON: what was done, and every check made

The camera is only ever *placed* from pulses that were recorded or rebuilt from recorded
messages; ``filled`` pulses sit between recorded ones. A rig whose camera never reached
the DAQ gets an all-zero ``CAMERA`` and no pulses: its video cannot be placed, and
post-processing says so with NaN frame times.

    repair_file          one session: read, rebuild the camera, write
    read_messages        every message, from the records and the reader's errors
    camera_from_bit,
    camera_from_repeats  the camera's pulses, from a channel or from repeated messages
    frame_zero           frame 0's pulse on the first message
    apply_camera         the camera line set from the pulses
    low_before_first     a camera-low copy of the first message
    _write               the new file
    Messages, Pulses,
    Assignment           the data passed between them
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

REPAIRED_BY = "hba2.legacy.red_rig_daq"

#: Bit order of the decoder that wrote each era's files (1-based arena names, as
#: recorded). ``None`` is a bit the decoder did not name.
_ARENA = ("SENSOR6", "SENSOR1", "SENSOR5", "SENSOR2", "SENSOR4", "SENSOR3",
          "LED_3", "LED_4", "LED_2", "LED_5", "LED_1", "LED_6",
          "VALVE4", "VALVE3", "VALVE5", "VALVE2", "VALVE6", "VALVE1")
BIT_ORDERS = {
    # 286a4f2: GO_CUE..LASER on 18-22.
    "go_cue": _ARENA + ("GO_CUE", "NOGO_CUE", "CAMERA", "SCALES", "LASER", None),
    # Between 9e48ea9 and d14298d; bit 21 named EXT_0, the rest as d14298d.
    "ext0": _ARENA + ("DAQ_LINK0", "DAQ_LINK1", "SCALES", "EXT_0", "LASER", "CAMERA"),
    # d14298d onwards, unchanged since.
    "ext1": _ARENA + ("DAQ_LINK0", "DAQ_LINK1", "SCALES", "EXT_1", "LASER", "CAMERA"),
}

#: The six non-arena channels every rewritten file has, under their current names.
OUTPUT_CHANNELS = ("DAQ_LINK0", "DAQ_LINK1", "SCALES", "EXT_1", "LASER", "CAMERA")

SOURCES = ("record", "split", "err10", "err8", "shifted", "inserted")
SHIFT = 1 << 24
FULL = (1 << 24) - 1
#: Bits a truncated packet still carries: its last byte (bits 0-7) is the one lost.
HIGH_BITS = FULL & ~0xFF

#: A camera interval this many periods long (firmware clock) has pulses missing.
GAP_PERIODS = 1.5
#: Gaps longer than this many periods are the camera or DAQ stopping, not lost
#: messages, and are left alone.
MAX_FILL_PERIODS = 30
#: A pulse edge within this many loop counts of an existing message is put on it.
SNAP_LOOPS = 2

#: How close to one camera period the DAQ's first message must sit before the first
#: pulse seen, for it to be taken as frame 0's own pulse.
FRAME_ZERO_TOLERANCE = 0.1


# =====================================================================
# One session
# =====================================================================

def repair_file(src, dst, assignment, n_trials=None, frame_span=None):
    """
    Read the rig's DAQ file at ``src``, write the repaired one at ``dst`` (each a Path
    or a string). Returns what was done and the checks made, which is also stored in
    the file.

    Args:
        assignment: the :class:`Assignment` for this rig and era.
        n_trials: the trials CSV's row count, for the marker check (None if unknown).
        frame_span: the tracker's frame-ID span, for the camera check (None if unknown).
    """
    bit_order = BIT_ORDERS[assignment.bit_order]

    # ---- 1. Every message ---------------------------------------------------
    m, recorded, attrs = read_messages(src, bit_order)
    bit = {name: b for b, name in enumerate(bit_order) if name is not None}
    counts = {s: int((m.source == SOURCES.index(s)).sum()) for s in SOURCES}

    # ---- 2. The camera's pulses, and its line from them --------------------
    if assignment.camera.startswith("label:"):
        pulses = camera_from_bit(m, bit[assignment.camera.split(":", 1)[1]])
    elif assignment.camera == "repeats":
        pulses = camera_from_repeats(m)
    else:
        pulses = None
    zero = None
    if pulses is not None and len(pulses.rising):
        pulses, zero = frame_zero(m, pulses)
        m, camera = apply_camera(m, pulses)
        if camera[0]:
            m, camera = low_before_first(m, camera)
    else:
        camera = np.zeros(len(m.ids), np.uint8)
    counts["inserted"] = int((m.source == SOURCES.index("inserted")).sum())

    # ---- 3. The channels, under their current names ------------------------
    out_channels = {}
    for name in bit_order[:18]:
        out_channels[name] = ((m.words >> bit[name]) & 1).astype(np.uint8)
    for name in OUTPUT_CHANNELS:
        label = assignment.channels.get(name)
        if label is not None:
            out_channels[name] = ((m.words >> bit[label]) & 1).astype(np.uint8)
        else:
            # Nothing was wired to it.
            out_channels[name] = np.zeros(len(m.ids), np.uint8)
    out_channels["CAMERA"] = camera
    used = set(bit_order[:18]) | {v for v in assignment.channels.values() if v}
    if assignment.camera.startswith("label:"):
        used.add(assignment.camera.split(":", 1)[1])
    unassigned = {name: ((m.words >> bit[name]) & 1).astype(np.uint8)
                  for name in recorded if name not in used and name in bit}
    unassigned = {name: data for name, data in unassigned.items() if data.any()}

    # ---- 4. What was done, and the checks ----------------------------------
    marker = out_channels["DAQ_LINK1"].astype(np.int8)
    camera_summary = None
    if pulses is not None:
        pulses_minus_frames = None
        if frame_span is not None:
            pulses_minus_frames = int(len(pulses.rising)) - int(frame_span)
        camera_summary = {
            "pulses": int(len(pulses.rising)), "filled": int(pulses.filled.sum()),
            "period_loops": pulses.period, "high_loops": pulses.high,
            "gaps_left_unfilled": pulses.unfilled_gaps,
            "frame_zero": zero,
            "frame_id_span": frame_span,
            "pulses_minus_frames": pulses_minus_frames}
    summary = {
        "repaired_by": REPAIRED_BY,
        "at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(src),
        "assignment": {"bit_order": assignment.bit_order, "channels": assignment.channels,
                       "camera": assignment.camera, "why": assignment.why},
        "messages": {"in_rig_file": int(len(m.original_row[m.original_row >= 0])),
                     "written": int(len(m.ids)), **counts,
                     "errors_in_rig_file": len(m.errors_original),
                     "errors_unrecovered": len(m.unrecovered),
                     "rows_with_unknown_bits_0_7": int((m.unknown != 0).sum())},
        "camera": camera_summary,
        "trial_marker": {"rising_edges": int(np.sum(np.diff(marker) == 1)),
                         "trials_csv_rows": n_trials},
        "unassigned_channels": {k: int(np.sum(np.diff(v.astype(np.int8)) == 1))
                                for k, v in unassigned.items()},
    }

    # ---- 5. Write -----------------------------------------------------------
    _write(dst, m, out_channels, unassigned, attrs, summary)
    return summary


# =====================================================================
# 1. Every message, from the records and the reader's errors
# =====================================================================

def _decode(payload):
    """Nine payload bytes -> (message ID, bits 0-23)."""
    p = payload
    return ((p[0] << 24) | (p[2] << 16) | (p[4] << 8) | p[6],
            ((p[3] << 24) | (p[5] << 16) | (p[7] << 8) | p[8]) & FULL)


def read_messages(path, bit_order):
    """
    The rig's file (a Path or a string) as one table of messages, with everything the
    reader broke undone. ``bit_order`` is one of :data:`BIT_ORDERS`.

    Returns the :class:`Messages`, the recorded channels (name -> per-row array, for
    anything the table may leave unassigned) and the file's attributes (a dict).
    """
    # ---- 1. The file --------------------------------------------------------
    with h5py.File(path, "r") as handle:
        ids = handle["message_ids"][:].astype(np.int64)
        times = handle["timestamps"][:].astype(np.float64)
        channels = {n: handle["channel_data"][n][:].astype(np.uint8)
                    for n in handle["channel_data"]}
        errors = []
        if "error_messages" in handle:
            errors = [e.decode() if isinstance(e, bytes) else str(e)
                      for e in handle["error_messages"][:]]
        attrs = {k: (v.item() if isinstance(v, np.generic) else v)
                 for k, v in handle.attrs.items()}

    words = np.zeros(len(ids), np.int64)
    for bit, name in enumerate(bit_order):
        if name is not None and name in channels:
            words |= channels[name].astype(np.int64) << bit

    rows = {"ids": [], "times": [], "words": [], "unknown": [], "source": [], "orig": []}

    def add(i, t, w, unknown, source, orig=-1):
        rows["ids"].append(i)
        rows["times"].append(t)
        rows["words"].append(w & FULL)
        rows["unknown"].append(unknown)
        rows["source"].append(SOURCES.index(source))
        rows["orig"].append(orig)

    # ---- 2. Records, and the shifted ones among them ------------------------
    # Shifted records: ID 2**24 + n where the IDs around them say otherwise. Real IDs
    # pass 2**24 by themselves after ~50 min, so the band alone is not enough.
    band = (ids >= SHIFT) & (ids < SHIFT + (1 << 16))
    shifted = np.zeros(len(ids), bool)
    if band.any() and (~band).sum() >= 2:
        expected = np.interp(times, times[~band], ids[~band].astype(float))
        shifted = band & (np.abs(ids - expected) > 50_000)
    for row in np.flatnonzero(~shifted):
        add(ids[row], times[row], words[row], 0, "record", row)
    shifted_rows = []
    for row in np.flatnonzero(shifted):
        # Bytes [01 id3 w4 id2 w3 id1 w2 id0 w1]: the saved word is id1 id0 w1 and the
        # saved ID is 01 w4 w3 w2. The ID's high half comes from its neighbours below.
        w = words[row]
        bits = ((ids[row] & 0xFF) << 16) | ((w & 0xFF) << 8)
        shifted_rows.append(len(rows["ids"]))
        add((w >> 8) & 0xFFFF, times[row], bits, 0xFF, "shifted", row)

    # ---- 3. The reader's errors ---------------------------------------------
    parsed = []
    for entry in errors:
        try:
            counter, hexed, t = ast.literal_eval(entry)
            parsed.append((int(counter), bytes.fromhex(hexed), float(t), entry))
        except (ValueError, SyntaxError, TypeError):
            parsed.append((None, None, None, entry))
    used = set()
    for k, (counter, raw, t, _) in enumerate(parsed):
        if k in used or raw is None:
            continue
        if len(raw) == 10 and raw[0] == 0x01:
            i, w = _decode(raw[1:])
            add(i, t, w, 0, "err10")
            used.add(k)
        elif len(raw) == 8:
            i, w = _decode(raw + b"\x00")
            add(i, t, w & HIGH_BITS, 0xFF, "err8")
            used.add(k)
        elif len(raw) <= 7 and k + 1 < len(parsed) and parsed[k + 1][1] is not None \
                and parsed[k + 1][0] == counter + 1 and len(raw) + len(parsed[k + 1][1]) == 7:
            # A payload cut at an internal 02 01: the two fragments, with those bytes back.
            i, w = _decode(raw + b"\x02\x01" + parsed[k + 1][1])
            add(i, t, w, 0, "split")
            used.update((k, k + 1))
    unrecovered = [p[3] for k, p in enumerate(parsed) if k not in used]

    # ---- 4. A shifted record's ID: its low 16 bits, the time says the rest --
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["ids"] = out["ids"].astype(np.int64)
    if shifted_rows:
        all_ids, all_times, at = out["ids"], out["times"], np.asarray(shifted_rows)
        known = np.ones(len(all_ids), bool)
        known[at] = False
        estimate = np.interp(all_times[at], all_times[known],
                             all_ids[known].astype(float)).astype(np.int64)
        candidate = (estimate & ~0xFFFF) | all_ids[at]
        candidate = np.where(candidate - estimate > 0x8000, candidate - 0x10000, candidate)
        candidate = np.where(estimate - candidate > 0x8000, candidate + 0x10000, candidate)
        all_ids = all_ids.copy()
        all_ids[at] = candidate
        out["ids"] = all_ids

    # ---- 5. In firmware order -----------------------------------------------
    order = np.lexsort((out["source"], out["ids"]))
    # A message rebuilt twice (it should not happen) keeps its first form only.
    keep = np.r_[True, np.diff(out["ids"][order]) != 0]
    order = order[keep]
    messages = Messages(
        ids=out["ids"][order], times=out["times"][order].astype(np.float64),
        words=out["words"][order].astype(np.int64),
        unknown=out["unknown"][order].astype(np.int64),
        source=out["source"][order].astype(np.uint8),
        original_row=out["orig"][order].astype(np.int64),
        unrecovered=unrecovered, errors_original=errors)

    # ---- 6. Bits a truncated packet lost, carried from the message before ---
    for row in np.flatnonzero(messages.unknown):
        before = messages.words[row - 1] if row else 0
        messages.words[row] = ((messages.words[row] & ~messages.unknown[row])
                               | (before & messages.unknown[row]))
    return messages, channels, attrs


# =====================================================================
# 2. The camera's pulses
# =====================================================================

def _pair_and_fill(rising, falling, period=None, high=None):
    """
    Each rising edge (loop counts) given its falling edge, and pulses inserted on the
    grid in gaps of up to :data:`MAX_FILL_PERIODS`. Returns :class:`Pulses`.
    """
    rising = np.asarray(rising, np.int64)
    falling = np.asarray(falling, np.int64)
    if len(rising) < 10:
        return Pulses(rising, falling[:len(rising)], np.zeros(len(rising), bool),
                      np.nan, np.nan)
    period = period or float(np.median(np.diff(rising)))
    # Each rising edge's own falling edge, or a typical pulse length after it.
    nxt = np.searchsorted(falling, rising, side="right")
    own = (nxt < len(falling)) & (falling[np.clip(nxt, 0, len(falling) - 1)] - rising < period / 2)
    if high is None or not np.isfinite(high):
        high = float(np.median(falling[nxt[own]] - rising[own])) if own.any() else period / 4
    fall = np.where(own, falling[np.clip(nxt, 0, len(falling) - 1)],
                    rising + round(high)).astype(np.int64)

    new_r, unfilled = [], 0
    for a, b in zip(rising[:-1], rising[1:]):
        n = int(round((b - a) / period)) - 1
        if (b - a) > GAP_PERIODS * period and n >= 1:
            if n > MAX_FILL_PERIODS:
                unfilled += 1
                continue
            new_r += list(np.round(a + (b - a) * np.arange(1, n + 1) / (n + 1)).astype(np.int64))
    new_r = np.asarray(new_r, np.int64)
    r = np.r_[rising, new_r]
    f = np.r_[fall, new_r + round(high)]
    filled = np.r_[np.zeros(len(rising), bool), np.ones(len(new_r), bool)]
    order = np.argsort(r, kind="stable")
    return Pulses(r[order], f[order], filled[order], period, high, unfilled)


def camera_from_bit(m, bit):
    """Pulses on a channel the camera was wired to: bit ``bit`` of the :class:`Messages`."""
    x = ((m.words >> bit) & 1).astype(np.int8)
    step = np.diff(x)
    rising = m.ids[1:][step == 1]
    falling = m.ids[1:][step == -1]
    return _pair_and_fill(rising, falling)


def camera_from_repeats(m):
    """
    Pulses on a bit the writer did not keep, from the messages its edges caused.

    A message whose 24 saved bits equal the message before was sent for a change in a
    bit that was not saved — on these rigs, only the camera toggled there at frame rate.
    The line is high for a short part of each frame, so an edge followed by another
    within half a period is a rising then a falling edge.
    """
    full = (m.unknown == 0)
    ids, words = m.ids[full], m.words[full]
    edge = np.r_[False, words[1:] == words[:-1]]
    candidates = ids[edge]
    if len(candidates) < 10:
        return Pulses(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, bool),
                      np.nan, np.nan)
    gaps = np.diff(candidates)
    period = float(np.median(gaps[:-1] + gaps[1:]))      # one short + one long interval
    rising, falling, k = [], [], 0
    while k < len(candidates) - 1:
        if candidates[k + 1] - candidates[k] < period / 2:
            rising.append(candidates[k])
            falling.append(candidates[k + 1])
            k += 2
        else:
            k += 1
    rising, falling = np.asarray(rising), np.asarray(falling)
    high = float(np.median(falling - rising)) if len(rising) else np.nan
    # A lone edge — its partner coincided with another change, or was lost: place it by
    # where it sits against the pulses around it.
    paired = set(rising.tolist()) | set(falling.tolist())
    lone = np.array([c for c in candidates if c not in paired], dtype=np.int64)
    if len(lone) and len(rising):
        before = rising[np.clip(np.searchsorted(rising, lone) - 1, 0, len(rising) - 1)]
        phase = np.mod(lone - before, period)
        as_falling = np.abs(phase - high) < np.abs(np.minimum(phase, period - phase))
        extra = np.where(as_falling, lone - high, lone).round().astype(np.int64)
        rising = np.sort(np.r_[rising, extra])
        falling = np.sort(np.r_[falling, extra + round(high)])
    return _pair_and_fill(rising, falling, period=period, high=high)


def frame_zero(m, pulses):
    """
    Frame 0's pulse, when it is the DAQ's first message.

    In most sessions where the DAQ and camera started together, the DAQ's first message
    is exactly one camera period before the first rising edge it holds (1.00 ± 0.01
    periods in 62 of 63 such sessions; a DAQ started earlier leads by 0.25-1 s). That
    message was sent for frame 0's strobe, but a first sample has nothing before it, so
    no rising edge can be seen there and every frame would be timed by the next frame's
    pulse — 33 ms late. Confirmed in the video: the mouse's nose disappears into the
    port one frame earlier than the beam break in exactly those sessions (red rig 1).

    Returns the pulses with frame 0's added on the first message when it was lost, and
    what was found: ``restored``, ``on first message`` (already there — the rebuild from
    repeats keeps it) or ``seen`` (an ordinary rising edge, nothing to do).
    """
    if not len(pulses.rising) or not np.isfinite(pulses.period):
        return pulses, "seen"
    lead = (pulses.rising[0] - m.ids[0]) / pulses.period
    if abs(lead - 1) <= FRAME_ZERO_TOLERANCE:
        r0 = int(m.ids[0])
        return Pulses(np.r_[r0, pulses.rising], np.r_[r0 + round(pulses.high), pulses.falling],
                      np.r_[True, pulses.filled], pulses.period, pulses.high,
                      pulses.unfilled_gaps), "restored"
    if pulses.rising[0] - m.ids[0] <= SNAP_LOOPS:
        return pulses, "on first message"
    return pulses, "seen"


def apply_camera(m, pulses):
    """
    The messages with the camera's line set from ``pulses``, adding a message for any
    edge no message sits on. Returns the messages and the new CAMERA channel.
    """
    edges = np.r_[pulses.rising, pulses.falling]
    pos = np.searchsorted(m.ids, edges)
    near = np.zeros(len(edges), bool)
    for side in (pos - 1, pos):
        ok = (side >= 0) & (side < len(m.ids))
        near[ok] |= np.abs(m.ids[np.clip(side, 0, len(m.ids) - 1)][ok] - edges[ok]) <= SNAP_LOOPS
    missing = np.unique(edges[~near])
    if len(missing):
        at = np.searchsorted(m.ids, missing)
        prev = np.clip(at - 1, 0, len(m.ids) - 1)
        m = Messages(
            ids=np.insert(m.ids, at, missing),
            times=np.insert(m.times, at, np.interp(missing, m.ids, m.times)),
            words=np.insert(m.words, at, m.words[prev]),
            unknown=np.insert(m.unknown, at, 0),
            source=np.insert(m.source, at, SOURCES.index("inserted")),
            original_row=np.insert(m.original_row, at, -1),
            unrecovered=m.unrecovered, errors_original=m.errors_original)
    # High from each rising edge up to (not including) its falling edge, by loop count:
    # a message snapped to an edge takes the edge's level.
    r = np.searchsorted(pulses.rising, m.ids + SNAP_LOOPS, side="right") - 1
    f_of_r = pulses.falling[np.clip(r, 0, len(pulses.falling) - 1)] if len(pulses.falling) else r
    camera = ((r >= 0) & (m.ids < np.asarray(f_of_r) - SNAP_LOOPS)).astype(np.uint8)
    return m, camera


def low_before_first(m, camera):
    """
    A copy of the first message with the camera low put before it, so a pulse on the
    first message is a rising edge post-processing can see. Same ID and time — the ID
    may already be 0 — and marked ``inserted``.
    """
    m = Messages(ids=np.r_[m.ids[0], m.ids], times=np.r_[m.times[0], m.times],
                 words=np.r_[m.words[0], m.words], unknown=np.r_[m.unknown[0], m.unknown],
                 source=np.r_[SOURCES.index("inserted"), m.source],
                 original_row=np.r_[-1, m.original_row],
                 unrecovered=m.unrecovered, errors_original=m.errors_original)
    return m, np.r_[np.uint8(0), camera]


# =====================================================================
# 3. The new file
# =====================================================================

def _plain(value):
    """What ``json.dumps`` makes of a value it cannot write itself."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write(dst, m, channels, unassigned, attrs, summary):
    """The repaired file at ``dst``, written beside it first and moved into place."""
    dst = Path(dst)
    tmp = dst.with_suffix(".h5.tmp")
    string = h5py.string_dtype()
    with h5py.File(tmp, "w") as out:
        out.create_dataset("message_ids", data=m.ids.astype(np.uint32))
        out.create_dataset("timestamps", data=m.times)
        group = out.create_group("channel_data")
        for name, data in channels.items():
            group.create_dataset(name, data=data, compression="gzip")
        out.create_dataset("error_messages",
                           data=np.array(m.unrecovered, dtype=object), dtype=string)
        repair = out.create_group("repair")
        repair.create_dataset("source", data=m.source, compression="gzip")
        repair.create_dataset("unknown_bits", data=m.unknown.astype(np.uint32), compression="gzip")
        repair.create_dataset("original_row", data=m.original_row, compression="gzip")
        repair.attrs["source_codes"] = json.dumps(dict(enumerate(SOURCES)))
        repair.create_dataset("error_messages_original",
                              data=np.array(m.errors_original, dtype=object), dtype=string)
        loose = repair.create_group("unassigned")
        for name, data in unassigned.items():
            loose.create_dataset(name, data=data, compression="gzip")
        for key, value in attrs.items():
            out.attrs[key] = value
        out.attrs["No_of_messages"] = len(m.ids)
        out.attrs["repair"] = json.dumps(summary, default=_plain)
    tmp.replace(dst)


# =====================================================================
# The data passed between them
# =====================================================================

@dataclass(frozen=True)
class Assignment:
    """Which recorded label holds each output channel, for one rig and era."""

    bit_order: str
    #: output channel -> recorded label; a label of ``None`` means nothing was wired.
    channels: dict
    #: ``"label:<name>"``, ``"repeats"`` (the hidden-bit rebuild) or ``"none"``.
    camera: str
    why: str = ""


@dataclass
class Messages:
    """Every message the DAQ sent that could be recovered, in firmware order."""

    ids: np.ndarray            #: int64, loop count
    times: np.ndarray          #: float64, PC seconds since acquisition start
    words: np.ndarray          #: int64, bits 0-23 in the era's decoder order
    unknown: np.ndarray        #: int64, bits carried over rather than read
    source: np.ndarray         #: uint8, index into SOURCES
    original_row: np.ndarray   #: int64, -1 when not a row of the rig's file
    unrecovered: list = field(default_factory=list)   #: raw error entries left over
    errors_original: list = field(default_factory=list)


@dataclass
class Pulses:
    rising: np.ndarray      #: loop counts
    falling: np.ndarray
    filled: np.ndarray      #: bool per pulse: inserted on the grid, not recorded
    period: float           #: loop counts per frame
    high: float             #: loop counts the line stays high
    unfilled_gaps: int = 0  #: gaps too long to be lost messages
