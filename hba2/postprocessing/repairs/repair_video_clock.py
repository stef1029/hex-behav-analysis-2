"""
Video frame times for sessions whose camera never reached the DAQ — a one-off fix.
=================================================================================

**Run after post-processing.** For sessions whose ``frame_times.csv`` is untimed because
no camera pulse was recorded (red rig 2 from 21 May 2026).

Why it can be done
------------------
Where the camera's pulses *were* recorded, frame time against frame ID is a straight
line to about a millisecond for a whole session, and its slope — the camera's own frame
clock against the DAQ's — is the same for every session on a rig (29.9989 fps on both
red rigs, within a part per million). So a session without pulses needs only one number,
the moment frame ID 0 was exposed, and::

    t(frame) = t0 + frame_id / rig_rate

``t0`` is found by lining up something the video shows with something the DAQ recorded:
the beam breaks. When a port's beam is broken the mouse is at that port, and the video
there changes. Two independent ways of seeing that are used:

* **brightness** — learned per rig from sessions with recorded pulses: for each port, the
  pixels whose brightness follows that port's sensor (the mouse is dark against a light
  floor). In a session to fix, those pixels' brightness is cross-correlated with the
  sensors over candidate ``t0``. Needs only the video.
* **occlusion** — from DeepLabCut: a beam is broken by the snout going into the port,
  which hides the nose. "Head at port p, nose hidden" is cross-correlated with beam p's
  state over candidate ``t0`` (section 8). Needs no lighting model, so it carries to a
  relit rig; the brightness ``t0`` centres its search.

Both are first run **blind on sessions with recorded pulses**, where the true ``t0`` is
known, to measure how often and how closely they get it right; a session is fixed only
when its own fit passes the checks that measurement justifies.

Stages, each a subcommand::

    extract   one downsampled brightness movie per session (cluster array job)
    learn     the per-port hotspot maps, per rig, from sessions with pulses
    validate  the brightness fit, blind, on sessions with pulses
    fit       the brightness fit on the sessions without
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

#: Brightness movies are stored this small: 1280x1024 / 16. A port mouth is a few of
#: these pixels, and a mouse at it several.
SMALL = (80, 64)
WORK = "alignment_work"
REPORT = "alignment_report"


# =====================================================================
# 1. Brightness movies
# =====================================================================

def extract(video, out):
    """
    Decode every frame, average to ``SMALL``, save as uint8 (frames, rows, cols).

    ``video`` and ``out`` are each a Path or a string. Returns what was decoded, as
    also written beside the movie.
    """
    import cv2
    capture = cv2.VideoCapture(str(video))
    n_header = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    movie = np.empty((max(n_header, 1) + 1000, SMALL[1], SMALL[0]), dtype=np.uint8)
    n = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if n == len(movie):
            movie = np.concatenate([movie, np.empty_like(movie[:10000])])
        gray = frame[:, :, 0] if frame.ndim == 3 else frame
        movie[n] = cv2.resize(gray, SMALL, interpolation=cv2.INTER_AREA)
        n += 1
    capture.release()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, movie[:n])
    tmp.replace(out)
    info = {"video": str(video), "frames_decoded": n, "frames_in_header": n_header,
            "size": SMALL}
    out.with_suffix(".json").write_text(json.dumps(info))
    return info


# =====================================================================
# 2. What the DAQ says, and how sure it is
# =====================================================================

N_PORTS = 6
#: Bits 0-5 of a rebuilt red-rig message are the sensors; where the message lost its
#: last byte they were carried over, not read (legacy/red_rig_daq.py).
SENSOR_BITS = 0x3F


#: The LEDs are bits 6-11; a truncated packet loses 6 and 7.
LED_BITS = 0xFC0


def sensors(daq_path):
    """Each port's beam state — see :func:`channel_states`."""
    return channel_states(daq_path, "SENSOR")


def leds(daq_path):
    """Each port's LED — see :func:`channel_states`."""
    return channel_states(daq_path, "LED_")


def channel_states(daq_path, family):
    """
    One per-port channel family on the DAQ clock, and the spans where it is not known.

    Returns ``times`` (message times), ``broken`` (messages x ports: 1 = beam broken —
    the sensors are active-low — or LED on) and ``uncertain`` (list of (start, end) DAQ
    times where those bits were carried over rather than read, left out of any fit).
    """
    import h5py
    from hba2.postprocessing.sources import daq as daq_source
    daq = daq_source.read(daq_path)
    raw = np.stack([daq.channel(f"{family}{p}").astype(np.int8) for p in range(N_PORTS)],
                   axis=1)
    broken = 1 - raw if family == "SENSOR" else raw
    mask = SENSOR_BITS if family == "SENSOR" else LED_BITS
    uncertain = []
    with h5py.File(daq_path, "r") as handle:
        if "repair/unknown_bits" in handle:
            unknown = (handle["repair/unknown_bits"][:] & mask) != 0
            rows = np.flatnonzero(unknown)
            known_rows = np.flatnonzero(~unknown)
            for row in rows:
                # Unknown from this message to the next one read, or to the end.
                if np.searchsorted(known_rows, row) < len(known_rows):
                    nxt = known_rows[np.searchsorted(known_rows, row)]
                    end = daq.timestamps[nxt]
                else:
                    end = daq.timestamps[-1]
                uncertain.append((float(daq.timestamps[row]), float(end)))

    # Overlapping spans merged into one.
    merged = []
    for a, b in sorted(uncertain):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    merged = [tuple(span) for span in merged]
    return {"times": daq.timestamps, "broken": broken, "uncertain": merged}


def state_at(s, t):
    """Beam state (len(t) x ports) at DAQ times ``t``; NaN where unknown or outside."""
    idx = np.searchsorted(s["times"], t, side="right") - 1
    out = s["broken"][np.clip(idx, 0, len(s["times"]) - 1)].astype(np.float32)
    out[(idx < 0) | (t > s["times"][-1])] = np.nan
    for a, b in s["uncertain"]:
        lo, hi = np.searchsorted(t, [a, b])
        out[lo:hi] = np.nan
    return out


# =====================================================================
# 3. Hotspots: which pixels follow each port's beam
# =====================================================================

#: Pixels kept per port: the most strongly darkened when that port's beam is broken.
HOTSPOT_PIXELS = 40
#: Every n-th frame is enough to learn a map from.
LEARN_STEP = 3


def session_map(movie, frame_t, s) -> np.ndarray | None:
    """
    Per port, mean frame with only that port's beam broken minus mean frame with none,
    in units of each pixel's standard deviation. ports x rows x cols.
    """
    pick = np.arange(0, min(len(movie), len(frame_t)), LEARN_STEP)
    pick = pick[np.isfinite(frame_t[pick])]
    state = state_at(s, frame_t[pick])
    ok = np.all(np.isfinite(state), axis=1)
    pick, state = pick[ok], state[ok]
    frames = movie[pick].astype(np.float32)
    sd = frames.std(axis=0) + 1.0
    none = state.sum(axis=1) == 0
    if none.sum() < 100:
        return None
    base = frames[none].mean(axis=0)
    maps = np.full((N_PORTS,) + movie.shape[1:], np.nan, np.float32)
    for p in range(N_PORTS):
        only = (state[:, p] == 1) & (state.sum(axis=1) == 1)
        if only.sum() >= 30:
            maps[p] = (frames[only].mean(axis=0) - base) / sd
    return maps


#: Frames either side of an onset compared for an edge map: after [0, EDGE_AFTER)
#: against before [-EDGE_BEFORE[0], -EDGE_BEFORE[1]). An LED lights within a frame.
EDGE_AFTER = 2
EDGE_BEFORE = (5, 1)


def session_edge_map(movie, frame_t, s) -> np.ndarray | None:
    """
    Per port, what changes in the picture *as* the channel switches on: frames just
    after each onset minus frames just before, averaged, in pixel standard deviations.

    Given LED states this finds each LED, which lights in the frame it is switched on
    in — a physical event with no behaviour in between, so it times a frame to within
    one on any rig and under any lighting. (Given beam states it finds the port mouth,
    which a mouse darkens a variable ~0.2 s before the beam deeper in breaks.)
    Brightening, so the matched filter takes the most *positive* pixels — see
    :func:`weights_from_maps`.
    """
    n = min(len(movie), len(frame_t))
    good = np.isfinite(frame_t[:n])
    if good.sum() < 1000:
        return None
    t = frame_t[:n]
    maps = np.full((N_PORTS,) + movie.shape[1:], np.nan, np.float32)
    sd = movie[:n:7].astype(np.float32).std(axis=0) + 1.0
    for p in range(N_PORTS):
        x = s["broken"][:, p].astype(np.int8)
        onsets = s["times"][1:][np.diff(x) == 1]
        # Onsets where the channel was carried over, not read, are left out.
        carried = np.zeros(len(onsets), bool)
        for a, b in s["uncertain"]:
            carried |= (onsets >= a) & (onsets < b)
        onsets = onsets[~carried]
        k = np.searchsorted(t, onsets)                 # first frame at or after the break
        k = k[(k >= EDGE_BEFORE[0]) & (k + EDGE_AFTER < n)]
        if len(k) < 15:
            continue
        acc = np.zeros(movie.shape[1:], np.float64)
        for kk in k:
            after = movie[kk:kk + EDGE_AFTER].astype(np.float32).mean(axis=0)
            before = movie[kk - EDGE_BEFORE[0]:kk - EDGE_BEFORE[1]].astype(np.float32).mean(axis=0)
            acc += after - before
        maps[p] = (acc / len(k)) / sd
    return maps


def weights_from_maps(maps, brighten=False):
    """
    A matched filter per port: its most darkened pixels (a mouse) — or, with
    ``brighten``, its most brightened (an LED) — weighted by how much.
    """
    w = np.zeros_like(maps)
    sign = 1.0 if brighten else -1.0
    for p in range(N_PORTS):
        if not np.isfinite(maps[p]).any():
            continue
        m = sign * np.nan_to_num(maps[p], nan=0.0)
        keep = np.argsort(m.ravel())[::-1][:HOTSPOT_PIXELS]
        flat = np.zeros(m.size, np.float32)
        flat[keep] = m.ravel()[keep]
        w[p] = flat.reshape(m.shape) / max(flat.sum(), 1e-6)
    return w


def port_signals(movie, weights, chunk=20000, detrend=True, darkness=True):
    """frames x ports: weighted darkness (or brightness) at each port's hotspot."""
    from scipy.ndimage import median_filter
    out = np.empty((len(movie), N_PORTS), np.float32)
    flat_w = weights.reshape(N_PORTS, -1)
    for start in range(0, len(movie), chunk):
        block = movie[start:start + chunk].reshape(-1, flat_w.shape[1]).astype(np.float32)
        out[start:start + chunk] = (-1 if darkness else 1) * (block @ flat_w.T)
    # Lighting drifts over minutes; a mouse visit lasts seconds. ~10 s running median out.
    if detrend:
        for p in range(N_PORTS):
            out[:, p] -= median_filter(out[:, p], size=301, mode="nearest")
    return out


# =====================================================================
# 4. The offset: brightness against beams over candidate t0
# =====================================================================

#: How far either side of the prior to look.
SEARCH_S = 120.0
#: The fine stage looks this far either side of the coarse peak.
FINE_S = 1.0
#: A candidate t0 is scored only where at least this share of the frames that could
#: overlap the DAQ do — short sessions otherwise score on a handful of frames.
MIN_OVERLAP = 0.8
#: A port needs this many beam breaks (onsets) to take part.
MIN_BREAKS = 15
#: Sessions whose coarse peak stands this far above the rest are trusted to measure the
#: fine stage's latency from.
TRUST_PEAK_Z = 8.0


def _correlate(x_grid, y, n_lags):
    """
    Summed per-port normalised cross-correlation; lags with poor overlap are NaN.

    Returns ``(score per lag, ports used)``.
    """
    score = np.zeros(n_lags)
    weight = np.zeros(n_lags)
    used = 0
    for p in range(N_PORTS):
        x, yy = x_grid[:, p], y[:, p]
        mx, my = np.isfinite(x), np.isfinite(yy)
        if mx.sum() < 100 or my.sum() < 100 or np.nanstd(x) == 0 or np.nanstd(yy) == 0:
            continue
        xz = np.where(mx, (x - np.nanmean(x)) / np.nanstd(x), 0.0)
        yz = np.where(my, (yy - np.nanmean(yy)) / np.nanstd(yy), 0.0)
        c = _xcorr(yz, xz, n_lags)
        overlap = _xcorr(my.astype(float), mx.astype(float), n_lags)
        valid = overlap >= MIN_OVERLAP * overlap.max()
        score += np.where(valid, c / np.maximum(overlap, 1.0), 0.0)
        weight += valid
        used += 1
    out = np.where(weight == used, score / max(used, 1), np.nan) if used else score * np.nan
    return out, used


def fit_t0(level, fines, frame_ids, rate, s, prior, search=SEARCH_S):
    """
    Candidate t0s for one session: a coarse one, and a fine one per signal in ``fines``.

    Each frame k sits at ``t0 + frame_ids[k] / rate``. For every candidate t0 on a
    one-frame grid the DAQ's states are read at those times and correlated with the
    hotspot signals, port by port, then averaged; one FFT cross-correlation per port
    over a uniform frame-ID grid, so the whole range costs seconds.

    * **coarse** — the *level* signals (where the mouse sits) against beam state over
      ±``search``: broad, but hard to fool, so it finds the right second;
    * **fine**, for each ``name: (edge_signal, states)`` in ``fines`` — the hotspots'
      frame-to-frame change against the channel's switching, within ±``FINE_S`` of the
      coarse peak, about a frame wide. ``"led"``: the port LEDs, which light in the frame
      they are switched on in; ``"beam"``: the port mouths, which a mouse darkens a
      little before the beam breaks. Which to believe on a rig, and its latency, is
      what validation measures.
    """
    breaks = np.diff(s["broken"].astype(np.int8), axis=0).clip(0).sum(axis=0)
    ports = breaks >= MIN_BREAKS
    if not ports.any():
        return {"ok": False, "reason": "no port with enough beam breaks"}
    n_id = int(frame_ids.max()) + 1
    lags = int(round(search * rate))
    t_first = prior - search
    t_grid = t_first + np.arange(n_id + 2 * lags) / rate
    beams = state_at(s, t_grid)
    beams[:, ~ports] = np.nan

    grid = np.full((n_id, N_PORTS), np.nan, np.float32)
    grid[frame_ids] = level[:len(frame_ids)]
    coarse, used = _correlate(grid, beams, 2 * lags + 1)
    if used == 0 or not np.isfinite(coarse).any():
        return {"ok": False, "reason": "no port could be scored"}
    k = int(np.nanargmax(coarse))
    away = np.isfinite(coarse) & (np.abs(np.arange(len(coarse)) - k) > int(rate * 2))
    noise = float(np.nanstd(coarse[away])) if away.sum() > 10 else np.nan
    peak_z = float((coarse[k] - np.nanmedian(coarse[away])) / noise) if noise > 0 else np.nan
    second = float(np.nanmax(coarse[away])) if away.any() else np.nan
    out = {"ok": True, "t0_coarse": float(t_first + k / rate), "peak_z": peak_z,
           "peak_over_second": float(coarse[k] / second) if second > 0 else np.nan,
           "ports_used": int(used), "breaks": breaks.tolist(), "score": coarse,
           "t_first": t_first, "curves": {}}
    w = int(round(FINE_S * rate))
    for name, (edge, states) in fines.items():
        egrid = np.full((n_id, N_PORTS), np.nan, np.float32)
        egrid[frame_ids] = edge[:len(frame_ids)]
        d_video = np.diff(egrid, axis=0, prepend=np.nan)
        st = state_at(states, t_grid)
        onsets = np.diff(states["broken"].astype(np.int8), axis=0).clip(0).sum(axis=0)
        st[:, onsets < MIN_BREAKS] = np.nan
        d_state = np.diff(st, axis=0, prepend=np.nan)
        # Only the window around the coarse peak is wanted: correlate just those lags.
        lo, hi = max(k - w, 0), min(k + w + 1, 2 * lags + 1)
        curve, fused = _correlate(d_video, d_state[lo:hi + n_id - 1], hi - lo)
        out[f"{name}_ports"] = int(fused)
        if not fused or not np.isfinite(curve).any():
            out[f"{name}_t0"], out[f"{name}_z"] = np.nan, np.nan
            continue
        j = int(np.nanargmax(curve))
        sub = 0.0
        if 0 < j < len(curve) - 1 and np.isfinite(curve[j - 1:j + 2]).all():
            a_, b_, c_ = curve[j - 1], curve[j], curve[j + 1]
            denom = a_ - 2 * b_ + c_
            sub = float(np.clip(0.5 * (a_ - c_) / denom, -0.5, 0.5)) if denom != 0 else 0.0
        rest = np.delete(curve, range(max(j - 3, 0), min(j + 4, len(curve))))
        out[f"{name}_t0"] = float(t_first + (lo + j + sub) / rate)
        if np.isfinite(rest).sum() > 5:
            out[f"{name}_z"] = float((curve[j] - np.nanmedian(rest)) / np.nanstd(rest))
        else:
            out[f"{name}_z"] = np.nan
        out["curves"][name] = (lo, curve)
    return out


def _xcorr(a, b, n_lags):
    """sum_i a[i + lag] * b[i] for lag = 0..n_lags-1 (a longer than b by n_lags - 1)."""
    from scipy.signal import fftconvolve
    full = fftconvolve(a, b[::-1], mode="valid")
    return full[:n_lags]


# =====================================================================
# 5. Sessions: what is known about each
# =====================================================================

def session_paths(cohort_base, session_id):
    """Where one session's files are: its folder, DAQ file, tracker, frame times, movie."""
    folder = next(cohort_base.glob(f"26*/{session_id}"))
    return {"folder": folder, "daq": folder / f"{session_id}-ArduinoDAQ.h5",
            "tracker": folder / f"{session_id}_Tracker_data.json",
            "frame_times": folder / "processed" / "frame_times.csv",
            "movie": cohort_base / WORK / "brightness" / f"{session_id}.npy"}


def wall_clock_prior(daq_path, tracker_path) -> float | None:
    """
    Frame 0's DAQ time from the two programs' wall clocks: the tracker's start (to the
    second) against the DAQ's start (its end time minus its length). Good to a few
    seconds — enough to centre the search, never used as the answer.
    """
    import h5py
    from datetime import datetime, timedelta
    with h5py.File(daq_path, "r") as handle:
        end = handle.attrs.get("time")
        taken = handle.attrs.get("time_taken")
    tracker = json.loads(Path(tracker_path).read_text())
    start = tracker.get("start_time")
    if end is None or taken is None or start is None:
        return None
    end = datetime.fromisoformat(end.decode() if isinstance(end, bytes) else str(end))
    daq_start = end - timedelta(seconds=float(taken))
    cam_start = datetime.strptime(str(start), "%y%m%d_%H%M%S")
    return (cam_start - daq_start).total_seconds()


def true_t0(frame_times_path) -> tuple[float, float] | None:
    """For a session with pulses: (t0, frames per second) from its placed frames."""
    import pandas as pd
    f = pd.read_csv(frame_times_path).dropna()
    if len(f) < 1000:
        return None
    slope, intercept = np.polyfit(f.frame_id.to_numpy(), f.t.to_numpy(), 1)
    return float(intercept), float(1 / slope)


# =====================================================================
# 6. Cohort steps
# =====================================================================

#: The maps learned per session, and the matched filter each becomes.
#: (file suffix, which DAQ channels it is learned against, brightening?)
KINDS = {"level": ("", "beam", False),
         "beam": ("_bedge", "beam", False),
         "led": ("_ledge", "led", True)}
FINES = ("beam", "led")


def _load_movie(path):
    return np.load(path, mmap_mode="r")


def _frame_ids(tracker):
    return np.asarray(json.loads(Path(tracker).read_text())["frame_IDs"], dtype=np.int64)


def learn_one(args):
    """Maps, true t0 and rate for one session with pulses. ``args`` is ``(base, session ID)``."""
    base, sid = args
    p = session_paths(Path(base), sid)
    movie = _load_movie(p["movie"])
    import pandas as pd
    frame_t = pd.read_csv(p["frame_times"])["t"].to_numpy()
    n = min(len(movie), len(frame_t))
    beams, lights = sensors(p["daq"]), leds(p["daq"])
    maps = {"level": session_map(movie[:n], frame_t[:n], beams),
            "beam": session_edge_map(movie[:n], frame_t[:n], beams),
            "led": session_edge_map(movie[:n], frame_t[:n], lights)}
    truth = true_t0(p["frame_times"])
    out = {"session": sid, "ok": maps["level"] is not None and truth is not None}
    if out["ok"]:
        for kind, m in maps.items():
            if m is not None:
                np.save(Path(base) / WORK / "maps" / f"{sid}{KINDS[kind][0]}.npy", m)
        out.update(t0_true=truth[0], fps=truth[1],
                   mean_frame=np.asarray(movie[:n:500]).mean(axis=0))
    return out


def signals_for(movie, weights):
    """The level signal and each fine stage's edge signal for one movie."""
    out = {"level": port_signals(movie, weights["level"])}
    for kind in FINES:
        out[kind] = port_signals(movie, weights[kind], detrend=False,
                                 darkness=not KINDS[kind][2])
    return out


def fit_one(args):
    """
    The brightness fit for one session, from maps learned on *other* sessions.
    ``args`` is ``(base, session ID, weights, rate, prior)``.

    The search is centred on ``prior`` — the usual t0 of sessions with pulses (the
    camera starts ~1.2 s after the DAQ). The wall clocks were meant to give it per
    session, but on these rigs they are tens of seconds out; that figure is kept only
    as a diagnostic.
    """
    base, sid, weights, rate, prior = args
    p = session_paths(Path(base), sid)
    movie = _load_movie(p["movie"])
    ids = _frame_ids(p["tracker"])
    n = min(len(movie), len(ids))
    sig = signals_for(movie[:n], weights)
    beams, lights = sensors(p["daq"]), leds(p["daq"])
    fit = fit_t0(sig["level"], {"beam": (sig["beam"], beams), "led": (sig["led"], lights)},
                 ids[:n], rate, beams, prior)
    fit.update(session=sid, prior=prior, wall_clock=wall_clock_prior(p["daq"], p["tracker"]),
               n_frames=n, n_uncertain_spans=len(beams["uncertain"]))
    return fit


@functools.lru_cache(maxsize=None)
def _rig(base, sid):
    folder = session_paths(Path(base), sid)["folder"]
    meta = folder / f"{sid}-metadata.json"
    if meta.exists():
        number = (json.loads(meta.read_text()).get("rig") or {}).get("number")
        if number is not None:
            return int(number)
    for log in folder.glob("daq_rig*.log"):
        return int(log.stem.removeprefix("daq_rig"))
    raise ValueError(f"{sid}: no rig in its metadata and no daq_rig<N>.log beside it")


def rig_of(base, sid):
    """
    Which rig a session ran on (``base`` the cohort folder, a Path or a string), from
    its own files: the metadata's ``rig.number``, else the DAQ log the rig wrote beside
    it (``daq_rig1.log``). Never from the mouse, since mice move between rigs.
    """
    return _rig(str(base), sid)


def learn(base, sessions, workers=8):
    """:func:`learn_one` on each of a list of session IDs, in parallel; by session ID."""
    from concurrent.futures import ProcessPoolExecutor
    (base / WORK / "maps").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        return {r["session"]: r for r in pool.map(learn_one, [(str(base), s) for s in sessions])}


def rig_weights(sessions, base, rig, kind, leave_out=None):
    """A rig's matched filter of one kind, from its sessions' maps."""
    import warnings
    suffix, _, brighten = KINDS[kind]
    paths = [base / WORK / "maps" / f"{sid}{suffix}.npy" for sid in sessions
             if rig_of(base, sid) == rig and sid != leave_out]
    maps = [np.load(p) for p in paths if p.exists()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # a port no session saw
        return weights_from_maps(np.nanmean(np.stack(maps), axis=0), brighten=brighten)


def all_weights(sessions, base, rig, leave_out=None):
    """A rig's matched filter of every kind, by kind."""
    return {kind: rig_weights(sessions, base, rig, kind, leave_out) for kind in KINDS}


#: The estimates validation compares, and the column each t0 is in.
METHODS = {"coarse": "t0_coarse", "beam": "beam_t0", "led": "led_t0"}


def validate(base, sessions, workers=8, log=print):
    """
    The brightness fit run blind on sessions with pulses: maps from the rig's *other*
    sessions, the search centred on the cohort's usual t0, each estimate compared with
    the t0 the pulses give. Per rig, each method's latency (its median error) is
    measured leave-one-out and taken off; the method with the smallest spread after
    that is the one the rig uses. Writes ``validation_brightness.csv``, the model, and
    pictures.
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    report = base / REPORT / "video_clock"
    report.mkdir(parents=True, exist_ok=True)
    learned = learn(base, sessions, workers)
    good = {s: r for s, r in learned.items() if r["ok"]}
    ids = list(good)
    rates = {}
    for rig in (1, 2):
        if any(rig_of(base, s) == rig for s in good):
            rates[rig] = float(np.median([r["fps"] for s, r in good.items() if rig_of(base, s) == rig]))
    prior = float(np.median([r["t0_true"] for r in good.values()]))
    log(f"learned {len(good)} session(s); rates {rates}; usual t0 {prior:.3f} s")
    _draw_hotspots(base, good, report / "hotspots.png")
    jobs = [(str(base), s, all_weights(ids, base, rig_of(base, s), leave_out=s), rates[rig_of(base, s)],
             prior) for s in good]
    with ProcessPoolExecutor(workers) as pool:
        fits = {f["session"]: f for f in pool.map(fit_one, jobs)}

    rows = []
    for sid, f in fits.items():
        row = {k: v for k, v in f.items() if k not in ("score", "curves", "t_first", "breaks")}
        row.update(t0_true=good[sid]["t0_true"], rig=rig_of(base, sid))
        for m, col in METHODS.items():
            row[f"{m}_raw_ms"] = (f[col] - row["t0_true"]) * 1000 if f.get("ok") else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    trusted = table.ok.astype(bool) & (table.peak_z >= TRUST_PEAK_Z)
    for m in METHODS:
        lat = []
        for r in table.itertuples():
            others = trusted & (table.rig == r.rig) & (table.session != r.session)
            vals = table.loc[others, f"{m}_raw_ms"].dropna()
            lat.append(float(np.median(vals)) if len(vals) >= 3 else np.nan)
        table[f"{m}_latency_ms"] = lat
        table[f"{m}_error_ms"] = table[f"{m}_raw_ms"] - table[f"{m}_latency_ms"]

    # Per rig: the method whose corrected errors spread least on trusted sessions.
    choice, latency, spread = {}, {}, {}
    for rig in sorted(int(r) for r in table.rig.unique()):
        t = table[trusted & (table.rig == rig)]
        mads = {m: float(np.nanmedian(np.abs(t[f"{m}_error_ms"]))) for m in METHODS
                if t[f"{m}_error_ms"].notna().sum() >= 3}
        best = min(mads, key=mads.get)
        choice[rig], spread[rig] = best, mads
        latency[rig] = {m: float(np.nanmedian(t[f"{m}_raw_ms"])) for m in METHODS}
        log(f"rig {rig}: median |error| after latency, by method {np.round(list(mads.values()), 1)} "
            f"({', '.join(mads)}) -> {best}")
    table["method"] = table.rig.map(choice)
    table["error_ms"] = [r[f"{r['method']}_error_ms"] for _, r in table.iterrows()]
    for sid, f in fits.items():
        if f.get("ok"):
            row = table[table.session == sid].iloc[0]
            _draw_fit(f, report / "validation" / f"{sid}.png", truth=row.t0_true,
                      latency={m: row[f"{m}_latency_ms"] for m in METHODS},
                      chosen=row.method)
    table.to_csv(report / "validation_brightness.csv", index=False)
    _draw_validation(table, report / "validation_brightness.png")
    meta = {"rates": rates, "prior_t0": prior, "method": choice, "latency_ms": latency,
            "median_abs_error_ms": spread, "sessions": ids}
    (base / WORK / "video_clock_model.json").write_text(json.dumps(meta, indent=2))
    return table


def _draw_hotspots(base, good, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, rig in enumerate((1, 2)):
        ses = [s for s in good if rig_of(base, s) == rig]
        if not ses:
            continue
        mean = np.mean([good[s]["mean_frame"] for s in ses], axis=0)
        for ax, kind in zip(axes[row], KINDS):
            w = rig_weights(ses, base, rig, kind)
            ax.imshow(mean, cmap="gray")
            for p in range(N_PORTS):
                ys, xs = np.nonzero(w[p])
                ax.scatter(xs, ys, s=10, c=colours[p], label=f"port {p}")
            ax.set_title(f"red rig {rig}: {kind} hotspots ({len(ses)} sessions)", fontsize=9)
            ax.axis("off")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _draw_fit(f, path, truth=None, latency=None, chosen=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    latency = latency or {}
    per_lag = 2 * SEARCH_S / (len(f["score"]) - 1)
    t_axis = f["t_first"] + np.arange(len(f["score"])) * per_lag
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
    ax[0].plot(t_axis, f["score"], lw=0.6, color="#555")
    ax[0].axvline(f["t0_coarse"], color="#2a78d6", lw=1, label=f"coarse {f['t0_coarse']:.3f} s")
    if truth is not None:
        ax[0].axvline(truth, color="#1baf7a", lw=1, ls="--", label=f"pulses {truth:.3f} s")
    ax[0].set(title=f"{f['session']}: coarse — mouse at port vs beam state "
                    f"(peak z {f['peak_z']:.1f})", xlabel="t0 (s, DAQ clock)",
              ylabel="mean correlation")
    ax[0].legend(fontsize=7)
    ref = truth if truth is not None else f["t0_coarse"]
    colours = {"beam": "#eb6834", "led": "#2a78d6"}
    for name, (lo, curve) in f["curves"].items():
        t = f["t_first"] + (lo + np.arange(len(curve))) * per_lag
        lat = latency.get(name, 0.0) or 0.0
        lat = 0.0 if not np.isfinite(lat) else lat
        y = (curve - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-12)
        est = f[f"{name}_t0"] - lat / 1000
        label = f"{name} (z {f[f'{name}_z']:.1f}): {(est - ref) * 1000:+.0f} ms"
        if name == chosen:
            label += "  ← used"
        ax[1].plot((t - lat / 1000 - ref) * 1000, y, ".-", ms=2, lw=0.8, color=colours[name],
                   label=label)
    ax[1].axvline(0, color="#1baf7a" if truth is not None else "#999", ls="--", lw=1)
    reference = "pulses" if truth is not None else "coarse t0"
    ax[1].set(xlim=(-600, 600), xlabel=f"ms from {reference} (latency taken off)",
              title="fine stages, scaled")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _draw_validation(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[(table.ok == True) & (table.peak_z >= TRUST_PEAK_Z)]       # noqa: E712
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    bins = np.arange(-10.5, 11, 1)
    for ax, rig in zip(axes[:2], (1, 2)):
        s = t[t.rig == rig]
        for m, c in (("coarse", "#bbb"), ("beam", "#eb6834"), ("led", "#2a78d6")):
            e = np.clip(s[f"{m}_error_ms"] / (1000 / 30), -10, 10)
            ax.hist(e, bins=bins, histtype="step", lw=1.5, color=c,
                    label=f"{m}: ≤1 frame {np.mean(np.abs(e) <= 1):.0%}")
        ax.set(title=f"red rig {rig}: error after latency, {len(s)} sessions "
                     f"(method used: {s.method.iloc[0] if len(s) else '-'})",
               xlabel="frames (clipped ±10)")
        ax.legend(fontsize=7)
    a = table[table.ok == True]                                           # noqa: E712
    axes[2].scatter(a.peak_z, np.abs(a.error_ms), s=12,
                    c=np.where(a.rig == 1, "#2a78d6", "#eb6834"))
    axes[2].axhline(33.3, color="k", lw=0.5, ls="--")
    axes[2].axvline(TRUST_PEAK_Z, color="k", lw=0.5, ls=":")
    axes[2].set(yscale="symlog", xscale="log", xlabel="coarse peak z",
                ylabel="|error| ms (method used)", title="error against peak (blue rig 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# 7. Fixing sessions without pulses
# =====================================================================

#: Accepted only above these — set from validation (validation_brightness.png).
ACCEPT_PEAK_Z = 8.0
ACCEPT_FINE_Z = 5.0
ACCEPT_PORTS = 3
#: Converter worker counts tried for a video the old converter chunked.
CORES_TRIED = tuple(range(16, 65))
UNPLACED = "frame_times_unplaced.csv"
BY = "hba2/postprocessing/repairs/repair_video_clock.py"


def movie_frame_ids(ids, n_movie, cores):
    """
    The frame ID each decoded video frame shows: in order (``cores`` None), or as the
    converter chunked it on that many cores.
    """
    if cores is None:
        return ids[:n_movie]
    from hba2.postprocessing.repairs.repair_video_timing import replay_chunks
    positions = replay_chunks(ids, cores)[:n_movie]
    return ids[np.clip(positions, 0, len(ids) - 1)]


def fit_unplaced(args):
    """
    Brightness fit for a session without pulses, by the rig's validated method and
    latency. For a chunked video every worker count is tried and the one whose coarse
    peak stands highest kept. ``args`` is ``(base, session ID, weights, rate, prior,
    method, latency_ms)``.
    """
    base, sid, weights, rate, prior, method, latency_ms = args
    p = session_paths(Path(base), sid)
    movie = _load_movie(p["movie"])
    ids = _frame_ids(p["tracker"])
    beams, lights = sensors(p["daq"]), leds(p["daq"])
    sig = signals_for(movie, weights)
    tries = [None] if len(movie) == len(ids) else list(CORES_TRIED)
    best = None
    for cores in tries:
        mids = movie_frame_ids(ids, len(movie), cores)
        order = np.argsort(mids, kind="stable")          # the fit wants IDs ascending
        _, first = np.unique(mids[order], return_index=True)
        keep = order[first]
        f = fit_t0(sig["level"][keep], {"beam": (sig["beam"][keep], beams),
                                        "led": (sig["led"][keep], lights)},
                   mids[keep], rate, beams, prior)
        if f["ok"] and (best is None or f["peak_z"] > best["peak_z"]):
            best = dict(f, cores=cores)
    if best is None:
        return {"session": sid, "ok": False, "reason": "no fit"}
    col = METHODS[method]
    best.update(session=sid, rate=rate, method=method, latency_ms=latency_ms[method],
                n_movie=len(movie), n_tracker=len(ids), chunked=len(tries) > 1,
                n_uncertain_spans=len(beams["uncertain"]),
                t0=best[col] - latency_ms[method] / 1000,
                fine_z=best.get(f"{method}_z", np.nan) if method != "coarse" else np.nan)
    # The other estimates, latency-corrected, for agreement checks.
    for m, c in METHODS.items():
        best[f"{m}_t0_corrected"] = best[c] - latency_ms[m] / 1000 if np.isfinite(best[c]) else np.nan
    fine_ok = method == "coarse" or np.nan_to_num(best["fine_z"]) >= ACCEPT_FINE_Z
    best["accepted"] = bool(best["peak_z"] >= ACCEPT_PEAK_Z and fine_ok
                            and best["ports_used"] >= ACCEPT_PORTS)
    return best


def apply_times(base, fit):
    """``frame_times.csv`` timed by the fit; the untimed one kept beside it."""
    import pandas as pd
    import shutil
    from datetime import datetime
    p = session_paths(base, fit["session"])
    ft_path = p["frame_times"]
    keep = ft_path.parent / UNPLACED
    if not keep.exists():
        shutil.copy2(ft_path, keep)
    ft = pd.read_csv(keep)
    ft["t"] = fit["t0"] + ft["frame_id"].to_numpy(float) / fit["rate"]
    ft.to_csv(ft_path, index=False, float_format="%.6f")
    sj = ft_path.parent / "session.json"
    doc = json.loads(sj.read_text())
    doc["video_clock"] = {
        "made_by": BY, "made_at": datetime.now().isoformat(timespec="seconds"),
        "why": "no camera pulse reached the DAQ: frame times fitted, not counted",
        "rule": "t = t0 + frame_id / rate; rate the rig's, from sessions with pulses; t0 "
                "from port-hotspot brightness against the DAQ (coarse: mouse at port vs "
                "beam state; fine: the rig's validated method), minus that method's "
                "measured latency",
        **{k: (float(v) if isinstance(v, np.floating) else v) for k, v in fit.items()
           if k not in ("score", "curves", "t_first", "breaks", "occ_curve")},
    }
    sj.write_text(json.dumps(doc, indent=2, default=str))


def fit_cohort(base, sessions, workers=8, apply=False, method=None, log=print):
    """
    Fit (and with ``apply``, write) every given session from the validated model.
    ``method`` overrides the model's per-rig choice, e.g. ``{2: "led"}``.
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    model = json.loads((base / WORK / "video_clock_model.json").read_text())
    report = base / REPORT / "video_clock" / "fits"
    weights = {rig: all_weights(model["sessions"], base, rig) for rig in (1, 2)}
    chosen = {int(k): v for k, v in model["method"].items()}
    chosen.update(method or {})
    jobs = [(str(base), s, weights[rig_of(base, s)], model["rates"][str(rig_of(base, s))],
             model["prior_t0"], chosen[rig_of(base, s)], model["latency_ms"][str(rig_of(base, s))])
            for s in sessions]
    with ProcessPoolExecutor(workers) as pool:
        fits = list(pool.map(fit_unplaced, jobs))
    for f in fits:
        if f.get("ok"):
            _draw_fit(f, report / f"{f['session']}.png",
                      latency=model["latency_ms"][str(rig_of(base, f["session"]))],
                      chosen=f["method"])
            if apply and f["accepted"]:
                apply_times(base, f)
    rows = []
    for f in fits:
        rows.append({k: v for k, v in f.items()
                     if k not in ("score", "curves", "t_first", "breaks")})
    table = pd.DataFrame(rows)
    report.mkdir(parents=True, exist_ok=True)
    table.to_csv(report.parent / "fits_brightness.csv", index=False)
    n_ok = int(table["accepted"].fillna(False).sum()) if "accepted" in table else 0
    line = f"{len(table)} fitted, {n_ok} accepted"
    if apply:
        line += " and applied"
    log(line)
    return table


# =====================================================================
# 8. Occlusion: the nose goes into the port as the beam breaks
# =====================================================================
#
# Brightness gives the right second everywhere, but its frame-level latency depends on
# the lighting, and red rig 2 was relit on the day its camera signal stopped. A beam is
# broken by the snout entering the port, which hides the nose from the camera — so "the
# head is at port p and DeepLabCut cannot see the nose" is, frame by frame, the same
# on/off state as beam p, with the same edges, whatever the lighting.
#
# On red rig 2 the camera keeps seeing the nose through most pokes (hidden in ~30% of
# beam-broken frames, against 97% on rig 1), so there "the nose is where it sits at that
# port during a poke" counts as at the port too. Each rig's latency is its own, since
# the two rigs see the poke differently.
#
# Tried first, and kept only as pictures in the report (validation_pose/): the t0 at
# which nose positions at beam breaks cluster tightest. The nose rests at the port for
# about half a second around a break, so that minimum is flat, and wandered by ±60 ms.

DLC_MODEL = "no_implant_superanimal"
MIN_LIKELIHOOD = 0.6
#: Head within this many pixels (full frame) of a port's head position is at that port.
AT_PORT_PX = 60
#: Nose within this many pixels of a port's poking nose position is in that port.
NOSE_AT_PORT_PX = 25
#: Search half-width around the brightness t0, which is right to the second.
OCC_WINDOW_S = 1.5
#: Frames a port needs on each side (beam broken, nose hidden there) to be scored.
OCC_MIN_FRAMES = 20
#: The peak is compared with the best score this far from it.
OCC_AWAY_S = 0.5


def dlc_csv(record_folder, model=DLC_MODEL) -> Path | None:
    """The first DeepLabCut CSV of ``model`` in a session folder, or None."""
    csvs = sorted((record_folder / "dlc" / model).glob("*.csv"))
    return csvs[0] if csvs else None


def occlusion_signal(table, frame_ids, rate, beams, centre):
    """
    Per frame and port: 1 where the mouse is poking that port as the video shows it —
    the head at the port with the nose hidden, or the nose where it sits during that
    port's pokes.

    Each port's head and nose positions are their medians while that beam is broken,
    with the frames placed by ``centre`` — a few hundred ms out moves them little, as a
    visit lasts longer than that. ``table`` is the DeepLabCut table, one row per entry
    of ``frame_ids``.
    """
    import warnings

    def part(name):
        xy = table[name][["x", "y"]].to_numpy(float)
        xy[table[name]["likelihood"].to_numpy() < MIN_LIKELIHOOD] = np.nan
        return xy

    def nearest_port(xy, state):
        ports = np.array([np.nanmedian(xy[state[:, q] == 1], axis=0)
                          if (state[:, q] == 1).sum() > OCC_MIN_FRAMES else (np.nan, np.nan)
                          for q in range(N_PORTS)])
        d = np.hypot(xy[:, None, 0] - ports[None, :, 0], xy[:, None, 1] - ports[None, :, 1])
        return d, d == np.nanmin(d, axis=1, keepdims=True)

    head, nose = part("head"), part("nose")
    hidden = np.isnan(nose[:, 0])
    state = state_at(beams, centre + frame_ids / rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # a port never visited
        dh, head_nearest = nearest_port(head, state)
        dn, nose_nearest = nearest_port(nose, state)
    at = (dh < AT_PORT_PX) & head_nearest & hidden[:, None]
    in_port = (dn < NOSE_AT_PORT_PX) & nose_nearest
    return (at | in_port).astype(np.float32)


def occlusion_score(t0s, signal, frame_ids, rate, beams):
    """Per candidate t0: the mean over ports of corr(video state, beam state), and how
    many ports were scored."""
    score, ports = np.full(len(t0s), np.nan), np.zeros(len(t0s), int)
    for i, t0 in enumerate(t0s):
        state = state_at(beams, t0 + frame_ids / rate)
        ok = np.isfinite(state).all(axis=1)
        r = [np.corrcoef(signal[ok, q], state[ok, q])[0, 1] for q in range(N_PORTS)
             if state[ok, q].sum() >= OCC_MIN_FRAMES and signal[ok, q].sum() >= OCC_MIN_FRAMES]
        if r:
            score[i], ports[i] = np.mean(r), len(r)
    return score, ports


def fit_occlusion(signal, frame_ids, rate, beams, centre):
    """The t0 within ±``OCC_WINDOW_S`` of ``centre`` that best lines the two states up:
    a search in whole frames, then in tenths of one around the best."""
    step = 1 / rate
    t0s = centre + np.arange(-OCC_WINDOW_S, OCC_WINDOW_S + step / 2, step)
    score, ports = occlusion_score(t0s, signal, frame_ids, rate, beams)
    if not np.isfinite(score).any():
        return {"occ_ok": False, "occ_reason": "no port with enough visits"}
    k = int(np.nanargmax(score))
    fine_t0s = t0s[k] + np.arange(-1.5, 1.51, 0.1) * step
    fine, _ = occlusion_score(fine_t0s, signal, frame_ids, rate, beams)
    j = int(np.nanargmax(fine))
    away = np.abs(t0s - t0s[k]) >= OCC_AWAY_S
    edge = k < 3 or k > len(score) - 4                  # best at the window's edge
    return {"occ_ok": not edge, "occ_reason": "peak at window edge" if edge else "",
            "occ_t0": float(fine_t0s[j]), "occ_peak": float(fine[j]),
            "occ_margin": float(fine[j] - np.nanmax(score[away])),
            "occ_ports": int(ports[k]), "occ_curve": (t0s, score)}


def occlusion_one(args):
    """
    The occlusion fit for one session, centred on ``centre``; ``args`` is ``(base,
    session ID, rate, centre, cores)``. For a chunked video
    (fewer decoded frames than frame IDs) every worker count in ``cores`` is tried and
    the one whose peak stands highest kept; ``cores=None`` means the video is in order.
    """
    import pandas as pd
    base, sid, rate, centre, cores = args
    p = session_paths(Path(base), sid)
    path = dlc_csv(p["folder"])
    if path is None:
        return {"session": sid, "occ_ok": False, "occ_reason": "no DeepLabCut output"}
    table = pd.read_csv(path, header=[1, 2], index_col=0)
    ids = _frame_ids(p["tracker"])
    beams = sensors(p["daq"])
    n = min(len(table), len(ids))
    tries = [None] if len(table) >= len(ids) or not cores else list(cores)
    best = last = None
    for c in tries:
        mids = movie_frame_ids(ids, n, c)
        order = np.argsort(mids, kind="stable")          # each frame ID once, ascending
        _, first = np.unique(mids[order], return_index=True)
        keep = order[first]
        fids = mids[keep].astype(float)
        last = dict(fit_occlusion(occlusion_signal(table.iloc[keep], fids, rate, beams, centre),
                                  fids, rate, beams, centre), cores=c)
        if last["occ_ok"] and (best is None or last["occ_peak"] > best["occ_peak"]):
            best = last
    best = best or last
    best.update(session=sid, rig=rig_of(base, sid), rate=rate, centre=centre,
                n_dlc=len(table), n_tracker=len(ids), chunked=len(tries) > 1,
                head_seen=float((table["head"]["likelihood"] >= MIN_LIKELIHOOD).mean()))
    return best


def validate_occlusion(base, workers=8, log=print):
    """
    The occlusion fit on every session with pulses, DeepLabCut and an unchunked video,
    against the t0 its pulses give. The search is centred on that t0 but spans ±1.5 s —
    90 frames either way, more than brightness is ever out — and the peak is the
    curve's own. Per rig, the latency is measured leave-one-out. Writes
    ``validation_occlusion.csv``, pictures, and the latencies into the model.
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    report = base / REPORT / "video_clock"
    model_path = base / WORK / "video_clock_model.json"
    model = json.loads(model_path.read_text())
    qc = pd.read_csv(base / "cohort_qc.csv")
    jobs, truth = [], {}
    for sid in qc.session[qc.use_video_timing.astype(str) == "True"]:
        p = session_paths(base, sid)
        tt = true_t0(p["frame_times"]) if p["frame_times"].exists() else None
        if tt is None or dlc_csv(p["folder"]) is None:
            continue
        truth[sid] = tt[0]
        jobs.append((str(base), sid, model["rates"][str(rig_of(base, sid))], tt[0], None))
    with ProcessPoolExecutor(workers) as pool:
        fits = list(pool.map(occlusion_one, jobs))
    rows = []
    for f in fits:
        row = {k: v for k, v in f.items() if k != "occ_curve"}
        if f.get("occ_ok"):
            row["occ_raw_ms"] = (f["occ_t0"] - truth[f["session"]]) * 1000
            _draw_occlusion(f, report / "validation_occlusion" / f"{f['session']}.png",
                            truth=truth[f["session"]])
        rows.append(row)
    table = pd.DataFrame(rows)
    ok = table.occ_ok.astype(bool)
    # Each session's latency from the other good sessions on its rig.
    latencies = []
    for r in table.itertuples():
        others = ok & (table.rig == r.rig) & (table.session != r.session)
        if r.occ_ok and others.sum() >= 3:
            latencies.append(float(np.median(table.occ_raw_ms[others])))
        else:
            latencies.append(np.nan)
    table["occ_latency_ms"] = latencies
    table["occ_error_ms"] = table.occ_raw_ms - table.occ_latency_ms
    table.to_csv(report / "validation_occlusion.csv", index=False)
    lat = {str(int(rig)): float(np.median(table.occ_raw_ms[ok & (table.rig == rig)]))
           for rig in table.rig.unique() if (ok & (table.rig == rig)).sum() >= 3}
    model["occlusion_latency_ms"] = lat
    model_path.write_text(json.dumps(model, indent=2))
    _draw_occlusion_validation(table, report / "validation_occlusion.png")
    for rig in sorted(lat):
        t = table[ok & (table.rig == int(rig))]
        log(f"occlusion, rig {rig}: {len(t)} sessions, latency {lat[rig]:.1f} ms, "
            f"median |error| {np.nanmedian(np.abs(t.occ_error_ms)):.1f} ms, "
            f"within 1 frame {np.mean(np.abs(t.occ_error_ms) <= 1000 / 29.9989):.0%}")
    return table


#: A fit is applied only if it looks like the validated ones on red rig 1 (peak 0.68-0.88,
#: margin 0.11-0.27, all but two with 4+ ports) and lands where brightness put it.
ACCEPT_OCC_PEAK = 0.5
ACCEPT_OCC_MARGIN = 0.08
ACCEPT_OCC_PORTS = 3
#: Brightness is right to the second but its latency on a relit rig is not known; this
#: much disagreement is allowed.
ACCEPT_OCC_VS_BRIGHTNESS_S = 0.6
OCC_RULE = ("t = t0 + frame_id / rate; rate the rig's, from sessions with pulses; t0 where "
            "'poking the port' in the video (DeepLabCut: head there with the nose hidden, or "
            "nose where it pokes) best matches each beam's state, minus the rig's latency "
            "measured on its sessions with pulses; searched within 1.5 s "
            "of the brightness fit")


def fit_occlusion_cohort(base, workers=8, apply=False, log=print):
    """
    The occlusion fit on every session the brightness fit was run on
    (``fits_brightness.csv``), centred on its coarse t0, less the rig's own latency from
    :func:`validate_occlusion`. Red rig 2's comes from its dim-era sessions with pulses,
    where DeepLabCut sees the head less often than in the relit sessions being fitted —
    the least certain number here. With ``apply``, accepted fits are written as frame
    times.
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    report = base / REPORT / "video_clock"
    model = json.loads((base / WORK / "video_clock_model.json").read_text())
    bright = pd.read_csv(report / "fits_brightness.csv")
    bright = bright[bright.ok.astype(bool)]
    jobs = [(str(base), r.session, model["rates"][str(rig_of(base, r.session))], r.t0_coarse,
             CORES_TRIED if r.n_movie < r.n_tracker else None) for r in bright.itertuples()]
    with ProcessPoolExecutor(workers) as pool:
        fits = list(pool.map(occlusion_one, jobs))
    rows = []
    for f, job in zip(fits, jobs):
        if f.get("occ_ok"):
            latency = model["occlusion_latency_ms"][str(rig_of(base, f["session"]))] / 1000
            f["t0"] = f["occ_t0"] - latency
            f["latency_ms"] = latency * 1000
            f["vs_brightness_s"] = f["t0"] - job[3]
            f["accepted"] = bool(f["occ_peak"] >= ACCEPT_OCC_PEAK
                                 and f["occ_margin"] >= ACCEPT_OCC_MARGIN
                                 and f["occ_ports"] >= ACCEPT_OCC_PORTS
                                 and abs(f["vs_brightness_s"]) <= ACCEPT_OCC_VS_BRIGHTNESS_S)
            _draw_occlusion(f, report / "fits_occlusion" / f"{f['session']}.png")
            if apply and f["accepted"]:
                apply_times(base, dict(f, method="occlusion", rule=OCC_RULE))
        else:
            f["accepted"] = False
        rows.append({k: v for k, v in f.items() if k != "occ_curve"})
    table = pd.DataFrame(rows)
    table.to_csv(report / "fits_occlusion.csv", index=False)
    line = f"{len(table)} fitted, {int(table.accepted.sum())} accepted"
    if apply:
        line += " and applied"
    log(line)
    return table


def _draw_occlusion(f, path, truth=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0s, score = f["occ_curve"]
    ref = truth if truth is not None else f["occ_t0"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot((t0s - ref) * 1000, score, ".-", ms=2, color="#555")
    ax.axvline((f["occ_t0"] - ref) * 1000, color="#2a78d6", label="occlusion peak")
    if truth is not None:
        ax.axvline(0, color="#1baf7a", ls="--", label="pulses")
    title = (f"{f['session']}: peak {f['occ_peak']:.2f}, margin {f['occ_margin']:.2f}, "
             f"{f['occ_ports']} ports")
    if f.get("cores"):
        title += f", cores {f['cores']}"
    reference = "pulses" if truth is not None else "peak"
    ax.set(xlabel=f"ms from {reference}", ylabel="corr(nose hidden at port, beam)",
           title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _draw_occlusion_validation(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = table[table.occ_ok.astype(bool)]
    frame = 1000 / 29.9989
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, rig in zip(axes, (1, 2)):
        t = ok[ok.rig == rig].sort_values("session")
        ax.axhspan(-frame, frame, color="#1baf7a", alpha=.12, label="±1 frame")
        ax.plot(range(len(t)), t.occ_error_ms, "o", color="#2a78d6", ms=4)
        ax.axhline(0, color="#999", lw=.8)
        ax.set(title=f"Red Rig {rig}: {len(t)} sessions, "
                     f"{np.mean(np.abs(t.occ_error_ms) <= frame):.0%} within 1 frame",
               xlabel="session (by date)", ylabel="error after latency (ms)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="video clock repair")
    sub = parser.add_subparsers(dest="step", required=True)
    e = sub.add_parser("extract", help="one brightness movie")
    e.add_argument("video")
    e.add_argument("out")
    args = parser.parse_args()
    if args.step == "extract":
        print(extract(args.video, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
