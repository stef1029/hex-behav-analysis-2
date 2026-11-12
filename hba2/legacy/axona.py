"""
The 2025 tetrode recordings (Axona DACQ, MountainSort5) in this package's layout.
=================================================================================

The 2025 tetrode cohort recorded with an Axona DACQ system — eight tetrodes, 48 kHz,
one recording per group folder (``ephys.set``, ``ephys.bin``, ``ephys.inp``) — sorted
per tetrode with MountainSort5 through SpikeInterface, combined into one Phy folder and
curated there. Everything downstream of a curation reads the same files whatever made
the sort, so this writes those files:

    <session>/processed/ephys_sync.csv    the Axona clock against the DAQ's
    <session>/processed/ephys_sync.json   how that was found
    <session>/ephys_sorts/<sort>/
    ├── sort.json                         what produced it, and that it is tetrodes
    ├── phy -> <group>/spikesorter_output_ephys/combined/phy_combined
    ├── analyzer -> .../combined/sorting_analyzer_combined
    └── curations/<curation>/             frozen from Phy (``ephys.freeze``), then
        ├── units.csv  spikes.parquet       exported here
        └── waveforms.npz

The sort's folders are linked, not moved or copied: the group folder stays as it was.

The clock
---------
The DAQ Arduino pulsed the Axona's digital input (``.inp``, pin 0) once every
:data:`MESSAGES_PER_PULSE` messages, so pulse *j* is DAQ message ``50 j`` — joined by
counting, as the current rig's ephys sync is. Pulses are exact to the ``.inp``'s 1 ms
timebase; the DAQ's own timestamps jitter by tens of milliseconds, so the mapping takes
the median offset over each second, as :func:`~..postprocessing.utils.clock
.map_ephys_to_daq` does. A pulse train is used only when its count matches the DAQ's messages,
as recorded or with gaps filled: one missed pulse would shift every spike after it by
25 ms (see :func:`_counted`).

Waveforms
---------
The sort whitened its input, so Phy's templates are not in volts. Each good unit's mean
waveform is recomputed from the raw recording in µV — band-passed 300–6000 Hz, up to
:data:`WAVEFORM_SPIKES` spikes, on its peak channel.

    process_axona_cohort   every session: sync, adopt, freeze, export, name
    sync                   the clock (_counted, else _by_runs)
    adopt                  the sort, linked in place
    export                 one curation's units, spikes and waveforms
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..cohort import CURATION
from ..postprocessing.utils import clock
from ..ephys.export import METRICS, UNIT_COLUMNS, GOOD

SORT_NAME = "ms5_axona"
MESSAGES_PER_PULSE = 50
#: Pulses past the DAQ's last message that are still accepted as trailing extras.
TRAILING_EXTRA = 10
#: Pulses per window when smoothing that offset.
OFFSET_WINDOW = 400
#: For trains that will not count (:func:`_by_runs`): the pulse interval the Arduino
#: ran at, how far a spacing may be from it in a regular run (the ``.inp`` timebase is
#: 1 ms), how many pulses make a run, how far a run's pulses may sit from their
#: messages, and how long a stretch without a placed pulse is worth reporting.
NOMINAL_INTERVAL_S = 0.025
RUN_TOLERANCE_S = 0.0025
MIN_RUN = 40
RUN_RESIDUAL_S = 0.005
HOLE_REPORT_S = 5.0
WAVEFORM_SPIKES = 300
WAVEFORM_WINDOW = (-24, 48)          # samples at 48 kHz: -0.5 ms .. +1 ms

SYNC_SUFFIX = "_ephys_sync_timestamps.h5"

#: Which clusters an export takes. ``good``: those labelled good in ``cluster_group.tsv``
#: — hba2's rule everywhere. ``read_phy``: what the thesis used, SpikeInterface's
#: ``read_phy(exclude_cluster_groups=["noise", "mua"])`` — every row of
#: ``cluster_info.tsv`` whose ``group`` is not noise or mua, so a cluster left
#: ``unsorted`` (or blank) is kept too — there is one such cluster in that cohort, and
#: the thesis used it. Exported under its own curation name, never over
#: ``good``'s.
UNIT_RULES = ("good", "read_phy")


def group_paths(record):
    """
    Where a session's Axona recording and sort live: a dict of Paths, keys ``group``,
    ``set``, ``phy``, ``analyzer`` and ``sync_h5``.
    """
    group = record.directory.parent
    combined = group / "spikesorter_output_ephys" / "combined"
    return {"group": group, "set": group / "ephys.set",
            "phy": combined / "phy_combined",
            "analyzer": combined / "sorting_analyzer_combined",
            "sync_h5": record.directory / f"{record.session_id}{SYNC_SUFFIX}"}


# =====================================================================
# A cohort
# =====================================================================

def process_axona_cohort(base, *, curation, note="", rule="good", sessions=None,
                         index_neurons=True, log=print):
    """
    Sync, adopt, freeze, export and name every session with an Axona recording.

    Each step only where its output is missing, as ``ephys.process_ephys_cohort`` does.
    Run after the behaviour is converted: the sync needs the session's DAQ.

    ``base`` is the cohort folder, a Path or a string. ``rule`` is the export's unit
    rule (:data:`UNIT_RULES`); ``sessions`` limits the run to those session IDs;
    ``index_neurons=False`` skips naming units and rebuilding the cohort's
    ``neurons.csv`` (for a curation made only to reproduce an old figure).

    Returns ``{"synced": [...], "exported": [...], "failed": {session ID: error}}``.
    """
    from ..cohort import Cohort, scan_session
    from ..ephys import freeze, name_unnamed
    from ..cohort.neuron_index import build_neuron_index
    from ..postprocessing.sources import daq as daq_source
    from .convert import _repair_synthetic_clock, find_files

    cohort = Cohort(base)
    report = {"synced": [], "exported": [], "failed": {}}

    # ---- 1. Each session with a recording, a sort and sync pulses -----------
    for record in cohort:
        if sessions is not None and record.session_id not in sessions:
            continue
        paths = group_paths(record)
        if not (paths["set"].exists() and paths["phy"].exists()
                and paths["sync_h5"].exists() and record.is_processed):
            continue
        try:
            if record.processed.get("ephys_sync") is None:
                daq, _ = _repair_synthetic_clock(daq_source.read(record.path("daq")),
                                                 find_files(record.directory))
                sync(record, daq)
                report["synced"].append(record.session_id)
            sort_dir = adopt(record)
            if not (sort_dir / "curations" / curation).exists():
                freeze(sort_dir, curation, note=note)
            record = scan_session(record.directory)      # sees the new sort folder
            files = record.curation(SORT_NAME, curation)
            if files.get("units") is None:
                summary = export(record, curation, rule=rule)
                report["exported"].append(record.session_id)
                log(f"  {record.session_id}: {summary['units']} good unit(s), "
                    f"{summary['spikes']:,} spikes")
        except Exception as exc:                  # noqa: BLE001 — reported per session
            report["failed"][record.session_id] = f"{type(exc).__name__}: {exc}"
            log(f"  {record.session_id}: FAILED — {type(exc).__name__}: {exc}")

    # ---- 2. Names and the neuron index --------------------------------------
    if index_neurons:
        name_unnamed(Cohort(base), sort=SORT_NAME, curation=curation)
        build_neuron_index(Cohort(base), sort=SORT_NAME, curation=curation)
    log(f"{Path(base).name}: {len(report['exported'])} exported, "
        f"{len(report['failed'])} failed")
    return report


# =====================================================================
# The clock
# =====================================================================

def sync(record, daq):
    """
    Write ``ephys_sync.csv`` / ``.json`` for one session, from its DAQ. Returns the
    summary written to the JSON. Raises if it cannot.
    """
    # ---- 1. The pulses --------------------------------------------------------
    paths = group_paths(record)
    with h5py.File(paths["sync_h5"], "r") as handle:
        pulses = np.asarray(handle["pins/pin_0/high_timestamps"][:], dtype=float)
        meta = dict(handle["metadata"].attrs) if "metadata" in handle else {}
    if len(pulses) < 100:
        raise ValueError(f"only {len(pulses)} sync pulses")

    # ---- 2. Counted onto the DAQ's messages ---------------------------------
    ids = daq.message_ids.astype(np.int64)
    daq_times = np.asarray(daq.timestamps, dtype=float)
    expected = int(ids.max()) // MESSAGES_PER_PULSE + 1
    interval = float(np.median(np.diff(pulses)))
    runs = None
    try:
        pulses, how = _counted(pulses, interval, expected, ids, daq_times)
        numbers = np.arange(len(pulses))
    except ValueError as refused:
        # The whole train will not count onto the DAQ. Fall back to the runs of it that
        # are regular, each numbered by where the DAQ puts it (see _by_runs).
        numbers, pulses, runs = _by_runs(pulses, NOMINAL_INTERVAL_S, expected, ids,
                                         daq_times)
        how = f"by regular runs ({refused})"

    # ---- 3. The mapping -------------------------------------------------------
    # Every message's ephys time, on the line through the pulses either side of it —
    # only between the first and last pulse placed: past them there is nothing to
    # draw a line through, and a clamped value would be an invented time.
    pulse_ids = numbers * MESSAGES_PER_PULSE
    per_message = np.interp(np.arange(pulse_ids[-1] + 1), pulse_ids, pulses)
    inside = (ids >= pulse_ids[0]) & (ids <= pulse_ids[-1])
    mapping = clock.map_ephys_to_daq(per_message, ids[inside], daq_times[inside])

    # ---- 4. Write -------------------------------------------------------------
    out = record.directory / "processed"
    pd.DataFrame({"pulse": mapping.pulse, "t_ephys": mapping.t_ephys,
                  "t_daq": mapping.t_daq, "n_messages": mapping.n_messages}
                 ).to_csv(out / "ephys_sync.csv", index=False)
    summary = {
        "made_by": "hba2.legacy.axona",
        "made_at": datetime.now().isoformat(timespec="seconds"),
        "recording": {"system": "axona_dacq", "set": str(paths["set"]),
                      "sync_pulses": str(paths["sync_h5"]), "index": 0},
        "pulses": {"n": int(len(pulses)), "expected_from_daq": expected,
                   "counting": how,
                   "interval_ms": round(interval * 1000, 4),
                   "messages_per_pulse": MESSAGES_PER_PULSE,
                   "timebase": meta.get("timebase")},
    }
    if runs is not None:
        summary["runs"] = runs
    summary["mapping"] = {"rows": int(len(mapping.pulse)), "n_matched": mapping.n_matched,
                          "n_unmatched": mapping.n_unmatched,
                          "residual_ms_median": mapping.residual_ms_median,
                          "residual_ms_p99": mapping.residual_ms_p99,
                          "n_glitch_rows": mapping.n_glitch_rows}
    (out / "ephys_sync.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _counted(pulses, interval, expected, ids, daq_times):
    """
    The pulse train, as pulse *j* = message ``50 j``, and how it was read (a string).

    Tried as recorded, then with gaps filled — a gap of *k* intervals hiding *k - 1*
    missed pulses — keeping the first whose count matches the DAQ's: within one, or up to
    :data:`TRAILING_EXTRA` over, since the Arduino can pulse on briefly after the DAQ
    file stops. A pause in the pulsing that hid nothing is a gap too, which is why
    filling is tried second.

    The count is the test, not the offset between pulse and DAQ time. That offset steps
    wherever the DAQ's own timestamps stalled, and a stall moves behaviour and spikes
    together — both are placed by message — so it is reported, not refused.
    """
    # The train with gaps filled: k - 1 pulses spread evenly across a gap of k intervals.
    gaps = np.diff(pulses)
    missing = np.maximum(np.round(gaps / interval).astype(int) - 1, 0)
    if not missing.any():
        gaps_filled = pulses
    else:
        pieces = [pulses[:1]]
        for i, k in enumerate(missing):
            if k:
                pieces.append(pulses[i] + np.arange(1, k + 1) * gaps[i] / (k + 1))
            pieces.append(pulses[i + 1:i + 2])
        gaps_filled = np.concatenate(pieces)

    readings = [("as recorded", pulses), ("gaps filled", gaps_filled)]
    tried = []
    for how, train in readings:
        if expected - 1 <= len(train) <= expected + TRAILING_EXTRA:
            train = train[:expected]
            # The largest step between neighbouring windows' median pulse-DAQ offsets.
            message = np.arange(len(train)) * MESSAGES_PER_PULSE
            keep = message <= ids.max()
            offset = train[keep] - np.interp(message[keep], ids, daq_times)
            n = len(offset) // OFFSET_WINDOW
            step = 0.0
            if n >= 3:
                medians = np.median(offset[:n * OFFSET_WINDOW].reshape(n, OFFSET_WINDOW),
                                    axis=1)
                step = float(np.max(np.abs(np.diff(medians))))
            return train, f"{how}; largest step in pulse-DAQ offset {step * 1000:.0f} ms"
        tried.append(f"{how}: {len(train):,} pulses for {expected:,} expected")
    raise ValueError("sync pulses cannot be counted onto the DAQ — " + "; ".join(tried))


def _by_runs(pulses, interval, expected, ids, daq_times):
    """
    The pulses that can be placed, when the whole train will not count onto the DAQ.

    Four 2504 recordings fail :func:`_counted` (checked 2026-10-15): one where the Axona
    stopped ~3 min before the DAQ (fewer pulses than messages), one where the Arduino
    pulsed on for a second after the DAQ file closed (51 extra), and two where the
    ``.inp`` holds stretches of pulses ~3 ms apart followed by 1-2 s silences — pulses
    bunched or invented by the recorder, which no count survives. Counting any of them
    blindly would let the behaviour-to-spike alignment drift after the first such
    stretch.

    Here only **regular runs** are used: pulses whose spacing is within
    :data:`RUN_TOLERANCE_S` of the interval, at least :data:`MIN_RUN` of them in a row.
    Within a run pulses are counted, as everywhere else. Which pulse number a run
    starts at is the one thing taken from time: the first run is numbered by its place
    in the train (pulse 0 = message 0, as :func:`_counted` assumes); each later run by
    where the DAQ's timestamps put it, through the offset of the run before. Runs are
    at least a second long and the median is taken over the whole run, so the DAQ's
    timestamp stalls (up to ~1 s, lasting tens of seconds) do not decide it; the
    ~20 ppm drift between the clocks moves a 25 ms pulse by half a pulse only after
    ~10 min without a run. A run whose pulses do not sit on the DAQ's messages
    (median residual over :data:`RUN_RESIDUAL_S`) or that would number backwards is
    left out. Spikes between runs are placed on the line through the runs either side.

    Returns pulse numbers, their ephys times, and a summary for ``ephys_sync.json``.
    """
    pulses = np.asarray(pulses, dtype=float)
    ids = np.asarray(ids, dtype=np.int64)
    daq_times = np.maximum.accumulate(np.asarray(daq_times, dtype=float))
    # The DAQ time of every pulse number's message, and its inverse.
    numbers_all = np.arange(expected)
    t_of = np.interp(numbers_all * MESSAGES_PER_PULSE, ids, daq_times)

    # ---- 1. The regular runs --------------------------------------------------
    regular = np.abs(np.diff(pulses) - interval) <= RUN_TOLERANCE_S
    edges = np.flatnonzero(np.diff(np.concatenate(([0], regular.astype(np.int8), [0]))))
    runs = [(a, b + 1) for a, b in zip(edges[::2], edges[1::2]) if b + 1 - a >= MIN_RUN]
    if not runs:
        raise ValueError("no regular run of sync pulses to place")

    # ---- 2. Each numbered, and checked against the DAQ ----------------------
    kept_numbers, kept_times, placed, dropped = [], [], [], []
    offset, last = None, -1
    for a, b in runs:
        train = pulses[a:b]
        within = np.arange(b - a)
        if offset is None:
            start = a
        else:
            guess = np.interp(train - offset, t_of, numbers_all) - within
            start = int(np.round(np.median(guess)))
        numbers = start + within
        ok = (numbers >= 0) & (numbers < expected)
        if ok.sum() < MIN_RUN or numbers[0] <= last:
            dropped.append([float(train[0]), float(train[-1]), "numbers backwards or "
                            "past the DAQ"])
            continue
        residual = train[ok] - t_of[numbers[ok]]
        # Against the run's own offset, window by window: the clocks drift apart by
        # tens of ms an hour, which is not a misplaced run.
        block = np.arange(len(residual)) // OFFSET_WINDOW
        local = np.array([np.median(residual[block == k]) for k in range(block[-1] + 1)])
        spread = float(np.median(np.abs(residual - local[block])))
        if spread > RUN_RESIDUAL_S:
            dropped.append([float(train[0]), float(train[-1]),
                            f"median residual {spread * 1000:.1f} ms"])
            continue
        run_offset = float(np.median(residual[-OFFSET_WINDOW:]))
        offset, last = run_offset, int(numbers[ok][-1])
        kept_numbers.append(numbers[ok])
        kept_times.append(train[ok])
        placed.append([int(numbers[ok][0]), int(numbers[ok][-1]), round(run_offset, 4)])

    # ---- 3. What was placed, and the holes left -----------------------------
    numbers = np.concatenate(kept_numbers)
    times = np.concatenate(kept_times)
    holes = np.diff(numbers) - 1
    big = holes * interval > HOLE_REPORT_S
    reported_holes = [[round(float(numbers[i]) * interval, 1),
                       round(float(holes[i]) * interval, 1)] for i in np.flatnonzero(big)]
    summary = {
        "method": "regular runs, each numbered by the DAQ's timestamps "
                  "(legacy.axona._by_runs)",
        "runs_placed": len(placed), "runs_dropped": len(dropped),
        "pulses_placed": int(len(numbers)), "pulses_recorded": int(len(pulses)),
        "pulse_numbers_spanned": [int(numbers[0]), int(numbers[-1])],
        "holes_over_s": HOLE_REPORT_S,
        "holes": reported_holes,
        "holes_note": "[ephys-pulse time from the first pulse (s), length (s)]: no "
                      "placed pulse here; spikes in it are on the line through the "
                      "runs either side",
        "runs": placed[:200], "dropped": dropped[:200],
    }
    return numbers, times, summary


# =====================================================================
# The sort, in place
# =====================================================================

def adopt(record):
    """``ephys_sorts/ms5_axona/`` with links to the group's sort and a sort.json. Returns its Path."""
    paths = group_paths(record)
    sort_dir = record.directory / "ephys_sorts" / SORT_NAME
    sort_dir.mkdir(parents=True, exist_ok=True)
    for name, target in (("phy", paths["phy"]), ("analyzer", paths["analyzer"])):
        link = sort_dir / name
        if not link.exists() and not link.is_symlink() and target.exists():
            link.symlink_to(os.path.relpath(target, sort_dir))
    record_path = sort_dir / "sort.json"
    if not record_path.exists():
        record_path.write_text(json.dumps({
            "name": SORT_NAME,
            "adopted_from": str(paths["phy"].parent),
            "adopted_at": datetime.now().isoformat(timespec="seconds"),
            "linked_not_moved": True,
            "sorter": "mountainsort5, per tetrode, via SpikeInterface; combined for Phy",
            "probe": {"description": "Axona DACQ tetrode drive, 8 tetrodes; positions as "
                                     "tetrodes_8.json gives them",
                      "group": "tetrode"},
            "pipeline": {"recording_index": 0},
        }, indent=2))
    return sort_dir


# =====================================================================
# Export — what ephys.export does, for an Axona recording
# =====================================================================

def export(record, curation, *, rule="good"):
    """
    Write ``units.csv``, ``spikes.parquet`` and ``waveforms.npz`` for one curation.

    ``rule`` chooses the clusters (:data:`UNIT_RULES`). Under ``read_phy`` the
    ``quality`` column holds each cluster's own label and ``units.csv`` gains
    ``si_unit_id``: the unit ID ``read_phy`` gave it (the sorter's ``si_unit_id`` from
    ``cluster_info.tsv``, renumbered past the largest where missing or shared, exactly
    as SpikeInterface 0.102 does) — the ID the thesis printed, e.g. unit 1001.

    Returns the summary stored under ``export`` in the curation's record.
    """
    if rule not in UNIT_RULES:
        raise ValueError(f"rule {rule!r}: use one of {UNIT_RULES}")

    # ---- 1. Which clusters ----------------------------------------------------
    files = record.curation(SORT_NAME, curation)
    phy = Path(record.sorts[SORT_NAME]["phy"])
    folder = Path(files["record"]).parent
    sync_rows = pd.read_csv(record.directory / "processed" / "ephys_sync.csv")
    rows_e, rows_d = sync_rows["t_ephys"].to_numpy(), sync_rows["t_daq"].to_numpy()

    info = pd.read_csv(files["cluster_info"] or phy / "cluster_info.tsv",
                       sep="\t").set_index("cluster_id")
    if rule == "good":
        labels = pd.read_csv(files["cluster_group"], sep="\t")
        labels["group"] = labels["group"].astype(str).str.strip()
        good = np.sort(labels.loc[labels["group"] == GOOD, "cluster_id"].to_numpy())
        quality = {int(u): GOOD for u in good}
        si_ids = None
    else:
        good, quality, si_ids = _read_phy_units(info)

    # ---- 2. Their spikes, on the DAQ clock ----------------------------------
    clusters = np.load(files["spike_clusters"]).ravel()
    samples_all = np.load(phy / "spike_times.npy").ravel()
    if len(samples_all) != len(clusters):
        raise ValueError("spike_times and the curation's spike_clusters differ in length")
    rate = None
    for line in (phy / "params.py").read_text().splitlines():
        if line.strip().startswith("sample_rate"):
            rate = float(line.split("=", 1)[1])
            break
    if rate is None:
        raise ValueError("no sample_rate in params.py")

    chosen = np.isin(clusters, good)
    samples, units = samples_all[chosen], clusters[chosen].astype(np.int32)
    order = np.argsort(samples, kind="stable")
    samples, units = samples[order], units[order]
    t_ephys = samples / rate
    t_daq = clock.ephys_to_daq(t_ephys, rows_e, rows_d)

    # ---- 3. The span the recording observed, on the DAQ clock ---------------
    import spikeinterface.extractors as se
    n_samples = se.read_axona(str(group_paths(record)["set"])).get_num_samples()
    first = max(0.0, rows_e[0] - clock.EPHYS_REACH_S)
    last = min(n_samples / rate, rows_e[-1] + clock.EPHYS_REACH_S)
    obs_start, obs_stop = (float(t) for t in clock.ephys_to_daq(
        np.array([first, last]), rows_e, rows_d))

    # ---- 4. One row per unit ------------------------------------------------
    positions = np.load(phy / "channel_positions.npy")
    groups = np.load(phy / "channel_groups.npy")
    names = np.load(phy / "channel_map_si.npy")
    peaks = {int(u): int(info.at[u, "ch"]) for u in good}
    waves, n_waves = _waveforms(record, good, peaks, samples_all, clusters)

    synced = ~np.isnan(t_daq)
    rows = []
    for i, unit in enumerate(good):
        mine = units == unit
        n = int((mine & synced).sum())
        peak = peaks[int(unit)]
        row = {
            "unit_id": int(unit), "shank": int(groups[peak]), "peak_channel": peak,
            "open_ephys_channel": str(names[peak]),
            "x_um": float(positions[peak, 0]), "depth_um": float(positions[peak, 1]),
            "n_spikes": n,
            "firing_rate_hz": n / (obs_stop - obs_start) if obs_stop > obs_start else np.nan,
            "amplitude_uv": float(np.ptp(waves[i])) if n_waves[i] else np.nan,
        }
        for metric in METRICS:
            row[metric] = float(info.at[unit, metric]) if metric in info.columns else np.nan
        row.update({
            "quality": quality[int(unit)], "keep": 1, "obs_start": obs_start,
            "obs_stop": obs_stop, "n_spikes_unsynced": int((mine & ~synced).sum()),
        })
        if si_ids is not None:
            row["si_unit_id"] = si_ids[int(unit)]
        rows.append(row)

    # ---- 5. Write -------------------------------------------------------------
    columns = list(UNIT_COLUMNS)
    if si_ids is not None:
        columns.append("si_unit_id")
    pd.DataFrame(rows, columns=columns).to_csv(folder / CURATION["units"], index=False)
    pd.DataFrame({"unit_id": units, "sample": samples, "t_ephys": t_ephys,
                  "t_daq": t_daq}).to_parquet(folder / CURATION["spikes"], index=False)
    np.savez(folder / CURATION["waveforms"], unit_id=good.astype(np.int32),
             waveform=waves.astype(np.float32), n_waveforms=n_waves,
             peak_channel=np.array([peaks[int(u)] for u in good], dtype=np.int32),
             sample_rate=rate)

    summary = {"exported_at": datetime.now().isoformat(timespec="seconds"),
               "exported_by": "hba2.legacy.axona",
               "rule": rule,
               "units": int(len(good)), "spikes": int(synced.sum()),
               "spikes_unsynced": int((~synced).sum()),
               "obs_daq_s": [obs_start, obs_stop],
               "waveforms": f"raw recording, 300-6000 Hz, up to {WAVEFORM_SPIKES} "
                            f"spikes, peak channel, uV"}
    record_path = Path(files["record"])
    document = json.loads(record_path.read_text())
    document["export"] = summary
    record_path.write_text(json.dumps(document, indent=2, default=str))
    return summary


def _read_phy_units(info):
    """
    The clusters, labels and unit IDs ``spikeinterface.extractors.read_phy`` gives with
    ``exclude_cluster_groups=["noise", "mua"]`` — ported from SpikeInterface 0.102.4,
    ``PhySortingExtractor.__init__``: the rows of ``cluster_info.tsv`` whose ``group``
    is not ``noise`` or ``mua`` (a blank group passes), each named by ``si_unit_id``
    unless that is blank or shared among the kept rows, when it gets the next number
    past the largest (checked against ``read_phy`` on all 36 thesis sessions).

    ``info`` is ``cluster_info.tsv`` indexed by ``cluster_id``. Returns the sorted
    cluster IDs, ``{cluster: label}`` and ``{cluster: unit ID}``.
    """
    group = info["group"].astype(str).str.strip()
    kept = info[(group != "noise") & (group != "mua")]
    clusters = kept.index.to_numpy()
    si = kept["si_unit_id"].to_numpy(dtype=float) if "si_unit_id" in kept else None
    names = {}
    if si is None:
        names = {int(c): int(c) for c in clusters}
    else:
        largest = -1 if np.all(np.isnan(si)) else int(np.nanmax(si))
        # SpikeInterface writes each new ID back into the array it counts duplicates
        # in (``unit_ids`` is the column's own ``.values``), so of two rows sharing an
        # ID the first is renumbered and the second, then unique, keeps it. Same here.
        for i, cluster in enumerate(clusters):
            value = si[i]
            if np.isnan(value) or np.count_nonzero(si == value) != 1:
                largest += 1
                si[i] = largest
            names[int(cluster)] = int(si[i])
    labels = {int(c): str(g) for c, g in zip(clusters, kept["group"].astype(str).str.strip())}
    return np.sort(clusters), labels, names


def _waveforms(record, good, peaks, samples_all, clusters):
    """
    Mean µV waveform per unit on its peak channel, from the raw recording band-passed
    300-6000 Hz. Returns the waveforms (units x samples) and how many spikes each took.
    """
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre

    raw = se.read_axona(str(group_paths(record)["set"]))
    recording = spre.bandpass_filter(raw, freq_min=300, freq_max=6000)
    channel_ids = recording.get_channel_ids()
    n_total = recording.get_num_samples()
    before, after = -WAVEFORM_WINDOW[0], WAVEFORM_WINDOW[1]
    rng = np.random.default_rng(0)
    waves = np.full((len(good), before + after), np.nan)
    counts = np.zeros(len(good), dtype=np.int32)
    for row, unit in enumerate(good):
        mine = samples_all[clusters == unit]
        mine = mine[(mine >= before) & (mine < n_total - after)]
        if not len(mine):
            continue
        picks = np.sort(rng.choice(mine, size=min(WAVEFORM_SPIKES, len(mine)),
                                   replace=False))
        channel = [channel_ids[peaks[int(unit)]]]
        snippets = [recording.get_traces(start_frame=int(s) - before,
                                         end_frame=int(s) + after,
                                         channel_ids=channel, return_scaled=True)[:, 0]
                    for s in picks]
        waves[row] = np.mean(snippets, axis=0)
        counts[row] = len(snippets)
    return waves, counts
