"""
Exporting a curation: its good units as plain files, their spikes on the DAQ clock.

Reads only the frozen curation, the sort's fixed output and the session's
``ephys_sync.csv``; never Phy's live files. So an export is always reproducible from the
curation it belongs to, whatever has happened in Phy since.

Writes three files into the curation's folder:

``units.csv`` — one row per good unit:

    unit_id               Phy's cluster ID — the name Phy, the sorter and UnitMatch all use
    shank                 the probe shank, from the peak channel
    peak_channel          Phy's channel index carrying the largest waveform
    open_ephys_channel    which recorded channel that is: CH1..CH128 for an Open Ephys
                          recording, SpikeGLX's channel name for a Neuropixels one
    x_um, depth_um        the peak channel's position on the probe
    n_spikes              spikes with a DAQ time
    firing_rate_hz        n_spikes / (obs_stop - obs_start)
    amplitude_uv          peak-to-trough of the mean waveform on the peak channel
    snr, isi_violations_ratio, isi_violations_count, presence_ratio, amplitude_cutoff
                          SpikeInterface's metrics as Phy carried them. NaN for a
                          cluster made in Phy, which the sorter never measured
    quality               'good' — the only label exported
    keep                  1, always, here. A figure marks units 0 to leave them out
    obs_start, obs_stop   the stretch of DAQ time the probe was recording (s)
    n_spikes_unsynced     spikes from before or after the DAQ ran, so with no DAQ time

``spikes.parquet`` — one row per spike of a good unit, by time: ``unit_id``, ``sample``
(the recording's own sample number), ``t_ephys`` and ``t_daq`` (s; NaN where the DAQ was
not running). ``t_ephys`` is Open Ephys's ``timestamps.npy`` at that sample, or for
SpikeGLX the sample over the probe's calibrated rate — the clock ``ephys_sync.csv`` is on
either way.

``waveforms.npz`` — ``unit_id``, ``waveform`` (units x samples, µV, peak channel),
``n_waveforms`` (how many spikes each mean is over) and ``sample_rate``. Taken from the
spikes Phy sampled for display, the same ones the unit tracking matched on.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..cohort import CURATION
from ..postprocessing.utils.clock import EPHYS_REACH_S, ephys_to_daq

#: SpikeInterface's quality metrics, carried through under the names it gave them.
METRICS = ("snr", "isi_violations_ratio", "isi_violations_count", "presence_ratio",
           "amplitude_cutoff")

GOOD = "good"

#: ``units.csv``'s columns, in order — see the module docstring.
UNIT_COLUMNS = ("unit_id", "shank", "peak_channel", "open_ephys_channel", "x_um",
                "depth_um", "n_spikes", "firing_rate_hz", "amplitude_uv", *METRICS,
                "quality", "keep", "obs_start", "obs_stop", "n_spikes_unsynced")


def _dominant_template(unit, phy, clusters):
    """The template most of ``unit``'s spikes came from, by the sorter's assignment."""
    templates = np.load(phy / "spike_templates.npy", mmap_mode="r").ravel()
    values, counts = np.unique(templates[clusters == unit], return_counts=True)
    return int(values[np.argmax(counts)])


def export(record, sort, curation):
    """
    Write a curation's ``units.csv``, ``spikes.parquet`` and ``waveforms.npz``.

    ``record`` is the session's :class:`~hba2.cohort.SessionRecord`;
    ``sort`` and ``curation`` are names. Needs the session's ``ephys_sync.csv`` (run
    post-processing, or ``sync_ephys``). Overwrites a previous export of the same
    curation: it is derived, and redoing it gives the same answer unless the sync was
    redone. Returns a dict of what it wrote, which is also recorded in ``curation.json``.
    """
    files = record.curation(sort, curation)
    phy = Path(record.sorts[sort]["phy"])
    folder = Path(files["record"]).parent
    sync_csv = record.processed.get("ephys_sync")
    sync_json = record.processed.get("ephys_sync_record")
    if sync_csv is None or sync_json is None:
        raise FileNotFoundError(f"{record.session_id} has no ephys_sync.csv — run "
                                "post-processing or sync_ephys first")

    # ---- 1. Which spikes -------------------------------------------------
    labels = pd.read_csv(files["cluster_group"], sep="\t")
    labels["group"] = labels["group"].astype(str).str.strip()
    good = np.sort(labels.loc[labels["group"] == GOOD, "cluster_id"].to_numpy())
    clusters = np.load(files["spike_clusters"]).ravel()
    samples_all = np.load(phy / "spike_times.npy").ravel()
    if len(samples_all) != len(clusters):
        raise ValueError(f"spike_times has {len(samples_all)} spikes but the curation's "
                         f"spike_clusters has {len(clusters)} — not the same sort")
    chosen = np.isin(clusters, good)
    samples, units = samples_all[chosen], clusters[chosen].astype(np.int32)
    order = np.argsort(samples, kind="stable")
    samples, units = samples[order], units[order]

    # ---- 2. Onto the DAQ clock -------------------------------------------
    sync_record = json.loads(Path(sync_json).read_text())

    # Refuse to put a sort of one recording on another's clock. A sort with no
    # sort.json, or either side not saying which recording, is not checked.
    sort_json = record.sorts[sort].get("record")
    if sort_json is not None:
        pipeline = (json.loads(Path(sort_json).read_text()).get("pipeline") or {})
        sorted_index = pipeline.get("recording_index")
        synced_index = sync_record["recording"].get("index")
        if sorted_index is not None and synced_index is not None \
                and int(sorted_index) != int(synced_index):
            raise ValueError(f"the sort is of recording #{sorted_index} but the sync is "
                             f"of #{synced_index}")

    # A sample's ephys time: Open Ephys keeps a clock file, one time per sample; a
    # SpikeGLX probe file starts at 0 and runs at its calibrated rate.
    recorded = sync_record["recording"]
    spikeglx = recorded.get("system") == "spikeglx"
    if spikeglx:
        n_recorded = int(recorded["n_samples"])
    else:
        clock = np.load(Path(recorded["continuous"]).parent / "timestamps.npy", mmap_mode="r")
        n_recorded = len(clock)
    if len(samples) and samples[-1] >= n_recorded:
        raise ValueError(f"spike at sample {samples[-1]} but the recording has "
                         f"{n_recorded} samples — the sort is of a different recording")
    if spikeglx:
        rate = float(recorded["sample_rate"])
        t_ephys = samples / rate
        clock_first, clock_last = 0.0, (n_recorded - 1) / rate
    else:
        t_ephys = np.asarray(clock[samples], dtype=np.float64)
        clock_first, clock_last = float(clock[0]), float(clock[-1])
    sync = pd.read_csv(sync_csv)
    rows_e, rows_d = sync["t_ephys"].to_numpy(), sync["t_daq"].to_numpy()
    t_daq = ephys_to_daq(t_ephys, rows_e, rows_d)

    # The stretch of DAQ time the probe covered: the recording, cut to what the sync
    # reaches. The same for every unit — whether a unit fired throughout it is
    # presence_ratio's question, not this one's.
    first = max(clock_first, rows_e[0] - EPHYS_REACH_S)
    last = min(clock_last, rows_e[-1] + EPHYS_REACH_S)
    obs_start, obs_stop = (float(t) for t in ephys_to_daq(np.array([first, last]),
                                                          rows_e, rows_d))

    # ---- 3. Phy's per-cluster table, indexed by cluster ID ---------------
    # The curation's own copy when it has one — Phy rewrites it on save, so it covers
    # the clusters made by merging. Otherwise the sorter's per-metric files, which only
    # know the clusters the sorter made.
    if files.get("cluster_info") is not None:
        info = pd.read_csv(files["cluster_info"], sep="\t").set_index("cluster_id")
    else:
        info = pd.DataFrame()
        for metric in METRICS:
            path = phy / f"cluster_{metric}.tsv"
            if path.exists():
                column = pd.read_csv(path, sep="\t").set_index("cluster_id")
                info = column if info.empty else info.join(column, how="outer")

    # ---- 4. The probe ----------------------------------------------------
    positions = np.load(phy / "channel_positions.npy")
    groups = np.load(phy / "channel_groups.npy")
    if (phy / "channel_map_si.npy").exists():
        names = np.load(phy / "channel_map_si.npy")
    else:
        names = np.array([f"CH{c + 1}" for c in np.load(phy / "channel_map.npy")])
    sample_rate = None
    for line in (phy / "params.py").read_text().splitlines():
        if line.strip().startswith("sample_rate"):
            sample_rate = float(line.split("=", 1)[1])
            break
    if sample_rate is None:
        raise ValueError(f"no sample_rate in {phy / 'params.py'}")

    # ---- 5. Each unit's peak channel -------------------------------------
    # The channel with the largest waveform, by Phy's index: Phy's own ``ch`` when it
    # has one for this cluster; otherwise the largest channel of the template most of
    # the cluster's spikes came from.
    #: unit -> its peak channel
    peaks = {}
    for unit in good:
        if "ch" in info.columns and unit in info.index and pd.notna(info.at[unit, "ch"]):
            peaks[unit] = int(info.at[unit, "ch"])
            continue
        template = _dominant_template(unit, phy, clusters)
        shape = np.load(phy / "templates.npy", mmap_mode="r")[template]
        channels = np.load(phy / "template_ind.npy", mmap_mode="r")[template]
        peaks[unit] = int(channels[int(np.argmax(np.ptp(shape, axis=0)))])

    # ---- 6. Each unit's mean waveform on its peak channel ----------------
    # From the spikes Phy sampled. A unit with none of its spikes sampled falls back to
    # its dominant template on that channel, and says so by a count of zero.
    subset = np.load(phy / "_phy_spikes_subset.spikes.npy")
    channels = np.load(phy / "_phy_spikes_subset.channels.npy", mmap_mode="r")
    stored = np.load(phy / "_phy_spikes_subset.waveforms.npy", mmap_mode="r")
    owner = clusters[subset]

    n_samples = stored.shape[1]
    waves = np.full((len(good), n_samples), np.nan)
    n_waves = np.zeros(len(good), dtype=np.int32)
    for row, unit in enumerate(good):
        picks = np.flatnonzero(owner == unit)
        if len(picks):
            on_peak = np.asarray(channels[picks]) == peaks[unit]
            has = on_peak.any(axis=1)
            if has.any():
                column = on_peak.argmax(axis=1)[has]
                data = np.asarray(stored[picks[has]])
                waves[row] = data[np.arange(len(column)), :, column].mean(axis=0)
                n_waves[row] = int(has.sum())
                continue
        template = _dominant_template(unit, phy, clusters)
        chans = np.load(phy / "template_ind.npy", mmap_mode="r")[template]
        if peaks[unit] in chans:
            shape = np.load(phy / "templates.npy", mmap_mode="r")[template]
            waves[row] = shape[:, int(np.flatnonzero(chans == peaks[unit])[0])]

    # ---- 7. A row per unit -----------------------------------------------
    synced = ~np.isnan(t_daq)
    rows = []
    for row, unit in enumerate(good):
        mine = units == unit
        n = int((mine & synced).sum())
        peak = peaks[unit]

        if obs_stop > obs_start:
            firing_rate = n / (obs_stop - obs_start)
        else:
            firing_rate = np.nan
        if n_waves[row]:
            amplitude = float(np.ptp(waves[row]))
        else:
            amplitude = np.nan

        # SpikeInterface's metrics as Phy carried them; NaN where Phy has no value,
        # as for a cluster made in Phy, which the sorter never measured.
        metrics = {}
        for metric in METRICS:
            metrics[metric] = np.nan
            if metric in info.columns and unit in info.index:
                value = pd.to_numeric(info.at[unit, metric], errors="coerce")
                if pd.notna(value):
                    metrics[metric] = float(value)

        rows.append({
            "unit_id": int(unit),
            "shank": int(groups[peak]),
            "peak_channel": int(peak),
            "open_ephys_channel": str(names[peak]),
            "x_um": float(positions[peak, 0]),
            "depth_um": float(positions[peak, 1]),
            "n_spikes": n,
            "firing_rate_hz": firing_rate,
            "amplitude_uv": amplitude,
            **metrics,
            "quality": GOOD,
            "keep": 1,
            "obs_start": obs_start,
            "obs_stop": obs_stop,
            "n_spikes_unsynced": int((mine & ~synced).sum()),
        })

    # ---- 8. Write --------------------------------------------------------
    # units.csv is written with its header even when there are no good units, so a
    # curation that kept none still reads back as an empty table rather than an
    # unreadable file.
    table = pd.DataFrame(rows, columns=UNIT_COLUMNS)
    table.to_csv(folder / CURATION["units"], index=False)
    pd.DataFrame({"unit_id": units, "sample": samples, "t_ephys": t_ephys,
                  "t_daq": t_daq}).to_parquet(folder / CURATION["spikes"], index=False)
    np.savez(folder / CURATION["waveforms"], unit_id=good.astype(np.int32),
             waveform=waves.astype(np.float32), n_waveforms=n_waves,
             peak_channel=np.array([peaks[u] for u in good], dtype=np.int32),
             sample_rate=sample_rate)

    # ---- 9. What was exported, into curation.json ------------------------
    # Added beside what was frozen, which is left alone.
    residual = sync_record.get("mapping", {}).get("residual_ms_p99")
    sync_summary = {"csv": str(sync_csv), "rows": int(len(sync)), "residual_ms_p99": residual}
    summary = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "units": int(len(good)),
        "spikes": int(synced.sum()),
        "spikes_unsynced": int((~synced).sum()),
        "obs_daq_s": [obs_start, obs_stop],
        "sync": sync_summary,
    }
    curation_json = Path(files["record"])
    frozen = json.loads(curation_json.read_text())
    frozen["export"] = summary
    curation_json.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return summary
