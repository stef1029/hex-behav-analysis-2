"""
``ephys_sync.csv`` — the ephys clock against the DAQ's. Ephys sessions only.

One row a second: ``pulse`` (the DAQ message ID and sync pulse the row is anchored on),
``t_ephys``, ``t_daq``, and ``n_messages`` (how many DAQ messages its offset is the
median of). An ephys time between two rows maps to the straight line between them —
see :func:`~..utils.clock.map_ephys_to_daq` for why the rows are smoothed.

This is a **mapping, not a conversion**. Nothing behavioural moves onto the ephys
clock; spikes come onto the DAQ's when a curation is exported, and nothing else ever
needs to.

``ephys_sync.json`` beside it says where the pulses were read from and how cleanly the
two clocks agreed, because the CSV alone cannot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ._result import OutputResult

COLUMNS = ("pulse", "t_ephys", "t_daq", "n_messages")

#: A smoothed line further than this from single messages, at the 99th percentile,
#: means something other than timestamp jitter — a dropped pulse would shift every
#: later row by a millisecond per pulse.
RESIDUAL_WARN_MS = 2.0


def write(out_dir, recording, pulses, mapping, source="recording"):
    """
    Write the mapping and the record of how it was made.

    ``source`` is a short text saying where the pulse times were read from.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ephys_sync.csv"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(
            (int(p), f"{e:.7f}", f"{d:.7f}", int(n))
            for p, e, d, n in zip(mapping.pulse, mapping.t_ephys, mapping.t_daq,
                                  mapping.n_messages))

    warnings = []
    if recording.n_recordings > 1:
        warnings.append(f"ephys folder holds {recording.n_recordings} recordings; "
                        f"synced #{recording.index} only")
    if mapping.residual_ms_p99 > RESIDUAL_WARN_MS:
        warnings.append(f"single DAQ messages sit up to {mapping.residual_ms_p99:.1f} ms "
                        f"(p99) from the ephys sync line — check for dropped pulses")
    if mapping.n_glitch_rows:
        warnings.append(f"{mapping.n_glitch_rows} sync row(s) jumped away from their "
                        "neighbours and were left out (a DAQ timestamp glitch)")
    if mapping.n_unmatched:
        warnings.append(f"{mapping.n_unmatched} DAQ message(s) have no sync pulse "
                        f"(ephys stopped first, or started late)")

    first_t_ephys = None
    last_t_ephys = None
    if len(pulses.times):
        first_t_ephys = float(pulses.times[0])
        last_t_ephys = float(pulses.times[-1])

    record = {
        "recording": {
            "directory": str(recording.directory),
            "continuous": str(recording.continuous),
            "stream": recording.stream,
            "index": recording.index,
            "n_recordings": recording.n_recordings,
            "sample_rate": recording.sample_rate,
            "n_samples": recording.n_samples,
        },
        "pulses": {
            # Where the pulse times were read: the recording itself, or the previous
            # pipeline's file of the same pulses after it was checked against it.
            "source": source,
            "channel": pulses.channel,
            "threshold_raw": int(pulses.threshold),
            "n": int(len(pulses.times)),
            "first_t_ephys": first_t_ephys,
            "last_t_ephys": last_t_ephys,
        },
        "mapping": {
            "rows": int(len(mapping.pulse)),
            "daq_messages_matched": mapping.n_matched,
            "daq_messages_unmatched": mapping.n_unmatched,
            "residual_ms_median": round(mapping.residual_ms_median, 4),
            "residual_ms_p99": round(mapping.residual_ms_p99, 4),
            "glitch_rows_dropped": mapping.n_glitch_rows,
        },
        "warnings": warnings,
    }
    (out_dir / "ephys_sync.json").write_text(json.dumps(record, indent=2),
                                             encoding="utf-8")

    return OutputResult(name="ephys_sync", path=path, n_rows=len(mapping.pulse),
                        n_unmatched=mapping.n_unmatched, warnings=tuple(warnings))
