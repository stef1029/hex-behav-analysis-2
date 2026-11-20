# Plan: Neuropixels (SpikeGLX) recordings in the analysis

Status (2026-11-13): **everything but the sorting is built.** Spike sorting waits for a
real Neuropixels recording to tune on; until then the rest is built assuming its output is
a Phy folder like the CamNT sort's. Paths are relative to `hba2/`.

What was built, against a real 28 s SpikeGLX test session (Rig 4):

- `postprocessing/sources/spikeglx.py`: finds the recording (`is_spikeglx`, `locate`) and
  reads the sync. On the test session: 28 of 28 1 Hz edges paired, probe 4.4 ppm fast of
  the NI card, edges within 31 us of the line; DAQ messages 0.03 ms (median) and
  0.18 ms (p99) from the sync rows.
- `postprocessing/pipeline.py`: `sync_ephys` reads Open Ephys or SpikeGLX by the folder,
  and refuses a DAQ that does not pulse on every loop (the hat firmware, below).
- `postprocessing/outputs/ephys_sync.py`: the JSON records `recording.system` and, for
  SpikeGLX, the NI-to-probe `alignment`.
- `postprocessing/utils/clock.py`: `map_ephys_to_daq`'s residual is measured against
  the extended line exports use. A flat clamp past the end rows reported 176 ms p99 on
  a 22 s session, for messages that sit 0.2 ms from the line.
- `ephys/export.py`: SpikeGLX spike times are sample over the probe's rate. Checked end
  to end on a disposable copy of the test session with a fake Phy sort: freeze and export
  run unchanged, and `t_daq` follows the sync rows.
- `nwb/build_nwb.py`: unit column descriptions no longer assume 8 shanks of 16. The probe
  device description already comes from the sort's `sort.json`.
- `tests/test_spikeglx_sync.py`: a synthetic recording with known clocks.

What a Neuropixels sorter needs from this side: `spikeglx.locate(record.ephys)`, whose
`.continuous` is the `.ap.bin` (the folder for SpikeInterface's `read_spikeglx`,
`stream_id` its `.stream`). It should write its sort into `ephys_sorts/<name>/` as the
CamNT sort does, with `sort.json`'s `probe` (`description`, `group`) and
`pipeline.recording_index` set to the probe's index. Everything after that already runs.

The rest of this page is the plan as first written.

## What the rig now writes

The control side (`hex_behav_control/SpikeGLXLink`, the Neuropixels section of the
Peripherals panel) records Neuropixels through SpikeGLX on the PXIe chassis: a PXIe-6341
NI card with a BNC-2110 breakout, and an imec basestation in slot 3. Files go straight
into the session group, named after the session ID:

```
<group>/ephys/<session id>/
  <session id>_g0/
    <session id>_g0_t0.nidq.bin / .meta      NI card, 25 kHz: AI0-7, then the P0 digital word
    <session id>_g0_t0.imec0.ap.bin / .meta  probe, ~30 kHz, 384 channels + SY word
  spikeglx_session.json                      whose probe, which sync line, where the
                                             session starts and ends in the files
```

On the NI card, AI0 carries SpikeGLX's 1 Hz sync from the imec card. Rig N's DAQ sync
output goes into AI*N*; the rig's channel is `rig_sync_line` in the JSON. The probe
records the same 1 Hz sync internally as bit 6 of its SY word, the last channel.

Example session: a 28 s SpikeGLX test session (Rig 4, NP2014
serial 23107820512).

## What already works

- `cohort/cohort.py` matches `ephys/<anything>_<mouse>` by the mouse ending, so these
  folders are found as they are.
- Everything after `processed/ephys_sync.csv` is system-independent: export (spikes
  onto the DAQ clock), names, `build_nwb --batch`, the plotters.

So the work is producing that table from SpikeGLX files, plus a few places that assume
Open Ephys.

## The clock chain

Open Ephys had one link: rig pulses recorded on the ephys board itself. SpikeGLX has two,
joined on the NI clock:

```
DAQ message_id --(rig pulses on AIn, counted)--> NI time
NI time --(1 Hz edges: AI0 on the NI card, SY bit 6 on the probe)--> probe time
```

The output keeps the existing `ephys_sync.csv` format (`pulse, t_ephys, t_daq,
n_messages`), with `t_ephys` in probe seconds, so nothing downstream changes.

## Steps

1. **`postprocessing/sources/spikeglx.py`** (new, the counterpart of `openephys.py`).
   - Read `spikeglx_session.json` and the `.meta` files.
   - Rising edges on the rig's channel and on AI0, from the NI file. It is small
     (about 1.6 GB an hour), so this is quick.
   - Bit 6 of the probe's SY channel, from the AP file. This is one pass over the file,
     the same cost as the Open Ephys pass, and is cached the same way.
   - Pair the 1 Hz edges by nearest neighbour. Both streams start saving at the same
     moment, within ms, and the edges are 1 s apart, so the pairing cannot be
     ambiguous.
2. **`postprocessing/utils/clock.py`.**
   - An NI-to-probe map: a straight line between the paired 1 Hz edges.
   - Then reuse `map_ephys_to_daq` for the rig pulses, with a **loops-per-pulse**
     setting: the 1 kHz firmware (`arduino_daq_giga_rigs`, Rig 4) pulses every loop;
     the hat firmware (`arduino_daq_giga_hat_rigs`, Rigs 1-3) pulses every 10th.
   - Confirm the exact pulse-to-`message_id` offset for each firmware from real data,
     by the residuals, as was done for Open Ephys. Do not assume it.
3. **`postprocessing/pipeline.py` (`sync_ephys`).** Dispatch on what the folder holds:
   `spikeglx_session.json` versus `Record Node*`. Post-processing on the rig PC then
   picks it up with no control-side change.
4. **`ephys/export.py`.** Spike times are `sample / imSampRate`, the calibrated rate in
   the `.meta`, in place of Open Ephys's `timestamps.npy`. The same-recording check
   compares the `.ap.bin` path.
5. **The session window.** In a multi-rig run the probe file starts at the *first*
   rig's session.
   - Spikes outside this rig's DAQ run already come out unsynced
     (`n_spikes_unsynced`), which follows "mark, don't delete".
   - `obs_start`/`obs_stop` are clipped to `file_sample_begin`/`file_sample_end` from
     the JSON.
   - Those are themselves clipped to the file length: a rig still running at Stop gets
     an end up to ~0.1 s past the last written sample.
6. **`nwb/build_nwb.py`.** The device and electrode descriptions assume a 128-channel Open
   Ephys probe. They should describe the NP2014 (4 shanks, 384 channels) from the
   probe's `.meta`.
7. **Tests.**
   - A small synthetic SpikeGLX recording (`.meta` + `.bin`) as a fixture.
   - An end-to-end check on a real SpikeGLX session: sync residuals should be well under 1 ms.
   - A multi-rig case once two rigs have recorded together.

## Decisions still open

1. **Spike sorting.** It lives in `~/Dev/projects/ephys_analysis` on hal
   (`stef1029/ephys_analysis`): `oeab/` and `sorting/` for the CamNT probes. A Neuropixels
   counterpart there waits for real data. Its recipe needs tuning of its own: the CamNT
   one (500 Hz notch, no CAR, no drift correction, per shank) was fitted to that probe.
   `ephys_analysis` also imports `hex_behav_analysis_2` by path, which after the rename to
   `hba2` works only through the symlink; worth moving to `hba2` then.
2. **Rigs 1-3 firmware.** The hat firmware pulses on every 10th loop, for 20 µs: shorter
   than the NI card's 40 µs sample, so about half are missed, and not one per loop, so
   `sync_ephys` refuses it. Before those rigs record Neuropixels, the firmware should
   pulse on every loop (or the pairing learn its ratio) for about 150 µs.
3. **`open_ephys_channel`** is kept as the column's name for both systems, holding
   SpikeGLX's channel name for Neuropixels. Renaming it to `recorded_channel` would touch
   export, `build_nwb` and any plotter that reads it.
4. **One probe per rig** for now. `sync_ephys(record, recording_index=n)` picks the probe;
   one `ephys_sync.csv` per session means a second probe would need its own file.
