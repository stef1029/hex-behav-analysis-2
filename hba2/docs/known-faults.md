# Known faults: symptoms, causes and fixes

Faults found in real cohorts, written up so the next one is recognised rather than
worked out again. Each entry gives **the symptom** as you meet it, **how to confirm it**,
**the cause** (or the candidates, where it is not settled) and **the fix**, with the
module that does it. The module docstrings hold the detail and the validation; this page
is the index to them by symptom.

Three rules hold for every fix here:

- **Repair a copy, never the original.** Copy the cohort (real files, not links), repair
  the copy, and leave the original as the record of what the rig wrote. Repairs that
  read the rig's files always read them from the original, so a repair can never be
  applied twice.
- **Mark, never delete.** A fix that is not accepted leaves the session as it was and
  says why. A session that cannot be used is labelled in `sessions.csv`, not removed.
- **Check a fix against something independent before trusting it.** Nearly every
  timing fix here is checked the same way: *when a port's beam breaks, the head is at
  that port.* A fix that moves touches away from the port is wrong, however plausible.

## Finding out what a cohort has

Survey before repairing. Per session, the numbers that separate the faults below:

| what | where | points to |
|---|---|---|
| DAQ messages per second, and the error records | `-ArduinoDAQ.h5` (`message_ids`, `timestamps`, `error_messages`) | a sparse stream (< 10 msg/s) means the camera is not reaching the DAQ; errors in bursts mean the serial reader fault |
| rising edges on every channel, `CAMERA` and `LASER` included | the same file | which channel the camera is really on |
| camera pulses minus frame IDs | DAQ against the tracker's frame list | extra pulses: video early; too few: lost messages or a late DAQ |
| frame IDs: skips, and the video's frame count against the tracker's | `_Tracker_data.json`, the video header | camera drops; the converter chunking bug; a truncated video |
| trial-marker pulses against rows of `-trials.csv` | `DAQ_LINK1`, the trials CSV | lost markers; no marker at all (recorded before it existed) |
| beam breaks per port per day | the DAQ sensors | dead, stuck or chattering ports |
| the rig, from each session's own metadata | `-metadata.json` `rig`, else the DAQ log file name | which era table row applies. **Never infer the rig from the mouse**: mice move between rigs |

Write these into one table per cohort before deciding anything. The faults cluster by rig
and date, and the table is what shows the eras.

---

## The DAQ file

### The camera has no pulses, or pulses on the wrong channel

**Symptom.** `CAMERA` has no edges, so post-processing writes NaN frame times. Or
another channel (often `LASER`) has a ~30 Hz train with as many edges as the video has
frames.

**Confirm.** Count rising edges on every channel; the camera's channel has within a few
edges of the frame count, at the camera's frame rate.

**Cause.** The DAQ firmware was reflashed with different bit layouts, and the decoder
assumed one. On the 24-bit layout the camera sits on the bit the decoder calls `LASER`.
On the 26-bit layout it sits on bit 24 and the scales on bit 25, but the writer keeps
only bits 0–23, so the camera's *value* is lost. Its *edges* are not: the firmware sends a
message whenever any bit changes, so each camera edge is a saved message identical, in
its 24 kept bits, to the one before. The same layout moves the trial marker onto the bit
called `EXT_1`.

**Fix.** [`legacy/red_rig_daq.py`](../legacy/red_rig_daq.py) rewrites the file with each
channel where it belongs. Which recorded label holds which channel is a per-rig,
per-date table in the cohort's own script (an `Assignment`: bit order, channel map,
camera source `"label:<name>"`, `"repeats"` or `"none"`), checked against each session's
frames and trial rows. Find the table's rows from the survey: eras change on a date and
apply to a whole rig. Pass `read_messages` the bit order itself, `BIT_ORDERS["ext1"]`,
not its name.

### No camera signal reaches the DAQ at all

**Symptom.** A sparse DAQ stream (a few messages per second, only behaviour changes), no
~30 Hz train on any channel, though the video is fine. Sometimes a ~49 Hz train that is
not the camera (its edge count does not follow the frames).

**Cause.** A wiring or firmware change on the rig. It cannot be recovered from the DAQ
file.

**Fix.** [`postprocessing/repairs/repair_video_clock.py`](../postprocessing/repairs/repair_video_clock.py).
The camera's clock against the DAQ's is a fixed rate per rig (measured from the rig's
sessions *with* pulses), so a session needs only one number, when frame 0 was exposed.
It is found by lining up what the video shows with the beam breaks: per-port pixel
brightness (learned per rig) and, from DeepLabCut, the nose hidden at the port. Both are
validated blind on sessions with pulses first, and a fit is applied only when it passes
the checks that validation justifies. Its steps: `extract` (one small brightness movie per
video, a cluster array), `learn`, `validate`, `fit_cohort`, `validate_occlusion`,
`fit_occlusion_cohort`. It needs sessions with pulses **from the same rig** to learn from. When a cohort has too
few, a model learned on another cohort from the same rig, camera and weeks can be used:
check first that the framing matches (the maps are per pixel), then run it blind on the
sessions with pulses that the cohort does have. The occlusion fit, centred on the
brightness one, is what usually passes; brightness alone gets the right second but often
fails its own frame-level checks on a relit rig.

The occlusion latency (how far the nose-hidden signal sits from the beam breaks) does
not carry over between lighting eras: one learned on a rig's dim sessions put every
blind fit on a relit cohort 33–44 ms late. Measure it on the cohort's own sessions with
pulses — run `occlusion_one` centred on the true t0 (and again started ±0.6 s off, to see
that the peak is the same) and take the median of `occ_t0 − true t0` — and use another
cohort's only if none have pulses. Where the occlusion fit disagrees with pulses by more
than a frame or two on an accepted fit, suspect the pulses' frame times first: most such
sessions are ones the offset repair shifted or refused.

A video that also lost converter chunks (short by whole chunks, below) cannot be fitted
this way, since its frames are out of place after the first lost chunk, and the chunk
search needs frame times. Fit both together: scan t0 one frame at a time, find the best
lost chunks for each, keep the t0 whose touches put the head at the port most, then run
the occlusion fit on the in-order frames. (A cohort-level script does this; see the lab
layer's cohort converters.)

### Error records, IDs of exactly 2^24, and a sparse stream

**Symptom.** Many `error_messages` of 8 and 10 bytes, or pairs whose lengths add to 7;
"accepted" messages with ID exactly 16777216 + n; whole stretches of a session missing
behaviour changes. Worst on rigs where the stream is sparse.

**Cause.** The serial reader splits the stream on `\x02\x01` (one packet's end byte and
the next one's start byte) with a one-second timeout, and drops the last two bytes it read.
Those bytes can occur inside a payload, which splits the packet in two. And with no next
packet within a second, a read times out and a packet is saved shifted by a byte, or as
a 10-byte and an 8-byte error. Note that real message IDs pass 2^24 on their own after about 50
minutes, so only the exact value marks a shifted record.

**Fix.** `red_rig_daq.read_messages` rebuilds every message: split pairs exactly
(`split`), whole payloads from 10-byte errors (`err10`), and 8-byte errors and shifted
records less their last byte (`err8`, `shifted`, with bits 0–7 carried from the message
before and marked in `repair/unknown_bits`). On a few rigs a session's file cannot be
read at all (an empty `timestamps` array): there is nothing to repair, so leave it out
and say so.

### Camera pulses missing in the middle of a session

**Symptom.** Fewer camera pulses than frames, by one or two at a time, spread through the
session; frame times step by two periods now and then.

**Cause.** Lost DAQ messages (above), not the camera stalling. Judged on the firmware's
own clock (message IDs count loop cycles), nearly every gap holds a split error record.
Judged on PC timestamps instead, many more gaps appear: those are late timestamps with
nothing missing.

**Fix.** `red_rig_daq` fills pulses into intervals longer than 1.5 periods **on the
message-ID clock**, never on PC time. Filled pulses only ever sit between recorded ones.

### Frame 0 timed by frame 1's pulse

**Symptom.** Every frame is one period late, and only in sessions where the DAQ and the
camera started together.

**Cause.** The DAQ's first message was frame 0's strobe, and a first sample can never
be a rising edge.

**Fix.** `red_rig_daq.frame_zero` puts the pulse on that message, with a camera-low copy
before it.

### The DAQ's clock was invented

**Symptom.** DAQ timestamps evenly spaced (exactly 1 ms, or the session's duration over
its row count); durations and reaction times about a fifth shorter than the same task
elsewhere.

**Cause.** The `.h5` was rebuilt from a backup CSV that had no times. The real message
rate is about 790 per second.

**Fix.** Rebuild time from camera pulses (frame k at k / fps): the legacy converter does
this ([`legacy/`](../legacy/__init__.py)).

### PWM channels with hundreds of thousands of edges

**Symptom.** `events.csv` of gigabytes; `build_nwb` taking hours or running out of memory
on one session.

**Cause.** A dimmed LED or a spotlight driven by ~1 kHz PWM: every edge was written as
an event.

**Fix.** Post-processing now writes PWM runs as one on and one off
(`led_dim_on` / `led_dim_off`), and marks PWM-only channels as not written. Re-run
post-processing on sessions processed before that.

### The DAQ's PC timestamps stall

**Symptom.** A stretch of messages whose PC timestamps do not advance, or arrive in a
burst, while message IDs advance steadily.

**Cause.** The PC was late reading the serial port.

**Fix.** Nothing to repair if the clocks are mapped on message IDs. Any decision about
missing pulses or messages is made on message IDs, never on these timestamps. Sync rows
inside a stall are dropped as glitches.

---

## Trials

### No trials at all, though the rig wrote a trials CSV

**Symptom.** `trials.csv` empty; `-trials.csv` has rows; `DAQ_LINK1` never pulses.

**Cause.** The session was recorded before the trial marker existed.

**Fix.** [`legacy/premarker.py`](../legacy/premarker.py): trials then ran from LED onset
to first touch or timeout, so the windows are rebuilt from the DAQ's cues and touches,
and the CSV rows paired with them by time. Note that the rig writes each row as its trial
*ends*: `time_since_start_s` is the end and `time_since_start_s - trial_duration_s` the
start.

### Every trial after some point has the next trial's data

**Symptom.** Marker count differs from row count by one or a few; outcomes and ports
stop matching what the DAQ shows partway through a session.

**Cause.** Post-processing pairs markers with rows by order. A marker lost in the middle
(a DAQ message never saved) shifts every row after it. A surplus marker at the end (the
session stopped mid-trial) is harmless.

**Fix.** [`legacy/marker_pairing.py`](../legacy/marker_pairing.py) `pair_cohort` pairs
by time on every session whose counts differ, allowing either side to skip, and fits
again after a step in the PC clock (seen: 1.35 s). A row with no marker gets a window
from its logged times and is marked `rig_log`. Sessions of one or two trials can fail to
pair; they carry no information and are left as they were.

### No `-trials.csv` at all

**Symptom.** Post-processing skips the session, so it has no events, frame times or
scales either.

**Cause.** An aborted run, or a protocol that writes no trials (optotagging, a blank
protocol).

**Fix.** Write a header-only `-trials.csv` with a note beside it saying it is not the
rig's, so the session still gets everything else, with no trials. This is a cohort-script
step, not part of the package.

### The mouse ID is wrong

**Symptom.** One animal's sessions split across two IDs that differ by a typo; the
folder name disagrees with the metadata.

**Cause.** On older rigs the ID was typed by hand; folders were renamed afterwards, the
headers were not.

**Fix.** [`postprocessing/repairs/repair_mouse_id.py`](../postprocessing/repairs/repair_mouse_id.py)
for legacy-converted sessions. A current-controller session's metadata is the rig's own
record: report the disagreement, do not overwrite it.

---

## Video

### The video is shorter than the tracker's frame list, and the head is not at the port

**Symptom.** Fewer video frames than tracker frame IDs; touches with the head far from
the touched port, increasingly so through the session; the camera dropped frames.

**Cause.** The old bin→avi converter chunked the frames over workers and seeked each
chunk by *frame ID* as if it were a *file position*. After the first camera drop, every
chunk starts too late and the frames between chunks are never written.

**Confirm.** Replaying the chunking on the tracker's frame IDs, for some number of
workers, reproduces the video's frame count exactly. The worker count is not known, so
try a range; a length match alone does not pin it.

**Fix.** [`repair_video_timing.py`](../postprocessing/repairs/repair_video_timing.py),
after DeepLabCut (it checks itself with pose). A session decided `unexplained` stays
that way on later runs; pass `force` (or `--force`) with `only` to decide it again. It re-pairs each video frame with the time
of the frame it really holds, and accepts a repair only if it does not move touches off the
port. It needs the video header's frame count, so the session's video must be present.
Run it again after any post-processing run, which overwrites its output.

### The video is short by whole chunks

**Symptom.** The video is shorter than the replay of the chunking says it should be,
often with no camera drops at all, and the touch check is bad from some point in the
session on. The shortfall is an exact multiple of the chunk size (`n // 40` frames): 2.5%,
5%, 7.5% of the tracker's frames.

**Cause.** The old converter's workers each wrote one chunk; some failed, and their chunks
were never written. Every frame after a lost chunk is paired with a time one chunk too
early. Lost chunks are often next to each other.

**Confirm.** The shortfall divided by the chunk size is a whole number.

**Fix.** `repair_video_timing.py` finds which chunks are missing from the touches (an
exact search over the chunks in order) and accepts the answer only if alternate halves of
the touches pick the same chunks and the result passes the check outright. It needs frame
times, so a session with no camera pulses is left `unchecked` until its clock is fitted.
**Run it before the offset repair:** the offset repair can approximate one lost chunk
as a jump, and if it ran first, its output must be set aside before the chunks are repaired.
A session that also has a camera offset can fail the check on chunks alone; the offset
repair, run after, may then fit it as two stretches.

### The video is short, not by whole chunks, and not by the chunking bug

**Cause.** Not established: a recording stopped early, an interrupted conversion, a
truncated copy.

**Fix.** None. Check whether the frames the video has are timed right (touches against
the head); if not, mark the session's video timing as unusable.

### The whole video sits a constant 0.1–5 s from the DAQ

**Symptom.** At almost every touch the head is away from the port; shifting the video
by a fixed amount puts it there. Sometimes a single jump partway through.

**Cause.** A mismatch between camera pulses and frame IDs at the start of recording.
Frame IDs restart at 0 when the camera's stream starts and do not count exposures made
without a stream, while the strobe fires on every exposure. Extra pulses before frame 0
(the camera was already exposing when the DAQ connected) make the video early; missing
first pulses make it late. Strongly associated with the DAQ's USB connection crashing in
the previous session, or in this one. Post-processing warns about too few pulses but not
about too many.

**Confirm.** Camera pulses minus frame IDs is positive (or negative) by about the
offset in frames.

**Not this.** When the fitted offset is tens of seconds (60–90 s) and the two halves
disagree by a second or so, while the occlusion fit (beam sensors against the video) is
clean, it is the trials table, not the video, that is off: look at how its trials were
paired to the DAQ's markers.

**Fix.** [`repair_video_offset.py`](../postprocessing/repairs/repair_video_offset.py),
after `repair_video_timing`: it fits one offset, or two with one jump, from the touches,
and applies it only when it is significant, large enough and constant. It never invents a
time.

### DeepLabCut does not see the mouse in dim video

**Symptom.** Low likelihoods for a whole session (median well under 0.5), "head far from
port" at most touches, though the timing checks out elsewhere.

**Cause.** The rig was lit differently; the model was trained on brighter video.

**Fix.** Do not read it as a timing fault. Mark the session's pose as unusable, or run
DeepLabCut again on contrast-enhanced video. Leave such sessions out of any repair that
is checked with pose, since that check cannot work without pose.

Very low likelihoods in a *short* session (a few minutes) usually mean no mouse was in
view: a test, blank or aborted run.

---

## Scales

### Weights untimed

**Symptom.** "N of N scales readings have no pulse and are untimed".

**Cause.** The scales were never wired into the DAQ, so the scales CSV has no pulse
index; readings carry only the scales server's own clock.

**Fix.** [`repair_scales_timing.py`](../postprocessing/repairs/repair_scales_timing.py):
trials start once the mouse has stood on the platform for a second, so "above threshold
for a second" events are lined up with trial starts by one constant offset per session
(or two, if the PC clock stepped), and applied only when the fit passes its checks.
Sessions with too few trials stay untimed.

---

## Ports and geometry

### A port is never credited, or two ports' lines move together

**Symptom.** Per port per day: a port with touches on the DAQ that is never the
rewarded response; a port that goes silent for days; a port with thousands of touches in
one session (chattering); one port's line exactly copying another's from some date on.

**Cause.** A faulty sensor or wiring. Not established from the data alone.

**Fix.** Write a note in the cohort saying which ports, which dates, what the data shows and what
to check on the rig. **Do not drop trials until someone has checked the rig.** If
confirmed, label it in `sessions.csv` and tag the trials cued at that port, so analyses
can choose. Report how much it moves the results (trials cued at the port, and
performance without them) so the size of the question is known.

### Port coordinates wrong for a rig, or a port barely touched

**Symptom.** Angles off by a constant; the port table does not sit on the port mouths in
the calibration picture; calibration refuses a port for too few touches.

**Cause.** Rig tables are per rig per era: cameras get moved, and rig numbering changed
when the rigs were rebuilt. Some ports are hardly visited.

**Fix.** Calibrate each cohort from its own touches with
[`pose/calibrate_ports.py`](../pose/calibrate_ports.py), into the cohort's own
`port_coordinates/`, with `fill_missing=<touches>` to place sparse ports from the hexagon
the others make. Check the picture, and the printed spacing sd (a good rig is under
2°): a port sensor that is mapped to the wrong port for even one session puts its
touches at a neighbour, and once those push the port over the `fill_missing` cut-off its
median lands on the neighbour. Find the session with
`measure_session` (each session's points per port), and raise the cut-off or leave the
session out.

---

## Cohort layout

### Sessions in subfolders, or split across two folders

**Symptom.** Sessions under an extra folder level; one session's video in one folder and
its other files in another; old analysis outputs (`.nwb`, empty `temp_videos/`) beside
the raw files.

**Fix.** Do it in the copy: put every session in `<launch>/<session>/`, merge split
sessions, and leave out the old outputs. Record what was moved in a summary.

### A cohort still being written to

Wait until nothing has changed for a while (check modification times) before copying, or
copy again with a step that never deletes.

---

## Cluster jobs

### Every DeepLabCut task fails within seconds with `No module named 'torch'`

**Cause.** `sbatch` exports the submitting shell's environment, and a `PATH` with another
Python ahead of the job's wins over the job's environment.

**Fix.** Submit from a clean login shell:
`env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin bash -lc '...'`.

### Tasks fail with "no kernel image is available"

**Cause.** The PyTorch build has no kernels for that node's GPU.

**Fix.** The job script checks the GPU against the build and fails fast, renaming its log
`*_CUDA_FAILED.log`. Re-queue those tasks on other nodes; exclude a node that keeps failing.

### "Could not find a shuffle with trainingset fraction ..."

**Cause (probable).** DeepLabCut rewrites `project_path` in a model's `config.yaml` to
the path the reading machine sees. A Windows machine opening a shared model writes a
Windows path into it.

**Fix.** Check the config's `project_path` and restore it. Avoid opening shared models'
configs from other machines.

### A shared index cannot be rewritten

**Symptom.** `sessions.csv` never updates when two people run the index on one cohort.

**Cause.** Copying with times onto another user's `.bak` fails.

**Fix.** The writer now uses a plain copy; move an old `.bak` you do not own aside.
